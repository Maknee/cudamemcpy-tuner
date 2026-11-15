#include <algorithm>
#include <csignal>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/wait.h>
#include <unistd.h>
#include <benchmark/benchmark.h>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <fmt/format.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <charconv>
#include <string>
#include <thread>
#include <vector>
#ifdef __GLIBC__
#include <gnu/libc-version.h>
#endif
#include "arch/copy.h"
#include "cuda/cuda.h"
#include "cuda/versions.h"
#include "measure/metrics.h"
#include "machine.h"

// Default pageable measurements use exact hook kernels without initializing CUDA.
// Explicit pinned controls initialize the selected CUDA device outside timing;
// the isolated driver-helper mode only loads the verified original libcuda.
// Warm means repeated buffers, not an LLC residency promise. Cold evicts only
// cacheable sides before a batch; WC memory is never passed to FlushRange.
// Small copies use independent buffer pairs to amortize clock calls without
// turning the second cold copy into a hot reuse. Preparation and full-byte
// verification are outside manual timing. Every copy includes its store fences;
// N participants means the caller plus N-1 persistent helpers owning whole chunks.
// Each pair retains the full payload until verification; these are component
// controls, not a reused staging ring, DMA overlap or complete pipeline timings.
namespace {
using cmt::Error;
using cmt::Failure;
using cmt::Result;
using Clock = cmt::Clock;
constexpr std::size_t kGuardBytes = 64;
constexpr std::size_t kBufferAlignment = 64;
constexpr std::size_t kMaxBatchCopies = 256;
// Past the hook's 16 copying lanes: a many-core host's copy scaling is measured to its saturation.
constexpr std::size_t kMaxWorkers = 32;
constexpr std::size_t kDefaultChunkBytes = 1U << 20;
constexpr std::size_t kMaxPreviewBytes = 16U << 20;
// Object fields, a uint64 byte count and comma; the settings object is counted separately.
constexpr std::size_t kPreviewPointEnvelopeBytes = 64;
constexpr std::size_t kPreviewArrayEnvelopeBytes = sizeof("[]\n") - 1;

enum class Cache : std::uint8_t
{
	Warm,
	Cold
};

enum class CpuMemory : std::uint8_t
{
	Pageable,
	PinnedCached,
	PinnedWc
};

enum class Control : std::uint8_t
{
	Pageable,
	FillCached,
	FillWc,
	DrainCached,
	DrainWc
};

const char* ControlName(Control control)
{
	switch (control)
	{
	case Control::Pageable:
		return "pageable";
	case Control::FillCached:
		return "fill-cached";
	case Control::FillWc:
		return "fill-wc";
	case Control::DrainCached:
		return "drain-cached";
	case Control::DrainWc:
		return "drain-wc";
	}
	cmt::Panic("invalid CPU control");
}

Result<Control> ParseControl(const std::string& text)
{
	for (auto control :
		 {Control::Pageable, Control::FillCached, Control::FillWc, Control::DrainCached, Control::DrainWc})
	{
		if (text == ControlName(control))
		{
			return control;
		}
	}
	return Failure(Error::Kind::Usage,
				   "--cpu-control expects pageable, fill-cached, fill-wc, drain-cached or drain-wc");
}

CpuMemory SourceMemory(Control control)
{
	return control == Control::DrainCached ? CpuMemory::PinnedCached
		   : control == Control::DrainWc   ? CpuMemory::PinnedWc
										   : CpuMemory::Pageable;
}

CpuMemory DestinationMemory(Control control)
{
	return control == Control::FillCached ? CpuMemory::PinnedCached
		   : control == Control::FillWc	  ? CpuMemory::PinnedWc
										  : CpuMemory::Pageable;
}

const char* MemoryName(CpuMemory memory)
{
	switch (memory)
	{
	case CpuMemory::Pageable:
		return "pageable";
	case CpuMemory::PinnedCached:
		return "pinned_cached";
	case CpuMemory::PinnedWc:
		return "pinned_wc";
	}
	cmt::Panic("invalid CPU memory");
}

const char* CacheName(Cache cache, Control control)
{
	if (cache == Cache::Warm)
	{
		return "reused_buffers";
	}
	if (SourceMemory(control) == CpuMemory::PinnedWc)
	{
		return "destination_evicted_wc_source_unflushed";
	}
	if (DestinationMemory(control) == CpuMemory::PinnedWc)
	{
		return "source_evicted_wc_destination_unflushed";
	}
	return "source_destination_evicted";
}

struct Options
{
	std::vector<std::size_t> sizes, workers{1};
	std::vector<Cache> caches{Cache::Warm};
	std::vector<cmt::Variant> variants;
	Control control = Control::Pageable;
	cmt::Pages pages = cmt::Pages::Small;
	std::optional<int> device;
	std::vector<unsigned> allowed_cpus;
	std::size_t chunk_bytes = kDefaultChunkBytes;
	std::size_t batch_bytes = 1U << 20;
	std::size_t reserve_bytes = 1ULL << 30; // Conservative policy, not a measured capacity threshold.
	std::optional<std::size_t> memory_limit;
	std::string driver_helper, driver_manifest;
	std::string output;
	bool extended = false, grid = false, self_test = false, help = false;
	bool prefetch = false;
	bool explicit_loops = false;
	bool preview = false;
};

bool failed = false;
double clock_pair_ns = 0;
std::vector<cmt::Measurement> measurements;
const cmt::PathSpec cpu_path = []
{
	cmt::PathSpec path;
	path.name = "cpu_copy";
	path.dirs = cmt::PathSpec::Dirs::None;
	path.device_bytes = 0;
	return path;
}();

Result<void> Publish(const Options& config)
{
	if (config.output.empty())
	{
		return {};
	}
	if (config.device)
	{
		auto machine = cmt::MachineReportFor(*config.device, false);
		if (!machine)
		{
			return cmt::Failure(machine.error());
		}
		return cmt::WriteMeasurements(config.output, machine->text, measurements);
	}
	// Host metadata must not initialize CUDA for the default pageable control or
	// the isolated private helper. An empty GPU array means not queried here.
	JsonWriter writer;
	writer.start_object();
	cmt::JsonField(writer, "schema", cmt::kMachineSchema, true);
	cmt::JsonKey(writer, "tool");
	cmt::WriteToolJson(writer);
	cmt::JsonKey(writer, "host");
	cmt::host::WriteHostJson(writer, "/", cmt::host::ReadCpuid());
	cmt::JsonKey(writer, "gpus");
	writer.start_array();
	writer.end_array();
	cmt::JsonField(writer, "gpu_error", nullptr);
	cmt::JsonKey(writer, "driver");
	cmt::nv::WriteDriverJson(writer, "/", cmt::ReadTextFile("/proc/self/maps").value_or(""), std::nullopt,
							 std::nullopt);
	writer.end_object();
	auto machine = cmt::FinishJson(writer);
	if (!machine)
	{
		return cmt::Failure(machine.error());
	}
	return cmt::WriteMeasurements(config.output, *machine, measurements);
}

bool StreamingStores(cmt::Method method)
{
	return method == cmt::Method::NtStore || method == cmt::Method::NtStorePrefetch;
}

Result<std::size_t> Number(std::string text)
{
	std::size_t scale = 1;
	if (!text.empty())
	{
		const auto unit = text.back();
		if (unit == 'K' || unit == 'M' || unit == 'G')
		{
			scale = std::size_t{1} << (unit == 'K' ? 10 : unit == 'M' ? 20 : 30);
			text.pop_back();
		}
	}
	if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
	{
		return Failure(Error::Kind::Usage, "expected unsigned bytes/count, optionally suffixed K, M or G");
	}
	std::size_t value = 0;
	const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
	if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
	{
		return Failure(Error::Kind::Usage, "byte/count value overflows size_t");
	}
	if (value > std::numeric_limits<std::size_t>::max() / scale)
	{
		return Failure(Error::Kind::Usage, "byte/count value overflows size_t");
	}
	return static_cast<std::size_t>(value) * scale;
}

/// Sizes or worker counts in gpu_copy's syntax (ParseSizes: K/M/G, ranges LO..HI), each once: a
/// repeated value would add its samples to the same measurement record (MeasurementFor).
Result<std::vector<std::size_t>> Numbers(const std::string& text)
{
	auto result = cmt::ParseSizes(text);
	if (!result)
	{
		return cmt::Failure(result.error());
	}
	for (std::size_t i = 0; i < std::size(*result); i++)
	{
		if (std::find(std::begin(*result), std::begin(*result) + i, (*result)[i]) != std::begin(*result) + i)
		{
			return Failure(Error::Kind::Usage, "sizes/workers must be unique");
		}
	}
	return result;
}

Result<std::vector<cmt::Variant>> SelectVariants(const std::optional<std::string>& names, bool prefetch)
{
	const auto catalog = cmt::Variants(cmt::kNativeArch);
	std::vector<cmt::Variant> selected;
	if (names)
	{
		if (names->empty() || names->back() == ',')
		{
			return Failure(Error::Kind::Usage, "--cpu-loops requires a nonempty list of exact catalog names");
		}
		std::istringstream words(*names);
		for (std::string name; std::getline(words, name, ',');)
		{
			const auto found = std::find_if(catalog.begin(), catalog.end(), [&](const cmt::Variant& variant)
											{ return cmt::VariantText(variant) == name; });
			if (found == catalog.end())
			{
				return Failure(Error::Kind::Usage, "--cpu-loops contains an empty or unknown catalog name");
			}
			if (std::any_of(selected.begin(), selected.end(), [&](const cmt::Variant& variant)
							{ return variant.method == found->method && variant.isa == found->isa; }))
			{
				return Failure(Error::Kind::Usage, "--cpu-loops contains a duplicate catalog name");
			}
			selected.push_back(*found);
		}
	}
	else
	{
		for (const auto& variant : catalog)
		{
			if ((variant.method == cmt::Method::NtStorePrefetch) == prefetch)
			{
				selected.push_back(variant);
			}
		}
	}
	if (selected.empty())
	{
		return Failure(Error::Kind::Usage, "selected CPU loop set is empty on this architecture");
	}
	return selected;
}

Result<Options> ParseOptions(int& argc, char** argv)
{
	Options result;
	std::optional<std::string> loop_names;
	bool driver_option = false;
	bool duplicate_cpu_option = false;
	std::vector<std::string> cpu_options;
	int kept = 1;
	for (int i = 1; i < argc; i++)
	{
		const std::string arg = argv[i];
		if (arg == "--out" || arg.rfind("--out=", 0) == 0)
		{
			if (!result.output.empty())
			{
				return Failure(Error::Kind::Usage, "duplicate --out");
			}
			if (arg == "--out")
			{
				if (++i == argc || argv[i][0] == '-')
				{
					return Failure(Error::Kind::Usage, "--out needs a path");
				}
				result.output = argv[i];
			}
			else
			{
				result.output = arg.substr(6);
			}
			if (result.output.empty())
			{
				return Failure(Error::Kind::Usage, "--out needs a nonempty path");
			}
			continue;
		}
		if (arg.rfind("--cpu-", 0) != 0)
		{
			argv[kept++] = argv[i];
			continue;
		}
		result.extended = true;
		const auto option = arg.substr(0, arg.find('='));
		if (std::find(cpu_options.begin(), cpu_options.end(), option) != cpu_options.end())
		{
			duplicate_cpu_option = true;
		}
		cpu_options.push_back(option);
		if (arg == "--cpu-help")
		{
			result.help = true;
			continue;
		}
		if (arg == "--cpu-preview")
		{
			result.preview = true;
			continue;
		}
		if (arg == "--cpu-grid")
		{
			result.grid = true;
			continue;
		}
		if (arg == "--cpu-self-test")
		{
			result.self_test = true;
			continue;
		}
		if (arg == "--cpu-prefetch")
		{
			result.prefetch = true;
			continue;
		}
		const auto equals = arg.find('=');
		if (equals == std::string::npos)
		{
			return Failure(Error::Kind::Usage, "CPU options use --cpu-name=value");
		}
		const auto key = arg.substr(0, equals), value = arg.substr(equals + 1);
		if (key == "--cpu-control")
		{
			auto control = ParseControl(value);
			if (!control)
			{
				return cmt::Failure(control.error());
			}
			result.control = *control;
		}
		else if (key == "--cpu-loops")
		{
			if (loop_names)
			{
				return Failure(Error::Kind::Usage, "duplicate --cpu-loops option");
			}
			loop_names = value;
			result.explicit_loops = true;
		}
		else if (key == "--cpu-pages")
		{
			auto pages = cmt::ParsePages(value);
			if (!pages || *pages == cmt::Pages::Hugetlb)
			{
				return Failure(Error::Kind::Usage, "--cpu-pages expects small (4k) or thp");
			}
			result.pages = *pages;
		}
		else if (key == "--cpu-device")
		{
			auto device = Number(value);
			if (!device || *device > static_cast<std::size_t>(std::numeric_limits<int>::max()))
			{
				return Failure(Error::Kind::Usage, "--cpu-device expects a nonnegative device ordinal");
			}
			result.device = static_cast<int>(*device);
		}
		else if (key == "--cpu-chunk-bytes")
		{
			auto bytes = Number(value);
			if (!bytes)
			{
				return cmt::Failure(bytes.error());
			}
			result.chunk_bytes = *bytes;
		}
		else if (key == "--cpu-driver-helper")
		{
			driver_option = true;
			result.driver_helper = value;
		}
		else if (key == "--cpu-driver-manifest")
		{
			driver_option = true;
			result.driver_manifest = value;
		}
		else if (key == "--cpu-sizes")
		{
			auto parsed = Numbers(value);
			if (!parsed)
			{
				return cmt::Failure(parsed.error());
			}
			result.sizes = std::move(*parsed);
		}
		else if (key == "--cpu-workers")
		{
			if (value == "all")
			{
				result.workers.clear();
				for (std::size_t workers = 1; workers <= kMaxWorkers; ++workers)
				{
					result.workers.push_back(workers);
				}
				continue;
			}
			auto parsed = Numbers(value);
			if (!parsed)
			{
				return cmt::Failure(parsed.error());
			}
			result.workers = std::move(*parsed);
		}
		else if (key == "--cpu-batch-bytes")
		{
			auto parsed = Number(value);
			if (!parsed)
			{
				return cmt::Failure(parsed.error());
			}
			result.batch_bytes = std::move(*parsed);
		}
		else if (key == "--cpu-reserve-bytes")
		{
			auto parsed = Number(value);
			if (!parsed)
			{
				return cmt::Failure(parsed.error());
			}
			result.reserve_bytes = std::move(*parsed);
		}
		else if (key == "--cpu-memory-limit")
		{
			auto parsed = Number(value);
			if (!parsed)
			{
				return cmt::Failure(parsed.error());
			}
			result.memory_limit = std::move(*parsed);
		}
		else if (key == "--cpu-cache")
		{
			if (value == "warm")
			{
				result.caches = {Cache::Warm};
			}
			else if (value == "cold")
			{
				result.caches = {Cache::Cold};
			}
			else if (value == "both")
			{
				result.caches = {Cache::Warm, Cache::Cold};
			}
			else
			{
				return Failure(Error::Kind::Usage, "--cpu-cache expects warm, cold or both");
			}
		}
		else
		{
			return Failure(Error::Kind::Usage, "unknown CPU option");
		}
	}
	argv[kept] = nullptr;
	argc = kept;
	if (result.preview && (!result.output.empty() || result.self_test || result.help || driver_option ||
						   duplicate_cpu_option || argc != 1))
	{
		return Failure(Error::Kind::Usage,
					   "--cpu-preview accepts CPU measurement options only; no --out, help, self-test, "
					   "driver-helper, duplicate options or Google Benchmark arguments");
	}
	if (result.grid && !std::empty(result.sizes))
	{
		return Failure(Error::Kind::Usage, "choose --cpu-grid or --cpu-sizes, not both");
	}
	if (result.grid)
	{
		auto grid = Numbers("4K,16K,64K,128K,512K,1M,2M,4M,16M,64M,256M,512M,1G,2G,4G,8G,16G,32G");
		if (!grid)
		{
			return cmt::Failure(grid.error());
		}
		result.sizes = std::move(*grid);
	}
	if (std::empty(result.sizes))
	{
		for (std::size_t n = 4096; n <= (64U << 20); n *= 4)
		{
			result.sizes.push_back(n);
		}
	}
	if (*std::max_element(result.sizes.begin(), result.sizes.end()) >
		static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
	{
		return Failure(Error::Kind::Usage, "size exceeds Google Benchmark argument range");
	}
	if (result.explicit_loops && (result.prefetch || driver_option || result.self_test))
	{
		return Failure(Error::Kind::Usage,
					   "--cpu-loops conflicts with --cpu-prefetch, driver-helper and self-test modes");
	}
	auto selected = SelectVariants(loop_names, result.prefetch);
	if (!selected)
	{
		return cmt::Failure(selected.error());
	}
	result.variants = std::move(*selected);
	if (!result.batch_bytes || !result.chunk_bytes ||
		*std::max_element(std::begin(result.workers), std::end(result.workers)) > kMaxWorkers)
	{
		return Failure(Error::Kind::Usage, "batch/chunk bytes must be positive; at most 32 copying lanes");
	}
	if (result.driver_helper.empty() != result.driver_manifest.empty())
	{
		return Failure(Error::Kind::Usage,
					   "driver helper requires both --cpu-driver-helper and --cpu-driver-manifest");
	}
	if ((result.control == Control::Pageable) == result.device.has_value())
	{
		return Failure(Error::Kind::Usage,
					   "pinned CPU controls require --cpu-device; pageable control does not accept it");
	}
	if (!result.driver_helper.empty() && (result.control != Control::Pageable || result.prefetch))
	{
		return Failure(
			Error::Kind::Usage,
			"driver helper is verified only for the isolated pageable CPU control, without --cpu-prefetch");
	}
	if ((!result.driver_helper.empty() || result.self_test) && result.pages != cmt::Pages::Small)
	{
		return Failure(
			Error::Kind::Usage,
			"CPU helper/self-test setup does not apply page policy; --cpu-pages=thp is unsupported");
	}
	if (result.self_test && result.control != Control::Pageable)
	{
		return Failure(Error::Kind::Usage, "--cpu-self-test exercises pageable kernels; pinned controls "
										   "require a measured byte-checked run");
	}
	if (result.self_test && !result.output.empty())
	{
		return Failure(Error::Kind::Usage, "--out requires measurements, not --cpu-self-test");
	}
	std::sort(std::begin(result.workers), std::end(result.workers));
	return result;
}

std::size_t ChunkCount(std::size_t bytes, std::size_t chunk_bytes)
{
	return bytes == 0 ? 0 : 1 + (bytes - 1) / chunk_bytes;
}

std::size_t CopyingLanes(std::size_t bytes, std::size_t chunk_bytes, std::size_t workers)
{
	return std::min(workers, ChunkCount(bytes, chunk_bytes));
}

struct AllocationPlan
{
	std::size_t stride;
	std::size_t buffer_bytes;
	std::size_t capacity_budget_bytes;
};

std::optional<std::size_t> BufferBudget(std::size_t bytes, CpuMemory memory)
{
	// HostBuffer reserves a THP-aligned mapping plus alignment slack even for
	// Pages::Small. Count that full virtual extent conservatively. CUDA controls
	// budget page-rounded requested bytes; driver-internal overhead is unknown.
	const auto alignment = memory == CpuMemory::Pageable ? cmt::ThpBytes() : cmt::BasePageBytes();
	const auto rounded = cmt::AlignUp(bytes, alignment);
	const auto slack = memory == CpuMemory::Pageable ? alignment : 0;
	if (!rounded || *rounded > std::numeric_limits<std::size_t>::max() - slack)
	{
		return std::nullopt;
	}
	return static_cast<std::size_t>(*rounded) + slack;
}

std::optional<AllocationPlan> Capacity(std::size_t bytes, std::size_t copies, std::size_t reserve,
									   std::size_t available, Control control)
{
	if (!copies || bytes > std::numeric_limits<std::size_t>::max() - kGuardBytes)
	{
		return std::nullopt;
	}
	const auto stride = cmt::AlignUp(bytes + kGuardBytes, kBufferAlignment);
	if (!stride || *stride > cmt::kMaxBytes / copies)
	{
		return std::nullopt;
	}
	const auto allocation = static_cast<std::size_t>(*stride) * copies;
	const auto source = BufferBudget(allocation, SourceMemory(control));
	const auto destination = BufferBudget(allocation, DestinationMemory(control));
	if (!source || !destination || reserve > available || *source > available - reserve ||
		*destination > available - reserve - *source)
	{
		return std::nullopt;
	}
	return AllocationPlan{static_cast<std::size_t>(*stride), allocation, *source + *destination};
}

struct Buffer
{
	cmt::HostBuffer pageable;
	cmt::PinnedAlloc pinned;

	Result<void> Create(std::size_t bytes, CpuMemory memory, cmt::Pages pages)
	{
		if (memory == CpuMemory::Pageable)
		{
			auto buffer = cmt::HostBuffer::Create(bytes, pages);
			if (!buffer)
			{
				return Failure(buffer.error());
			}
			pageable = std::move(*buffer);
			return {};
		}
		const auto kind = memory == CpuMemory::PinnedWc ? cmt::HostMem::WriteCombined : cmt::HostMem::Cached;
		auto buffer = cmt::PinnedAlloc::Create(bytes, kind);
		if (!buffer)
		{
			return Failure(buffer.error());
		}
		pinned = std::move(*buffer);
		for (std::size_t offset = 0; offset < bytes; offset += cmt::BasePageBytes())
		{
			pinned.data()[offset] = 0;
		}
		cmt::StoreFence();
		return {};
	}

	unsigned char* data() const
	{
		return pageable.data() ? pageable.data() : pinned.data();
	}

	std::optional<long> HugeKiB() const
	{
		// CUDA-owned memory is deliberately unqueried, not observed as zero huge pages.
		return pageable.data() ? pageable.HugeKiB() : std::nullopt;
	}
};

void Prepare(unsigned char* src, unsigned char* dst, std::size_t bytes, std::uint64_t epoch)
{
	// A new pattern on every timed batch makes draining an old buffer detectable.
	std::uint64_t value = epoch * 0x9e3779b97f4a7c15ULL;
	std::size_t i = 0;
	for (; i + sizeof value <= bytes; i += sizeof value)
	{
		value += 0x517cc1b727220a95ULL;
		// SAFETY: the loop bounds leave sizeof(value) live bytes; memcpy needs no alignment.
		std::memcpy(src + i, &value, sizeof value);
		const auto poison = ~value;
		std::memcpy(dst + i, &poison, sizeof poison);
	}
	for (; i < bytes; i++)
	{
		src[i] = static_cast<unsigned char>(value + i);
		dst[i] = static_cast<unsigned char>((value + i) ^ 255U);
	}
}

#if defined(__x86_64__)
// Independent equality check, not a candidate copy kernel. Streaming loads avoid
// slow ordinary reads from WC memory; both fences stay outside the copy timer.
__attribute__((target("avx2"))) bool VerifiedWc(const unsigned char* wc, const unsigned char* cached,
												 std::size_t bytes)
{
	constexpr std::size_t cache_line_bytes = 64;
	constexpr std::size_t vector_bytes = sizeof(__m256i);
	constexpr int all_bytes_equal = -1;
	_mm_mfence();
	bool equal = true;
	while (bytes && (reinterpret_cast<std::uintptr_t>(wc) & (cache_line_bytes - 1)))
	{
		equal &= *wc++ == *cached++;
		--bytes;
	}
	while (bytes >= cache_line_bytes)
	{
		// Read both adjacent halves before moving to the next WC cache line.
		const auto wc_first = _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(wc));
		const auto wc_second =
			_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(wc + vector_bytes));
		const auto cached_first = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cached));
		const auto cached_second =
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cached + vector_bytes));
		equal &= _mm256_movemask_epi8(_mm256_cmpeq_epi8(wc_first, cached_first)) == all_bytes_equal;
		equal &= _mm256_movemask_epi8(_mm256_cmpeq_epi8(wc_second, cached_second)) == all_bytes_equal;
		wc += cache_line_bytes;
		cached += cache_line_bytes;
		bytes -= cache_line_bytes;
	}
	if (bytes >= vector_bytes)
	{
		const auto wc_value = _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(wc));
		const auto cached_value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cached));
		equal &= _mm256_movemask_epi8(_mm256_cmpeq_epi8(wc_value, cached_value)) == all_bytes_equal;
		wc += vector_bytes;
		cached += vector_bytes;
		bytes -= vector_bytes;
	}
	while (bytes)
	{
		equal &= *wc++ == *cached++;
		--bytes;
	}
	_mm_mfence(); // Also reached on a mismatch; finish reads before any buffer reuse.
	return equal;
}
#endif

bool Verified(const unsigned char* src, const unsigned char* dst, std::size_t bytes,
			  CpuMemory source_memory = CpuMemory::Pageable,
			  CpuMemory destination_memory = CpuMemory::Pageable)
{
#if defined(__x86_64__)
	const bool source_wc = source_memory == CpuMemory::PinnedWc;
	const bool destination_wc = destination_memory == CpuMemory::PinnedWc;
	if (source_wc != destination_wc && cmt::hwcopy::GetCpu().Has(cmt::Isa::Avx2))
	{
		return VerifiedWc(source_wc ? src : dst, source_wc ? dst : src, bytes);
	}
#else
	(void)source_memory;
	(void)destination_memory;
#endif
	return std::memcmp(src, dst, bytes) == 0;
}

bool GuardsIntact(const unsigned char* src, const unsigned char* dst, std::size_t bytes)
{
	return std::all_of(src + bytes, src + bytes + kGuardBytes,
					   [](unsigned char value) { return value == 0xa5; }) &&
		   std::all_of(dst + bytes, dst + bytes + kGuardBytes,
					   [](unsigned char value) { return value == 0x5a; });
}

template <class Kernel>
class Workers
{
	struct Job
	{
		unsigned char* dst;
		const unsigned char* src;
		std::size_t bytes;
	} pending_job{};

	const std::size_t worker_count;
	const std::size_t chunk_bytes;
	cmt::CopyCtx copy_context;
	Mutex mutex;
	ConditionVariable work_done;

	struct Helper
	{
		cmt::Thread thread;
		ConditionVariable work_ready;
		std::size_t generation = 0;
	};

	std::vector<std::unique_ptr<Helper>> helpers;
	std::size_t ready_helpers = 0;
	std::size_t remaining = 0;
	bool stopping = false;

	void Stop()
	{
		{
			std::lock_guard<Mutex> guard(mutex);
			stopping = true;
		}
		for (auto& helper : helpers)
		{
			helper->work_ready.Signal();
			auto joined = helper->thread.Join();
			if (!joined)
			{
				cmt::Panic(joined.error().Message());
			}
		}
	}

	void Copy(Job job, std::size_t worker)
	{
		const auto chunks = ChunkCount(job.bytes, chunk_bytes);
		const auto active_lanes = std::min(chunks, worker_count);
		for (std::size_t chunk = worker; chunk < chunks;)
		{
			const auto begin = chunk * chunk_bytes;
			const auto length = std::min(chunk_bytes, job.bytes - begin);
			// SAFETY: each whole chunk belongs to exactly one lane. The final
			// chunk alone may be shorter. No lane shares a chunk or destination.
			Kernel::Copy(copy_context, job.dst + begin, job.src + begin, length);
			cmt::StoreFence();
			if (chunks - chunk <= active_lanes)
			{
				break;
			}
			chunk += active_lanes;
		}
	}

public:
	Workers(std::size_t count, std::size_t chunk, cmt::CopyCtx context)
		: worker_count(count), chunk_bytes(chunk), copy_context(context)
	{
	}

	Result<void> Start()
	{
		if (worker_count == 1)
		{
			return {};
		}
		helpers.reserve(worker_count - 1);
		for (std::size_t worker = 1; worker < worker_count; worker++)
		{
			helpers.push_back(std::make_unique<Helper>());
			auto& helper = *helpers.back();
			auto started = helper.thread.Start(
				[this, worker, &helper]
				{
					std::size_t seen = 0;
					std::unique_lock<Mutex> guard(mutex);
					if (++ready_helpers == worker_count - 1)
					{
						work_done.Signal();
					}
					for (;;)
					{
						while (!stopping && helper.generation == seen)
						{
							helper.work_ready.Wait(guard.mutex());
						}
						if (stopping)
						{
							return;
						}
						seen = helper.generation;
						const auto job = pending_job;
						guard.unlock();
						Copy(job, worker);
						guard.lock();
						if (--remaining == 0)
						{
							work_done.Signal();
						}
					}
				},
				cmt::ThreadPlacement{});
			if (!started)
			{
				Stop();
				return cmt::Failure(started.error());
			}
		}
		std::unique_lock<Mutex> guard(mutex);
		while (ready_helpers != worker_count - 1)
		{
			work_done.Wait(guard.mutex());
		}
		return {};
	}

	~Workers()
	{
		Stop();
	}

	void Run(unsigned char* dst, const unsigned char* src, std::size_t bytes)
	{
		const auto active_lanes = CopyingLanes(bytes, chunk_bytes, worker_count);
		if (active_lanes == 1)
		{
			Copy({dst, src, bytes}, 0);
			return;
		}
		std::unique_lock<Mutex> guard(mutex);
		pending_job = {dst, src, bytes};
		remaining = active_lanes - 1;
		for (std::size_t worker = 1; worker < active_lanes; ++worker)
		{
			auto& helper = *helpers[worker - 1];
			++helper.generation;
			helper.work_ready.Signal();
		}
		guard.unlock();
		Copy({dst, src, bytes}, 0);
		guard.lock();
		// The mutex publishes the job before helpers read it. Each lane fences
		// every completed chunk, like a staging lane handing its chunk onward.
		// The caller cannot reuse buffers until all helpers reported completion;
		// destructor stop/join similarly precedes buffer destruction on failure.
		while (remaining != 0)
		{
			work_done.Wait(guard.mutex());
		}
	}
};

// This is an isolated CPU helper, not a CUDA memcpy API or a GPU transfer.
// dlopen constructors and identity/self-tests run before manual timing. No CUDA
// entry point is called, and no library bytes or permissions are changed.
using DriverCopy = void (*)(void*, const void*, std::size_t);
DriverCopy driver_copy = nullptr;

struct DriverKernel
{
	static void Copy(cmt::CopyCtx, void* dst, const void* src, std::size_t bytes)
	{
		driver_copy(dst, src, bytes);
	}
};

struct DriverHelper
{
	void* handle = nullptr;
	int fd = -1;
	std::string path;
	std::string version;
	std::string sha256;
	std::string mapping;
	std::string reason;
	std::string gpu_fds_before;
	std::string gpu_fds_after;
	struct stat identity{};
	const cmt::patch::DriverVersion* row = nullptr;
	bool fatal = false;
	double setup_seconds = 0;

	~DriverHelper()
	{
		driver_copy = nullptr;
		if (handle)
		{
			dlclose(handle);
		}
		if (fd >= 0)
		{
			close(fd);
		}
	}

	static bool SameFile(const struct stat& a, const struct stat& b)
	{
		return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
			   a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec;
	}

	static Result<std::string> GpuFds()
	{
		std::string result;
		std::error_code iteration_error;
		std::filesystem::directory_iterator it("/proc/self/fd", iteration_error), end;
		for (; !iteration_error && it != end; it.increment(iteration_error))
		{
			const auto& entry = *it;
			std::error_code error;
			const auto target = std::filesystem::read_symlink(entry.path(), error).string();
			if (!error && target.rfind("/dev/nvidia", 0) == 0)
			{
				result += target + ";";
			}
		}
		if (iteration_error)
		{
			return Failure(Error::Kind::Os, "cannot inspect GPU file descriptors", iteration_error.value());
		}
		return result.empty() ? "none observed" : result;
	}

	static pid_t WaitOwned(pid_t child, int* status, int wait_flags)
	{
		pid_t result;
		do
		{
			result = waitpid(child, status, wait_flags);
		} while (result < 0 && errno == EINTR);
		return result;
	}

	Result<std::string> Hash() const
	{
		// No shell interpretation or new crypto-library dependency. The child
		// hashes the already-open original inode, not a newly resolved pathname.
		int channel[2];
		if (pipe(channel))
		{
			return Failure(Error::Kind::Os, "sha256 pipe failed");
		}
		const auto source = std::string("/proc/self/fd/") + std::to_string(fd);
		const auto child = fork();
		if (child < 0)
		{
			close(channel[0]);
			close(channel[1]);
			return Failure(Error::Kind::Os, "sha256 fork failed");
		}
		if (child == 0)
		{
			close(channel[0]);
			if (dup2(channel[1], STDOUT_FILENO) < 0)
			{
				_exit(126);
			}
			if (channel[1] != STDOUT_FILENO)
			{
				close(channel[1]);
			}
			execl("/usr/bin/sha256sum", "sha256sum", "--", source.c_str(), static_cast<char*>(nullptr));
			_exit(127);
		}
		close(channel[1]);
		std::string output;
		char chunk[256];
		bool read_failed = false;
		for (;;)
		{
			const auto count = read(channel[0], chunk, sizeof chunk);
			if (count < 0 && errno == EINTR)
			{
				continue;
			}
			if (count <= 0)
			{
				read_failed = count < 0;
				break;
			}
			output.append(chunk, static_cast<std::size_t>(count));
			if (output.size() >= 8192)
			{
				read_failed = true;
				break;
			}
		}
		close(channel[0]);
		int status = 0;
		const auto waited = WaitOwned(child, &status, 0);
		if (waited < 0 && errno != ECHILD)
		{
			kill(child, SIGKILL);
			WaitOwned(child, &status, 0);
		}
		if (read_failed || waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
			output.size() < 65 || output[64] != ' ' ||
			output.substr(0, 64).find_first_not_of("0123456789abcdef") != std::string::npos)
		{
			return Failure(Error::Kind::Os, "sha256 utility failed");
		}
		return output.substr(0, 64);
	}

	Result<void> SelfTest()
	{
		const auto child = fork();
		if (child < 0)
		{
			return Failure(Error::Kind::Os, "self-test fork failed");
		}
		if (child == 0)
		{
			// Separate process, bounded storage, changing input and poisoned output.
			// Full guards cover arbitrary tails and all source/destination alignments.
			alignas(64) unsigned char source[66048], destination[66048], expected[66048];
			const std::size_t lengths[]{0,	1,	2,	3,	4,	7,	 8,	  15,	16,	  17,	31,
										32, 33, 63, 64, 65, 127, 255, 1023, 4095, 4096, 65539};
			for (std::size_t epoch = 1; epoch <= 2; epoch++)
			{
				for (const auto bytes : lengths)
				{
					for (std::size_t offset = 0; offset < 16; offset++)
					{
						for (std::size_t dst_alignment = 0; dst_alignment < 16; dst_alignment++)
						{
							std::memset(source, 0xa5, sizeof source);
							std::memset(destination, 0x5a, sizeof destination);
							const auto src_offset = 64 + offset, dst_offset = 64 + dst_alignment;
							Prepare(source + src_offset, destination + dst_offset, bytes, epoch);
							std::memcpy(expected, source, sizeof source);
							driver_copy(destination + dst_offset, source + src_offset, bytes);
							cmt::StoreFence();
							if (std::memcmp(source, expected, sizeof source) ||
								!Verified(source + src_offset, destination + dst_offset, bytes))
							{
								_exit(10);
							}
							for (std::size_t i = 0; i < sizeof destination; i++)
							{
								if ((i < dst_offset || i >= dst_offset + bytes) && destination[i] != 0x5a)
								{
									_exit(11);
								}
							}
						}
					}
				}
			}
			_exit(0);
		}
		const auto deadline = Clock::now() + std::chrono::seconds(5);
		int status = 0;
		for (;;)
		{
			const auto result = WaitOwned(child, &status, WNOHANG);
			if (result == child)
			{
				break;
			}
			if (result < 0)
			{
				if (errno != ECHILD)
				{
					kill(child, SIGKILL);
					WaitOwned(child, &status, 0);
				}
				return Failure(Error::Kind::Os, "self-test wait failed");
			}
			if (Clock::now() >= deadline)
			{
				kill(child, SIGKILL);
				WaitOwned(child, &status, 0);
				fatal = true;
				reason = "failure=driver_selftest timeout";
				return {};
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		if (WIFSIGNALED(status) && WTERMSIG(status) == SIGILL)
		{
			reason = "skip=unsupported_driver_helper instruction_set";
		}
		else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
		{
			fatal = true;
			reason = fmt::format("failure=driver_selftest wait_status={}", status);
		}
		return {};
	}

	Result<void> LoadChecked(const Options& config)
	{
#if !defined(__x86_64__)
		return Failure(Error::Kind::Os, "no reviewed in-driver internal_memcpy helper on this architecture");
#else
		std::error_code path_error;
		path = std::filesystem::canonical(config.driver_helper, path_error).string();
		if (path_error)
		{
			return Failure(Error::Kind::Os, "cannot resolve driver", path_error.value());
		}
		const auto name = std::filesystem::path(path).filename().string();
		if (name.rfind("libcuda.so.", 0) != 0)
		{
			return Failure(Error::Kind::Os, "explicit original libcuda filename required");
		}
		version = name.substr(std::strlen("libcuda.so."));
		row = cmt::patch::FindVersion(cmt::patch::NativeVersionTable(), version.c_str());
		// Reviewed original 615.71.09 body: [0x285900,0x285bb8), SSE and scalar
		// loads/stores, branches and stack only. NT stores are not fenced by
		// the helper; Workers fences before ending every timed copy.
		if (!row || version != "615.71.09" || row->internal_memcpy_va != 0x285900)
		{
			return Failure(Error::Kind::Os, "no reviewed helper ABI/body for this exact version row");
		}
		std::ifstream manifest(config.driver_manifest);
		std::string expected, line;
		while (std::getline(manifest, line))
		{
			if (line.empty() || line[0] == '#')
			{
				continue;
			}
			std::istringstream fields(line);
			std::string arch, ver, url, package_sha, member, lib_sha;
			if (fields >> arch >> ver >> url >> package_sha >> member >> lib_sha && arch == "x86_64" &&
				ver == version)
			{
				if (!expected.empty())
				{
					return Failure(Error::Kind::Os, "duplicate driver identity manifest row");
				}
				expected = lib_sha;
			}
		}
		if (expected != "85a8cc8cf00419ceb295ec62c481d05b4cdbdab10f25c5375d6d24bdbe00b4d9")
		{
			return Failure(Error::Kind::Os, "missing exact libcuda SHA256 manifest row");
		}
		fd = open(path.c_str(), O_RDONLY);
		if (fd >= 0 && fd <= STDERR_FILENO)
		{
			const auto duplicate = fcntl(fd, F_DUPFD, STDERR_FILENO + 1);
			close(fd);
			fd = duplicate;
		}
		if (fd < 0 || fstat(fd, &identity) || !S_ISREG(identity.st_mode))
		{
			return Failure(Error::Kind::Os, "cannot open original driver inode");
		}
		auto hashed = Hash();
		if (!hashed)
		{
			return cmt::Failure(hashed.error());
		}
		sha256 = std::move(*hashed);
		struct stat after_hash{};
		if (sha256 != expected || fstat(fd, &after_hash) || !SameFile(identity, after_hash))
		{
			return Failure(Error::Kind::Os, "original driver file hash/identity mismatch");
		}
		{
			auto fds = GpuFds();
			if (!fds)
			{
				return cmt::Failure(fds.error());
			}
			gpu_fds_before = std::move(*fds);
		}
		handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
		if (!handle)
		{
			return Failure(Error::Kind::Os, "original driver dlopen failed");
		}
		{
			auto fds = GpuFds();
			if (!fds)
			{
				return cmt::Failure(fds.error());
			}
			gpu_fds_after = std::move(*fds);
		}
		Dl_info info{};
		const auto symbol = dlsym(handle, "cuInit"); // Identify the image; NEVER call cuInit.
		if (!symbol || !dladdr(symbol, &info) || !info.dli_fbase || !info.dli_fname ||
			std::filesystem::canonical(info.dli_fname, path_error) != path || path_error)
		{
			return Failure(Error::Kind::Os, "loaded object does not match explicit driver");
		}
		struct stat loaded_identity{};
		if (stat(info.dli_fname, &loaded_identity) || !SameFile(identity, loaded_identity))
		{
			return Failure(Error::Kind::Os, "loaded driver inode changed");
		}

		struct Bounds
		{
			std::uintptr_t base, entry;
			std::uint64_t file_offset = 0;
			bool found = false;
		} bounds{reinterpret_cast<std::uintptr_t>(info.dli_fbase),
				 reinterpret_cast<std::uintptr_t>(info.dli_fbase) + row->internal_memcpy_va};

		constexpr std::size_t reviewed_bytes = 0x2b8;
		dl_iterate_phdr(
			[](dl_phdr_info* image, std::size_t, void* opaque)
			{
				auto& b = *static_cast<Bounds*>(opaque);
				if (image->dlpi_addr != b.base)
				{
					return 0;
				}
				for (std::size_t i = 0; i < image->dlpi_phnum; i++)
				{
					const auto& p = image->dlpi_phdr[i];
					const auto segment_begin = b.base + p.p_vaddr;
					if (p.p_type == PT_LOAD && (p.p_flags & (PF_R | PF_X | PF_W)) == (PF_R | PF_X) &&
						b.entry >= segment_begin && b.entry - segment_begin <= p.p_filesz &&
						reviewed_bytes <= p.p_filesz - (b.entry - segment_begin))
					{
						b.found = true;
						b.file_offset = p.p_offset + (b.entry - segment_begin);
					}
				}
				return 1;
			},
			&bounds);
		if (!bounds.found)
		{
			return Failure(Error::Kind::Os, "helper is not fully inside an original RX ELF segment");
		}
		unsigned char original[reviewed_bytes];
		if (pread(fd, original, sizeof original, static_cast<off_t>(bounds.file_offset)) !=
				static_cast<ssize_t>(sizeof original) ||
			std::memcmp(reinterpret_cast<const void*>(bounds.entry), row->internal_memcpy_prologue,
						sizeof row->internal_memcpy_prologue) ||
			std::memcmp(reinterpret_cast<const void*>(bounds.entry), original, sizeof original))
		{
			return Failure(Error::Kind::Os, "helper prologue/body differs from original file");
		}
		std::ifstream maps("/proc/self/maps");
		while (std::getline(maps, line))
		{
			unsigned long begin = 0, end = 0, offset = 0, inode = 0;
			unsigned major_id = 0, minor_id = 0;
			char permissions[5]{};
			if (std::sscanf(line.c_str(), "%lx-%lx %4s %lx %x:%x %lu", &begin, &end, permissions, &offset,
							&major_id, &minor_id, &inode) == 7 &&
				begin <= bounds.entry && bounds.entry < end && reviewed_bytes <= end - bounds.entry &&
				std::strcmp(permissions, "r-xp") == 0 &&
				offset + bounds.entry - begin == bounds.file_offset && inode == identity.st_ino &&
				major_id == major(identity.st_dev) && minor_id == minor(identity.st_dev))
			{
				mapping = line;
			}
		}
		if (mapping.empty())
		{
			return Failure(Error::Kind::Os, "helper executable mapping identity is not verified");
		}
		driver_copy = reinterpret_cast<DriverCopy>(bounds.entry);
		auto tested = SelfTest();
		if (!tested)
		{
			fatal = true;
			reason = std::string("failure=driver_selftest ") + tested.error().Message();
		}
#endif
		return {};
	}

	void Load(const Options& config)
	{
		const auto start = Clock::now();
		auto loaded = LoadChecked(config);
		if (!loaded)
		{
			reason = std::string("skip=unsupported_driver_helper ") + loaded.error().Message();
			driver_copy = nullptr;
		}
		setup_seconds = std::chrono::duration<double>(Clock::now() - start).count();
	}

	Result<bool> Unchanged() const
	{
		struct stat current{};
		if (fd < 0 || fstat(fd, &current) || !SameFile(identity, current))
		{
			return false;
		}
		auto hashed = Hash();
		if (!hashed)
		{
			return cmt::Failure(hashed.error());
		}
		return *hashed == sha256;
	}
};

std::string UnsupportedReason(cmt::Variant variant, const cmt::CpuIsas& cpu, Cache cache)
{
	if (!cpu.Has(variant.isa))
	{
		return "skip=unsupported_isa isa=" + std::string(cmt::IsaName(variant.isa));
	}
	if (cache == Cache::Cold && !cmt::FlushSupported())
	{
		return "skip=unsupported_cache_eviction";
	}
	return {};
}

Result<void> MeasureChecked(benchmark::State& state, cmt::CopyLoop loop, cmt::Variant variant, Cache cache,
							std::size_t workers, Options config, bool driver_helper, cmt::Measurement* record)
{
	const auto bytes = static_cast<std::size_t>(state.range(0));
	const auto copies = std::max<std::size_t>(1, std::min(kMaxBatchCopies, config.batch_bytes / bytes));
	const auto reason = UnsupportedReason(variant, cmt::hwcopy::GetCpu(), cache);
	if (!std::empty(reason))
	{
		state.SkipWithError(reason.c_str());
		if (record)
		{
			record->status = "unsupported";
			record->reason = reason;
		}
		return {};
	}
	const auto available_result = cmt::HostMemoryAvailable(config.pages);
	if (!available_result)
	{
		state.SkipWithError("skip=capacity reason=host_capacity_unknown");
		if (record)
		{
			record->status = "capacity";
			record->reason = "host_capacity_unknown";
		}
		return {};
	}
	const auto available =
		std::min(*available_result, config.memory_limit.value_or(std::numeric_limits<std::size_t>::max()));
	const auto allocation = Capacity(bytes, copies, config.reserve_bytes, available, config.control);
	if (!allocation)
	{
		const auto message =
			fmt::format("skip=capacity bytes={} batch_copies={} available_bytes={} reserve_bytes={}", bytes,
						copies, available, config.reserve_bytes);
		state.SkipWithError(message.c_str());
		if (record)
		{
			record->status = "capacity";
			record->reason = "insufficient_host_capacity";
			record->observations["host_available_bytes"] = static_cast<double>(*available_result);
			record->observations["capacity_available_bytes"] = static_cast<double>(available);
			record->observations["capacity_reserve_bytes"] = static_cast<double>(config.reserve_bytes);
		}
		return {};
	}
	if (config.device)
	{
		CMT_CUDA(cudaSetDevice(*config.device));
	}
	const auto source_memory = SourceMemory(config.control);
	const auto destination_memory = DestinationMemory(config.control);
	Buffer src, dst;
	for (const auto& [buffer, memory] : {std::pair{&src, source_memory}, std::pair{&dst, destination_memory}})
	{
		auto allocated = buffer->Create(allocation->buffer_bytes, memory, config.pages);
		if (!allocated)
		{
			if (memory == CpuMemory::Pageable && config.pages == cmt::Pages::Thp &&
				allocated.error().IsUnsupported())
			{
				state.SkipWithError("skip=unsupported reason=cpu_page_policy_unavailable");
				if (record)
				{
					record->status = "unsupported";
					record->reason = "cpu_page_policy_unavailable";
				}
				return {};
			}
			return cmt::Failure(allocated.error());
		}
	}
	const auto stride = allocation->stride;
	double total_batch_seconds = 0;
	std::uint64_t epoch = 0;
	const auto run = [&](auto kernel) -> Result<void>
	{
		Workers<decltype(kernel)> team(workers, config.chunk_bytes, loop.ctx);
		CMT_RETURN_IF_ERROR(team.Start());
		const auto prepare = [&]
		{
			for (std::size_t batch = 0; batch < copies; batch++)
			{
				auto *source = src.data() + batch * stride, *destination = dst.data() + batch * stride;
				Prepare(source, destination, bytes, ++epoch);
				std::memset(source + bytes, 0xa5, kGuardBytes);
				std::memset(destination + bytes, 0x5a, kGuardBytes);
			}
			// Publish prepared WC stores before any lane can read its source.
			cmt::StoreFence();
		};
		if (cache == Cache::Warm)
		{
			// One complete untimed batch establishes real reuse even when --out
			// requests a single timed batch per Google repetition. Keep every byte
			// for validation; the next Prepare changes the source and poisons the
			// destination so warmup output cannot satisfy the timed byte check.
			prepare();
			for (std::size_t batch = 0; batch < copies; batch++)
			{
				team.Run(dst.data() + batch * stride, src.data() + batch * stride, bytes);
			}
			for (std::size_t batch = 0; batch < copies; batch++)
			{
				if (!Verified(src.data() + batch * stride, dst.data() + batch * stride, bytes, source_memory,
							  destination_memory) ||
					!GuardsIntact(src.data() + batch * stride, dst.data() + batch * stride, bytes))
				{
					return Failure(Error::Kind::Verify,
								   "CPU reuse warmup failed full-byte/guard verification");
				}
			}
		}
		for (auto iteration : state)
		{
			(void)iteration;
			cmt::SampleOut sample;
			if (record)
			{
				// Read smaps before preparing/conditioning data, never inside the copy timer.
				sample.cpu_backing.emplace();
				sample.cpu_backing->source_before.Record(src.HugeKiB());
				sample.cpu_backing->destination_before.Record(dst.HugeKiB());
			}
			prepare();
			if (cache == Cache::Cold)
			{
				for (std::size_t batch = 0; batch < copies; batch++)
				{
					if (source_memory != CpuMemory::PinnedWc)
					{
						cmt::FlushRange(src.data() + batch * stride, bytes);
					}
					if (destination_memory != CpuMemory::PinnedWc)
					{
						cmt::FlushRange(dst.data() + batch * stride, bytes);
					}
				}
			}
			const auto start = Clock::now();
			for (std::size_t batch = 0; batch < copies; batch++)
			{
				team.Run(dst.data() + batch * stride, src.data() + batch * stride, bytes);
			}
			benchmark::ClobberMemory();
			const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
			if (!std::isfinite(seconds) || seconds <= 0)
			{
				return Failure(Error::Kind::Verify, "CPU copy timer did not advance to a finite duration");
			}
			total_batch_seconds += seconds;
			state.SetIterationTime(seconds);
			for (std::size_t batch = 0; batch < copies; batch++)
			{
				if (!Verified(src.data() + batch * stride, dst.data() + batch * stride, bytes, source_memory,
							  destination_memory) ||
					!GuardsIntact(src.data() + batch * stride, dst.data() + batch * stride, bytes))
				{
					failed = true;
					state.SkipWithError("failure=verification full_byte_compare");
					return {};
				}
			}
			if (record)
			{
				// Correctness and all timed copying finished before the second snapshot.
				sample.cpu_backing->source_after.Record(src.HugeKiB());
				sample.cpu_backing->destination_after.Record(dst.HugeKiB());
				sample.seconds = seconds;
				record->samples.push_back(std::move(sample));
				record->copies_per_sample = static_cast<std::uint32_t>(copies);
			}
		}
		return {};
	};
	if (driver_helper)
	{
		CMT_RETURN_IF_ERROR(run(DriverKernel{}));
	}
	else
	{
		CMT_RETURN_IF_ERROR(cmt::hwcopy::WithLoop(loop.id, run));
	}
	// All helper lifetimes end inside run before either allocation is released.
	if (state.skipped())
	{
		return {};
	}
	if (static_cast<std::uint64_t>(state.iterations()) >
		static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / bytes / copies)
	{
		failed = true;
		state.SkipWithError("failure=counter_overflow physical_bytes");
		return {};
	}
	state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(bytes) *
							static_cast<std::int64_t>(copies));
	state.counters["payload_bytes"] = static_cast<double>(bytes);
	state.counters["available_bytes_before_alloc"] = static_cast<double>(available);
	state.counters["buffer_bytes_requested"] = static_cast<double>(allocation->buffer_bytes) * 2;
	state.counters["pinned_bytes_requested"] =
		config.control == Control::Pageable ? 0 : static_cast<double>(allocation->buffer_bytes);
	state.counters["capacity_budget_bytes"] = static_cast<double>(allocation->capacity_budget_bytes);
	state.counters["reserve_bytes"] = static_cast<double>(config.reserve_bytes);
	state.counters["streaming"] = StreamingStores(variant.method) ? 1 : 0;
	state.counters["streaming_load"] = variant.method == cmt::Method::NtLoad ? 1 : 0;
	state.counters["store_policy"] =
		StreamingStores(variant.method)													  ? 2
		: (variant.method == cmt::Method::Memcpy || variant.method == cmt::Method::Movsb) ? 0
																						  : 1;
	state.counters["isa_id"] = static_cast<unsigned>(variant.isa);
	state.counters["requested_copying_lanes"] = static_cast<double>(workers);
	state.counters["actual_copying_lanes"] =
		static_cast<double>(CopyingLanes(bytes, config.chunk_bytes, workers));
	state.counters["helper_threads"] = static_cast<double>(workers - 1);
	state.counters["active_helpers"] =
		static_cast<double>(CopyingLanes(bytes, config.chunk_bytes, workers) - 1);
	state.counters["chunk_bytes"] = static_cast<double>(config.chunk_bytes);
	state.counters["chunks_per_copy"] = static_cast<double>(ChunkCount(bytes, config.chunk_bytes));
	state.counters["allowed_cpu_count"] = static_cast<double>(config.allowed_cpus.size());
	state.counters["affinity_oversubscribed"] =
		CopyingLanes(bytes, config.chunk_bytes, workers) > config.allowed_cpus.size() ? 1 : 0;
	state.counters["cache_cold"] = cache == Cache::Cold ? 1 : 0;
	state.counters["batch_copies"] = static_cast<double>(copies);
	state.counters["untimed_warmup_copies"] = cache == Cache::Warm ? static_cast<double>(copies) : 0;
	state.counters["batch_payload_bytes"] = static_cast<double>(bytes) * static_cast<double>(copies);
	state.counters["physical_copies"] = static_cast<double>(state.iterations()) * static_cast<double>(copies);
	state.counters["physical_bytes"] =
		static_cast<double>(state.iterations()) * static_cast<double>(copies) * static_cast<double>(bytes);
	state.counters["batch_wall_seconds"] = total_batch_seconds;
	state.counters["copy_seconds"] = benchmark::Counter(total_batch_seconds / static_cast<double>(copies),
														benchmark::Counter::kAvgIterations);
	if (!driver_helper)
	{
		state.counters["clock_pair_ns_per_copy"] = clock_pair_ns / static_cast<double>(copies);
	}
	state.counters["verified"] = 1;
	state.counters["libc_runtime_dispatch"] = !driver_helper && variant.method == cmt::Method::Memcpy ? 1 : 0;
	if (driver_helper)
	{
		state.counters["driver_internal_helper"] = 1;
	}
	state.SetLabel(fmt::format(
		"CPU component {}; full-buffer footprint, no ring reuse or DMA; {}; caller plus {} helpers; "
		"whole chunks; handoff and per-chunk store fences timed; setup/first-touch/check/release excluded",
		ControlName(config.control), CacheName(cache, config.control), workers - 1));
	return {};
}

Result<std::string> SettingsFor(std::size_t bytes, cmt::Variant variant, Cache cache, std::size_t workers,
								const Options& config, bool driver_helper)
{
	const auto copies = std::max<std::size_t>(1, std::min(kMaxBatchCopies, config.batch_bytes / bytes));
	const auto allocation =
		Capacity(bytes, copies, 0, std::numeric_limits<std::size_t>::max(), config.control);
	JsonWriter writer;
	writer.start_object();
	cmt::JsonField(writer, "method", "cpu_copy", true);
	cmt::JsonField(writer, "cpu_loop",
				   driver_helper ? "verified_libcuda_quickcopy" : cmt::VariantText(variant));
	cmt::JsonField(writer, "control", ControlName(config.control));
	cmt::JsonField(writer, "requested_copying_lanes", workers);
	cmt::JsonField(writer, "actual_copying_lanes", CopyingLanes(bytes, config.chunk_bytes, workers));
	cmt::JsonField(writer, "helper_threads", workers - 1);
	cmt::JsonField(writer, "active_helpers", CopyingLanes(bytes, config.chunk_bytes, workers) - 1);
	cmt::JsonField(writer, "chunk_bytes", config.chunk_bytes);
	cmt::JsonField(writer, "chunks_per_copy", ChunkCount(bytes, config.chunk_bytes));
	cmt::JsonField(writer, "chunk_ownership", "cyclic_whole_chunks_caller_lane_zero");
	cmt::JsonField(writer, "cache_policy", CacheName(cache, config.control));
	cmt::JsonField(writer, "source_memory", MemoryName(SourceMemory(config.control)));
	cmt::JsonField(writer, "destination_memory", MemoryName(DestinationMemory(config.control)));
	cmt::JsonField(writer, "pageable_pages", cmt::PagesName(config.pages));
	cmt::JsonField(writer, "pinned_allocator", config.device ? "cudaHostAlloc" : "none");
	cmt::JsonField(writer, "cuda_device", config.device);
	cmt::JsonField(writer, "gpu_transfer_in_timer", false);
	cmt::JsonField(writer, "allocation_scope", "independent_full_buffer_pairs_no_ring_reuse");
	cmt::JsonField(writer, "resource_lifetime", "one_benchmark_invocation_recreated_per_google_repetition");
	cmt::JsonField(writer, "buffer_bytes_requested_per_side",
				   allocation ? std::optional<std::size_t>(allocation->buffer_bytes) : std::nullopt);
	cmt::JsonField(writer, "pinned_bytes_requested",
				   allocation ? std::optional<std::size_t>(config.device ? allocation->buffer_bytes : 0)
							  : std::nullopt);
	cmt::JsonField(writer, "capacity_budget_bytes",
				   allocation ? std::optional<std::size_t>(allocation->capacity_budget_bytes) : std::nullopt);
	cmt::JsonField(writer, "capacity_budget_scope",
				   "pageable mapping extent plus page-rounded pinned request; CUDA internal overhead "
				   "unknown; no reservation");
	cmt::JsonField(writer, "base_alignment_bytes", kBufferAlignment);
	cmt::JsonField(writer, "batch_target_bytes", config.batch_bytes);
	cmt::JsonField(writer, "batch_copies", copies);
	cmt::JsonField(writer, "untimed_warmup_copies", cache == Cache::Warm ? copies : 0);
	cmt::JsonField(writer, "reserve_bytes", config.reserve_bytes);
	cmt::JsonField(writer, "memory_limit_bytes", config.memory_limit);
	cmt::JsonField(writer, "store_fence", "per_chunk_inside_timer");
	cmt::JsonField(writer, "verification", "every_payload_byte_and_pair_tail_guards_after_each_batch");
	cmt::JsonField(writer, "placement", "helpers_inherit_caller_affinity_no_numa_memory_binding");
	cmt::JsonField(writer, "first_touch", "caller_before_timer");
	cmt::JsonField(writer, "allowed_cpu_count", config.allowed_cpus.size());
	cmt::JsonKey(writer, "allowed_cpus");
	writer.start_array();
	for (std::size_t i = 0; i < config.allowed_cpus.size(); ++i)
	{
		if (i != 0)
		{
			writer.append_comma();
		}
		writer.append_raw(fmt::format("{}", config.allowed_cpus[i]));
	}
	writer.end_array();
	cmt::JsonField(writer, "affinity_oversubscribed",
				   CopyingLanes(bytes, config.chunk_bytes, workers) > config.allowed_cpus.size());
	writer.end_object();
	auto settings = cmt::FinishJson(writer);
	if (!settings)
	{
		return cmt::Failure(settings.error());
	}
	// FinishJson adds a document newline; settings_json is an embedded object.
	settings->pop_back();
	return settings;
}

Result<cmt::Measurement*> MeasurementFor(std::size_t bytes, cmt::Variant variant, Cache cache,
										 std::size_t workers, const Options& config, bool driver_helper)
{
	if (config.output.empty())
	{
		return static_cast<cmt::Measurement*>(nullptr);
	}
	auto settings = SettingsFor(bytes, variant, cache, workers, config, driver_helper);
	if (!settings)
	{
		return cmt::Failure(settings.error());
	}
	cmt::Measurement* record = nullptr;
	auto found = std::find_if(measurements.begin(), measurements.end(), [&](const auto& value)
							  { return value.bytes == bytes && value.settings_json == *settings; });
	if (found == measurements.end())
	{
		measurements.emplace_back();
		record = &measurements.back();
		record->config.path = &cpu_path;
		record->config.flush = cache == Cache::Cold;
		record->direction_name = "cpu";
		record->allocation = ControlName(config.control);
		record->config.pages = config.pages;
		record->pages_requested = cmt::PagesName(config.pages);
		record->bytes = bytes;
		record->copies_per_sample = static_cast<std::uint32_t>(
			std::max<std::size_t>(1, std::min(kMaxBatchCopies, config.batch_bytes / bytes)));
		record->settings_json = std::move(*settings);
		record->execution = driver_helper ? "isolated_driver_cpu_helper" : "direct_cpu_loop";
		record->status = "pending";
	}
	else
	{
		record = &*found;
	}
	return record;
}

Result<void> PreviewPoints(const Options& config)
{
	JsonWriter writer;
	writer.start_array();
	std::size_t remaining_bytes = kMaxPreviewBytes - kPreviewArrayEnvelopeBytes;
	bool first = true;
	for (const auto workers : config.workers)
	{
		for (const auto cache : config.caches)
		{
			for (const auto& variant : config.variants)
			{
				for (const auto bytes : config.sizes)
				{
					auto settings = SettingsFor(bytes, variant, cache, workers, config, false);
					if (!settings)
					{
						return cmt::Failure(settings.error());
					}
					if (remaining_bytes < kPreviewPointEnvelopeBytes ||
						settings->size() > remaining_bytes - kPreviewPointEnvelopeBytes)
					{
						return Failure(
							Error::Kind::Usage,
							"CPU preview exceeds 16 MiB; select an explicit smaller --cpu-sizes list");
					}
					remaining_bytes -= settings->size() + kPreviewPointEnvelopeBytes;
					if (!first)
					{
						writer.append_comma();
					}
					first = false;
					writer.start_object();
					cmt::JsonField(writer, "direction", "cpu", true);
					cmt::JsonField(writer, "bytes", bytes);
					cmt::JsonKey(writer, "settings");
					writer.append_raw(*settings);
					writer.end_object();
				}
			}
		}
	}
	writer.end_array();
	auto points = cmt::FinishJson(writer);
	if (!points)
	{
		return cmt::Failure(points.error());
	}
	return cmt::WriteStdout(*points);
}

void MeasureCommon(benchmark::State& state, cmt::CopyLoop loop, cmt::Variant variant, Cache cache,
				   std::size_t workers, Options config, bool driver_helper)
{
	auto found = MeasurementFor(static_cast<std::size_t>(state.range(0)), variant, cache, workers, config,
								driver_helper);
	if (!found)
	{
		failed = true;
		state.SkipWithError(found.error().Message());
		return;
	}
	auto* record = *found;
	if (record)
	{
		if (record->status == "failed" || record->status == "capacity" || record->status == "unsupported")
		{
			// A later repetition must not turn an incomplete cell into a successful
			// shorter sample set. Retry this cell in a new process instead.
			state.SkipWithError("previous repetition failed or was skipped; cell is terminal");
			return;
		}
	}
	auto measured = MeasureChecked(state, loop, variant, cache, workers, config, driver_helper, record);
	if (!measured)
	{
		failed = true;
		const auto message = fmt::format("failure=cpu_component {} code={}", measured.error().Message(),
										 measured.error().GetCode());
		state.SkipWithError(message.c_str());
		if (record)
		{
			record->status = "failed";
			record->reason = message;
		}
	}
	if (record)
	{
		if (state.skipped())
		{
			if (record->status != "capacity" && record->status != "unsupported")
			{
				record->status = "failed";
				if (record->reason == "not measured")
				{
					record->reason = "copy or counter verification failed";
				}
			}
			record->verified = false;
			record->samples.clear();
		}
		else
		{
			record->status = "complete";
			record->reason.clear();
			record->verified = true;
		}
	}
}

void Measure(benchmark::State& state, cmt::CopyLoop loop, cmt::Variant variant, Cache cache,
			 std::size_t workers, Options config)
{
	MeasureCommon(state, loop, variant, cache, workers, config, false);
}

Result<int> DriverMain(const Options& config)
{
	DriverHelper helper;
	helper.Load(config);
	if (config.self_test)
	{
		fmt::print("driver helper self-test: {}\n", helper.reason.empty() ? "passed" : helper.reason);
		benchmark::Shutdown();
		return helper.fatal ? 6 : helper.reason.empty() ? 0 : 77;
	}
	benchmark::AddCustomContext("cpu_driver_helper_schema_version", "1");
	benchmark::AddCustomContext("driver_helper_status", helper.reason.empty() ? "verified_for_measurement"
														: helper.fatal		  ? "failed"
																			  : "unsupported");
	benchmark::AddCustomContext("driver_helper_reason", helper.reason);
	benchmark::AddCustomContext("driver_file", helper.path);
	benchmark::AddCustomContext("driver_sha256", helper.sha256);
	benchmark::AddCustomContext("driver_version", helper.version);
	benchmark::AddCustomContext("driver_table_manifest", config.driver_manifest);
	benchmark::AddCustomContext("driver_quickcopy_offset",
								helper.row ? fmt::format("{:#x}", helper.row->internal_memcpy_va) : "absent");
	benchmark::AddCustomContext("driver_prologue", helper.row ? [&] {
        std::string value;
        for (auto byte : helper.row->internal_memcpy_prologue){
            value += fmt::format("{:02x}", byte);
}
        return value;
    }()
                                                              : "absent");
	benchmark::AddCustomContext("driver_executable_mapping", helper.mapping);
	benchmark::AddCustomContext("driver_reviewed_body_bytes", "696");
	benchmark::AddCustomContext("driver_original_body_equal", helper.reason.empty() ? "1" : "0");
	benchmark::AddCustomContext("driver_bounded_selftest", helper.reason.empty() ? "passed"
														   : helper.fatal		 ? "failed"
																				 : "unsupported");
	benchmark::AddCustomContext("driver_file_inode", fmt::format("{}", helper.identity.st_ino));
	benchmark::AddCustomContext("driver_file_device", fmt::format("{}:{}", major(helper.identity.st_dev),
																  minor(helper.identity.st_dev)));

	benchmark::AddCustomContext("driver_gpu_fds_before_dlopen", helper.gpu_fds_before);
	benchmark::AddCustomContext("driver_gpu_fds_after_dlopen", helper.gpu_fds_after);
	benchmark::AddCustomContext("driver_setup_seconds", fmt::format("{}", helper.setup_seconds));
	benchmark::AddCustomContext(
		"driver_timing_scope",
		"Original private libcuda internal_memcpy CPU helper only, not CUDA API CPU time or a GPU transfer. "
		"Library loading/constructors, identity checks, bounded self-test, allocation, first touch, "
		"preparation, eviction and verification excluded. Caller plus N-1 helpers copy whole chunks; "
		"per-chunk completion fence and handoff inside manual timing. Pageable buffers only; no "
		"cuInit/context/GPU-buffer calls or patching by benchmark; constructor code "
		"runs "
		"outside timing.");
	benchmark::AddCustomContext("driver_body_review",
								"x86_64 615.71.09 offset0x285900..0x285bb8; original full-file SHA "
								"pinned; loaded prologue and all "
								"reviewed body bytes equal original RX inode. SSE non-temporal bulk "
								"stores, scalar/ordinary tails; "
								"no internal completion fence. No fixed per-size store policy claimed.");
	for (const auto cache : config.caches)
	{
		for (const auto workers : config.workers)
		{
			const auto name = fmt::format("driver_internal/libcuda_quickcopy/{}/workers_{}/chunk_{}",
										  CacheName(cache, config.control), workers, config.chunk_bytes);
			auto* registered =
				benchmark::RegisterBenchmark(
					name.c_str(),
					[&helper, cache, workers, config](benchmark::State& state)
					{
						if (!helper.reason.empty())
						{
							auto found = MeasurementFor(static_cast<std::size_t>(state.range(0)),
														cmt::Variant{cmt::Method::Memcpy, cmt::Isa::Sse2},
														cache, workers, config, true);
							if (!found)
							{
								failed = true;
								state.SkipWithError(found.error().Message());
								return;
							}
							auto* record = *found;
							if (record)
							{
								record->status = helper.fatal ? "failed" : "unsupported";
								record->reason = helper.reason;
							}
							failed = failed || helper.fatal;
							state.SkipWithError(helper.reason.c_str());
							return;
						}
						MeasureCommon(state, cmt::CopyLoop{},
									  cmt::Variant{cmt::Method::Memcpy, cmt::Isa::Sse2}, cache, workers,
									  config, true);
					})
					->UseManualTime();
			if (!config.output.empty())
			{
				registered->Iterations(1);
			}
			for (const auto bytes : config.sizes)
			{
				registered->Arg(static_cast<std::int64_t>(bytes));
			}
		}
	}
	benchmark::RunSpecifiedBenchmarks();
	auto Unchanged = helper.reason.empty() ? helper.Unchanged() : Result<bool>(true);
	if (!Unchanged)
	{
		return cmt::Failure(Unchanged.error());
	}
	if (!*Unchanged)
	{
		failed = true;
		std::fprintf(stderr, "cpu_loops: failure=driver_identity_changed_after_measurement\n");
	}
	if (failed)
	{
		for (auto& record : measurements)
		{
			record.status = "failed";
			record.reason = "driver helper validation failed";
			record.verified = false;
			record.samples.clear();
		}
	}
	for (auto& record : measurements)
	{
		// The helper hash is verified hexadecimal; retain its exact binary identity with each row.
		if (helper.sha256.size() == 64 &&
			helper.sha256.find_first_not_of("0123456789abcdef") == std::string::npos)
		{
			record.settings_json.pop_back();
			record.settings_json += fmt::format(R"(,"driver_sha256":"{}"}})", helper.sha256);
		}
	}
	CMT_RETURN_IF_ERROR(Publish(config));
	benchmark::Shutdown();
	return failed ? 6 : helper.reason.empty() ? 0 : 77;
}

Result<void> SelfTest()
{
	std::size_t variants = 0, copies = 0, stale_checks = 0;
	if (StreamingStores(cmt::Method::NtLoad) || StreamingStores(cmt::Method::Memcpy) ||
		!StreamingStores(cmt::Method::NtStore) || !StreamingStores(cmt::Method::NtStorePrefetch))
	{
		return Failure(Error::Kind::Os, "load hints/libc were mislabeled as explicit streaming stores");
	}
	if (Capacity(4096, 1, 1024, 1024, Control::Pageable) ||
		Capacity(std::numeric_limits<std::size_t>::max(), kMaxBatchCopies, 1, 1, Control::FillWc))
	{
		return Failure(Error::Kind::Os, "capacity preflight did not reject insufficient/overflowing sizes");
	}
	if (CopyingLanes(4096, kDefaultChunkBytes, kMaxWorkers) != 1 ||
		CopyingLanes(kDefaultChunkBytes + 1, kDefaultChunkBytes, kMaxWorkers) != 2)
	{
		return Failure(Error::Kind::Os, "copying lane count split one chunk across participants");
	}
	cmt::CpuIsas absent;
	if (std::empty(UnsupportedReason({cmt::Method::Cached, cmt::Isa::Avx2}, absent, Cache::Warm)))
	{
		return Failure(Error::Kind::Os, "unsupported ISA did not produce an explicit skip");
	}
	for (const auto& variant : cmt::Variants(cmt::kNativeArch))
	{
		if (!cmt::hwcopy::GetCpu().Has(variant.isa))
		{
			continue;
		}
		const auto id = cmt::hwcopy::GetVariantId(variant);
		if (!id)
		{
			return Failure(Error::Kind::Os, "compiled variant missing its dispatch id");
		}
		variants++;
		auto tested = cmt::hwcopy::WithLoop(
			*id,
			[&](auto kernel) -> Result<void>
			{
				for (const auto workers : {std::size_t{1}, std::size_t{3}, std::size_t{7}})
				{
					for (const auto chunk_bytes : {std::size_t{17}, std::size_t{4096}, kDefaultChunkBytes})
					{
						Workers<decltype(kernel)> team(workers, chunk_bytes, {});
						CMT_RETURN_IF_ERROR(team.Start());
						for (const auto bytes :
							 {std::size_t{1}, std::size_t{17}, std::size_t{4097}, std::size_t{196624}})
						{
							std::vector<unsigned char> src(bytes + 2, 0), dst(bytes + 2, 0);
							for (std::uint64_t epoch = 1; epoch <= 3; epoch++)
							{
								Prepare(std::data(src) + 1, std::data(dst) + 1, bytes, epoch);
								if (cmt::FlushSupported())
								{
									cmt::FlushRange(std::data(src) + 1, bytes);
									cmt::FlushRange(std::data(dst) + 1, bytes);
								}
								team.Run(std::data(dst) + 1, std::data(src) + 1, bytes);
								copies++;
								if (!Verified(std::data(src) + 1, std::data(dst) + 1, bytes) || dst.front() ||
									dst.back())
								{
									return Failure(Error::Kind::Os,
												   "full-byte/tail/guard copy verification failed");
								}
							}
							Prepare(std::data(src) + 1, std::data(dst) + 1, bytes, 4);
							stale_checks++;
							if (Verified(std::data(src) + 1, std::data(dst) + 1, bytes))
							{
								return Failure(Error::Kind::Os,
											   "verification accepted stale destination bytes");
							}
						}
					}
				}
				return {};
			});
		CMT_RETURN_IF_ERROR(tested);
	}
	fmt::print("cpu_loops self-test: {} supported variants, {} full-byte/guard/tail copies, {} stale-output "
			   "checks; capacity/overflow/unsupported-ISA checks passed\n",
			   variants, copies, stale_checks);
	return {};
}

} // namespace

Result<int> Run(int argc, char** argv)
{
	auto parsed = ParseOptions(argc, argv);
	if (!parsed)
	{
		return cmt::Failure(parsed.error());
	}
	auto config = std::move(*parsed);
	if (config.help)
	{
		fmt::print(
			"CPU options (Google Benchmark flags remain available):\n"
			"  --cpu-preview             exact Point settings JSON on stdout; no measurement or CUDA init\n"
			"  --out PATH                 canonical verified samples; one batch per Google repetition\n"
			"  --cpu-grid                 exact 18 "
			"sizes: 4K,16K,64K,128K,512K,1M,2M,4M,16M,64M,256M,512M,1G,2G,4G,8G,16G,32G\n"
			"  --cpu-sizes=4K,1M..1G      explicit unique sizes, LO..HI powers of two, instead of the grid\n"
			"  --cpu-control=pageable    pageable baseline (default); or fill-cached, fill-wc, "
			"drain-cached, drain-wc\n"
			"  --cpu-device=0            explicit CUDA device required only for pinned controls\n"
			"  --cpu-pages=small|thp     pageable endpoint policy (default small); CUDA pinned backing "
			"unchanged\n"
			"  --cpu-cache=warm|cold|both repeated buffers or eviction of cacheable sides; never flush WC\n"
			"  --cpu-workers=all         1 through 16 requested copying lanes; explicit lists also accepted\n"
			"  --cpu-chunk-bytes=1M      caller lane 0 plus N-1 helpers cyclically own whole chunks\n"
			"  --cpu-loops=NAME,...      exact native catalog names in list order, including prefetch; "
			"no aliases\n"
			"  --cpu-prefetch            separate explicit-prefetch loops only; default excludes them\n"
			"  --cpu-batch-bytes=1M       target independent-pair payload per batch; at most 256 copies\n"
			"  --cpu-reserve-bytes=1G     headroom required in addition to both buffer allocations\n"
			"  --cpu-memory-limit=1G      optional additional capacity ceiling; never bypasses "
			"host/cgroup limits\n"
			"  --cpu-driver-helper=PATH   separate opt-in original private libcuda CPU helper "
			"measurement\n"
			"  --cpu-driver-manifest=PATH exact source driver identity manifest; currently "
			"reviewed615.71.09 only\n"
			"  --cpu-self-test            changing-pattern bytes, tails, guards, workers, stale output "
			"and skip checks\n"
			"Default: pageable warm 4K..64M, one caller, 1M chunks, non-prefetch loops, no CUDA init.\n"
			"--cpu-loops rejects empty/unknown/duplicate names and conflicts with --cpu-prefetch, "
			"driver-helper and self-test. Unsupported CPU ISAs remain explicit skipped cells.\n"
			"Preview includes unsupported compiled variants for explicit skip coverage; it rejects --out, "
			"help, self-test, driver-helper, duplicate CPU options and Google Benchmark flags.\n"
			"Preview JSON is limited to 16 MiB; use explicit smaller --cpu-sizes lists when needed.\n"
			"THP is a request, not a guarantee. Canonical samples observe overlapping pageable VMAs before "
			"preparation and after correctness; these totals do not prove exact payload huge-page coverage.\n"
			"CUDA owns pinned backing; it remains unqueried. Both sides retain full payloads "
			"until checks; this does not model ring reuse or DMA overlap.\n"
			"Actual copying lanes = min(requested lanes, ceil(payload/chunk)); small payloads can use "
			"only the caller; inactive persistent helpers are not dispatched.\n"
			"Helpers inherit caller affinity (set externally); no NUMA memory binding. One Google iteration "
			"is one independent-pair batch; payload_bytes is the total per copy.\n"
			"Warm runs first complete and byte-check one untimed batch, then change the source and poison "
			"the destination before timing. No cache residency is promised.\n"
			"Each Google repetition recreates resources and its warmup; process wall time includes that "
			"untimed copying and checking.\n"
			"real_time is manual batch time; copy_seconds is per-copy latency. cpu_time includes untimed "
			"preparation/verification.\n");
		fmt::print("Compiled native loop names (unsupported ISAs become skipped cells; defaults exclude "
				   "prefetch):\n");
		for (const auto& variant : cmt::Variants(cmt::kNativeArch))
		{
			fmt::print("  {}\n", cmt::VariantText(variant));
		}
		return 0;
	}
	if (config.self_test && config.driver_helper.empty())
	{
		CMT_RETURN_IF_ERROR(SelfTest());
		return 0;
	}
	auto allowed_cpus = cmt::GetThreadAffinity();
	if (!allowed_cpus)
	{
		return cmt::Failure(allowed_cpus.error());
	}
	if (allowed_cpus->empty())
	{
		return Failure(Error::Kind::Os, "CPU benchmark has no allowed logical CPUs");
	}
	config.allowed_cpus = std::move(*allowed_cpus);
	if (config.preview)
	{
		CMT_RETURN_IF_ERROR(PreviewPoints(config));
		return 0;
	}
	benchmark::Initialize(&argc, argv);
	if (benchmark::ReportUnrecognizedArguments(argc, argv))
	{
		return 2;
	}
	if (!config.driver_helper.empty())
	{
		return DriverMain(config);
	}
	benchmark::AddCustomContext("cpu_schema_version", "3");
	benchmark::AddCustomContext("cpu_control", ControlName(config.control));
	benchmark::AddCustomContext("cpu_pageable_pages", cmt::PagesName(config.pages));
	benchmark::AddCustomContext("cpu_chunk_bytes", fmt::format("{}", config.chunk_bytes));
	benchmark::AddCustomContext("cpu_cuda_device",
								config.device ? fmt::format("{}", *config.device) : "none");
	benchmark::AddCustomContext("cpu_loop_set", config.explicit_loops ? "selected_catalog_variants"
												: config.prefetch	  ? "explicit_prefetch"
																	  : "production_nonprefetch");
	benchmark::AddCustomContext("cpu_store_policy_ids",
								"0=libc/microcode runtime-selected;1=ordinary vector stores;2=explicit "
								"non-temporal stores; streaming_load=1 denotes load hint only");
	benchmark::AddCustomContext(
		"cpu_cache_semantics",
		"warm=reused buffers without residency promise; cold=only cacheable sides evicted before "
		"independent-pair batch; WC side unflushed");
	benchmark::AddCustomContext(
		"cpu_timing_semantics",
		"manual seconds per prepared batch; allocation, preparation, eviction, full-byte verification "
		"and release outside timing; caller/helper handoff, copying, per-chunk store fences and completion "
		"inside");
	benchmark::AddCustomContext(
		"cpu_output_clock",
		"real_time is manual batch time; copy_seconds is batch time/copies; cpu_time includes untimed "
		"preparation/verification and must not be used as copy latency");
	benchmark::AddCustomContext("cpu_batch_semantics",
								"one Google iteration is a batch; bytes_processed counts N*batch_copies; "
								"copy_seconds normalizes per N-byte copy");
	benchmark::AddCustomContext(
		"cpu_reuse_semantics",
		"warm runs one untimed fully checked batch before measured batches; fresh pattern and poisoned "
		"destination before every timed batch; resources and warmup recreated for each Google repetition; "
		"warmup work remains in process wall time");
	benchmark::AddCustomContext("cpu_reserve_bytes", fmt::format("{}", config.reserve_bytes));
#ifdef __GLIBC__
	benchmark::AddCustomContext("libc_version", gnu_get_libc_version());
#endif
	benchmark::AddCustomContext(
		"cpu_host_memory",
		fmt::format("{} -> {}; pageable uses HostBuffer {} request; pinned uses cudaHostAlloc cached/WC; "
					"full-buffer "
					"pairs, no staging ring reuse or DMA; caller first touch; no NUMA memory binding; "
					"streaming labels describe instructions independently of allocation policy",
					MemoryName(SourceMemory(config.control)), MemoryName(DestinationMemory(config.control)),
					cmt::PagesName(config.pages)));
	benchmark::AddCustomContext(
		"cpu_placement",
		"helpers inherit caller affinity; CMT_WORKER_CPUS/NUMA overrides are not applied; caller lane 0; "
		"requested lanes can exceed actual copying lanes or allowed logical CPUs");
	std::string affinity;
	for (auto cpu : config.allowed_cpus)
	{
		if (!affinity.empty())
		{
			affinity += ',';
		}
		affinity += fmt::format("{}", cpu);
	}
	benchmark::AddCustomContext("cpu_allowed_cpus", affinity);
	benchmark::AddCustomContext("libc_memcpy_isa",
								"runtime-selected by libc; no fixed SSE/AVX implementation claimed");
	double minimum_clock_seconds = std::numeric_limits<double>::max();
	for (std::size_t i = 0; i < 1024; i++)
	{
		const auto start = Clock::now();
		minimum_clock_seconds =
			std::min(minimum_clock_seconds, std::chrono::duration<double>(Clock::now() - start).count());
	}
	clock_pair_ns = minimum_clock_seconds * 1e9;
	benchmark::AddCustomContext("minimum_two_clock_reads_ns", fmt::format("{}", minimum_clock_seconds * 1e9));
	// Worker order is explicit: finish the one-thread matrix before multi-worker cases.
	for (const auto workers : config.workers)
	{
		for (const auto cache : config.caches)
		{
			for (const auto& variant : config.variants)
			{
				const auto id = cmt::hwcopy::GetVariantId(variant);
				if (!id)
				{
					return Failure(Error::Kind::Os, "compiled variant missing dispatch id");
				}
				std::string name = cmt::VariantText(variant);
				if (config.extended)
				{
					const bool streaming = StreamingStores(variant.method);
					name = fmt::format("{}/{}/{}/{}/workers_{}/chunk_{}", ControlName(config.control),
									   streaming ? "streaming" : "ordinary", name,
									   CacheName(cache, config.control), workers, config.chunk_bytes);
				}
				auto* registered = benchmark::RegisterBenchmark(name.c_str(), Measure, cmt::CopyLoop{*id, {}},
																variant, cache, workers, config)
									   ->UseManualTime();
				if (!config.output.empty())
				{
					registered->Iterations(1);
				}
				for (const auto bytes : config.sizes)
				{
					registered->Arg(static_cast<std::int64_t>(bytes));
				}
			}
		}
	}
	benchmark::RunSpecifiedBenchmarks();
	CMT_RETURN_IF_ERROR(Publish(config));
	benchmark::Shutdown();
	return failed ? 6 : 0;
}

int main(int argc, char** argv)
{
	cmt::InstallFatalAllocationHandler();
	auto result = Run(argc, argv);
	if (!result)
	{
		std::fprintf(stderr, "cpu_loops: %s\n", result.error().Message());
		return result.error().ExitCode();
	}
	return *result;
}
