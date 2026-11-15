#include "config.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "arch/copy.h"
#include "common/aliases.h"
#include "common/defines.h"
#include "common/error.h"
#include "common/helpers.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "common/log.h"

namespace cmt {
namespace {

template <class T, std::size_t N, class Lookup>
Result<void> ReadNumber(const Lookup& lookup, const char (&name)[N], T& value)
{
	const char* text = lookup(name);
	if (text == nullptr || *text == '\0')
	{
		return {};
	}
	using Number = std::conditional_t<std::is_same_v<T, bool>, int, T>;
	Number number{};
	const char* end = text + std::strlen(text);
	const auto parsed = std::from_chars(text, end, number);
	if (parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(number))
	{
		const int code =
			parsed.ec == std::errc::result_out_of_range || !std::isfinite(number) ? ERANGE : EINVAL;
		return Failure(Error::Kind::Usage, name, code);
	}
	if constexpr (std::is_same_v<T, bool>)
	{
		if (number != 0 && number != 1)
		{
			return Failure(Error::Kind::Usage, name, EINVAL);
		}
	}
	value = static_cast<T>(number);
	return {};
}

std::optional<std::string> OptionalString(const char* v)
{
	if (v == nullptr)
	{
		return std::nullopt;
	}
	return std::string(v);
}

template <std::size_t N>
Result<void> ValidateCopyLoop(const char (&setting)[N], const std::optional<std::string>& name,
							  const HookConfig& config, const CpuIsas& cpu)
{
	if (!name)
	{
		return {};
	}
	CopyCtx context;
	context.flush_batch = config.copy_flush_bytes;
	const auto selected = hwcopy::Select(*name, context, config.copy_isa, cpu);
	if (!selected)
	{
		return Failure(Error::Kind::Usage, setting, EINVAL);
	}
	if (selected->fell_back)
	{
		return Failure(Error::Kind::Unsupported, setting, ENOTSUP);
	}
	return {};
}

} // namespace

const char* StagingAllocationName(StagingAllocation allocation)
{
	switch (allocation)
	{
	case StagingAllocation::CachedPinned:
		return "hostalloc";
	case StagingAllocation::WriteCombinedPinned:
		return "hostalloc_wc";
	case StagingAllocation::Registered4K:
		return "mmap_4k";
	}
	return "invalid";
}

bool DebugLoggingEnabled()
{
	static const bool enabled = []
	{
		bool value = false;
		const auto parsed = ReadNumber(std::getenv, "MEMCPY_HOOK_DEBUG", value);
		return parsed && value;
	}();
	return enabled;
}

bool HookConfig::WantH2HPatch() const
{
	return !fixed_benchmark && tune_table_path.empty() &&
		   (impl.has_value() || impl_h2d.has_value() || impl_d2h.has_value() || bench_dma);
}

bool HookConfig::WantInternalMemcpyPatch() const
{
	return !fixed_benchmark && tune_table_path.empty() && impl.has_value();
}

Result<std::vector<unsigned>> ParseCpuList(const char* text)
{
	std::vector<unsigned> cpus;
	if (text == nullptr || *text == '\0')
	{
		return cpus;
	}
	std::string_view remaining(text);
	for (;;)
	{
		const auto comma = remaining.find(',');
		auto field = remaining.substr(0, comma);
		const auto first = field.find_first_not_of(" \t\r\n\v\f");
		if (first == std::string_view::npos)
		{
			return UsageError("worker CPU list contains an empty entry");
		}
		field = field.substr(first, field.find_last_not_of(" \t\r\n\v\f") - first + 1);
		auto parsed = ParseThreadCpus(field);
		if (!parsed)
		{
			return Failure(parsed.error());
		}
		cpus.insert(cpus.end(), parsed->begin(), parsed->end());
		if (comma == std::string_view::npos)
		{
			break;
		}
		remaining.remove_prefix(comma + 1);
	}
	return cpus;
}

PlacementInputs HookPlacementInputs(const HookConfig& config)
{
	if (config.tune_table)
	{
		return {{}, &config.tune_table->worker_placement};
	}
	return {config.pool_cpus, nullptr};
}

Result<RunOverrides> ParseRunOverrides(std::string_view text, const CpuIsas& cpu)
{
	RunOverrides overrides;
	std::set<std::string_view> keys;
	for (;;)
	{
		const auto comma = text.find(',');
		const auto entry = text.substr(0, comma);
		const auto equal = entry.find('=');
		if (equal == std::string_view::npos || equal == 0 || equal + 1 == entry.size())
		{
			return UsageError("override requires nonempty KEY=VALUE assignments");
		}
		const auto key = entry.substr(0, equal);
		const auto value = entry.substr(equal + 1);
		if (!keys.insert(key).second)
		{
			return UsageError("duplicate run override key");
		}
		if (key == "progressive_pinning")
		{
			if (value == "auto")
			{
				overrides.progressive_pinning = ProgressivePinning::Auto;
			}
			else if (value == "on")
			{
				overrides.progressive_pinning = ProgressivePinning::On;
			}
			else if (value == "off")
			{
				overrides.progressive_pinning = ProgressivePinning::Off;
			}
			else
			{
				return UsageError("progressive_pinning override must be auto, on or off");
			}
		}
		else if (key == "numa")
		{
			if (value != "on" && value != "off")
			{
				return UsageError("numa override must be on or off");
			}
			overrides.numa = value == "on";
		}
		else if (key == "pinning_max_mib")
		{
			std::uint64_t mib = 0;
			const auto parsed = std::from_chars(value.data(), value.data() + value.size(), mib);
			if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || mib == 0 ||
				mib > (kMaxBytes >> 20))
			{
				return UsageError(
					"pinning_max_mib override must be a decimal MiB count from 1 through 1048576");
			}
			overrides.pinning_max_bytes = mib << 20;
		}
		else if (key == "cpu_loop")
		{
			const auto selected = hwcopy::Select(value, CopyCtx{}, std::nullopt, cpu);
			if (!selected)
			{
				return UsageError("unknown cpu_loop override");
			}
			if (selected->fell_back)
			{
				return Unsupported("cpu_loop override is unavailable on this CPU");
			}
			overrides.cpu_loop = selected->loop;
		}
		else if (key == "staging_memory")
		{
			if (value != "cached_pinned" && value != "wc_pinned")
			{
				return UsageError("staging_memory override must be cached_pinned or wc_pinned");
			}
			overrides.wc_slots = value == "wc_pinned";
		}
		else if (key == "pipeline_workers" || key == "slots_per_worker" || key == "chunk_bytes")
		{
			std::size_t number = 0;
			const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
			if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || number == 0)
			{
				return UsageError("numeric staging overrides require positive decimal integers");
			}
			if (key == "pipeline_workers")
			{
				if (number > kMaxPipelineWorkers)
				{
					return UsageError("pipeline_workers override exceeds the copying-thread limit");
				}
				overrides.pipeline_workers = static_cast<unsigned>(number);
			}
			else if (key == "slots_per_worker")
			{
				if (number > kMaxStagingBuffers)
				{
					return UsageError("slots_per_worker override exceeds the staging-buffer limit");
				}
				overrides.slots_per_worker = static_cast<uint32_t>(number);
			}
			else
			{
				if (number < kMinSlotBytes || number > kMaxSlotBytes || number % kChunkSizeGranularity != 0)
				{
					return UsageError("chunk_bytes override must be 64 KiB through 64 MiB, in 4 KiB units");
				}
				overrides.chunk_bytes = number;
			}
		}
		else
		{
			return UsageError("unknown run override key");
		}
		if (comma == std::string_view::npos)
		{
			return overrides;
		}
		text.remove_prefix(comma + 1);
	}
}

std::string FormatRunOverrides(const RunOverrides& overrides)
{
	std::string text;
	const auto Append = [&](std::string_view key, std::string_view value)
	{
		if (!text.empty())
		{
			text += ',';
		}
		text += key;
		text += '=';
		text += value;
	};
	if (overrides.pipeline_workers)
	{
		Append("pipeline_workers", std::to_string(*overrides.pipeline_workers));
	}
	if (overrides.slots_per_worker)
	{
		Append("slots_per_worker", std::to_string(*overrides.slots_per_worker));
	}
	if (overrides.chunk_bytes)
	{
		Append("chunk_bytes", std::to_string(*overrides.chunk_bytes));
	}
	if (overrides.cpu_loop)
	{
		const auto variant = hwcopy::VariantOf(overrides.cpu_loop->id);
		Append("cpu_loop",
			   LoopNameText(PinnedName(variant.method, variant.isa, overrides.cpu_loop->ctx.write_back)));
	}
	if (overrides.numa)
	{
		Append("numa", *overrides.numa ? "on" : "off");
	}
	if (overrides.pinning_max_bytes)
	{
		Append("pinning_max_mib", std::to_string(*overrides.pinning_max_bytes >> 20));
	}
	if (overrides.wc_slots)
	{
		Append("staging_memory", *overrides.wc_slots ? "wc_pinned" : "cached_pinned");
	}
	if (overrides.progressive_pinning)
	{
		const auto mode = *overrides.progressive_pinning;
		Append("progressive_pinning", mode == ProgressivePinning::Auto ? "auto"
									  : mode == ProgressivePinning::On ? "on"
																	   : "off");
	}
	return text;
}

const char* PathVariable(const EnvLookup& getenv_fn, const char* name, bool secure)
{
	const char* value = getenv_fn(name);
	if (value == nullptr || *value == '\0')
	{
		return nullptr;
	}
	if (secure)
	{
		LogLine("config: {} ignored: this process runs with elevated privileges (AT_SECURE)", name);
		return nullptr;
	}
	return value;
}

Result<HookConfig> ParseConfig(const EnvLookup& getenv_fn, const cmt::CpuIsas& cpu)
{
	HookConfig c;
	const bool secure = SecureExecution();
	if (const char* expected_path = PathVariable(getenv_fn, "CMT_EXPECTED_RUN", secure))
	{
		c.fixed_benchmark = true;
		c.progressive_pinning = ProgressivePinning::Off;
		auto pinning = ReadBenchmarkPinning(expected_path);
		if (!pinning)
		{
			return Failure(pinning.error());
		}
		c.benchmark_pinning = std::move(*pinning);
	}
	const char* v = nullptr;

	static constexpr Error removed_controls[] = {
		{Error::Kind::Usage, "STAGING_PIPELINE_MODE", EINVAL},
		{Error::Kind::Usage, "STAGING_HMM_D2H", EINVAL},
		{Error::Kind::Usage, "STAGING_HMM_FALLBACK", EINVAL},
		{Error::Kind::Usage, "STAGING_HMM_BLOCKS", EINVAL},
		{Error::Kind::Usage, "STAGING_HMM_THREADS", EINVAL},
		{Error::Kind::Usage, "STAGING_HMM_KERNEL", EINVAL},
		{Error::Kind::Usage, "STAGING_HMM_CHUNK_KIB", EINVAL},
		{Error::Kind::Usage, "STAGING_HMM_MIN_KIB", EINVAL},
		{Error::Kind::Usage, "STAGING_PIPELINE_PACE_GBPS", EINVAL},
		{Error::Kind::Usage, "STAGING_PIPELINE_PACE_QUANTUM_KIB", EINVAL},

		{Error::Kind::Usage, "STAGING_THREADS_COPY_NUM_THREADS", EINVAL},
		{Error::Kind::Usage, "STAGING_THREADS_COPY_MPMC", EINVAL},
		{Error::Kind::Usage, "STAGING_THREADS_COPY_POOL_H2D", EINVAL},
		{Error::Kind::Usage, "STAGING_THREADS_COPY_POOL_D2H", EINVAL},
		{Error::Kind::Usage, "STAGING_THREADS_COPY_SPIN_US", EINVAL},
		{Error::Kind::Usage, "STAGING_AUTO_CPUS", EINVAL},
		{Error::Kind::Usage, "STAGING_AUTO_CPUS_LOG", EINVAL},
		{Error::Kind::Usage, "STAGING_PIPELINE_STREAMS", EINVAL},
	};
	for (const Error& removed : removed_controls)
	{
		if (getenv_fn(removed.Message()) != nullptr)
		{
			return Failure(removed);
		}
	}
	int buffers = static_cast<int>(c.num_buffers);
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "STAGING_THREADS_COPY_NUM_BUFFERS", buffers));
	c.num_buffers = static_cast<unsigned>(std::clamp(buffers, 0, static_cast<int>(kMaxStagingBuffers)));

	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "STAGING_BUFFER_SIZE", c.staging_buffer_kib));
	constexpr std::uint32_t kMaxDriverStagingKiB = std::numeric_limits<std::uint32_t>::max() / 1024;
	if (c.staging_buffer_kib > 0 && static_cast<std::uint32_t>(c.staging_buffer_kib) > kMaxDriverStagingKiB)
	{
		return Failure(Error::Kind::Usage, "STAGING_BUFFER_SIZE exceeds the driver byte-size limit", ERANGE);
	}
	if (c.staging_buffer_kib > 0)
	{
		c.pipeline_chunk_bytes = static_cast<uint64_t>(c.staging_buffer_kib) * 1024ull;
	}

	int pinning_min_kib = static_cast<int>(c.pinning_min_bytes / 1024);
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "CMT_PINNING_MIN_KIB", pinning_min_kib));
	c.pinning_min_bytes = pinning_min_kib > 0 ? static_cast<uint64_t>(pinning_min_kib) * 1024 : 0;
	int pinning_step_mib = static_cast<int>(c.pinning_step_bytes >> 20);
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "CMT_PINNING_STEP_MIB", pinning_step_mib));
	c.pinning_step_bytes = static_cast<uint64_t>(std::clamp(pinning_step_mib, 1, 65536)) << 20;
	int pinning_first_mib = static_cast<int>(c.pinning_first_bytes >> 20);
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "CMT_PINNING_FIRST_MIB", pinning_first_mib));
	c.pinning_first_bytes = static_cast<uint64_t>(std::clamp(pinning_first_mib, 1, 65536)) << 20;
	int pinning_read_only = 0;
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "CMT_PINNING_READ_ONLY", pinning_read_only));
	if (pinning_read_only != 0 && pinning_read_only != 1)
	{
		return UsageError("CMT_PINNING_READ_ONLY must be 0 or 1");
	}
	c.pinning_read_only = pinning_read_only == 1;
	std::int64_t pinning_max_mib = static_cast<std::int64_t>(c.pinning_max_bytes >> 20);
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "CMT_PINNING_MAX_MIB", pinning_max_mib));
	if (pinning_max_mib > 0 &&
		static_cast<std::uint64_t>(pinning_max_mib) > (std::numeric_limits<std::uint64_t>::max() >> 20))
	{
		return Failure(Error::Kind::Usage, "CMT_PINNING_MAX_MIB exceeds the byte-size limit", ERANGE);
	}
	c.pinning_max_bytes = pinning_max_mib > 0 ? static_cast<std::uint64_t>(pinning_max_mib) << 20 : 0;

	int pipeline_workers = static_cast<int>(c.pipeline_workers);
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "STAGING_PIPELINE_WORKERS", pipeline_workers));
	c.pipeline_workers =
		static_cast<unsigned>(std::clamp(pipeline_workers, 1, static_cast<int>(kMaxPipelineWorkers)));
	if ((v = getenv_fn("STAGING_SLOT_ALLOC")) != nullptr)
	{
		const std::string_view s(v);
		if (s == "hostalloc")
		{
			c.slot_alloc = StagingAllocation::CachedPinned;
		}
		else if (s == "hostalloc_wc")
		{
			c.slot_alloc = StagingAllocation::WriteCombinedPinned;
		}
		else if (s == "mmap_4k")
		{
			c.slot_alloc = StagingAllocation::Registered4K;
		}

		else
		{
			return Failure(Error::Kind::Usage,
						   "STAGING_SLOT_ALLOC must be hostalloc, hostalloc_wc or mmap_4k", EINVAL);
		}
	}

	c.impl = OptionalString(getenv_fn("STAGING_COPY_NT_IMPL"));
	c.impl_h2d = OptionalString(getenv_fn("STAGING_COPY_NT_IMPL_H2D"));
	c.impl_d2h = OptionalString(getenv_fn("STAGING_COPY_NT_IMPL_D2H"));
	if ((v = getenv_fn("STAGING_COPY_ISA")) != nullptr)
	{
		const auto isa = ParseIsa(NativeArch(), v);
		if (!isa)
		{
			return Failure(Error::Kind::Usage, "STAGING_COPY_ISA is not an ISA of this architecture", EINVAL);
		}
		if (!cpu.Has(*isa))
		{
			return Failure(Error::Kind::Unsupported, "STAGING_COPY_ISA is unavailable on this CPU", ENOTSUP);
		}
		c.copy_isa = isa;
	}

	auto pool_cpus = ParseCpuList(getenv_fn("STAGING_THREADS_COPY_CPUS"));
	if (!pool_cpus)
	{
		return Failure(Error::Kind::Usage, "STAGING_THREADS_COPY_CPUS contains an invalid CPU list", EINVAL);
	}
	c.pool_cpus = std::move(*pool_cpus);

	long copy_flush_bytes = static_cast<long>(c.copy_flush_bytes);
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "STAGING_COPY_FLUSH_BYTES", copy_flush_bytes));
	c.copy_flush_bytes =
		copy_flush_bytes <= 0
			? 0
			: std::max(kFlushLineBytes, static_cast<std::size_t>(copy_flush_bytes)) & ~(kFlushLineBytes - 1);
	CMT_RETURN_IF_ERROR(ValidateCopyLoop("STAGING_COPY_NT_IMPL", c.impl, c, cpu));
	CMT_RETURN_IF_ERROR(ValidateCopyLoop("STAGING_COPY_NT_IMPL_H2D", c.impl_h2d, c, cpu));
	CMT_RETURN_IF_ERROR(ValidateCopyLoop("STAGING_COPY_NT_IMPL_D2H", c.impl_d2h, c, cpu));
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "STAGING_BENCH_DMA", c.bench_dma));
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "MEMCPY_HOOK_DEBUG", c.debug));
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "MEMCPY_HOOK_NO_EAGER", c.no_eager));
	int allow_unverified_rows = 0;
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS", allow_unverified_rows));
	c.allow_unverified_rows = allow_unverified_rows == 1;

	RunOverrides overrides;
	if (const char* text = getenv_fn("CMT_OVERRIDE"))
	{
		auto parsed = ParseRunOverrides(text, cpu);
		if (!parsed)
		{
			return Failure(parsed.error());
		}
		overrides = std::move(*parsed);
		if (c.fixed_benchmark)
		{
			return UsageError("CMT_OVERRIDE cannot change a fixed measured benchmark");
		}
	}
	c.progressive_pinning = overrides.progressive_pinning.value_or(c.progressive_pinning);
	c.pinning_max_bytes = overrides.pinning_max_bytes.value_or(c.pinning_max_bytes);
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "CMT_REMOTE_CONFIG", c.remote_config));
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "CMT_CUDA_HOST_ALLOC_HOOK", c.cuda_host_alloc_hook));
	CMT_RETURN_IF_ERROR(ReadNumber(getenv_fn, "CMT_CUDA_HOST_ALLOC_PRIVATE", c.cuda_host_alloc_private));

	if ((v = PathVariable(getenv_fn, "CMT_CONFIG", secure)) != nullptr)
	{
		if (c.fixed_benchmark)
		{
			return UsageError("CMT_CONFIG cannot change a fixed measured benchmark");
		}
		c.tune_table_path = v;
		auto table = tuning::Load(v, overrides);
		if (!table)
		{
			return Failure(table.error());
		}
		c.tune_table = std::move(*table);
	}
	else if (overrides.HasStaging())
	{
		return UsageError("staging overrides require CMT_CONFIG and an existing staged selection");
	}
	c.valid = true;
	return c;
}

namespace {

std::string ShowBool(bool b)
{
	return b ? "1" : "0";
}

std::string ShowOptional(const std::optional<std::string>& s)
{
	return s ? *s : "<unset>";
}

struct Knob
{
	const char* name;
	const char* default_value;
	const char* meaning;
	std::string (*effective)(const HookConfig&);
};

const Knob kKnobs[] = {
	{"STAGING_THREADS_COPY_NUM_BUFFERS", "0",
	 "pinned staging slots of the API pipeline (>=1 enables, max 32)",
	 [](const HookConfig& c) { return std::to_string(c.num_buffers); }},
	{"STAGING_BUFFER_SIZE", "<unset>",
	 "staging size in KiB: driver pool and pipeline chunk (pipeline default 4096)",
	 [](const HookConfig& c)
	 {
		 return std::to_string(c.staging_buffer_kib) + " -> chunk " +
				std::to_string(c.pipeline_chunk_bytes / 1024) + " KiB";
	 }},
	{"CMT_PINNING_MIN_KIB", "4096",
	 "automatic progressive pinning requires verified loaded kernel optimizations; "
	 "smaller ranges are never registered; the kernel validates large-page mapping eligibility",
	 [](const HookConfig& c) { return std::to_string(c.pinning_min_bytes / 1024); }},
	{"CMT_PINNING_STEP_MIB", "64",
	 "progressive pinning: maximum piece size; piece k+1 registers while piece k's DMA runs",
	 [](const HookConfig& c) { return std::to_string(c.pinning_step_bytes >> 20); }},
	{"CMT_PINNING_FIRST_MIB", "4", "progressive pinning: first piece size; later pieces double up to STEP",
	 [](const HookConfig& c) { return std::to_string(c.pinning_first_bytes >> 20); }},
	{"CMT_PINNING_READ_ONLY", "0",
	 "1 = use supported read-only registration for HtoD; unsupported capability uses writable "
	 "registration",
	 [](const HookConfig& c) { return std::string(c.pinning_read_only ? "1" : "0"); }},
	{"CMT_PINNING_MAX_MIB", "0",
	 "progressive pinning: most bytes this process keeps pinned at once (0 = the memory of the GPU each "
	 "copy targets; `cmt run --override pinning_max_mib=N` replaces it)",
	 [](const HookConfig& c) { return std::to_string(c.pinning_max_bytes >> 20); }},
	{"STAGING_PIPELINE_WORKERS", "2",
	 "copying threads per CUDA context: caller plus N-1 helpers, each with slots and a stream (1..16)",
	 [](const HookConfig& c) { return std::to_string(c.pipeline_workers); }},
	{"STAGING_SLOT_ALLOC", "hostalloc",
	 "slot memory: hostalloc (cuMemHostAlloc, cached), hostalloc_wc (write-combined stores, "
	 "slow CPU reads; the pool also serves DtoH), mmap_4k (mmap + cuMemHostRegister)",
	 [](const HookConfig& c) { return std::string(StagingAllocationName(c.slot_alloc)); }},
	{"STAGING_COPY_ISA", "<unset>",
	 "one ISA for every copy loop (sse2|avx2|avx512, neon|sve); unset = the widest this CPU runs",
	 [](const HookConfig& c)
	 { return c.copy_isa ? std::string(IsaName(*c.copy_isa)) : std::string("<unset>"); }},
	{"STAGING_COPY_NT_IMPL", "(architecture default)",
	 "host copy loop: METHOD[@ISA][+flush|+clean] (arch/copy.h)",
	 [](const HookConfig& c) { return ShowOptional(c.impl); }},
	{"STAGING_COPY_NT_IMPL_H2D", "<unset>", "copy loop override for staging fills",
	 [](const HookConfig& c) { return ShowOptional(c.impl_h2d); }},
	{"STAGING_COPY_NT_IMPL_D2H", "<unset>", "copy loop override for staging drains",
	 [](const HookConfig& c) { return ShowOptional(c.impl_d2h); }},
	{"STAGING_THREADS_COPY_CPUS", "<unset>", "CPU list for staging workers, worker i -> cpus[i % n]",
	 [](const HookConfig& c)
	 {
		 std::string s;
		 for (unsigned cpu : c.pool_cpus)
		 {
			 s += (std::empty(s) ? "" : ",") + std::to_string(cpu);
		 }
		 return std::empty(s) ? std::string("<none>") : s;
	 }},
	{"STAGING_COPY_FLUSH_BYTES", "4096",
	 "batch of the *_flush loops (copy, flush, repeat); 0 = flush after the whole copy",
	 [](const HookConfig& c) { return std::to_string(c.copy_flush_bytes); }},
	{"STAGING_BENCH_DMA", "0", "1 = skip host copy legs of staged copies (bandwidth aid, stale data)",
	 [](const HookConfig& c) { return ShowBool(c.bench_dma); }},
	{"CMT_CONFIG", "<unset>",
	 "measured tune.json: derive per-size driver/staging settings once; binary patches are disabled",
	 [](const HookConfig& c)
	 {
		 if (std::empty(c.tune_table_path))
		 {
			 return std::string("<unset>");
		 }
		 return c.tune_table != nullptr ? std::to_string(std::size(c.tune_table->rows)) + " rows"
										: std::string("<invalid: ignored>");
	 }},
	{"CMT_REMOTE_CONFIG", "0",
	 "1 = use a CMT_CONFIG measured on another CPU/GPU model (set by `cmt run --remote-config`); "
	 "0 = such a table is ignored, with one line on stderr",
	 [](const HookConfig& c) { return ShowBool(c.remote_config); }},
	{"CMT_CUDA_HOST_ALLOC_HOOK", "0",
	 "1 = serve cuMemHostAlloc/cudaHostAlloc of 2 MiB or more with pre-faulted anonymous memory that is then "
	 "registered (set by `cmt run --cuda-host-alloc-hook on`); 0 = the driver allocates",
	 [](const HookConfig& c) { return ShowBool(c.cuda_host_alloc_hook); }},
	{"CMT_CUDA_HOST_ALLOC_PRIVATE", "0",
	 "1 = with CMT_CUDA_HOST_ALLOC_HOOK=1, private instead of shared memory: huge pages under THP madvise, "
	 "but a "
	 "fork() child gets a copy, not a share (set by `cmt run --cuda-host-alloc-hook private`)",
	 [](const HookConfig& c) { return ShowBool(c.cuda_host_alloc_private); }},
	{"MEMCPY_HOOK_DEBUG", "0", "1 = load/patch logging on stderr",
	 [](const HookConfig& c) { return ShowBool(c.debug); }},
	{"MEMCPY_HOOK_NO_EAGER", "0", "1 = do not dlopen libcuda from the constructor",
	 [](const HookConfig& c) { return ShowBool(c.no_eager); }},
	{"MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS", "0",
	 "1 = patch with table rows not yet verified on this architecture's hardware (aarch64)",
	 [](const HookConfig& c) { return ShowBool(c.allow_unverified_rows); }},
};

} // namespace

void PrintConfig(const HookConfig& cfg, int fd)
{
	LogTo(fd, "{:<34} {:<24} {:<9} {}", "knob", "effective", "default", "meaning");
	for (const Knob& k : kKnobs)
	{
		LogTo(fd, "{:<34} {:<24} {:<9} {}", k.name, k.effective(cfg).c_str(), k.default_value, k.meaning);
	}
}

} // namespace cmt
