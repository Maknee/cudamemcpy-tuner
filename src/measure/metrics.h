#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <benchmark/benchmark.h>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <sched.h>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

#include "common/aliases.h"
#include "common/helpers.h"
#include "arch/copy.h"
#include "common/defines.h"
#include "common/error.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "machine.h"

namespace cmt {

struct PathSpec;

enum class CapacityLimit
{
	None,
	Device,
	Host
};

struct SizeCapacity
{
	std::size_t bytes;
	CapacityLimit limit;
	std::optional<std::size_t> caller_host_bytes;
	const char* reason = nullptr;
};

struct HostAllocationRequirements
{
	std::optional<std::size_t> buffer_alignment = 1;
	std::size_t fixed_pinned_bytes = 0;
	bool separate_pinned_budget = false;
	std::optional<std::size_t> pinned_available;
};

struct SizeCoverage
{
	std::size_t device_available;
	std::size_t device_usable;
	std::optional<std::size_t> host_available;
	std::optional<std::size_t> host_usable;
	unsigned device_multiplier;
	unsigned host_multiplier;
	std::vector<SizeCapacity> sizes;
	HostAllocationRequirements host_allocation;
	std::optional<std::size_t> pinned_usable;
	std::vector<std::size_t> Eligible() const;
};

SizeCoverage ComputeSizeCoverage(const std::vector<std::size_t>&, std::size_t device_available,
								 std::optional<std::size_t> host_available, unsigned device_multiplier,
								 unsigned host_multiplier, HostAllocationRequirements host_allocation = {});
std::string SizeCoverageJson(const SizeCoverage&);

enum class Alloc : std::uint8_t
{
	Register,
	HostAlloc,
	HostAllocWc
};
enum class Unpin : std::uint8_t
{
	End,
	Eager,
	Prereg
};

const char* AllocName(Alloc a);
const char* UnpinName(Unpin u);
Result<Dir> ParseDir(const std::string& s);
Result<Alloc> ParseAlloc(const std::string& s);
Result<Unpin> ParseUnpin(const std::string& s);

enum class Knob : std::uint16_t
{
	Direction = 1u << 0,
	Pages = 1u << 1,
	Alloc = 1u << 2,
	Fill = 1u << 3,
	Threads = 1u << 4,
	Slots = 1u << 5,
	SlotBytes = 1u << 6,
	Wc = 1u << 7,
	NoDma = 1u << 8,
	PinStep = 1u << 9,
	Unpin = 1u << 10,
};

struct Knobs
{
	unsigned bits = 0;
	constexpr Knobs() = default;

	constexpr Knobs(std::initializer_list<Knob> ks)
	{
		for (Knob k : ks)
		{
			bits |= static_cast<unsigned>(k);
		}
	}

	void Set(Knob k)
	{
		bits |= static_cast<unsigned>(k);
	}

	constexpr bool Has(Knob k) const
	{
		return bits & static_cast<unsigned>(k);
	}
};

struct Config
{
	const PathSpec* path = nullptr;
	Dir dir = Dir::HtoD;
	Pages pages = Pages::Small;
	Alloc alloc = Alloc::Register;
	LoopName cpu_loop;
	bool cpu_copy = true;
	int threads = 1;
	int slots = 4;
	std::size_t slot_bytes = 1u << 20;
	bool wc_slots = false;
	std::size_t pin_step = 32u << 20;
	Unpin unpin = Unpin::End;
	bool dma = true;
	bool flush = true;
	bool verify = true;
	int samples = 3;
	std::size_t min_bytes = 64u << 20;
	double budget_s = 0;
	int cpu = -1;
	int device = 0;
	std::string tag;
	std::vector<std::size_t> sizes;
	Knobs set_knobs;
	std::vector<unsigned> worker_allowed_cpus{};

	Result<void> Validate() const;
};

constexpr int kCsvSchema = 3;

enum class Verified : std::uint8_t
{
	Ok,
	Bad,
	Skip,
};
const char* VerifiedName(Verified v);

struct MeasurementRow
{
	std::string tag;
	std::string gpu;
	std::string path;
	std::string dir;
	std::string pages;
	std::string alloc;
	std::string fill;
	std::string unpin;
	int threads = 0;
	int slots = 0;
	std::size_t slot_bytes = 0;
	std::size_t pin_step = 0;
	bool flush = false;
	std::size_t size_bytes = 0;
	std::uint32_t reps = 0;
	std::size_t samples = 0;
	double gbps_best = 0;
	double gbps_med = 0;
	double gbps_min = 0;
	double gbps_h2d = 0;
	double gbps_d2h = 0;
	std::optional<long> huge_kib;
	Verified verified = Verified::Skip;
	bool dma = true;
	double setup_s = 0;
	double copy_s = 0;
	double teardown_s = 0;
	std::string phase = "-";
};

std::string CsvHeader();
std::string CsvRow(const MeasurementRow& r);

struct DeviceInfo;
struct PathSpec;
struct SampleOut;
class PathRunner;

const char* DirColumn(const Config&, const PathSpec&);
MeasurementRow ResultIdentity(const Config&, const DeviceInfo&, const PathRunner&, std::size_t bytes,
							  std::uint32_t reps, bool flushing, Verified);
void SummarizeResult(MeasurementRow&, const std::vector<SampleOut>&, unsigned moved_bytes);

Result<std::size_t> ParseSize(const std::string& s, std::size_t* error_offset = nullptr);

Result<std::vector<std::size_t>> ParseSizes(const std::string& s, std::size_t* error_offset = nullptr);

Result<std::vector<std::size_t>> Pow2Range(std::size_t lo, std::size_t hi);

std::string HumanSize(std::size_t bytes);

std::uint32_t RepsFor(std::size_t bytes, std::size_t min_bytes);

struct Summary
{
	double best = 0;
	double median = 0;
	double min = 0;
	std::size_t n = 0;
};

Summary Summarize(std::vector<double> rates);

double Gbps(double bytes, double seconds);

void PatternFill(void* p, std::size_t bytes, std::uint64_t seed, std::size_t threads = 16);

std::size_t PatternCheck(const void* p, std::size_t bytes, std::uint64_t seed, std::size_t threads = 16);

inline constexpr std::array<const char*, 7> kLifecycleCounters = {
	"lifecycle_host_alloc_s", "lifecycle_device_alloc_s", "lifecycle_pin_s",	  "lifecycle_copy_s",
	"lifecycle_unpin_s",	  "lifecycle_device_free_s",  "lifecycle_host_free_s"};

struct SampleOut
{
	double seconds = 0;

	struct VmaPages
	{
		std::size_t allocations = 0;
		std::size_t observed_allocations = 0;
		std::optional<long> min_huge_kib;
		std::optional<long> max_huge_kib;

		void Record(std::optional<long> huge_kib)
		{
			++allocations;
			if (huge_kib)
			{
				++observed_allocations;
				min_huge_kib = min_huge_kib ? std::min(*min_huge_kib, *huge_kib) : huge_kib;
				max_huge_kib = max_huge_kib ? std::max(*max_huge_kib, *huge_kib) : huge_kib;
			}
		}
	};

	struct HostBacking
	{
		VmaPages before_pin;
		VmaPages after_pin;
	};

	std::optional<HostBacking> host_backing;

	struct CpuBacking
	{
		VmaPages source_before;
		VmaPages source_after;
		VmaPages destination_before;
		VmaPages destination_after;
	};

	std::optional<CpuBacking> cpu_backing;

	struct Directions
	{
		double htod_gbps = 0;
		double dtoh_gbps = 0;
	};

	using Phases = cmt::CopyPhases;
	std::optional<Directions> directions;
	std::optional<Phases> phases;
	std::optional<std::array<double, kLifecycleCounters.size()>> lifecycle = std::nullopt;
};

struct Measurement
{
	Config config;
	std::size_t bytes = 0;
	unsigned process_round = 1;
	std::uint32_t copies_per_sample = 1;
	std::vector<SampleOut> samples;
	bool verified = false;
	std::string verification = "byte_equality";
	std::string status = "failed";
	std::string reason = "not measured";
	std::string execution = "bare_driver";
	std::string coverage = "measured";
	std::string measurement_class = "primary";
	std::string direction_name;
	std::string allocation = "not_recorded";
	std::string pages_requested;
	std::string settings_json;
	std::map<std::string, std::string> experiment;
	std::map<std::string, double> observations;
	std::optional<long> huge_kib;
};

const char* TimingBoundary(const Config& config);
Result<void> WriteMeasurements(const std::string& path, const std::string& source_machine,
							   const std::vector<Measurement>& measurements,
							   const std::map<std::string, double>& timings_seconds = {},
							   const std::string& diagnostics = {});

class PathRunner
{
public:
	virtual ~PathRunner() = default;
	virtual Result<void> Prepare(std::size_t b) = 0;
	virtual Result<void> ClearDestination(std::size_t b) = 0;
	virtual Result<SampleOut> Sample(std::size_t b, std::uint32_t reps) = 0;
	virtual Result<std::optional<std::size_t>> Verify(std::size_t b) = 0;
	virtual Result<void> Flush(std::size_t b) = 0;

	virtual std::optional<long> HugeKiB() const
	{
		return std::nullopt;
	}

	virtual std::string Alloc() const = 0;
	virtual std::vector<MeasurementRow> Rows(const MeasurementRow& base,
											 const std::vector<SampleOut>& samples) const;
};

struct RunContext
{
	const Config& cfg;
	const DeviceInfo& dev;
	std::size_t max_bytes;
};

struct PathSpec
{
	std::string name;
	std::string summary;
	std::string nvbandwidth;
	enum class Dirs : std::uint8_t
	{
		Either,
		HtoDOnly,
		Both,
		None,
	} dirs = Dirs::Either;
	unsigned device_bytes =
		1;
	unsigned host_bytes = 1;
	unsigned moved_bytes = 1;
	Knobs accepts;
	Function<Result<void>(const Config&) const> validate;
	Function<Result<std::unique_ptr<PathRunner>>(const RunContext&) const> make;
};

const std::array<PathSpec, 10>& BenchmarkPaths();
const PathSpec* FindPath(const std::string& name);
Result<const PathSpec*> PathNamed(const std::string& name);

namespace detail {
inline std::uint64_t SeedFor(std::size_t bytes, int which)
{
	return 0x5eed0000ull ^ (bytes * 31) ^ (std::uint64_t(which) << 56);
}
} // namespace detail

} // namespace cmt

namespace cmt {
namespace bench {
class SchemaReporter final : public benchmark::BenchmarkReporter
{
public:
	using Format = Function<std::vector<cmt::MeasurementRow>(
		const std::string&, const std::vector<cmt::SampleOut>&, cmt::Verified)>;

	explicit SchemaReporter(Format format, bool emit_header = true)
		: format_callback(std::move(format)), header(emit_header)
	{
	}

	bool ReportContext(const Context&) override;
	void ReportRuns(const std::vector<Run>&) override;
	void Finalize() override;

private:
	struct Group
	{
		std::vector<cmt::SampleOut> samples;
		cmt::Verified verified = cmt::Verified::Ok;
		bool failed = false;
	};

	std::map<std::string, Group> groups;
	Format format_callback;
	bool header;
};
} // namespace bench
} // namespace cmt
