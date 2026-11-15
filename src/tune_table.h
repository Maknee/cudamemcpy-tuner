#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "arch/copy.h"
#include "common/aliases.h"
#include "common/cpu.h"
#include "common/defines.h"
#include "common/error.h"
#include "common/memory.h"

namespace cmt {

constexpr unsigned kMaxPipelineWorkers = 16;
constexpr unsigned kMaxStagingBuffers = 32;
constexpr std::size_t kMinSlotBytes = std::size_t{64} << 10;
constexpr std::size_t kMaxSlotBytes = std::size_t{64} << 20;
constexpr std::size_t kChunkSizeGranularity = 4096;
constexpr uint64_t kDefaultPipelineChunk = 4ull << 20;
constexpr std::size_t kFlushLineBytes = 64;

constexpr char kSelectionPolicyVersion[] = "measured_steps_v4";
constexpr unsigned kMinMeasuredSamples = 2;

enum class StagingAllocation : std::uint8_t
{
	CachedPinned,
	WriteCombinedPinned,
	Registered4K
};
const char* StagingAllocationName(StagingAllocation allocation);

enum class Mode : std::uint8_t
{
	Driver,
	Chunks,
	ProgressivePinning
};

constexpr char kRegistrationLimitKey[] = "registration_limit";
constexpr char kExplicitOnly[] = "explicit_only";
constexpr std::uint64_t kLegacyDefaultRegistrationMiB = 1024;

struct PinningSettings
{
	uint64_t first_chunk_bytes = 4ull << 20;
	uint64_t max_chunk_bytes = 64ull << 20;
	std::optional<uint64_t> max_registered_bytes;
	bool read_only = false;
};

struct StagingSettings
{
	int workers = 0;
	int slots = 0;
	std::size_t slot_bytes = 0;
	std::string loop;
	bool wc_slots = false;

	bool operator==(const StagingSettings& other) const
	{
		return workers == other.workers && slots == other.slots && slot_bytes == other.slot_bytes &&
			   loop == other.loop && wc_slots == other.wc_slots;
	}
};

struct CopySettings
{
	unsigned workers = 0;
	uint32_t slots = 0;
	uint64_t chunk_bytes = 0;
	CopyLoop loop;
	bool wc_slots = false;
};

enum class ProgressivePinning : std::uint8_t
{
	Auto,
	On,
	Off
};

struct RunOverrides
{
	std::optional<unsigned> pipeline_workers;
	std::optional<uint32_t> slots_per_worker;
	std::optional<std::size_t> chunk_bytes;
	std::optional<CopyLoop> cpu_loop;
	std::optional<bool> wc_slots;
	std::optional<ProgressivePinning> progressive_pinning;
	std::optional<std::uint64_t> pinning_max_bytes;
	std::optional<bool>
		numa;

	bool HasStaging() const
	{
		return pipeline_workers || slots_per_worker || chunk_bytes || cpu_loop || wc_slots;
	}

	bool Empty() const
	{
		return !HasStaging() && !progressive_pinning && !pinning_max_bytes && !numa;
	}
};

Result<RunOverrides> ParseRunOverrides(std::string_view text, const CpuIsas& cpu = hwcopy::GetCpu());
std::string FormatRunOverrides(const RunOverrides& overrides);

struct Row
{
	Dir dir = Dir::HtoD;
	std::uint64_t min_bytes = 0;
	std::uint64_t max_bytes = 0;
	Mode mode = Mode::Driver;
	CopySettings settings;
	PinningSettings pinning;
	bool loop_fell_back = true;
	std::string loop_name{};
};

struct Table
{
	WorkerPlacement worker_placement;
	std::string source_cpu_model;
	std::string source_cpu_arch;
	std::string source_gpu_name;
	std::size_t max_pinned_bytes = 0;
	std::vector<Row> rows;

	const Row* Find(Dir d, std::uint64_t bytes) const;
	bool HasStagedCopies() const;
	bool HasProgressivePinning() const;
	std::size_t MaxPinnedBytes() const;
	std::size_t MaxPinnedBytes(bool wc_slots) const;
};

struct CopyRun
{
	Dir direction = Dir::HtoD;
	std::uint64_t bytes = 0;
	unsigned process_round = 1;
	Mode mode = Mode::Driver;
	StagingSettings settings;
	PinningSettings pinning;
	double seconds = 0;
	double gbps = 0;
	std::vector<double> sample_gbps;
	bool verified = false;
	std::string status = "failed";
	std::string execution;
	std::string timing_boundary = "completed_pageable_copy";
	std::string coverage = "measured";
	std::string measurement_class = "primary";
	std::string reason;
};

struct SelectionPolicy
{
	std::string version = kSelectionPolicyVersion;
	unsigned process_rounds = 1;
	double minimum_gain = 0.02;
	double tie_band = 0;
};

struct LocalCapabilities
{
	CpuIsas cpu = DetectCpuIsas();
	std::size_t max_pinned_bytes = std::numeric_limits<std::size_t>::max();
	bool progressive_pinning = false;
};

Result<void> ValidateCopyRun(const CopyRun& run);
Result<StagingSettings> ParseChunksSettings(Json settings);
Result<PinningSettings> ParsePinningSettings(Json settings);
Result<std::vector<CopyRun>> ReadCopyRuns(Json document);
double Median(std::vector<double> values);
void WriteCopyRun(JsonWriter& writer, const CopyRun& run, std::optional<Pages> pages = std::nullopt);
void WriteSettings(JsonWriter& writer, Mode mode, const StagingSettings& settings,
				   const PinningSettings& pinning = {});
Result<Table> BuildCopyLookup(const std::vector<CopyRun>& runs, const SelectionPolicy& policy,
							  const LocalCapabilities& capabilities);
Result<Table> ParseTuningResult(const std::string& text, const LocalCapabilities& capabilities = {});
const char* ModeName(Mode mode);

constexpr std::size_t kMaxSelectionLineBytes = std::size_t{1} << 20;

std::optional<std::size_t> SelectionLineBytes(std::string_view text);
Result<std::string> WithSelection(std::string_view rest, const std::vector<CopyRun>& runs,
								  std::string_view policy, Json request, Json source_machine);
Result<std::string> SelectionRequest(Json request);
Result<std::string> SelectionMachine(Json machine);
Result<std::string> RewriteSelection(std::string_view text, std::string_view rest);
Result<std::string> Reselect(const std::string& text);

Result<std::optional<PinningSettings>> ReadBenchmarkPinning(const char* path);

} // namespace cmt

namespace cmt {
namespace tuning {

constexpr std::size_t kMaxConfigBytes = std::size_t{256} << 20;
[[nodiscard]] cmt::Result<std::shared_ptr<const cmt::Table>> Load(const char* path,
																  const RunOverrides& overrides = {});
cmt::Result<cmt::Table> ReadSelection(std::string_view text, const cmt::LocalCapabilities& local,
									  std::uint64_t& file_bytes);

void Print(const cmt::Table& table, const std::string& path, int fd);

struct HostAllocArm
{
	std::vector<double> alloc_seconds;
	std::vector<double> free_seconds;
	std::vector<std::uint64_t> huge_kib;
};

struct HostAllocation
{
	std::string state = "not_measured";
	std::string reason;
	std::uint64_t bytes = 0;
	unsigned threads = 0;
	std::optional<std::string>
		thp;
	HostAllocArm driver;
	HostAllocArm shared;
	std::string private_state = "not_allowed";
	HostAllocArm private_memory;
};

inline constexpr const char* kHostAllocReasons[] = {"host_alloc off", "instant search",
													"max_pinned_mib below 2 MiB", "measurement failed"};
bool HostAllocClearlyFaster(const std::vector<double>& faster, const std::vector<double>& baseline,
							double minimum_gain = SelectionPolicy{}.minimum_gain);
void WriteHostAllocation(JsonWriter& writer, const HostAllocation& section);
Result<std::optional<HostAllocation>> ReadHostAllocation(Json document);

} // namespace tuning
} // namespace cmt
