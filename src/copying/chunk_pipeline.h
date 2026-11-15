#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include "arch/copy.h"
#include "common/aliases.h"
#include "common/helpers.h"
#include "config.h"
#include "copying/progressive_pinning.h"
#include "cuda/driver.h"

namespace cmt {
namespace ring {

constexpr std::uint64_t kStagingSlotAlignmentBytes = 4096;

inline std::uint64_t SlotStride(std::uint64_t chunk_bytes)
{
	return cmt::AlignUp(chunk_bytes, kStagingSlotAlignmentBytes).value_or(0);
}

constexpr cu::CUresult kNotStarted = -3;

struct PipelineCapacity
{
	unsigned workers = 0;
	uint32_t slots = 0;
	uint64_t pool_bytes = 0;
};

inline uint64_t RequiredPinnedBytes(const CopySettings& settings)
{
	const auto stride = SlotStride(settings.chunk_bytes);
	const auto slots = static_cast<uint64_t>(settings.workers) * settings.slots;
	if (stride == 0 || slots == 0 || stride > UINT64_MAX / slots)
	{
		return 0;
	}
	return slots * stride;
}

inline bool SettingsFit(const PipelineCapacity& capacity, const CopySettings& settings)
{
	const auto pinned_bytes = RequiredPinnedBytes(settings);
	return settings.workers >= 1 && settings.workers <= capacity.workers && settings.slots >= 1 &&
		   settings.slots <= capacity.slots && settings.chunk_bytes >= 1 && pinned_bytes != 0 &&
		   pinned_bytes <= capacity.pool_bytes;
}

struct CopyObservation
{
	uint64_t completed = 0;
	uint64_t requested_workers = 0;
	uint64_t actual_workers = 0;
	uint64_t slots = 0;
	uint64_t max_slots_used = 0;
	uint64_t slot_bytes = 0;
	uint64_t variant = 0;
	uint64_t write_back = 0;
	uint64_t flush_bytes = 0;
	uint64_t bench_dma = 0;
	uint64_t wc_slots = 0;
	uint64_t pace_quantum = 0;
	double pace_gbps = 0;
};

class ChunkPipeline
{
public:
	[[nodiscard]] static Result<std::unique_ptr<ChunkPipeline>> Create(cu::CUcontext context,
																	   const HookConfig& config,
																	   const PipelineCapacity& capacity,
																	   const ThreadPlacement& workers = {});
	~ChunkPipeline();
	ChunkPipeline(const ChunkPipeline&) = delete;
	ChunkPipeline& operator=(const ChunkPipeline&) = delete;

	[[nodiscard]] Result<void> Start(const ThreadPlacement& workers, bool one_cpu_each = false);

	[[nodiscard]] Result<void> Finish();

	bool Fits(const CopySettings& settings) const;
	cu::CUresult HtoD(const CopySettings& settings, cu::CUdeviceptr dst, const uint8_t* src,
					  std::size_t bytes, cu::CUstream stream, CopyObservation* observation = nullptr);
	cu::CUresult DtoH(const CopySettings& settings, uint8_t* dst, cu::CUdeviceptr src, std::size_t bytes,
					  cu::CUstream stream, CopyObservation* observation = nullptr);

private:
	struct State;
	explicit ChunkPipeline(std::unique_ptr<State> owned_state);
	std::unique_ptr<State> state;
};

struct PinnedPool
{
	void* staging = nullptr;
	void* mapping = nullptr;
	std::size_t mapping_len = 0;
};

[[nodiscard]] Result<void> AllocatePinnedPool(PinnedPool& pool, std::size_t bytes,
											  StagingAllocation allocation);
void ReleasePinnedPool(PinnedPool& pool);

} // namespace ring

namespace tune_audit {
constexpr uint32_t kVersion = 3;

enum class Outcome : uint32_t
{
	Unknown,
	Chunks,
	DriverRow,
	Fork,
	Capture,
	ProgressivePinning,
	NonPageable,
	NoPipeline,
	KnobsMismatch,
	NotStarted,
	Env,
	Driver,
	NotReady,
	LoopFallback,
	Nested,
	Count
};

struct Snapshot
{
	uint32_t version = kVersion;
	uint32_t size = sizeof(Snapshot);
	uint64_t calls = 0;
	uint64_t bytes = 0;
	uint64_t sync_calls = 0;
	uint64_t async_calls = 0;
	uint64_t errors = 0;
	uint64_t mixed = 0;
	uint64_t direction = 0;
	uint64_t copy_bytes = 0;
	uint64_t row_min = 0;
	uint64_t row_max = 0;
	uint64_t outcomes[static_cast<unsigned>(Outcome::Count)] = {};
	ring::CopyObservation job;
	pinning::Observation pinning;
};

static_assert(std::is_trivially_copyable_v<Snapshot>, "observation ABI copies only fixed-width values");
using Begin = int (*)(uint32_t version, uint32_t size);
using End = int (*)(Snapshot*, uint32_t size);
} // namespace tune_audit
} // namespace cmt

extern "C"
{
	__attribute__((visibility("default"))) int cmt_tune_audit_begin(uint32_t version, uint32_t size) noexcept;
	__attribute__((visibility("default"))) int cmt_tune_audit_end(cmt::tune_audit::Snapshot*,
																  uint32_t size) noexcept;
}
