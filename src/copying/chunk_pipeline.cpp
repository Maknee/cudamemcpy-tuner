#include "copying/chunk_pipeline.h"

namespace cmt {
namespace ring {
namespace {

constexpr std::size_t kThreadNameBytes = 16;

using namespace cu;

bool CanAllocatePinnedPool(StagingAllocation allocation)
{
	switch (allocation)
	{
	case StagingAllocation::CachedPinned:
		return (g_api.cuMemHostAlloc || g_api.cuMemAllocHost) && g_api.cuMemFreeHost;
	case StagingAllocation::WriteCombinedPinned:
		return g_api.cuMemHostAlloc && g_api.cuMemFreeHost;
	case StagingAllocation::Registered4K:
		return g_api.cuMemHostRegister && g_api.cuMemHostUnregister;
	}
	return false;
}

Result<void> MapPinnedPool(PinnedPool& pool, std::size_t bytes)
{
	const std::size_t page_bytes = BasePageBytes();
	const std::size_t alignment = page_bytes;
	const auto rounded = AlignUp(bytes, alignment);
	if (bytes == 0 || alignment < page_bytes || alignment % page_bytes != 0 || !rounded ||
		*rounded > SIZE_MAX - alignment)
	{
		return UsageError("Invalid or overflowing staging mapping size/alignment");
	}
	const std::size_t length = static_cast<std::size_t>(*rounded);
	const std::size_t mapped_bytes = length + alignment;
	void* mapping = mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (mapping == MAP_FAILED)
	{
		return OsError("Cannot map staging pool");
	}
	pool.mapping = mapping;
	pool.mapping_len = mapped_bytes;
	const auto aligned = AlignUp(reinterpret_cast<uintptr_t>(mapping), alignment);
	if (!aligned)
	{
		ReleasePinnedPool(pool);
		return UsageError("Staging mapping address alignment overflows");
	}
	auto* base = reinterpret_cast<uint8_t*>(*aligned);
	if (madvise(base, length, MADV_NOHUGEPAGE) != 0)
	{
		const int code = errno;
		ReleasePinnedPool(pool);
		return OsError("Cannot set staging page policy", code);
	}
	for (std::size_t offset = 0; offset < length; offset += page_bytes)
	{
		base[offset] = 0;
	}
	const CUresult status = g_api.cuMemHostRegister(base, length, 0);
	if (status != kSuccess)
	{
		ReleasePinnedPool(pool);
		return Failure(Error::Kind::Cuda, "Cannot register staging pool", status);
	}
	pool.staging = base;
	return {};
}

} // namespace

Result<void> AllocatePinnedPool(PinnedPool& pool, std::size_t bytes, StagingAllocation allocation)
{
	if (bytes == 0 || pool.staging != nullptr || pool.mapping != nullptr)
	{
		return UsageError("Pinned pool allocation requires nonzero bytes and an empty owner");
	}
	if (!CanAllocatePinnedPool(allocation))
	{
		return Unsupported("CUDA allocation/release API unavailable for staging memory");
	}
	switch (allocation)
	{
	case StagingAllocation::CachedPinned:
	case StagingAllocation::WriteCombinedPinned:
	{
		const unsigned flags =
			allocation == StagingAllocation::WriteCombinedPinned ? kMemHostAllocWriteCombined : 0;
		const CUresult status = HostAlloc(g_api, &pool.staging, bytes, flags);
		if (status != kSuccess)
		{
			pool.staging = nullptr;
			return Failure(Error::Kind::Cuda, "Cannot allocate pinned staging pool", status);
		}
		return {};
	}
	case StagingAllocation::Registered4K:
		return MapPinnedPool(pool, bytes);
	}
	return UsageError("invalid staging allocation policy");
}

void ReleasePinnedPool(PinnedPool& pool)
{
	if (pool.mapping != nullptr)
	{
		if ((pool.staging == nullptr ||
			 CheckCleanupResult(g_api.cuMemHostUnregister(pool.staging), "cuMemHostUnregister")) &&
			munmap(pool.mapping, pool.mapping_len) != 0)
		{
			LogLine("error: staging munmap failed errno={}; the mapping is leaked", errno);
		}
	}
	else if (pool.staging != nullptr)
	{
		(void)CheckCleanupResult(g_api.cuMemFreeHost(pool.staging), "cuMemFreeHost");
	}
	pool = PinnedPool{};
}

using namespace cu;

namespace {

struct SlotEvent
{
	CUevent event = nullptr;
	bool event_used = false;
};

struct Lane
{
	std::array<SlotEvent, kMaxStagingBuffers> slots{};
	uint32_t cursor = 0;
	CUstream stream = nullptr;
	CUevent join = nullptr;
	uint32_t ticket = 0;
	ConditionVariable work_ready;
	Thread thread;
};

struct Job
{
	bool to_device = true;
	uint8_t* host = nullptr;
	CUdeviceptr device = 0;
	std::size_t bytes = 0;
	unsigned active_lanes = 0;
	uint32_t slots_per_lane = 0;
	uint64_t chunk_bytes = 0;
	uint64_t chunk_count = 0;
	CopyLoop loop;
	uint8_t* pool = nullptr;
	uint64_t slot_stride_bytes = 0;

	uint8_t* SlotData(unsigned lane_index, uint32_t slot_index) const
	{
		return pool + (static_cast<uint64_t>(lane_index) * slots_per_lane + slot_index) * slot_stride_bytes;
	}
};

struct Layout
{
	unsigned lane_count = 0;
	uint32_t slots_per_lane = 0;
	uint64_t slot_stride_bytes = 0;

	bool operator==(const Layout& other) const
	{
		return lane_count == other.lane_count && slots_per_lane == other.slots_per_lane &&
			   slot_stride_bytes == other.slot_stride_bytes;
	}

	bool operator!=(const Layout& other) const
	{
		return !(*this == other);
	}
};

} // namespace

struct ChunkPipeline::State
{
	CUcontext context = nullptr;
	PipelineCapacity capacity;
	PinnedPool pool;
	Layout previous_layout;
	bool bench_dma = false;
	bool wc_slots = false;
	std::vector<std::unique_ptr<Lane>> lanes;
	CUevent fence = nullptr;

	Mutex mutex;
	ConditionVariable work_done;
	Job pending_job;
	uint32_t helpers_remaining = 0;
	std::atomic<CUresult> first_error{kSuccess};
	bool quit = false;
	std::atomic<bool> unsettled{false};
	unsigned helpers_ready = 0;
	CUresult helper_context_status = kSuccess;
	ConditionVariable helper_ready;

	unsigned LaneCount() const
	{
		return static_cast<unsigned>(std::size(lanes));
	}

	void RunHelper(unsigned lane_index);
	CUresult ExecuteLane(unsigned lane_index, const Job& job);
	template <class Kernel>
	CUresult CopyHtoD(Lane& lane, unsigned lane_index, const Job& job);
	template <class Kernel>
	CUresult CopyDtoH(Lane& lane, unsigned lane_index, const Job& job);
	CUresult Call(Job copy, const CopySettings& settings, CUstream caller_stream,
				  CopyObservation* observation);
	CUresult Quiesce();
	CUresult SettleFailure(CUresult status);
	void RecordFailure(CUresult error);
	CUresult SettleUnrecorded(Lane& lane, CUresult record_status);
};

namespace {
void ReportUnsettledOnce(CUresult status)
{
	static std::atomic<bool> said{false};
	if (!said.exchange(true, std::memory_order_relaxed))
	{
		LogLine("error: a staged copy failed and its DMA completion cannot be established (rc={}, usually a "
				"sticky CUDA error); that context's staging is retired and kept allocated, and its copies use "
				"the original driver. Reported once.",
				status);
	}
}
} // namespace

CUresult ChunkPipeline::State::SettleUnrecorded(Lane& lane, CUresult record_status)
{
	const CUresult settled =
		g_api.cuStreamSynchronize != nullptr ? g_api.cuStreamSynchronize(lane.stream) : kErrorNotSupported;
	if (settled != kSuccess)
	{
		unsettled.store(true);
		ReportUnsettledOnce(settled);
		return settled;
	}
	return record_status;
}

CUresult ChunkPipeline::State::SettleFailure(CUresult error)
{
	for (const auto& lane : lanes)
	{
		const CUresult settled = lane->stream == nullptr			 ? kSuccess
								 : g_api.cuStreamSynchronize == nullptr ? kErrorNotSupported
																		: g_api.cuStreamSynchronize(lane->stream);
		if (settled != kSuccess)
		{
			unsettled.store(true);
			ReportUnsettledOnce(settled);
		}
	}
	if (!unsettled.load())
	{
		for (const auto& lane : lanes)
		{
			for (auto& slot : lane->slots)
			{
				slot.event_used = false;
			}
		}
	}
	return error;
}

void ChunkPipeline::State::RecordFailure(CUresult error)
{
	CUresult expected = kSuccess;
	first_error.compare_exchange_strong(expected, error);
}

void ChunkPipeline::State::RunHelper(unsigned lane_index)
{
	char name[kThreadNameBytes];
	*fmt::format_to_n(name, sizeof(name) - 1, "cmt-pipe-{}", lane_index).out = '\0';
	pthread_setname_np(pthread_self(), name);
	pinning::MarkInternalThread();
	const CUresult context_status = g_api.cuCtxSetCurrent(context);
	if (context_status != kSuccess)
	{
		LogDebug("pipeline lane {}: cuCtxSetCurrent failed rc={}", lane_index, context_status);
	}
	{
		std::lock_guard<Mutex> guard(mutex);
		if (context_status != kSuccess && helper_context_status == kSuccess)
		{
			helper_context_status = context_status;
		}
		++helpers_ready;
		helper_ready.SignalAll();
	}
	if (context_status != kSuccess)
	{
		return;
	}
	RelaxCaptureModeForThisThread();
	Lane& lane = *lanes[lane_index];
	uint32_t seen_ticket = 0;
	for (;;)
	{
		std::unique_lock<Mutex> guard(mutex);
		while (lane.ticket == seen_ticket && !quit)
		{
			lane.work_ready.Wait(guard.mutex());
		}
		if (quit)
		{
			return;
		}
		seen_ticket = lane.ticket;
		const Job job = pending_job;
		guard.unlock();

		const CUresult copy_status = ExecuteLane(lane_index, job);
		if (copy_status != kSuccess)
		{
			RecordFailure(copy_status);
		}
		guard.lock();
		if (--helpers_remaining == 0)
		{
			work_done.Signal();
		}
	}
}

CUresult ChunkPipeline::State::ExecuteLane(unsigned lane_index, const Job& job)
{
	Lane& lane = *lanes[lane_index];
	return hwcopy::WithLoop(job.loop.id,
							[&](auto kernel)
							{
								if (job.to_device)
								{
									return CopyHtoD<decltype(kernel)>(lane, lane_index, job);
								}
								return CopyDtoH<decltype(kernel)>(lane, lane_index, job);
							});
}

template <class Kernel>
CUresult ChunkPipeline::State::CopyHtoD(Lane& lane, unsigned lane_index, const Job& job)
{
	const CopyCtx copy_context = job.loop.ctx;
	const auto fill = [copy_context](void* destination, const void* source, std::size_t bytes)
	{ Kernel::Copy(copy_context, destination, source, bytes); };
	lane.cursor %= job.slots_per_lane;
	for (uint64_t chunk_index = lane_index; chunk_index < job.chunk_count;)
	{
		const uint64_t offset = chunk_index * job.chunk_bytes;
		if (first_error.load(std::memory_order_relaxed) != kSuccess)
		{
			return kSuccess;
		}
		const std::size_t chunk_bytes = static_cast<std::size_t>(
			job.bytes - offset < job.chunk_bytes ? job.bytes - offset : job.chunk_bytes);
		lane.cursor = (lane.cursor + 1) % job.slots_per_lane;
		SlotEvent& slot = lane.slots[lane.cursor];
		uint8_t* const staging = job.SlotData(lane_index, lane.cursor);
		if (slot.event_used)
		{
			const CUresult copy_status = g_api.cuEventSynchronize(slot.event);
			if (copy_status != kSuccess)
			{
				return copy_status;
			}
			slot.event_used = false;
		}
		if (!bench_dma)
		{
			fill(staging, job.host + offset, chunk_bytes);
			StoreFence();
		}
		CUresult copy_status =
			g_api.cuMemcpyHtoDAsync_v2(job.device + offset, staging, chunk_bytes, lane.stream);
		if (copy_status != kSuccess)
		{
			return copy_status;
		}
		copy_status = g_api.cuEventRecord(slot.event, lane.stream);
		if (copy_status != kSuccess)
		{
			return SettleUnrecorded(lane, copy_status);
		}
		slot.event_used = true;
		if (job.chunk_count - chunk_index <= job.active_lanes)
		{
			break;
		}
		chunk_index += job.active_lanes;
	}
	return kSuccess;
}

template <class Kernel>
CUresult ChunkPipeline::State::CopyDtoH(Lane& lane, unsigned lane_index, const Job& job)
{
	const CopyCtx copy_context = job.loop.ctx;
	const auto drain = [copy_context](void* destination, const void* source, std::size_t bytes)
	{ Kernel::Copy(copy_context, destination, source, bytes); };
	uint8_t* pending_destinations[kMaxStagingBuffers] = {};
	std::size_t pending_bytes[kMaxStagingBuffers] = {};
	lane.cursor %= job.slots_per_lane;
	const uint32_t start_cursor = lane.cursor;
	uint64_t chunks_issued = 0;

	auto drain_slot = [&](uint32_t slot_index)
	{
		SlotEvent& slot = lane.slots[slot_index];
		const CUresult copy_status = g_api.cuEventSynchronize(slot.event);
		if (copy_status != kSuccess)
		{
			return copy_status;
		}
		slot.event_used = false;
		if (!bench_dma)
		{
			drain(pending_destinations[slot_index], job.SlotData(lane_index, slot_index),
					   pending_bytes[slot_index]);
			StoreFence();
		}
		pending_destinations[slot_index] = nullptr;
		return kSuccess;
	};

	for (uint64_t chunk_index = lane_index; chunk_index < job.chunk_count;)
	{
		const uint64_t offset = chunk_index * job.chunk_bytes;
		if (first_error.load(std::memory_order_relaxed) != kSuccess)
		{
			break;
		}
		const std::size_t chunk_bytes = static_cast<std::size_t>(
			job.bytes - offset < job.chunk_bytes ? job.bytes - offset : job.chunk_bytes);
		lane.cursor = (lane.cursor + 1) % job.slots_per_lane;
		const uint32_t slot_index = lane.cursor;
		SlotEvent& slot = lane.slots[slot_index];
		CUresult copy_status = kSuccess;
		if (pending_destinations[slot_index] != nullptr)
		{
			copy_status = drain_slot(slot_index);
		}
		else if (slot.event_used)
		{
			copy_status = g_api.cuEventSynchronize(slot.event);
			slot.event_used = copy_status != kSuccess;
		}
		if (copy_status == kSuccess)
		{
			copy_status = g_api.cuMemcpyDtoHAsync_v2(job.SlotData(lane_index, slot_index),
													 job.device + offset, chunk_bytes, lane.stream);
		}
		if (copy_status != kSuccess)
		{
			return copy_status;
		}
		copy_status = g_api.cuEventRecord(slot.event, lane.stream);
		if (copy_status != kSuccess)
		{
			return SettleUnrecorded(lane, copy_status);
		}
		slot.event_used = true;
		pending_destinations[slot_index] = job.host + offset;
		pending_bytes[slot_index] = chunk_bytes;
		chunks_issued++;
		if (job.chunk_count - chunk_index <= job.active_lanes)
		{
			break;
		}
		chunk_index += job.active_lanes;
	}
	for (uint64_t pending_index = chunks_issued > job.slots_per_lane ? chunks_issued - job.slots_per_lane : 0;
		 pending_index < chunks_issued; pending_index++)
	{
		const uint32_t slot_index = static_cast<uint32_t>(
			(start_cursor + 1 + pending_index % job.slots_per_lane) % job.slots_per_lane);
		if (pending_destinations[slot_index] != nullptr)
		{
			const CUresult copy_status = drain_slot(slot_index);
			if (copy_status != kSuccess)
			{
				return copy_status;
			}
		}
	}
	return kSuccess;
}

CUresult ChunkPipeline::State::Quiesce()
{
	for (auto& lane : lanes)
	{
		for (uint32_t slot_index = 0; slot_index < capacity.slots; slot_index++)
		{
			SlotEvent& slot = lane->slots[slot_index];
			if (!slot.event_used)
			{
				continue;
			}
			const CUresult copy_status = g_api.cuEventSynchronize(slot.event);
			if (copy_status != kSuccess)
			{
				return copy_status;
			}
			slot.event_used = false;
		}
	}
	return kSuccess;
}

CUresult ChunkPipeline::State::Call(Job copy, const CopySettings& settings, CUstream caller_stream,
									CopyObservation* observation)
{
	if (!SettingsFit(capacity, settings))
	{
		return kErrorInvalidValue;
	}
	if (copy.bytes == 0)
	{
		return kSuccess;
	}
	if (unsettled.load())
	{
		return kNotStarted;
	}
	const auto host_address = reinterpret_cast<uintptr_t>(copy.host);
	if (!copy.host || copy.device == 0 || copy.bytes - 1 > UINTPTR_MAX - host_address ||
		copy.bytes - 1 > UINT64_MAX - copy.device)
	{
		return kErrorInvalidValue;
	}
	const uint64_t chunk_count = 1 + (copy.bytes - 1) / settings.chunk_bytes;
	copy.active_lanes =
		chunk_count < settings.workers ? static_cast<unsigned>(chunk_count) : settings.workers;
	copy.slots_per_lane = settings.slots;
	copy.chunk_bytes = settings.chunk_bytes;
	copy.chunk_count = chunk_count;
	copy.loop = settings.loop;
	copy.pool = static_cast<uint8_t*>(pool.staging);
	copy.slot_stride_bytes = SlotStride(settings.chunk_bytes);
	const unsigned active_lanes = copy.active_lanes;
	pinning::NoDriverScope driver_scope;

	const Layout layout{settings.workers, settings.slots, copy.slot_stride_bytes};
	CUresult copy_status = kSuccess;
	if (layout != previous_layout)
	{
		copy_status = Quiesce();
		if (copy_status != kSuccess)
		{
			return SettleFailure(copy_status);
		}
		previous_layout = layout;
	}

	copy_status = g_api.cuEventRecord(fence, caller_stream);
	for (unsigned lane_index = 0; copy_status == kSuccess && lane_index < active_lanes; lane_index++)
	{
		copy_status = g_api.cuStreamWaitEvent(lanes[lane_index]->stream, fence, 0);
	}
	if (copy_status != kSuccess)
	{
		LogDebug("pipeline: stream fence failed rc={}; the copy uses the original driver", copy_status);
		return kNotStarted;
	}

	first_error.store(kSuccess, std::memory_order_relaxed);
	if (active_lanes > 1)
	{
		std::unique_lock<Mutex> guard(mutex);
		pending_job = copy;
		helpers_remaining = active_lanes - 1;
		for (unsigned lane_index = 1; lane_index < active_lanes; lane_index++)
		{
			Lane& lane = *lanes[lane_index];
			++lane.ticket;
			lane.work_ready.Signal();
		}
	}
	copy_status = ExecuteLane(0, copy);
	if (copy_status != kSuccess)
	{
		RecordFailure(copy_status);
	}
	if (active_lanes > 1)
	{
		std::unique_lock<Mutex> guard(mutex);
		while (helpers_remaining != 0)
		{
			work_done.Wait(guard.mutex());
		}
	}
	copy_status = first_error.load(std::memory_order_acquire);
	if (copy_status != kSuccess)
	{
		return SettleFailure(copy_status);
	}

	for (unsigned lane_index = 0; copy_status == kSuccess && lane_index < active_lanes; lane_index++)
	{
		copy_status = g_api.cuEventRecord(lanes[lane_index]->join, lanes[lane_index]->stream);
		if (copy_status == kSuccess)
		{
			copy_status = g_api.cuStreamWaitEvent(caller_stream, lanes[lane_index]->join, 0);
		}
	}
	if (copy_status == kSuccess && observation != nullptr)
	{
		observation->completed = 1;
		observation->requested_workers = settings.workers;
		observation->actual_workers = active_lanes;
		observation->slots = copy.slots_per_lane;
		observation->max_slots_used =
			std::min<uint64_t>(copy.slots_per_lane, 1 + (chunk_count - 1) / active_lanes);
		observation->slot_bytes = copy.chunk_bytes;
		observation->variant = static_cast<uint64_t>(copy.loop.id);
		observation->write_back = static_cast<uint64_t>(copy.loop.ctx.write_back);
		observation->flush_bytes = copy.loop.ctx.flush_batch;
		observation->bench_dma = bench_dma;
		observation->wc_slots = wc_slots;
	}
	return copy_status == kSuccess ? copy_status : SettleFailure(copy_status);
}

ChunkPipeline::ChunkPipeline(std::unique_ptr<State> owned_state) : state(std::move(owned_state))
{
}

namespace {

Result<void> AllocatePlacedPool(CUcontext context, PinnedPool& pool, std::size_t bytes,
								StagingAllocation allocation, const ThreadPlacement& placement)
{
	Result<void> allocated;
	Thread thread;
	CMT_RETURN_IF_ERROR(thread.Start(
		[&]
		{
			pthread_setname_np(pthread_self(), "cmt-pool");
			pinning::MarkInternalThread();
			RelaxCaptureModeForThisThread();
			const CUresult status = g_api.cuCtxSetCurrent(context);
			allocated = status == kSuccess ? AllocatePinnedPool(pool, bytes, allocation)
										   : Failure(Error::Kind::Cuda,
													 "Cannot make the context current for the pool", status);
		},
		placement));
	CMT_RETURN_IF_ERROR(thread.Join());
	return allocated;
}

} // namespace

Result<std::unique_ptr<ChunkPipeline>> ChunkPipeline::Create(CUcontext context, const HookConfig& config,
															 const PipelineCapacity& capacity,
															 const ThreadPlacement& workers)
{
	if (!g_api.cuStreamCreate || !g_api.cuStreamWaitEvent || !g_api.cuStreamDestroy ||
		!g_api.cuStreamSynchronize || !g_api.cuCtxGetCurrent || !g_api.cuCtxSetCurrent ||
		!g_api.cuEventCreate || !g_api.cuEventRecord || !g_api.cuEventSynchronize || !g_api.cuEventDestroy ||
		!g_api.cuMemcpyHtoDAsync_v2 || !g_api.cuMemcpyDtoHAsync_v2 ||
		!CanAllocatePinnedPool(config.slot_alloc))
	{
		return Unsupported("CUDA copy, allocation, or completion API unavailable");
	}
	if (capacity.workers == 0 || capacity.workers > kMaxPipelineWorkers || capacity.slots == 0 ||
		capacity.slots > kMaxStagingBuffers || capacity.pool_bytes == 0 || capacity.pool_bytes > SIZE_MAX)
	{
		return Failure(Error::Kind::Usage, "Invalid chunk pipeline capacity");
	}
	CUcontext current = nullptr;
	const CUresult context_status = g_api.cuCtxGetCurrent(&current);
	if (context_status != kSuccess || context == nullptr || current != context)
	{
		return Failure(Error::Kind::Cuda, "Chunk pipeline requires its CUDA context to be current",
					   context_status != kSuccess ? context_status : kErrorInvalidContext);
	}

	auto created_state = std::make_unique<State>();
	State& resources = *created_state;
	resources.context = context;
	resources.capacity = capacity;
	resources.bench_dma = config.bench_dma;
	resources.wc_slots = config.slot_alloc == StagingAllocation::WriteCombinedPinned;
	std::unique_ptr<ChunkPipeline> pipeline(new ChunkPipeline(std::move(created_state)));
	auto status = g_api.cuEventCreate(&resources.fence, kEventDisableTiming);
	if (status != kSuccess)
	{
		resources.fence = nullptr;
		return Failure(Error::Kind::Cuda, "Cannot create pipeline fence", status);
	}
	const auto pool_bytes = static_cast<std::size_t>(capacity.pool_bytes);
	CMT_RETURN_IF_ERROR(
		std::empty(workers.cpus) && !workers.numa_node
			? AllocatePinnedPool(resources.pool, pool_bytes, config.slot_alloc)
			: AllocatePlacedPool(context, resources.pool, pool_bytes, config.slot_alloc, workers));
	for (unsigned index = 0; index < capacity.workers; ++index)
	{
		resources.lanes.push_back(std::make_unique<Lane>());
		Lane& lane = *resources.lanes.back();
		for (uint32_t slot = 0; slot < capacity.slots; ++slot)
		{
			status = g_api.cuEventCreate(&lane.slots[slot].event, kEventDisableTiming);
			if (status != kSuccess)
			{
				lane.slots[slot].event = nullptr;
				return Failure(Error::Kind::Cuda, "Cannot create slot event", status);
			}
		}
		status = g_api.cuStreamCreate(&lane.stream, kStreamNonBlocking);
		if (status != kSuccess)
		{
			lane.stream = nullptr;
			return Failure(Error::Kind::Cuda, "Cannot create lane stream", status);
		}
		status = g_api.cuEventCreate(&lane.join, kEventDisableTiming);
		if (status != kSuccess)
		{
			lane.join = nullptr;
			return Failure(Error::Kind::Cuda, "Cannot create lane completion event", status);
		}
	}
	return pipeline;
}

Result<void> ChunkPipeline::Start(const ThreadPlacement& workers, bool one_cpu_each)
{
	State& resources = *state;
	for (unsigned index = 1; index < resources.LaneCount(); ++index)
	{
		auto worker_placement = workers;
		if (one_cpu_each && !std::empty(workers.cpus))
		{
			worker_placement.cpus = {workers.cpus[(index - 1) % std::size(workers.cpus)]};
		}
		auto& thread = resources.lanes[index]->thread;
		CMT_RETURN_IF_ERROR(
			thread.Start([&resources, index] { resources.RunHelper(index); }, worker_placement));
		std::unique_lock<Mutex> guard(resources.mutex);
		while (resources.helpers_ready < index)
		{
			resources.helper_ready.Wait(guard.mutex());
		}
		if (resources.helper_context_status != kSuccess)
		{
			return Failure(Error::Kind::Cuda, "A pipeline helper cannot make the CUDA context current",
						   resources.helper_context_status);
		}
	}
	return {};
}

ChunkPipeline::~ChunkPipeline()
{
	State& resources = *state;
	{
		std::lock_guard<Mutex> guard(resources.mutex);
		resources.quit = true;
		for (auto& lane : resources.lanes)
		{
			lane->work_ready.Signal();
		}
	}
	for (auto& lane : resources.lanes)
	{
		if (lane->thread.Joinable())
		{
			const auto joined = lane->thread.Join();
			if (!joined)
			{
				Panic(joined.error().Message());
			}
		}
	}
	bool drained = !resources.unsettled.load();
	for (const auto& lane : resources.lanes)
	{
		if (!drained || lane->stream == nullptr)
		{
			continue;
		}
		const auto status =
			g_api.cuStreamSynchronize ? g_api.cuStreamSynchronize(lane->stream) : kErrorNotSupported;
		if (status != kSuccess)
		{
			if (status != kErrorDeinitialized)
			{
				LogLine("error: staging teardown cannot drain a lane stream (rc={}); its pinned pool and "
						"handles are retained (leaked)",
						status);
			}
			drained = false;
		}
	}
	if (!drained)
	{
		(void)state.release();
		return;
	}
	for (auto& lane : resources.lanes)
	{
		for (SlotEvent& slot : lane->slots)
		{
			if (slot.event != nullptr)
			{
				(void)CheckCleanupResult(g_api.cuEventDestroy(slot.event), "cuEventDestroy");
			}
		}
		if (lane->stream != nullptr)
		{
			(void)CheckCleanupResult(g_api.cuStreamDestroy(lane->stream), "cuStreamDestroy");
		}
		if (lane->join != nullptr)
		{
			(void)CheckCleanupResult(g_api.cuEventDestroy(lane->join), "cuEventDestroy");
		}
	}
	if (resources.fence != nullptr)
	{
		(void)CheckCleanupResult(g_api.cuEventDestroy(resources.fence), "cuEventDestroy");
	}
	ReleasePinnedPool(resources.pool);
}

Result<void> ChunkPipeline::Finish()
{
	for (const auto& lane : state->lanes)
	{
		if (lane->stream == nullptr)
		{
			continue;
		}
		const CUresult status = g_api.cuStreamSynchronize(lane->stream);
		if (status != kSuccess)
		{
			state->SettleFailure(status);
			return Failure(Error::Kind::Cuda, "pipeline completion failed", status);
		}
	}
	return {};
}

bool ChunkPipeline::Fits(const CopySettings& settings) const
{
	return SettingsFit(state->capacity, settings);
}

CUresult ChunkPipeline::HtoD(const CopySettings& settings, CUdeviceptr dst, const uint8_t* src,
							 std::size_t bytes, CUstream stream, CopyObservation* observation)
{
	Job job;
	job.to_device = true;
	job.host = const_cast<uint8_t*>(src);
	job.device = dst;
	job.bytes = bytes;
	return state->Call(job, settings, stream, observation);
}

CUresult ChunkPipeline::DtoH(const CopySettings& settings, uint8_t* dst, CUdeviceptr src, std::size_t bytes,
							 CUstream stream, CopyObservation* observation)
{
	Job job;
	job.to_device = false;
	job.host = dst;
	job.device = src;
	job.bytes = bytes;
	return state->Call(job, settings, stream, observation);
}

} // namespace ring
} // namespace cmt
