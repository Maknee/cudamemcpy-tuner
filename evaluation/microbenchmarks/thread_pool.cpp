#include "evaluation/microbenchmarks/thread_pool.h"

// Publish preallocated shares, then park through a library condition variable.
// Predicate checks and updates use one mutex, including shutdown and completion.
// Workers fence stores before reporting completion; the caller also fences
// before returning, so a subsequent DMA sees every share's writes.

namespace cmt {

// ------------------------------------------------------------------
// Copy share planning
// ------------------------------------------------------------------

bool Plan1D(CopyShare* shares, unsigned num_shares, CopyLoop loop, void* dst, const void* src, uint64_t bytes)
{
	if (bytes < kMinParallelCopyBytes)
	{
		return false;
	}
	const uint64_t per_share = (bytes / num_shares) & ~(kSplitAlignmentBytes - 1);
	if (per_share == 0)
	{
		return false;
	}
	for (unsigned i = 0; i < num_shares; i++)
	{
		CopyShare& s = shares[i];
		s = CopyShare{};
		s.loop = loop;
		s.dst = static_cast<uint8_t*>(dst);
		s.src = static_cast<const uint8_t*>(src);
		s.byte_offset = i * per_share;
		s.byte_count = (i + 1 == num_shares) ? bytes - s.byte_offset : per_share;
	}
	return true;
}

bool Plan2D(CopyShare* shares, unsigned num_shares, CopyLoop loop, const Copy2DView& c)
{
	const uint64_t total_rows = c.height * c.depth;
	if (c.width * total_rows < kMinParallelCopyBytes)
	{
		return false;
	}
	const bool byte_split = (total_rows == 1);
	uint64_t rows_per_share = 0;
	uint64_t bytes_per_share = 0;
	if (byte_split)
	{
		bytes_per_share = (c.width / num_shares) & ~(kSplitAlignmentBytes - 1);
		if (bytes_per_share == 0)
		{
			return false;
		}
	}
	else
	{
		rows_per_share = (total_rows + num_shares - 1) / num_shares;
	}
	for (unsigned i = 0; i < num_shares; i++)
	{
		CopyShare& s = shares[i];
		s = CopyShare{};
		s.loop = loop;
		s.dst = c.dst;
		s.src = c.src;
		s.dst_pitch = c.dst_pitch;
		s.src_pitch = c.src_pitch;
		s.dst_height = c.dst_height;
		s.src_height = c.src_height;
		s.width = c.width;
		s.height = c.height;
		if (byte_split)
		{
			s.byte_offset = i * bytes_per_share;
			s.byte_count = (i + 1 == num_shares) ? c.width - s.byte_offset : bytes_per_share;
		}
		else
		{
			s.first_row = i * rows_per_share;
			if (s.first_row < total_rows)
			{
				const uint64_t left = total_rows - s.first_row;
				s.row_count = rows_per_share < left ? rows_per_share : left;
			} // else: empty share (row_count == byte_count == 0)
		}
	}
	return true;
}

namespace {

// Choose a direct kernel call once per share, then reuse it for every row.
template <class L>
void RunShareWith(const CopyShare& s)
{
	if (s.row_count == 0)
	{
		if (s.byte_count != 0)
		{
			L::Copy(s.loop.ctx, s.dst + s.byte_offset, s.src + s.byte_offset,
					static_cast<size_t>(s.byte_count));
		}
		return;
	}
	const uint64_t row_end = s.first_row + s.row_count;
	for (uint64_t global_row = s.first_row; global_row < row_end; global_row++)
	{
		const uint64_t slice = global_row / s.height;
		const uint64_t row = global_row % s.height;
		uint8_t* dst_row = s.dst + slice * s.dst_pitch * s.dst_height + row * s.dst_pitch;
		const uint8_t* src_row = s.src + slice * s.src_pitch * s.src_height + row * s.src_pitch;
		L::Copy(s.loop.ctx, dst_row, src_row, static_cast<size_t>(s.width));
	}
}

} // namespace

// The loop is chosen here, once per share, not per row.
void RunShare(const CopyShare& s)
{
	hwcopy::WithLoop(s.loop.id, [&](auto l) { RunShareWith<decltype(l)>(s); });
}

// ------------------------------------------------------------------
// Pool state
// ------------------------------------------------------------------

struct ThreadPool::State
{
	PoolOptions options;
	std::array<CopyShare, kMaxCopyParticipants> shares{};
	std::vector<cmt::Thread> threads;
	Mutex mutex;
	ConditionVariable work_ready;
	ConditionVariable work_done;
	ConditionVariable caller_ready;
	bool caller_active = false;

	// All worker handoff predicates are protected by mutex.
	uint32_t generation = 0;
	uint32_t pending = 0;
	bool quit = false;
	alignas(64) std::atomic<bool> caller_reserved{false};

	void WorkerLoop(unsigned index);
	void Dispatch(const CopyShare* planned);
};

void ThreadPool::State::WorkerLoop(unsigned index)
{
	char name[16];
	*fmt::format_to_n(name, sizeof(name) - 1, "mh-copy-{}", index).out = '\0';
	pthread_setname_np(pthread_self(), name);
	uint32_t seen = 0;
	for (;;)
	{
		std::unique_lock<Mutex> guard(mutex);
		while (generation == seen && !quit)
		{
			work_ready.Wait(guard.mutex());
		}
		if (quit)
		{
			return;
		}
		seen = generation;
		const auto share = shares[index];
		guard.unlock();

		RunShare(share);
		StoreFence();

		guard.lock();
		if (--pending == 0)
		{
			work_done.Signal();
		}
	}
}

void ThreadPool::State::Dispatch(const CopyShare* planned)
{
	const bool atomic_reservation = options.protocol == PoolProtocol::Mpmc;
	if (atomic_reservation)
	{
		// Compatibility mode keeps its atomic caller reservation. Only reservation
		// acquisition differs; it does not claim a separate worker-wait algorithm.
		auto reserve = [&] { return !caller_reserved.exchange(true, std::memory_order_acquire); };
		if (!reserve())
		{
			std::unique_lock<Mutex> guard(mutex);
			while (!reserve())
			{
				caller_ready.Wait(guard.mutex());
			}
		}
	}

	std::unique_lock<Mutex> guard(mutex);
	if (!atomic_reservation)
	{
		while (caller_active)
		{
			caller_ready.Wait(guard.mutex());
		}
		caller_active = true;
	}
	std::memcpy(std::data(shares), planned, (options.workers + 1) * sizeof(CopyShare));
	pending = options.workers;
	++generation;
	work_ready.SignalAll();
	guard.unlock();

	RunShare(shares[options.workers]);
	StoreFence();

	guard.lock();
	while (pending != 0)
	{
		work_done.Wait(guard.mutex());
	}
	if (atomic_reservation)
	{
		caller_reserved.store(false, std::memory_order_release);
	}
	else
	{
		caller_active = false;
	}
	caller_ready.Signal();
}

// ------------------------------------------------------------------
// ThreadPool
// ------------------------------------------------------------------

ThreadPool::ThreadPool(std::unique_ptr<State> owned_state) : state(std::move(owned_state))
{
}

cmt::Result<std::unique_ptr<ThreadPool>> ThreadPool::Create(const PoolOptions& options)
{

	if (options.workers == 0)
	{
		return std::unique_ptr<ThreadPool>{};
	}
	if (options.workers >= kMaxCopyParticipants)
	{
		return UsageError("diagnostic CPU pool supports at most 31 helpers plus the caller");
	}
	auto placement = cu::CurrentWorkerPlacement(cu::g_api, options.cpus);
	if (!placement)
	{
		return cmt::Failure(placement.error());
	}
	auto created_state = std::make_unique<State>();
	created_state->options = options;
	State* worker_state = created_state.get();
	std::unique_ptr<ThreadPool> pool(new ThreadPool(std::move(created_state)));
	worker_state->threads.resize(worker_state->options.workers);
	for (unsigned index = 0; index < worker_state->options.workers; ++index)
	{
		auto worker_placement = *placement;
		if (!options.cpus.empty())
		{
			worker_placement.cpus = {options.cpus[index % options.cpus.size()]};
		}
		auto started = worker_state->threads[index].Start(
			[worker_state, index] { worker_state->WorkerLoop(index); }, worker_placement);
		if (!started)
		{
			return cmt::Failure(std::move(started.error()));
		}
	}
	return pool;
}

ThreadPool::~ThreadPool()
{
	State& s = *state;
	{
		std::lock_guard<Mutex> lk(s.mutex);
		s.quit = true;
		++s.generation;
		s.work_ready.SignalAll();
	}
	for (auto& thread : s.threads)
	{
		const auto joined = thread.Join();
		if (!joined)
		{
			cmt::Panic(joined.error().Message());
		}
	}
}

unsigned ThreadPool::Workers() const
{
	return state->options.workers;
}

cmt::Result<void> ThreadPool::PinWorkers(const std::vector<unsigned>& cpus)
{
	if (cpus.empty())
	{
		return {};
	}
	for (std::size_t index = 0; index < state->threads.size(); ++index)
	{
		CMT_RETURN_IF_ERROR(
			cmt::SetThreadAffinity(state->threads[index].NativeHandle(), {cpus[index % cpus.size()]}));
	}
	return {};
}

bool ThreadPool::Dispatch(const CopyShare* planned)
{
	state->Dispatch(planned);
	return true;
}

bool ThreadPool::Copy1D(CopyLoop loop, void* dst, const void* src, uint64_t bytes)
{
	CopyShare planned[kMaxCopyParticipants];
	if (!Plan1D(planned, state->options.workers + 1, loop, dst, src, bytes))
	{
		return false;
	}
	return Dispatch(planned);
}

bool ThreadPool::Copy2D(CopyLoop loop, const Copy2DView& copy)
{
	CopyShare planned[kMaxCopyParticipants];
	if (!Plan2D(planned, state->options.workers + 1, loop, copy))
	{
		return false;
	}
	return Dispatch(planned);
}

} // namespace cmt
