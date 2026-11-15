#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <new>
#include <system_error>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "arch/copy.h"
#include "common/aliases.h"
#include "common/defines.h"
#include "common/error.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "config.h"
#include "copying/cpu_copy.h"
#include "cuda/driver.h"

namespace cmt {

// Diagnostic CPU-copy experiment only: this pool divides one buffer among
// helper threads and the caller. The hook never uses it; ChunkPipeline workers
// each own whole staging slots instead. These limits are not tuning defaults.
constexpr unsigned kMaxCopyParticipants = 32;		   // caller plus at most 31 helpers
constexpr uint64_t kMinParallelCopyBytes = 1ull << 20; // diagnostic split threshold: 1 MiB
constexpr uint64_t kSplitAlignmentBytes = 4096;		   // alignment of CPU copy partitions

// One thread's part of a copy. row_count == 0 selects byte-split mode. Trivially copyable (the mpmc protocol
// publishes shares with memcpy); the loop is chosen once per share (RunShare), not per row.
struct alignas(64) CopyShare
{
	CopyLoop loop;
	uint8_t* dst;
	const uint8_t* src;
	uint64_t dst_pitch;
	uint64_t src_pitch;
	uint64_t dst_height;
	uint64_t src_height;
	uint64_t width;		  // row width in bytes (row mode)
	uint64_t height;	  // rows per slice (row mode)
	uint64_t first_row;	  // first global row (row mode)
	uint64_t row_count;	  // rows to copy; 0 = byte mode
	uint64_t byte_offset; // byte mode: range start
	uint64_t byte_count;  // byte mode: range length
};

// Fill shares[0..num_shares) for a flat copy. Returns false when the copy
// is too small to split.
bool Plan1D(CopyShare* shares, unsigned num_shares, CopyLoop loop, void* dst, const void* src,
			uint64_t bytes);
// Same for a pitched copy.
bool Plan2D(CopyShare* shares, unsigned num_shares, CopyLoop loop, const Copy2DView& copy);

void RunShare(const CopyShare& share);

enum class PoolProtocol : std::uint8_t
{
	Condvar, // Mutex reservation and condition-variable handoff.
	Mpmc	 // Atomic caller reservation; the same condition-variable worker handoff.
};

struct PoolOptions
{
	unsigned workers = 0; // 1..31; 0 = no pool
	PoolProtocol protocol = PoolProtocol::Condvar;
	// Worker i uses cpus[i % size]; empty selects CMT_WORKER_* or inherited affinity.
	std::vector<unsigned> cpus;
};

// Persistent workers plus the calling thread execute one copy at a time.
// Each call waits for every share and fences stores before returning. Callers
// may submit copies concurrently. Serialize CPU changes and destruction with all callers.
// Both modes use library condition variables for worker handoff. Condvar reserves
// each caller's turn under the mutex; Mpmc uses an atomic reservation first, then
// parks on contention. Neither mode spins during handoff.
// This diagnostic-only CPU split pool is not used by the hook. ChunkPipeline has persistent
// workers, each of which copies whole chunks without dispatching into this pool.
class ThreadPool
{
public:
	// Zero workers succeeds with an empty pool. Allocation/thread failures are
	// typed errors; a partially created pool joins its workers before returning.
	// Callers may fall back to serial copies without claiming workers ran.
	[[nodiscard]] static cmt::Result<std::unique_ptr<ThreadPool>> Create(const PoolOptions& options);
	~ThreadPool();
	ThreadPool(const ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;

	unsigned Workers() const;

	// Re-pins running worker i to cpus[i % n] (empty = leave as is).
	[[nodiscard]] cmt::Result<void> PinWorkers(const std::vector<unsigned>& cpus);

	// Split and run one copy. All shares use this job's loop; a later job may
	// select a different loop. Returns false (nothing copied) below the split threshold.
	bool Copy1D(CopyLoop loop, void* dst, const void* src, uint64_t bytes);
	bool Copy2D(CopyLoop loop, const Copy2DView& copy);

	struct State;

private:
	explicit ThreadPool(std::unique_ptr<State> owned_state);
	bool Dispatch(const CopyShare* shares);
	std::unique_ptr<State> state;
};

} // namespace cmt
