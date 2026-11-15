#if defined(CMT_FAKE_RUNTIME_ALT)
#include <cstddef>
#include <cstring>

extern "C" __attribute__((visibility("default"))) int cudaMemcpy(void* dst, const void* src, std::size_t n,
																 int)
{
	std::memcpy(dst, src, n);
	return 41; // distinguish this original from the independent fake driver
}

extern "C" __attribute__((visibility("default"))) int cuMemcpyHtoD_v2(unsigned long long dst, const void* src,
																	  std::size_t n)
{
	std::memcpy(reinterpret_cast<void*>(dst), src, n);
	return 55;
}
#elif defined(CMT_FAKE_IMPORT_CALLER)
#include <cstddef>
#include <cstdlib>
#include <cstring>

extern "C"
{
	int cudaMemcpy(void*, const void*, std::size_t, int);
	int cudaMemcpyAsync(void*, const void*, std::size_t, int, void*);
	int cudaHostAlloc(void**, std::size_t, unsigned);
	int cudaFreeHost(void*);
	int cudaHostRegister(void*, std::size_t, unsigned);
	int cudaHostUnregister(void*);
	int cuMemcpyHtoD_v2(unsigned long long, const void*, std::size_t);
	int cuMemcpyDtoH_v2(void*, unsigned long long, std::size_t);
	int cuCtxSynchronize();
	unsigned fake_trace_queries();
}

// This separately built DSO IMPORTS CUDA symbols from its RTLD_LOCAL dependency.
// It never looks those imports up with dlsym. It catches a preloaded wrapper
// incorrectly assuming that RTLD_NEXT can see every runtime dependency.
extern "C" __attribute__((visibility("default"))) int fake_import_calls()
{
	unsigned char device[64]{}, host[64]{}, result[64]{};
	void* pinned = nullptr;
	int rc = cudaHostAlloc(&pinned, sizeof(host), 0);
	if (rc != 0)
	{
		return rc;
	}
	std::memset(pinned, 42, sizeof(host));
	if ((rc = cudaMemcpy(device, pinned, sizeof(host), 1)) != 0)
	{
		return rc;
	}
	if ((rc = cudaMemcpyAsync(result, device, sizeof(host), 2, nullptr)) != 0)
	{
		return rc;
	}
	if ((rc = cuCtxSynchronize()) != 0)
	{
		return rc;
	}
	if (std::memcmp(pinned, result, sizeof(host)) != 0)
	{
		return 31;
	}
	if ((rc = cudaFreeHost(pinned)) != 0)
	{
		return rc;
	}
	if ((rc = cudaHostRegister(host, sizeof(host), 0)) != 0)
	{
		return rc;
	}
	if ((rc = cudaHostUnregister(host)) != 0)
	{
		return rc;
	}
	if ((rc = cuMemcpyHtoD_v2(reinterpret_cast<unsigned long long>(device), host, sizeof(host))) != 0)
	{
		return rc;
	}
	if ((rc = cuMemcpyDtoH_v2(result, reinterpret_cast<unsigned long long>(device), sizeof(host))) != 0)
	{
		return rc;
	}
	if (std::memcmp(host, result, sizeof(host)) != 0)
	{
		return 32;
	}
	const char* gate = std::getenv("CUDAMEMCPY_TUNER_TRACE_ENABLE");
	const bool traced = gate != nullptr && std::strcmp(gate, "1") == 0;
	return (fake_trace_queries() != 0) == traced ? 0 : 33;
}
#else
// fake_libcuda.cpp - a host-only stand-in for libcuda.so.1 that lets the
// pipeline run end to end without a GPU.
//
// "Device" memory is plain host memory (CUdeviceptr = address). Async
// copies are NOT executed when issued: each stream keeps a FIFO of ops
// (copy, event record, wait-event) that only run when something waits
// (cuEventSynchronize, cuStreamSynchronize, cuCtxSynchronize, a sync
// memcpy). So a pipeline that reuses a staging slot before its event, or
// forgets a cross-stream fence/join, reads or writes the wrong bytes and
// the test sees it. Pageable async copies follow CUDA's documented
// behaviour instead: an H2D source is captured at call time, a D2H into
// pageable memory has completed when the call returns.
//
// Primary-context teardown (cuDevicePrimaryCtxReset{,_v2}, and
// cuDevicePrimaryCtxRelease{,_v2} when the retain count reaches 0) really
// invalidates the context, as the driver does: every stream, event and
// cuMemHostAlloc allocation made before it becomes stale, and any later use
// of one fails with CUDA_ERROR_INVALID_HANDLE (400) and is counted
// (fake_stale_uses), so a hook that keeps a ring across a reset fails the test.
// The context handle stays 0x1000, which is what makes a stale ring findable:
// the driver, too, usually hands the same handle back. cuCtxDestroy does NOT
// invalidate (the fake has one context and the existing tests keep using
// streams they made before a destroy).
//
// Further behaviours, each off until a test turns it on:
//   - stream capture (fake_set_capture): cuStreamIsCapturing reports a stream as capturing, or the legacy
//     stream as CUDA_ERROR_STREAM_CAPTURE_IMPLICIT; a copy on the capturing stream is "captured" (counted,
//     not run), pageable or pinned, as on driver 615.71.09 (a captured pageable copy returns 0); a
//     pageable copy on the legacy stream during an implicit capture fails with 900 (not measured);
//   - per-thread default stream entry points (*_ptds / *_ptsz), returned by cuGetProcAddress for flags & 2;
//   - an unexported primary-context reset, returned by cuGetProcAddress for cudaVersion 1 (address-identity
//     check);
//   - fake_fail_record_after_dma: the next cuEventRecord after a pinned DMA fails (999) once;
//   - FAKE_STRICT_LEGACY=1: the synchronous copies run the legacy stream and the BLOCKING streams only, as
//   CUDA's
//     legacy stream does; streams made with CU_STREAM_NON_BLOCKING are not waited for (by default the fake
//     runs every stream, which hides a fallback that leaves the caller's non-blocking stream).
//
// Registration-cache redesign additions:
//   - host callbacks (cuLaunchHostFunc, cuStreamAddCallback): queued like any op; when a wait reaches one, it
//   runs
//     on the fake's callback thread (made with pthread_create from this library, as libcuda's are) while the
//     waiter holds the fake's lock, i.e. the stream does not progress until the callback returns;
//   - a driver thread (fake_driver_thread_post): runs a test function while holding the fake's lock, the
//   stand-in
//     for a driver-internal lock that cuMemHostRegister and every other call also need;
//   - API calls made on either of those threads fail with CUDA_ERROR_NOT_PERMITTED (800) and are counted
//     (fake_internal_api_calls): CUDA forbids them in callbacks, and on a driver thread they would
//     self-deadlock;
//   - stale registrations: fake_pages_changed(p, n) says that the physical pages behind [p, p + n) were
//   replaced
//     (munmap, mremap, madvise, an allocator returning a block; fake_alloc calls it for blocks of 64 KiB or
//     more). A DMA through a registration made BEFORE such a change reads or writes the old pages, as on a
//     real driver: the fake counts it (fake_stale_dmas), an HtoD delivers 0xDB bytes, a DtoH leaves the
//     destination unwritten.
//   - pageable HtoD snapshots are freed once the copy ran (a multi-second stress test copies gigabytes).
//
// Registration-limit additions (progressive pinning without a default limit, test_pinning_limits):
//   - cuDeviceTotalMem_v2 reports 24 GiB per device, or what fake_device_memory(device, bytes, error) sets, or
//     that error; fake_device_memory_queries counts the calls;
//   - fake_rw_register_error_at(rc, n): the n-th writable registration attempt from now and every later one fail
//     with rc (fake_rw_register_error(rc) is n = 1), a refusal mid-copy as from ENOMEM or a locked-memory limit.
//
// Generic-copy additions (cuMemcpy / cuMemcpyAsync, test_tune_table_e2e "generic"):
//   - device memory has a kind (fake_device_alloc; cuMemAlloc_v2 makes the plain kind), and
//     cuPointerGetAttributes answers MEMORY_TYPE, IS_MANAGED, DEVICE_ORDINAL, CONTEXT, RANGE_START_ADDR,
//     RANGE_SIZE, MAPPING_BASE_ADDR and MAPPING_SIZE per kind the way the RTX 3090 driver 615.71.09 does
//     (measured): VMM memory has CONTEXT NULL and
//     one mapping per cuMemMap, a hole in a reservation is type 0 with the reservation's range and no
//     mapping, a pool allocation's mapping is the larger pool chunk, managed memory is type 2 with IS_MANAGED
//     1, pinned and registered memory is type 1, everything else reads as pageable (type 0, ordinal -1, range
//     [0, +0));
//   - a copy that touches a hole fails with 1 (INVALID_VALUE) before writing, and is counted
//     (fake_hole_copies): the real driver refuses such a synchronous or pageable copy that way, and dies
//     (SIGSEGV) on a pinned asynchronous one, which is what a staged chunk is;
//   - fake_fail_pointer_queries(address): cuPointerGetAttributes for that address fails (201), for every
//     address with ~0, until cleared with 0;
//   - cuMemcpy / cuMemcpyAsync (and their _ptds / _ptsz) copy with the driver's semantics for the memory they
//     see and are counted (fake_generic_calls): a call that reaches the fake was passed through by the hook.
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <thread>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

#define FAKE_API extern "C" __attribute__((visibility("default")))

namespace {

struct Stream;

// Context generation: bumped by a primary-context teardown. A handle made in an older generation is stale.
std::atomic<uint64_t> g_gen{0};
std::atomic<long> g_stale_uses{0};

struct Event
{
	Stream* stream = nullptr; // stream of the latest record
	uint64_t index = 0;		  // op index of that record
	bool recorded = false;
	uint64_t gen = g_gen.load();
};

struct Op
{
	enum Kind : std::uint8_t
	{
		Copy,
		Record,
		Wait,
		Host
	} kind;

	void* dst;
	const void* src;
	std::size_t n;
	Stream* wait_stream;				   // Wait: stream whose op must have run
	uint64_t wait_index;				   // Wait: that op's index
	std::vector<uint8_t>* owned = nullptr; // Copy: a staged source or a lost write, freed once the copy ran
	void (*host_fn)(void*) = nullptr;	   // Host: the callback and its argument
	void* host_arg = nullptr;
};

struct Stream
{
	std::vector<Op> ops;
	uint64_t executed = 0;
	uint64_t gen = g_gen.load();
	bool nonblocking = false; // CU_STREAM_NON_BLOCKING: not synchronized with the legacy stream
};

// Process-lifetime fake-driver state. The preload library's detached reaper may
// still call this DSO during process finalization, after this DSO's C++ static
// destructors would have run. Destroying a map concurrently with unregister is
// a test-fixture race (reproduced under QEMU on the pre-migration build).
// Keep the harness bookkeeping alive, as for g_cb_mu/g_drv_mu below. Individual
// CUDA resources are still explicitly released and checked by the tests. This
// fixture does not validate real-driver process-exit teardown ordering.
std::recursive_mutex& g_lock = *new std::recursive_mutex;
Stream& g_default_stream = *new Stream; // legacy and per-thread default stream
std::vector<Stream*>& g_streams = *new std::vector<Stream*>{&g_default_stream};
std::map<uintptr_t, std::size_t>& g_pinned = *new std::map<uintptr_t, std::size_t>; // base -> size

// Device memory by kind (fake_device_alloc). The test's names for the kinds are the values.
enum DeviceKind : std::uint8_t
{
	kPlainDevice = 0,	 // cudaMalloc: CONTEXT is the context
	kVmmDevice = 1,		 // cuMemCreate + cuMemMap: CONTEXT is NULL, ordinal 0
	kManagedDevice = 2,	 // cudaMallocManaged: type 2, IS_MANAGED 1
	kPeerDevice = 3,	 // another GPU: ordinal 1
	kShortDevice = 4,	 // a pool allocation: range = first half, mapping (pool chunk) = all
	kUnmappedDevice = 5, // a reservation with nothing mapped: a hole throughout
	kSplitDevice = 6,	 // a reservation mapped by 4 adjacent cuMemMap calls (PyTorch expandable segments)
	kManyMappings = 7,	 // mapped 4 KiB at a time: more mappings than the hook walks
	kMiddleHole = 8,	 // mapped [0, n/4), hole [n/4, n/2), mapped [n/2, n)
	kTailHole = 9,		 // mapped [0, n/2), hole [n/2, n)
	kMixedDevice = 10,	 // mapped [0, n/2) from this device, [n/2, n) from device 1
};

struct DeviceAlloc
{
	std::size_t bytes;
	DeviceKind kind;
};

std::map<uintptr_t, DeviceAlloc>& g_device = *new std::map<uintptr_t, DeviceAlloc>; // base -> allocation
std::atomic<unsigned long long> g_fail_pointer_queries{0}; // 0 none, ~0 all, else that address
std::atomic<long> g_generic_calls{0};
long g_captured_generic = 0;		// cuMemcpyAsync calls on the capturing stream (under g_lock)
std::atomic<long> g_hole_copies{0}; // copies refused for touching a hole (the real driver: rc 1, or a crash)

// The cuMemMap mapping that holds address a of allocation [base, base + d.bytes); size 0 for a hole.
struct Mapping
{
	uintptr_t base;
	std::size_t size;
};

Mapping mapping_at(uintptr_t base, const DeviceAlloc& d, uintptr_t a)
{
	const std::size_t n = d.bytes, off = a - base, q = n / 4, half = n / 2;
	switch (d.kind)
	{
	case kUnmappedDevice:
		return {0, 0};
	case kSplitDevice:
		return {base + off / q * q, q};
	case kManyMappings:
		return {base + off / 4096 * 4096, 4096};
	case kMiddleHole:
		return off < q ? Mapping{base, q} : off < half ? Mapping{0, 0} : Mapping{base + half, n - half};
	case kTailHole:
		return off < half ? Mapping{base, half} : Mapping{0, 0};
	case kMixedDevice:
		return off < half ? Mapping{base, half} : Mapping{base + half, n - half};
	default:
		return {base, n};
	}
}

// [p, p + n) touches a hole of some device allocation. Caller holds g_lock.
bool touches_hole(const void* p, std::size_t n)
{
	const uintptr_t a = reinterpret_cast<uintptr_t>(p);
	auto it = g_device.upper_bound(a);
	if (it != std::begin(g_device))
	{
		--it;
	}
	for (; it != std::end(g_device) && it->first < a + n; ++it)
	{
		const uintptr_t lo = std::max(a, it->first), hi = std::min(a + n, it->first + it->second.bytes);
		for (uintptr_t at = lo; at < hi;)
		{
			const Mapping m = mapping_at(it->first, it->second, at);
			if (m.size == 0)
			{
				return true;
			}
			at = m.base + m.size;
		}
	}
	return false;
}
std::map<uintptr_t, uint64_t>& g_alloc_gen =
	*new std::map<uintptr_t, uint64_t>; // cuMemHostAlloc base -> generation (registrations are not here)
std::map<uintptr_t, std::size_t>& g_stale_alloc =
	*new std::map<uintptr_t,
				  std::size_t>; // cuMemHostAlloc memory of a torn-down context (kept, not freed)
int g_primary_refs = 0;			// cuDevicePrimaryCtxRetain count
// Host allocation accounting: bytes, attempts, and injected failures.
long g_host_alloc_bytes = 0, g_host_free_bytes = 0, g_host_alloc_attempts = 0;
long g_host_alloc_flag_attempts[16]{}; // cuMemHostAlloc attempts by flags value (0..15)
int g_host_alloc_failures = 0;		   // positive = the next N fail; -1 = every one fails until cleared
// cuMemHostAlloc flags by base (cuMemHostGetFlags); registrations of any flags that fail (fake_fail_register);
// what cuMemHostGetDevicePointer_v2 adds to the host address (fake_device_pointer_offset).
std::map<uintptr_t, unsigned>& g_alloc_flags = *new std::map<uintptr_t, unsigned>;
std::map<uintptr_t, unsigned>& g_reg_flags = *new std::map<uintptr_t, unsigned>; // registrations' flags
int g_register_failures = 0, g_register_failure_rc = 0;
long g_device_pointer_offset = 0;
// Test callbacks, run at the start of the call and outside g_lock (a callback may block on a hook lock whose
// holder needs the fake).
std::atomic<void (*)()> g_on_host_alloc{nullptr}, g_on_register{nullptr}, g_on_pinned_htod{nullptr};

void run_callback(const std::atomic<void (*)()>& cb)
{
	if (void (*fn)() = cb.load())
	{
		fn();
	}
}

// Threads of the fake (callback thread, driver thread): any API call there fails with 800 and is counted.
thread_local bool t_fake_internal = false;
std::atomic<long> g_internal_api_calls{0};
#define NOT_ON_FAKE_THREAD()                                                                                 \
	do                                                                                                       \
	{                                                                                                        \
		if (t_fake_internal)                                                                                 \
		{                                                                                                    \
			g_internal_api_calls++;                                                                          \
			return 800; /* NOT_PERMITTED */                                                                  \
		}                                                                                                    \
	} while (0)

// fake_set_sticky: a sticky error (CUDA_ERROR_ILLEGAL_ADDRESS, say) killed the context. Every later stream, event,
// copy, synchronization and host-memory call fails with it and no queued work runs, until cuCtxDestroy (which
// discards the queued work and gives a clean context, as a new context would be).
std::atomic<int> g_sticky{0};
#define STICKY()                                                                                             \
	do                                                                                                       \
	{                                                                                                        \
		if (const int sticky_rc = g_sticky.load(); sticky_rc != 0)                                           \
		{                                                                                                    \
			return sticky_rc;                                                                                \
		}                                                                                                    \
	} while (0)

// Stream capture interaction modes (cuThreadExchangeStreamCaptureMode). fake_set_global_capture: some other
// thread captures in CU_STREAM_CAPTURE_MODE_GLOBAL, so a "potentially unsafe" call from a thread in the global
// mode (0, every thread's default) fails with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED and invalidates that capture
// (counted); a thread in relaxed mode (2) may make it.
thread_local int t_capture_mode = 0;
std::atomic<bool> g_global_capture{false};
std::atomic<long> g_capture_invalidations{0};
#define CAPTURE_UNSAFE()                                                                                     \
	do                                                                                                       \
	{                                                                                                        \
		if (g_global_capture.load() && t_capture_mode == 0)                                                  \
		{                                                                                                    \
			g_capture_invalidations++;                                                                       \
			return 900;                                                                                      \
		}                                                                                                    \
	} while (0)

// fake_fail_record_on: cuEventRecord on this stream fails (400), as for a stream of another context.
std::atomic<void*> g_fail_record_stream{nullptr};
// fake_fail_ctx_set_off_main: cuCtxSetCurrent(non-null) fails (999) on every thread but the main one.
std::atomic<bool> g_fail_ctx_set_off_main{false};
// fake_teardown_drops_registrations: cuCtxDestroy and a primary-context teardown also forget every registration,
// as the real driver does for the registrations of the context that goes away.
std::atomic<bool> g_teardown_drops_registrations{false};
void drop_registrations_locked();

// Stale registrations: the epoch of each registration, and every reported page change.
std::atomic<uint64_t> g_epoch{0};
std::map<uintptr_t, uint64_t>& g_reg_epoch =
	*new std::map<uintptr_t, uint64_t>; // cuMemHostRegister base -> epoch (under g_lock)

struct PageChange
{
	uintptr_t s, e;
	uint64_t epoch;
};

// A fixed table (fake_pages_changed runs inside free(): it must not allocate, or the free of the old storage
// would re-enter it), used as a ring: the newest kChanges reports count.
std::mutex& g_change_mu =
	*new std::mutex; // taken alone by fake_pages_changed (from any free()), or after g_lock
constexpr std::size_t kChanges = 1u << 16;
PageChange g_changes[kChanges];
std::size_t g_change_count = 0;
std::atomic<long> g_stale_dmas{0};
std::atomic<long> g_hold_tid{0}; // fake_hold_unregister: only this thread's cuMemHostUnregister runs
// Stream capture: the capturing stream (nullptr = none), and whether the legacy stream reports IMPLICIT.
void* g_capture_stream = nullptr;
int g_capture_legacy_implicit = 0;
long g_captured_ops = 0, g_capture_rejects = 0;
// fake_fail_record_after_dma: 1 = armed, 2 = a pinned DMA was queued, the next record fails.
int g_record_fail = 0;
std::atomic<bool> g_hold_stream_sync{false};
std::atomic<unsigned> g_waiting_stream_sync{0};
int g_stream_sync_failures = 0;
long g_stream_sync_attempts = 0;
long g_ptds_calls = 0;
void* g_ctx = reinterpret_cast<void*>(0x1000);

void run_all();

struct Stats
{
	long htod_async_pageable_src; // HtoDAsync calls whose src is not pinned (pass-through)
	long htod_async_pinned_src;
	long dtoh_async_pageable_dst;
	long dtoh_async_pinned_dst;
	long htod_sync;
	long dtoh_sync;
	long stream_create;
	long stream_wait_event;
	long HostAlloc;
	long host_free;
	long ctx_destroy;
	long host_register;
	long host_unregister;
};

Stats g_stats{};
long g_register_attempts = 0;
long g_unregister_attempts = 0;
int g_unregister_failures = 0; // test injection: positive = next N; -1 = persist until cleared
std::map<uintptr_t, std::size_t>& g_readonly = *new std::map<uintptr_t, std::size_t>;
long g_flag_attempts[16]{};
int g_ro_device = 0, g_ro_supported = 1, g_ro_query_error = 0, g_ro_ctx_error = 0;
int g_ro_register_error = 0, g_ro_fail_at = 1, g_ro_seen = 0, g_rw_register_error = 0;
long g_rw_fail_at = 1, g_rw_seen = 0; // fake_rw_register_error_at: the fail_at-th writable attempt on fails
// cuDeviceTotalMem_v2, progressive pinning's default limit: an RTX 3090's 24 GiB unless fake_device_memory says
// otherwise, per device ordinal below kFakeDevices (cuCtxGetDevice reports g_ro_device), or an error to return.
constexpr int kFakeDevices = 32;
std::size_t g_device_memory[kFakeDevices] = {}; // 0: the 24 GiB default
int g_device_memory_error = 0;
long g_device_memory_queries = 0;
long g_ro_queries = 0;
int g_ro_queried_device = -1;

// Every async copy with a pinned host side (a pipeline DMA), in issue order,
// so a test can see which stream, which staging slot and which chunk size a
// copy used (test_tune_table_e2e). Read through fake_dma_log_*.
struct DmaRecord
{
	Stream* stream;
	const void* host; // the pinned side: a staging slot
	std::size_t n;
	bool h2d;
};

std::vector<DmaRecord>& g_dma_log = *new std::vector<DmaRecord>;

bool readonly_destination(const void* p, std::size_t n)
{
	const uintptr_t a = reinterpret_cast<uintptr_t>(p);
	for (const auto& r : g_readonly)
	{
		if (r.first < a + n && a < r.first + r.second)
		{
			return true;
		}
	}
	return false;
}

// Context per thread, as in CUDA: the process main thread starts with the
// context current; any other thread must call cuCtxSetCurrent first, or
// stream/event/copy calls fail with CUDA_ERROR_INVALID_CONTEXT (201).
thread_local int t_ctx_state = -1; // -1 unknown, 0 none, 1 current

bool ctx_current()
{
	if (t_ctx_state < 0)
	{
		t_ctx_state = syscall(SYS_gettid) == getpid() ? 1 : 0;
	}
	return t_ctx_state == 1;
}

#define REQUIRE_CTX()                                                                                        \
	do                                                                                                       \
	{                                                                                                        \
		if (!ctx_current())                                                                                  \
		{                                                                                                    \
			return 201;                                                                                      \
		}                                                                                                    \
	} while (0)

Stream* stream_of(void* s)
{
	const uintptr_t v = reinterpret_cast<uintptr_t>(s);
	return (v == 0 || v == 1 || v == 2) ? &g_default_stream : static_cast<Stream*>(s);
}

// A handle of a torn-down context: counted, and the call fails with 400 (CUDA_ERROR_INVALID_HANDLE).
bool stale_stream(void* s)
{
	Stream* st = stream_of(s);
	const bool stale = st != &g_default_stream && st->gen < g_gen.load();
	g_stale_uses += stale;
	return stale;
}

bool stale_event(void* e)
{
	const bool stale = static_cast<Event*>(e)->gen < g_gen.load();
	g_stale_uses += stale;
	return stale;
}

bool stale_host(const void* p)
{
	const uintptr_t a = reinterpret_cast<uintptr_t>(p);
	auto it = g_stale_alloc.upper_bound(a);
	const bool stale = it != std::begin(g_stale_alloc) && a < std::prev(it)->first + std::prev(it)->second;
	g_stale_uses += stale;
	return stale;
}

// The primary context is torn down: queued work completes, then everything made before is stale. Caller holds
// g_lock.
void invalidate_context()
{
	run_all();
	g_gen++;
	if (g_teardown_drops_registrations.load())
	{
		drop_registrations_locked();
	}
	for (auto it = std::begin(g_alloc_gen); it != std::end(g_alloc_gen);)
	{
		auto pinned = g_pinned.find(it->first);
		if (pinned != std::end(g_pinned))
		{
			g_stale_alloc[pinned->first] = pinned->second;
			g_pinned.erase(pinned);
		}
		it = g_alloc_gen.erase(it);
	}
}

bool pinned(const void* p)
{
	const uintptr_t a = reinterpret_cast<uintptr_t>(p);
	auto it = g_pinned.upper_bound(a);
	if (it == std::begin(g_pinned))
	{
		return false;
	}
	--it;
	return a < it->first + it->second;
}

// [p, p + n) is inside a cuMemHostRegister'ed range whose pages changed after it was registered. Caller holds
// g_lock.
// The cuMemHostAlloc allocation holding p, or 0. Caller holds g_lock.
uintptr_t host_alloc_base(const void* p)
{
	const uintptr_t a = reinterpret_cast<uintptr_t>(p);
	const auto alloc = g_alloc_flags.upper_bound(a);
	if (alloc == std::begin(g_alloc_flags))
	{
		return 0;
	}
	const auto base = std::prev(alloc)->first;
	const auto range = g_pinned.find(base);
	return range != std::end(g_pinned) && a < base + std::max<std::size_t>(range->second, 1) ? base : 0;
}

bool stale_registration(const void* p, std::size_t n)
{
	const uintptr_t a = reinterpret_cast<uintptr_t>(p);
	auto it = g_pinned.upper_bound(a);
	if (it == std::begin(g_pinned))
	{
		return false;
	}
	--it;
	auto ep = g_reg_epoch.find(it->first);
	if (ep == std::end(g_reg_epoch) || a >= it->first + it->second)
	{
		return false;
	}
	std::lock_guard<std::mutex> g(g_change_mu);
	for (std::size_t i = 0; i < std::min(g_change_count, kChanges); i++)
	{
		const PageChange& c = g_changes[i];
		if (c.epoch > ep->second && c.s < a + n && a < c.e)
		{
			g_stale_dmas++;
			return true;
		}
	}
	return false;
}

// The callback thread: one job at a time (callers hold g_lock), run outside every fake lock.
// Leaked: the callback thread still waits on them at exit, and destroying a waited-on condition variable
// blocks.
std::mutex& g_cb_mu = *new std::mutex;
std::condition_variable& g_cb_cv = *new std::condition_variable;

struct CbJob
{
	void (*fn)(void*);
	void* arg;
	bool done;
};

CbJob* g_cb_job = nullptr;
bool g_cb_started = false;

void* callback_main(void*)
{
	t_fake_internal = true;
	std::unique_lock<std::mutex> lk(g_cb_mu);
	for (;;)
	{
		g_cb_cv.wait(lk, [] { return g_cb_job != nullptr && !g_cb_job->done; });
		CbJob* job = g_cb_job;
		lk.unlock();
		job->fn(job->arg);
		lk.lock();
		job->done = true;
		g_cb_cv.notify_all();
	}
}

// Runs a host callback on the callback thread and waits for it. Caller holds g_lock (the stream is stuck
// meanwhile).
void run_host(void (*fn)(void*), void* arg)
{
	std::unique_lock<std::mutex> lk(g_cb_mu);
	if (!g_cb_started)
	{
		pthread_t t;
		if (pthread_create(&t, nullptr, &callback_main, nullptr) != 0)
		{
			std::abort();
		}
		pthread_detach(t);
		g_cb_started = true;
	}
	CbJob job{fn, arg, false};
	g_cb_job = &job;
	g_cb_cv.notify_all();
	g_cb_cv.wait(lk, [&] { return job.done; });
	g_cb_job = nullptr;
}

void run_until(Stream* s, uint64_t count)
{
	while (s->executed < count && s->executed < std::size(s->ops))
	{
		Op op = s->ops[s->executed];
		if (op.kind == Op::Wait)
		{
			run_until(op.wait_stream, op.wait_index + 1);
		}
		else if (op.kind == Op::Copy)
		{
			std::memcpy(op.dst, op.src, op.n);
			delete op.owned;
			s->ops[s->executed].owned = nullptr;
		}
		else if (op.kind == Op::Host)
		{
			run_host(op.host_fn, op.host_arg);
		}
		s->executed++;
	}
}

// The legacy stream's view of "everything before": the default stream and every blocking stream
// (FAKE_STRICT_LEGACY), else every stream.
void run_legacy()
{
	static const bool strict = []
	{
		const char* v = std::getenv("FAKE_STRICT_LEGACY");
		return v != nullptr && std::atoi(v) != 0;
	}();
	if (!strict)
	{
		run_all();
		return;
	}
	for (Stream* s : g_streams)
	{
		if (!s->nonblocking)
		{
			run_until(s, std::size(s->ops));
		}
	}
}

// The copy is on a capturing stream: CUDA records it into the graph instead of running it, pageable or
// pinned. On the legacy stream during an implicit capture, a pageable copy fails with 900. Returns -1 when
// not capturing, else the result of the call. Caller holds g_lock.
int captured(void* stream, bool pinned_host)
{
	const bool legacy = stream == nullptr || stream == reinterpret_cast<void*>(1);
	if (g_capture_stream == nullptr || (stream != g_capture_stream && !(legacy && g_capture_legacy_implicit)))
	{
		return -1;
	}
	if (!pinned_host && stream != g_capture_stream)
	{
		g_capture_rejects++;
		return 900;
	}
	g_captured_ops++;
	return 0;
}

void run_all()
{
	bool progress = true;
	while (progress)
	{
		progress = false;
		// Newest stream first: independent streams get no accidental order,
		// only what wait-event ops enforce.
		for (auto it = g_streams.rbegin(); it != g_streams.rend(); it++)
		{
			Stream* s = *it;
			if (s->executed < std::size(s->ops))
			{
				run_until(s, std::size(s->ops));
				progress = true;
			}
		}
	}
}

// FAKE_FAIL_PINNED_DMA_AT=n: the n-th and every later async copy with a
// pinned host side (a pipeline DMA) fails with 700 (CUDA_ERROR_ILLEGAL_ADDRESS).
bool inject_failure()
{
	static long count = 0;
	static const long at = []
	{
		const char* v = std::getenv("FAKE_FAIL_PINNED_DMA_AT");
		return v ? std::atol(v) : 0L;
	}();
	return at > 0 && ++count >= at;
}

void enqueue_copy(void* stream, void* dst, const void* src, std::size_t n,
				  std::vector<uint8_t>* owned = nullptr)
{
	Op op{Op::Copy, dst, src, n, nullptr, 0};
	op.owned = owned;
	stream_of(stream)->ops.push_back(op);
}

} // namespace

#if !defined(FAKE_OMIT_CU_INIT) // fake_cuda_no_init: an image the hook's install cannot identify
FAKE_API int cuInit(unsigned) { return 0; }
#endif

FAKE_API int cuCtxGetCurrent(void** ctx)
{
	NOT_ON_FAKE_THREAD();
	*ctx = ctx_current() ? g_ctx : nullptr;
	return 0;
}

FAKE_API int cuCtxSetCurrent(void* ctx)
{
	NOT_ON_FAKE_THREAD();
	if (ctx != nullptr && g_fail_ctx_set_off_main.load() && syscall(SYS_gettid) != getpid())
	{
		return 999;
	}
	t_ctx_state = ctx == g_ctx ? 1 : 0;
	return ctx == g_ctx || ctx == nullptr ? 0 : 201;
}

FAKE_API int cuCtxSynchronize()
{
	NOT_ON_FAKE_THREAD();
	STICKY();
	CAPTURE_UNSAFE();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	run_all();
	return 0;
}

FAKE_API int cuCtxDestroy_v2(void*)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (g_sticky.exchange(0) != 0)
	{
		for (Stream* s : g_streams)
		{
			s->executed = std::size(s->ops); // the dead context's work never runs
		}
	}
	if (g_teardown_drops_registrations.load())
	{
		drop_registrations_locked();
	}
	run_all();
	g_stats.ctx_destroy++;
	return 0;
}

FAKE_API int cuCtxDestroy(void* ctx) { return cuCtxDestroy_v2(ctx); }

FAKE_API int cuPointerGetAttribute(void* data, unsigned attr, unsigned long long ptr)
{
	NOT_ON_FAKE_THREAD();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (attr != 2 || !pinned(reinterpret_cast<const void*>(static_cast<uintptr_t>(ptr))))
	{
		return 1; // untracked
	}
	*static_cast<unsigned*>(data) = 1; // CU_MEMORYTYPE_HOST
	return 0;
}

// Plural lookup has CUDA's successful/default response for an unregistered
// pointer. The trace tests also force a query error; errors must stay unknown.
std::atomic<unsigned> trace_queries{0};

FAKE_API unsigned fake_trace_queries() { return trace_queries.load(); }

namespace {
// What the driver reports for one address (see the header). Caller holds g_lock.
struct PointerFacts
{
	unsigned type = 0;
	unsigned managed = 0;
	int ordinal = -1;
	void* context = nullptr;
	unsigned long long start = 0;
	std::size_t size = 0;
	unsigned long long mapping_base = 0;
	std::size_t mapping_size = 0;
};

PointerFacts facts_of(uintptr_t a)
{
	PointerFacts f;
	auto device = g_device.upper_bound(a);
	if (device != std::begin(g_device) && a < std::prev(device)->first + std::prev(device)->second.bytes)
	{
		--device;
		const DeviceKind kind = device->second.kind;
		f.start = device->first;
		f.size = kind == kShortDevice ? device->second.bytes / 2 : device->second.bytes;
		const Mapping m = mapping_at(device->first, device->second, a);
		if (m.size == 0)
		{
			return f; // a hole: type 0, ordinal -1, but inside the reservation's range
		}
		f.type = 2;
		f.managed = kind == kManagedDevice ? 1 : 0;
		f.ordinal = kind == kPeerDevice || (kind == kMixedDevice && m.base != device->first) ? 1 : 0;
		f.context = kind == kVmmDevice || kind >= kUnmappedDevice ? nullptr : g_ctx;
		f.mapping_base = m.base;
		f.mapping_size = m.size;
		return f;
	}
	auto host = g_pinned.upper_bound(a);
	if (host != std::begin(g_pinned) && a < std::prev(host)->first + std::prev(host)->second)
	{
		--host;
		f.type = 1;
		f.ordinal = 0;
		f.context = g_ctx;
		f.start = host->first;
		f.size = host->second;
		f.mapping_base = host->first;
		f.mapping_size = host->second;
	}
	return f;
}

bool is_device(const void* p)
{
	return facts_of(reinterpret_cast<uintptr_t>(p)).type == 2;
}
} // namespace

FAKE_API int cuPointerGetAttributes(unsigned count, const unsigned* attrs, void** values,
									unsigned long long ptr)
{
	trace_queries.fetch_add(1);
	const unsigned long long failing = g_fail_pointer_queries.load();
	if (std::getenv("FAKE_TRACE_QUERY_FAIL") || failing == ~0ull || (failing != 0 && failing == ptr))
	{
		return 201;
	}
	std::lock_guard<std::recursive_mutex> g(g_lock);
	const PointerFacts f = facts_of(static_cast<uintptr_t>(ptr));
	for (unsigned i = 0; i < count; ++i)
	{
		switch (attrs[i])
		{
		case 1: // CONTEXT
			*static_cast<void**>(values[i]) = f.context;
			break;
		case 2: // MEMORY_TYPE
			*static_cast<unsigned*>(values[i]) = f.type;
			break;
		case 8: // IS_MANAGED
			*static_cast<unsigned*>(values[i]) = f.managed;
			break;
		case 9: // DEVICE_ORDINAL
			*static_cast<int*>(values[i]) = f.ordinal;
			break;
		case 11: // RANGE_START_ADDR
			*static_cast<unsigned long long*>(values[i]) = f.start;
			break;
		case 12: // RANGE_SIZE
			*static_cast<std::size_t*>(values[i]) = f.size;
			break;
		case 18: // MAPPING_SIZE
			*static_cast<std::size_t*>(values[i]) = f.mapping_size;
			break;
		case 19: // MAPPING_BASE_ADDR
			*static_cast<unsigned long long*>(values[i]) = f.mapping_base;
			break;
		default:
			return 1;
		}
	}
	return 0;
}

FAKE_API int cuMemHostAlloc(void** p, std::size_t n, unsigned flags)
{
	NOT_ON_FAKE_THREAD();
	STICKY();
	CAPTURE_UNSAFE();
	run_callback(g_on_host_alloc);
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_host_alloc_attempts++;
	if (flags < 16)
	{
		g_host_alloc_flag_attempts[flags]++;
	}
	if (g_host_alloc_failures != 0)
	{
		if (g_host_alloc_failures > 0)
		{
			--g_host_alloc_failures;
		}
		return 2; // CUDA_ERROR_OUT_OF_MEMORY
	}
	if (posix_memalign(p, 4096, n == 0 ? 1 : n) != 0)
	{
		return 2;
	}
	g_pinned[reinterpret_cast<uintptr_t>(*p)] = n;
	g_alloc_gen[reinterpret_cast<uintptr_t>(*p)] = g_gen.load();
	g_alloc_flags[reinterpret_cast<uintptr_t>(*p)] = flags;
	g_stats.HostAlloc++;
	g_host_alloc_bytes += static_cast<long>(n);
	return 0;
}

FAKE_API int cuMemAllocHost(void** p, std::size_t n) { return cuMemHostAlloc(p, n, 0); }
FAKE_API int cuMemAllocHost_v2(void** p, std::size_t n) { return cuMemHostAlloc(p, n, 0); }

// As the 615 driver answers (tools/semantics.cpp of results/rtx3090-hostalloc-hook-20261111): cuMemHostAlloc memory
// reports its flags with DEVICEMAP added, at any address inside it; other memory is CUDA_ERROR_INVALID_VALUE.
FAKE_API int cuMemHostGetFlags(unsigned* flags, void* p)
{
	NOT_ON_FAKE_THREAD();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (flags == nullptr)
	{
		return 1;
	}
	if (const uintptr_t base = host_alloc_base(p))
	{
		*flags = g_alloc_flags[base] | 2;
		return 0;
	}
	const uintptr_t a = reinterpret_cast<uintptr_t>(p);
	const auto reg = g_reg_flags.upper_bound(a);
	if (reg == std::begin(g_reg_flags) || !pinned(p) || a >= std::prev(reg)->first + g_pinned[std::prev(reg)->first])
	{
		return 1;
	}
	*flags = std::prev(reg)->second | 2; // a registration reports its own flags, with DEVICEMAP
	return 0;
}

// Pinned memory is mapped at its host address (unified addressing), plus fake_device_pointer_offset.
FAKE_API int cuMemHostGetDevicePointer_v2(unsigned long long* d, void* p, unsigned flags)
{
	NOT_ON_FAKE_THREAD();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (d == nullptr || flags != 0 || !pinned(p))
	{
		return 1;
	}
	*d = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p) + g_device_pointer_offset);
	return 0;
}

// Registration of existing memory: tracked like cuMemHostAlloc memory; overlapping a registered range fails
// like the driver (CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712), unregistering an unknown base fails
// (713).
FAKE_API int cuMemHostRegister_v2(void* p, std::size_t n, unsigned flags)
{
	NOT_ON_FAKE_THREAD();
	STICKY();
	CAPTURE_UNSAFE();
	run_callback(g_on_register);
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_register_attempts++;
	if (flags < 16)
	{
		++g_flag_attempts[flags];
	}
	if (g_register_failures != 0)
	{
		if (g_register_failures > 0)
		{
			--g_register_failures;
		}
		return g_register_failure_rc;
	}
	if ((flags & 8) != 0)
	{
		g_ro_seen++;
		if (g_ro_register_error && g_ro_seen >= g_ro_fail_at)
		{
			return g_ro_register_error;
		}
		if (!g_ro_supported)
		{
			return 801;
		}
	}
	else if (flags == 1 && g_rw_register_error && ++g_rw_seen >= g_rw_fail_at)
	{
		return g_rw_register_error;
	}
	const uintptr_t a = reinterpret_cast<uintptr_t>(p);
	for (const auto& e : g_pinned)
	{
		if (e.first < a + n && a < e.first + e.second)
		{
			// Over cuMemHostAlloc memory the 615 driver says INVALID_VALUE; over a registration, 712.
			return g_alloc_flags.count(e.first) != 0 ? 1 : 712;
		}
	}
	g_pinned[a] = n;
	g_reg_epoch[a] = ++g_epoch;
	g_reg_flags[a] = flags;
	if ((flags & 8) != 0)
	{
		g_readonly[a] = n;
	}
	g_stats.host_register++;
	return 0;
}

FAKE_API int cuMemHostRegister(void* p, std::size_t n, unsigned f) { return cuMemHostRegister_v2(p, n, f); }

FAKE_API int cuMemHostUnregister(void* p)
{
	NOT_ON_FAKE_THREAD();
	// fake_hold_unregister: other threads' unregistrations wait here, outside every fake lock.
	for (long holder; (holder = g_hold_tid.load()) != 0 && holder != static_cast<long>(syscall(SYS_gettid));)
	{
		usleep(100);
	}
	STICKY();
	CAPTURE_UNSAFE();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_unregister_attempts++;
	run_all();
	if (host_alloc_base(p) != 0)
	{
		return 1; // cuMemHostAlloc memory, as the 615 driver answers
	}
	auto it = g_pinned.find(reinterpret_cast<uintptr_t>(p));
	if (it == std::end(g_pinned))
	{
		return 713;
	}
	if (g_unregister_failures != 0)
	{
		if (g_unregister_failures > 0)
		{
			--g_unregister_failures;
		}
		return 700; // fail without releasing the registered range
	}
	g_readonly.erase(it->first);
	g_reg_epoch.erase(it->first);
	g_reg_flags.erase(it->first);
	g_pinned.erase(it);
	g_stats.host_unregister++;
	return 0;
}

FAKE_API int cuMemFreeHost(void* p)
{
	NOT_ON_FAKE_THREAD();
	STICKY();
	CAPTURE_UNSAFE();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	run_all();
	if (stale_host(p))
	{
		// The context is gone. Retain the memory so a use-after-reset reads garbage.
		return 400;
	}
	if (p != nullptr && g_alloc_flags.count(reinterpret_cast<uintptr_t>(p)) == 0)
	{
		return 1; // not cuMemHostAlloc memory (CUDA_ERROR_INVALID_VALUE): never free() a foreign pointer
	}
	auto it = g_pinned.find(reinterpret_cast<uintptr_t>(p));
	if (it != std::end(g_pinned))
	{
		g_host_free_bytes += static_cast<long>(it->second);
		g_pinned.erase(it);
	}
	g_alloc_gen.erase(reinterpret_cast<uintptr_t>(p));
	g_alloc_flags.erase(reinterpret_cast<uintptr_t>(p));
	std::free(p);
	g_stats.host_free++;
	return 0;
}

// Device memory is host memory. FAKE_CUMEMALLOC_FAIL=1 makes it fail.
FAKE_API int cuMemAlloc_v2(unsigned long long* dptr, std::size_t n)
{
	const char* fail = std::getenv("FAKE_CUMEMALLOC_FAIL");
	if (fail != nullptr && std::atoi(fail) != 0)
	{
		return 2; // CUDA_ERROR_OUT_OF_MEMORY
	}
	void* p = nullptr;
	if (posix_memalign(&p, 4096, n == 0 ? 1 : n) != 0)
	{
		return 2;
	}
	*dptr = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p));
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_device[reinterpret_cast<uintptr_t>(p)] = DeviceAlloc{n == 0 ? 1 : n, kPlainDevice};
	return 0;
}

// Device memory of a kind (DeviceKind above): what cuPointerGetAttributes reports for it. Freed by
// cuMemFree_v2.
FAKE_API int fake_device_alloc(unsigned long long* dptr, std::size_t n, int kind)
{
	const int rc = cuMemAlloc_v2(dptr, n);
	if (rc == 0)
	{
		std::lock_guard<std::recursive_mutex> g(g_lock);
		g_device[static_cast<uintptr_t>(*dptr)].kind = static_cast<DeviceKind>(kind);
	}
	return rc;
}

FAKE_API int cuMemFree_v2(unsigned long long dptr)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	run_all();
	g_device.erase(static_cast<uintptr_t>(dptr));
	std::free(reinterpret_cast<void*>(static_cast<uintptr_t>(dptr)));
	return 0;
}

FAKE_API void fake_fail_pointer_queries(unsigned long long address)
{
	g_fail_pointer_queries.store(address);
}

FAKE_API long fake_generic_calls()
{
	return g_generic_calls.load();
}

FAKE_API long fake_captured_generic()
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	return g_captured_generic;
}

FAKE_API long fake_hole_copies()
{
	return g_hole_copies.load();
}

FAKE_API int cuEventCreate(void** e, unsigned)
{
	*e = new Event;
	return 0;
}

FAKE_API int cuEventDestroy_v2(void* e)
{
	STICKY();
	if (stale_event(e))
	{
		return 400; // destroyed with its context (leaked here, so the test fails on the count, not a crash)
	}
	delete static_cast<Event*>(e);
	return 0;
}

FAKE_API int cuEventDestroy(void* e) { return cuEventDestroy_v2(e); }

FAKE_API int cuEventRecord(void* ev, void* stream)
{
	NOT_ON_FAKE_THREAD();
	REQUIRE_CTX();
	STICKY();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (stale_event(ev) || stale_stream(stream))
	{
		return 400;
	}
	if (stream != nullptr && stream == g_fail_record_stream.load())
	{
		return 400; // CUDA_ERROR_INVALID_HANDLE: the stream belongs to another context
	}
	if (g_record_fail == 2)
	{
		g_record_fail = 0;
		return 999; // CUDA_ERROR_UNKNOWN; the DMA before it stays queued
	}
	Event* e = static_cast<Event*>(ev);
	Stream* s = stream_of(stream);
	e->stream = s;
	e->index = std::size(s->ops);
	e->recorded = true;
	s->ops.push_back(Op{Op::Record, nullptr, nullptr, 0, nullptr, 0});
	return 0;
}

FAKE_API int cuEventSynchronize(void* ev)
{
	NOT_ON_FAKE_THREAD();
	REQUIRE_CTX();
	STICKY();
	CAPTURE_UNSAFE(); // as the 615 driver answers a global-mode thread during another thread's global capture
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (stale_event(ev))
	{
		return 400;
	}
	Event* e = static_cast<Event*>(ev);
	if (e->recorded)
	{
		run_until(e->stream, e->index + 1);
	}
	return 0;
}

FAKE_API int cuStreamCreate(void** s, unsigned flags)
{
	NOT_ON_FAKE_THREAD();
	REQUIRE_CTX();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	Stream* st = new Stream;
	st->nonblocking = (flags & 1) != 0;
	g_streams.push_back(st);
	*s = st;
	g_stats.stream_create++;
	return 0;
}

FAKE_API int cuStreamDestroy_v2(void* s)
{
	STICKY();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (stale_stream(s))
	{
		return 400;
	}
	run_all(); // streams are leaked (kept in g_streams) to keep run_all simple
	(void)s;
	return 0;
}

FAKE_API int cuStreamWaitEvent(void* stream, void* ev, unsigned)
{
	NOT_ON_FAKE_THREAD();
	REQUIRE_CTX();
	STICKY();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (stale_stream(stream) || stale_event(ev))
	{
		return 400;
	}
	Event* e = static_cast<Event*>(ev);
	g_stats.stream_wait_event++;
	if (e->recorded)
	{
		stream_of(stream)->ops.push_back(Op{Op::Wait, nullptr, nullptr, 0, e->stream, e->index});
	}
	return 0;
}

// Test-only completion gate: hold outside the driver's lock so an ordinary
// unmap thread can reach the hook's registration retirement logic.
FAKE_API void fake_hold_stream_sync(int hold) { g_hold_stream_sync.store(hold != 0); }

FAKE_API unsigned fake_waiting_stream_sync() { return g_waiting_stream_sync.load(); }

FAKE_API int cuStreamSynchronize(void* stream)
{
	NOT_ON_FAKE_THREAD();
	if (g_hold_stream_sync.load())
	{
		g_waiting_stream_sync.fetch_add(1);
		while (g_hold_stream_sync.load())
		{
			std::this_thread::yield();
		}
		g_waiting_stream_sync.fetch_sub(1);
	}
	std::lock_guard<std::recursive_mutex> g(g_lock);
	++g_stream_sync_attempts;
	STICKY();
	CAPTURE_UNSAFE();
	if (g_stream_sync_failures != 0)
	{
		if (g_stream_sync_failures > 0)
		{
			--g_stream_sync_failures;
		}
		return 700; // Do not execute pending DMA: cleanup must keep its storage live.
	}
	if (stale_stream(stream))
	{
		return 400;
	}
	Stream* s = stream_of(stream);
	run_until(s, std::size(s->ops));
	return 0;
}

FAKE_API int cuMemcpyHtoDAsync_v2(unsigned long long dst, const void* src, std::size_t n, void* stream)
{
	NOT_ON_FAKE_THREAD();
	REQUIRE_CTX();
	if (g_on_pinned_htod.load() != nullptr)
	{
		bool is_pinned = false;
		{
			std::lock_guard<std::recursive_mutex> g(g_lock);
			is_pinned = pinned(src);
		}
		if (is_pinned)
		{
			run_callback(g_on_pinned_htod);
		}
	}
	STICKY();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (touches_hole(reinterpret_cast<const void*>(static_cast<uintptr_t>(dst)), n) || touches_hole(src, n))
	{
		g_hole_copies++;
		return 1;
	}
	if (stale_stream(stream) || stale_host(src))
	{
		return 400;
	}
	const int cap = captured(stream, pinned(src));
	if (cap >= 0)
	{
		return cap;
	}
	if (pinned(src) && g_record_fail == 1)
	{
		g_record_fail = 2;
	}
	if (!pinned(src))
	{
		// CUDA stages a pageable source before returning: snapshot it.
		g_stats.htod_async_pageable_src++;
		auto* copy =
			new std::vector<uint8_t>(static_cast<const uint8_t*>(src), static_cast<const uint8_t*>(src) + n);
		enqueue_copy(stream, reinterpret_cast<void*>(static_cast<uintptr_t>(dst)), std::data(*copy), n, copy);
		return 0;
	}
	if (inject_failure())
	{
		return 700;
	}
	g_stats.htod_async_pinned_src++;
	g_dma_log.push_back(DmaRecord{stream_of(stream), src, n, true});
	if (stale_registration(src, n))
	{ // the old pages: not what the application wrote at this address
		auto* old = new std::vector<uint8_t>(n, 0xDB);
		enqueue_copy(stream, reinterpret_cast<void*>(static_cast<uintptr_t>(dst)), std::data(*old), n, old);
		return 0;
	}
	enqueue_copy(stream, reinterpret_cast<void*>(static_cast<uintptr_t>(dst)), src, n);
	return 0;
}

FAKE_API int cuMemcpyDtoHAsync_v2(void* dst, unsigned long long src, std::size_t n, void* stream)
{
	NOT_ON_FAKE_THREAD();
	REQUIRE_CTX();
	STICKY();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (touches_hole(reinterpret_cast<const void*>(static_cast<uintptr_t>(src)), n) || touches_hole(dst, n))
	{
		g_hole_copies++;
		return 1;
	}
	if (stale_stream(stream) || stale_host(dst))
	{
		return 400;
	}
	if (readonly_destination(dst, n))
	{
		return 1;
	}
	const int cap = captured(stream, pinned(dst));
	if (cap >= 0)
	{
		return cap;
	}
	if (pinned(dst) && inject_failure())
	{
		return 700;
	}
	if (pinned(dst) && g_record_fail == 1)
	{
		g_record_fail = 2;
	}
	if (pinned(dst) && stale_registration(dst, n))
	{ // written to the old pages: lost for the application
		auto* lost = new std::vector<uint8_t>(n);
		enqueue_copy(stream, std::data(*lost), reinterpret_cast<const void*>(static_cast<uintptr_t>(src)), n,
					 lost);
	}
	else
	{
		enqueue_copy(stream, dst, reinterpret_cast<const void*>(static_cast<uintptr_t>(src)), n);
	}
	if (!pinned(dst))
	{
		// CUDA returns from a D2H into pageable memory only when it is done.
		g_stats.dtoh_async_pageable_dst++;
		Stream* s = stream_of(stream);
		run_until(s, std::size(s->ops));
	}
	else
	{
		g_stats.dtoh_async_pinned_dst++;
		g_dma_log.push_back(DmaRecord{stream_of(stream), dst, n, false});
	}
	return 0;
}

// Synchronous copies: everything queued before them runs first.
FAKE_API int cuMemcpyHtoD_v2(unsigned long long dst, const void* src, std::size_t n)
{
	NOT_ON_FAKE_THREAD();
	STICKY();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (touches_hole(reinterpret_cast<const void*>(static_cast<uintptr_t>(dst)), n) || touches_hole(src, n))
	{
		g_hole_copies++;
		return 1;
	}
	const int cap = captured(nullptr, pinned(src));
	if (cap >= 0)
	{
		return cap;
	}
	g_stats.htod_sync++;
	run_legacy();
	if (pinned(src) && stale_registration(src, n))
	{
		std::memset(reinterpret_cast<void*>(static_cast<uintptr_t>(dst)), 0xDB, n);
		return 0;
	}
	std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(dst)), src, n);
	return 0;
}

FAKE_API int cuMemcpyDtoH_v2(void* dst, unsigned long long src, std::size_t n)
{
	NOT_ON_FAKE_THREAD();
	STICKY();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (touches_hole(reinterpret_cast<const void*>(static_cast<uintptr_t>(src)), n) || touches_hole(dst, n))
	{
		g_hole_copies++;
		return 1;
	}
	if (readonly_destination(dst, n))
	{
		return 1;
	}
	const int cap = captured(nullptr, pinned(dst));
	if (cap >= 0)
	{
		return cap;
	}
	g_stats.dtoh_sync++;
	run_legacy();
	if (pinned(dst) && stale_registration(dst, n))
	{
		return 0;
	}
	std::memcpy(dst, reinterpret_cast<const void*>(static_cast<uintptr_t>(src)), n);
	return 0;
}

FAKE_API int cuMemcpyHtoD(unsigned int dst, const void* src, unsigned int n)
{
	return cuMemcpyHtoD_v2(dst, src, n);
}

namespace {
// Whether each end is device memory. The typed fakes run outside this lock: some run test callbacks first.
std::pair<bool, bool> devices_of(const void* dst, const void* src)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	return {is_device(dst), is_device(src)};
}
} // namespace

// The generic copies infer the direction as the driver does and run the typed fake for it, so pinned,
// pageable, stale and capture behaviour is the typed one. Device-to-device and host-to-host run in stream
// order (sync: after the legacy stream's work). nullptr fails (INVALID_VALUE) before any write, as on the
// driver.
FAKE_API int cuMemcpy(unsigned long long dst, unsigned long long src, std::size_t n)
{
	NOT_ON_FAKE_THREAD();
	g_generic_calls++;
	if (dst == 0 || src == 0)
	{
		return 1;
	}
	auto* d = reinterpret_cast<void*>(static_cast<uintptr_t>(dst));
	const auto* s = reinterpret_cast<const void*>(static_cast<uintptr_t>(src));
	const auto [to_device, from_device] = devices_of(d, s);
	if (to_device != from_device)
	{
		return to_device ? cuMemcpyHtoD_v2(dst, s, n) : cuMemcpyDtoH_v2(d, src, n);
	}
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (touches_hole(d, n) || touches_hole(s, n))
	{
		g_hole_copies++;
		return 1;
	}
	run_legacy();
	std::memcpy(d, s, n);
	return 0;
}

FAKE_API int cuMemcpyAsync(unsigned long long dst, unsigned long long src, std::size_t n, void* stream)
{
	NOT_ON_FAKE_THREAD();
	REQUIRE_CTX();
	g_generic_calls++;
	if (dst == 0 || src == 0)
	{
		return 1;
	}
	auto* d = reinterpret_cast<void*>(static_cast<uintptr_t>(dst));
	const auto* s = reinterpret_cast<const void*>(static_cast<uintptr_t>(src));
	{
		std::lock_guard<std::recursive_mutex> g(g_lock);
		g_captured_generic += g_capture_stream != nullptr && stream == g_capture_stream ? 1 : 0;
	}
	const auto [to_device, from_device] = devices_of(d, s);
	if (to_device != from_device)
	{
		return to_device ? cuMemcpyHtoDAsync_v2(dst, s, n, stream) : cuMemcpyDtoHAsync_v2(d, src, n, stream);
	}
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (touches_hole(d, n) || touches_hole(s, n))
	{
		g_hole_copies++;
		return 1;
	}
	enqueue_copy(stream, d, s, n);
	if (!to_device)
	{
		Stream* st = stream_of(stream);
		run_until(st, std::size(st->ops)); // a host destination is written when the call returns
	}
	return 0;
}

FAKE_API int cuMemcpy_ptds(unsigned long long dst, unsigned long long src, std::size_t n)
{
	g_ptds_calls++;
	return cuMemcpy(dst, src, n);
}

FAKE_API int cuMemcpyAsync_ptsz(unsigned long long dst, unsigned long long src, std::size_t n, void* stream)
{
	g_ptds_calls++;
	return cuMemcpyAsync(dst, src, n, stream);
}

// Per-thread default stream variants: the fake has one default stream, so they behave like the legacy ones;
// they are separate symbols so a test can see which one an application got (and count their calls).
FAKE_API int cuMemcpyHtoD_v2_ptds(unsigned long long dst, const void* src, std::size_t n)
{
	g_ptds_calls++;
	return cuMemcpyHtoD_v2(dst, src, n);
}

FAKE_API int cuMemcpyDtoH_v2_ptds(void* dst, unsigned long long src, std::size_t n)
{
	g_ptds_calls++;
	return cuMemcpyDtoH_v2(dst, src, n);
}

FAKE_API int cuMemcpyHtoDAsync_v2_ptsz(unsigned long long dst, const void* src, std::size_t n, void* stream)
{
	g_ptds_calls++;
	return cuMemcpyHtoDAsync_v2(dst, src, n, stream);
}

FAKE_API int cuMemcpyDtoHAsync_v2_ptsz(void* dst, unsigned long long src, std::size_t n, void* stream)
{
	g_ptds_calls++;
	return cuMemcpyDtoHAsync_v2(dst, src, n, stream);
}

// CU_STREAM_CAPTURE_STATUS_NONE = 0, ACTIVE = 1; CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906 for the legacy
// stream while a blocking stream captures.
FAKE_API int cuStreamIsCapturing(void* stream, int* status)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	const bool legacy = stream == nullptr || stream == reinterpret_cast<void*>(1);
	if (legacy && g_capture_stream != nullptr && g_capture_legacy_implicit)
	{
		return 906;
	}
	*status = g_capture_stream != nullptr && stream == g_capture_stream ? 1 : 0;
	return 0;
}

// Primary context: one per process (the fake's only context, 0x1000). Retain counts; the release that brings
// the count to 0, and every reset, tear the context down (invalidate_context). The v1 and _v2 entry points
// behave the same here; they are separate symbols so the hook's choice between them can be checked by
// address.
FAKE_API int cuDevicePrimaryCtxRetain(void** ctx, int)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_primary_refs++;
	*ctx = g_ctx;
	return 0;
}

FAKE_API int cuDevicePrimaryCtxRelease_v2(int)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (g_primary_refs > 0 && --g_primary_refs == 0)
	{
		invalidate_context();
	}
	return 0;
}

FAKE_API int cuDevicePrimaryCtxRelease(int dev) { return cuDevicePrimaryCtxRelease_v2(dev); }

FAKE_API int cuDevicePrimaryCtxReset_v2(int)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_primary_refs = 0;
	invalidate_context();
	return 0;
}

FAKE_API int cuDevicePrimaryCtxReset(int dev) { return cuDevicePrimaryCtxReset_v2(dev); }

namespace {
// Not exported: what a driver could return from cuGetProcAddress for some version without exporting it.
int primary_reset_internal(int dev) { return cuDevicePrimaryCtxReset_v2(dev); }

template <int Index>
int primary_reset_distinct(int dev)
{
	return cuDevicePrimaryCtxReset_v2(dev) + Index;
}

template <int Index>
int context_destroy_distinct(void* ctx)
{
	return cuCtxDestroy_v2(ctx) + 77 + Index;
}
} // namespace

FAKE_API int cuGetProcAddress(const char* symbol, void** pfn, int, unsigned long long);

// CUDA 12 form: same lookup, writes the query result (0 = found, 1 = symbol not found).
FAKE_API int cuGetProcAddress_v2(const char* symbol, void** pfn, int v, unsigned long long f, void* status)
{
	// A completed lookup may reject the requested symbol/ABI independently of rc.
	if (symbol != nullptr && pfn != nullptr && f <= 2 && v == 3010 &&
		std::strcmp(symbol, "cuMemcpyHtoD_v2") == 0)
	{
		*pfn = nullptr;
		if (status != nullptr)
		{
			*static_cast<int*>(status) = 2; // VERSION_NOT_SUFFICIENT
		}
		return 0;
	}
	const int rc = cuGetProcAddress(symbol, pfn, v, f);
	if (status != nullptr)
	{
		*static_cast<int*>(status) = rc == 0 ? 0 : 1;
	}
	return rc;
}

FAKE_API int cuGetProcAddress(const char* symbol, void** pfn, int version, unsigned long long flags)
{
	struct Entry
	{
		const char* name;
		void* fn;
	};

	if (symbol == nullptr || pfn == nullptr || version < 0 || version > 13000 || flags > 2)
	{
		return 1; // INVALID_VALUE; do not modify the caller's pointer
	}
	if (version >= 100 && version <= 108)
	{
		const auto index = static_cast<std::size_t>(version - 100);
		const Entry primary[] = {
			{"", reinterpret_cast<void*>(&primary_reset_distinct<0>)},
			{"", reinterpret_cast<void*>(&primary_reset_distinct<1>)},
			{"", reinterpret_cast<void*>(&primary_reset_distinct<2>)},
			{"", reinterpret_cast<void*>(&primary_reset_distinct<3>)},
			{"", reinterpret_cast<void*>(&primary_reset_distinct<4>)},
			{"", reinterpret_cast<void*>(&primary_reset_distinct<5>)},
			{"", reinterpret_cast<void*>(&primary_reset_distinct<6>)},
			{"", reinterpret_cast<void*>(&primary_reset_distinct<7>)},
			{"", reinterpret_cast<void*>(&primary_reset_distinct<8>)},
		};
		const Entry contexts[] = {
			{"", reinterpret_cast<void*>(&context_destroy_distinct<0>)},
			{"", reinterpret_cast<void*>(&context_destroy_distinct<1>)},
			{"", reinterpret_cast<void*>(&context_destroy_distinct<2>)},
			{"", reinterpret_cast<void*>(&context_destroy_distinct<3>)},
			{"", reinterpret_cast<void*>(&context_destroy_distinct<4>)},
			{"", reinterpret_cast<void*>(&context_destroy_distinct<5>)},
			{"", reinterpret_cast<void*>(&context_destroy_distinct<6>)},
			{"", reinterpret_cast<void*>(&context_destroy_distinct<7>)},
			{"", reinterpret_cast<void*>(&context_destroy_distinct<8>)},
		};
		if (std::strcmp(symbol, "cuDevicePrimaryCtxReset") == 0)
		{
			*pfn = primary[index].fn;
			return 0;
		}
		if (std::strcmp(symbol, "cuCtxDestroy") == 0)
		{
			*pfn = contexts[index].fn;
			return 0;
		}
	}
	if (version < 3020 && std::strcmp(symbol, "cuMemcpyHtoD") == 0)
	{
		*pfn = reinterpret_cast<void*>(&cuMemcpyHtoD);
		return 0;
	}
	if (version < 3020 && std::strcmp(symbol, "cuMemAllocHost") == 0)
	{
		*pfn = reinterpret_cast<void*>(&cuMemAllocHost); // the 32-bit-size ABI of the real driver
		return 0;
	}
	if (version == 1 && std::strcmp(symbol, "cuDevicePrimaryCtxReset") == 0)
	{
		*pfn = reinterpret_cast<void*>(&primary_reset_internal);
		return 0;
	}
	const Entry ptds[] = {
		{"cuMemcpyHtoD", reinterpret_cast<void*>(&cuMemcpyHtoD_v2_ptds)},
		{"cuMemcpyDtoH", reinterpret_cast<void*>(&cuMemcpyDtoH_v2_ptds)},
		{"cuMemcpyHtoDAsync", reinterpret_cast<void*>(&cuMemcpyHtoDAsync_v2_ptsz)},
		{"cuMemcpyDtoHAsync", reinterpret_cast<void*>(&cuMemcpyDtoHAsync_v2_ptsz)},
		{"cuMemcpy", reinterpret_cast<void*>(&cuMemcpy_ptds)},
		{"cuMemcpyAsync", reinterpret_cast<void*>(&cuMemcpyAsync_ptsz)},
	};
	if ((flags & 2) != 0)
	{ // CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
		for (const Entry& e : ptds)
		{
			if (std::strcmp(e.name, symbol) == 0)
			{
				*pfn = e.fn;
				return 0;
			}
		}
	}
	if (std::strcmp(symbol, "cuGetProcAddress") == 0)
	{
		*pfn = version >= 12000 ? reinterpret_cast<void*>(&cuGetProcAddress_v2)
								: reinterpret_cast<void*>(&cuGetProcAddress);
		return 0;
	}
	const Entry table[] = {
		{"cuMemcpyDtoH", reinterpret_cast<void*>(&cuMemcpyDtoH_v2)},
		{"cuMemcpyDtoHAsync", reinterpret_cast<void*>(&cuMemcpyDtoHAsync_v2)},
		{"cuMemcpyHtoD", reinterpret_cast<void*>(&cuMemcpyHtoD_v2)},
		{"cuMemcpyHtoDAsync", reinterpret_cast<void*>(&cuMemcpyHtoDAsync_v2)},
		{"cuMemcpy", reinterpret_cast<void*>(&cuMemcpy)},
		{"cuMemcpyAsync", reinterpret_cast<void*>(&cuMemcpyAsync)},
		{"cuCtxSynchronize", reinterpret_cast<void*>(&cuCtxSynchronize)},
		{"cuCtxDestroy", reinterpret_cast<void*>(&cuCtxDestroy_v2)},
		// As a CUDA >= 11 driver answers: the unversioned names give the _v2 functions.
		{"cuDevicePrimaryCtxRelease", reinterpret_cast<void*>(&cuDevicePrimaryCtxRelease_v2)},
		{"cuDevicePrimaryCtxReset", reinterpret_cast<void*>(&cuDevicePrimaryCtxReset_v2)},
		{"cuDevicePrimaryCtxRetain", reinterpret_cast<void*>(&cuDevicePrimaryCtxRetain)},
		{"cuMemHostAlloc", reinterpret_cast<void*>(&cuMemHostAlloc)},
		{"cuMemAllocHost", reinterpret_cast<void*>(&cuMemAllocHost_v2)},
		{"cuMemFreeHost", reinterpret_cast<void*>(&cuMemFreeHost)},
		{"cuMemHostGetFlags", reinterpret_cast<void*>(&cuMemHostGetFlags)},
		{"cuMemHostUnregister", reinterpret_cast<void*>(&cuMemHostUnregister)},
	};
	for (const Entry& e : table)
	{
		if (std::strcmp(e.name, symbol) == 0)
		{
			*pfn = e.fn;
			return 0;
		}
	}
	*pfn = nullptr;
	return 500; // CUDA_ERROR_NOT_FOUND
}

FAKE_API int cuCtxGetDevice(int* dev)
{
	REQUIRE_CTX();
	if (g_ro_ctx_error)
	{
		return g_ro_ctx_error;
	}
	*dev = g_ro_device;
	return 0;
}

FAKE_API int cuDeviceTotalMem_v2(std::size_t* bytes, int device)
{
	NOT_ON_FAKE_THREAD();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_device_memory_queries++;
	if (g_device_memory_error)
	{
		return g_device_memory_error;
	}
	if (bytes == nullptr || device < 0 || device >= kFakeDevices)
	{
		return 101; // CUDA_ERROR_INVALID_DEVICE
	}
	*bytes = g_device_memory[device] != 0 ? g_device_memory[device] : std::size_t{24} << 30;
	return 0;
}

FAKE_API int cuDeviceGetName(char* name, int length, int device)
{
	if (!name || length <= 0 || device < 0)
	{
		return 1;
	}
	std::snprintf(name, static_cast<std::size_t>(length), "CMT fake GPU");
	return 0;
}

// One device, as `cmt run` and `cmt tune` find it before they start (runner.cpp, DetectThisMachine).
FAKE_API int cuDeviceGetCount(int* count)
{
	if (!count)
	{
		return 1;
	}
	*count = 1;
	return 0;
}

FAKE_API int cuDeviceGet(int* device, int ordinal)
{
	if (!device || ordinal != 0)
	{
		return 101; // CUDA_ERROR_INVALID_DEVICE
	}
	*device = 0;
	return 0;
}

#ifndef FAKE_OMIT_DEVICE_ATTRIBUTE
FAKE_API int cuDeviceGetAttribute(int* value, int attr, int device)
{
	if (attr == 113)
	{
		g_ro_queries++;
		g_ro_queried_device = device;
		if (g_ro_query_error)
		{
			return g_ro_query_error;
		}
		*value = g_ro_supported;
	}
	else if (attr == 88)
	{
		const char* v = std::getenv("FAKE_PAGEABLE_ACCESS");
		*value = v ? std::atoi(v) : 1;
	}
	else if (attr == 16)
	{
		*value = 108; // A100 SM count
	}
	else
	{
		*value = 0;
	}
	return 0;
}
#endif

// Test-only: counters.
FAKE_API const Stats* fake_stats() { return &g_stats; }

FAKE_API void fake_dma_log_reset()
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_dma_log.clear();
}

FAKE_API long fake_dma_log_count()
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	return static_cast<long>(std::size(g_dma_log));
}

// Record i: its stream (stream 0 and the per-thread/legacy handles all read as
// the default stream, reported by *is_default), pinned host pointer, size and
// direction. Returns 0, or 1 when i is out of range.
FAKE_API int fake_dma_log_at(long i, void** stream, int* is_default, const void** host, std::size_t* n,
							 int* h2d)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (i < 0 || i >= static_cast<long>(std::size(g_dma_log)))
	{
		return 1;
	}
	const DmaRecord& r = g_dma_log[static_cast<std::size_t>(i)];
	*stream = r.stream;
	*is_default = r.stream == &g_default_stream ? 1 : 0;
	*host = r.host;
	*n = r.n;
	*h2d = r.h2d ? 1 : 0;
	return 0;
}

FAKE_API void fake_fail_unregister(int count)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_unregister_failures = count;
}

FAKE_API long fake_registered_ranges()
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	return static_cast<long>(std::size(g_pinned));
}

FAKE_API long fake_registered_overlap(const void* p, std::size_t n)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	const auto a = reinterpret_cast<uintptr_t>(p);
	long count = 0;
	for (const auto& range : g_pinned)
	{
		count += range.first < a + n && a < range.first + range.second;
	}
	return count;
}

FAKE_API long fake_register_attempts() { return g_register_attempts; }

FAKE_API long fake_unregister_attempts() { return g_unregister_attempts; }

FAKE_API void fake_ro_config(int device, int supported, int query_error, int ctx_error, int register_error,
							 int fail_at)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_ro_device = device;
	g_ro_supported = supported;
	g_ro_query_error = query_error;
	g_ro_ctx_error = ctx_error;
	g_ro_register_error = register_error;
	g_ro_fail_at = fail_at;
	g_ro_seen = 0;
	g_rw_register_error = 0;
}

FAKE_API void fake_rw_register_error(int rc)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_rw_register_error = rc;
	g_rw_fail_at = 1;
	g_rw_seen = 0;
}

// From now on, the fail_at-th writable (flags 1) registration attempt and every later one return rc; 0 clears.
FAKE_API void fake_rw_register_error_at(int rc, long fail_at)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_rw_register_error = rc;
	g_rw_fail_at = fail_at;
	g_rw_seen = 0;
}

// cuDeviceTotalMem_v2 reports `bytes` for `device`, or fails with `error` (nonzero) for every device.
FAKE_API void fake_device_memory(int device, std::size_t bytes, int error)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (device >= 0 && device < kFakeDevices)
	{
		g_device_memory[device] = bytes;
	}
	g_device_memory_error = error;
}

FAKE_API long fake_device_memory_queries() { return g_device_memory_queries; }

FAKE_API long fake_flag_attempts(unsigned flags) { return flags < 16 ? g_flag_attempts[flags] : 0; }

FAKE_API long fake_ro_queries() { return g_ro_queries; }

FAKE_API int fake_ro_queried_device() { return g_ro_queried_device; }

// Host allocations (cuMemHostAlloc / cuMemFreeHost): bytes allocated and freed so far, attempts (failed ones
// too).
FAKE_API long fake_host_alloc_bytes()
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	return g_host_alloc_bytes;
}

FAKE_API long fake_host_free_bytes()
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	return g_host_free_bytes;
}

FAKE_API long fake_host_alloc_attempts()
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	return g_host_alloc_attempts;
}

// cuMemHostAlloc attempts made with exactly `flags` (0 = cached, 4 = CU_MEMHOSTALLOC_WRITECOMBINED).
FAKE_API long fake_host_alloc_flag_attempts(unsigned flags)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	return flags < 16 ? g_host_alloc_flag_attempts[flags] : 0;
}

// The next `count` cuMemHostRegister calls, whatever their flags, fail with `rc`; -1 = all of them, 0 = none.
FAKE_API void fake_fail_register(int rc, int count)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_register_failure_rc = rc;
	g_register_failures = count;
}

namespace {
// Caller holds g_lock.
void drop_registrations_locked()
{
	for (const auto& reg : g_reg_flags)
	{
		g_pinned.erase(reg.first);
		g_reg_epoch.erase(reg.first);
		g_readonly.erase(reg.first);
	}
	g_reg_flags.clear();
}
} // namespace

// Every registration is forgotten, as when the context that made it is destroyed: a later cuMemHostUnregister is
// 713.
FAKE_API void fake_drop_registrations()
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	drop_registrations_locked();
}

FAKE_API void fake_teardown_drops_registrations(int on) { g_teardown_drops_registrations.store(on != 0); }

FAKE_API void fake_set_sticky(int rc) { g_sticky.store(rc); }

FAKE_API void fake_fail_record_on(void* stream) { g_fail_record_stream.store(stream); }

FAKE_API void fake_fail_ctx_set_off_main(int on) { g_fail_ctx_set_off_main.store(on != 0); }

FAKE_API void fake_set_global_capture(int on) { g_global_capture.store(on != 0); }

FAKE_API long fake_capture_invalidations() { return g_capture_invalidations.load(); }

FAKE_API int cuThreadExchangeStreamCaptureMode(int* mode)
{
	if (mode == nullptr || *mode < 0 || *mode > 2)
	{
		return 1;
	}
	std::swap(*mode, t_capture_mode);
	return 0;
}

// cuMemHostGetDevicePointer_v2 answers host address + `offset` (0: the host address, as under unified addressing).
FAKE_API void fake_device_pointer_offset(long offset)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_device_pointer_offset = offset;
}

// The next `count` cuMemHostAlloc calls fail with CUDA_ERROR_OUT_OF_MEMORY; -1 = all of them, 0 = none.
FAKE_API void fake_fail_host_alloc(int count)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_host_alloc_failures = count;
}

// Uses of a stream, event or host allocation after its primary context was torn down.
FAKE_API long fake_stale_uses() { return g_stale_uses.load(); }

// Callbacks at the start of cuMemHostAlloc, cuMemHostRegister and a pinned-source cuMemcpyHtoDAsync (nullptr:
// none).
FAKE_API void fake_on_host_alloc(void (*fn)()) { g_on_host_alloc.store(fn); }

FAKE_API void fake_on_register(void (*fn)()) { g_on_register.store(fn); }

FAKE_API void fake_on_pinned_htod(void (*fn)()) { g_on_pinned_htod.store(fn); }

// Review additions (see the top of the file).
// Capture: `stream` captures from now on (nullptr: no capture); legacy_implicit: the legacy stream then
// reports CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, as when the capturing stream is a blocking one.
FAKE_API void fake_set_capture(void* stream, int legacy_implicit)
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_capture_stream = stream;
	g_capture_legacy_implicit = legacy_implicit;
}

FAKE_API long fake_captured_ops() { return g_captured_ops; }

FAKE_API long fake_capture_rejects() { return g_capture_rejects; }

FAKE_API long fake_ptds_calls() { return g_ptds_calls; }

// The next cuEventRecord that follows a pinned DMA fails once with 999.
FAKE_API void fake_fail_record_after_dma()
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	g_record_fail = 1;
}

// Explicit cleanup failure controls; disabled unless the test arms them.
FAKE_API void fake_fail_stream_sync(int count)
{
	std::lock_guard<std::recursive_mutex> guard(g_lock);
	g_stream_sync_failures = count;
}

FAKE_API long fake_stream_sync_attempts()
{
	std::lock_guard<std::recursive_mutex> guard(g_lock);
	return g_stream_sync_attempts;
}

FAKE_API void* fake_first_registered_base()
{
	std::lock_guard<std::recursive_mutex> guard(g_lock);
	return g_pinned.empty() ? nullptr : reinterpret_cast<void*>(g_pinned.begin()->first);
}

// ------------------------------------------------------------------
// Registration-cache redesign additions (see the top of the file)
// ------------------------------------------------------------------

FAKE_API int cuLaunchHostFunc(void* stream, void (*fn)(void*), void* data)
{
	NOT_ON_FAKE_THREAD();
	REQUIRE_CTX();
	std::lock_guard<std::recursive_mutex> g(g_lock);
	if (stale_stream(stream))
	{
		return 400;
	}
	Op op{Op::Host, nullptr, nullptr, 0, nullptr, 0};
	op.host_fn = fn;
	op.host_arg = data;
	stream_of(stream)->ops.push_back(op);
	return 0;
}

namespace {
struct StreamCallback
{
	void (*cb)(void*, int, void*);
	void* stream;
	void* data;
};

void run_stream_callback(void* p)
{
	StreamCallback* c = static_cast<StreamCallback*>(p);
	c->cb(c->stream, 0, c->data);
	delete c;
}
} // namespace

FAKE_API int cuStreamAddCallback(void* stream, void (*cb)(void*, int, void*), void* data, unsigned)
{
	return cuLaunchHostFunc(stream, &run_stream_callback, new StreamCallback{cb, stream, data});
}

namespace {
// The driver thread: runs one posted job at a time while holding g_lock.
std::mutex& g_drv_mu = *new std::mutex; // leaked, as the callback thread's
std::condition_variable& g_drv_cv = *new std::condition_variable;
void (*g_drv_fn)(void*) = nullptr;
void* g_drv_arg = nullptr;
int g_drv_state = 0; // 0 idle, 1 posted, 2 holding g_lock and running, 3 done
bool g_drv_started = false;

void* driver_main(void*)
{
	t_fake_internal = true;
	std::unique_lock<std::mutex> lk(g_drv_mu);
	for (;;)
	{
		g_drv_cv.wait(lk, [] { return g_drv_state == 1; });
		void (*fn)(void*) = g_drv_fn;
		void* arg = g_drv_arg;
		lk.unlock();
		{
			std::lock_guard<std::recursive_mutex> g(g_lock); // a driver-internal lock
			{
				std::lock_guard<std::mutex> l2(g_drv_mu);
				g_drv_state = 2;
				g_drv_cv.notify_all();
			}
			fn(arg);
		}
		lk.lock();
		g_drv_state = 3;
		g_drv_cv.notify_all();
	}
}
} // namespace

// Runs fn(arg) on the fake's driver thread with the fake's lock held; returns once that thread holds the
// lock.
FAKE_API void fake_driver_thread_post(void (*fn)(void*), void* arg)
{
	std::unique_lock<std::mutex> lk(g_drv_mu);
	if (!g_drv_started)
	{
		pthread_t t;
		if (pthread_create(&t, nullptr, &driver_main, nullptr) != 0)
		{
			std::abort();
		}
		pthread_detach(t);
		g_drv_started = true;
	}
	g_drv_fn = fn;
	g_drv_arg = arg;
	g_drv_state = 1;
	g_drv_cv.notify_all();
	g_drv_cv.wait(lk, [] { return g_drv_state >= 2; });
}

// Waits up to `ms` for the posted job to finish: 0 done, 1 timed out.
FAKE_API int fake_driver_thread_wait(int ms)
{
	std::unique_lock<std::mutex> lk(g_drv_mu);
	const bool done = g_drv_cv.wait_for(lk, std::chrono::milliseconds(ms), [] { return g_drv_state == 3; });
	if (done)
	{
		g_drv_state = 0;
	}
	return done ? 0 : 1;
}

FAKE_API long fake_internal_api_calls() { return g_internal_api_calls.load(); }

// The physical pages behind [p, p + n) were replaced. Takes no fake lock but its own (callable from any
// free()).
FAKE_API void fake_pages_changed(const void* p, std::size_t n)
{
	std::lock_guard<std::mutex> g(g_change_mu);
	const uintptr_t a = reinterpret_cast<uintptr_t>(p);
	g_changes[g_change_count++ % kChanges] = PageChange{a, a + n, ++g_epoch};
}

// The same, for a change whose epoch was taken (fake_epoch_now) just before the syscall that made it: a
// registration made after that moment (of the new pages, by another thread) is not stale.
FAKE_API unsigned long long fake_epoch_now() { return ++g_epoch; }

FAKE_API void fake_pages_changed_at(const void* p, std::size_t n, unsigned long long epoch)
{
	std::lock_guard<std::mutex> g(g_change_mu);
	const uintptr_t a = reinterpret_cast<uintptr_t>(p);
	g_changes[g_change_count++ % kChanges] = PageChange{a, a + n, epoch};
}

FAKE_API long fake_stale_dmas() { return g_stale_dmas.load(); }

// on: from now on only the calling thread's cuMemHostUnregister runs; every other thread's waits (outside the
// fake's locks) until off. Makes "the reaper unregisters later" observable: a release that relies on it
// leaves a registration over replaced pages for as long as the test wants.
FAKE_API void fake_hold_unregister(int on)
{
	g_hold_tid.store(on != 0 ? static_cast<long>(syscall(SYS_gettid)) : 0);
}

// Live cuMemHostRegister ranges over pages that changed after they were registered: a DMA through one would
// use the old pages. 0 at every moment is the cache's invariant (except inside the window
// copying/registration_cache.h documents).
FAKE_API long fake_stale_registrations()
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	std::lock_guard<std::mutex> c(g_change_mu);
	long n = 0;
	for (const auto& r : g_reg_epoch)
	{
		const uintptr_t a = r.first, b = a + g_pinned[a];
		for (std::size_t i = 0; i < std::min(g_change_count, kChanges); i++)
		{
			const PageChange& ch = g_changes[i];
			if (ch.epoch > r.second && ch.s < b && a < ch.e)
			{
				n++;
				break;
			}
		}
	}
	return n;
}

// Live cuMemHostRegister ranges (cuMemHostAlloc memory not included).
FAKE_API long fake_registrations()
{
	std::lock_guard<std::recursive_mutex> g(g_lock);
	return static_cast<long>(std::size(g_reg_epoch));
}

// Runtime stand-ins call the driver through its exported entry. One runtime
// request must produce one trace record even when this nests a driver wrapper.
FAKE_API int cudaMemcpy(void* dst, const void* src, std::size_t n, int kind)
{
	if (kind == 1)
	{
		return cuMemcpyHtoD_v2(reinterpret_cast<uintptr_t>(dst), src, n);
	}
	if (kind == 2)
	{
		return cuMemcpyDtoH_v2(dst, reinterpret_cast<uintptr_t>(src), n);
	}
	return 1;
}

FAKE_API int cudaMemcpyAsync(void* dst, const void* src, std::size_t n, int kind, void* stream)
{
	if (kind == 1)
	{
		return cuMemcpyHtoDAsync_v2(reinterpret_cast<uintptr_t>(dst), src, n, stream);
	}
	if (kind == 2)
	{
		return cuMemcpyDtoHAsync_v2(dst, reinterpret_cast<uintptr_t>(src), n, stream);
	}
	return 1;
}

FAKE_API int cudaHostAlloc(void** p, std::size_t n, unsigned flags) { return cuMemHostAlloc(p, n, flags); }

FAKE_API int cudaMallocHost(void** p, std::size_t n) { return cuMemHostAlloc(p, n, 0); }

FAKE_API int cudaFreeHost(void* p) { return cuMemFreeHost(p); }

FAKE_API int cudaHostRegister(void* p, std::size_t n, unsigned flags)
{
	return cuMemHostRegister_v2(p, n, flags);
}

FAKE_API int cudaHostUnregister(void* p) { return cuMemHostUnregister(p); }
#endif
