// test_ring_lifecycle.cpp - how a context's ring is built, kept, lost and torn down, through the preloaded
// hook and fake_libcuda. Each mode is one ctest (tests/CMakeLists.txt sets the environment); every mode
// that could hang runs under alarm(), so a deadlock fails the test instead of stalling ctest.
//
// Modes cover staging allocation and cleanup, context resets, stream capture/order,
// fork-before-use, and loader interception. Progressive-registration checks skip
// when verified patched-kernel provenance is unavailable.
#include "cuda/kernel_status.h"
#include <dlfcn.h>
#include <link.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <climits>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "common/memory.h"
#include "common/cpu.h"

// install_race: the hook's install calls realpath() through the PLT; this executable is linked
// with ENABLE_EXPORTS, so this definition wins for the whole process. When armed it holds the installing
// thread inside the install until the racing thread has done its lookups and its copy (or 2 s passed), which
// is the race window: the handle known to the hook, g_api not filled yet.
std::atomic<int> g_gate{0}; // 0 off, 1 armed, 2 the installer is inside
std::atomic<bool> g_racer_done{false};

extern "C" __attribute__((visibility("default"))) char* realpath(const char* path, char* resolved)
{
	int armed = 1;
	if (g_gate.compare_exchange_strong(armed, 2))
	{
		const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
		while (!g_racer_done.load() && std::chrono::steady_clock::now() < end)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	char* r = canonicalize_file_name(path);
	if (r == nullptr || resolved == nullptr)
	{
		return r;
	}
	std::snprintf(resolved, PATH_MAX, "%s", r);
	std::free(r);
	return resolved;
}

// With _FORTIFY_SOURCE (the distribution default at -O2) the hook calls the checked variant instead.
extern "C" __attribute__((visibility("default"))) char* __realpath_chk(const char* path, char* resolved,
																	   std::size_t)
{
	return realpath(path, resolved);
}

namespace {

using Dptr = unsigned long long;
int (*p_htod)(Dptr, const void*, std::size_t);
int (*p_dtoh)(void*, Dptr, std::size_t);
int (*p_htod_async)(Dptr, const void*, std::size_t, void*);
int (*p_ctx_sync)();
int (*p_ctx_set)(void*);
int (*p_ctx_destroy)(void*);
int (*p_get_proc)(const char*, void**, int, unsigned long long);
long (*p_alloc_attempts)();
void (*p_fail_alloc)(int);
long (*p_stale_uses)();
long (*p_registered_overlap)(const void*, std::size_t);
long (*p_log_count)();
void (*p_log_reset)();

void* const kCtx = reinterpret_cast<void*>(0x1000); // the fake's only context
void* g_lib = nullptr;

template <class Fn>
void load(Fn* slot, const char* name)
{
	void* p = dlsym(g_lib, name);
	if (p == nullptr)
	{
		std::fprintf(stderr, "missing %s\n", name);
		std::exit(2);
	}
	std::memcpy(slot, &p, sizeof(p));
}

Dptr dev(void* p)
{
	return static_cast<Dptr>(reinterpret_cast<uintptr_t>(p));
}

// One HtoD and one DtoH of `bytes` through the hook, data checked; returns false on a failed call or bad
// data.
bool roundtrip(std::size_t bytes, unsigned seed)
{
	check::Buffer host(bytes), device(bytes, 0xEE), back(bytes, 0x11);
	for (std::size_t i = 0; i < bytes; i++)
	{
		host[i] = check::pattern(i, seed);
	}
	if (p_htod(dev(std::data(device)), std::data(host), bytes) != 0 || p_ctx_sync() != 0)
	{
		return false;
	}
	if (p_dtoh(std::data(back), dev(std::data(device)), bytes) != 0)
	{
		return false;
	}
	return std::memcmp(std::data(back), std::data(host), bytes) == 0;
}

// Vary participation, force parking, race callers, then tear down and rebuild.
// Checks actual bytes through the preload hook; a lost wakeup hits the alarm.
int handoff_stress(bool shared_cpu)
{
	alarm(45);
	cpu_set_t saved;
	if (shared_cpu)
	{
		const int got = sched_getaffinity(0, sizeof(saved), &saved);
		CHECK(got == 0);
		if (got != 0)
		{
			return check::exit_code();
		}
		cpu_set_t one;
		CPU_ZERO(&one);
		for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu)
		{
			if (CPU_ISSET(cpu, &saved))
			{
				CPU_SET(cpu, &one);
				break;
			}
		}
		const int pinned = sched_setaffinity(0, sizeof(one), &one);
		CHECK(pinned == 0);
		if (pinned != 0)
		{
			return check::exit_code();
		}
	}
	std::atomic<unsigned> failures{0};
	for (unsigned epoch = 0; epoch < 8; ++epoch)
	{
		std::vector<std::thread> callers;
		p_log_reset();
		for (unsigned caller = 0; caller < 3; ++caller)
		{
			callers.emplace_back(
				[&, caller]
				{
					if (p_ctx_set(kCtx) != 0)
					{
						failures.fetch_add(1);
						return;
					}
					const std::size_t sizes[] = {4096, 65537, 196625};
					for (unsigned i = 0; i < 64; ++i)
					{
						if (!roundtrip(sizes[i % 3], epoch * 192 + caller * 64 + i))
						{
							failures.fetch_add(1);
						}
						if (i % 4 == 0)
						{
							std::this_thread::sleep_for(std::chrono::microseconds(200));
						}
					}
				});
		}
		for (auto& caller : callers)
		{
			caller.join();
		}
		CHECK(p_log_count() > 0); // the test must reach the staging path
		CHECK(p_ctx_destroy(kCtx) == 0);
	}
	CHECK(failures.load() == 0);
	CHECK(p_stale_uses() == 0);
	if (shared_cpu)
	{
		CHECK(sched_setaffinity(0, sizeof(saved), &saved) == 0);
	}
	alarm(0);
	return check::exit_code();
}

// An anonymous mapping of `bytes`, 2 MiB aligned, filled with a pattern.
uint8_t* mapped(std::size_t bytes, unsigned seed)
{
	const std::size_t align = std::size_t{2} << 20;
	void* m = mmap(nullptr, bytes + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (m == MAP_FAILED)
	{
		std::abort();
	}
	auto* p = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(m) + align - 1) & ~(align - 1));
	for (std::size_t i = 0; i < bytes; i++)
	{
		p[i] = check::pattern(i, seed);
	}
	return p;
}

std::string read_file(const std::string& path)
{
	std::ifstream in(path);
	std::stringstream ss;
	ss << in.rdbuf();
	return ss.str();
}

int count_lines_with(const std::string& text, const char* needle)
{
	int n = 0;
	std::istringstream in(text);
	for (std::string line; std::getline(in, line);)
	{
		n += line.find(needle) != std::string::npos;
	}
	return n;
}

// ------------------------------------------------------------------
// alloc_fail
// ------------------------------------------------------------------

int alloc_fail()
{
	// The hook's error line goes to stderr: capture it in a file of the test's working directory.
	const std::string path = "ring_lifecycle_alloc_fail." + std::to_string(getpid()) + ".stderr";
	std::fflush(stderr);
	const int saved = dup(2);
	std::FILE* capture = std::fopen(path.c_str(), "w");
	CHECK(capture != nullptr && saved >= 0);
	dup2(fileno(capture), 2);

	p_fail_alloc(-1);
	const long before = p_alloc_attempts();
	bool ok = true;
	for (unsigned i = 0; i < 20; i++)
	{
		ok = roundtrip((256u << 10) + 7 * i, i) && ok;
	}
	const long first_context = p_alloc_attempts() - before;
	// A new context (same handle, as after a destroy) tries once more, and only once.
	CHECK(p_ctx_destroy(kCtx) == 0);
	for (unsigned i = 0; i < 10; i++)
	{
		ok = roundtrip(300000 + i, 40 + i) && ok;
	}
	const long second_context = p_alloc_attempts() - before - first_context;
	// Memory back, same context: still the driver (remembered). Then a fresh context builds its ring.
	p_fail_alloc(0);
	ok = roundtrip(256u << 10, 70) && ok;
	const long after_clear = p_alloc_attempts() - before - first_context - second_context;
	CHECK(p_ctx_destroy(kCtx) == 0);
	p_log_reset();
	ok = roundtrip(256u << 10, 71) && ok;
	const long rebuilt = p_alloc_attempts() - before - first_context - second_context - after_clear;
	const long pipeline_dmas = p_log_count();

	std::fflush(stderr);
	dup2(saved, 2);
	close(saved);
	std::fclose(capture);
	const std::string err = read_file(path);
	std::remove(path.c_str());

	CHECKF(ok, "a copy failed or returned wrong data while the ring could not be built");
	CHECKF(first_context == 1, "first context: %ld cuMemHostAlloc attempts for 40 copies, 1 expected",
		   first_context);
	CHECKF(second_context == 1, "second context: %ld attempts, 1 expected", second_context);
	CHECKF(after_clear == 0, "a context whose ring failed tried again (%ld attempts)", after_clear);
	CHECKF(rebuilt == 1 && pipeline_dmas > 0,
		   "a new context after the failure: %ld attempts, %ld pipeline DMAs", rebuilt, pipeline_dmas);
	const int lines = count_lines_with(err, "[cmt]");
	CHECKF(lines == 1 && count_lines_with(err, "staging setup failed") == 1,
		   "%d hook lines on stderr, 1 expected:\n%s", lines, err.c_str());
	return check::exit_code();
}

// ------------------------------------------------------------------
// slot_alloc: STAGING_SLOT_ALLOC=hostalloc_wc reaches cuMemHostAlloc as the WC flag
// ------------------------------------------------------------------

int slot_alloc(const std::string& want)
{
	long (*flag_attempts)(unsigned) = nullptr;
	load(&flag_attempts, "fake_host_alloc_flag_attempts");
	constexpr unsigned kWc = 0x04; // CU_MEMHOSTALLOC_WRITECOMBINED
	const long cached_before = flag_attempts(0), wc_before = flag_attempts(kWc);
	p_log_reset();
	bool ok = true;
	for (unsigned i = 0; i < 8; i++)
	{
		ok = roundtrip((512u << 10) + 4099 * i, 90 + i) && ok;
	}
	const long cached = flag_attempts(0) - cached_before, wc = flag_attempts(kWc) - wc_before;
	CHECKF(ok, "a copy failed or returned wrong data through %s staging", want.c_str());
	CHECKF(p_log_count() > 0, "no staging DMA: the test did not reach the pipeline");
	if (want == "wc")
	{
		CHECKF(wc > 0 && cached == 0, "WC requested: %ld WC and %ld cached staging allocations", wc, cached);
	}
	else
	{
		CHECKF(cached > 0 && wc == 0, "cached requested: %ld cached and %ld WC staging allocations", cached,
			   wc);
	}
	CHECK(p_ctx_destroy(kCtx) == 0);
	CHECK(p_stale_uses() == 0);
	return check::exit_code();
}

// ------------------------------------------------------------------
// primary_reset
// ------------------------------------------------------------------

int primary_reset()
{
	int (*retain)(void**, int) = nullptr;
	int (*reset_v2)(int) = nullptr;
	int (*release_v2)(int) = nullptr;
	int (*release_proc)(int) =
		nullptr; // "cuDevicePrimaryCtxRelease" through cuGetProcAddress (the _v2 function)
	load(&retain, "cuDevicePrimaryCtxRetain");
	load(&reset_v2, "cuDevicePrimaryCtxReset_v2");
	load(&release_v2, "cuDevicePrimaryCtxRelease_v2");
	void* p = nullptr;
	CHECK(p_get_proc("cuDevicePrimaryCtxRelease", &p, 12000, 0) == 0 && p != nullptr);
	std::memcpy(&release_proc, &p, sizeof(p));

	void* ctx = nullptr;
	CHECK(retain(&ctx, 0) == 0 && ctx == kCtx);
	CHECK(roundtrip(200000, 1)); // builds the ring: streams, events, pinned slots of this context
	CHECK(reset_v2(0) == 0);	 // the fake invalidates all of them
	CHECK(retain(&ctx, 0) == 0 && ctx == kCtx); // the same handle comes back, as from the driver
	CHECKF(roundtrip(200000, 2), "copy after cuDevicePrimaryCtxReset_v2 failed (stale ring?)");
	CHECKF(p_stale_uses() == 0, "%ld uses of a torn-down context's handles after a reset", p_stale_uses());

	// A release that does not end the context: the hook cannot know, frees the ring anyway; the copy rebuilds
	// it.
	CHECK(retain(&ctx, 0) == 0); // count 2
	CHECK(roundtrip(200000, 3));
	CHECK(release_proc(0) == 0); // count 1: context alive
	CHECKF(roundtrip(200000, 4), "copy after a release that kept the context failed");
	CHECK(release_v2(0) == 0); // count 0: the fake invalidates
	CHECK(retain(&ctx, 0) == 0);
	CHECKF(roundtrip(200000, 5), "copy after the last cuDevicePrimaryCtxRelease_v2 failed (stale ring?)");
	CHECKF(p_stale_uses() == 0, "%ld uses of a torn-down context's handles after a release", p_stale_uses());

	return check::exit_code();
}

// ------------------------------------------------------------------
// Interposition and teardown
// ------------------------------------------------------------------

template <class Fn>
Fn sym(const char* name)
{
	Fn fn = nullptr;
	void* p = dlsym(g_lib, name);
	if (p == nullptr)
	{
		p = dlsym(RTLD_DEFAULT, name);
	}
	if (p == nullptr)
	{
		std::fprintf(stderr, "missing %s\n", name);
		std::exit(2);
	}
	std::memcpy(&fn, &p, sizeof(p));
	return fn;
}

std::string object_of(const void* p);

// The fake's counters (fake_libcuda.cpp, Stats; same field order).
struct FakeStats
{
	long htod_async_pageable_src, htod_async_pinned_src, dtoh_async_pageable_dst, dtoh_async_pinned_dst;
	long htod_sync, dtoh_sync, stream_create, stream_wait_event, HostAlloc, host_free, ctx_destroy;
	long host_register, host_unregister;
};

// Redirects stderr to a file while `fn` runs; returns what was written.
template <class F>
std::string capture_stderr(const char* tag, F fn)
{
	const std::string path =
		std::string("ring_lifecycle_") + tag + "." + std::to_string(getpid()) + ".stderr";
	std::fflush(stderr);
	const int saved = dup(2);
	std::FILE* capture = std::fopen(path.c_str(), "w");
	if (capture == nullptr || saved < 0)
	{
		std::abort();
	}
	dup2(fileno(capture), 2);
	fn();
	std::fflush(stderr);
	dup2(saved, 2);
	close(saved);
	std::fclose(capture);
	const std::string err = read_file(path);
	std::remove(path.c_str());
	return err;
}

// A: free()/realloc() through the hook reach the allocator preloaded after it. Without the fix the hook calls
// glibc's __libc_free on fake_alloc's pointers and glibc aborts (already during startup).
int foreign_alloc()
{
	auto frees = sym<long (*)()>("fake_alloc_frees");
	const long before = frees();
	void* volatile sink = nullptr;
	for (unsigned i = 0; i < 100u; i++)
	{
		sink = std::malloc(1000 + static_cast<std::size_t>(i));
		std::free(sink);
	}
	sink = std::malloc(10);
	sink = std::realloc(sink, 100000); // the old block is freed inside fake_alloc's realloc
	std::free(sink);
	const long after = frees();
	CHECKF(after - before >= 102, "%ld frees reached the allocator behind the hook, 102 expected",
		   after - before);
	CHECK(roundtrip(256u << 10, 1));
	return check::exit_code();
}

// B: stream capture. Mutation: return false from capturing() -> the hook stages the copy (DMAs logged, result
// 0).
int capture()
{
	auto stream_create = sym<int (*)(void**, unsigned)>("cuStreamCreate");
	auto set_capture = sym<void (*)(void*, int)>("fake_set_capture");
	auto rejects = sym<long (*)()>("fake_capture_rejects");
	auto captured_ops = sym<long (*)()>("fake_captured_ops");
	auto dtoh_async = sym<int (*)(void*, Dptr, std::size_t, void*)>("cuMemcpyDtoHAsync_v2");
	CHECK(roundtrip(256u << 10, 1)); // the ring exists
	void* s = nullptr;
	CHECK(stream_create(&s, 1) == 0);
	const std::size_t n = std::size_t{4} << 20;
	// A fresh pageable range must reach the driver's capture check without staging or registration.
	uint8_t* host = mapped(n, 2);
	check::Buffer device(n);
	p_log_reset();
	const long rejects0 = rejects();
	const long captured0 = captured_ops();
	set_capture(s, 0);
	const int rc1 = p_htod_async(dev(std::data(device)), host, n, s);
	const int rc2 = dtoh_async(host, dev(std::data(device)), n, s);
	set_capture(s, 1); // the legacy stream now reports CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
	const int rc3 = p_htod(dev(std::data(device)), host, n);
	const int rc4 = p_dtoh(host, dev(std::data(device)), n);
	set_capture(nullptr, 0);
	CHECKF(rc1 == 0 && rc2 == 0,
		   "copies on a capturing stream returned %d / %d, the driver captures them (0)", rc1, rc2);
	CHECKF(rc3 == 900 && rc4 == 900, "sync copies during an implicit capture returned %d / %d, 900 expected",
		   rc3, rc4);
	CHECKF(p_log_count() == 0 && captured_ops() - captured0 == 2,
		   "%ld staging DMAs and %ld copies the driver captured during capture (0 and 2 expected)",
		   p_log_count(), captured_ops() - captured0);
	CHECKF(rejects() - rejects0 == 2, "the driver refused %ld of the 2 implicit-capture copies",
		   rejects() - rejects0);
	CHECK(roundtrip(256u << 10, 3)); // the hook still works afterwards
	CHECKF(p_log_count() > 0, "no pipeline DMA after the capture ended");
	return check::exit_code();
}

// C: per-thread default stream. Mutation: ignore per_thread_stream in ring::RedirectLookup -> the hook's
// legacy bodies come back for a PTDS query.
int ptds()
{
	auto get_proc_v2 =
		sym<int (*)(const char*, void**, int, unsigned long long, void*)>("cuGetProcAddress_v2");
	auto ptds_calls = sym<long (*)()>("fake_ptds_calls");
	const unsigned long long kPtds = 2; // CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
	void* p = nullptr;
	CHECK(p_get_proc("cuMemcpyHtoD", &p, 12000, kPtds) == 0);
	CHECKF(p == dlsym(g_lib, "cuMemcpyHtoD_v2_ptds"), "PTDS cuMemcpyHtoD gave %p in %s", p,
		   object_of(p).c_str());
	void* q = nullptr;
	int status = -1;
	CHECK(get_proc_v2("cuMemcpyDtoHAsync", &q, 12000, kPtds, &status) == 0 && status == 0);
	CHECKF(q == dlsym(g_lib, "cuMemcpyDtoHAsync_v2_ptsz"), "PTDS cuMemcpyDtoHAsync (v2) gave %p in %s", q,
		   object_of(q).c_str());
	void* legacy = nullptr;
	CHECK(p_get_proc("cuMemcpyHtoD", &legacy, 12000, 0) == 0);
	CHECKF(object_of(legacy).find("cmt_hook") != std::string::npos, "legacy cuMemcpyHtoD gave %s",
		   object_of(legacy).c_str());
	void* destroy = nullptr;
	CHECK(p_get_proc("cuCtxDestroy", &destroy, 12000, kPtds) == 0);
	CHECKF(object_of(destroy).find("cmt_hook") != std::string::npos, "PTDS cuCtxDestroy lost its teardown");
	// The per-thread function the application got copies (through the driver: no pipeline DMA).
	int (*htod_ptds)(Dptr, const void*, std::size_t) = nullptr;
	std::memcpy(&htod_ptds, &p, sizeof(p));
	const std::size_t n = 256u << 10;
	check::Buffer host(n), device(n);
	for (std::size_t i = 0; i < n; i++)
	{
		host[i] = check::pattern(i, 4);
	}
	p_log_reset();
	const long calls0 = ptds_calls();
	CHECK(htod_ptds != nullptr && htod_ptds(dev(std::data(device)), std::data(host), n) == 0 &&
		  p_ctx_sync() == 0);
	CHECK(std::memcmp(std::data(device), std::data(host), n) == 0);
	CHECK(ptds_calls() - calls0 == 1 && p_log_count() == 0);
	return check::exit_code();
}

// Registration failure must preserve the caller's non-blocking stream order.
// Its queued pinned HtoD must run before the pageable DtoH fallback reads the device.
int stream_order()
{
	if (!cmt::ProgressivePinningAvailable())
	{
		std::puts("SKIP: progressive pinning requires verified patched-kernel provenance");
		return 77;
	}
	auto stream_create = sym<int (*)(void**, unsigned)>("cuStreamCreate");
	auto HostAlloc = sym<int (*)(void**, std::size_t, unsigned)>("cuMemHostAlloc");
	auto dtoh_async = sym<int (*)(void*, Dptr, std::size_t, void*)>("cuMemcpyDtoHAsync_v2");
	auto stats = sym<const FakeStats* (*)()>("fake_stats");
	auto rw_register_error = sym<void (*)(int)>("fake_rw_register_error");
	auto register_attempts = sym<long (*)()>("fake_register_attempts");
	void* s = nullptr;
	CHECK(stream_create(&s, 1) == 0); // CU_STREAM_NON_BLOCKING
	const std::size_t n = std::size_t{8} << 20;
	void* pinned = nullptr;
	CHECK(HostAlloc(&pinned, n, 0) == 0);
	auto* src = static_cast<uint8_t*>(pinned);
	for (std::size_t i = 0; i < n; i++)
	{
		src[i] = check::pattern(i, 9);
	}
	check::Buffer device(n, 0);
	uint8_t* dst = mapped(n, 0);
	std::memset(dst, 0, n);
	rw_register_error(2);
	const long registrations0 = register_attempts();
	const long fallback0 = stats()->dtoh_async_pageable_dst;
	CHECK(p_htod_async(dev(std::data(device)), src, n, s) == 0); // pinned: the driver, queued on S
	CHECK(dtoh_async(dst, dev(std::data(device)), n, s) == 0); // failed registration, then driver fallback
	rw_register_error(0);
	std::size_t first_bad = n;
	for (std::size_t i = 0; i < n; i++)
	{
		if (dst[i] != src[i])
		{
			first_bad = i;
			break;
		}
	}
	CHECKF(first_bad == n, "byte %zu of the DtoH was read before S's earlier HtoD ran", first_bad);
	CHECKF(register_attempts() > registrations0, "progressive registration was not attempted");
	CHECKF(stats()->dtoh_async_pageable_dst > fallback0, "no copy took the pageable driver fallback");
	return check::exit_code();
}

// Context teardown must remain safe when a thread-local allocation would fail.
// The next copy rebuilds staging without using the destroyed context's handles.
int teardown_oom(const std::string& variant)
{
	auto fail_next = sym<void (*)(int)>("fake_alloc_fail_next");
	int (*retain)(void**, int) = nullptr;
	int (*reset_v2)(int) = nullptr;
	load(&retain, "cuDevicePrimaryCtxRetain");
	load(&reset_v2, "cuDevicePrimaryCtxReset_v2");
	void* ctx = nullptr;
	CHECK(retain(&ctx, 0) == 0);
	CHECK(roundtrip(200000, 1));
	int rc = -1;
	fail_next(1);
	if (variant == "destroy")
	{
		rc = p_ctx_destroy(kCtx);
	}
	else
	{
		rc = reset_v2(0);
	}
	fail_next(0);
	CHECKF(rc == 0, "%s returned %d", variant.c_str(), rc);
	CHECK(retain(&ctx, 0) == 0);
	CHECKF(roundtrip(200000, 3), "copy after an out-of-memory %s failed (stale ring?)", variant.c_str());
	CHECKF(p_stale_uses() == 0, "%ld uses of a torn-down context's handles", p_stale_uses());
	return check::exit_code();
}

// G: the install race. With MEMCPY_HOOK_NO_EAGER=1, main's first explicit CUDA lookup installs; realpath()
// holds it. Mutation: publish g_libcuda_handle before resolve_driver_api again -> the racer's copy calls a
// null g_api entry (SIGSEGV) and its release lookup gets the driver's function.
void* g_racer_release = nullptr;
bool g_racer_copy_ok = false;

void racer()
{
	const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (g_gate.load() != 2 && std::chrono::steady_clock::now() < end)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
	if (h != nullptr && g_gate.load() == 2)
	{
		g_racer_release = dlsym(h, "cuDevicePrimaryCtxRelease_v2");
		void* htod = dlsym(h, "cuMemcpyHtoD_v2");
		void* setc = dlsym(h, "cuCtxSetCurrent");
		int (*htod_fn)(Dptr, const void*, std::size_t) = nullptr;
		int (*setc_fn)(void*) = nullptr;
		std::memcpy(&htod_fn, &htod, sizeof(htod));
		std::memcpy(&setc_fn, &setc, sizeof(setc));
		const std::size_t n = 256u << 10;
		check::Buffer host(n), device(n);
		for (std::size_t i = 0; i < n; i++)
		{
			host[i] = check::pattern(i, 5);
		}
		g_racer_copy_ok = setc_fn != nullptr && setc_fn(kCtx) == 0 && htod_fn != nullptr &&
						  htod_fn(dev(std::data(device)), std::data(host), n) == 0 &&
						  std::memcmp(std::data(device), std::data(host), n) == 0 && g_gate.load() == 2;
	}
	g_racer_done.store(true);
}

int install_race_open()
{
	std::thread t(racer);
	g_gate.store(1);
	g_lib = dlopen("libcuda.so.1", RTLD_NOW);
	if (g_lib != nullptr)
	{
		(void)dlsym(g_lib, "cuInit"); // installs; realpath() holds it until the racer is done
	}
	t.join();
	return g_lib != nullptr ? 0 : 2;
}

int install_race()
{
	CHECKF(g_racer_done.load() && g_gate.load() == 2, "the race was not set up");
	CHECKF(g_racer_copy_ok, "the copy made through a lookup during the install failed");
	void* now = dlsym(g_lib, "cuDevicePrimaryCtxRelease_v2");
	CHECKF(object_of(g_racer_release).find("cmt_hook") != std::string::npos && g_racer_release == now,
		   "a release looked up during the install is %p in %s, after it %p", g_racer_release,
		   object_of(g_racer_release).c_str(), now);
	CHECK(roundtrip(256u << 10, 6));
	return check::exit_code();
}

// G (second part): cuGetProcAddress returns an address that matches none of the exported primary-context
// functions. Mutation: drop the wrap_unmatched_primary fallback -> the application gets the driver's
// function, the reset leaves the ring behind, and the next copy uses its freed streams.
int unknown_primary()
{
	int (*retain)(void**, int) = nullptr;
	load(&retain, "cuDevicePrimaryCtxRetain");
	void *p = nullptr, *again = nullptr;
	const std::string err =
		capture_stderr("unknown_primary",
					   [&]
					   {
						   CHECK(p_get_proc("cuDevicePrimaryCtxReset", &p, 1, 0) == 0);
						   CHECK(p_get_proc("cuDevicePrimaryCtxReset", &again, 1, 0) == 0);
					   });
	CHECKF(object_of(p).find("cmt_hook") != std::string::npos && p == again,
		   "the unmatched reset is %p in %s (again %p)", p, object_of(p).c_str(), again);
	CHECKF(count_lines_with(err, "[cmt] error: cuGetProcAddress") == 1, "stderr:\n%s", err.c_str());
	int (*reset)(int) = nullptr;
	std::memcpy(&reset, &p, sizeof(p));
	void* ctx = nullptr;
	CHECK(retain(&ctx, 0) == 0);
	CHECK(roundtrip(200000, 1));
	CHECK(reset != nullptr && reset(0) == 0);
	CHECK(retain(&ctx, 0) == 0);
	CHECKF(roundtrip(200000, 2), "copy after the unmatched reset failed (stale ring?)");
	CHECKF(p_stale_uses() == 0, "%ld uses of a torn-down context's handles", p_stale_uses());
	return check::exit_code();
}

// H: a DtoH whose slot record fails after its DMA was queued, then an HtoD through the same (only) slot. The
// fake runs DMAs lazily: without the fix the old DtoH lands in the slot after the HtoD filled it, and the
// HtoD delivers the DtoH's bytes. Mutation: return the record error without settle_unrecorded.
int record_fail()
{
	auto fail_record = sym<void (*)()>("fake_fail_record_after_dma");
	const std::size_t n = 64u << 10; // one chunk (STAGING_BUFFER_SIZE=64): one slot
	check::Buffer dev1(n), dst1(n), dev2(n, 0), src2(n);
	for (std::size_t i = 0; i < n; i++)
	{
		dev1[i] = check::pattern(i, 10);
		src2[i] = check::pattern(i, 11);
	}
	CHECK(roundtrip(n, 1)); // the ring exists
	fail_record();
	const int rc1 = p_dtoh(std::data(dst1), dev(std::data(dev1)), n);
	CHECKF(rc1 != 0, "the DtoH whose record failed returned success");
	CHECK(p_htod(dev(std::data(dev2)), std::data(src2), n) == 0);
	CHECK(p_ctx_sync() == 0);
	CHECKF(std::memcmp(std::data(dev2), std::data(src2), n) == 0,
		   "the HtoD after a failed slot record delivered other bytes");
	CHECK(roundtrip(3 * n + 5, 2));
	return check::exit_code();
}

// Cleanup must drain queued worker DMA before releasing a registered mmap pool.
// A failed drain or unregister must not release memory a DMA may still use: the pool is retained (leaked), and
// the application's teardown call still succeeds; it never terminates the process.
int CleanupLifetime(const std::string& mode)
{
	constexpr std::size_t bytes = 64u << 10;
	CHECK(roundtrip(bytes, 20));
	auto base = sym<void* (*)()>("fake_first_registered_base")();
	CHECK(base != nullptr);
	if (!base)
	{
		return check::exit_code();
	}
	auto sync_attempts = sym<long (*)()>("fake_stream_sync_attempts");
	auto unregister_attempts = sym<long (*)()>("fake_unregister_attempts");
	const auto before_sync = sync_attempts();
	const auto before_unregister = unregister_attempts();
	check::Buffer source(bytes), destination(bytes, 0);
	for (std::size_t i = 0; i < bytes; ++i)
	{
		source[i] = check::pattern(i, 21);
	}
	CHECK(p_htod_async(dev(std::data(destination)), std::data(source), bytes, nullptr) == 0);
	if (mode == "cleanup_drain_failure")
	{
		sym<void (*)(int)>("fake_fail_stream_sync")(1);
	}
	if (mode == "cleanup_unregister_failure")
	{
		sym<void (*)(int)>("fake_fail_unregister")(1);
	}
	CHECK(p_ctx_destroy(kCtx) == 0);
	CHECK(sync_attempts() > before_sync);
	CHECK(std::memcmp(std::data(source), std::data(destination), bytes) == 0);
	unsigned char residency = 0;
	const int mapped = mincore(base, 4096, &residency);
	if (mode == "cleanup_success")
	{
		CHECK(p_registered_overlap(base, bytes) == 0);
		CHECK(mapped == -1 && errno == ENOMEM);
		CHECK(unregister_attempts() > before_unregister);
	}
	else
	{
		CHECK(p_registered_overlap(base, bytes) == 1);
		CHECK(mapped == 0);
		if (mode == "cleanup_drain_failure")
		{
			CHECK(unregister_attempts() == before_unregister);
		}
	}
	return check::exit_code();
}

// B1: a sticky error (the context is dead: every later call fails with 700 and runs no work) appears while a
// staged copy is in flight. The copy returns the error, later copies return it from the driver, the teardown
// (cuCtxDestroy) completes without terminating, and a new context stages again. Mutation: restore the Panic in
// SettleFailure, or in the pipeline destructor's drain -> SIGABRT.
std::atomic<bool> g_make_sticky{false};
void BecomeSticky()
{
	if (g_make_sticky.exchange(false))
	{
		sym<void (*)(int)>("fake_set_sticky")(700);
	}
}

int sticky()
{
	alarm(30);
	auto on_pinned_htod = sym<void (*)(void (*)())>("fake_on_pinned_htod");
	const std::size_t n = 256u << 10;
	CHECK(roundtrip(n, 1)); // the ring exists
	check::Buffer host(n), device(n), back(n);
	for (std::size_t i = 0; i < n; i++)
	{
		host[i] = check::pattern(i, 2);
	}
	g_make_sticky = true;
	on_pinned_htod(&BecomeSticky); // the first staged DMA of the next copy finds the context dead
	const int rc1 = p_htod(dev(std::data(device)), std::data(host), n);
	on_pinned_htod(nullptr);
	CHECKF(rc1 == 700, "the staged copy that met the sticky error returned %d, 700 expected", rc1);
	p_log_reset();
	const int rc2 = p_dtoh(std::data(back), dev(std::data(device)), n);
	const int rc3 = p_htod_async(dev(std::data(device)), std::data(host), n, nullptr);
	CHECKF(rc2 == 700 && rc3 == 700, "later copies returned %d / %d, the driver's 700 expected", rc2, rc3);
	CHECKF(p_log_count() == 0, "%ld staged DMAs after the pipeline could not settle", p_log_count());
	CHECK(p_ctx_sync() == 700);
	CHECKF(p_ctx_destroy(kCtx) == 0, "cuCtxDestroy after a sticky error failed");
	// The fake's destroy gives a clean context, as a new one would be: the hook builds a new ring for it.
	p_log_reset();
	CHECKF(roundtrip(n, 3), "copies in the new context failed");
	CHECKF(p_log_count() > 0, "the new context's copies were not staged");
	alarm(0);
	return check::exit_code();
}

// M1: two threads copy through one context's staging. While the first one is inside the pipeline (its staged
// DMA is being submitted), the second one's copy must not wait for it: it goes to the driver and returns. The
// first copy's lane could otherwise be waiting for stream work that only the second thread's later actions
// release. Mutation: lock copy_mutex instead of try_lock -> the second copy blocks until the first finishes
// (timeout below).
std::atomic<bool> g_second_started{false};
std::atomic<bool> g_second_done{false};
std::atomic<int> g_second_rc{-1};
std::vector<uint8_t>* g_second_src = nullptr;
std::vector<uint8_t>* g_second_dev = nullptr;

void SecondCopy()
{
	if (g_second_started.exchange(true))
	{
		return; // only the first staged DMA starts it
	}
	std::thread([]
				{
					p_ctx_set(kCtx);
					g_second_rc = p_htod(dev(std::data(*g_second_dev)), std::data(*g_second_src),
										 std::size(*g_second_src));
					g_second_done = true;
				})
		.detach();
	const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(3);
	while (!g_second_done.load() && std::chrono::steady_clock::now() < end)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}

int concurrent_copy()
{
	alarm(30);
	auto on_pinned_htod = sym<void (*)(void (*)())>("fake_on_pinned_htod");
	const std::size_t n = 256u << 10;
	CHECK(roundtrip(n, 1)); // the ring exists
	std::vector<uint8_t> src1(n), dev1(n), src2(n), dev2(n);
	for (std::size_t i = 0; i < n; i++)
	{
		src1[i] = check::pattern(i, 5);
		src2[i] = check::pattern(i, 6);
	}
	g_second_src = &src2;
	g_second_dev = &dev2;
	on_pinned_htod(&SecondCopy);
	const auto start = std::chrono::steady_clock::now();
	const int rc1 = p_htod(dev(std::data(dev1)), std::data(src1), n);
	const bool second_done_inside = g_second_done.load();
	on_pinned_htod(nullptr);
	while (!g_second_done.load() && std::chrono::steady_clock::now() - start < std::chrono::seconds(10))
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	CHECK(rc1 == 0 && p_ctx_sync() == 0);
	CHECKF(second_done_inside, "the second thread's copy waited for the first one's staging");
	CHECKF(g_second_done.load() && g_second_rc.load() == 0, "the second copy returned %d", g_second_rc.load());
	CHECK(src1 == dev1);
	CHECK(src2 == dev2);
	CHECK(roundtrip(n, 7)); // the staging is still usable
	alarm(0);
	return check::exit_code();
}

// M2: another thread captures a stream in global mode while this thread's first copy builds the context's staging
// (pinned pool allocation, a "potentially unsafe" call) and then stages (event and stream waits, which the real
// driver also refuses then). Both run in relaxed mode: the copy is staged, no capture is invalidated, and this
// thread's own mode is restored. Mutation: drop the RelaxedCaptureScope in ResourceForCurrentContext -> the pool
// allocation fails with 900 and the copy goes to the driver; drop the one in StagingResources::Copy -> a slot or
// settle wait fails with 900.
int capture_global()
{
	alarm(30);
	auto stream_create = sym<int (*)(void**, unsigned)>("cuStreamCreate");
	auto stream_sync = sym<int (*)(void*)>("cuStreamSynchronize");
	auto set_global_capture = sym<void (*)(int)>("fake_set_global_capture");
	auto invalidations = sym<long (*)()>("fake_capture_invalidations");
	auto exchange_mode = sym<int (*)(int*)>("cuThreadExchangeStreamCaptureMode");
	void* stream = nullptr;
	CHECK(stream_create(&stream, 1) == 0);
	const std::size_t n = 256u << 10;
	std::vector<uint8_t> host(n), device(n);
	for (std::size_t i = 0; i < n; i++)
	{
		host[i] = check::pattern(i, 8);
	}
	p_log_reset();
	set_global_capture(1);
	const int rc = p_htod_async(dev(std::data(device)), std::data(host), n, stream);
	int mode = 0;
	const int exchanged = exchange_mode(&mode); // reads this thread's mode (and sets global, 0, again)
	set_global_capture(0);
	CHECK(rc == 0 && stream_sync(stream) == 0);
	CHECK(host == device);
	CHECKF(invalidations() == 0, "%ld global-mode captures invalidated by the hook's own calls", invalidations());
	CHECKF(p_log_count() > 0, "the copy was not staged (staging setup failed under the capture?)");
	CHECKF(exchanged == 0 && mode == 0, "this thread's capture mode is %d after the copy, 0 (global) expected",
		   mode);
	alarm(0);
	return check::exit_code();
}

// A pipeline helper that cannot make the context current fails the staging setup, so the context's copies use
// the driver and succeed. Mutation: ignore the handshake in ChunkPipeline::Start -> the helper's lane fails every
// copy with the context error.
int helper_ctx_fail()
{
	alarm(30);
	sym<void (*)(int)>("fake_fail_ctx_set_off_main")(1);
	p_log_reset();
	const bool ok = roundtrip(1u << 20, 1);
	sym<void (*)(int)>("fake_fail_ctx_set_off_main")(0);
	CHECKF(ok, "copies failed when the pipeline's helpers could not select the context");
	CHECKF(p_log_count() == 0, "%ld staged DMAs without a working pipeline", p_log_count());
	alarm(0);
	return check::exit_code();
}

// A caller stream the pipeline's fence event cannot be recorded on (another context's, CUDA_ERROR_INVALID_HANDLE)
// started no work: the copy goes to the driver and succeeds. Mutation: return the record error -> rc 400.
int fence_fail()
{
	alarm(30);
	auto stream_create = sym<int (*)(void**, unsigned)>("cuStreamCreate");
	auto stream_sync = sym<int (*)(void*)>("cuStreamSynchronize");
	auto fail_record_on = sym<void (*)(void*)>("fake_fail_record_on");
	CHECK(roundtrip(256u << 10, 1)); // the ring exists
	void* stream = nullptr;
	CHECK(stream_create(&stream, 1) == 0);
	const std::size_t n = 256u << 10;
	std::vector<uint8_t> host(n), device(n);
	for (std::size_t i = 0; i < n; i++)
	{
		host[i] = check::pattern(i, 9);
	}
	fail_record_on(stream);
	p_log_reset();
	const int rc = p_htod_async(dev(std::data(device)), std::data(host), n, stream);
	fail_record_on(nullptr);
	CHECKF(rc == 0, "a copy whose stream fence failed returned %d", rc);
	CHECK(stream_sync(stream) == 0);
	CHECK(host == device);
	CHECKF(p_log_count() == 0, "%ld staged DMAs for a copy whose fence failed", p_log_count());
	CHECK(roundtrip(256u << 10, 2)); // the ring still serves other streams
	alarm(0);
	return check::exit_code();
}

// The libcuda image has no cuInit (fake_cuda_no_init), so the install cannot identify it: the hook's entry points
// handed out must still reach the driver for every call (through g_api, resolved once), without staging.
int install_failed()
{
	alarm(30);
	CHECKF(object_of(reinterpret_cast<void*>(p_htod)).find("cmt_hook") != std::string::npos,
		   "cuMemcpyHtoD_v2 is not the hook's entry point");
	p_log_reset();
	CHECK(roundtrip(256u << 10, 1));
	CHECK(roundtrip(3u << 20, 2));
	CHECKF(p_log_count() == 0, "%ld staged DMAs without an install", p_log_count());
	CHECK(p_ctx_destroy(kCtx) == 0);
	alarm(0);
	return check::exit_code();
}

// I: the parent never used the hook. Mutation: bypass in on_fork_child unconditionally -> the
// child's copies go to the driver (no pipeline DMA).
int fork_unused()
{
	alarm(30);
	const pid_t pid = fork();
	if (pid == 0)
	{
		alarm(10);
		p_log_reset();
		const bool ok = roundtrip(256u << 10, 1);
		const long dmas = p_log_count();
		if (!ok || dmas == 0)
		{
			std::fprintf(stderr, "child: copies %s, %ld pipeline DMAs\n", ok ? "right" : "wrong", dmas);
		}
		std::exit(ok && dmas > 0 ? 0 : 1);
	}
	CHECKF(pid > 0, "fork failed");
	if (pid > 0)
	{
		int status = 0;
		CHECK(waitpid(pid, &status, 0) == pid);
		CHECKF(WIFEXITED(status) && WEXITSTATUS(status) == 0,
			   "the child of a parent without hook state %s %d",
			   WIFSIGNALED(status) ? "was killed by signal" : "exited with",
			   WIFSIGNALED(status) ? WTERMSIG(status) : WEXITSTATUS(status));
	}
	CHECK(roundtrip(256u << 10, 2)); // and the parent's own hook works after the fork
	alarm(0);
	return check::exit_code();
}

// ------------------------------------------------------------------
// early_dlsym
// ------------------------------------------------------------------

// The file name of the object that contains `p`.
std::string object_of(const void* p)
{
	Dl_info info{};
	return p != nullptr && dladdr(p, &info) != 0 && info.dli_fname != nullptr ? info.dli_fname : "?";
}

int early_dlsym()
{
	void* const* early = static_cast<void* const*>(dlsym(RTLD_DEFAULT, "early_dlsym_htod"));
	const int* ran = static_cast<const int*>(dlsym(RTLD_DEFAULT, "early_dlsym_ran"));
	const int* before = static_cast<const int*>(dlsym(RTLD_DEFAULT, "early_dlsym_before_hook"));
	CHECKF(early != nullptr && ran != nullptr && *ran == 1, "the early library's constructor did not run");
	// The setup itself: the early constructor must have run before the hook's initializers, or this proves
	// nothing.
	CHECKF(before != nullptr && *before == 1, "the early constructor ran after the hook's constructor");
	void* now = dlsym(g_lib, "cuMemcpyHtoD_v2");
	CHECKF(object_of(now).find("cmt_hook") != std::string::npos, "dlsym now gives %p in %s", now,
		   object_of(now).c_str());
	CHECKF(early != nullptr && *early == now,
		   "a lookup before the hook's static initializers got %p in %s; the hook's entry point is %p",
		   early != nullptr ? *early : nullptr, object_of(early != nullptr ? *early : nullptr).c_str(), now);
	return check::exit_code();
}

// Diagnostic tracing is independent of binary patches and works with this
// unversioned fake driver. This is also an end-to-end test of the CLI launcher.
int trace_sample(bool enabled)
{
	check::Buffer host(64, 42), device(64), back(64);
	auto query_count = sym<unsigned (*)()>("fake_trace_queries");
	auto reg = sym<int (*)(void*, std::size_t, unsigned)>("cuMemHostRegister_v2");
	auto unreg = sym<int (*)(void*)>("cuMemHostUnregister");
	auto runtime = sym<int (*)(void*, const void*, std::size_t, int)>("cudaMemcpy");
	auto async = sym<int (*)(void*, const void*, std::size_t, int, void*)>("cudaMemcpyAsync");
	CHECK(p_htod(dev(device.data()), host.data(), 64) == 0);
	CHECK(reg(host.data(), 64, 0) == 0);
	CHECK(p_htod(dev(device.data()), host.data(), 64) == 0);
	CHECK(runtime(device.data(), host.data(), 64, 1) == 0);
	CHECK(async(device.data(), host.data(), 64, 1, nullptr) == 0);
	CHECK(p_ctx_sync() == 0);
	setenv("FAKE_TRACE_QUERY_FAIL", "1", 1);
	CHECK(p_dtoh(back.data(), dev(device.data()), 64) == 0);
	unsetenv("FAKE_TRACE_QUERY_FAIL");
	void* entry = nullptr;
	auto get_proc = sym<int (*)(const char*, void**, int, unsigned long long)>("cuGetProcAddress");
	CHECK(get_proc("cuMemcpyHtoD", &entry, 12000, 2) == 0);
	CHECK(entry != nullptr);
	if (entry)
	{
		CHECK(reinterpret_cast<decltype(p_htod)>(entry)(dev(device.data()), host.data(), 64) == 0);
	}
	// The returned get-proc-address ABI follows the requested version.
	auto get_v2 = sym<int (*)(const char*, void**, int, unsigned long long, void*)>("cuGetProcAddress_v2");
	void* proc = nullptr;
	int query_status = -1;
	CHECK(get_v2("cuGetProcAddress", &proc, 11030, 0, &query_status) == 0 && query_status == 0);
	CHECK(proc != nullptr);
	if (proc)
	{
		CHECK(reinterpret_cast<decltype(get_proc)>(proc)("cuMemcpyHtoD", &entry, 12000, 2) == 0);
	}
	CHECK(get_proc("cuGetProcAddress", &proc, 12000, 0) == 0);
	CHECK(proc != nullptr);
	query_status = -1;
	if (proc)
	{
		CHECK(reinterpret_cast<decltype(get_v2)>(proc)("cuMemcpyDtoHAsync", &entry, 12000, 2,
													   &query_status) == 0 &&
			  query_status == 0);
	}
	CHECK(get_v2("cuMemcpyDtoHAsync", &entry, 12000, 2, &query_status) == 0 && query_status == 0);
	CHECK(entry != nullptr);
	if (entry)
	{
		CHECK(reinterpret_cast<int (*)(void*, Dptr, std::size_t, void*)>(entry)(
				  back.data(), dev(device.data()), 64, nullptr) == 0);
	}
	CHECK(p_ctx_sync() == 0);
	CHECK(runtime(device.data(), host.data(), 17, 99) == 1);
	CHECK(unreg(host.data()) == 0);
	CHECK(std::memcmp(host.data(), back.data(), 64) == 0);
	CHECK(enabled ? query_count() > 0 : query_count() == 0);
	return check::exit_code();
}

int trace_imports(const char* module)
{
	// No libcuda dlsym lookup before this call: forwarding must use the calling
	// DSO's dependency scope, not a driver handle captured by earlier tests.
	void* handle = dlopen(module, RTLD_NOW | RTLD_LOCAL);
	CHECKF(handle != nullptr, "%s", dlerror());
	if (handle == nullptr)
	{
		return check::exit_code();
	}
	auto run = reinterpret_cast<int (*)()>(dlsym(handle, "fake_import_calls"));
	CHECK(run != nullptr);
	if (run)
	{
		CHECK(run() == 0);
	}
	dlclose(handle);
	return check::exit_code();
}

int trace_conflicts(const char* alternate, const char* caller)
{
	void* alt = dlopen(alternate, RTLD_NOW | RTLD_LOCAL);
	CHECK(alt != nullptr);
	if (!alt)
	{
		return check::exit_code();
	}
	using Runtime = int (*)(void*, const void*, std::size_t, int);
	using Driver = int (*)(Dptr, const void*, std::size_t);
	auto first_runtime = reinterpret_cast<Runtime>(dlsym(alt, "cudaMemcpy"));
	auto first_driver = reinterpret_cast<Driver>(dlsym(alt, "cuMemcpyHtoD_v2"));
	CHECK(first_runtime && first_driver);
	if (!first_runtime || !first_driver)
	{
		return check::exit_code();
	}
	unsigned char src[64]{42}, dst[64]{};
	CHECK(first_runtime(dst, src, sizeof(src), 1) == 41);
	CHECK(first_driver(dev(dst), src, sizeof(src)) == 55);
	CHECK(std::memcmp(src, dst, sizeof(src)) == 0);

	// Direct imports must not use the originals captured from the other image.
	void* module = dlopen(caller, RTLD_NOW | RTLD_LOCAL);
	CHECK(module != nullptr);
	if (!module)
	{
		return check::exit_code();
	}
	auto imported = reinterpret_cast<int (*)()>(dlsym(module, "fake_import_calls"));
	CHECK(imported != nullptr);
	if (imported)
	{
		CHECK(imported() == 0);
	}
	auto other_runtime = reinterpret_cast<Runtime>(dlsym(module, "cudaMemcpy"));
	auto other_driver = reinterpret_cast<Driver>(dlsym(module, "cuMemcpyHtoD_v2"));
	CHECK(other_runtime && other_driver);
	CHECK(other_runtime != first_runtime && other_driver != first_driver);
	if (other_runtime)
	{
		CHECK(other_runtime(dst, src, sizeof(src), 1) == 0);
	}
	if (other_driver)
	{
		CHECK(other_driver(dev(dst), src, sizeof(src)) == 0);
	}
	CHECK(dlsym(alt, "cudaMemcpy") == reinterpret_cast<void*>(first_runtime));
	CHECK(dlsym(alt, "cuMemcpyHtoD_v2") == reinterpret_cast<void*>(first_driver));
	CHECK(first_runtime(dst, src, sizeof(src), 1) == 41);
	CHECK(first_driver(dev(dst), src, sizeof(src)) == 55);
	dlclose(module);
	dlclose(alt);
	return check::exit_code();
}

int trace_launch(const char* self, const char* cli, const char* hook, const char* caller,
				 const char* alternate)
{
	char directory[] = "/tmp/cmt-trace-test-XXXXXX";
	CHECK(mkdtemp(directory) != nullptr);
	const auto run = [&](std::vector<std::string> words, bool gate_off = false)
	{
		const pid_t child = fork();
		CHECK(child >= 0);
		if (child == 0)
		{
			if (gate_off)
			{
				setenv("LD_PRELOAD", hook, 1);
				unsetenv("CUDAMEMCPY_TUNER_TRACE_ENABLE");
			}
			std::vector<char*> args;
			for (auto& word : words)
			{
				args.push_back(word.data());
			}
			args.push_back(nullptr);
			execv(args.front(), args.data());
			_exit(127);
		}
		int status = 0;
		while (waitpid(child, &status, 0) < 0 && errno == EINTR)
		{
		}
		return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
	};
	const auto path = [&](const char* name) { return std::string(directory) + "/" + name; };
	const std::string output = path("copies.jsonl");
	CHECK(run({cli, "trace", "--out", output, "--hook", hook, "--", self, "trace_sample", "on"}) == 0);
	std::ifstream input(output);
	std::ostringstream buffer;
	buffer << input.rdbuf();
	const std::string records = buffer.str();
	CHECK(records.find("\"copy_calls\":8") != std::string::npos);
	CHECK(records.find("\"failed_copy_calls\":1") != std::string::npos);
	for (const char* text : {"\"src_first\":\"pageable\"", "\"src_first\":\"pinned\"",
							 "\"src_first\":\"unknown\"", "\"bytes_requested\":17", "cuMemcpyHtoD_v2_ptds",
							 "\"api\":\"cudaMemcpy\"", "\"api\":\"cudaMemcpyAsync\"", "trace_end"})
	{
		CHECKF(records.find(text) != std::string::npos, "missing %s in %s", text, records.c_str());
	}
	const std::string local_output = path("local-runtime.jsonl");
	CHECK(run({cli, "trace", "--out", local_output, "--hook", hook, "--", self, "trace_sample", "local"}) ==
		  0);
	std::ifstream local_input(local_output);
	std::ostringstream local_buffer;
	local_buffer << local_input.rdbuf();
	const std::string local_records = local_buffer.str();
	CHECK(local_records.find("\"api\":\"cudaMemcpy") != std::string::npos);
	CHECK(local_records.find("\"copy_calls\":8") != std::string::npos);
	CHECK(local_records.find("\"failed_copy_calls\":1") != std::string::npos);
	CHECK(run({cli, "trace", "--out", output, "--hook", hook, "--", "/bin/true"}) == 125); // never overwrite: cmt itself fails (125)
	CHECK(run({cli, "trace", "--out", path("empty.jsonl"), "--hook", hook, "--", "/bin/true"}) == 0);
	CHECK(run({cli, "trace", "--out", path("failed.jsonl"), "--hook", hook, "--", "/bin/sh", "-c",
			   "exit 7"}) == 7);
	CHECK(run({cli, "trace", "--out", path("signal.jsonl"), "--hook", hook, "--", "/bin/sh", "-c",
			   "kill -TERM $$"}) == 143);
	CHECK(run({self, "trace_sample", "off"}, true) == 0);
	const std::string imports_output = path("local-imports.jsonl");
	CHECK(run({cli, "trace", "--out", imports_output, "--hook", hook, "--", self, "trace_imports", caller}) ==
		  0);
	std::ifstream imports_input(imports_output);
	std::ostringstream imports_buffer;
	imports_buffer << imports_input.rdbuf();
	const std::string imports_records = imports_buffer.str();
	CHECK(imports_records.find("\"copy_calls\":4") != std::string::npos);
	CHECK(imports_records.find("\"failed_copy_calls\":0") != std::string::npos);
	for (const char* api : {"cudaMemcpy", "cudaMemcpyAsync", "cudaHostAlloc", "cudaFreeHost",
							"cudaHostRegister", "cudaHostUnregister", "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2"})
	{
		CHECKF(imports_records.find(std::string("\"api\":\"") + api + "\"") != std::string::npos,
			   "missing %s", api);
	}
	CHECK(run({self, "trace_imports", caller}, true) == 0);
	const std::string conflicts_output = path("binding-conflicts.jsonl");
	CHECK(run({cli, "trace", "--out", conflicts_output, "--hook", hook, "--", self, "trace_conflicts",
			   alternate, caller}) == 0);
	std::ifstream conflicts_input(conflicts_output);
	std::ostringstream conflicts_buffer;
	conflicts_buffer << conflicts_input.rdbuf();
	const std::string conflicts_records = conflicts_buffer.str();
	CHECK(conflicts_records.find("\"copy_calls\":10") != std::string::npos);
	CHECK(conflicts_records.find("\"failed_copy_calls\":4") != std::string::npos);
	CHECK(run({self, "trace_conflicts", alternate, caller}, true) == 0);
	if (check::failures() == 0)
	{
		for (const char* name : {"copies.jsonl", "local-runtime.jsonl", "local-imports.jsonl",
								 "binding-conflicts.jsonl", "empty.jsonl", "failed.jsonl", "signal.jsonl"})
		{
			unlink(path(name).c_str());
		}
		rmdir(directory);
	}
	else
	{
		std::fprintf(stderr, "Trace fixture retained at %s\n", directory);
	}
	return check::exit_code();
}

} // namespace

int main(int argc, char** argv)
{
	const std::string mode = argc > 1 ? argv[1] : "";
	const std::string arg = argc > 2 ? argv[2] : "";
	if (mode == "trace_launch" && argc == 6)
	{
		return trace_launch(argv[0], argv[2], argv[3], argv[4], argv[5]);
	}
	if (mode == "trace_conflicts" && argc == 4)
	{
		return trace_conflicts(argv[2], argv[3]);
	}
	if (mode == "trace_imports")
	{
		return trace_imports(arg.c_str());
	}
	if (mode == "install_race")
	{
		if (install_race_open() != 0)
		{
			std::fprintf(stderr, "dlopen libcuda.so.1 failed: %s\n", dlerror());
			return 2;
		}
	}
	else
	{
		const int visibility = mode == "trace_sample" && arg != "local" ? RTLD_GLOBAL : RTLD_LOCAL;
		g_lib = dlopen("libcuda.so.1", RTLD_NOW | visibility);
	}
	if (g_lib == nullptr)
	{
		std::fprintf(stderr, "dlopen libcuda.so.1 failed: %s\n", dlerror());
		return 2;
	}
	load(&p_htod, "cuMemcpyHtoD_v2");
	load(&p_dtoh, "cuMemcpyDtoH_v2");
	load(&p_htod_async, "cuMemcpyHtoDAsync_v2");
	load(&p_ctx_sync, "cuCtxSynchronize");
	load(&p_ctx_set, "cuCtxSetCurrent");
	load(&p_ctx_destroy, "cuCtxDestroy_v2");
	load(&p_get_proc, "cuGetProcAddress");
	load(&p_alloc_attempts, "fake_host_alloc_attempts");
	load(&p_fail_alloc, "fake_fail_host_alloc");
	load(&p_stale_uses, "fake_stale_uses");
	load(&p_registered_overlap, "fake_registered_overlap");
	load(&p_log_count, "fake_dma_log_count");
	load(&p_log_reset, "fake_dma_log_reset");

	if (mode == "cleanup_success" || mode == "cleanup_drain_failure" || mode == "cleanup_unregister_failure")
	{
		return CleanupLifetime(mode);
	}
	if (mode == "trace_sample")
	{
		return trace_sample(arg != "off");
	}
	if (mode == "alloc_fail")
	{
		return alloc_fail();
	}
	if (mode == "primary_reset")
	{
		return primary_reset();
	}
	if (mode == "slot_alloc")
	{
		return slot_alloc(arg);
	}
	if (mode == "handoff_stress")
	{
		return handoff_stress(arg == "shared_cpu");
	}
	if (mode == "early_dlsym")
	{
		return early_dlsym();
	}
	if (mode == "foreign_alloc")
	{
		return foreign_alloc();
	}
	if (mode == "capture")
	{
		return capture();
	}
	if (mode == "ptds")
	{
		return ptds();
	}
	if (mode == "stream_order" && arg == "regfail")
	{
		return stream_order();
	}
	if (mode == "teardown_oom" && (arg == "destroy" || arg == "reset"))
	{
		return teardown_oom(arg);
	}
	if (mode == "install_race")
	{
		return install_race();
	}
	if (mode == "unknown_primary")
	{
		return unknown_primary();
	}
	if (mode == "record_fail")
	{
		return record_fail();
	}
	if (mode == "fork_unused")
	{
		return fork_unused();
	}
	if (mode == "sticky")
	{
		return sticky();
	}
	if (mode == "concurrent_copy")
	{
		return concurrent_copy();
	}
	if (mode == "capture_global")
	{
		return capture_global();
	}
	if (mode == "helper_ctx_fail")
	{
		return helper_ctx_fail();
	}
	if (mode == "fence_fail")
	{
		return fence_fail();
	}
	if (mode == "install_failed")
	{
		return install_failed();
	}
	std::fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
	return 2;
}
