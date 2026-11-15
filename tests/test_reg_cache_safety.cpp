// Progressive registration lifetime through the preloaded hook and fake CUDA.
// Reuse/munmap holds an actual registration during pending DMA. Stress checks
// byte preservation and complete retirement across concurrent copies/releases. Foreign and hole check the
// typed-copy end checks against registrations (hook.cpp, TypedEnds).
// Every case skips unless the real immutable kernel eligibility passes.
#include "cuda/kernel_status.h"
#include <sys/wait.h>
#include <dlfcn.h>
#include <malloc.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "common/memory.h"
#include "common/cpu.h"

namespace {

using Dptr = unsigned long long;
void* const kCtx = reinterpret_cast<void*>(0x1000); // the fake's only context
void* g_lib = nullptr;

int (*p_htod)(Dptr, const void*, std::size_t);
int (*p_dtoh)(void*, Dptr, std::size_t);
int (*p_htod_async)(Dptr, const void*, std::size_t, void*);
int (*p_unseen_htod_async)(Dptr, const void*, std::size_t, void*);
int (*p_ctx_set)(void*);
int (*p_ctx_sync)();
int (*p_stream_create)(void**, unsigned);
int (*p_stream_sync)(void*);
long (*p_registered_overlap)(const void*, std::size_t);
long (*p_registrations)();
long (*p_register_attempts)();
long (*p_stale_dmas)();
long (*p_internal_api_calls)();
unsigned long long (*p_epoch_now)();
void (*p_pages_changed_at)(const void*, std::size_t, unsigned long long);

// munmap / madvise(DONTNEED) that tell the fake which pages changed, stamped before the syscall
// (fake_epoch_now).
int unmap_reported(void* p, std::size_t n)
{
	const unsigned long long e = p_epoch_now();
	const int rc = munmap(p, n);
	p_pages_changed_at(p, n, e);
	return rc;
}

int dontneed_reported(void* p, std::size_t n)
{
	const unsigned long long e = p_epoch_now();
	const int rc = madvise(p, n, MADV_DONTNEED);
	p_pages_changed_at(p, n, e);
	return rc;
}

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

Dptr dev(const void* p)
{
	return static_cast<Dptr>(reinterpret_cast<uintptr_t>(p));
}

void fill(uint8_t* p, std::size_t n, unsigned seed)
{
	for (std::size_t i = 0; i < n; i++)
	{
		p[i] = check::pattern(i, seed);
	}
}

// First differing byte, or n.
std::size_t diff(const uint8_t* a, const uint8_t* b, std::size_t n)
{
	for (std::size_t i = 0; i < n; i++)
	{
		if (a[i] != b[i])
		{
			return i;
		}
	}
	return n;
}

// An anonymous mapping of `bytes`, 2 MiB aligned (the alignment slop stays mapped), filled with a pattern.
uint8_t* mapped(std::size_t bytes, unsigned seed)
{
	const std::size_t align = std::size_t{2} << 20;
	void* m = mmap(nullptr, bytes + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (m == MAP_FAILED)
	{
		std::abort();
	}
	auto* p = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(m) + align - 1) & ~(align - 1));
	fill(p, bytes, seed);
	return p;
}

bool wait_until(const std::function<bool()>& cond, int seconds)
{
	const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
	while (!cond())
	{
		if (std::chrono::steady_clock::now() > end)
		{
			return false;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return true;
}

bool registered(const void* p, std::size_t n)
{
	return p_registered_overlap(p, n) > 0;
}

// A registered source must remain mapped until the queued DMA has completed.
// This runs inside the existing reuse/munmap case, with real interposed munmap.
void CheckPendingDmaUnmap()
{
	auto hold_sync = reinterpret_cast<void (*)(int)>(dlsym(g_lib, "fake_hold_stream_sync"));
	auto waiting_sync = reinterpret_cast<unsigned (*)()>(dlsym(g_lib, "fake_waiting_stream_sync"));
	CHECK(hold_sync && waiting_sync);
	if (!hold_sync || !waiting_sync)
	{
		return;
	}
	constexpr std::size_t bytes = 8u << 20;
	uint8_t* host = mapped(bytes, 27);
	check::Buffer device(bytes);
	void* stream = nullptr;
	CHECK(p_stream_create(&stream, 1) == 0);
	std::atomic<int> copy_status{-1};
	std::atomic<int> unmap_status{-1};
	std::atomic<bool> unmap_started{false};
	std::atomic<bool> unmap_finished{false};
	hold_sync(1);
	std::thread copying(
		[&]
		{
			p_ctx_set(kCtx);
			copy_status.store(p_htod_async(dev(std::data(device)), host, bytes, stream));
		});
	const bool waiting = wait_until([&] { return waiting_sync() != 0; }, 5);
	CHECKF(waiting, "copy never reached completion gate");
	CHECKF(registered(host, bytes), "copy reached completion without registering its source");
	if (!waiting || !registered(host, bytes))
	{
		hold_sync(0);
		copying.join();
		CHECK(p_stream_sync(stream) == 0);
		unmap_reported(host, bytes);
		return;
	}
	std::thread unmapping(
		[&]
		{
			p_ctx_set(kCtx);
			unmap_started.store(true);
			unmap_status.store(unmap_reported(host, bytes));
			unmap_finished.store(true);
		});
	CHECK(wait_until([&] { return unmap_started.load(); }, 5));
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	CHECKF(!unmap_finished.load(), "munmap returned while a registered DMA was pending");
	hold_sync(0);
	copying.join();
	unmapping.join();
	CHECK(copy_status.load() == 0);
	CHECK(unmap_status.load() == 0);
	CHECK(!registered(host, bytes));
	for (std::size_t index = 0; index < bytes; ++index)
	{
		if (device[index] != check::pattern(index, 27))
		{
			CHECKF(false, "pending-unmap DMA corrupted byte %zu", index);
			break;
		}
	}
}

// A sticky error (the context is dead: every later synchronization, copy and unregistration fails with 700 and
// runs no work) appears while two progressively pinned copies wait for their DMA: one from an mmap'd buffer, one
// from a heap block. Each returns 700, as the driver would, and the process keeps running. Their registrations
// cannot be proven idle, so they are kept (leaked): the fake still has them. Later copies in the dead context
// return the driver's 700 without registering; munmap of the mapped buffer returns at once; free() of the heap
// block keeps it allocated (malloc cannot hand it out again); cuCtxDestroy completes. The destroy gives a clean
// context (fake_libcuda.cpp, cuCtxDestroy_v2), whose copies pin again and release everything they pin.
// Mutation: restore the Panic in WaitForDma -> SIGABRT; make Settle wait for a LEAKED slot -> munmap hangs (alarm);
// let Drain or OnFree release a block over a LEAKED slot -> malloc returns the old block.
void CheckStickyError()
{
	auto hold_sync = reinterpret_cast<void (*)(int)>(dlsym(g_lib, "fake_hold_stream_sync"));
	auto waiting_sync = reinterpret_cast<unsigned (*)()>(dlsym(g_lib, "fake_waiting_stream_sync"));
	auto set_sticky = reinterpret_cast<void (*)(int)>(dlsym(g_lib, "fake_set_sticky"));
	auto teardown_drops = reinterpret_cast<void (*)(int)>(dlsym(g_lib, "fake_teardown_drops_registrations"));
	auto ctx_destroy = reinterpret_cast<int (*)(void*)>(dlsym(g_lib, "cuCtxDestroy_v2"));
	CHECK(hold_sync && waiting_sync && set_sticky && teardown_drops && ctx_destroy);
	if (!hold_sync || !waiting_sync || !set_sticky || !teardown_drops || !ctx_destroy)
	{
		return;
	}
	teardown_drops(1); // as the real driver: a destroyed context's registrations go with it
	p_ctx_set(kCtx);
	constexpr std::size_t bytes = 8u << 20;
	uint8_t* mapped_host = mapped(bytes, 41);
	auto* heap_host = static_cast<uint8_t*>(std::malloc(bytes));
	CHECK(heap_host != nullptr);
	fill(heap_host, bytes, 43);
	check::Buffer device_a(bytes);
	check::Buffer device_b(bytes);
	void* stream_a = nullptr;
	void* stream_b = nullptr;
	CHECK(p_stream_create(&stream_a, 1) == 0 && p_stream_create(&stream_b, 1) == 0);
	std::atomic<int> status_a{-1};
	std::atomic<int> status_b{-1};
	hold_sync(1);
	std::thread a(
		[&]
		{
			p_ctx_set(kCtx);
			status_a.store(p_htod_async(dev(std::data(device_a)), mapped_host, bytes, stream_a));
		});
	std::thread b(
		[&]
		{
			p_ctx_set(kCtx);
			status_b.store(p_htod_async(dev(std::data(device_b)), heap_host, bytes, stream_b));
		});
	const bool gated = wait_until([&] { return waiting_sync() == 2; }, 5);
	CHECKF(gated && registered(mapped_host, bytes) && registered(heap_host, bytes),
		   "the copies never held their registrations at the completion gate");
	set_sticky(700); // the context dies while both DMAs are unconfirmed
	hold_sync(0);
	a.join();
	b.join();
	CHECKF(status_a.load() == 700 && status_b.load() == 700,
		   "copies whose DMA wait met the sticky error returned %d / %d, 700 expected", status_a.load(),
		   status_b.load());
	CHECKF(registered(mapped_host, bytes) && registered(heap_host, bytes),
		   "a registration whose DMA was not confirmed was released");
	// The dead context: the driver's error, and no new registration.
	const long attempts = p_register_attempts();
	check::Buffer other(bytes);
	const int rc1 = p_htod(dev(std::data(device_a)), mapped_host, bytes);
	const int rc2 = p_dtoh(heap_host, dev(std::data(device_b)), bytes);
	const int rc3 = p_htod_async(dev(std::data(device_a)), std::data(other), bytes, stream_a);
	CHECKF(rc1 == 700 && rc2 == 700 && rc3 == 700, "copies in the dead context returned %d / %d / %d, 700 expected",
		   rc1, rc2, rc3);
	CHECKF(p_register_attempts() == attempts, "%ld registrations attempted in the dead context",
		   p_register_attempts() - attempts);
	CHECK(p_ctx_sync() == 700);
	// Releases over the kept registrations: neither waits for them nor hands the heap block back.
	CHECK(unmap_reported(mapped_host, bytes) == 0);
	std::free(heap_host);
	void* again = std::malloc(bytes);
	CHECKF(again != heap_host, "malloc returned a freed block that a kept registration still covers");
	CHECKF(registered(heap_host, bytes), "the kept registration of the freed heap block is gone");
	std::free(again);
	CHECKF(ctx_destroy(kCtx) == 0, "cuCtxDestroy after a sticky error failed");
	// The clean context pins again and releases what it pins.
	p_ctx_set(kCtx);
	uint8_t* fresh = mapped(bytes, 47);
	const long attempts_after = p_register_attempts();
	CHECK(p_htod(dev(std::data(device_a)), fresh, bytes) == 0 && p_ctx_sync() == 0);
	CHECK(diff(std::data(device_a), fresh, bytes) == bytes);
	CHECKF(p_register_attempts() > attempts_after, "the new context's copy was not pinned");
	CHECKF(!registered(fresh, bytes), "the new context's copy kept a registration");
	CHECK(unmap_reported(fresh, bytes) == 0);
	CHECKF(p_stale_dmas() == 0, "%ld DMAs through a registration whose pages changed", p_stale_dmas());
	teardown_drops(0);
}

// Device memory of a fake kind (fake_libcuda.cpp, DeviceKind): 2 managed, 8 mapped / hole / mapped.
Dptr device_of_kind(std::size_t bytes, int kind)
{
	auto alloc = reinterpret_cast<int (*)(Dptr*, std::size_t, int)>(dlsym(g_lib, "fake_device_alloc"));
	Dptr d = 0;
	CHECK(alloc != nullptr && alloc(&d, bytes, kind) == 0);
	return d;
}

void free_device(Dptr d)
{
	auto free_fn = reinterpret_cast<int (*)(Dptr)>(dlsym(g_lib, "cuMemFree_v2"));
	CHECK(free_fn != nullptr && free_fn(d) == 0);
}

// A typed copy whose device end the hook does not stage (managed memory: TypedEnds says Foreign) over host
// pages another copy holds registered. It must still take TryCopy's obligations: borrow the live pins and
// wait for its DMA before returning, so the registration cannot retire under it. Copy A holds its
// registration at the completion gate (fake_hold_stream_sync); copy B, to managed memory from the same pages,
// must not return while A is held. Mutation: Foreign straight to the driver -> B returns at once, its DMA
// still queued.
void CheckForeignTakesObligations()
{
	auto hold_sync = reinterpret_cast<void (*)(int)>(dlsym(g_lib, "fake_hold_stream_sync"));
	auto waiting_sync = reinterpret_cast<unsigned (*)()>(dlsym(g_lib, "fake_waiting_stream_sync"));
	CHECK(hold_sync && waiting_sync);
	if (!hold_sync || !waiting_sync)
	{
		return;
	}
	constexpr std::size_t bytes = 8u << 20;
	uint8_t* host = mapped(bytes, 31);
	check::Buffer device(bytes);
	const Dptr managed = device_of_kind(bytes, 2);
	void* stream_a = nullptr;
	void* stream_b = nullptr;
	CHECK(p_stream_create(&stream_a, 1) == 0 && p_stream_create(&stream_b, 1) == 0);
	std::atomic<int> status_a{-1};
	std::atomic<int> status_b{-1};
	std::atomic<bool> b_returned{false};
	hold_sync(1);
	std::thread a(
		[&]
		{
			p_ctx_set(kCtx);
			status_a.store(p_htod_async(dev(std::data(device)), host, bytes, stream_a));
		});
	CHECKF(wait_until([&] { return waiting_sync() != 0 && registered(host, bytes); }, 5),
		   "copy A never held a registration at its completion gate");
	std::thread b(
		[&]
		{
			p_ctx_set(kCtx);
			status_b.store(p_htod_async(managed, host, bytes, stream_b));
			b_returned.store(true);
		});
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	CHECKF(!b_returned.load(), "a copy to managed memory over live registered pages returned without waiting "
							   "for its DMA (it skipped the registration's obligations)");
	hold_sync(0);
	a.join();
	b.join();
	CHECK(p_stream_sync(stream_b) == 0);
	CHECK(status_a.load() == 0 && status_b.load() == 0);
	CHECK(!registered(host, bytes));
	CHECK(diff(std::data(device), host, bytes) == bytes);
	CHECK(diff(reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(managed)), host, bytes) == bytes);
	free_device(managed);
	unmap_reported(host, bytes);
}

// A typed copy into a VMM hole with progressive pinning on: TypedEnds says Refused and the driver refuses it
// (rc 1) before pinning registers a page or a DMA reaches the hole (on the real driver, a staged chunk into
// the hole kills the process). Mutation: the result of TypedEnds ignored -> pinning registers and copies.
void CheckHoleNotPinned()
{
	auto hole_copies = reinterpret_cast<long (*)()>(dlsym(g_lib, "fake_hole_copies"));
	CHECK(hole_copies != nullptr);
	constexpr std::size_t bytes = 8u << 20;
	uint8_t* host = mapped(bytes, 33);
	const Dptr hole = device_of_kind(bytes, 8);
	auto* mapped_part = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(hole));
	std::memset(mapped_part, 0xCD, bytes / 4);
	const long attempts = p_register_attempts();
	const long refused = hole_copies();
	for (const bool async : {false, true})
	{
		void* stream = nullptr;
		CHECK(p_stream_create(&stream, 1) == 0);
		const int rc = async ? p_htod_async(hole, host, bytes, stream) : p_htod(hole, host, bytes);
		CHECK(p_stream_sync(stream) == 0);
		CHECKF(rc == 1, "a %s copy into a hole returned %d, the driver's 1 expected",
			   async ? "async" : "sync", rc);
	}
	CHECKF(p_register_attempts() == attempts, "%ld registrations attempted for a copy into a hole",
		   p_register_attempts() - attempts);
	CHECKF(hole_copies() - refused == 2, "%ld copies touched the hole, 2 (the driver's refusals) expected",
		   hole_copies() - refused);
	bool untouched = true;
	for (std::size_t i = 0; i < bytes / 4 && untouched; i++)
	{
		untouched = mapped_part[i] == 0xCD;
	}
	CHECKF(untouched, "a refused copy wrote the hole's mapped part");
	free_device(hole);
	unmap_reported(host, bytes);
}

std::atomic<bool> g_stop{false};
std::atomic<long> g_bad{0};
std::atomic<long> g_ops{0};

void bad(const char* what, std::size_t at)
{
	if (g_bad.fetch_add(1) < 10)
	{
		std::fprintf(stderr, "stress: %s (byte %zu)\n", what, at);
	}
}

void copy_worker(unsigned id)
{
	p_ctx_set(kCtx);
	std::mt19937 rng(1000 + id);
	void* s = nullptr;
	p_stream_create(&s, 1);
	const std::size_t max = std::size_t{6} << 20;
	check::Buffer device(max);
	unsigned seed = id * 100000;
	while (!g_stop.load(std::memory_order_relaxed))
	{
		const std::size_t n = ((1 + rng() % 5) << 20) + rng() % 4096;
		const unsigned kind = rng() % 4; // 0 malloc, 1 mmap + munmap, 2 malloc + realloc, 3 mmap + madvise
		uint8_t* buf;
		uint8_t* map = nullptr;
		std::size_t map_n = 0;
		if (kind == 1 || kind == 3)
		{
			map_n = (n + 4095) & ~std::size_t{4095};
			map = static_cast<uint8_t*>(
				mmap(nullptr, map_n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
			buf = map;
		}
		else
		{
			buf = static_cast<uint8_t*>(std::malloc(n));
		}
		fill(buf, n, ++seed);
		for (unsigned round = 0; round < 2u; round++)
		{ // Every copy releases its registrations before the next one.
			// A synchronous pageable HtoD may return before its last DMA (CUDA's semantics, and the ring's):
			// sync.
			if (p_htod(dev(std::data(device)), buf, n) != 0 || p_ctx_sync() != 0)
			{
				bad("htod failed", 0);
			}
			else if (std::size_t at = diff(std::data(device), buf, n); at != n)
			{
				bad("htod bytes", at);
			}
		}
		// Pageable contract of an async copy: the source may be overwritten as soon as the call returns.
		fill(buf, n, ++seed);
		std::vector<uint8_t> expect(buf, buf + n);
		if (p_htod_async(dev(std::data(device)), buf, n, s) != 0)
		{
			bad("htod async failed", 0);
		}
		std::memset(buf, 0xEE, n);
		p_stream_sync(s);
		if (std::size_t at = diff(std::data(device), std::data(expect), n); at != n)
		{
			bad("async htod bytes", at);
		}
		auto* back = static_cast<uint8_t*>(std::malloc(n));
		if (p_dtoh(back, dev(std::data(device)), n) != 0)
		{
			bad("dtoh failed", 0);
		}
		else if (std::size_t at = diff(back, std::data(expect), n); at != n)
		{
			bad("dtoh bytes", at);
		}
		std::free(back);
		if (kind == 0)
		{
			std::free(buf);
		}
		else if (kind == 1)
		{
			unmap_reported(map, map_n);
		}
		else if (kind == 2)
		{
			fill(buf, n, ++seed);
			std::vector<uint8_t> keep(buf, buf + n / 2);
			buf = static_cast<uint8_t*>(std::realloc(buf, n / 2));
			if (std::size_t at = diff(buf, std::data(keep), n / 2); at != n / 2)
			{
				bad("realloc lost bytes", at);
			}
			const int rc = p_htod(dev(std::data(device)), buf, n / 2) | p_ctx_sync();
			if (rc != 0)
			{
				bad("htod after realloc failed", static_cast<std::size_t>(rc));
			}
			else if (std::size_t at = diff(std::data(device), buf, n / 2); at != n / 2)
			{
				bad("htod after realloc: bytes", at);
			}
			if ((seed & 1) != 0)
			{
				// glibc: realloc(p, 0) frees p and returns NULL, also for a block pinning still holds.
				void* gone = std::realloc(buf, 0);
				if (gone != nullptr)
				{
					bad("realloc(p, 0) returned a block", 0);
					std::free(gone);
				}
			}
			else
			{
				std::free(buf);
			}
		}
		else
		{
			dontneed_reported(map, map_n);
			fill(map, n, ++seed);
			if (p_htod(dev(std::data(device)), map, n) != 0 || p_ctx_sync() != 0 ||
				diff(std::data(device), map, n) != n)
			{
				bad("htod after madvise", 0);
			}
			unmap_reported(map, map_n);
		}
		g_ops.fetch_add(1, std::memory_order_relaxed);
	}
}

void unseen_worker()
{
	p_ctx_set(kCtx);
	std::mt19937 rng(99);
	check::Buffer device(std::size_t{5} << 20);
	unsigned seed = 900000;
	while (!g_stop.load(std::memory_order_relaxed))
	{
		const std::size_t n = ((1 + rng() % 4) << 20) + rng() % 4096;
		const bool map = rng() % 2 == 0;
		const std::size_t mn = (n + 4095) & ~std::size_t{4095};
		auto* p = map ? static_cast<uint8_t*>(
							mmap(nullptr, mn, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0))
					  : static_cast<uint8_t*>(std::malloc(n));
		fill(p, n, ++seed);
		if (p_unseen_htod_async(dev(std::data(device)), p, n, nullptr) != 0 || p_ctx_sync() != 0)
		{
			bad("unseen copy failed", 0);
		}
		else if (std::size_t at = diff(std::data(device), p, n); at != n)
		{
			bad("unseen copy bytes (a stale registration?)", at);
		}
		if (map)
		{
			unmap_reported(p, mn);
		}
		else
		{
			std::free(p);
		}
		g_ops.fetch_add(1, std::memory_order_relaxed);
	}
}

int stress()
{
	const char* sv = std::getenv("STRESS_SECONDS");
	const int seconds = sv != nullptr && std::atoi(sv) > 0 ? std::atoi(sv) : 3;
	alarm(static_cast<unsigned>(seconds + 40));
	auto env_or = [](const char* name, int dflt)
	{
		const char* v = std::getenv(name);
		return v != nullptr ? std::atoi(v) : dflt;
	};
	const long attempts_before = p_register_attempts();
	std::vector<std::thread> threads;
	const unsigned copy_threads = static_cast<unsigned>(std::max(1, env_or("STRESS_COPY_THREADS", 4)));
	for (unsigned i = 0; i < copy_threads; i++)
	{
		threads.emplace_back(copy_worker, i);
	}
	if (env_or("STRESS_UNSEEN", 1) != 0)
	{
		threads.emplace_back(unseen_worker);
	}
	std::this_thread::sleep_for(std::chrono::seconds(seconds));
	g_stop.store(true);
	for (std::thread& t : threads)
	{
		t.join();
	}
	std::printf("stress: %ld iterations in %d s\n", g_ops.load(), seconds);
	CHECKF(p_register_attempts() > attempts_before, "stress never exercised progressive registration");
	CHECKF(g_bad.load() == 0, "%ld wrong results", g_bad.load());
	CHECKF(p_stale_dmas() == 0, "%ld DMAs through a registration whose pages changed", p_stale_dmas());
	CHECKF(p_internal_api_calls() == 0, "%ld driver calls on fake threads", p_internal_api_calls());
	CHECKF(g_ops.load() > 20, "only %ld iterations", g_ops.load());
	// Every buffer is gone: every registration must go too (the reaper releases the freed ones).
	CHECKF(wait_until([] { return p_registrations() == 0; }, 10),
		   "%ld registrations left after every buffer was freed", p_registrations());
	alarm(0);
	return check::exit_code();
}

} // namespace

int main(int argc, char** argv)
{
	if (!cmt::ProgressivePinningAvailable())
	{
		std::puts("SKIP: progressive lifetime checks require verified patched-kernel provenance");
		return 77;
	}
	g_lib = dlopen("libcuda.so.1", RTLD_NOW);
	if (g_lib == nullptr)
	{
		std::fprintf(stderr, "dlopen libcuda.so.1 failed: %s\n", dlerror());
		return 2;
	}
	load(&p_htod, "cuMemcpyHtoD_v2");
	load(&p_dtoh, "cuMemcpyDtoH_v2");
	load(&p_htod_async, "cuMemcpyHtoDAsync_v2");
	load(&p_unseen_htod_async, "cuMemcpyHtoDAsync_v2_ptsz");
	load(&p_ctx_set, "cuCtxSetCurrent");
	load(&p_ctx_sync, "cuCtxSynchronize");
	load(&p_stream_create, "cuStreamCreate");
	load(&p_stream_sync, "cuStreamSynchronize");
	load(&p_registered_overlap, "fake_registered_overlap");
	load(&p_registrations, "fake_registrations");
	load(&p_register_attempts, "fake_register_attempts");
	load(&p_stale_dmas, "fake_stale_dmas");
	load(&p_internal_api_calls, "fake_internal_api_calls");
	load(&p_epoch_now, "fake_epoch_now");
	load(&p_pages_changed_at, "fake_pages_changed_at");
	const std::string mode = argc > 1 ? argv[1] : "";
	const std::string arg = argc > 2 ? argv[2] : "";
	if (mode == "reuse" && arg == "munmap")
	{
		alarm(45);
		CheckPendingDmaUnmap();
		alarm(0);
		return check::exit_code();
	}
	if (mode == "stress")
	{
		return stress();
	}
	if (mode == "foreign")
	{
		alarm(45);
		CheckForeignTakesObligations();
		alarm(0);
		return check::exit_code();
	}
	if (mode == "sticky")
	{
		alarm(45);
		CheckStickyError();
		alarm(0);
		return check::exit_code();
	}
	if (mode == "hole")
	{
		alarm(45);
		CheckHoleNotPinned();
		alarm(0);
		return check::exit_code();
	}
	std::fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
	return 2;
}
