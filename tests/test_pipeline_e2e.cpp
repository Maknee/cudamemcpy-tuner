// test_pipeline_e2e.cpp - the preloaded hook against fake_libcuda.
//
// ctest runs this with LD_PRELOAD=libcmt_hook.so and the fake
// libcuda.so.1 first on LD_LIBRARY_PATH, under several knob settings. It
// reads the same knobs to know what to expect:
//   - dlsym/cuGetProcAddress always redirect; with STAGING_THREADS_COPY_NUM_BUFFERS unset
//     every copy still reaches the driver as the application issued it
//   - every size round-trips byte-exact (sync, async on a user stream,
//     async on stream 0), with the fake executing DMAs lazily
//   - stream order: a pass-through op on the user stream before/after a
//     pipelined copy is observed in order (fence and join events)
//   - each worker has its own stream; cuCtxDestroy frees the shared staging pool
#include "cuda/kernel_status.h"
#include <vector>
#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sched.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#include "check.h"
#include "common/memory.h"
#include "common/cpu.h"

namespace {

struct Stats
{
	long htod_async_pageable_src, htod_async_pinned_src, dtoh_async_pageable_dst, dtoh_async_pinned_dst;
	long htod_sync, dtoh_sync, stream_create, stream_wait_event, HostAlloc, host_free, ctx_destroy;
	long host_register, host_unregister;
};

using Dptr = unsigned long long;
int (*p_htod)(Dptr, const void*, std::size_t);
int (*p_dtoh)(void*, Dptr, std::size_t);
int (*p_htod_async)(Dptr, const void*, std::size_t, void*);
int (*p_dtoh_async)(void*, Dptr, std::size_t, void*);
int (*p_ctx_sync)();
int (*p_stream_create)(void**, unsigned);
int (*p_stream_sync)(void*);
int (*p_host_alloc)(void**, std::size_t, unsigned);
int (*p_ctx_destroy)(void*);
int (*p_get_proc)(const char*, void**, int, unsigned long long);
int (*p_get_proc_v2)(const char*, void**, int, unsigned long long, void*);
const Stats* (*p_stats)();
long (*p_host_free_bytes)();

int env_int(const char* name)
{
	const char* v = std::getenv(name);
	return v ? std::atoi(v) : 0;
}

template <class Fn>
void load(void* h, Fn* slot, const char* name)
{
	void* p = dlsym(h, name);
	if (p == nullptr)
	{
		std::fprintf(stderr, "missing %s\n", name);
		std::exit(2);
	}
	std::memcpy(slot, &p, sizeof(p));
}

// True when p lives in the preloaded hook rather than the fake driver.
bool from_hook(void* p)
{
	Dl_info info;
	if (dladdr(p, &info) == 0 || info.dli_fname == nullptr)
	{
		return false;
	}
	const char* base = std::strrchr(info.dli_fname, '/');
	base = base ? base + 1 : info.dli_fname;
	return std::strncmp(base, "libcmt_hook", 11) == 0;
}

Dptr dev(void* p)
{
	return static_cast<Dptr>(reinterpret_cast<uintptr_t>(p));
}

enum class Mode : std::uint8_t
{
	Sync,
	AsyncUser,
	AsyncNull
};

void roundtrip(std::size_t bytes, Mode mode, void* user_stream, unsigned seed)
{
	check::Buffer host_src(bytes);
	check::Buffer device(bytes);
	check::Buffer host_dst(bytes, 0xCD);
	for (std::size_t i = 0; i < bytes; i++)
	{
		host_src[i] = check::pattern(i, seed);
	}
	void* stream = mode == Mode::AsyncUser ? user_stream : nullptr;
	if (mode == Mode::Sync)
	{
		CHECK(p_htod(dev(std::data(device)), std::data(host_src), bytes) == 0);
	}
	else
	{
		CHECK(p_htod_async(dev(std::data(device)), std::data(host_src), bytes, stream) == 0);
	}
	// Scribble the source: a correct H2D has already captured it.
	std::memset(std::data(host_src), 0x11, bytes);
	if (mode == Mode::Sync)
	{
		CHECK(p_dtoh(std::data(host_dst), dev(std::data(device)), bytes) == 0);
	}
	else
	{
		CHECK(p_dtoh_async(std::data(host_dst), dev(std::data(device)), bytes, stream) == 0);
		p_stream_sync(stream);
	}
	bool ok = true;
	for (std::size_t i = 0; i < bytes && ok; i++)
	{
		ok = host_dst[i] == check::pattern(i, seed);
	}
	CHECKF(ok, "roundtrip bytes=%zu mode=%d", bytes, static_cast<int>(mode));
}

// Pass-through (pinned) H2D, then pipelined D2H on the same stream: the
// D2H must see the pass-through data (fence). Then pipelined H2D followed
// by pass-through D2H into pinned memory: it must see the pipelined data
// (join).
void stream_order(void* stream, std::size_t bytes)
{
	void* pinned = nullptr;
	CHECK(p_host_alloc(&pinned, bytes, 0) == 0);
	check::Buffer device(bytes);
	check::Buffer pageable(bytes);
	auto* pin = static_cast<uint8_t*>(pinned);
	for (std::size_t i = 0; i < bytes; i++)
	{
		pin[i] = check::pattern(i, 21);
	}
	CHECK(p_htod_async(dev(std::data(device)), pinned, bytes, stream) == 0);
	CHECK(p_dtoh_async(std::data(pageable), dev(std::data(device)), bytes, stream) == 0);
	p_stream_sync(stream);
	CHECKF(std::memcmp(std::data(pageable), pinned, bytes) == 0, "fence: pipelined D2H overtook a queued op");

	for (std::size_t i = 0; i < bytes; i++)
	{
		pageable[i] = check::pattern(i, 22);
	}
	CHECK(p_htod_async(dev(std::data(device)), std::data(pageable), bytes, stream) == 0);
	std::memset(pin, 0, bytes);
	CHECK(p_dtoh_async(pinned, dev(std::data(device)), bytes, stream) == 0);
	p_stream_sync(stream);
	CHECKF(std::memcmp(std::data(pageable), pinned, bytes) == 0, "join: queued op overtook pipelined H2D");

	// Pass-through H2D of A, then pipelined H2D of B to the same buffer:
	// the device must end with B (fence before the pipelined DMAs).
	for (std::size_t i = 0; i < bytes; i++)
	{
		pin[i] = check::pattern(i, 23);
		pageable[i] = check::pattern(i, 24);
	}
	CHECK(p_htod_async(dev(std::data(device)), pinned, bytes, stream) == 0);
	CHECK(p_htod_async(dev(std::data(device)), std::data(pageable), bytes, stream) == 0);
	p_ctx_sync();
	CHECKF(std::memcmp(std::data(device), std::data(pageable), bytes) == 0,
		   "fence: pipelined H2D overtook a queued H2D");
}

int count_threads(const char* prefix, bool pinned)
{
	DIR* d = opendir("/proc/self/task");
	if (d == nullptr)
	{
		return 0;
	}
	int count = 0;
	while (dirent* e = readdir(d))
	{
		if (e->d_name[0] == '.')
		{
			continue;
		}
		const std::string dir = std::string("/proc/self/task/") + e->d_name;
		std::FILE* f = std::fopen((dir + "/comm").c_str(), "r");
		char comm[64] = {0};
		if (f == nullptr)
		{
			continue;
		}
		const bool got = std::fgets(comm, sizeof(comm), f) != nullptr;
		std::fclose(f);
		if (!got || std::strncmp(comm, prefix, std::strlen(prefix)) != 0)
		{
			continue;
		}
		cpu_set_t set;
		if (pinned &&
			(sched_getaffinity(std::atoi(e->d_name), sizeof(set), &set) != 0 || CPU_COUNT(&set) != 1))
		{
			closedir(d);
			return -1;
		}
		count++;
	}
	closedir(d);
	return count;
}



} // namespace

int main(int argc, char** argv)
{
	const bool progressive_case = argc == 2 && std::strcmp(argv[1], "progressive_pinning") == 0;
	if (progressive_case && !cmt::ProgressivePinningAvailable())
	{
		std::puts("SKIP: progressive pinning requires verified patched-kernel provenance");
		return 77;
	}
	const int buffers = env_int("STAGING_THREADS_COPY_NUM_BUFFERS");
	const bool pipeline = buffers >= 1;
	const int kib = env_int("STAGING_BUFFER_SIZE") > 0 ? env_int("STAGING_BUFFER_SIZE") : 4096;
	const std::size_t chunk = static_cast<std::size_t>(kib) * 1024;
	const bool redirect = true; // Supported wrappers select their route when the copy runs.
	const int pipe_workers =
		env_int("STAGING_PIPELINE_WORKERS") > 0 ? env_int("STAGING_PIPELINE_WORKERS") : 2;


	void* h = dlopen("libcuda.so.1", RTLD_NOW);
	if (h == nullptr)
	{
		std::fprintf(stderr, "dlopen libcuda.so.1 failed: %s\n", dlerror());
		return 2;
	}
	load(h, &p_htod, "cuMemcpyHtoD_v2");
	load(h, &p_dtoh, "cuMemcpyDtoH_v2");
	load(h, &p_htod_async, "cuMemcpyHtoDAsync_v2");
	load(h, &p_dtoh_async, "cuMemcpyDtoHAsync_v2");
	load(h, &p_ctx_sync, "cuCtxSynchronize");
	load(h, &p_stream_create, "cuStreamCreate");
	load(h, &p_stream_sync, "cuStreamSynchronize");
	load(h, &p_host_alloc, "cuMemHostAlloc");
	load(h, &p_ctx_destroy, "cuCtxDestroy_v2");
	load(h, &p_get_proc, "cuGetProcAddress");
	load(h, &p_get_proc_v2, "cuGetProcAddress_v2");
	load(h, &p_stats, "fake_stats");
	load(h, &p_host_free_bytes, "fake_host_free_bytes");

	void* htod_ptr;
	std::memcpy(&htod_ptr, &p_htod, sizeof(htod_ptr));
	CHECKF(from_hook(htod_ptr) == redirect, "dlsym redirect expected=%d", redirect);
	void* gpa;
	std::memcpy(&gpa, &p_get_proc, sizeof(gpa));
	CHECKF(from_hook(gpa) == redirect, "cuGetProcAddress wrapper expected=%d", redirect);
	void* via_gpa = nullptr;
	CHECK(p_get_proc("cuMemcpyHtoD", &via_gpa, 12000, 0) == 0);
	CHECKF(from_hook(via_gpa) == redirect, "cuGetProcAddress(cuMemcpyHtoD) redirect");
	void* sync_fn = nullptr;
	CHECK(p_get_proc("cuCtxSynchronize", &sync_fn, 12000, 0) == 0);
	CHECK(sync_fn != nullptr && !from_hook(sync_fn));
	// CUDA 12 runtimes (PyTorch cu12x) resolve the driver through cuGetProcAddress_v2.
	void* gpa2;
	std::memcpy(&gpa2, &p_get_proc_v2, sizeof(gpa2));
	CHECKF(from_hook(gpa2) == redirect, "cuGetProcAddress_v2 wrapper expected=%d", redirect);
	int qstatus = -1;
	void* via_gpa2 = nullptr;
	CHECK(p_get_proc_v2("cuMemcpyHtoD", &via_gpa2, 12000, 0, &qstatus) == 0 && qstatus == 0);
	CHECKF(from_hook(via_gpa2) == redirect, "cuGetProcAddress_v2(cuMemcpyHtoD) redirect");
	if (redirect)
	{ // The driver selects the query ABI; the hook preserves that selection.
		void* self2 = nullptr;
		qstatus = -1;
		CHECK(p_get_proc_v2("cuGetProcAddress", &self2, 12000, 0, &qstatus) == 0 && qstatus == 0);
		CHECKF(from_hook(self2), "cuGetProcAddress_v2(cuGetProcAddress) returns the v2 wrapper");
	}
	void* sync2 = nullptr;
	qstatus = -1;
	CHECK(p_get_proc_v2("cuCtxSynchronize", &sync2, 12000, 0, &qstatus) == 0 && qstatus == 0);
	CHECK(sync2 != nullptr && !from_hook(sync2));

	void* user_stream = nullptr;
	CHECK(p_stream_create(&user_stream, 0) == 0);

	// Progressive pinning registers pieces and unregisters all of
	// them before it returns; nothing stays registered between copies, in either direction.
	if (progressive_case)
	{
		const std::size_t bytes = std::size_t{8} << 20;
		std::vector<uint8_t> device(bytes);
		auto* buf = static_cast<uint8_t*>(std::malloc(bytes));
		CHECK(buf != nullptr);
		for (unsigned round = 0; round < 3u; round++)
		{
			const long reg0 = p_stats()->host_register, unreg0 = p_stats()->host_unregister;
			for (std::size_t i = 0; i < bytes; i++)
			{
				buf[i] = static_cast<uint8_t>(i * 5 + round);
			}
			CHECK(p_htod(dev(std::data(device)), buf, bytes) == 0);
			CHECKF(std::memcmp(std::data(device), buf, bytes) == 0, "progressive htod data, round %d", round);
			const long regs = p_stats()->host_register - reg0, unregs = p_stats()->host_unregister - unreg0;
			CHECKF(regs >= 1 && regs == unregs,
				   "progressive htod round %d: %ld registrations, %ld unregistrations", round, regs, unregs);
			for (std::size_t i = 0; i < bytes; i++)
			{
				device[i] = static_cast<uint8_t>(i * 3 + round);
			}
			const long reg1 = p_stats()->host_register, unreg1 = p_stats()->host_unregister;
			CHECK(p_dtoh(buf, dev(std::data(device)), bytes) == 0);
			CHECKF(std::memcmp(std::data(device), buf, bytes) == 0, "progressive dtoh data, round %d", round);
			CHECKF(p_stats()->host_register - reg1 >= 1 &&
					   p_stats()->host_register - reg1 == p_stats()->host_unregister - unreg1,
				   "progressive dtoh round %d: registrations and unregistrations differ", round);
		}
		std::free(buf);
		std::printf("test_pipeline_e2e: progressive pinning ok\n");
		return check::failures() == 0 ? 0 : 1;
	}
	// Error propagation: a failing pipeline DMA must come back to the caller.
	if (env_int("FAKE_FAIL_PINNED_DMA_AT") > 0)
	{
		const std::size_t bytes = chunk * 9 + 5;
		check::Buffer host(bytes);
		check::Buffer device(bytes);
		const int h2d_rc = p_htod_async(dev(std::data(device)), std::data(host), bytes, user_stream);
		const int d2h_rc = p_dtoh_async(std::data(host), dev(std::data(device)), bytes, user_stream);
		CHECKF(h2d_rc == 700, "h2d returned %d", h2d_rc);
		CHECKF(d2h_rc == 700, "d2h returned %d", d2h_rc);
		return check::exit_code();
	}
	const long created_before = p_stats()->stream_create;

	const std::size_t sizes[] = {1,
								 4095,
								 chunk,
								 chunk * 3 + 17,
								 chunk * buffers + 1,
								 chunk * (buffers + 2) * 2 + 4097,
								 (5u << 20) + 1};
	unsigned seed = 1;
	for (std::size_t bytes : sizes)
	{
		for (Mode m : {Mode::Sync, Mode::AsyncUser, Mode::AsyncNull})
		{
			roundtrip(bytes, m, user_stream, seed++);
		}
	}
	// Back-to-back async copies through the persistent rotation.
	for (unsigned i = 0; i < 5u; i++)
	{
		roundtrip(chunk * 2 + 123, Mode::AsyncUser, user_stream, 100 + i);
	}
	stream_order(user_stream, chunk * 5 + 99);
	p_ctx_sync();





	const Stats* st = p_stats();
	if (pipeline)
	{
		CHECKF(st->stream_create - created_before == pipe_workers,
			   "worker streams created=%ld", st->stream_create - created_before);
		CHECK(st->stream_wait_event > 0);
		const bool pin = std::getenv("STAGING_THREADS_COPY_CPUS") != nullptr;
		// Lane zero runs on this caller; only the remaining lanes create helpers.
		CHECKF(count_threads("cmt-pipe-", pin) == pipe_workers - 1, "pipeline helpers=%d (pinned check %d)",
			   count_threads("cmt-pipe-", pin), pin);
	}

	else
	{
		CHECK(st->stream_create - created_before == 0);
	}
	// Pass-through count: async calls with a pageable host side.
	const long passthrough = st->htod_async_pageable_src + st->dtoh_async_pageable_dst;
	if (!pipeline)
	{
		// Nothing configured (no staging, no table, pinning off): the wrappers are installed but every copy
		// is the application's own driver call, async stays async (hook.cpp, CopyThroughHook's first check).
		CHECKF(passthrough > 0, "with staging off, async pageable copies must reach the driver unchanged");
	}

	else
	{
		CHECKF(passthrough == 0, "pageable pass-through with the pipeline on: %ld", passthrough);
	}

	if (redirect)
	{
		const long freed = st->host_free;
		const long freed_bytes = p_host_free_bytes();
		void* ctx = reinterpret_cast<void*>(0x1000);
		CHECK(p_ctx_destroy(ctx) == 0);
		// One allocation holds the workers' staging slots.
		const long allocations = pipeline ? 1 : 0;
		const long bytes = pipeline ? static_cast<long>(buffers) * pipe_workers *
			static_cast<long>((chunk + 4095) / 4096 * 4096) : 0;
		CHECKF(p_stats()->host_free - freed == allocations, "ctx destroy freed %ld allocations, %ld expected",
			   p_stats()->host_free - freed, allocations);
		CHECKF(p_host_free_bytes() - freed_bytes == bytes, "ctx destroy freed %ld bytes, %ld expected",
			   p_host_free_bytes() - freed_bytes, bytes);
		CHECK(p_stats()->ctx_destroy == 1);


		roundtrip(chunk + 5, Mode::Sync, user_stream, 77); // a new ring is built

	}
	return check::exit_code();
}
