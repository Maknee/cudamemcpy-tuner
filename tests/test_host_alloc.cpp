// The opt-in pinned host allocation (copying/host_alloc.h), through the preloaded hook and the fake driver (no GPU):
//   test_host_alloc off   CMT_CUDA_HOST_ALLOC_HOOK unset or 0: dlsym and cuGetProcAddress hand out the driver's own
//                         allocation entry points, and every allocation is the driver's
//   test_host_alloc on    CMT_CUDA_HOST_ALLOC_HOOK=1: an eligible allocation is the hook's registered anonymous
//                         memory, freed by cuMemFreeHost; the flags it registers and reports; what goes to the
//                         driver unchanged (write-combined, unknown flags, small sizes, no context, foreign
//                         pointers, a fork() child); every failure (mmap, registration, device address, release)
//                         undoes its work and leaves the driver's answer; threads allocating and freeing at once
//   test_host_alloc private  CMT_CUDA_HOST_ALLOC_HOOK=1 and CMT_CUDA_HOST_ALLOC_PRIVATE=1: the same, with private
//                         memory, which a fork() child gets a copy of instead of a share
#include <dlfcn.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "common/memory.h"
#include "common/cpu.h"

namespace {
using Dptr = unsigned long long;
constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr int kInvalidValue = 1;
constexpr int kOutOfMemory = 2;
constexpr int kNotRegistered = 713;
constexpr int kAlreadyRegistered = 712;
constexpr unsigned kWriteCombined = 4;

void* g_lib = nullptr;
bool g_private = false; // test_host_alloc private
int (*host_alloc)(void**, std::size_t, unsigned);
int (*alloc_host)(void**, std::size_t);
int (*free_host)(void*);
int (*get_flags)(unsigned*, void*);
int (*unregister)(void*);
int (*register_memory)(void*, std::size_t, unsigned);
int (*device_pointer)(Dptr*, void*, unsigned);
int (*pointer_attribute)(void*, unsigned, Dptr);
int (*get_proc)(const char*, void**, int, unsigned long long);
int (*ctx_get)(void**);
int (*ctx_set)(void*);
long (*alloc_attempts)();
long (*alloc_flag_attempts)(unsigned);
long (*register_attempts)();
long (*register_flag_attempts)(unsigned);
long (*unregister_attempts)();
long (*registered_ranges)();
long (*registered_overlap)(const void*, std::size_t);
void (*fail_register)(int, int);
void (*fail_unregister)(int);
void (*device_pointer_offset)(long);
void (*drop_registrations)();
int (*ctx_destroy)(void*);
int (*primary_retain)(void**, int);
int (*primary_release)(int);
int (*primary_reset)(int);
void (*teardown_drops_registrations)(int);

void Require(bool ok, const char* what)
{
	if (!ok)
	{
		std::fprintf(stderr, "HOST ALLOC SETUP FAILED: %s\n", what);
		std::exit(EXIT_FAILURE);
	}
}

template <class F>
void Load(F& f, const char* name)
{
	void* p = dlsym(g_lib, name);
	Require(p != nullptr, name);
	static_assert(sizeof(f) == sizeof(p), "function-pointer ABI");
	std::memcpy(&f, &p, sizeof(f));
}

// The file that defines the function at `p`: the hook or the fake driver.
std::string Owner(const void* p)
{
	Dl_info info{};
	return dladdr(p, &info) != 0 && info.dli_fname != nullptr ? std::string(info.dli_fname) : std::string("?");
}

bool FromHook(const void* p)
{
	return Owner(p).find("libcmt_hook") != std::string::npos;
}

bool FromDriver(const void* p)
{
	return Owner(p).find("libcuda.so") != std::string::npos;
}

void* Query(const char* symbol, int version, unsigned long long flags = 0)
{
	void* fn = nullptr;
	return get_proc(symbol, &fn, version, flags) == 0 ? fn : nullptr;
}

bool Mapped(const void* p)
{
	unsigned char resident = 0;
	const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
	void* start = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(p) / page * page);
	return mincore(start, page, &resident) == 0;
}

long RssKiB()
{
	std::ifstream in("/proc/self/status");
	std::string line;
	while (std::getline(in, line))
	{
		if (line.rfind("VmRSS:", 0) == 0)
		{
			return std::strtol(line.c_str() + 6, nullptr, 10);
		}
	}
	return -1;
}

// The permissions of the mapping holding `p` in /proc/self/maps ("rw-s", "rw-p"), or empty.
std::string Permissions(const void* p)
{
	std::ifstream maps("/proc/self/maps");
	const auto address = reinterpret_cast<uintptr_t>(p);
	for (std::string line; std::getline(maps, line);)
	{
		unsigned long long low = 0, high = 0;
		char permissions[5] = {};
		if (std::sscanf(line.c_str(), "%llx-%llx %4s", &low, &high, permissions) == 3 && low <= address &&
			address < high)
		{
			return permissions;
		}
	}
	return {};
}

// The VmFlags of the mapping holding `p` in /proc/self/smaps (" rd wr mr ..."), or empty.
std::string VmFlags(const void* p)
{
	std::ifstream smaps("/proc/self/smaps");
	const auto address = reinterpret_cast<uintptr_t>(p);
	bool inside = false;
	for (std::string line; std::getline(smaps, line);)
	{
		unsigned long long low = 0, high = 0;
		if (std::sscanf(line.c_str(), "%llx-%llx ", &low, &high) == 2 && line.find(':') > line.find(' '))
		{
			inside = low <= address && address < high;
		}
		else if (inside && line.rfind("VmFlags:", 0) == 0)
		{
			return line.substr(8) + " ";
		}
	}
	return {};
}

unsigned MemoryType(const void* p)
{
	unsigned type = 0;
	return pointer_attribute(&type, 2, static_cast<Dptr>(reinterpret_cast<uintptr_t>(p))) == 0 ? type : 0;
}

// What the hook serves: the fake's allocation count does not move, one registration of the request rounded up
// to whole pages with PORTABLE | DEVICEMAP appears, aligned to the THP size, host memory type, mapped at its host
// address, writable.
void* HookAllocation(std::size_t bytes, unsigned flags)
{
	const long allocs = alloc_attempts();
	const long registrations = register_flag_attempts(3);
	void* p = nullptr;
	CHECK_EQ(host_alloc(&p, bytes, flags), 0);
	CHECK(p != nullptr);
	CHECK_EQ(alloc_attempts(), allocs);
	CHECK_EQ(register_flag_attempts(3), registrations + 1);
	if (p == nullptr)
	{
		return nullptr;
	}
	CHECK_EQ(reinterpret_cast<uintptr_t>(p) % cmt::ThpBytes(), uintptr_t{0});
	CHECK_EQ(Permissions(p), std::string(g_private ? "rw-p" : "rw-s"));
	CHECK_EQ(MemoryType(p), 1u);
	CHECK_EQ(registered_overlap(p, bytes), 1L);
	Dptr device = 0;
	CHECK_EQ(device_pointer(&device, p, 0), 0);
	CHECK_EQ(device, static_cast<Dptr>(reinterpret_cast<uintptr_t>(p)));
	unsigned reported = 0;
	CHECK_EQ(get_flags(&reported, static_cast<char*>(p) + bytes - 1), 0);
	CHECK_EQ(reported, flags | 2u);
	auto* bytes_view = static_cast<unsigned char*>(p);
	CHECK(bytes_view[0] == 0 && bytes_view[bytes - 1] == 0);
	std::memset(p, 0x5a, bytes);
	return p;
}

// The driver served it: the fake's count moved and no registration was made.
void* DriverAllocation(std::size_t bytes, unsigned flags)
{
	const long allocs = alloc_attempts();
	const long registrations = register_attempts();
	void* p = nullptr;
	CHECK_EQ(host_alloc(&p, bytes, flags), 0);
	CHECK_EQ(alloc_attempts(), allocs + 1);
	CHECK_EQ(register_attempts(), registrations);
	return p;
}

void Free(void* p)
{
	const long ranges = registered_ranges();
	CHECK_EQ(free_host(p), 0);
	CHECK_EQ(registered_ranges(), ranges - 1);
}

int Off()
{
	// dlsym takes export names, cuGetProcAddress the unversioned API names.
	for (const char* name : {"cuMemHostAlloc", "cuMemAllocHost_v2", "cuMemFreeHost", "cuMemHostGetFlags",
							 "cuMemHostRegister_v2", "cuMemHostUnregister"})
	{
		CHECK(FromDriver(dlsym(g_lib, name)));
	}
	for (const char* name :
		 {"cuMemHostAlloc", "cuMemAllocHost", "cuMemFreeHost", "cuMemHostGetFlags", "cuMemHostUnregister"})
	{
		CHECK(FromDriver(Query(name, 12000)));
	}
	CHECK(FromDriver(Query("cuMemHostAlloc", 12000, 2)));
	void* p = DriverAllocation(64 * kMiB, 0);
	CHECK_EQ(free_host(p), 0);
	CHECK_EQ(registered_ranges(), 0L);
	return check::exit_code();
}

TEST(entry_points_are_the_hook_s)
{
	for (const char* name : {"cuMemHostAlloc", "cuMemAllocHost_v2", "cuMemFreeHost", "cuMemHostGetFlags",
							 "cuMemHostRegister_v2", "cuMemHostUnregister"})
	{
		CHECK(FromHook(dlsym(g_lib, name)));
	}
	for (const char* name :
		 {"cuMemHostAlloc", "cuMemAllocHost", "cuMemFreeHost", "cuMemHostGetFlags", "cuMemHostUnregister"})
	{
		CHECK(FromHook(Query(name, 12000)));
		CHECK(FromHook(Query(name, 12000, 2))); // per-thread default stream: no stream argument, same function
	}
	CHECK(dlsym(g_lib, "cuMemAllocHost") != nullptr && FromDriver(dlsym(g_lib, "cuMemAllocHost")));
	CHECK(FromDriver(Query("cuMemAllocHost", 3010))); // 32-bit size: never the size_t replacement
	CHECK(FromDriver(dlsym(g_lib, "cuMemHostRegister"))); // the legacy export is left alone
}

TEST(round_trip_registers_then_frees)
{
	const long unregistrations = unregister_attempts();
	for (std::size_t bytes : {2 * kMiB, 64 * kMiB + 100, 192 * kMiB})
	{
		void* p = HookAllocation(bytes, 0);
		Free(p);
		CHECK(!Mapped(p));
	}
	CHECK_EQ(unregister_attempts(), unregistrations + 3);
	void* q = nullptr;
	const long allocs = alloc_attempts();
	CHECK_EQ(alloc_host(&q, 8 * kMiB), 0);
	CHECK_EQ(alloc_attempts(), allocs);
	CHECK_EQ(registered_overlap(q, 8 * kMiB), 1L);
	Free(q);
	CHECK_EQ(registered_ranges(), 0L);
}

TEST(requested_flags_register_portable_and_mapped)
{
	for (unsigned flags : {0u, 1u, 2u, 3u})
	{
		Free(HookAllocation(4 * kMiB, flags));
	}
	CHECK_EQ(register_flag_attempts(0) + register_flag_attempts(1) + register_flag_attempts(2), 0L);
}

TEST(write_combined_unknown_flags_small_and_empty_go_to_the_driver)
{
	for (unsigned flags : {kWriteCombined, kWriteCombined | 2u, 0x100u})
	{
		const long wc = alloc_flag_attempts(kWriteCombined);
		void* p = DriverAllocation(8 * kMiB, flags);
		CHECK_EQ(alloc_flag_attempts(kWriteCombined), wc + (flags == kWriteCombined ? 1 : 0));
		Free(p);
	}
	Free(DriverAllocation(2 * kMiB - 1, 0));
	void* empty = DriverAllocation(0, 0);
	CHECK_EQ(free_host(empty), 0);
	void* small = nullptr;
	const long allocs = alloc_attempts();
	CHECK_EQ(alloc_host(&small, 4096), 0);
	CHECK_EQ(alloc_attempts(), allocs + 1);
	Free(small);
}

TEST(foreign_pointers_reach_the_driver)
{
	// A driver allocation freed through the hook's cuMemFreeHost.
	void* driver = DriverAllocation(8 * kMiB, kWriteCombined);
	unsigned flags = 0;
	CHECK_EQ(get_flags(&flags, driver), 0);
	CHECK_EQ(flags, kWriteCombined | 2u);
	Free(driver);
	// The application's own registration: unregistered by the driver, not refused.
	void* own = mmap(nullptr, 4 * kMiB, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	Require(own != MAP_FAILED, "mmap");
	CHECK_EQ(register_memory(own, 4 * kMiB, 0), 0);
	CHECK_EQ(unregister(own), 0);
	CHECK_EQ(registered_overlap(own, 4 * kMiB), 0L);
	// The application registering over its own registration: the driver's 712.
	CHECK_EQ(register_memory(own, 4 * kMiB, 0), 0);
	CHECK_EQ(register_memory(static_cast<char*>(own) + 4096, 4096, 0), kAlreadyRegistered);
	CHECK_EQ(unregister(own), 0);
	munmap(own, 4 * kMiB);
	// The hook's allocation answers as driver memory does (INVALID_VALUE, measured on 615.71.09) to
	// cuMemHostUnregister and to cuMemHostRegister over it, and stays registered.
	void* p = HookAllocation(4 * kMiB, 0);
	const long registrations = register_attempts();
	CHECK_EQ(unregister(p), kInvalidValue);
	CHECK_EQ(unregister(static_cast<char*>(p) + 4096), kInvalidValue);
	CHECK_EQ(register_memory(static_cast<char*>(p) + 4096, 4096, 0), kInvalidValue);
	CHECK_EQ(register_memory(static_cast<char*>(p) - 4096, 8192, 0), kInvalidValue);
	CHECK_EQ(register_attempts(), registrations);
	CHECK_EQ(registered_overlap(p, 4 * kMiB), 1L);
	// And so does driver memory itself, through the hook (the fake answers as the driver).
	void* wc = DriverAllocation(4 * kMiB, kWriteCombined);
	CHECK_EQ(unregister(wc), kInvalidValue);
	CHECK_EQ(register_memory(wc, 4096, 0), kInvalidValue);
	Free(wc);
	unsigned stack_flags = 0;
	CHECK_EQ(get_flags(&stack_flags, &stack_flags), kInvalidValue);
	Free(p);
}

TEST(no_current_context_goes_to_the_driver)
{
	std::thread([]
				{
					void* p = DriverAllocation(8 * kMiB, 0); // this thread has no context (the fake's rule)
					CHECK_EQ(free_host(p), 0);
				})
		.join();
}

TEST(mmap_failure_uses_the_driver_and_its_error)
{
	rlimit saved{};
	Require(getrlimit(RLIMIT_AS, &saved) == 0, "getrlimit");
	// Room for small mappings, none for 1 GiB: the hook's mmap fails, and so does the fake's own allocation.
	std::ifstream statm("/proc/self/statm");
	unsigned long pages = 0;
	statm >> pages;
	rlimit low = saved;
	low.rlim_cur = pages * static_cast<unsigned long>(sysconf(_SC_PAGESIZE)) + 256 * kMiB;
	Require(setrlimit(RLIMIT_AS, &low) == 0, "setrlimit");
	const long allocs = alloc_attempts();
	const long registrations = register_attempts();
	void* p = nullptr;
	const int rc = host_alloc(&p, 1024 * kMiB, 0);
	Require(setrlimit(RLIMIT_AS, &saved) == 0, "restore rlimit");
	CHECK_EQ(rc, kOutOfMemory);
	CHECK_EQ(alloc_attempts(), allocs + 1);
	CHECK_EQ(register_attempts(), registrations);
}

TEST(registration_failure_unmaps_and_uses_the_driver)
{
	const long before = RssKiB();
	const long allocs = alloc_attempts();
	const long registrations = register_attempts();
	fail_register(kOutOfMemory, 1);
	void* p = nullptr;
	CHECK_EQ(host_alloc(&p, 64 * kMiB, 0), 0);
	CHECK_EQ(register_attempts(), registrations + 1); // the hook's, refused
	CHECK_EQ(alloc_attempts(), allocs + 1);			  // then the driver's own
	CHECK(p != nullptr && registered_overlap(p, 64 * kMiB) == 1);
	Free(p);
	// The hook touched 64 MiB before the refusal: it must be unmapped again.
	CHECK(RssKiB() - before < 16 * 1024);
	fail_register(0, 0);
}

TEST(device_address_mismatch_unregisters_and_uses_the_driver)
{
	const long before = RssKiB();
	const long unregistrations = unregister_attempts();
	const long ranges = registered_ranges();
	device_pointer_offset(4096);
	void* p = nullptr;
	const long allocs = alloc_attempts();
	CHECK_EQ(host_alloc(&p, 64 * kMiB, 0), 0);
	device_pointer_offset(0);
	CHECK_EQ(alloc_attempts(), allocs + 1);
	CHECK_EQ(unregister_attempts(), unregistrations + 1);
	CHECK_EQ(registered_ranges(), ranges + 1); // the driver's allocation only
	Free(p);
	CHECK(RssKiB() - before < 16 * 1024);
}

TEST(failed_release_keeps_the_allocation)
{
	void* p = HookAllocation(4 * kMiB, 1);
	fail_unregister(1);
	CHECK_EQ(free_host(p), 700);
	CHECK(Mapped(p));
	CHECK_EQ(registered_overlap(p, 4 * kMiB), 1L);
	unsigned flags = 0;
	CHECK_EQ(get_flags(&flags, p), 0);
	CHECK_EQ(flags, 3u);
	Free(p); // the retry frees it
	CHECK(!Mapped(p));
	// Freed: a second free is the driver's (the fake knows no such pointer), not the hook's.
	const long unregistrations = unregister_attempts();
	CHECK_EQ(get_flags(&flags, p), kInvalidValue);
	CHECK_EQ(unregister(p), kNotRegistered);
	CHECK_EQ(unregister_attempts(), unregistrations + 1);
}

TEST(free_after_the_context_dropped_the_registration)
{
	// The context that registered it is gone (cuCtxDestroy): the driver no longer knows the registration
	// (cuMemHostUnregister: 713). Nothing maps the pages any more, so the free unmaps them and answers what the
	// driver answers for its own memory then: INVALID_VALUE.
	void* p = HookAllocation(8 * kMiB, 0);
	drop_registrations();
	CHECK_EQ(registered_overlap(p, 8 * kMiB), 0L);
	CHECK_EQ(free_host(p), kInvalidValue);
	CHECK(!Mapped(p));
	unsigned flags = 0;
	CHECK_EQ(get_flags(&flags, p), kInvalidValue); // no longer the hook's
}

TEST(fork_child_uses_the_driver)
{
	void* p = HookAllocation(4 * kMiB, 0);
	const pid_t child = fork();
	Require(child >= 0, "fork");
	if (child == 0)
	{
		const long allocs = alloc_attempts();
		void* q = nullptr;
		const bool driver = host_alloc(&q, 8 * kMiB, 0) == 0 && alloc_attempts() == allocs + 1;
		// The driver answers, not the hook's table: the fake reports the registration's own flags (3), the hook
		// would report the requested 0 | DEVICEMAP (2).
		unsigned flags = 0;
		const bool driver_answers = get_flags(&flags, p) == 0 && flags == 3u;
		// The parent's memory is shared with the child, as driver memory is (private memory: copied).
		static_cast<volatile unsigned char*>(p)[0] = 0xa5;
		_exit(driver && driver_answers ? 0 : 1);
	}
	int status = 0;
	Require(waitpid(child, &status, 0) == child, "waitpid");
	CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
	// Shared: the child's write, as with the driver's MAP_SHARED memory. Private: the child wrote its own copy.
	CHECK_EQ(static_cast<unsigned char*>(p)[0], g_private ? 0x5a : 0xa5);
	Free(p);
}

TEST(threads_allocate_and_free_concurrently)
{
	void* context = nullptr;
	Require(ctx_get(&context) == 0 && context != nullptr, "main thread context");
	const long allocs = alloc_attempts();
	const long ranges = registered_ranges();
	std::atomic<long> wrong{0};
	std::vector<std::thread> threads;
	for (unsigned t = 0; t < 8; ++t)
	{
		threads.emplace_back(
			[&, t]
			{
				if (ctx_set(context) != 0)
				{
					++wrong;
					return;
				}
				void* live[3] = {};
				std::size_t sizes[3] = {};
				for (unsigned i = 0; i < 24; ++i)
				{
					const unsigned slot = i % 3;
					if (live[slot] != nullptr)
					{
						const auto* bytes = static_cast<const unsigned char*>(live[slot]);
						for (std::size_t at = 0; at < sizes[slot]; at += 4096)
						{
							wrong += bytes[at] != check::pattern(at / 4096, t * 100 + slot);
						}
						wrong += free_host(live[slot]) != 0;
					}
					sizes[slot] = (2 + (i + t) % 5) * kMiB + i * 4096;
					if (host_alloc(&live[slot], sizes[slot], (i + t) % 4) != 0 || live[slot] == nullptr)
					{
						++wrong;
						live[slot] = nullptr;
						continue;
					}
					auto* bytes = static_cast<unsigned char*>(live[slot]);
					for (std::size_t at = 0; at < sizes[slot]; at += 4096)
					{
						bytes[at] = check::pattern(at / 4096, t * 100 + slot);
					}
				}
				for (void* p : live)
				{
					wrong += p != nullptr && free_host(p) != 0;
				}
			});
	}
	for (auto& thread : threads)
	{
		thread.join();
	}
	CHECK_EQ(wrong.load(), 0L);
	CHECK_EQ(alloc_attempts(), allocs);
	CHECK_EQ(registered_ranges(), ranges);
}

TEST(many_threads_touch_one_allocation)
{
	// 512 MiB with the default thread count: several helpers fault it in; every page reads zero.
	void* p = HookAllocation(512 * kMiB + 4096, 0);
	Free(p);
}

// Private memory on a kernel that does not copy DMA-pinned pages at fork() (before 5.11) is excluded from fork()
// (MADV_DONTFORK, VmFlags "dc"); on a newer kernel, and for shared memory, it is not (a child copies or shares it,
// fork_child_uses_the_driver). Only this kernel's side can be checked here.
TEST(private_memory_is_excluded_from_fork_only_where_fork_could_steal_its_pages)
{
	utsname name{};
	unsigned major = 0, minor = 0;
	Require(uname(&name) == 0 && std::sscanf(name.release, "%u.%u", &major, &minor) == 2, "kernel version");
	const bool early_cow = major > 5 || (major == 5 && minor >= 11);
	void* p = HookAllocation(4 * kMiB, 0);
	const std::string flags = VmFlags(p);
	CHECK(!flags.empty());
	CHECK_EQ(flags.find(" dc ") != std::string::npos, g_private && !early_cow);
	Free(p);
}

// The driver frees its own pinned memory with the context that made it, and forgets that context's
// registrations. The hook's allocations go with it too (host_alloc.h, AfterContextTeardown): unmapped, unknown to
// the hook, and a later cuMemFreeHost gets the driver's INVALID_VALUE. An allocation whose registration is still
// alive (a primary-context release that left the context alive) stays. Last: the context resets.
TEST(context_teardown_releases_allocations_as_the_driver_does)
{
	teardown_drops_registrations(1);
	// A release that does not destroy the primary context (two retains, one release) keeps everything.
	void* kept = HookAllocation(4 * kMiB, 0);
	void* context = nullptr;
	CHECK_EQ(primary_retain(&context, 0), 0);
	CHECK_EQ(primary_retain(&context, 0), 0);
	CHECK_EQ(primary_release(0), 0);
	CHECK(Mapped(kept));
	CHECK_EQ(MemoryType(kept), 1u);
	Free(kept);
	// cuCtxDestroy.
	void* first = HookAllocation(8 * kMiB, 0);
	void* second = HookAllocation(4 * kMiB, 1);
	CHECK_EQ(ctx_destroy(context), 0);
	for (void* p : {first, second})
	{
		CHECK(!Mapped(p));
		unsigned flags = 0;
		CHECK_EQ(get_flags(&flags, p), kInvalidValue); // no longer the hook's
		CHECK_EQ(free_host(p), kInvalidValue);		   // the driver's answer for memory it does not know
	}
	// A primary-context reset (cudaDeviceReset).
	void* third = HookAllocation(4 * kMiB, 0);
	CHECK_EQ(primary_reset(0), 0);
	CHECK(!Mapped(third));
	CHECK_EQ(free_host(third), kInvalidValue);
	teardown_drops_registrations(0);
	// New allocations work as before.
	Free(HookAllocation(4 * kMiB, 0));
}
} // namespace

int main(int argc, char** argv)
{
	Require(argc == 2, "expected off, on or private");
	g_lib = dlopen("libcuda.so.1", RTLD_NOW);
	Require(g_lib != nullptr, "load the fake driver");
	Load(host_alloc, "cuMemHostAlloc");
	Load(alloc_host, "cuMemAllocHost_v2");
	Load(free_host, "cuMemFreeHost");
	Load(get_flags, "cuMemHostGetFlags");
	Load(unregister, "cuMemHostUnregister");
	Load(register_memory, "cuMemHostRegister_v2");
	Load(device_pointer, "cuMemHostGetDevicePointer_v2");
	Load(pointer_attribute, "cuPointerGetAttribute");
	Load(get_proc, "cuGetProcAddress");
	Load(ctx_get, "cuCtxGetCurrent");
	Load(ctx_set, "cuCtxSetCurrent");
	Load(alloc_attempts, "fake_host_alloc_attempts");
	Load(alloc_flag_attempts, "fake_host_alloc_flag_attempts");
	Load(register_attempts, "fake_register_attempts");
	Load(register_flag_attempts, "fake_flag_attempts");
	Load(unregister_attempts, "fake_unregister_attempts");
	Load(registered_ranges, "fake_registered_ranges");
	Load(registered_overlap, "fake_registered_overlap");
	Load(fail_register, "fake_fail_register");
	Load(fail_unregister, "fake_fail_unregister");
	Load(device_pointer_offset, "fake_device_pointer_offset");
	Load(drop_registrations, "fake_drop_registrations");
	Load(ctx_destroy, "cuCtxDestroy_v2");
	Load(primary_retain, "cuDevicePrimaryCtxRetain");
	Load(primary_release, "cuDevicePrimaryCtxRelease_v2");
	Load(primary_reset, "cuDevicePrimaryCtxReset_v2");
	Load(teardown_drops_registrations, "fake_teardown_drops_registrations");
	const std::string mode(argv[1]);
	if (mode == "off")
	{
		return Off();
	}
	Require(mode == "on" || mode == "private", "expected off, on or private");
	g_private = mode == "private";
	return check::run_all();
}
