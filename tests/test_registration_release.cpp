// Fake-driver regression: a refused unregistration keeps (leaks) the registration and never terminates;
// progressive copies release borrowed registrations before address reuse.
#include "cuda/kernel_status.h"
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "common/memory.h"
#include "common/cpu.h"

namespace {
using Dptr = unsigned long long;
int (*h2d)(Dptr, const void*, std::size_t);
int (*d2h)(void*, Dptr, std::size_t);
int (*h2da)(Dptr, const void*, std::size_t, void*);
int (*d2ha)(void*, Dptr, std::size_t, void*);
int (*stream_create)(void**, unsigned);
int (*stream_sync)(void*);
int (*HostAlloc)(void**, std::size_t, unsigned);
int (*host_free)(void*);
int (*host_register)(void*, std::size_t, unsigned);
int (*host_unregister)(void*);
void (*fail_unregister)(int);
long (*ranges)();
long (*overlap)(const void*, std::size_t);
long (*register_attempts)();
long (*unregister_attempts)();

void Require(bool value, const char* message)
{
	if (!value)
	{
		std::fprintf(stderr, "REGRESSION FAILED: %s\n", message);
		std::exit(EXIT_FAILURE);
	}
}

template <class Fn>
void Load(void* lib, Fn& fn, const char* name)
{
	void* symbol = dlsym(lib, name);
	Require(symbol != nullptr, name);
	static_assert(sizeof(fn) == sizeof(symbol), "function-pointer ABI");
	std::memcpy(&fn, &symbol, sizeof(fn));
}

void Init()
{
	void* lib = dlopen("libcuda.so.1", RTLD_NOW);
	Require(lib != nullptr, "fake driver not loaded");
	Load(lib, h2d, "cuMemcpyHtoD_v2");
	Load(lib, d2h, "cuMemcpyDtoH_v2");
	Load(lib, h2da, "cuMemcpyHtoDAsync_v2");
	Load(lib, d2ha, "cuMemcpyDtoHAsync_v2");
	Load(lib, stream_create, "cuStreamCreate");
	Load(lib, stream_sync, "cuStreamSynchronize");
	Load(lib, HostAlloc, "cuMemHostAlloc");
	Load(lib, host_free, "cuMemFreeHost");
	Load(lib, host_register, "cuMemHostRegister");
	Load(lib, host_unregister, "cuMemHostUnregister");
	Load(lib, fail_unregister, "fake_fail_unregister");
	Load(lib, ranges, "fake_registered_ranges");
	Load(lib, overlap, "fake_registered_overlap");
	Load(lib, register_attempts, "fake_register_attempts");
	Load(lib, unregister_attempts, "fake_unregister_attempts");
}

constexpr std::size_t bytes = 4u << 20;

Dptr DevicePointer(const void* p)
{
	return static_cast<Dptr>(reinterpret_cast<uintptr_t>(p));
}

// The driver refuses one unregistration after a completed copy (as after a sticky error). The copy's DMA finished,
// so it returns success with the right bytes; the refused registration is kept (leaked, never unregistered) instead
// of the process terminating, and a later copy over it goes to the driver and is right too.
// Mutation: restore the Panic in UnregisterClaimed -> the child dies with SIGABRT.
void CheckRegistrationFailures()
{
	for (bool to_device : {true, false})
	{
		const pid_t child = fork();
		Require(child >= 0, "fork cleanup failure child");
		if (child == 0)
		{
			check::Buffer host(bytes);
			std::vector<uint8_t> device(bytes, 0x71);
			for (std::size_t i = 0; i < bytes; ++i)
			{
				host[i] = static_cast<uint8_t>(i * 7 + 3);
			}
			fail_unregister(1);
			for (int round = 0; round < 2; ++round)
			{
				const int rc = to_device ? h2d(DevicePointer(std::data(device)), std::data(host), bytes)
										 : d2h(std::data(host), DevicePointer(std::data(device)), bytes);
				if (rc != 0)
				{
					_exit(10 + round);
				}
				if (std::memcmp(std::data(device), std::data(host), bytes) != 0)
				{
					_exit(20 + round);
				}
				if (round == 0 && overlap(std::data(host), bytes) == 0)
				{
					_exit(30); // the registration whose release the driver refused must be kept
				}
				std::memset(to_device ? std::data(device) : std::data(host), 0x5a, bytes / 2);
			}
			_exit(0);
		}
		int status = 0;
		Require(waitpid(child, &status, 0) == child, "wait cleanup failure child");
		if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
		{
			std::fprintf(stderr, "child: %s %d\n", WIFEXITED(status) ? "exit" : "signal",
						 WIFEXITED(status) ? WEXITSTATUS(status) : WTERMSIG(status));
		}
		Require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
				"a refused unregistration did not leave the copy correct and the process running");
	}
}

void CheckAddressReuse()
{
	auto* old = static_cast<uint8_t*>(
		mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
	Require(old != MAP_FAILED, "initial mmap");
	std::memset(old, 0x31, bytes);
	std::vector<uint8_t> device(bytes);
	const long attempts_before = register_attempts();
	Require(h2d(DevicePointer(std::data(device)), old, bytes) == 0, "first progressive copy");
	Require(register_attempts() > attempts_before, "progressive registration path was not exercised");
	Require(overlap(old, bytes) == 0, "progressive registration survived completed copy");
	Require(munmap(old, bytes) == 0, "munmap old mapping");
	auto* fresh = static_cast<uint8_t*>(
		mmap(old, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0));
	Require(fresh == old, "reserve old address for reuse regression");
	std::memset(fresh, 0x93, bytes);
	Require(h2d(DevicePointer(std::data(device)), fresh, bytes) == 0, "reused-address progressive copy");
	Require(overlap(fresh, bytes) == 0 && std::memcmp(std::data(device), fresh, bytes) == 0,
			"address reuse copied stale data or retained registration");
	Require(munmap(fresh, bytes) == 0, "munmap new mapping");
}

void CheckOwnedPinnedAsync()
{
	void* stream = nullptr;
	Require(stream_create(&stream, 0) == 0, "create stream");
	for (bool allocated : {true, false})
	{
		void* memory = nullptr;
		if (allocated)
		{
			Require(HostAlloc(&memory, bytes, 0) == 0, "host allocation");
		}
		else
		{
			Require(posix_memalign(&memory, 4096, bytes) == 0, "aligned allocation");
			Require(host_register(memory, bytes, 0) == 0, "application registration");
		}
		auto* host = static_cast<uint8_t*>(memory);
		std::vector<uint8_t> device(bytes, 0x11);
		std::memset(host, 0x51, bytes);
		const long reg_before = register_attempts(), unreg_before = unregister_attempts();
		Require(h2da(DevicePointer(std::data(device)), host, bytes, stream) == 0, "pinned HtoDAsync");
		// Fake pinned DMAs are lazy. This assertion detects a synchronous
		// fallback; it makes no claim that real DMA cannot finish early.
		Require(device[0] == 0x11 && device.back() == 0x11, "hook forced pinned HtoDAsync to synchronize");
		Require(stream_sync(stream) == 0, "HtoD stream wait");
		Require(std::memcmp(std::data(device), host, bytes) == 0, "pinned HtoD data");
		std::memset(std::data(device), 0x71, bytes);
		Require(d2ha(host, DevicePointer(std::data(device)), bytes, stream) == 0, "pinned DtoHAsync");
		Require(host[0] == 0x51 && host[bytes - 1] == 0x51, "hook forced pinned DtoHAsync to synchronize");
		Require(stream_sync(stream) == 0, "DtoH stream wait");
		Require(std::memcmp(std::data(device), host, bytes) == 0, "pinned DtoH data");
		Require(register_attempts() == reg_before && unregister_attempts() == unreg_before,
				"hook changed application-owned registration");
		if (allocated)
		{
			Require(host_free(host) == 0, "host free");
		}
		else
		{
			Require(host_unregister(host) == 0, "application unregister");
			std::free(host);
		}
	}
	Require(ranges() == 0, "pinned regression leaked allocations");
	std::puts("Application-owned pinned async: hostalloc and register paths preserved in both directions");
}
} // namespace

int main(int argc, char** argv)
{
	if (!cmt::ProgressivePinningAvailable())
	{
		std::puts("SKIP: progressive pinning requires verified patched-kernel provenance");
		return 77;
	}
	Init();
	Require(argc == 2, "expected failures or pinned");
	if (std::string(argv[1]) == "failures")
	{
		CheckRegistrationFailures();
		CheckAddressReuse();
	}
	else if (std::string(argv[1]) == "pinned")
	{
		CheckOwnedPinnedAsync();
	}
	else
	{
		Require(false, "unknown test case");
	}
	return 0;
}
