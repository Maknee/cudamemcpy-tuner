// Real preload path, fake CUDA: directional permissions and failure ownership.
#include "cuda/kernel_status.h"
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/memory.h"
#include "common/cpu.h"

namespace {
using Dptr = unsigned long long;
int (*h2d)(Dptr, const void*, std::size_t);
int (*d2h)(void*, Dptr, std::size_t);
int (*context_sync)();
void (*configure)(int, int, int, int, int, int);
void (*rw_error)(int);
void (*fail_unregister)(int);
long (*flag_attempts)(unsigned);
long (*overlap)(const void*, std::size_t);
long (*queries)();
int (*queried_device)();

void Require(bool ok, const char* message)
{
	if (!ok)
	{
		std::fprintf(stderr, "READONLY REGRESSION FAILED: %s\n", message);
		std::exit(EXIT_FAILURE);
	}
}

template <class F>
void Load(void* lib, F& f, const char* name)
{
	void* p = dlsym(lib, name);
	Require(p != nullptr, name);
	static_assert(sizeof(f) == sizeof(p), "function ABI");
	std::memcpy(&f, &p, sizeof(f));
}

constexpr std::size_t bytes = 4u << 20;

struct Memory
{
	uint8_t* host = nullptr;
	std::vector<uint8_t> device = std::vector<uint8_t>(bytes);

	Memory()
	{
		void* p = nullptr;
		Require(posix_memalign(&p, 4096, bytes) == 0, "allocate");
		host = static_cast<uint8_t*>(p);
	}

	~Memory()
	{
		std::free(host);
	}

	Dptr DevicePointer()
	{
		return static_cast<Dptr>(reinterpret_cast<uintptr_t>(std::data(device)));
	}

	void Fill()
	{
		for (std::size_t i = 0; i < bytes; i++)
		{
			host[i] = static_cast<uint8_t>((i * 131 + (i >> 9)) & 255);
		}
		std::memset(std::data(device), 0, bytes);
	}

	void CheckEqual()
	{
		Require(std::memcmp(host, std::data(device), bytes) == 0, "all-byte mismatch");
	}

	void CheckReleased()
	{
		Require(overlap(host, bytes) == 0, "user range still registered");
	}
};

void Transfer(Memory& m, bool ro)
{
	m.Fill();
	const long before = flag_attempts(9), writable = flag_attempts(1);
	Require(h2d(m.DevicePointer(), m.host, bytes) == 0, "HtoD transfer");
	m.CheckEqual();
	Require(ro ? flag_attempts(9) > before : flag_attempts(9) == before,
			"wrong HtoD read-only flag selection");
	Require(ro || flag_attempts(1) > writable, "missing writable registration");
	m.CheckReleased();
	std::memset(std::data(m.device), 0x35, bytes);
	const long before_d2h = flag_attempts(9), query_before = queries();
	Require(d2h(m.host, m.DevicePointer(), bytes) == 0, "DtoH after HtoD registration");
	m.CheckEqual();
	Require(flag_attempts(9) == before_d2h && queries() == query_before,
			"DtoH used/queried read-only registration");
	m.CheckReleased();
}

void Enabled()
{
	Memory m;
	configure(7, 1, 0, 0, 0, 1);
	Transfer(m, true);
	Require(queried_device() == 7, "did not query current device");
	configure(9, 0, 0, 0, 0, 1);
	Transfer(m, false);
	Require(queried_device() == 9, "capability cached across devices");
	configure(12, 1, 999, 0, 0, 1);
	Transfer(m, false);
	configure(14, 1, 0, 201, 0, 1);
	const long q = queries();
	Transfer(m, false);
	Require(queries() == q, "queried attribute after context-device failure");
	configure(17, 1, 0, 0, 0, 1);
	Transfer(m, true);
	Require(queried_device() == 17, "capability did not recover on new current device");

	// Capability claimed support, but actual registration rejects flags9.
	// There must be one failed flags9 attempt, then flags1 for all this copy.
	configure(17, 1, 0, 0, 801, 1);
	m.Fill();
	long r9 = flag_attempts(9), r1 = flag_attempts(1);
	Require(h2d(m.DevicePointer(), m.host, bytes) == 0, "unsupported flags9 did not fall back");
	Require(flag_attempts(9) == r9 + 1 && flag_attempts(1) > r1,
			"unsupported flags9 retry was unbounded or absent");
	m.CheckEqual();
	m.CheckReleased();
	configure(17, 1, 0, 0, 801, 2);
	m.Fill();
	r9 = flag_attempts(9);
	r1 = flag_attempts(1);
	Require(h2d(m.DevicePointer(), m.host, bytes) == 0, "unsupported flag after owned piece did not recover");
	Require(flag_attempts(9) == r9 + 2 && flag_attempts(1) > r1,
			"late fallback did not downgrade remaining pieces");
	m.CheckEqual();
	m.CheckReleased();
	// Fallback itself can fail: preserve the established pageable-driver path.
	configure(17, 1, 0, 0, 801, 1);
	rw_error(2);
	m.Fill();
	r9 = flag_attempts(9);
	r1 = flag_attempts(1);
	Require(h2d(m.DevicePointer(), m.host, bytes) == 0,
			"writable register failure did not use pageable fallback");
	Require(flag_attempts(9) == r9 + 1 && flag_attempts(1) == r1 + 1,
			"failed fallback registration repeated");
	m.CheckReleased();
	// No registration succeeded, so the ordinary pageable HtoD path applies.
	// Staging consumes the source before returning but can leave DMA pending.
	// Keep every successful progressive transfer's immediate equality check.
	std::fputs("Read-only check: writable registration failed before DMA; waiting for ordinary fallback\n",
			   stderr);
	Require(context_sync() == 0, "ordinary fallback completion");
	m.CheckEqual();

	// A non-support-related error after a successful earlier piece must return
	// the original error and still release all earlier owned registrations.
	configure(17, 1, 0, 0, 700, 2);
	m.Fill();
	r9 = flag_attempts(9);
	r1 = flag_attempts(1);
	Require(h2d(m.DevicePointer(), m.host, bytes) == 700, "non-801 read-only error was hidden");
	Require(flag_attempts(9) == r9 + 2 && flag_attempts(1) == r1, "unexpected retry after non-801 failure");
	m.CheckReleased();

	// Fatal unregister failures in both directions are covered by the existing
	// readonly_release_reuse test (test_registration_release).
	std::puts("Read-only progressive pinning: directional lifecycle, device query and registration fallback passed");
}
} // namespace

int main(int argc, char** argv)
{
	if (!cmt::ProgressivePinningAvailable())
	{
		std::puts("SKIP: read-only pinning requires verified patched-kernel provenance");
		return 77;
	}
	Require(argc == 2, "expected enabled/disabled/missing_api");
	void* lib = dlopen("libcuda.so.1", RTLD_NOW);
	Require(lib != nullptr, "load fake CUDA");
	Load(lib, h2d, "cuMemcpyHtoD_v2");
	Load(lib, d2h, "cuMemcpyDtoH_v2");
	Load(lib, context_sync, "cuCtxSynchronize");
	Load(lib, configure, "fake_ro_config");
	Load(lib, rw_error, "fake_rw_register_error");
	Load(lib, fail_unregister, "fake_fail_unregister");
	Load(lib, flag_attempts, "fake_flag_attempts");
	Load(lib, overlap, "fake_registered_overlap");
	Load(lib, queries, "fake_ro_queries");
	Load(lib, queried_device, "fake_ro_queried_device");
	const std::string mode(argv[1]);
	Require(mode == "enabled" || mode == "disabled" || mode == "missing_api",
			"unknown test case");
	if (mode == "enabled")
	{
		Enabled();
	}
	else
	{
		Memory m;
		configure(7, 1, 0, 0, 0, 1);
		const long q = queries();
		Transfer(m, false);
		Require(queries() == q, "disabled/missing path queried read-only capability");
		Require(flag_attempts(9) == 0, "read-only used without read-only configuration");
	}
	return 0;
}
