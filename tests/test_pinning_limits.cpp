// Progressive pinning's registration limit and registration refusals, through the preloaded hook and the fake
// driver (no GPU):
//   test_pinning_limits default    no limit set: a copy may pin up to its GPU's memory (cuDeviceTotalMem_v2), read
//                                  once per device; unreadable, nothing is pinned
//   test_pinning_limits explicit   CMT_PINNING_MAX_MIB=8 or CMT_OVERRIDE pinning_max_mib=8: a 16 MiB copy is not
//                                  pinned, an 8 MiB one is, and the GPU's memory is never asked
//   test_pinning_limits refusals   the driver refuses the first registration, or the fourth (as a refused
//                                  pin_user_pages, ENOMEM, comes back: CUDA_ERROR_OUT_OF_MEMORY): the copy
//                                  completes, every registration is released
// Every copy, both directions: each byte arrives, no registration over the buffer outlives the call, and the hook's
// audit records the route (pinned, or the staging pipeline it falls back to here) and the pinned/pageable bytes.
// Exit 77 without verified patched-kernel provenance (ProgressivePinningAvailable), as every pinning fixture.
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
#include "copying/chunk_pipeline.h"

namespace {
using Dptr = unsigned long long;
using cmt::tune_audit::Outcome;
using cmt::tune_audit::Snapshot;

int (*h2d)(Dptr, const void*, std::size_t);
int (*d2h)(void*, Dptr, std::size_t);
int (*context_sync)();
void (*ro_config)(int, int, int, int, int, int);
void (*rw_error_at)(int, long);
void (*device_memory)(int, std::size_t, int);
long (*device_memory_queries)();
long (*overlap)(const void*, std::size_t);
long (*register_attempts)();
long (*unregister_attempts)();
cmt::tune_audit::Begin audit_begin;
cmt::tune_audit::End audit_end;

constexpr std::size_t kMiB = std::size_t{1} << 20;
// CUDA_ERROR_OUT_OF_MEMORY: what cuMemHostRegister returns when the kernel refuses to pin (ENOMEM).
constexpr int kOutOfMemory = 2;

void Require(bool ok, const char* message)
{
	if (!ok)
	{
		std::fprintf(stderr, "PINNING LIMIT REGRESSION FAILED: %s\n", message);
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

// What one copy did: the hook's audit of it, and the registrations the fake saw it make and release.
struct Copied
{
	Snapshot audit;
	long registrations = 0;
	long unregistrations = 0;

	bool Pinned() const
	{
		return audit.outcomes[static_cast<unsigned>(Outcome::ProgressivePinning)] == 1;
	}

	// Declined before any DMA: nothing registered, and the staging pipeline (STAGING_THREADS_COPY_NUM_BUFFERS)
	// copied it, as the driver would without one.
	bool Declined() const
	{
		return audit.outcomes[static_cast<unsigned>(Outcome::Env)] == 1 && audit.pinning.registered_bytes == 0 &&
			   unregistrations == 0;
	}
};

// One synchronous copy of `bytes` through the hook. Fails the test unless every byte arrives and no registration
// over the buffer survives the call; every registration it made was released (or it made none).
Copied Copy(bool to_device, std::size_t bytes)
{
	void* memory = nullptr;
	Require(posix_memalign(&memory, 4096, bytes) == 0, "allocate");
	auto* host = static_cast<uint8_t*>(memory);
	std::vector<uint8_t> device(bytes);
	for (std::size_t i = 0; i < bytes; i++)
	{
		host[i] = static_cast<uint8_t>((i * 131 + (i >> 12)) & 255);
		device[i] = static_cast<uint8_t>((i * 7 + 3) & 255);
	}
	auto* source = to_device ? host : std::data(device);
	auto* destination = to_device ? std::data(device) : host;
	const std::vector<uint8_t> expected(source, source + bytes);
	const auto dev = static_cast<Dptr>(reinterpret_cast<uintptr_t>(std::data(device)));
	Copied copied;
	const long registrations = register_attempts(), unregistrations = unregister_attempts();
	Require(audit_begin(cmt::tune_audit::kVersion, sizeof(Snapshot)) == 1, "audit begin");
	const int rc = to_device ? h2d(dev, host, bytes) : d2h(host, dev, bytes);
	Require(audit_end(&copied.audit, sizeof(Snapshot)) == 1, "audit end");
	Require(rc == 0, "the copy returned an error");
	// A staged fallback consumes the source before returning but may leave its last DMA queued.
	Require(context_sync() == 0, "wait for the fallback's DMA");
	Require(std::memcmp(destination, std::data(expected), bytes) == 0, "a byte differs after the copy");
	Require(overlap(host, bytes) == 0, "a registration over the buffer outlived the copy");
	copied.registrations = register_attempts() - registrations;
	copied.unregistrations = unregister_attempts() - unregistrations;
	Require(copied.audit.calls == 1 && copied.audit.errors == 0, "audit: one call, no error");
	const auto& pinning = copied.audit.pinning;
	Require(pinning.released_bytes == pinning.registered_bytes, "audit: registered bytes not all released");
	std::free(host);
	return copied;
}

void Default()
{
	for (bool to_device : {true, false})
	{
		// Fresh device ordinals per direction: the hook reads each device's memory once and keeps it.
		const int large = to_device ? 1 : 4, small = large + 1, unknown_device = large + 2;
		// `large` has 64 MiB: a 16 MiB copy is pinned whole, with no limit recorded (0 in the audit).
		ro_config(large, 0, 0, 0, 0, 1);
		device_memory(large, 64 * kMiB, 0);
		const auto pinned = Copy(to_device, 16 * kMiB);
		Require(pinned.Pinned() && pinned.audit.pinning.registered_bytes == 16 * kMiB &&
					pinned.audit.pinning.registered_dma_bytes == 16 * kMiB &&
					pinned.audit.pinning.pageable_dma_bytes == 0 && pinned.audit.pinning.max_registered_bytes == 0,
				"no limit set: a copy within the GPU's memory is pinned whole");
		Require(pinned.registrations > 0 && pinned.unregistrations == pinned.registrations,
				"every registration released");
		// Read once per device: the next copy on it does not ask again.
		const long queries = device_memory_queries();
		Require(Copy(to_device, 16 * kMiB).Pinned() && device_memory_queries() == queries,
				"the GPU's memory is read once per device");
		// `small` has 8 MiB: a 16 MiB copy is more than the limit, so nothing is registered; 8 MiB fits.
		ro_config(small, 0, 0, 0, 0, 1);
		device_memory(small, 8 * kMiB, 0);
		const auto over = Copy(to_device, 16 * kMiB);
		Require(over.Declined() && over.registrations == 0, "a copy above its GPU's memory is not pinned");
		Require(Copy(to_device, 8 * kMiB).Pinned(), "a copy within a smaller GPU's memory is pinned");
		// `unknown_device`'s memory cannot be read: nothing is pinned; a later successful read is used.
		ro_config(unknown_device, 0, 0, 0, 0, 1);
		device_memory(unknown_device, 64 * kMiB, 999);
		const auto unknown = Copy(to_device, 4 * kMiB);
		Require(unknown.Declined() && unknown.registrations == 0, "unknown GPU memory: no registration");
		device_memory(unknown_device, 64 * kMiB, 0);
		Require(Copy(to_device, 4 * kMiB).Pinned(), "the GPU's memory is read again after a failed read");
		// The devices read before that error keep their sizes: a different GPU in the same process.
		ro_config(large, 0, 0, 0, 0, 1);
		Require(Copy(to_device, 16 * kMiB).Pinned(), "the larger GPU still pins 16 MiB");
		ro_config(small, 0, 0, 0, 0, 1);
		Require(Copy(to_device, 16 * kMiB).Declined(), "the smaller GPU still declines 16 MiB");
	}
	std::puts("Default registration limit: each GPU's memory, read once per device; unreadable pins nothing");
}

void Explicit()
{
	for (bool to_device : {true, false})
	{
		const long queries = device_memory_queries();
		const auto over = Copy(to_device, 16 * kMiB);
		Require(over.Declined() && over.registrations == 0, "a copy above the explicit 8 MiB limit is not pinned");
		const auto within = Copy(to_device, 8 * kMiB);
		Require(within.Pinned() && within.audit.pinning.registered_bytes == 8 * kMiB,
				"a copy within the explicit limit is pinned whole");
		Require(device_memory_queries() == queries, "an explicit limit replaces the GPU's memory");
	}
	std::puts("Explicit registration limit: applied instead of the GPU's memory");
}

void Refusals()
{
	for (bool to_device : {true, false})
	{
		// The first registration is refused: nothing was submitted, so the copy is declined and takes the
		// fallback; one attempt, no retry.
		rw_error_at(kOutOfMemory, 1);
		const auto first = Copy(to_device, 16 * kMiB);
		Require(first.Declined() && first.registrations == 1, "a refused first registration declines the copy");
		// The fourth is refused: three pieces (1 MiB at most) are pinned and copied, the rest goes as pageable DMA
		// on the same stream, and the three are released before the call returns.
		rw_error_at(kOutOfMemory, 4);
		const auto partial = Copy(to_device, 16 * kMiB);
		const auto& pinning = partial.audit.pinning;
		Require(partial.Pinned() && partial.registrations == 4 && partial.unregistrations == 3,
				"a refusal mid-copy stops registering and releases what was pinned");
		// Piece ends also fall on 2 MiB boundaries (PiecesOf), so how many bytes three pieces hold depends on
		// where the buffer starts; what holds everywhere: three pinned pieces, the rest pageable, nothing twice.
		Require(pinning.completed == 1 && pinning.chunks == 3 && pinning.registered_bytes > 0 &&
					pinning.registered_dma_bytes == pinning.registered_bytes &&
					pinning.registered_dma_bytes + pinning.pageable_dma_bytes == 16 * kMiB,
				"audit: three pieces pinned, the rest pageable after the refusal");
		rw_error_at(0, 1);
		// Registration works again for the next copy: no state survived the refusals.
		const auto after = Copy(to_device, 16 * kMiB);
		Require(after.Pinned() && after.audit.pinning.registered_bytes == 16 * kMiB,
				"the copy after a refusal pins again");
	}
	std::puts("Registration refusals: the copy completes, every registration is released");
}
} // namespace

int main(int argc, char** argv)
{
	if (!cmt::ProgressivePinningAvailable())
	{
		std::puts("SKIP: progressive pinning requires verified patched-kernel provenance");
		return 77;
	}
	Require(argc == 2, "expected default, explicit or refusals");
	void* lib = dlopen("libcuda.so.1", RTLD_NOW);
	Require(lib != nullptr, "load the fake driver");
	Load(lib, h2d, "cuMemcpyHtoD_v2");
	Load(lib, d2h, "cuMemcpyDtoH_v2");
	Load(lib, context_sync, "cuCtxSynchronize");
	Load(lib, ro_config, "fake_ro_config");
	Load(lib, rw_error_at, "fake_rw_register_error_at");
	Load(lib, device_memory, "fake_device_memory");
	Load(lib, device_memory_queries, "fake_device_memory_queries");
	Load(lib, overlap, "fake_registered_overlap");
	Load(lib, register_attempts, "fake_register_attempts");
	Load(lib, unregister_attempts, "fake_unregister_attempts");
	Load(RTLD_DEFAULT, audit_begin, "cmt_tune_audit_begin");
	Load(RTLD_DEFAULT, audit_end, "cmt_tune_audit_end");
	const std::string mode(argv[1]);
	if (mode == "default")
	{
		Default();
	}
	else if (mode == "explicit")
	{
		Explicit();
	}
	else if (mode == "refusals")
	{
		Refusals();
	}
	else
	{
		Require(false, "unknown test case");
	}
	return 0;
}
