// Test-only runtime fault injection, linked only into cmt_cleanup_fault and gpu_copy_cleanup_fault
// (tests/CMakeLists.txt). Those relink the frontends with -Wl,--wrap=NAME, so every call the frontend makes to
// NAME arrives here as __wrap_NAME and the static CUDA runtime's NAME is reachable as __real_NAME. An
// LD_PRELOAD library cannot do this: with the static runtime the calls never go through the dynamic linker.
// Failed frees intentionally leave tiny allocations to process teardown; this file is never in a product.
#include <cuda_runtime.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

namespace {
bool fail(const char* name)
{
	const char* selected = std::getenv("CMT_TEST_FAIL_CLEANUP");
	if (selected == nullptr || std::strcmp(selected, name) != 0)
	{
		return false;
	}
	constexpr char message[] = "INJECTED cleanup failure\n";
	if (write(STDERR_FILENO, message, sizeof(message) - 1) < 0)
	{
		_exit(125);
	}
	return true;
}
} // namespace

// The __wrap_/__real_ names are the GNU linker's --wrap contract, not identifiers of ours.
#define FAIL_FREE(name, type)                                                                                \
	extern "C" cudaError_t __real_##name(type value);                                                        \
	extern "C" cudaError_t __wrap_##name(type value)                                                         \
	{                                                                                                        \
		if (fail(#name))                                                                                     \
			return cudaErrorUnknown;                                                                         \
		return __real_##name(value);                                                                         \
	}
FAIL_FREE(cudaFreeHost, void*)
FAIL_FREE(cudaHostUnregister, void*)
FAIL_FREE(cudaFree, void*)
FAIL_FREE(cudaStreamDestroy, cudaStream_t)
FAIL_FREE(cudaEventDestroy, cudaEvent_t)
#undef FAIL_FREE
