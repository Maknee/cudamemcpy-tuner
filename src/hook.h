#pragma once

#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <malloc.h>
#include <memory>
#include <new>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "arch/copy.h"
#include "arch/disasm/plan.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "common/error.h"
#include "config.h"
#include "context.h"
#include "copying/cpu_copy.h"
#include "copying/direct.h"
#include "copying/progressive_pinning.h"
#include "cuda/driver.h"
#include "cuda/versions.h"

namespace cmt {
namespace driver {

void InstallPatches(const patch::DriverVersion* vi, uintptr_t libcuda_base);

void CopyHostDescriptor(const patch::DriverVersion& vi, uint8_t* copy);

} // namespace driver

void* CudaSymbol(const char* name);

bool DriverApiReady();

bool DriverApiResolved();

void FreeToAllocator(void* p);

namespace ring {

void* RedirectLookup(const char* symbol, bool per_thread_stream);

void* RedirectForAddress(void* real);

void* WrapPrimaryByName(void* real);

void* WrapContextByName(void* real);

bool DisableCopiesForUnwrappedTeardown(const char* kind);

cu::CUresult CopyHtoD(cu::CUdeviceptr dst, const void* src, std::size_t bytes, cu::CUstream stream,
					  bool async);
cu::CUresult CopyDtoH(void* dst, cu::CUdeviceptr src, std::size_t bytes, cu::CUstream stream, bool async);

bool EnvironmentStagingEnabled();

} // namespace ring

#if defined(__x86_64__)
inline constexpr char kRequiredGlibcSymbolVersion[] = "GLIBC_2.2.5";
#elif defined(__aarch64__)
inline constexpr char kRequiredGlibcSymbolVersion[] = "GLIBC_2.17";
#endif

struct LoaderFunctions
{
	void* (*lookup)(void*, const char*);
	void* (*open)(const char*, int);
};

LoaderFunctions GetLoaderFunctions();
bool IsCudaDriverAddress(uintptr_t address) noexcept;

namespace trace {
void Start();
void Stop();
bool BypassTuning();
void* Redirect(const char* symbol, void* real, bool per_thread = false, int version = 0);
} // namespace trace
} // namespace cmt
