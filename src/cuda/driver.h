#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <mutex>
#include "common/log.h"
#include "common/memory.h"
#include "common/cpu.h"

namespace cmt {
namespace cu {

using CUresult = int;
using CUdeviceptr = unsigned long long;
using CUstream = void*;
using CUevent = void*;
using CUcontext = void*;
using CUdevice = int;
using CUmodule = void*;
using CUfunction = void*;

constexpr CUresult kSuccess = 0;

enum RuntimeMemcpyKind : int
{
	cudaMemcpyHostToHost = 0,
	cudaMemcpyHostToDevice = 1,
	cudaMemcpyDeviceToHost = 2,
	cudaMemcpyDeviceToDevice = 3,
	cudaMemcpyDefault = 4
};

bool IsCapturing(CUstream stream);

enum class HostPointerKind
{
	Pageable,
	Pinned,
	Managed,
	Device,
	Unknown
};

HostPointerKind ClassifyHostPointer(const void* pointer);

bool CheckCleanupResult(CUresult result, const char* operation) noexcept;

constexpr CUresult kErrorInvalidValue = 1;
constexpr CUresult kErrorDeinitialized = 4;
constexpr CUresult kErrorInvalidContext = 201;
constexpr CUresult kErrorNotFound = 500;
constexpr CUresult kErrorNotSupported = 801;
constexpr CUresult kErrorStreamCaptureUnsupported = 900;
constexpr CUresult kErrorStreamCaptureImplicit = 906;
constexpr int kStreamCaptureStatusNone = 0;
constexpr unsigned long long kGetProcAddressPerThreadDefaultStream = 1ull << 1;
constexpr unsigned kMemHostAllocWriteCombined = 0x04;
constexpr unsigned kMemHostRegisterPortable = 0x01;
constexpr unsigned kMemHostRegisterReadOnly = 0x08;
constexpr unsigned kEventDisableTiming = 0x2;
constexpr unsigned kStreamNonBlocking = 0x1;
constexpr unsigned kPointerAttributeMemoryType = 2;
constexpr unsigned kPointerAttributeIsManaged = 8;
constexpr unsigned kPointerAttributeDeviceOrdinal = 9;	 // int
constexpr unsigned kPointerAttributeRangeStartAddr = 11; // CUdeviceptr
constexpr unsigned kPointerAttributeRangeSize = 12;		 // size_t
constexpr unsigned kPointerAttributeMappingSize = 18;	 // size_t
constexpr unsigned kPointerAttributeMappingBaseAddr = 19; // CUdeviceptr
constexpr unsigned kMemoryTypeHost = 1;
constexpr unsigned kMemoryTypeDevice = 2;
constexpr unsigned kMemoryTypeUnified = 4;
constexpr int kDevAttrMultiprocessorCount = 16;
constexpr int kDevAttrPageableMemoryAccess = 88;
constexpr int kDevAttrReadOnlyHostRegisterSupported = 113;

struct ImageExtent
{
	uintptr_t inside;
	uintptr_t lo;
	uintptr_t hi;
};

ImageExtent FindImageExtent(uintptr_t address);

constexpr int kStreamCaptureModeRelaxed = 2;
class RelaxedCaptureScope
{
public:
	RelaxedCaptureScope() noexcept;
	~RelaxedCaptureScope();
	RelaxedCaptureScope(const RelaxedCaptureScope&) = delete;
	RelaxedCaptureScope& operator=(const RelaxedCaptureScope&) = delete;

private:
	int previous = kStreamCaptureModeRelaxed;
	bool exchanged = false;
};
void RelaxCaptureModeForThisThread() noexcept;

struct DriverApi
{
	CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr, const void*, std::size_t) = nullptr;
	CUresult (*cuMemcpyDtoH_v2)(void*, CUdeviceptr, std::size_t) = nullptr;
	CUresult (*cuMemcpyHtoDAsync_v2)(CUdeviceptr, const void*, std::size_t, CUstream) = nullptr;
	CUresult (*cuMemcpyDtoHAsync_v2)(void*, CUdeviceptr, std::size_t, CUstream) = nullptr;
	CUresult (*cuMemcpy)(CUdeviceptr, CUdeviceptr, std::size_t) = nullptr;
	CUresult (*cuMemcpyAsync)(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) = nullptr;
	CUresult (*cuPointerGetAttribute)(void*, unsigned, CUdeviceptr) = nullptr;
	CUresult (*cuPointerGetAttributes)(unsigned, unsigned*, void**, CUdeviceptr) = nullptr;
	CUresult (*cuMemHostAlloc)(void**, std::size_t, unsigned) = nullptr;
	CUresult (*cuMemAllocHost)(void**, std::size_t) = nullptr;
	CUresult (*cuMemFreeHost)(void*) = nullptr;
	CUresult (*cuMemHostRegister)(void*, std::size_t, unsigned) = nullptr;
	CUresult (*cuMemHostUnregister)(void*) = nullptr;
	CUresult (*cuEventCreate)(CUevent*, unsigned) = nullptr;
	CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
	CUresult (*cuEventSynchronize)(CUevent) = nullptr;
	CUresult (*cuEventDestroy)(CUevent) = nullptr;
	CUresult (*cuCtxGetCurrent)(CUcontext*) = nullptr;
	CUresult (*cuCtxDestroy_v2)(CUcontext) = nullptr;
	CUresult (*cuCtxDestroy)(CUcontext) = nullptr;
	CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
	CUresult (*cuStreamCreate)(CUstream*, unsigned) = nullptr;
	CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned) = nullptr;
	CUresult (*cuStreamDestroy)(CUstream) = nullptr;
	CUresult (*cuCtxGetDevice)(CUdevice*) = nullptr;
	CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
	CUresult (*cuDeviceGetPCIBusId)(char*, int, CUdevice) = nullptr;
	CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
	CUresult (*cuDeviceTotalMem_v2)(std::size_t*, CUdevice) = nullptr;
	CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
	CUresult (*cuStreamIsCapturing)(CUstream, int*) = nullptr;
	CUresult (*cuThreadExchangeStreamCaptureMode)(int*) = nullptr;
	CUresult (*cuCtxSynchronize)() = nullptr;
	CUresult (*cuDevicePrimaryCtxReset)(CUdevice) = nullptr;
	CUresult (*cuDevicePrimaryCtxReset_v2)(CUdevice) = nullptr;
	CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
	CUresult (*cuDevicePrimaryCtxRelease_v2)(CUdevice) = nullptr;
};

extern DriverApi g_api;

using SymbolLookup = void* (*)(const char*);
void ResolveDriverApi(SymbolLookup lookup, bool debug);
[[nodiscard]] cmt::Result<void> InitializeDriverApi();

[[nodiscard]] cmt::Result<cmt::ThreadPlacement>
CurrentWorkerPlacement(const DriverApi& api, const std::vector<unsigned>& explicit_cpus = {},
					   CUcontext expected_context = nullptr, const WorkerPlacement* recorded = nullptr,
					   std::string_view sysfs = "/sys", const GpuCpus& nvml = NvmlGpuCpus);

[[nodiscard]] cmt::Result<cmt::ThreadPlacement>
BackgroundPlacement(const std::vector<unsigned>& explicit_cpus, const WorkerPlacement* recorded,
					std::string_view sysfs = "/sys", const GpuCpus& nvml = {});

CUresult HostAlloc(const DriverApi& api, void** p, std::size_t bytes, unsigned flags) noexcept;

} // namespace cu
} // namespace cmt

namespace cmt {
namespace gpu {
namespace nv {
struct CopyApi
{
	using DevicePtr = cmt::cu::CUdeviceptr;
	using Stream = cmt::cu::CUstream;

	static cmt::cu::CUresult H2D(DevicePtr dst, const void* src, std::size_t bytes, Stream stream, bool async)
	{
		const auto& api = cmt::cu::g_api;
		return async ? api.cuMemcpyHtoDAsync_v2(dst, src, bytes, stream)
					 : api.cuMemcpyHtoD_v2(dst, src, bytes);
	}

	static cmt::cu::CUresult DtoH(void* dst, DevicePtr src, std::size_t bytes, Stream stream, bool async)
	{
		const auto& api = cmt::cu::g_api;
		return async ? api.cuMemcpyDtoHAsync_v2(dst, src, bytes, stream)
					 : api.cuMemcpyDtoH_v2(dst, src, bytes);
	}
};
} // namespace nv
} // namespace gpu
} // namespace cmt
