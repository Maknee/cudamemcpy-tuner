#include "driver.h"

namespace cmt {
namespace cu {

DriverApi g_api;

namespace {
int LocateImageExtent(struct dl_phdr_info* info, std::size_t, void* data)
{
	auto* x = static_cast<ImageExtent*>(data);
	uintptr_t lo = UINTPTR_MAX;
	uintptr_t hi = 0;
	for (ElfW(Half) i = 0; i < info->dlpi_phnum; i++)
	{
		if (info->dlpi_phdr[i].p_type == PT_LOAD)
		{
			const uintptr_t s = info->dlpi_addr + info->dlpi_phdr[i].p_vaddr;
			lo = s < lo ? s : lo;
			hi = s + info->dlpi_phdr[i].p_memsz > hi ? s + info->dlpi_phdr[i].p_memsz : hi;
		}
	}
	if (x->inside < lo || x->inside >= hi)
	{
		return 0;
	}
	x->lo = lo;
	x->hi = hi;
	return 1;
}

} // namespace

ImageExtent FindImageExtent(uintptr_t address)
{
	ImageExtent extent{address, 0, 0};
	(void)dl_iterate_phdr(&LocateImageExtent, &extent);
	return extent;
}

bool IsCapturing(CUstream stream)
{
	if (g_api.cuStreamIsCapturing == nullptr)
	{
		return false;
	}
	int status = kStreamCaptureStatusNone;
	const CUresult rc = g_api.cuStreamIsCapturing(stream, &status);
	return rc != kSuccess || status != kStreamCaptureStatusNone;
}

HostPointerKind ClassifyHostPointer(const void* pointer)
{
	if (pointer == nullptr)
	{
		return HostPointerKind::Unknown;
	}
	const auto address = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(pointer));
	unsigned memory_type = 0;
	unsigned managed = 0;
	if (g_api.cuPointerGetAttributes != nullptr)
	{
		unsigned attributes[] = {kPointerAttributeMemoryType, kPointerAttributeIsManaged};
		void* values[] = {&memory_type, &managed};
		if (g_api.cuPointerGetAttributes(std::size(attributes), attributes, values, address) != kSuccess)
		{
			return HostPointerKind::Unknown;
		}
		if (memory_type == 0 && managed == 0)
		{
			return HostPointerKind::Pageable;
		}
	}
	else if (g_api.cuPointerGetAttribute == nullptr ||
			 g_api.cuPointerGetAttribute(&memory_type, kPointerAttributeMemoryType, address) != kSuccess)
	{
		return HostPointerKind::Unknown;
	}
	if (managed != 0 || memory_type == kMemoryTypeUnified)
	{
		return HostPointerKind::Managed;
	}
	if (memory_type == kMemoryTypeHost)
	{
		return HostPointerKind::Pinned;
	}
	return memory_type == kMemoryTypeDevice ? HostPointerKind::Device : HostPointerKind::Unknown;
}

bool CheckCleanupResult(CUresult result, const char* operation) noexcept
{
	if (result == kSuccess)
	{
		return true;
	}
	LogLine("error: hook cleanup {} failed rc={}; the resource is retained (leaked), not released", operation,
			result);
	return false;
}

RelaxedCaptureScope::RelaxedCaptureScope() noexcept
{
	if (g_api.cuThreadExchangeStreamCaptureMode != nullptr)
	{
		previous = kStreamCaptureModeRelaxed;
		exchanged = g_api.cuThreadExchangeStreamCaptureMode(&previous) == kSuccess;
	}
}

RelaxedCaptureScope::~RelaxedCaptureScope()
{
	if (exchanged && previous != kStreamCaptureModeRelaxed)
	{
		(void)g_api.cuThreadExchangeStreamCaptureMode(&previous);
	}
}

void RelaxCaptureModeForThisThread() noexcept
{
	if (g_api.cuThreadExchangeStreamCaptureMode != nullptr)
	{
		int mode = kStreamCaptureModeRelaxed;
		(void)g_api.cuThreadExchangeStreamCaptureMode(&mode);
	}
}

namespace {

template <class Fn>
void Resolve(SymbolLookup lookup, bool debug, Fn* slot, const char* name)
{
	void* fn = lookup(name);
	if (debug && fn == nullptr)
	{
		cmt::logging::WriteLine(STDERR_FILENO, "[memcpy_hook] ", "missing libcuda symbol {}", name);
	}
	*slot = reinterpret_cast<Fn>(fn);
}

template <class Fn>
void ResolveFallback(SymbolLookup lookup, bool debug, Fn* slot, const char* name, const char* fallback)
{
	void* fn = lookup(name);
	if (fn == nullptr)
	{
		fn = lookup(fallback);
	}
	if (debug && fn == nullptr)
	{
		cmt::logging::WriteLine(STDERR_FILENO, "[memcpy_hook] ", "missing libcuda symbol {} (and {})", name,
								fallback);
	}
	*slot = reinterpret_cast<Fn>(fn);
}

} // namespace

void ResolveDriverApi(SymbolLookup lookup, bool debug)
{
	DriverApi& a = g_api;
	Resolve(lookup, debug, &a.cuMemcpyHtoD_v2, "cuMemcpyHtoD_v2");
	Resolve(lookup, debug, &a.cuMemcpyDtoH_v2, "cuMemcpyDtoH_v2");
	Resolve(lookup, debug, &a.cuMemcpyHtoDAsync_v2, "cuMemcpyHtoDAsync_v2");
	Resolve(lookup, debug, &a.cuMemcpyDtoHAsync_v2, "cuMemcpyDtoHAsync_v2");
	Resolve(lookup, debug, &a.cuMemcpy, "cuMemcpy");
	Resolve(lookup, debug, &a.cuMemcpyAsync, "cuMemcpyAsync");
	Resolve(lookup, debug, &a.cuPointerGetAttribute, "cuPointerGetAttribute");
	Resolve(lookup, debug, &a.cuPointerGetAttributes, "cuPointerGetAttributes");
	Resolve(lookup, debug, &a.cuMemHostAlloc, "cuMemHostAlloc");
	Resolve(lookup, debug, &a.cuMemAllocHost, "cuMemAllocHost_v2");
	Resolve(lookup, debug, &a.cuMemFreeHost, "cuMemFreeHost");
	ResolveFallback(lookup, debug, &a.cuMemHostRegister, "cuMemHostRegister_v2", "cuMemHostRegister");
	Resolve(lookup, debug, &a.cuMemHostUnregister, "cuMemHostUnregister");
	Resolve(lookup, debug, &a.cuCtxSynchronize, "cuCtxSynchronize");
	ResolveFallback(lookup, debug, &a.cuEventCreate, "cuEventCreate_v2", "cuEventCreate");
	Resolve(lookup, debug, &a.cuEventRecord, "cuEventRecord");
	Resolve(lookup, debug, &a.cuEventSynchronize, "cuEventSynchronize");
	ResolveFallback(lookup, debug, &a.cuEventDestroy, "cuEventDestroy_v2", "cuEventDestroy");
	Resolve(lookup, debug, &a.cuCtxGetCurrent, "cuCtxGetCurrent");
	ResolveFallback(lookup, debug, &a.cuCtxDestroy_v2, "cuCtxDestroy_v2", "cuCtxDestroy");
	Resolve(lookup, debug, &a.cuCtxDestroy, "cuCtxDestroy");
	Resolve(lookup, debug, &a.cuCtxSetCurrent, "cuCtxSetCurrent");
	Resolve(lookup, debug, &a.cuStreamCreate, "cuStreamCreate");
	Resolve(lookup, debug, &a.cuStreamWaitEvent, "cuStreamWaitEvent");
	ResolveFallback(lookup, debug, &a.cuStreamDestroy, "cuStreamDestroy_v2", "cuStreamDestroy");
	Resolve(lookup, debug, &a.cuCtxGetDevice, "cuCtxGetDevice");
	Resolve(lookup, debug, &a.cuDeviceGetName, "cuDeviceGetName");
	Resolve(lookup, debug, &a.cuDeviceGetPCIBusId, "cuDeviceGetPCIBusId");
	Resolve(lookup, debug, &a.cuDeviceGetAttribute, "cuDeviceGetAttribute");
	Resolve(lookup, debug, &a.cuDeviceTotalMem_v2, "cuDeviceTotalMem_v2");
	Resolve(lookup, debug, &a.cuStreamSynchronize, "cuStreamSynchronize");
	Resolve(lookup, debug, &a.cuStreamIsCapturing, "cuStreamIsCapturing");
	Resolve(lookup, debug, &a.cuThreadExchangeStreamCaptureMode, "cuThreadExchangeStreamCaptureMode");
	Resolve(lookup, debug, &a.cuDevicePrimaryCtxReset, "cuDevicePrimaryCtxReset");
	Resolve(lookup, debug, &a.cuDevicePrimaryCtxReset_v2, "cuDevicePrimaryCtxReset_v2");
	Resolve(lookup, debug, &a.cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease");
	Resolve(lookup, debug, &a.cuDevicePrimaryCtxRelease_v2, "cuDevicePrimaryCtxRelease_v2");
}

cmt::Result<void> InitializeDriverApi()
{
	static std::once_flag once;
	static cmt::Result<void> initialized;
	static void* library = nullptr;
	std::call_once(once,
				   []
				   {
					   library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
					   if (library == nullptr)
					   {
						   initialized = cmt::Failure(cmt::Error::Kind::Unsupported, "cannot load libcuda");
						   return;
					   }
					   ResolveDriverApi([](const char* name) { return dlsym(library, name); }, false);
					   if (g_api.cuCtxGetCurrent == nullptr || g_api.cuMemcpyHtoDAsync_v2 == nullptr ||
						   g_api.cuMemcpyDtoHAsync_v2 == nullptr)
					   {
						   initialized = cmt::Failure(cmt::Error::Kind::Unsupported,
													  "libcuda lacks required copy APIs");
					   }
				   });
	return initialized;
}

cmt::Result<cmt::ThreadPlacement> CurrentWorkerPlacement(const DriverApi& api,
														 const std::vector<unsigned>& explicit_cpus,
														 CUcontext expected_context,
														 const WorkerPlacement* recorded,
														 std::string_view sysfs, const GpuCpus& nvml)
{
	if (!std::empty(explicit_cpus))
	{
		return cmt::ThreadPlacement{explicit_cpus, std::nullopt};
	}
	auto request = recorded ? Result<WorkerPlacement>{*recorded} : WorkerPlacementFromEnv();
	if (!request)
	{
		return cmt::Failure(request.error());
	}
	const auto gpu_pci = [&]() -> Result<std::optional<std::string>>
	{
		if (!api.cuCtxGetCurrent || !api.cuCtxGetDevice || !api.cuDeviceGetPCIBusId)
		{
			return std::optional<std::string>{};
		}
		CUcontext context = nullptr;
		auto status = api.cuCtxGetCurrent(&context);
		if (status != kSuccess)
		{
			return cmt::Failure(cmt::Error::Kind::Cuda, "Cannot query CUDA context for worker placement",
								status);
		}
		if (expected_context && context != expected_context)
		{
			return cmt::Failure(cmt::Error::Kind::Usage,
								"Worker placement requires the pipeline's CUDA context to be current");
		}
		if (!context)
		{
			return std::optional<std::string>{};
		}
		CUdevice device;
		status = api.cuCtxGetDevice(&device);
		if (status != kSuccess)
		{
			return cmt::Failure(cmt::Error::Kind::Cuda, "Cannot query CUDA device for worker placement",
								status);
		}
		char bus[32] = {};
		status = api.cuDeviceGetPCIBusId(bus, sizeof(bus), device);
		if (status != kSuccess)
		{
			return cmt::Failure(cmt::Error::Kind::Cuda, "Cannot query CUDA PCI address for worker placement",
								status);
		}
		if (!std::memchr(bus, '\0', sizeof(bus)))
		{
			return cmt::Failure(cmt::Error::Kind::Usage, "CUDA PCI address is not terminated");
		}
		return std::optional<std::string>{bus};
	};
	return ResolveWorkerPlacement(*request, gpu_pci, sysfs, nvml);
}

cmt::Result<cmt::ThreadPlacement> BackgroundPlacement(const std::vector<unsigned>& explicit_cpus,
													  const WorkerPlacement* recorded, std::string_view sysfs,
													  const GpuCpus& nvml)
{
	auto request = recorded ? Result<WorkerPlacement>{*recorded} : WorkerPlacementFromEnv();
	if (!request)
	{
		return cmt::Failure(request.error());
	}
	request->gpu_numa = false;
	return CurrentWorkerPlacement(g_api, explicit_cpus, nullptr, &*request, sysfs, nvml);
}

CUresult HostAlloc(const DriverApi& api, void** p, std::size_t bytes, unsigned flags) noexcept
{
	if (api.cuMemHostAlloc != nullptr)
	{
		return api.cuMemHostAlloc(p, bytes, flags);
	}
	if (api.cuMemAllocHost == nullptr)
	{
		return kErrorNotFound;
	}
	if (flags != 0)
	{
		return kErrorNotSupported;
	}
	return api.cuMemAllocHost(p, bytes);
}

} // namespace cu
} // namespace cmt
