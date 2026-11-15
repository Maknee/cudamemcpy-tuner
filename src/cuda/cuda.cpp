#include "cuda.h"
#include <algorithm>
#include <dlfcn.h>
#include "common/defines.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "common/log.h"

namespace cmt {

Result<ThreadPlacement> CudaWorkerPlacement(int device, const std::vector<unsigned>& explicit_cpus)
{
	if (!std::empty(explicit_cpus))
	{
		return ThreadPlacement{explicit_cpus, std::nullopt};
	}
	auto request = WorkerPlacementFromEnv();
	if (!request)
	{
		return Failure(request.error());
	}
	return ResolveWorkerPlacement(*request,
								  [&]() -> Result<std::optional<std::string>>
								  {
									  if (device < 0)
									  {
										  CMT_CUDA(cudaGetDevice(&device));
									  }
									  char bus[32] = {};
									  CMT_CUDA(cudaDeviceGetPCIBusId(bus, sizeof(bus), device));
									  return std::optional<std::string>{bus};
								  });
}

namespace {

struct RuntimeResources
{
	static RuntimeResources& Get()
	{
		static RuntimeResources resources;
		return resources;
	}

	std::atomic<long> device{0};
	std::atomic<long> pinned{0};
	std::atomic<long> registered{0};
	std::atomic<long> streams{0};
	std::atomic<long> events{0};
};

} // namespace

void CheckCudaCleanup(cudaError_t status) noexcept
{
	if (status != cudaSuccess)
	{
		Panic("CUDA cleanup failed; resource lifetime cannot be established");
	}
}

long Live::DeviceBuffers() noexcept
{
	return RuntimeResources::Get().device.load();
}

long Live::PinnedAllocs() noexcept
{
	return RuntimeResources::Get().pinned.load();
}

long Live::Registrations() noexcept
{
	return RuntimeResources::Get().registered.load();
}

long Live::Streams() noexcept
{
	return RuntimeResources::Get().streams.load();
}

long Live::Events() noexcept
{
	return RuntimeResources::Get().events.load();
}

Result<int> DeviceCountFrom(cudaError_t status, int count, bool driver_installed)
{
	if (status == cudaErrorNoDevice || status == cudaErrorStubLibrary ||
		(status == cudaErrorInsufficientDriver && !driver_installed))
	{
		return 0;
	}
	if (status != cudaSuccess)
	{
		return Failure(Error::Kind::Cuda, "cudaGetDeviceCount failed", static_cast<int>(status));
	}
	return count;
}

Result<int> CudaDeviceCount()
{
	int count = 0;
	const auto status = cudaGetDeviceCount(&count);
	if (status != cudaSuccess)
	{
		(void)cudaGetLastError();
	}
	bool driver_installed = true;
	if (status == cudaErrorInsufficientDriver)
	{
		void* driver = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
		driver_installed = driver != nullptr;
		if (driver)
		{
			dlclose(driver);
		}
	}
	return DeviceCountFrom(status, count, driver_installed);
}

Result<DeviceBuffer> DeviceBuffer::Create(std::size_t allocation_bytes)
{
	if (allocation_bytes == 0)
	{
		Panic("DeviceBuffer of zero bytes");
	}
	void* allocation_pointer = nullptr;
	const auto status = cudaMalloc(&allocation_pointer, allocation_bytes);
	if (status == cudaErrorMemoryAllocation)
	{
		(void)cudaGetLastError();
		return Failure(Error::Kind::Unsupported, "cudaMalloc failed: insufficient device memory",
					   static_cast<int>(status));
	}
	CMT_RETURN_IF_ERROR(CudaCheck(status, "cudaMalloc failed"));
	DeviceBuffer buffer;
	buffer.pointer = static_cast<unsigned char*>(allocation_pointer);
	buffer.bytes = allocation_bytes;
	++RuntimeResources::Get().device;
	return buffer;
}

void DeviceBuffer::Reset() noexcept
{
	if (pointer)
	{
		CheckCudaCleanup(cudaFree(pointer));
		--RuntimeResources::Get().device;
	}
	pointer = nullptr;
	bytes = 0;
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
	: pointer(std::exchange(other.pointer, nullptr)), bytes(std::exchange(other.bytes, 0))
{
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept
{
	if (this != &other)
	{
		Reset();
		pointer = std::exchange(other.pointer, nullptr);
		bytes = std::exchange(other.bytes, 0);
	}
	return *this;
}

Result<PinnedAlloc> PinnedAlloc::Create(std::size_t allocation_bytes, HostMem kind)
{
	if (allocation_bytes == 0)
	{
		Panic("PinnedAlloc of zero bytes");
	}
	void* allocation_pointer = nullptr;
	CMT_CUDA(
		cudaHostAlloc(&allocation_pointer, allocation_bytes,
					  kind == HostMem::WriteCombined ? cudaHostAllocWriteCombined : cudaHostAllocDefault));
	PinnedAlloc allocation;
	allocation.pointer = static_cast<unsigned char*>(allocation_pointer);
	allocation.bytes = allocation_bytes;
	++RuntimeResources::Get().pinned;
	return allocation;
}

void PinnedAlloc::Reset() noexcept
{
	if (pointer)
	{
		CheckCudaCleanup(cudaFreeHost(pointer));
		--RuntimeResources::Get().pinned;
	}
	pointer = nullptr;
	bytes = 0;
}

PinnedAlloc::PinnedAlloc(PinnedAlloc&& other) noexcept
	: pointer(std::exchange(other.pointer, nullptr)), bytes(std::exchange(other.bytes, 0))
{
}

PinnedAlloc& PinnedAlloc::operator=(PinnedAlloc&& other) noexcept
{
	if (this != &other)
	{
		Reset();
		pointer = std::exchange(other.pointer, nullptr);
		bytes = std::exchange(other.bytes, 0);
	}
	return *this;
}

Result<HostRegistration> HostRegistration::Create(void* allocation_pointer, std::size_t allocation_bytes)
{
	if (!allocation_pointer || allocation_bytes == 0)
	{
		Panic("HostRegistration of an empty range");
	}
	CMT_CUDA(cudaHostRegister(allocation_pointer, allocation_bytes, cudaHostRegisterDefault));
	HostRegistration registration;
	registration.pointer = allocation_pointer;
	++RuntimeResources::Get().registered;
	return registration;
}

Result<void> HostRegistration::Release()
{
	if (!pointer)
	{
		return {};
	}
	CMT_CUDA(cudaHostUnregister(pointer));
	pointer = nullptr;
	--RuntimeResources::Get().registered;
	return {};
}

void HostRegistration::Reset() noexcept
{
	if (pointer)
	{
		CheckCudaCleanup(cudaHostUnregister(pointer));
		--RuntimeResources::Get().registered;
	}
	pointer = nullptr;
}

HostRegistration::HostRegistration(HostRegistration&& other) noexcept
	: pointer(std::exchange(other.pointer, nullptr))
{
}

HostRegistration& HostRegistration::operator=(HostRegistration&& other) noexcept
{
	if (this != &other)
	{
		Reset();
		pointer = std::exchange(other.pointer, nullptr);
	}
	return *this;
}

Result<Stream> Stream::Create()
{
	cudaStream_t handle = nullptr;
	CMT_CUDA(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking));
	Stream created;
	created.stream = handle;
	++RuntimeResources::Get().streams;
	return created;
}

void Stream::Reset() noexcept
{
	if (stream)
	{
		CheckCudaCleanup(cudaStreamSynchronize(stream));
		CheckCudaCleanup(cudaStreamDestroy(stream));
		--RuntimeResources::Get().streams;
	}
	stream = nullptr;
}

Stream::Stream(Stream&& other) noexcept : stream(std::exchange(other.stream, nullptr))
{
}

Stream& Stream::operator=(Stream&& other) noexcept
{
	if (this != &other)
	{
		Reset();
		stream = std::exchange(other.stream, nullptr);
	}
	return *this;
}

Result<void> Stream::Sync() const
{
	return CudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

Result<Event> Event::Create(Timing timing)
{
	cudaEvent_t handle = nullptr;
	CMT_CUDA(cudaEventCreateWithFlags(&handle,
									  timing == Timing::Enabled ? cudaEventDefault : cudaEventDisableTiming));
	Event created;
	created.event = handle;
	++RuntimeResources::Get().events;
	return created;
}

void Event::Reset() noexcept
{
	if (event)
	{
		CheckCudaCleanup(cudaEventDestroy(event));
		--RuntimeResources::Get().events;
	}
	event = nullptr;
}

Event::Event(Event&& other) noexcept : event(std::exchange(other.event, nullptr))
{
}

Event& Event::operator=(Event&& other) noexcept
{
	if (this != &other)
	{
		Reset();
		event = std::exchange(other.event, nullptr);
	}
	return *this;
}

Result<void> Event::Record(const Stream& stream) const
{
	return CudaCheck(cudaEventRecord(event, stream.Get()), "cudaEventRecord");
}

Result<void> Event::Sync() const
{
	return CudaCheck(cudaEventSynchronize(event), "cudaEventSynchronize");
}

Result<double> Event::ElapsedMs(const Event& start, const Event& end)
{
	float milliseconds = 0;
	CMT_CUDA(cudaEventElapsedTime(&milliseconds, start.event, end.event));
	return milliseconds;
}

namespace {

constexpr std::size_t kPatternPieceBytes = std::size_t{64} << 20;

template <class F>
Result<void> PatternPieces(Dir dir, const unsigned char* d, std::size_t bytes, cudaStream_t s, F words)
{
	const std::size_t total = bytes / 8 * 8;
	if (total == 0)
	{
		return {};
	}
	auto created = PinnedAlloc::Create(std::min(total, kPatternPieceBytes), HostMem::Cached);
	if (!created)
	{
		return Failure(std::move(created.error()));
	}
	const PinnedAlloc host = std::move(*created);
	auto* w = reinterpret_cast<std::uint64_t*>(std::data(host));
	for (std::size_t at = 0; at < total; at += std::size(host))
	{
		const std::size_t n = std::min(std::size(host), total - at);
		if (dir == Dir::HtoD)
		{
			words(w, at / 8, n / 8);
			CMT_CUDA(cudaMemcpyAsync(const_cast<unsigned char*>(d + at), w, n, cudaMemcpyHostToDevice, s));
		}
		else
		{
			CMT_CUDA(cudaMemcpyAsync(w, d + at, n, cudaMemcpyDeviceToHost, s));
		}
		CMT_CUDA(cudaStreamSynchronize(s));
		if (dir == Dir::DtoH)
		{
			words(w, at / 8, n / 8);
		}
	}
	return {};
}

} // namespace

Result<void> LaunchPatternFill(void* d, std::size_t bytes, std::uint64_t seed, cudaStream_t s)
{
	return PatternPieces(Dir::HtoD, static_cast<const unsigned char*>(d), bytes, s,
						 [seed](std::uint64_t* w, std::size_t first, std::size_t n)
						 {
							 for (std::size_t i = 0; i < n; i++)
							 {
								 w[i] = PatternWord(first + i, seed);
							 }
						 });
}

Result<std::size_t> DevicePatternCheck(const void* d, std::size_t bytes, std::uint64_t seed, cudaStream_t s)
{
	std::size_t bad = 0;
	CMT_RETURN_IF_ERROR(PatternPieces(Dir::DtoH, static_cast<const unsigned char*>(d), bytes, s,
									  [seed, &bad](const std::uint64_t* w, std::size_t first, std::size_t n)
									  {
										  for (std::size_t i = 0; i < n; i++)
										  {
											  bad += w[i] != PatternWord(first + i, seed);
										  }
									  }));
	return bad;
}

} // namespace cmt
