#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/error.h"

namespace cmt {

struct ThreadPlacement;

template <std::size_t N>
[[nodiscard]] Result<void> CudaCheck(cudaError_t status, const char (&operation)[N])
{
	if (status == cudaSuccess)
	{
		return {};
	}
	(void)cudaGetLastError();
	return Failure(Error::Kind::Cuda, operation, static_cast<int>(status));
}

[[nodiscard]] Result<int> CudaDeviceCount();
[[nodiscard]] Result<int> DeviceCountFrom(cudaError_t status, int count, bool driver_installed);
[[nodiscard]] Result<ThreadPlacement> CudaWorkerPlacement(int device = -1,
														  const std::vector<unsigned>& explicit_cpus = {});

void CheckCudaCleanup(cudaError_t status) noexcept;

#define CMT_CUDA(call)                                                                                       \
	do                                                                                                       \
	{                                                                                                        \
		auto cuda_result = ::cmt::CudaCheck((call), "CUDA call failed: " #call);                             \
		if (!cuda_result)                                                                                    \
		{                                                                                                    \
			return cmt::Failure(std::move(cuda_result.error()));                                             \
		}                                                                                                    \
	} while (false)

struct Live
{
	static long DeviceBuffers() noexcept;
	static long PinnedAllocs() noexcept;
	static long Registrations() noexcept;
	static long Streams() noexcept;
	static long Events() noexcept;
};

class DeviceBuffer
{
public:
	DeviceBuffer() = default;
	[[nodiscard]] static Result<DeviceBuffer> Create(std::size_t allocation_bytes);

	~DeviceBuffer()
	{
		Reset();
	}

	DeviceBuffer(DeviceBuffer&& other) noexcept;
	DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;

	unsigned char* data() const noexcept
	{
		return pointer;
	}

	std::size_t size() const noexcept
	{
		return bytes;
	}

private:
	void Reset() noexcept;
	unsigned char* pointer = nullptr;
	std::size_t bytes = 0;
};

enum class HostMem : std::uint8_t
{
	Cached,
	WriteCombined
};

class PinnedAlloc
{
public:
	PinnedAlloc() = default;
	[[nodiscard]] static Result<PinnedAlloc> Create(std::size_t allocation_bytes, HostMem kind);

	~PinnedAlloc()
	{
		Reset();
	}

	PinnedAlloc(PinnedAlloc&& other) noexcept;
	PinnedAlloc& operator=(PinnedAlloc&& other) noexcept;
	PinnedAlloc(const PinnedAlloc&) = delete;
	PinnedAlloc& operator=(const PinnedAlloc&) = delete;

	unsigned char* data() const noexcept
	{
		return pointer;
	}

	std::size_t size() const noexcept
	{
		return bytes;
	}

private:
	void Reset() noexcept;
	unsigned char* pointer = nullptr;
	std::size_t bytes = 0;
};

class HostRegistration
{
public:
	HostRegistration() = default;
	[[nodiscard]] static Result<HostRegistration> Create(void* allocation_pointer,
														 std::size_t allocation_bytes);

	~HostRegistration()
	{
		Reset();
	}

	HostRegistration(HostRegistration&& other) noexcept;
	HostRegistration& operator=(HostRegistration&& other) noexcept;
	HostRegistration(const HostRegistration&) = delete;
	HostRegistration& operator=(const HostRegistration&) = delete;

	[[nodiscard]] Result<void> Release();

	bool Active() const noexcept
	{
		return pointer != nullptr;
	}

private:
	void Reset() noexcept;
	void* pointer = nullptr;
};

class Stream
{
public:
	Stream() = default;
	[[nodiscard]] static Result<Stream> Create();

	~Stream()
	{
		Reset();
	}

	Stream(Stream&& other) noexcept;
	Stream& operator=(Stream&& other) noexcept;
	Stream(const Stream&) = delete;
	Stream& operator=(const Stream&) = delete;

	cudaStream_t Get() const noexcept
	{
		return stream;
	}

	[[nodiscard]] Result<void> Sync() const;

private:
	void Reset() noexcept;
	cudaStream_t stream = nullptr;
};

struct DrainOnExit
{
	const Stream& s;

	~DrainOnExit()
	{
		CheckCudaCleanup(cudaStreamSynchronize(s.Get()));
	}
};

enum class Timing : std::uint8_t
{
	Enabled,
	Disabled
};

class Event
{
public:
	Event() = default;
	[[nodiscard]] static Result<Event> Create(Timing timing);

	~Event()
	{
		Reset();
	}

	Event(Event&& other) noexcept;
	Event& operator=(Event&& other) noexcept;
	Event(const Event&) = delete;
	Event& operator=(const Event&) = delete;

	cudaEvent_t Get() const noexcept
	{
		return event;
	}

	[[nodiscard]] Result<void> Record(const Stream& stream) const;
	[[nodiscard]] Result<void> Sync() const;
	[[nodiscard]] static Result<double> ElapsedMs(const Event& start, const Event& end);

private:
	void Reset() noexcept;
	cudaEvent_t event = nullptr;
};

[[nodiscard]] Result<void> LaunchPatternFill(void* d, std::size_t bytes, std::uint64_t seed, cudaStream_t s);
[[nodiscard]] Result<std::size_t> DevicePatternCheck(const void* d, std::size_t bytes, std::uint64_t seed,
													 cudaStream_t s);

} // namespace cmt

namespace cmt {
namespace gpu {

struct RuntimeApi
{
	using DevicePtr = unsigned char*;
	using Stream = cudaStream_t;

	static cudaError_t H2D(DevicePtr destination, const void* source, std::size_t bytes, Stream stream,
						   bool async)
	{
		return async ? cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream)
					 : cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);
	}

	static cudaError_t DtoH(void* destination, DevicePtr source, std::size_t bytes, Stream stream, bool async)
	{
		return async ? cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, stream)
					 : cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);
	}
};

} // namespace gpu
} // namespace cmt
