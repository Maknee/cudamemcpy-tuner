#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <vector>

#include "common/memory.h"
#include "common/cpu.h"
#include "copying/chunk_pipeline.h"
#include "cuda/cuda.h"

// This executable measures ChunkPipeline directly. Registration-cache fallback
// and allocator interception must never participate in this component benchmark.
namespace cmt::copying::services {
void ReleaseAllocation(void*)
{
	cmt::Panic("registration-cache release reached the standalone pipeline benchmark");
}

cu::CUresult StagingCopy(bool, cu::CUdeviceptr, uint8_t*, std::size_t, cu::CUstream)
{
	cmt::Panic("registration-cache fallback reached the standalone pipeline benchmark");
}
} // namespace cmt::copying::services

// Component measurement: call the hook's persistent pipeline directly. No
// interception, registration cache, automatic tuning or internal driver patch.
// Every timed transfer includes caller-stream completion in both directions.
namespace {
using cmt::CopySettings;
using cmt::Result;
using cmt::ring::ChunkPipeline;
enum class Path
{
	ReusedPipeline,
	LifecyclePipeline,
	PinnedDma,
	PageableDriver
};
bool failed = false;
void* driver = nullptr;

void* lookup(const char* name)
{
	return dlsym(driver, name);
}

template <std::size_t N>
Result<void> CheckDriver(cmt::cu::CUresult status, const char (&operation)[N])
{
	if (status != cmt::cu::kSuccess)
	{
		return cmt::Failure(cmt::Error::Kind::Cuda, operation, status);
	}
	return {};
}

Result<std::unique_ptr<ChunkPipeline>> MakePipeline(const CopySettings& knobs)
{
	cmt::cu::CUcontext context = nullptr;
	CMT_RETURN_IF_ERROR(CheckDriver(cmt::cu::g_api.cuCtxGetCurrent(&context), "cuCtxGetCurrent"));
	if (!context)
	{
		return cmt::Failure(cmt::Error::Kind::Verify, "no current CUDA context");
	}
	cmt::HookConfig config;
	config.slot_alloc =
		cmt::StagingAllocation::CachedPinned; // CPU-cached pinned staging, unchanged hook default
	const cmt::ring::PipelineCapacity shape{knobs.workers, knobs.slots,
											cmt::ring::RequiredPinnedBytes(knobs)};
	auto placement = cmt::cu::CurrentWorkerPlacement(cmt::cu::g_api, {}, context);
	if (!placement)
	{
		return cmt::Failure(placement.error());
	}
	auto pipeline = ChunkPipeline::Create(context, config, shape, *placement);
	if (!pipeline)
	{
		return cmt::Failure(std::move(pipeline.error()));
	}
	CMT_RETURN_IF_ERROR((*pipeline)->Start(*placement));
	return std::move(*pipeline);
}

Result<void> Measure(benchmark::State& state, Path path, bool htod)
{
	const auto bytes = static_cast<std::size_t>(state.range(0));
	const CopySettings knobs{static_cast<unsigned>(state.range(1)),
							 static_cast<uint32_t>(state.range(2)),
							 static_cast<uint64_t>(state.range(3)),
							 {cmt::VariantId::Memcpy, {}}};
	const bool use_pipeline = path == Path::ReusedPipeline || path == Path::LifecyclePipeline;
	cmt::ring::CopyObservation observed;
	{
		constexpr std::size_t guard = 64;
		std::vector<uint8_t> expected(bytes + 2 * guard, 0xa5);
		for (std::size_t i = 0; i < bytes; ++i)
		{
			expected[guard + i] = static_cast<uint8_t>((i * 31) ^ (i >> 8) ^ (i >> 16));
		}
		std::vector<uint8_t> pageable(expected.size(), 0xa5), readback(expected.size());
		cmt::PinnedAlloc pinned;
		if (path == Path::PinnedDma)
		{
			auto allocated = cmt::PinnedAlloc::Create(expected.size(), cmt::HostMem::Cached);
			if (!allocated)
			{
				return cmt::Failure(std::move(allocated.error()));
			}
			pinned = std::move(*allocated);
		}
		auto* host = path == Path::PinnedDma ? pinned.data() : pageable.data();
		std::memcpy(host, expected.data(), expected.size());
		auto allocated_device = cmt::DeviceBuffer::Create(expected.size());
		if (!allocated_device)
		{
			return cmt::Failure(std::move(allocated_device.error()));
		}
		auto device = std::move(*allocated_device);
		CMT_CUDA(cudaMemcpy(device.data(), expected.data(), expected.size(), cudaMemcpyHostToDevice));
		// Destroy pipeline first, then drain/destroy the caller stream, then
		// release host/device memory. The CUDA context outlives this whole scope.
		auto created_stream = cmt::Stream::Create();
		if (!created_stream)
		{
			return cmt::Failure(std::move(created_stream.error()));
		}
		auto stream = std::move(*created_stream);
		std::unique_ptr<ChunkPipeline> pipeline;
		if (use_pipeline)
		{
			auto created = MakePipeline(knobs);
			if (!created)
			{
				return cmt::Failure(std::move(created.error()));
			}
			pipeline = std::move(*created);
		}
		const auto device_range = reinterpret_cast<cmt::cu::CUdeviceptr>(device.data() + guard);
		auto transfer = [&](cmt::ring::CopyObservation* observation) -> Result<void>
		{
			if (use_pipeline)
			{
				const auto status =
					htod
						? pipeline->HtoD(knobs, device_range, host + guard, bytes, stream.Get(), observation)
						: pipeline->DtoH(knobs, host + guard, device_range, bytes, stream.Get(), observation);
				CMT_RETURN_IF_ERROR(CheckDriver(status, "ChunkPipeline copy"));
			}
			else
			{
				CMT_CUDA(cudaMemcpyAsync(
					htod ? device.data() + guard : host + guard, htod ? host + guard : device.data() + guard,
					bytes, htod ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, stream.Get()));
			}
			return stream.Sync(); // HtoD submission alone is not a complete-copy sample.
		};
		auto verify = [&]() -> Result<void>
		{
			if (htod)
			{
				CMT_CUDA(cudaMemcpy(readback.data(), device.data(), expected.size(), cudaMemcpyDeviceToHost));
				if (readback != expected)
				{
					return cmt::Failure(cmt::Error::Kind::Verify, "HtoD byte or guard verification failed");
				}
			}
			else if (std::memcmp(host, expected.data(), expected.size()) != 0)
			{
				return cmt::Failure(cmt::Error::Kind::Verify, "DtoH byte or guard verification failed");
			}
			return {};
		};
		// Changed source and poisoned destination detect skipped/stale jobs.
		// Checks and preparation are outside timing. Timed copies reuse memory
		// without cache flushing; these are not cold-page measurements.
		for (unsigned round = 0; round < 2; ++round)
		{
			for (std::size_t i = 0; i < bytes; ++i)
			{
				expected[guard + i] ^= 0x3d;
			}
			if (htod)
			{
				std::memcpy(host, expected.data(), expected.size());
				CMT_CUDA(cudaMemset(device.data() + guard, 0, bytes));
			}
			else
			{
				CMT_CUDA(cudaMemcpy(device.data(), expected.data(), expected.size(), cudaMemcpyHostToDevice));
				std::memset(host + guard, 0, bytes);
			}
			// cudaMemset on the default stream must finish before the independent stream starts.
			CMT_CUDA(cudaDeviceSynchronize());
			CMT_RETURN_IF_ERROR(transfer(&observed));
			CMT_RETURN_IF_ERROR(verify());
		}
		if (use_pipeline && (observed.completed != 1 || observed.actual_workers == 0))
		{
			return cmt::Failure(cmt::Error::Kind::Verify, "pipeline did not confirm the requested route");
		}
		if (path == Path::LifecyclePipeline)
		{
			pipeline.reset();
		}
		for (auto iteration : state)
		{
			(void)iteration;
			if (path == Path::LifecyclePipeline)
			{
				auto created = MakePipeline(knobs);
				if (!created)
				{
					return cmt::Failure(std::move(created.error()));
				}
				pipeline = std::move(*created);
			}
			CMT_RETURN_IF_ERROR(transfer(nullptr));
			if (path == Path::LifecyclePipeline)
			{
				pipeline.reset();
			}
			benchmark::ClobberMemory();
		}
		CMT_RETURN_IF_ERROR(verify());
	}
	state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(bytes));
	state.counters["pool_bytes"] =
		use_pipeline ? static_cast<double>(cmt::ring::RequiredPinnedBytes(knobs)) : 0;
	state.counters["actual_workers"] = static_cast<double>(observed.actual_workers);
	state.counters["max_slots_used_per_worker"] = static_cast<double>(observed.max_slots_used);
	return {};
}

void MeasureCallback(benchmark::State& state, Path path, bool htod)
{
	auto result = Measure(state, path, htod);
	if (!result)
	{
		state.SkipWithError(result.error().Message());
		failed = true;
	}
}

// Registration calls reaching the stubs below. This mode loads no libcuda, so g_api would be empty and
// AllocatePinnedPool would answer Unsupported before it looks at the size; the stubs make the registered
// policies available and record any registration attempt.
long overflow_registrations = 0;

cmt::cu::CUresult CountRegister(void*, std::size_t, unsigned)
{
	overflow_registrations++;
	return cmt::cu::kErrorNotSupported;
}

cmt::cu::CUresult CountUnregister(void*)
{
	overflow_registrations++;
	return cmt::cu::kErrorNotSupported;
}

Result<int> CheckAllocationOverflow()
{
	cmt::cu::g_api.cuMemHostRegister = CountRegister;
	cmt::cu::g_api.cuMemHostUnregister = CountUnregister;
	constexpr auto max_bytes = std::numeric_limits<std::size_t>::max();
	for (const auto allocation : {cmt::StagingAllocation::Registered4K})
	{
		// MapPinnedPool's alignment for this policy (ThpBytes() is the base page without THP geometry).
		const std::size_t alignment =
			cmt::BasePageBytes();
		// SIZE_MAX overflows the round-up. SIZE_MAX - (alignment - 1) is aligned already, so it rounds to
		// itself and only the extra `alignment` mapped to align the base overflows.
		for (const auto bytes : {max_bytes, max_bytes - (alignment - 1)})
		{
			cmt::ring::PinnedPool slot;
			const auto result = cmt::ring::AllocatePinnedPool(slot, bytes, allocation);
			if (result || !result.error().IsUsage() || slot.staging != nullptr || slot.mapping != nullptr ||
				slot.mapping_len != 0 || overflow_registrations != 0)
			{
				std::fprintf(stderr,
							 "allocation overflow: allocation=%u bytes=%zu, result=%s, mapping_bytes=%zu, "
							 "registrations=%ld\n",
							 static_cast<unsigned>(allocation), bytes,
							 result ? "unexpected success" : result.error().Message(), slot.mapping_len,
							 overflow_registrations);
				return cmt::Failure(cmt::Error::Kind::Verify, "staging allocation overflow check failed");
			}
		}
	}
	return 0;
}
} // namespace

Result<int> RunMain(int argc, char** argv)
{
	if (argc == 2 && std::strcmp(argv[1], "--check-allocation-overflow") == 0)
	{
		return CheckAllocationOverflow();
	}
	benchmark::Initialize(&argc, argv);
	if (benchmark::ReportUnrecognizedArguments(argc, argv))
	{
		return 2;
	}
	// Even an unrelated preload can change allocation/copy timing. Keep this
	// executable a direct component measurement; hook comparisons belong in evaluation/.
	if (const char* preload = std::getenv("LD_PRELOAD"); preload && *preload)
	{
		return cmt::Failure(cmt::Error::Kind::Verify, "run component benchmarks with LD_PRELOAD unset");
	}
	driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!driver)
	{
		const char* detail = dlerror();
		std::fprintf(stderr, "libcuda load failed: %s\n", detail ? detail : "unknown loader error");
		return cmt::Failure(cmt::Error::Kind::Verify, "cannot load libcuda");
	}
	cmt::cu::ResolveDriverApi(lookup, false);
	if (!cmt::cu::g_api.cuCtxGetCurrent)
	{
		return cmt::Failure(cmt::Error::Kind::Verify, "driver lacks cuCtxGetCurrent");
	}
	auto count = cmt::CudaDeviceCount();
	if (!count)
	{
		return cmt::Failure(std::move(count.error()));
	}
	if (*count == 0)
	{
		return 77;
	}
	CMT_CUDA(cudaSetDevice(0));
	CMT_CUDA(cudaFree(nullptr)); // Initialize the primary context outside every measurement.
	constexpr std::int64_t mib = 1 << 20;
	for (const bool htod : {true, false})
	{
		for (const auto path :
			 {Path::ReusedPipeline, Path::LifecyclePipeline, Path::PinnedDma, Path::PageableDriver})
		{
			const char* name = path == Path::ReusedPipeline		 ? "pipeline/reused"
							   : path == Path::LifecyclePipeline ? "pipeline/lifecycle"
							   : path == Path::PinnedDma		 ? "pinned_dma/reused"
																 : "pageable_driver";
			const std::string label = std::string(name) + (htod ? "/htod" : "/dtoh");
			auto* bench = benchmark::RegisterBenchmark(label.c_str(), MeasureCallback, path, htod);
			bench->ArgNames({"bytes", "workers", "slots_per_worker", "chunk_bytes"})->UseRealTime();
			const std::int64_t sizes[] = {4096, mib, 16 * mib, 64 * mib, 256 * mib, 1024 * mib};
			for (const auto bytes : sizes)
			{
				if (path == Path::PinnedDma || path == Path::PageableDriver)
				{
					bench->Args({bytes, 0, 0, 0});
				}
				else
				{
					// Each layout owns exactly 32 MiB of pinned staging.
					bench->Args({bytes, 1, 2, 16 * mib});
					bench->Args({bytes, 1, 4, 8 * mib});
					bench->Args({bytes, 1, 8, 4 * mib});
					bench->Args({bytes, 2, 2, 8 * mib});
					bench->Args({bytes, 4, 2, 4 * mib});
				}
			}
		}
	}
	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
	// Keep libcuda loaded: CUDA's own exit handlers can still need its code.
	return failed ? 6 : 0;
}

int main(int argc, char** argv)
{
	cmt::InstallFatalAllocationHandler();
	auto result = RunMain(argc, argv);
	if (result)
	{
		return *result;
	}
	std::fprintf(stderr, "chunk_pipeline_bench: %s\n", result.error().Message());
	return result.error().ExitCode();
}
