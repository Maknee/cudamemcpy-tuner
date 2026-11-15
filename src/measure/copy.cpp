#include <charconv>
#include <fstream>
#include "measure/copy.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "copying/direct.h"
#include "measure/pin_pipeline.h"
#include "machine.h"
#include "cuda/cuda.h"
#include "log.h"

namespace cmt {

struct StagedCopyBenchmark::Impl
{
	Result<void> Setup(const StagedBenchmarkOptions& shape)
	{
		if (shape.cpu_copy)
		{
			const auto selected =
				hwcopy::Select(LoopNameText(shape.cpu_loop), CopyCtx{}, std::nullopt, hwcopy::GetCpu());
			if (!selected || selected->fell_back)
			{
				return Unsupported("staged CPU loop cannot execute exactly");
			}
			knobs.loop = selected->loop;
		}
		CMT_CUDA(cudaSetDevice(shape.device));
		auto created_stream = Stream::Create();
		if (!created_stream)
		{
			return Failure(created_stream.error());
		}
		caller = std::move(*created_stream);
		CMT_RETURN_IF_ERROR(cu::InitializeDriverApi());
		cu::CUcontext context = nullptr;
		const auto context_status = cu::g_api.cuCtxGetCurrent(&context);
		if (context_status != cu::kSuccess || context == nullptr)
		{
			return Failure(Error::Kind::Cuda, "staged benchmark has no CUDA context", context_status);
		}
		knobs.workers = static_cast<unsigned>(shape.threads);
		knobs.slots = static_cast<uint32_t>(shape.slots);
		knobs.chunk_bytes = shape.slot_bytes;
		knobs.wc_slots = shape.wc_slots;
		const ring::PipelineCapacity capacity{knobs.workers, knobs.slots, ring::RequiredPinnedBytes(knobs)};
		HookConfig config;
		config.slot_alloc =
			shape.wc_slots ? StagingAllocation::WriteCombinedPinned : StagingAllocation::CachedPinned;
		config.bench_dma = !shape.cpu_copy;
		std::vector<unsigned> cpus;
		if (shape.cpu >= 0)
		{
			for (unsigned index = 1; index < knobs.workers; ++index)
			{
				cpus.push_back(static_cast<unsigned>(shape.cpu) + index);
			}
		}
		Result<ThreadPlacement> placement{ThreadPlacement{}};
		if (knobs.workers > 1)
		{
			placement = cu::CurrentWorkerPlacement(cu::g_api, cpus, context);
		}
		if (!placement)
		{
			return Failure(placement.error());
		}
		placement->allowed_cpus = shape.cpu >= 0 ? shape.worker_allowed_cpus : std::vector<unsigned>{};
		auto created = ring::ChunkPipeline::Create(context, config, capacity, *placement);
		if (!created)
		{
			return Failure(created.error());
		}
		CMT_RETURN_IF_ERROR((*created)->Start(*placement, !std::empty(cpus)));
		pipeline = std::move(*created);
		return {};
	}

	Result<void> Finish()
	{
		CMT_RETURN_IF_ERROR(pipeline->Finish());
		return caller.Sync();
	}

	Result<double> Copy(Dir direction, unsigned char* host, unsigned char* device, std::size_t bytes,
						std::uint32_t repetitions)
	{
		const double started = NowSeconds();
		for (std::uint32_t repetition = 0; repetition < repetitions; ++repetition)
		{
			const auto device_pointer = reinterpret_cast<cu::CUdeviceptr>(device);
			const auto stream = reinterpret_cast<cu::CUstream>(caller.Get());
			observation = {};
			const auto status =
				direction == Dir::HtoD
					? pipeline->HtoD(knobs, device_pointer, host, bytes, stream, &observation)
					: pipeline->DtoH(knobs, host, device_pointer, bytes, stream, &observation);
			if (status != cu::kSuccess)
			{
				return Failure(Error::Kind::Cuda, "staged pipeline copy failed", status);
			}
		}
		CMT_RETURN_IF_ERROR(Finish());
		return NowSeconds() - started;
	}

	Stream caller;
	std::unique_ptr<ring::ChunkPipeline> pipeline;
	CopySettings knobs;
	ring::CopyObservation observation;
};

StagedCopyBenchmark::StagedCopyBenchmark() = default;
StagedCopyBenchmark::~StagedCopyBenchmark() = default;

Result<void> StagedCopyBenchmark::Setup(const StagedBenchmarkOptions& shape)
{
	if (shape.threads < 1 || shape.threads > kMaxPipelineWorkers || shape.slots < 1 ||
		shape.slots > kMaxStagingBuffers || shape.slot_bytes == 0 || shape.slot_bytes % 16 != 0)
	{
		return Failure(Error::Kind::Usage, "staged layout exceeds production pipeline limits");
	}
	if (!shape.dma)
	{
		return Failure(Error::Kind::Unsupported, "CPU-only copying uses the CPU microbenchmark");
	}
	auto prepared = std::make_unique<Impl>();
	CMT_RETURN_IF_ERROR(prepared->Setup(shape));
	impl = std::move(prepared);
	return {};
}

Result<double> StagedCopyBenchmark::Copy(Dir direction, unsigned char* host, unsigned char* device,
										 std::size_t bytes, std::uint32_t repetitions)
{
	if (!impl)
	{
		Panic("staged copy before setup");
	}
	if ((direction != Dir::HtoD && direction != Dir::DtoH) || bytes == 0 || repetitions == 0)
	{
		return Failure(Error::Kind::Usage, "staged copy requires HtoD/DtoH, bytes and repetitions");
	}
	return impl->Copy(direction, host, device, bytes, repetitions);
}

Result<void> StagedCopyBenchmark::Finish()
{
	if (!impl)
	{
		return {};
	}
	return impl->Finish();
}

void StagedCopyBenchmark::Teardown() noexcept
{
	impl.reset();
}

namespace {

using cmt::copying::CopyDirect;
using DirectRequest = cmt::copying::DirectRequest<cmt::gpu::RuntimeApi>;
using detail::SeedFor;

class RegistrationCopy final
{
public:
	void Setup() noexcept
	{
	}

	[[nodiscard]] Result<CopyPhases> Copy(void* host, std::size_t bytes, std::uint32_t reps);

	void Teardown() noexcept
	{
	}
};

class RegisterCopy final
{
public:
	void Setup() noexcept
	{
	}

	[[nodiscard]] Result<CopyPhases> Copy(const DirectRequest&, const Stream&, std::uint32_t reps);

	void Teardown() noexcept
	{
	}
};

template <bool WithData>
Result<CopyPhases> Transfer(DirectRequest request, const Stream* stream, std::uint32_t reps)
{
	CopyPhases phases;
	for (std::uint32_t r = 0; r < reps; r++)
	{
		const double t0 = NowSeconds();
		auto registered = HostRegistration::Create(request.host, request.bytes);
		if (!registered)
		{
			return Failure(std::move(registered.error()));
		}
		auto registration = std::move(*registered);

		struct Drain
		{
			const Stream* stream;

			~Drain()
			{
				if constexpr (WithData)
				{
					CheckCudaCleanup(cudaStreamSynchronize(stream->Get()));
				}
			}
		} drain{stream};

		const double t1 = NowSeconds();
		double t2 = t1;
		if constexpr (WithData)
		{
			request.stream = stream->Get();
			request.asynchronous = true;
			CMT_CUDA(CopyDirect(request));
			CMT_RETURN_IF_ERROR(stream->Sync());
			t2 = NowSeconds();
		}
		CMT_RETURN_IF_ERROR(registration.Release());
		const double t3 = NowSeconds();
		phases.setup += t1 - t0;
		phases.copy += t2 - t1;
		phases.teardown += t3 - t2;
	}
	return phases;
}

Result<CopyPhases> RegistrationCopy::Copy(void* host, std::size_t bytes, std::uint32_t reps)
{
	return Transfer<false>({Dir::HtoD, nullptr, host, bytes}, nullptr, reps);
}

Result<CopyPhases> RegisterCopy::Copy(const DirectRequest& request, const Stream& stream, std::uint32_t reps)
{
	return Transfer<true>(request, &stream, reps);
}

template <class T, class... Args>
Result<std::unique_ptr<PathRunner>> MakeRunner(const RunContext& ctx, Args... args)
{
	auto runner = std::make_unique<T>(ctx, args...);
	CMT_RETURN_IF_ERROR(runner->Setup(ctx));
	return std::unique_ptr<PathRunner>(std::move(runner));
}

template <class T>
Result<void> Acquire(T& destination, Result<T> resource)
{
	if (!resource)
	{
		return cmt::Failure(std::move(resource.error()));
	}
	destination = std::move(*resource);
	return {};
}

struct PinnedHost
{
	HostBuffer buf;
	HostRegistration reg;
	PinnedAlloc alloc;
	unsigned char* p = nullptr;

	Result<void> Setup(std::size_t bytes, Alloc a, Pages pages)
	{
		if (a == Alloc::Register)
		{
			CMT_RETURN_IF_ERROR(Acquire(buf, HostBuffer::Create(bytes, pages)));
			CMT_RETURN_IF_ERROR(Acquire(reg, HostRegistration::Create(std::data(buf), bytes)));
			p = std::data(buf);
		}
		else
		{
			CMT_RETURN_IF_ERROR(
				Acquire(alloc, PinnedAlloc::Create(bytes, a == Alloc::HostAllocWc ? HostMem::WriteCombined
																				  : HostMem::Cached)));
			for (std::size_t i = 0; i < bytes; i += BasePageBytes())
			{
				std::data(alloc)[i] = 0;
			}
			p = std::data(alloc);
		}
		return {};
	}

	std::optional<long> HugeKiB() const
	{
		if (std::data(buf))
		{
			return buf.HugeKiB();
		}
		return std::nullopt;
	}
};

class PinnedRunner final : public PathRunner
{
public:
	explicit PinnedRunner(const RunContext& ctx) : config(ctx.cfg)
	{
	}

	Result<void> Setup(const RunContext& ctx)
	{
		CMT_RETURN_IF_ERROR(host.Setup(ctx.max_bytes, config.alloc, config.pages));
		CMT_RETURN_IF_ERROR(Acquire(device_buffer, DeviceBuffer::Create(ctx.max_bytes)));
		return Acquire(stream, Stream::Create());
	}

	Result<void> Prepare(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			PatternFill(host.p, b, SeedFor(b, 0));
			StoreFence();
		}
		else
		{
			CMT_RETURN_IF_ERROR(LaunchPatternFill(std::data(device_buffer), b, SeedFor(b, 1), stream.Get()));
		}
		CMT_RETURN_IF_ERROR(ClearDestination(b));
		return {};
	}

	Result<void> ClearDestination(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			CMT_CUDA(cudaMemsetAsync(std::data(device_buffer), 0, b, stream.Get()));
		}
		else
		{
			std::memset(host.p, 0, b);
			StoreFence();
		}
		CMT_RETURN_IF_ERROR(stream.Sync());
		return {};
	}

	Result<SampleOut> Sample(std::size_t b, std::uint32_t reps) override
	{
		const double t0 = NowSeconds();
		for (std::uint32_t r = 0; r < reps; r++)
		{
			CMT_CUDA(
				CopyDirect(DirectRequest{config.dir, std::data(device_buffer), host.p, b, stream.Get(), true}));
		}
		CMT_RETURN_IF_ERROR(stream.Sync());
		SampleOut o;
		o.seconds = NowSeconds() - t0;
		return o;
	}

	Result<std::optional<std::size_t>> Verify(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			return DevicePatternCheck(std::data(device_buffer), b, SeedFor(b, 0), stream.Get());
		}
		return PatternCheck(host.p, b, SeedFor(b, 1));
	}

	Result<void> Flush(std::size_t b) override
	{
		if (config.alloc != Alloc::HostAllocWc)
		{
			FlushRange(host.p, b);
		}
		return {};
	}

	std::optional<long> HugeKiB() const override
	{
		return host.HugeKiB();
	}

	std::string Alloc() const override
	{
		return AllocName(config.alloc);
	}

private:
	const Config config;
	PinnedHost host;
	DeviceBuffer device_buffer;
	Stream stream;
};

class BidirRunner final : public PathRunner
{
public:
	explicit BidirRunner(const RunContext& ctx) : config(ctx.cfg)
	{
	}

	Result<void> Setup(const RunContext& ctx)
	{
		CMT_RETURN_IF_ERROR(up.Setup(ctx.max_bytes, config.alloc, config.pages));
		CMT_RETURN_IF_ERROR(down.Setup(ctx.max_bytes, config.alloc, config.pages));
		CMT_RETURN_IF_ERROR(Acquire(device_up, DeviceBuffer::Create(ctx.max_bytes)));
		CMT_RETURN_IF_ERROR(Acquire(device_down, DeviceBuffer::Create(ctx.max_bytes)));
		CMT_RETURN_IF_ERROR(Acquire(up_stream, Stream::Create()));
		return Acquire(down_stream, Stream::Create());
	}

	Result<void> Prepare(std::size_t b) override
	{
		PatternFill(up.p, b, SeedFor(b, 0));
		CMT_RETURN_IF_ERROR(LaunchPatternFill(std::data(device_down), b, SeedFor(b, 1), up_stream.Get()));
		CMT_RETURN_IF_ERROR(ClearDestination(b));
		return {};
	}

	Result<void> ClearDestination(std::size_t b) override
	{
		std::memset(down.p, 0, b);
		CMT_CUDA(cudaMemsetAsync(std::data(device_up), 0, b, up_stream.Get()));
		CMT_RETURN_IF_ERROR(up_stream.Sync());
		return {};
	}

	Result<SampleOut> Sample(std::size_t b, std::uint32_t reps) override
	{
		Event a0, a1, b0, b1;
		CMT_RETURN_IF_ERROR(Acquire(a0, Event::Create(Timing::Enabled)));
		CMT_RETURN_IF_ERROR(Acquire(a1, Event::Create(Timing::Enabled)));
		CMT_RETURN_IF_ERROR(Acquire(b0, Event::Create(Timing::Enabled)));
		CMT_RETURN_IF_ERROR(Acquire(b1, Event::Create(Timing::Enabled)));
		const double t0 = NowSeconds();
		CMT_RETURN_IF_ERROR(a0.Record(up_stream));
		CMT_RETURN_IF_ERROR(b0.Record(down_stream));
		for (std::uint32_t r = 0; r < reps; r++)
		{
			CMT_CUDA(
				CopyDirect(DirectRequest{Dir::HtoD, std::data(device_up), up.p, b, up_stream.Get(), true}));
			CMT_CUDA(CopyDirect(
				DirectRequest{Dir::DtoH, std::data(device_down), down.p, b, down_stream.Get(), true}));
		}
		CMT_RETURN_IF_ERROR(a1.Record(up_stream));
		CMT_RETURN_IF_ERROR(b1.Record(down_stream));
		CMT_RETURN_IF_ERROR(up_stream.Sync());
		CMT_RETURN_IF_ERROR(down_stream.Sync());
		SampleOut o;
		o.seconds = NowSeconds() - t0;
		const double moved = static_cast<double>(b) * reps;
		const auto up_ms = Event::ElapsedMs(a0, a1);
		if (!up_ms)
		{
			return cmt::Failure(up_ms.error());
		}
		const auto down_ms = Event::ElapsedMs(b0, b1);
		if (!down_ms)
		{
			return cmt::Failure(down_ms.error());
		}
		o.directions = SampleOut::Directions{Gbps(moved, *up_ms * 1e-3), Gbps(moved, *down_ms * 1e-3)};
		return o;
	}

	Result<std::optional<std::size_t>> Verify(std::size_t b) override
	{
		auto bad = DevicePatternCheck(std::data(device_up), b, SeedFor(b, 0), up_stream.Get());
		if (!bad)
		{
			return cmt::Failure(std::move(bad.error()));
		}
		return std::optional<std::size_t>(*bad + PatternCheck(down.p, b, SeedFor(b, 1)));
	}

	Result<void> Flush(std::size_t b) override
	{
		FlushRange(up.p, b);
		FlushRange(down.p, b);
		return {};
	}

	std::optional<long> HugeKiB() const override
	{
		const auto u = up.HugeKiB(), d = down.HugeKiB();
		if (!u || !d)
		{
			return std::nullopt;
		}
		return *u + *d;
	}

	std::string Alloc() const override
	{
		return AllocName(config.alloc);
	}

	std::vector<MeasurementRow> Rows(const MeasurementRow& base,
									 const std::vector<SampleOut>& s) const override
	{
		MeasurementRow r = base;
		const auto best =
			std::min_element(std::begin(s), std::end(s),
							 [](const SampleOut& x, const SampleOut& y) { return x.seconds < y.seconds; });
		if (best != std::end(s) && best->directions)
		{
			r.gbps_h2d = best->directions->htod_gbps, r.gbps_d2h = best->directions->dtoh_gbps;
		}
		return {r};
	}

private:
	const Config config;
	PinnedHost up, down;
	DeviceBuffer device_up, device_down;
	Stream up_stream, down_stream;
};

PathSpec PinnedCeSpec()
{
	PathSpec p;
	p.name = "pinned_ce";
	p.summary = "completed cudaMemcpyAsync on already-pinned memory; small-copy route needs verification";
	p.nvbandwidth = "host_to_device_memcpy_ce / device_to_host_memcpy_ce";
	p.accepts = {Knob::Direction, Knob::Pages, Knob::Alloc};
	p.make = [](const RunContext& ctx) { return MakeRunner<PinnedRunner>(ctx); };
	return p;
}

PathSpec BidirCeSpec()
{
	PathSpec p;
	p.name = "bidir_ce";
	p.summary = "HtoD and DtoH at the same time on two streams (sum; each direction in gbps_h2d/gbps_d2h)";
	p.nvbandwidth = "host_to_device_bidirectional_memcpy_ce";
	p.dirs = PathSpec::Dirs::Both;
	p.device_bytes = 2, p.host_bytes = 2, p.moved_bytes = 2;
	p.accepts = {Knob::Pages, Knob::Alloc};
	p.validate = [](const Config& c) -> Result<void>
	{
		if (c.alloc == Alloc::HostAllocWc)
		{
			return UsageError("write-combined memory is supported by pinned_ce, not bidir_ce");
		}
		return {};
	};
	p.make = [](const RunContext& ctx) { return MakeRunner<BidirRunner>(ctx); };
	return p;
}

class PageableRunner final : public PathRunner
{
public:
	explicit PageableRunner(const RunContext& ctx) : config(ctx.cfg)
	{
	}

	Result<void> Setup(const RunContext& ctx)
	{
		CMT_RETURN_IF_ERROR(Acquire(host, HostBuffer::Create(ctx.max_bytes, config.pages)));
		CMT_RETURN_IF_ERROR(Acquire(device_buffer, DeviceBuffer::Create(ctx.max_bytes)));
		return {};
	}

	Result<void> Prepare(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			PatternFill(std::data(host), b, SeedFor(b, 0));
		}
		else
		{
			CMT_RETURN_IF_ERROR(LaunchPatternFill(std::data(device_buffer), b, SeedFor(b, 1), nullptr));
		}
		CMT_RETURN_IF_ERROR(ClearDestination(b));
		return {};
	}

	Result<void> ClearDestination(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			CMT_CUDA(cudaMemset(std::data(device_buffer), 0, b));
		}
		else
		{
			std::memset(std::data(host), 0, b);
			StoreFence();
		}
		CMT_CUDA(cudaDeviceSynchronize());
		return {};
	}

	Result<SampleOut> Sample(std::size_t b, std::uint32_t reps) override
	{
		const double t0 = NowSeconds();
		for (std::uint32_t r = 0; r < reps; r++)
		{
			CMT_CUDA(CopyDirect(DirectRequest{config.dir, std::data(device_buffer), std::data(host), b}));
		}
		CMT_CUDA(cudaDeviceSynchronize());
		SampleOut o;
		o.seconds = NowSeconds() - t0;
		return o;
	}

	Result<std::optional<std::size_t>> Verify(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			return DevicePatternCheck(std::data(device_buffer), b, SeedFor(b, 0), nullptr);
		}
		return PatternCheck(std::data(host), b, SeedFor(b, 1));
	}

	Result<void> Flush(std::size_t b) override
	{
		FlushRange(std::data(host), b);
		return {};
	}

	std::optional<long> HugeKiB() const override
	{
		return host.HugeKiB();
	}

	std::string Alloc() const override
	{
		return "mmap";
	}

private:
	const Config config;
	HostBuffer host;
	DeviceBuffer
		device_buffer;
};

PathSpec PageableSpec()
{
	PathSpec p;
	p.name = "pageable";
	p.summary = "plain cudaMemcpy on pageable mmap memory (the runtime, or an LD_PRELOAD hook)";
	p.accepts = {Knob::Direction, Knob::Pages};
	p.make = [](const RunContext& ctx) { return MakeRunner<PageableRunner>(ctx); };
	return p;
}

class StagedRunner final : public PathRunner
{
public:
	explicit StagedRunner(const RunContext& ctx, bool measure_lifecycle = false)
		: config(ctx.cfg), device(ctx.dev.ordinal), lifecycle(measure_lifecycle)
	{
	}

	Result<void> Setup(const RunContext& ctx)
	{
		CMT_RETURN_IF_ERROR(Acquire(host, HostBuffer::Create(ctx.max_bytes, config.pages)));
		CMT_RETURN_IF_ERROR(Acquire(device_buffer, DeviceBuffer::Create(ctx.max_bytes)));
		if (lifecycle)
		{
			return {};
		}
		return copier.Setup(Shape());
	}

	Result<void> Prepare(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			PatternFill(std::data(host), b, SeedFor(b, 0));
		}
		else
		{
			CMT_RETURN_IF_ERROR(LaunchPatternFill(std::data(device_buffer), b, SeedFor(b, 1), nullptr));
		}
		CMT_RETURN_IF_ERROR(ClearDestination(b));
		return {};
	}

	Result<void> ClearDestination(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			CMT_CUDA(cudaMemset(std::data(device_buffer), 0, b));
		}
		else
		{
			std::memset(std::data(host), 0, b);
			StoreFence();
		}
		CMT_CUDA(cudaDeviceSynchronize());
		return {};
	}

	Result<SampleOut> Sample(std::size_t b, std::uint32_t reps) override
	{
		SampleOut out;
		if (lifecycle)
		{
			out.phases = SampleOut::Phases{};
			for (std::uint32_t repetition = 0; repetition < reps; ++repetition)
			{
				const double started = NowSeconds();
				CMT_RETURN_IF_ERROR(copier.Setup(Shape()));
				const double prepared = NowSeconds();
				auto copied = copier.Copy(config.dir, std::data(host), std::data(device_buffer), b, 1);
				if (!copied)
				{
					return Failure(copied.error());
				}
				const double completed = NowSeconds();
				copier.Teardown();
				const double released = NowSeconds();
				out.phases->setup += prepared - started;
				out.phases->copy += completed - prepared;
				out.phases->teardown += released - completed;
				out.seconds += released - started;
			}
			return out;
		}
		auto copied = copier.Copy(config.dir, std::data(host), std::data(device_buffer), b, reps);
		if (!copied)
		{
			return cmt::Failure(std::move(copied.error()));
		}
		out.seconds = *copied;
		return out;
	}

	Result<std::optional<std::size_t>> Verify(std::size_t b) override
	{
		if (!config.cpu_copy || !config.dma)
		{
			return std::nullopt;
		}
		if (config.dir == Dir::HtoD)
		{
			return DevicePatternCheck(std::data(device_buffer), b, SeedFor(b, 0), nullptr);
		}
		return PatternCheck(std::data(host), b, SeedFor(b, 1));
	}

	Result<void> Flush(std::size_t b) override
	{
		FlushRange(std::data(host), b);
		return {};
	}

	std::optional<long> HugeKiB() const override
	{
		return host.HugeKiB();
	}

	std::string Alloc() const override
	{
		return config.wc_slots ? "slots_wc" : "slots";
	}

private:
	StagedBenchmarkOptions Shape() const
	{
		return {device,
				config.cpu,
				static_cast<std::size_t>(config.threads),
				static_cast<std::size_t>(config.slots),
				config.slot_bytes,
				config.cpu_loop,
				config.cpu_copy,
				config.wc_slots,
				config.dma,
				config.worker_allowed_cpus};
	}

	const Config config;
	int device;
	bool lifecycle;
	HostBuffer host;
	DeviceBuffer device_buffer;
	cmt::StagedCopyBenchmark copier;
};

PathSpec StagedSpec()
{
	PathSpec p;
	p.name = "staged";
	p.summary = "reusable pinned slots: CPU fill/drain + cudaMemcpyAsync per slot, T copier threads";
	p.accepts = {Knob::Direction, Knob::Pages,	   Knob::Fill, Knob::Threads,
				 Knob::Slots,	  Knob::SlotBytes, Knob::Wc,   Knob::NoDma};
	p.validate = [](const Config& c) -> Result<void>
	{
		if (c.threads < 1 || c.threads > static_cast<int>(kMaxPipelineWorkers))
		{
			return UsageError("--threads must be 1..16 for the production pipeline");
		}
		if (c.slots < 1 || c.slots > static_cast<int>(kMaxStagingBuffers))
		{
			return UsageError("--slots must be 1..32 for the production pipeline");
		}
		if (c.slot_bytes == 0 || c.slot_bytes % 16)
		{
			return UsageError("--slot-bytes must be a positive multiple of 16");
		}
		if (!c.dma && !c.cpu_copy)
		{
			return UsageError("--no-dma with --fill none measures nothing");
		}
		if (c.cpu_copy && !Resolve(NativeArch(), c.cpu_loop, hwcopy::GetCpu(), std::nullopt))
		{
			return Unsupported("CPU copy loop is not supported on this CPU");
		}
		return {};
	};
	p.make = [](const RunContext& ctx) { return MakeRunner<StagedRunner>(ctx); };
	return p;
}

PathSpec StagedLifecycleSpec()
{
	PathSpec path = StagedSpec();
	path.name = "staged_lifecycle";
	path.summary = "create pinned pool/workers, complete first staged copy, and release pool/workers";
	path.make = [](const RunContext& ctx) { return MakeRunner<StagedRunner>(ctx, true); };
	return path;
}

class PinPipelineRunner final : public PathRunner
{
public:
	explicit PinPipelineRunner(const RunContext& ctx) : config(ctx.cfg), device(ctx.dev.ordinal)
	{
	}

	Result<void> Setup(const RunContext& ctx)
	{
		CMT_RETURN_IF_ERROR(Acquire(host, HostBuffer::Create(ctx.max_bytes, config.pages)));
		CMT_RETURN_IF_ERROR(Acquire(device_buffer, DeviceBuffer::Create(ctx.max_bytes)));
		CMT_RETURN_IF_ERROR(Acquire(stream, Stream::Create()));
		return {};
	}

	Result<void> Prepare(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			PatternFill(std::data(host), b, SeedFor(b, 0));
		}
		else
		{
			CMT_RETURN_IF_ERROR(LaunchPatternFill(std::data(device_buffer), b, SeedFor(b, 1), stream.Get()));
		}
		CMT_RETURN_IF_ERROR(ClearDestination(b));
		using Policy = cmt::copying::runtime::ReleasePolicy;
		const Policy release = config.unpin == Unpin::Prereg  ? Policy::Preregister
							   : config.unpin == Unpin::Eager ? Policy::Eager
															  : Policy::End;
		return copier.Setup(cmt::copying::runtime::PinPipelineOptions{device, config.pin_step, release},
							std::data(host), b, stream);
	}

	Result<void> ClearDestination(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			CMT_CUDA(cudaMemsetAsync(std::data(device_buffer), 0, b, stream.Get()));
		}
		else
		{
			std::memset(std::data(host), 0, b);
		}
		CMT_RETURN_IF_ERROR(stream.Sync());
		return {};
	}

	Result<SampleOut> Sample(std::size_t b, std::uint32_t reps) override
	{
		(void)b;
		SampleOut out;
		auto copied = copier.Copy(config.dir, std::data(device_buffer), reps);
		if (!copied)
		{
			return cmt::Failure(std::move(copied.error()));
		}
		out.seconds = *copied;
		return out;
	}

	Result<std::optional<std::size_t>> Verify(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			return DevicePatternCheck(std::data(device_buffer), b, SeedFor(b, 0), stream.Get());
		}
		return PatternCheck(std::data(host), b, SeedFor(b, 1));
	}

	Result<void> Flush(std::size_t b) override
	{
		FlushRange(std::data(host), b);
		return {};
	}

	std::optional<long> HugeKiB() const override
	{
		return host.HugeKiB();
	}

	std::string Alloc() const override
	{
		return "mmap";
	}

private:
	const Config config;
	int device;
	HostBuffer host;
	DeviceBuffer device_buffer;
	Stream stream;
	cmt::copying::runtime::PinPipelineCopy copier;
};

class RegisterRunner final : public PathRunner
{
public:
	explicit RegisterRunner(const RunContext&)
	{
	}

	Result<void> Setup(const RunContext& ctx)
	{
		return Acquire(host, HostBuffer::Create(ctx.max_bytes, ctx.cfg.pages));
	}

	Result<void> Prepare(std::size_t b) override
	{
		PatternFill(std::data(host), b, SeedFor(b, 0));
		return {};
	}

	Result<void> ClearDestination(std::size_t) override
	{
		return {};
	}

	Result<SampleOut> Sample(std::size_t b, std::uint32_t reps) override
	{
		const auto ph = copier.Copy(std::data(host), b, reps);
		SampleOut o;
		if (!ph)
		{
			return cmt::Failure(ph.error());
		}
		o.seconds = ph->setup + ph->teardown;
		o.phases = *ph;
		return o;
	}

	Result<std::optional<std::size_t>> Verify(std::size_t bytes) override
	{
		return PatternCheck(std::data(host), bytes, SeedFor(bytes, 0));
	}

	Result<void> Flush(std::size_t) override
	{
		return {};
	}

	std::optional<long> HugeKiB() const override
	{
		return host.HugeKiB();
	}

	std::string Alloc() const override
	{
		return "mmap";
	}

	std::vector<MeasurementRow> Rows(const MeasurementRow& base,
									 const std::vector<SampleOut>& s) const override
	{
		const double moved = static_cast<double>(base.size_bytes) * base.reps;
		std::vector<double> pin, unpin;
		for (const auto& o : s)
		{
			if (o.phases)
			{
				pin.push_back(Gbps(moved, o.phases->setup)), unpin.push_back(Gbps(moved, o.phases->teardown));
			}
		}
		std::vector<MeasurementRow> out;
		for (std::size_t k = 0; k < 2; k++)
		{
			const Summary sum = Summarize(k == 0 ? pin : unpin);
			MeasurementRow r = base;
			r.phase = k == 0 ? "pin" : "unpin";
			r.gbps_best = sum.best, r.gbps_med = sum.median, r.gbps_min = sum.min;
			out.push_back(r);
		}
		return out;
	}

private:
	RegistrationCopy copier;
	HostBuffer host;
};

class PinCopyUnpinRunner final : public PathRunner
{
public:
	explicit PinCopyUnpinRunner(const RunContext& ctx) : config(ctx.cfg)
	{
	}

	Result<void> Setup(const RunContext& ctx)
	{
		CMT_RETURN_IF_ERROR(Acquire(host, HostBuffer::Create(ctx.max_bytes, config.pages)));
		CMT_RETURN_IF_ERROR(Acquire(device_buffer, DeviceBuffer::Create(ctx.max_bytes)));
		CMT_RETURN_IF_ERROR(Acquire(stream, Stream::Create()));
		return {};
	}

	Result<void> Prepare(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			PatternFill(std::data(host), b, SeedFor(b, 0));
		}
		else
		{
			CMT_RETURN_IF_ERROR(LaunchPatternFill(std::data(device_buffer), b, SeedFor(b, 1), stream.Get()));
		}
		CMT_RETURN_IF_ERROR(ClearDestination(b));
		return {};
	}

	Result<void> ClearDestination(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			CMT_CUDA(cudaMemsetAsync(std::data(device_buffer), 0, b, stream.Get()));
		}
		else
		{
			std::memset(std::data(host), 0, b);
		}
		CMT_RETURN_IF_ERROR(stream.Sync());
		return {};
	}

	Result<SampleOut> Sample(std::size_t b, std::uint32_t reps) override
	{
		const auto ph = copier.Copy(DirectRequest{config.dir, std::data(device_buffer), std::data(host), b},
									stream, reps);
		SampleOut o;
		if (!ph)
		{
			return cmt::Failure(ph.error());
		}
		o.seconds = ph->setup + ph->copy + ph->teardown;
		o.phases = *ph;
		return o;
	}

	Result<std::optional<std::size_t>> Verify(std::size_t b) override
	{
		if (config.dir == Dir::HtoD)
		{
			return DevicePatternCheck(std::data(device_buffer), b, SeedFor(b, 0), stream.Get());
		}
		return PatternCheck(std::data(host), b, SeedFor(b, 1));
	}

	Result<void> Flush(std::size_t b) override
	{
		FlushRange(std::data(host), b);
		return {};
	}

	std::optional<long> HugeKiB() const override
	{
		return host.HugeKiB();
	}

	std::string Alloc() const override
	{
		return "mmap";
	}

	std::vector<MeasurementRow> Rows(const MeasurementRow& base,
									 const std::vector<SampleOut>& s) const override
	{
		MeasurementRow r = base;
		const auto best =
			std::min_element(std::begin(s), std::end(s),
							 [](const SampleOut& x, const SampleOut& y) { return x.seconds < y.seconds; });
		if (best != std::end(s) && best->phases && base.reps > 0)
		{
			r.setup_s = best->phases->setup / base.reps;
			r.copy_s = best->phases->copy / base.reps;
			r.teardown_s = best->phases->teardown / base.reps;
		}
		return {r};
	}

private:
	RegisterCopy copier;
	const Config config;
	HostBuffer host;
	DeviceBuffer device_buffer;
	Stream stream;
};

Result<void> PinStepIsWholePages(const Config& c)
{
	if (c.pin_step == 0 || c.pin_step % BasePageBytes())
	{
		return UsageError("--pin-step must be a positive multiple of the system page size");
	}
	return {};
}

PathSpec PinPipelineSpec()
{
	PathSpec p;
	p.name = "pin_pipeline";
	p.summary = "register step i+1 while step i copies in either direction; --unpin end|eager|prereg";
	p.accepts = {Knob::Direction, Knob::Pages, Knob::PinStep, Knob::Unpin};
	p.validate = PinStepIsWholePages;
	p.make = [](const RunContext& ctx) { return MakeRunner<PinPipelineRunner>(ctx); };
	return p;
}

PathSpec RegisterSpec()
{
	PathSpec p;
	p.name = "register";
	p.summary = "cudaHostRegister + cudaHostUnregister of the whole buffer (rows: phase pin, phase unpin)";
	p.dirs = PathSpec::Dirs::None;
	p.device_bytes = 0;
	p.accepts = {Knob::Pages};
	p.make = [](const RunContext& ctx) { return MakeRunner<RegisterRunner>(ctx); };
	return p;
}

PathSpec PinCopyUnpinSpec()
{
	PathSpec p;
	p.name = "pin_copy_unpin";
	p.summary = "register + one copy + unregister, each phase timed (setup_s, copy_s, teardown_s)";
	p.accepts = {Knob::Direction, Knob::Pages};
	p.make = [](const RunContext& ctx) { return MakeRunner<PinCopyUnpinRunner>(ctx); };
	return p;
}

class LifecycleRunner final : public PathRunner
{
public:
	explicit LifecycleRunner(const RunContext& ctx, bool copy = true)
		: config(ctx.cfg), copy_payload(copy)
	{
	}

	Result<void> Setup(const RunContext&)
	{
		return Acquire(stream, Stream::Create());
	}

	Result<void> Prepare(std::size_t) override
	{
		wrong = 0;
		return {};
	}

	Result<void> ClearDestination(std::size_t) override
	{
		wrong = 0;
		return {};
	}

	Result<SampleOut> Sample(std::size_t b, std::uint32_t reps) override
	{
		SampleOut out;
		out.lifecycle.emplace();
		out.host_backing.emplace();
		auto& phases = *out.lifecycle;
		const bool register_host = config.alloc == cmt::Alloc::Register;
		const auto timed = [&](std::size_t phase, auto operation) -> Result<void>
		{
			const double start = NowSeconds();
			CMT_RETURN_IF_ERROR(operation());
			phases[phase] += NowSeconds() - start;
			return {};
		};
		for (std::uint32_t rep = 0; rep < reps; rep++)
		{
			HostBuffer host;
			PinnedAlloc pinned;
			DeviceBuffer device;
			HostRegistration registration;
			cmt::DrainOnExit drain{stream};
			const auto allocate_host = [&]() -> Result<void>
			{
				if (register_host)
				{
					CMT_RETURN_IF_ERROR(Acquire(host, HostBuffer::Create(b, config.pages)));
				}
				else
				{
					const HostMem memory =
						config.alloc == cmt::Alloc::HostAllocWc ? HostMem::WriteCombined : HostMem::Cached;
					CMT_RETURN_IF_ERROR(Acquire(pinned, PinnedAlloc::Create(b, memory)));
					for (std::size_t offset = 0; offset < b; offset += BasePageBytes())
					{
						std::data(pinned)[offset] = 0;
					}
				}
				StoreFence();
				return {};
			};
			CMT_RETURN_IF_ERROR(timed(0, allocate_host));
			unsigned char* host_data = register_host ? std::data(host) : std::data(pinned);
			out.host_backing->before_pin.Record(register_host ? host.HugeKiB() : std::nullopt);
			if (copy_payload)
			{
				CMT_RETURN_IF_ERROR(timed(1, [&] { return Acquire(device, DeviceBuffer::Create(b)); }));
			}
			const auto seed = SeedFor(b, !copy_payload || config.dir == Dir::HtoD ? 0 : 1);
			if (!copy_payload || config.dir == Dir::HtoD)
			{
				PatternFill(host_data, b, seed);
				if (copy_payload)
				{
					CMT_CUDA(cudaMemsetAsync(std::data(device), 0, b, stream.Get()));
				}
			}
			else
			{
				std::memset(host_data, 0xD3, b);
				CMT_RETURN_IF_ERROR(LaunchPatternFill(std::data(device), b, seed, stream.Get()));
			}
			if (copy_payload)
			{
				CMT_RETURN_IF_ERROR(stream.Sync());
				if (config.flush && FlushSupported())
				{
					FlushRange(host_data, b);
				}
			}
			if (register_host)
			{
				CMT_RETURN_IF_ERROR(timed(
					2, [&] { return Acquire(registration, HostRegistration::Create(host_data, b)); }));
			}
			out.host_backing->after_pin.Record(register_host ? host.HugeKiB() : std::nullopt);
			if (copy_payload)
			{
				CMT_RETURN_IF_ERROR(timed(
					3,
					[&]() -> Result<void>
					{
						CMT_CUDA(CopyDirect(DirectRequest{config.dir, std::data(device), host_data, b,
														 stream.Get(), true}));
						return stream.Sync();
					}));
			}
			if (copy_payload && config.verify)
			{
				if (config.dir == Dir::HtoD)
				{
					auto bad = DevicePatternCheck(std::data(device), b, seed, stream.Get());
					if (!bad)
					{
						return cmt::Failure(std::move(bad.error()));
					}
					wrong += *bad;
				}
				else
				{
					wrong += PatternCheck(host_data, b, seed);
				}
			}
			if (register_host)
			{
				CMT_RETURN_IF_ERROR(timed(4, [&] { return registration.Release(); }));
			}
			if (!copy_payload && config.verify)
			{
				wrong += PatternCheck(host_data, b, seed);
			}
			if (copy_payload)
			{
				CMT_RETURN_IF_ERROR(timed(
					5,
					[&]() -> Result<void>
					{
						device = DeviceBuffer{};
						return {};
					}));
			}
			CMT_RETURN_IF_ERROR(timed(
				6,
				[&]() -> Result<void>
				{
					if (register_host)
					{
						host = HostBuffer{};
					}
					else
					{
						pinned = PinnedAlloc{};
					}
					return {};
				}));
		}
		for (double phase : phases)
		{
			out.seconds += phase;
		}
		out.phases = SampleOut::Phases{phases[0] + phases[1] + phases[2], phases[3],
									   phases[4] + phases[5] + phases[6]};
		return out;
	}

	Result<std::optional<std::size_t>> Verify(std::size_t) override
	{
		return config.verify ? std::optional<std::size_t>(wrong) : std::nullopt;
	}

	Result<void> Flush(std::size_t) override
	{
		return {};
	}

	std::optional<long> HugeKiB() const override
	{
		return std::nullopt;
	}

	std::string Alloc() const override
	{
		return config.alloc == cmt::Alloc::Register ? "mmap" : AllocName(config.alloc);
	}

	std::vector<MeasurementRow> Rows(const MeasurementRow& base,
									 const std::vector<SampleOut>& samples) const override
	{
		MeasurementRow row = base;
		std::vector<const SampleOut*> ordered;
		for (const auto& sample : samples)
		{
			ordered.push_back(&sample);
		}
		std::stable_sort(std::begin(ordered), std::end(ordered),
						 [](auto a, auto b) { return a->seconds < b->seconds; });
		if (!std::empty(ordered) && base.reps > 0)
		{
			const auto& selected = *ordered[std::size(ordered) / 2];
			row.gbps_med = Gbps(static_cast<double>(base.size_bytes), selected.seconds / base.reps);
			const auto& ph = selected.phases;
			if (ph)
			{
				row.setup_s = ph->setup / base.reps;
				row.copy_s = ph->copy / base.reps;
				row.teardown_s = ph->teardown / base.reps;
			}
		}
		return {row};
	}

private:
	const Config config;
	const bool copy_payload;
	Stream stream;
	std::size_t wrong = 0;
};

PathSpec LifecycleSpec()
{
	PathSpec p;
	p.name = "lifecycle";
	p.summary = "sum of host allocation/first-touch, device allocation, pin, completed copy, unpin and frees";
	p.accepts = {Knob::Direction, Knob::Pages, Knob::Alloc};
	p.make = [](const RunContext& ctx) { return MakeRunner<LifecycleRunner>(ctx); };
	return p;
}

PathSpec HostAllocationSpec()
{
	PathSpec p;
	p.name = "host_allocation";
	p.summary = "host allocation/first-touch, pin, unpin and free; no GPU copy";
	p.dirs = PathSpec::Dirs::None;
	p.device_bytes = 0;
	p.accepts = {Knob::Pages, Knob::Alloc};
	p.make = [](const RunContext& ctx) { return MakeRunner<LifecycleRunner>(ctx, false); };
	return p;
}

} // namespace

const std::array<PathSpec, 10>& BenchmarkPaths()
{
	static const std::array<PathSpec, 10> paths = {
		PinnedCeSpec(), BidirCeSpec(), PageableSpec(), StagedSpec(), StagedLifecycleSpec(),
		LifecycleSpec(), HostAllocationSpec(), PinPipelineSpec(), RegisterSpec(), PinCopyUnpinSpec()};
	return paths;
}

const PathSpec* FindPath(const std::string& name)
{
	for (const PathSpec& p : BenchmarkPaths())
	{
		if (p.name == name)
		{
			return &p;
		}
	}
	return nullptr;
}

Result<const PathSpec*> PathNamed(const std::string& name)
{
	if (const PathSpec* p = FindPath(name))
	{
		return p;
	}
	return UsageError("unknown benchmark path (see --help)");
}

Result<std::vector<std::size_t>> SizesThatFit(const Config& c, const PathSpec& p, SizeCoverage* coverage)
{
	std::size_t free_b = 0, total_b = 0;
	CMT_CUDA(cudaMemGetInfo(&free_b, &total_b));
	const bool driver_allocated = p.accepts.Has(Knob::Alloc) && c.alloc != Alloc::Register;
	const Pages caller_pages = driver_allocated ? Pages::Small : c.pages;
	HostAllocationRequirements host_allocation;
	host_allocation.buffer_alignment =
		caller_pages == Pages::Hugetlb ? HugetlbPageBytes() : std::optional<std::size_t>(BasePageBytes());
	if (p.name == "staged" || p.name == "staged_lifecycle")
	{
		CopySettings settings;
		settings.workers = static_cast<unsigned>(c.threads);
		settings.slots = static_cast<std::uint32_t>(c.slots);
		settings.chunk_bytes = c.slot_bytes;
		host_allocation.fixed_pinned_bytes = ring::RequiredPinnedBytes(settings);
		if (host_allocation.fixed_pinned_bytes == 0)
		{
			return UsageError("staging pool size is empty or overflows");
		}
		host_allocation.separate_pinned_budget = caller_pages == Pages::Hugetlb;
		if (host_allocation.separate_pinned_budget)
		{
			host_allocation.pinned_available = HostMemoryAvailable(Pages::Small);
		}
	}
	const auto pending_ordinary_bytes =
		host_allocation.separate_pinned_budget ? host_allocation.fixed_pinned_bytes : 0;
	const auto caller_available = HostMemoryAvailable(caller_pages, pending_ordinary_bytes);
	const auto result =
		ComputeSizeCoverage(c.sizes, free_b, caller_available, p.device_bytes, p.host_bytes, host_allocation);
	for (const auto& size : result.sizes)
	{
		if (size.limit == CapacityLimit::Device)
		{
			spdlog::warn("cmt: skip {}: needs {} x that in device memory, {:.1f} GiB free",
						 HumanSize(size.bytes), p.device_bytes, static_cast<double>(free_b) / (1u << 30));
		}
		else if (size.limit == CapacityLimit::Host)
		{
			spdlog::warn("cmt: skip {}: {} (caller host {}, fixed pinned pool {})", HumanSize(size.bytes),
						 size.reason, size.caller_host_bytes ? HumanSize(*size.caller_host_bytes) : "unknown",
						 HumanSize(host_allocation.fixed_pinned_bytes));
		}
	}
	if (coverage)
	{
		*coverage = result;
	}
	return result.Eligible();
}

namespace {
Result<void> RunSize(const Config& c, const DeviceInfo& dev, PathRunner& runner, std::size_t bytes,
					 bool flushing, bool& any_bad, std::size_t& emitted,
					 const Function<Result<void>(const MeasurementRow&) const>& emit)
{
	const PathSpec& path = *c.path;
	const auto reps = RepsFor(bytes, c.min_bytes);
	CMT_RETURN_IF_ERROR(runner.Prepare(bytes));
	CMT_RETURN_IF_ERROR(runner.Sample(bytes, 1));
	CMT_RETURN_IF_ERROR(runner.ClearDestination(bytes));
	std::vector<SampleOut> samples;
	const double deadline = NowSeconds() + c.budget_s;
	for (int i = 0; i < c.samples; ++i)
	{
		if (c.budget_s > 0 && i >= 2 && NowSeconds() > deadline)
		{
			break;
		}
		if (flushing)
		{
			CMT_RETURN_IF_ERROR(runner.Flush(bytes));
		}
		auto sample = runner.Sample(bytes, reps);
		if (!sample)
		{
			return cmt::Failure(std::move(sample.error()));
		}
		samples.push_back(std::move(*sample));
	}
	std::optional<std::size_t> bad;
	if (c.verify)
	{
		auto verified = runner.Verify(bytes);
		if (!verified)
		{
			return cmt::Failure(std::move(verified.error()));
		}
		bad = *verified;
	}
	if (bad && *bad > 0)
	{
		any_bad = true;
		spdlog::error("cmt: {} {} {}: {} wrong words", path.name, DirColumn(c, path), HumanSize(bytes), *bad);
	}
	MeasurementRow base = ResultIdentity(c, dev, runner, bytes, reps, flushing,
										 !bad		 ? Verified::Skip
										 : *bad == 0 ? Verified::Ok
													 : Verified::Bad);
	SummarizeResult(base, samples, path.moved_bytes);
	for (const MeasurementRow& row : runner.Rows(base, samples))
	{
		CMT_RETURN_IF_ERROR(emit(row));
		++emitted;
	}
	return {};
}
} // namespace

Result<std::optional<DeviceInfo>> PrepareBenchmark(Config& config)
{
	if (config.cpu >= 0)
	{
		auto allowed = GetThreadAffinity();
		if (!allowed)
		{
			return Failure(allowed.error());
		}
		config.worker_allowed_cpus = std::move(*allowed);
	}
	CMT_RETURN_IF_ERROR(PinThread(config.cpu));
	const auto count = CudaDeviceCount();
	if (!count)
	{
		return Failure(count.error());
	}
	if (*count == 0)
	{
		return std::optional<DeviceInfo>{};
	}
	if (config.device >= *count)
	{
		return Failure(Error::Kind::Usage, "--device is outside the visible CUDA device range");
	}
	auto device = QueryDevice(config.device);
	if (!device)
	{
		return Failure(device.error());
	}
	return std::optional<DeviceInfo>{std::move(*device)};
}

Result<void> Run(const Config& c, const DeviceInfo& dev,
				 const Function<Result<void>(const MeasurementRow&) const>& emit)
{
	if (!c.path)
	{
		Panic("Run without a path");
	}
	CMT_CUDA(cudaSetDevice(dev.ordinal));
	auto sizes = SizesThatFit(c, *c.path);
	if (!sizes)
	{
		return cmt::Failure(std::move(sizes.error()));
	}
	if (sizes->empty())
	{
		return Unsupported("no size fits in device and host memory");
	}
	const auto max_bytes = *std::max_element(sizes->begin(), sizes->end());
	auto created = c.path->make(RunContext{c, dev, max_bytes});
	if (!created)
	{
		return cmt::Failure(std::move(created.error()));
	}
	auto runner = std::move(*created);
	const bool pinned_wc = c.path->name == "pinned_ce" && c.alloc == Alloc::HostAllocWc;
	const bool flushing = c.flush && FlushSupported() && c.path->dirs != PathSpec::Dirs::None && !pinned_wc;
	std::size_t emitted = 0;
	bool any_bad = false;
	for (const auto bytes : *sizes)
	{
		auto result = RunSize(c, dev, *runner, bytes, flushing, any_bad, emitted, emit);
		if (!result)
		{
			const auto& error = result.error();
			if (error.IsUnsupported() && emitted > 0)
			{
				return Failure(Error::Kind::Cuda, "benchmark failed after emitting rows", error.GetCode());
			}
			return cmt::Failure(std::move(result.error()));
		}
	}
	if (any_bad)
	{
		return Failure(Error::Kind::Verify, "data verification failed (rows marked 'bad')");
	}
	runner.reset();
	return {};
}

} // namespace cmt

namespace cmt {
namespace {

constexpr std::uint64_t kMaxProbeCpuBytes = 16ull << 20;
constexpr std::uint64_t kMaxProbePinnedBytes = 64ull << 20;

struct CpuProbe
{
	Dir dir = Dir::HtoD;
	LoopName loop;
	std::size_t bytes = 0;
};

Result<std::string> MeasureCpuCopies(const std::vector<CpuProbe>& probes, int samples, const HostBuffer& ordinary,
									 const PinnedAlloc& pinned)
{
	auto allowed = GetThreadAffinity();
	if (!allowed)
	{
		return Failure(allowed.error());
	}
	if (allowed->empty())
	{
		return Unsupported("CPU probe has no allowed logical CPU");
	}
	const unsigned cpu = allowed->front();
	CMT_RETURN_IF_ERROR(SetThreadAffinity(pthread_self(), {cpu}));
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "status", "complete", true);
	JsonField(writer, "pages_requested", PagesName(ordinary.PagePolicy()));
	JsonField(writer, "ordinary_allocation_bytes", ordinary.size());
	JsonField(writer, "ordinary_mapping_huge_kib", ordinary.HugeKiB());
	JsonField(writer, "huge_page_observation",
			  "whole overlapping VMA before CPU samples, not the copied prefix; THP is not guaranteed");
	JsonField(writer, "purpose", "one core's CPU copy rate for the thread estimate; not DMA or pipeline throughput");
	JsonField(writer, "placement",
			  "first allowed CPU under helper-placement policy; not caller-lane throughput; no core sweep");
	JsonField(writer, "preparation",
			  "allocation and first touch on the benchmark caller; pattern fill on the probe CPU; "
			  "no NUMA memory binding; preparation, eviction and checking untimed");
	JsonKey(writer, "cpu_copy_probes");
	writer.start_array();
	for (std::size_t i = 0; i < std::size(probes); i++)
	{
		const auto& probe = probes[i];
		const auto name = LoopNameText(probe.loop);
		const auto selected = hwcopy::Select(name, CopyCtx{}, std::nullopt, hwcopy::GetCpu());
		if (!selected || selected->fell_back)
		{
			return Unsupported("CPU probe cannot execute the requested copy loop");
		}
		auto* source = probe.dir == Dir::HtoD ? ordinary.data() : pinned.data();
		auto* destination = probe.dir == Dir::HtoD ? pinned.data() : ordinary.data();
		const auto seed = SeedFor(probe.bytes, probe.dir == Dir::HtoD ? 0 : 1);
		std::vector<double> seconds;
		std::vector<double> rates;
		for (int sample = 0; sample < samples; sample++)
		{
			PatternFill(source, probe.bytes, seed, 1);
			std::memset(destination, 0, probe.bytes);
			StoreFence();
			FlushRange(ordinary.data(), probe.bytes);
			FlushRange(pinned.data(), probe.bytes);
			const auto started = Clock::now();
			hwcopy::Run(selected->loop, destination, source, probe.bytes);
			StoreFence();
			const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
			if (!std::isfinite(elapsed) || elapsed <= 0 || PatternCheck(destination, probe.bytes, seed, 1) != 0)
			{
				return Failure(Error::Kind::Verify, "direct CPU-copy probe failed timing or correctness");
			}
			seconds.push_back(elapsed);
			rates.push_back(static_cast<double>(probe.bytes) / elapsed / 1e9);
		}
		writer.append_raw(i == 0 ? "" : ",");
		writer.start_object();
		JsonField(writer, "direction", DirName(probe.dir), true);
		JsonField(writer, "bytes", probe.bytes);
		JsonField(writer, "pages_requested", PagesName(ordinary.PagePolicy()));
		JsonField(writer, "cpu", cpu);
		JsonField(writer, "requested_loop", name);
		JsonField(writer, "executed_loop", hwcopy::Describe(selected->loop));
		JsonField(writer, "cache_eviction_supported", FlushSupported());
		JsonField(writer, "execution", "direct_cpu_loop");
		JsonField(writer, "timing_boundary", "cpu_copy_and_store_fence");
		JsonField(writer, "status", "complete");
		JsonField(writer, "verified", true);
		JsonField(writer, "gbps", Summarize(rates).median);
		JsonKey(writer, "seconds");
		writer.start_array();
		for (std::size_t sample = 0; sample < seconds.size(); sample++)
		{
			writer.append_raw(sample == 0 ? "" : ",");
			writer.append_raw(fmt::format("{:.17g}", seconds[sample]));
		}
		writer.end_array();
		writer.end_object();
	}
	writer.end_array();
	writer.end_object();
	return FinishJson(writer);
}

Result<std::string> ProbeCpuCopies(const std::vector<CpuProbe>& probes, int samples, Pages pages,
								   double& setup_seconds)
{
	const auto setup_started = Clock::now();
	std::size_t bytes = 0;
	for (const auto& probe : probes)
	{
		bytes = std::max(bytes, probe.bytes);
	}
	HostBuffer ordinary;
	PinnedAlloc pinned;
	CMT_RETURN_IF_ERROR(Acquire(ordinary, HostBuffer::Create(bytes, pages)));
	CMT_RETURN_IF_ERROR(Acquire(pinned, PinnedAlloc::Create(bytes, HostMem::Cached)));
	CMT_RETURN_IF_ERROR(cu::InitializeDriverApi());
	auto placement = cu::CurrentWorkerPlacement(cu::g_api);
	if (!placement)
	{
		return Failure(placement.error());
	}
	setup_seconds += std::chrono::duration<double>(Clock::now() - setup_started).count();
	Result<std::string> diagnostics = std::string{};
	Thread thread;
	CMT_RETURN_IF_ERROR(
		thread.Start([&] { diagnostics = MeasureCpuCopies(probes, samples, ordinary, pinned); }, *placement));
	CMT_RETURN_IF_ERROR(thread.Join());
	return diagnostics;
}

Result<void> MeasureProbePoint(Measurement& measurement, PathRunner& runner)
{
	const auto bytes = measurement.bytes;
	CMT_RETURN_IF_ERROR(runner.Prepare(bytes));
	CMT_RETURN_IF_ERROR(runner.Sample(bytes, 1));
	CMT_RETURN_IF_ERROR(runner.ClearDestination(bytes));
	for (int i = 0; i < measurement.config.samples; ++i)
	{
		if (measurement.config.flush && FlushSupported())
		{
			CMT_RETURN_IF_ERROR(runner.Flush(bytes));
		}
		auto sample = runner.Sample(bytes, 1);
		if (!sample)
		{
			return Failure(sample.error());
		}
		measurement.samples.push_back(*sample);
	}
	auto wrong = runner.Verify(bytes);
	if (!wrong)
	{
		return Failure(wrong.error());
	}
	if (!*wrong || **wrong != 0)
	{
		return Failure(Error::Kind::Verify, "probe destination failed correctness check");
	}
	measurement.verified = true;
	measurement.huge_kib = runner.HugeKiB();
	measurement.status = "complete";
	measurement.reason.clear();
	return {};
}

Result<Measurement> MeasurePinnedCopy(const Config& config, const DeviceInfo& device, double& setup_seconds)
{
	Measurement measurement;
	measurement.config = config;
	measurement.bytes = config.sizes.front();
	measurement.coverage = "estimate";
	auto fitting = SizesThatFit(config, *config.path, nullptr);
	if (!fitting)
	{
		return Failure(fitting.error());
	}
	if (fitting->empty())
	{
		return Unsupported("the pinned-copy probe does not fit free device or host memory");
	}
	const auto setup_started = Clock::now();
	auto runner = config.path->make(RunContext{config, device, measurement.bytes});
	setup_seconds += std::chrono::duration<double>(Clock::now() - setup_started).count();
	if (!runner)
	{
		return Failure(runner.error());
	}
	measurement.allocation = (*runner)->Alloc();
	CMT_RETURN_IF_ERROR(MeasureProbePoint(measurement, **runner));
	return measurement;
}

Result<Dir> ProbeDirection(Json value)
{
	std::string_view direction;
	if (value.get(direction) || (direction != "htod" && direction != "dtoh"))
	{
		return UsageError("probe direction must be htod or dtoh");
	}
	return direction == "htod" ? Dir::HtoD : Dir::DtoH;
}

} // namespace

Result<int> RunTuneProbe(const std::string& request_path, const std::string& result_path)
{
	const auto started = Clock::now();
	if (std::getenv("LD_PRELOAD") || std::getenv("CMT_CONFIG"))
	{
		return UsageError("tuning probe requires a clean child without preload or selected config");
	}
	auto text = ReadTextFile(request_path, tuning::kMaxConfigBytes);
	if (!text)
	{
		return Failure(text.error());
	}
	JsonParser parser;
	Json request;
	std::string_view schema;
	uint64_t version = 0;
	int64_t device = 0;
	bool metadata_only = false;
	if (parser.parse(*text).get(request) || request["schema"].get(schema) || schema != "cmt.probe" ||
		request["version"].get(version) || version != 2 || request["device"].get(device) || device < 0 ||
		device > INT_MAX || request["metadata_only"].get(metadata_only))
	{
		return UsageError("invalid cmt.probe request");
	}
	auto machine =
		MachineReportFor(static_cast<int>(device), false, std::nullopt, true);
	if (!machine)
	{
		return Failure(machine.error());
	}
	const auto initialized = Clock::now();
	const double initialization_seconds = std::chrono::duration<double>(initialized - started).count();
	if (metadata_only || machine->exit_code != 0)
	{
		CMT_RETURN_IF_ERROR(WriteMeasurements(
			result_path, machine->text, {},
			{{"initialization", initialization_seconds}, {"total_before_output", initialization_seconds}}));
		return machine->exit_code;
	}
	std::string_view requested_pages;
	if (request["pages"].get(requested_pages) || (requested_pages != "small" && requested_pages != "thp"))
	{
		return UsageError("probe caller pages must be small or thp");
	}
	const Pages pages = requested_pages == "thp" ? Pages::Thp : Pages::Small;
	auto device_info = QueryDevice(static_cast<int>(device));
	if (!device_info)
	{
		return Failure(device_info.error());
	}
	CMT_CUDA(cudaSetDevice(static_cast<int>(device)));
	uint64_t samples = 0;
	uint64_t pinned_bytes = 0;
	simdjson::dom::array directions, cpu_copies;
	if (request["samples"].get(samples) || samples < 1 || samples > 10 || request["directions"].get(directions) ||
		directions.size() == 0 || directions.size() > 2 || request["cpu_copies"].get(cpu_copies) ||
		cpu_copies.size() != directions.size() || request["pinned_copy_bytes"].get(pinned_bytes) ||
		pinned_bytes == 0 || pinned_bytes % 16 || pinned_bytes > kMaxProbePinnedBytes)
	{
		return UsageError("probe requires samples, directions, one CPU copy per direction and a pinned copy size");
	}
	std::vector<Config> pinned;
	for (Json value : directions)
	{
		auto dir = ProbeDirection(value);
		if (!dir)
		{
			return Failure(dir.error());
		}
		if (std::any_of(pinned.begin(), pinned.end(), [&](const Config& config) { return config.dir == *dir; }))
		{
			return UsageError("probe directions must be unique");
		}
		Config config;
		config.dir = *dir;
		config.device = static_cast<int>(device);
		config.path = FindPath("pinned_ce");
		config.alloc = Alloc::HostAlloc;
		config.samples = static_cast<int>(samples);
		config.sizes = {static_cast<std::size_t>(pinned_bytes)};
		CMT_RETURN_IF_ERROR(config.Validate());
		pinned.push_back(std::move(config));
	}
	std::vector<CpuProbe> probes;
	for (Json entry : cpu_copies)
	{
		Json direction;
		std::string_view loop_name;
		uint64_t bytes = 0;
		if (entry["direction"].get(direction) || entry["loop"].get(loop_name) || entry["bytes"].get(bytes) ||
			bytes == 0 || bytes % 8 || bytes > kMaxProbeCpuBytes)
		{
			return UsageError("a probe CPU copy needs a direction, a loop and 8-byte-aligned bytes up to 16 MiB");
		}
		auto dir = ProbeDirection(direction);
		auto loop = ParseLoopName(NativeArch(), loop_name);
		if (!dir || !loop ||
			std::none_of(pinned.begin(), pinned.end(), [&](const Config& config) { return config.dir == *dir; }) ||
			std::any_of(probes.begin(), probes.end(), [&](const CpuProbe& probe) { return probe.dir == *dir; }))
		{
			return UsageError("probe CPU copies need a known loop and one requested direction each");
		}
		probes.push_back({*dir, *loop, static_cast<std::size_t>(bytes)});
	}
	const auto cpu_started = Clock::now();
	double setup_seconds = 0;
	auto diagnostics = ProbeCpuCopies(probes, static_cast<int>(samples), pages, setup_seconds);
	if (!diagnostics)
	{
		return Failure(diagnostics.error());
	}
	const double cpu_probe_seconds = std::chrono::duration<double>(Clock::now() - cpu_started).count();
	std::vector<Measurement> measurements;
	for (const auto& config : pinned)
	{
		auto measurement = MeasurePinnedCopy(config, *device_info, setup_seconds);
		if (!measurement)
		{
			return Failure(measurement.error());
		}
		measurements.push_back(std::move(*measurement));
	}
	const auto finished = Clock::now();
	const double probe_seconds = std::chrono::duration<double>(finished - initialized).count();
	CMT_RETURN_IF_ERROR(WriteMeasurements(
		result_path, machine->text, measurements,
		{{"initialization", initialization_seconds},
		 {"resource_setup", setup_seconds},
		 {"cpu_probe_wall", cpu_probe_seconds},
		 {"probe", std::max(0.0, probe_seconds - setup_seconds)},
		 {"total_before_output", std::chrono::duration<double>(finished - started).count()}},
		*diagnostics));
	return 0;
}

namespace {

std::optional<std::string> ServedBy(const void* p)
{
	std::ifstream smaps("/proc/self/smaps");
	const auto address = reinterpret_cast<std::uintptr_t>(p);
	bool in = false;
	bool shared = false;
	for (std::string line; std::getline(smaps, line);)
	{
		const std::size_t dash = line.find('-');
		const std::size_t space = line.find(' ');
		if (dash != std::string::npos && space != std::string::npos && dash < space &&
			line.find_first_not_of("0123456789abcdef") == dash)
		{
			unsigned long long low = 0;
			unsigned long long high = 0;
			std::from_chars(line.data(), line.data() + dash, low, 16);
			std::from_chars(line.data() + dash + 1, line.data() + space, high, 16);
			in = low <= address && address < high;
			shared = space + 4 < std::size(line) && line[space + 4] == 's';
			continue;
		}
		if (in && line.rfind("VmFlags:", 0) == 0)
		{
			const bool advised = (line + " ").find(" hg ") != std::string::npos;
			return std::string(!advised ? "driver" : shared ? "shared" : "private");
		}
	}
	return std::nullopt;
}

} // namespace

Result<int> RunTuneHostAlloc(const std::string& request_path, const std::string& result_path)
{
	const auto started = Clock::now();
	auto text = ReadTextFile(request_path, tuning::kMaxConfigBytes);
	if (!text)
	{
		return Failure(text.error());
	}
	JsonParser parser;
	Json request;
	std::string_view schema;
	std::string_view arm;
	std::uint64_t version = 0;
	std::int64_t device = 0;
	std::uint64_t bytes = 0;
	constexpr std::uint64_t kCheckBytes = 4096;
	if (parser.parse(*text).get(request) || request["schema"].get(schema) || schema != "cmt.host_alloc_probe" ||
		request["version"].get(version) || version != 1 || request["device"].get(device) || device < 0 ||
		device > INT_MAX || request["bytes"].get(bytes) || bytes < 2 * kCheckBytes || bytes % kCheckBytes != 0 ||
		request["arm"].get(arm) || (arm != "driver" && arm != "shared" && arm != "private"))
	{
		return UsageError("invalid cmt.host_alloc_probe request");
	}
	if ((arm == "driver") != (std::getenv("LD_PRELOAD") == nullptr))
	{
		return UsageError("the host allocation child's preload does not match its arm");
	}
	CMT_CUDA(cudaSetDevice(static_cast<int>(device)));
	CMT_CUDA(cudaFree(nullptr));
	const auto initialized = Clock::now();
	void* host = nullptr;
	const auto allocating = Clock::now();
	CMT_CUDA(cudaHostAlloc(&host, bytes, cudaHostAllocDefault));
	const auto allocated = Clock::now();
	struct Release
	{
		void* host;
		~Release()
		{
			if (host != nullptr)
			{
				cudaFreeHost(host);
			}
		}
	} release{host};
	const auto served_by = ServedBy(host);
	const auto anon_huge = SmapsKiB(host, bytes, "AnonHugePages");
	const auto shmem_huge = SmapsKiB(host, bytes, "ShmemPmdMapped");
	if (!served_by || !anon_huge || !shmem_huge)
	{
		return Failure(Error::Kind::Os, "cannot read the allocation's mapping in /proc/self/smaps");
	}
	auto* bytes_at = static_cast<unsigned char*>(host);
	for (std::uint64_t i = 0; i < kCheckBytes; i++)
	{
		bytes_at[i] = static_cast<unsigned char>(i * 7 + 1);
	}
	auto check = DeviceBuffer::Create(kCheckBytes);
	if (!check)
	{
		return Failure(check.error());
	}
	CMT_CUDA(cudaMemcpy(std::data(*check), host, kCheckBytes, cudaMemcpyHostToDevice));
	CMT_CUDA(cudaMemcpy(bytes_at + bytes - kCheckBytes, std::data(*check), kCheckBytes, cudaMemcpyDeviceToHost));
	if (std::memcmp(bytes_at, bytes_at + bytes - kCheckBytes, kCheckBytes) != 0)
	{
		return Failure(Error::Kind::Verify, "the pinned allocation's copy through the GPU returned other bytes");
	}
	release.host = nullptr;
	const auto freeing = Clock::now();
	CMT_CUDA(cudaFreeHost(host));
	const auto freed = Clock::now();
	const auto seconds = [](Clock::time_point from, Clock::time_point to)
	{ return std::chrono::duration<double>(to - from).count(); };
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "schema", "cmt.host_alloc_result", true);
	JsonField(writer, "version", 1);
	JsonField(writer, "arm", arm);
	JsonField(writer, "bytes", bytes);
	JsonField(writer, "served_by", *served_by);
	JsonField(writer, "alloc_seconds", seconds(allocating, allocated));
	JsonField(writer, "free_seconds", seconds(freeing, freed));
	JsonField(writer, "huge_kib", static_cast<std::uint64_t>(std::max(0L, *anon_huge + *shmem_huge)));
	JsonField(writer, "verified", true);
	JsonKey(writer, "timings_seconds");
	writer.start_object();
	JsonField(writer, "initialization", seconds(started, initialized), true);
	JsonField(writer, "total_before_output", seconds(started, Clock::now()));
	writer.end_object();
	writer.end_object();
	auto result = FinishJson(writer);
	if (!result)
	{
		return Failure(result.error());
	}
	CMT_RETURN_IF_ERROR(WriteTextFile(result_path, *result));
	return 0;
}

} // namespace cmt
