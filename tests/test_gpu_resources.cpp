// Lifetime tracking (registry pattern): every CUDA object a path creates is released, after a normal Run AND
// after a Run that fails half-way with copies in flight. A leaked registration per sample would otherwise
// surface only as a slow drift (or an out-of-memory) late in a long sweep.
#include <utility>

#include "tests/check.h"
#include "cli.h"
#include "cuda/cuda.h"
#include "machine.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "measure/copy.h"
#include "measure/metrics.h"

using namespace cmt;

namespace {

struct Counts
{
	long dev, pin, reg, str, ev, host;

	static Counts now()
	{
		return {Live::DeviceBuffers(), Live::PinnedAllocs(), Live::Registrations(),
				Live::Streams(),	   Live::Events(),		 HostBuffer::Live()};
	}
};

void expect_same(const Counts& a, const Counts& b, const std::string& where)
{
	if (a.dev != b.dev || a.pin != b.pin || a.reg != b.reg || a.str != b.str || a.ev != b.ev ||
		a.host != b.host)
	{
		check::failures()++;
		std::printf("  %s: dev %ld->%ld pinned %ld->%ld reg %ld->%ld Streams %ld->%ld Events %ld->%ld "
					"host %ld->%ld\n",
					where.c_str(), a.dev, b.dev, a.pin, b.pin, a.reg, b.reg, a.str, b.str, a.ev, b.ev, a.host,
					b.host);
	}
}

DeviceInfo device_info;

const DeviceInfo& dev()
{
	return device_info;
}

Result<void> run_quiet(const std::vector<std::string>& args)
{
	const auto o = ParseArgs(args);
	if (!o)
	{
		return cmt::Failure(o.error());
	}
	return Run(o->cfg, dev(), [](const MeasurementRow&) -> Result<void> { return {}; });
}

/// A path that queues copies and then fails inside the timed sample: the stream must be drained and every
/// object released when the failed Result propagates.
class FailMidSample final : public PathRunner
{
public:
	static Result<std::unique_ptr<PathRunner>> Create(const RunContext& ctx)
	{
		auto owned = std::unique_ptr<FailMidSample>(new FailMidSample);
		auto host = HostBuffer::Create(ctx.max_bytes, Pages::Small);
		if (!host)
		{
			return cmt::Failure(host.error());
		}
		owned->host_ = std::move(*host);
		auto registration = HostRegistration::Create(std::data(owned->host_), ctx.max_bytes);
		if (!registration)
		{
			return cmt::Failure(registration.error());
		}
		owned->reg_ = std::move(*registration);
		auto device = DeviceBuffer::Create(ctx.max_bytes);
		if (!device)
		{
			return cmt::Failure(device.error());
		}
		owned->dev_ = std::move(*device);
		auto stream = Stream::Create();
		if (!stream)
		{
			return cmt::Failure(stream.error());
		}
		owned->s_ = std::move(*stream);
		return std::unique_ptr<PathRunner>(std::move(owned));
	}

	Result<void> Prepare(std::size_t) override
	{
		return {};
	}

	Result<void> ClearDestination(std::size_t) override
	{
		return {};
	}

	Result<SampleOut> Sample(std::size_t b, std::uint32_t) override
	{
		for (std::size_t i = 0; i < 8; i++)
		{
			CMT_CUDA(cudaMemcpyAsync(std::data(dev_), std::data(host_), b, cudaMemcpyHostToDevice, s_.Get()));
		}
		if (++calls_ > 1)
		{
			return Failure(Error::Kind::Cuda, "injected failure with 8 copies in flight");
		}
		CMT_RETURN_IF_ERROR(s_.Sync());
		SampleOut out;
		out.seconds = 1e-3;
		return out;
	}

	Result<std::optional<std::size_t>> Verify(std::size_t) override
	{
		return std::optional<std::size_t>{};
	}

	Result<void> Flush(std::size_t) override
	{
		return {};
	}

	std::string Alloc() const override
	{
		return "register";
	}

private:
	std::size_t calls_ = 0;
	HostBuffer host_;
	HostRegistration reg_;
	DeviceBuffer dev_;
	Stream s_;
};

} // namespace

TEST(every_path_and_variant_releases_everything)
{
	const std::vector<std::vector<std::string>> variants = {
		{"--path", "pinned_ce", "--alloc", "hostalloc"},
		{"--path", "bidir_ce", "--alloc", "hostalloc"},
		{"--path", "staged", "--threads", "2", "--wc"},
		{"--path", "staged", "--dir", "dtoh", "--threads", "3", "--slot-bytes", "64K"},
		{"--path", "staged", "--dir", "dtoh", "--threads", "3", "--slot-bytes", "64K", "--wc"},
		{"--path", "pin_pipeline", "--unpin", "eager", "--pin-step", "1M"},
		{"--path", "pin_pipeline", "--unpin", "prereg", "--pin-step", "1M"},
		{"--path", "pin_pipeline", "--dir", "dtoh", "--unpin", "end", "--pin-step", "1M"},
		{"--path", "pin_pipeline", "--dir", "dtoh", "--unpin", "eager", "--pin-step", "1M"},
		{"--path", "pin_pipeline", "--dir", "dtoh", "--unpin", "prereg", "--pin-step", "1M"}};
	std::vector<std::vector<std::string>> all;
	for (const PathSpec& p : BenchmarkPaths())
	{
		all.push_back({"--path", p.name});
	}
	all.insert(std::end(all), std::begin(variants), std::end(variants));
	for (auto args : all)
	{
		args.insert(std::end(args), {"--sizes", "64K,3M", "--samples", "2", "--min-bytes", "1M"});
		const Counts before = Counts::now();
		CHECK(run_quiet(args));
		std::string l;
		for (const auto& a : args)
		{
			l += a + " ";
		}
		expect_same(before, Counts::now(), l);
	}
}

TEST(a_failure_with_copies_in_flight_releases_everything)
{
	PathSpec path;
	path.name = "test_fail_mid_sample";
	path.dirs = PathSpec::Dirs::None;
	path.make = [](const RunContext& ctx) { return FailMidSample::Create(ctx); };
	Config config;
	config.path = &path;
	config.sizes = {16u << 20};
	config.samples = 2;
	CHECK(config.Validate());
	const Counts before = Counts::now();
	CHECK_ERROR_KIND(Run(config, dev(), [](const MeasurementRow&) -> Result<void> { return {}; }), Cuda);
	expect_same(before, Counts::now(), "failure mid-sample");
}

TEST(registration_lifetime)
{
	const long base = Live::Registrations();
	auto b_result = HostBuffer::Create(4u << 20, Pages::Small);
	CHECK(b_result);
	if (!b_result)
	{
		return;
	}
	auto b = std::move(*b_result);
	{
		auto r_result = HostRegistration::Create(std::data(b), std::size(b));
		CHECK(r_result);
		if (!r_result)
		{
			return;
		}
		auto r = std::move(*r_result);
		CHECK_EQ(Live::Registrations(), base + 1);
		HostRegistration m(std::move(r));
		CHECK(!r.Active());
		CHECK(m.Active());
		CHECK_EQ(Live::Registrations(), base + 1);
		CHECK(m.Release());
		CHECK_EQ(Live::Registrations(), base);
		CHECK(m.Release()); // second release is a no-op
		CHECK_EQ(Live::Registrations(), base);
		auto again_result = HostRegistration::Create(std::data(b), std::size(b));
		CHECK(again_result);
		if (!again_result)
		{
			return;
		}
		auto again = std::move(*again_result); // the range can be registered again after release
		CHECK_EQ(Live::Registrations(), base + 1);
	}
	CHECK_EQ(Live::Registrations(), base);
}

TEST(double_registration_is_an_error_and_leaks_nothing)
{
	const Counts before = Counts::now();
	{
		auto b_result = HostBuffer::Create(2u << 20, Pages::Small);
		CHECK(b_result);
		if (!b_result)
		{
			return;
		}
		auto b = std::move(*b_result);
		auto r_result = HostRegistration::Create(std::data(b), std::size(b));
		CHECK(r_result);
		if (!r_result)
		{
			return;
		}
		auto r = std::move(*r_result);
		CHECK_ERROR_KIND(HostRegistration::Create(std::data(b), std::size(b)), Cuda); // already registered
	}
	expect_same(before, Counts::now(), "double registration");
}

TEST(too_large_device_buffer_is_unsupported_and_zero_is_a_bug)
{
	const Counts before = Counts::now();
	CHECK_ERROR_KIND(DeviceBuffer::Create(std::size_t{1} << 46), Unsupported);
	CHECK_PANICS((void)DeviceBuffer::Create(0));
	CHECK_PANICS(
		(void)PinnedAlloc::Create(0, HostMem::Cached)); // no counter drift from a zero-size object
	expect_same(before, Counts::now(), "oversized and zero-size allocations");
}

int main()
{
	auto count = CudaDeviceCount();
	if (!count)
	{
		std::fprintf(stderr, "%s\n", count.error().Message());
		return count.error().ExitCode();
	}
	if (*count == 0)
	{
		std::printf("no CUDA device: skipped\n");
		return 77;
	}
	auto info = QueryDevice(0);
	if (!info)
	{
		std::fprintf(stderr, "%s\n", info.error().Message());
		return info.error().ExitCode();
	}
	device_info = std::move(*info);
	return check::run_all();
}
