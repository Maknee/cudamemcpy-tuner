// Test-owned paths exercise the measurement loop's verification and cleanup.
// A deliberately broken path copies only during warm-up; samples must expose it.
#include <cstdio>
#include <utility>
#include <string>
#include <vector>

#include "tests/check.h"
#include "cuda/cuda.h"
#include "machine.h"
#include "measure/copy.h"
#include "measure/metrics.h"

using namespace cmt;

namespace {

/// A device-to-device copy: what a new benchmark looks like, start to end.
class DtoD final : public PathRunner
{
public:
	static Result<std::unique_ptr<PathRunner>> Create(const RunContext& ctx, bool lazy)
	{
		auto owned = std::unique_ptr<DtoD>(new DtoD(lazy));
		auto source = DeviceBuffer::Create(ctx.max_bytes);
		if (!source)
		{
			return cmt::Failure(source.error());
		}
		owned->src_ = std::move(*source);
		auto destination = DeviceBuffer::Create(ctx.max_bytes);
		if (!destination)
		{
			return cmt::Failure(destination.error());
		}
		owned->dst_ = std::move(*destination);
		auto stream = Stream::Create();
		if (!stream)
		{
			return cmt::Failure(stream.error());
		}
		owned->s_ = std::move(*stream);
		return std::unique_ptr<PathRunner>(std::move(owned));
	}

	Result<void> Prepare(std::size_t b) override
	{
		first_since_prepare_ = true;
		CMT_RETURN_IF_ERROR(LaunchPatternFill(std::data(src_), b, 7 + b, s_.Get()));
		return ClearDestination(b);
	}

	Result<void> ClearDestination(std::size_t b) override
	{
		CMT_CUDA(cudaMemsetAsync(std::data(dst_), 0, b, s_.Get()));
		CMT_RETURN_IF_ERROR(s_.Sync());
		return {};
	}

	Result<SampleOut> Sample(std::size_t b, std::uint32_t reps) override
	{
		// the lazy path copies only in the first sample after prepare(), which the loop uses as its warm-up
		const bool copy = !lazy_ || std::exchange(first_since_prepare_, false);
		for (std::uint32_t r = 0; r < reps && copy; r++)
		{
			CMT_CUDA(
				cudaMemcpyAsync(std::data(dst_), std::data(src_), b, cudaMemcpyDeviceToDevice, s_.Get()));
		}
		CMT_RETURN_IF_ERROR(s_.Sync());
		SampleOut o;
		o.seconds = 1e-3; // the rate is not the point of this test
		return o;
	}

	Result<std::optional<std::size_t>> Verify(std::size_t b) override
	{
		auto checked = DevicePatternCheck(std::data(dst_), b, 7 + b, s_.Get());
		if (!checked)
		{
			return cmt::Failure(checked.error());
		}
		return std::optional<std::size_t>(*checked);
	}

	Result<void> Flush(std::size_t) override
	{
		return {};
	}

	std::string Alloc() const override
	{
		return "device";
	}

private:
	explicit DtoD(bool lazy) : lazy_(lazy)
	{
	}

	bool lazy_;
	bool first_since_prepare_ = false;
	DeviceBuffer src_, dst_;
	Stream s_;
};

PathSpec spec(const char* name, bool lazy)
{
	PathSpec p;
	p.name = name;
	p.summary = lazy ? "test: copies only in the warm-up" : "test: device-to-device copy";
	p.dirs = PathSpec::Dirs::None;
	p.device_bytes = 2;
	p.host_bytes = 0;
	p.make = [lazy](const RunContext& ctx) { return DtoD::Create(ctx, lazy); };
	return p;
}

DeviceInfo device_info;

const DeviceInfo& dev()
{
	return device_info;
}

} // namespace

TEST(the_loop_runs_and_verifies_it_and_nothing_leaks)
{
	const long before = Live::DeviceBuffers() + Live::Streams();
	const auto path = spec("test_dtod", false);
	Config config;
	config.path = &path;
	config.sizes = {4u << 10, 1u << 20};
	config.samples = 2;
	config.min_bytes = 1u << 20;
	CHECK(config.Validate());
	std::vector<MeasurementRow> rows;
	CHECK(Run(config, dev(),
			  [&](const MeasurementRow& r) -> Result<void>
			  {
				  rows.push_back(r);
				  return {};
			  }));
	CHECK_EQ(std::size(rows), std::size_t{2});
	for (const MeasurementRow& r : rows)
	{
		CHECK_EQ(r.path, std::string("test_dtod"));
		CHECK(r.verified == Verified::Ok);
		CHECK_EQ(r.dir, std::string("-"));
		CHECK_EQ(r.alloc, std::string("device"));
	}
	CHECK_EQ(Live::DeviceBuffers() + Live::Streams(), before);
}

TEST(a_sample_that_copies_nothing_fails_verification)
{
	const auto path = spec("test_lazy", true);
	Config config;
	config.path = &path;
	config.sizes = {1u << 20};
	config.samples = 2;
	config.min_bytes = 1u << 20;
	CHECK(config.Validate());
	std::vector<MeasurementRow> rows;
	CHECK_ERROR_KIND(Run(config, dev(),
						 [&](const MeasurementRow& r) -> Result<void>
						 {
							 rows.push_back(r);
							 return {};
						 }),
					 Verify);
	CHECK_EQ(std::size(rows), std::size_t{1});
	if (!std::empty(rows))
	{
		CHECK(rows[0].verified == Verified::Bad);
	}
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
