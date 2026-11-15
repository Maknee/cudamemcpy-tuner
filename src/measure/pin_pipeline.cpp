#include "measure/pin_pipeline.h"

namespace cmt {
namespace copying {
namespace runtime {

namespace {
class StepQueue
{
public:
	void Publish(std::size_t submitted_count)
	{
		std::lock_guard<Mutex> l(m);
		submitted = submitted_count;
		cv.SignalAll();
	}

	void Released(std::size_t n)
	{
		std::lock_guard<Mutex> l(m);
		released = n;
		cv.SignalAll();
	}

	void Close()
	{
		std::lock_guard<Mutex> l(m);
		closed = true;
		cv.SignalAll();
	}

	bool WaitSubmitted(std::size_t i)
	{
		std::unique_lock<Mutex> l(m);
		while (submitted <= i && !closed)
		{
			cv.Wait(l.mutex());
		}
		return submitted > i;
	}

	bool WaitReleased(std::size_t n)
	{
		std::unique_lock<Mutex> l(m);
		while (released < n && !closed)
		{
			cv.Wait(l.mutex());
		}
		return released >= n;
	}

private:
	Mutex m;
	ConditionVariable cv;
	std::size_t submitted = 0;
	std::size_t released = 0;
	bool closed = false;
};

struct CloseOnExit
{
	StepQueue& q;

	~CloseOnExit()
	{
		q.Close();
	}
};

} // namespace

struct PinPipelineCopy::Impl
{
	PinPipelineOptions shape;
	unsigned char* host;
	std::size_t b;
	const Stream& s;
	std::vector<HostRegistration> pre;

	Impl(const PinPipelineOptions& configuration, unsigned char* host_memory, std::size_t bytes,
		 const Stream& stream)
		: shape(configuration), host(host_memory), b(bytes), s(stream)
	{
	}

	Result<void> Setup()
	{
		if (shape.release == ReleasePolicy::Preregister)
		{
			for (std::size_t off = 0; off < b;)
			{
				const auto length = std::min(shape.step_bytes, b - off);
				auto registration = HostRegistration::Create(host + off, length);
				if (!registration)
				{
					return Failure(std::move(registration.error()));
				}
				pre.push_back(std::move(*registration));
				off += length;
			}
		}
		return {};
	}

	Result<double> Copy(Dir direction, unsigned char* device, std::uint32_t reps)
	{
		const std::size_t n = 1 + (b - 1) / shape.step_bytes;
		if (n > std::numeric_limits<std::size_t>::max() / reps)
		{
			return Failure(Error::Kind::Usage, "pin pipeline: step count times repetitions overflows");
		}
		const std::size_t total = n * static_cast<std::size_t>(reps);
		const bool eager = shape.release == ReleasePolicy::Eager;
		StepQueue q;
		std::vector<HostRegistration> reg;
		std::vector<Event> ev;
		if (total > ev.max_size() || (shape.release != ReleasePolicy::Preregister && total > reg.max_size()))
		{
			return Failure(Error::Kind::Usage, "pin pipeline: step count exceeds container capacity");
		}
		if (shape.release != ReleasePolicy::Preregister)
		{
			reg.resize(total);
		}
		ev.reserve(total);
		for (std::size_t i = 0; i < total; i++)
		{
			auto event = Event::Create(Timing::Disabled);
			if (!event)
			{
				return Failure(std::move(event.error()));
			}
			ev.push_back(std::move(*event));
		}
		std::unique_ptr<Worker> unpinner;
		if (eager)
		{
			auto worker = Worker::Create(
				[&]() -> Result<void>
				{
					CloseOnExit close_on_failure{
						q};
					CMT_CUDA(cudaSetDevice(shape.device));
					for (std::size_t i = 0; i < total; i++)
					{
						if (!q.WaitSubmitted(i))
						{
							return {};
						}
						CMT_RETURN_IF_ERROR(ev[i].Sync());
						CMT_RETURN_IF_ERROR(reg[i].Release());
						q.Released(i + 1);
					}
					return {};
				},
				shape.device);
			if (!worker)
			{
				return Failure(std::move(worker.error()));
			}
			unpinner = std::move(*worker);
		}
		CloseOnExit close_guard{
			q};
		DrainOnExit drain{s};
		const double t0 = NowSeconds();
		for (std::uint32_t r = 0; r < reps; r++)
		{
			for (std::size_t i = 0; i < n; i++)
			{
				const std::size_t idx = static_cast<std::size_t>(r) * n + i;
				const std::size_t off = i * shape.step_bytes;
				const std::size_t len = std::min(shape.step_bytes, b - off);
				if (shape.release != ReleasePolicy::Preregister)
				{
					auto registration = HostRegistration::Create(host + off, len);
					if (!registration)
					{
						return Failure(std::move(registration.error()));
					}
					reg[idx] = std::move(*registration);
				}
				if (direction == Dir::HtoD)
				{
					CMT_CUDA(cudaMemcpyAsync(device + off, host + off, len, cudaMemcpyHostToDevice, s.Get()));
				}
				else
				{
					CMT_CUDA(cudaMemcpyAsync(host + off, device + off, len, cudaMemcpyDeviceToHost, s.Get()));
				}
				CMT_RETURN_IF_ERROR(ev[idx].Record(s));
				q.Publish(idx + 1);
			}
			CMT_RETURN_IF_ERROR(s.Sync());
			if (eager)
			{
				if (!q.WaitReleased((static_cast<std::size_t>(r) + 1) * n))
				{
					break;
				}
			}
			else if (shape.release == ReleasePolicy::End)
			{
				for (std::size_t i = 0; i < n; i++)
				{
					CMT_RETURN_IF_ERROR(reg[r * n + i].Release());
				}
			}
		}
		const double t1 = NowSeconds();
		if (unpinner)
		{
			CMT_RETURN_IF_ERROR(unpinner->Join());
		}
		return t1 - t0;
	}
};

PinPipelineCopy::PinPipelineCopy() = default;

PinPipelineCopy::~PinPipelineCopy()
{
	Teardown();
}

Result<void> PinPipelineCopy::Setup(const PinPipelineOptions& shape, unsigned char* host, std::size_t bytes,
									const Stream& stream)
{
	if (shape.step_bytes == 0 || shape.step_bytes % BasePageBytes())
	{
		return Failure(Error::Kind::Usage,
					   "pin pipeline: steps must be a positive whole number of Linux pages");
	}
	if (!host || bytes == 0)
	{
		return Failure(Error::Kind::Usage, "pin pipeline: host range must be nonempty");
	}
	if (shape.release != ReleasePolicy::End && shape.release != ReleasePolicy::Eager &&
		shape.release != ReleasePolicy::Preregister)
	{
		return Failure(Error::Kind::Usage, "pin pipeline: unknown registration release policy");
	}
	Teardown();
	auto next = std::make_unique<Impl>(shape, host, bytes, stream);
	CMT_RETURN_IF_ERROR(next->Setup());
	impl = std::move(next);
	return {};
}

Result<double> PinPipelineCopy::Copy(Dir direction, unsigned char* device, std::uint32_t reps)
{
	if (!impl)
	{
		Panic("pin pipeline copy before setup");
	}
	if (direction != Dir::HtoD && direction != Dir::DtoH)
	{
		return Failure(Error::Kind::Usage, "pin pipeline direction must be htod or dtoh");
	}
	if (!device || reps == 0)
	{
		return Failure(Error::Kind::Usage, "pin pipeline: device range and repetitions must be nonzero");
	}
	return impl->Copy(direction, device, reps);
}

void PinPipelineCopy::Teardown() noexcept
{
	impl.reset();
}
} // namespace runtime
} // namespace copying
} // namespace cmt
