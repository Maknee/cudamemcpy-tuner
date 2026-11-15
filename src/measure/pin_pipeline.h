#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "common/defines.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "common/aliases.h"
#include "cuda/cuda.h"

namespace cmt {
namespace copying {
namespace runtime {

class Worker
{
public:
	template <class Callable>
	[[nodiscard]] static Result<std::unique_ptr<Worker>>
	Create(Callable task, int device = -1, const std::vector<unsigned>& explicit_cpus = {},
		   const std::vector<unsigned>& allowed_cpus = {})
	{
		auto placement = CudaWorkerPlacement(device, explicit_cpus);
		if (!placement)
		{
			return Failure(placement.error());
		}
		placement->allowed_cpus = allowed_cpus;
		auto worker = std::unique_ptr<Worker>(new Worker(std::move(task)));
		auto started =
			worker->thread.Start([state = worker.get()] { state->result = state->function(); }, *placement);
		if (!started)
		{
			return Failure(std::move(started.error()));
		}
		return worker;
	}

	Worker(const Worker&) = delete;
	Worker& operator=(const Worker&) = delete;

	[[nodiscard]] Result<void> Join()
	{
		auto joined = thread.Join();
		if (!joined)
		{
			return Failure(std::move(joined.error()));
		}
		return result;
	}

private:
	explicit Worker(Function<Result<void>()> task) : function(std::move(task))
	{
	}

	Function<Result<void>()> function;
	Result<void> result;
	Thread thread;
};

enum class ReleasePolicy : std::uint8_t
{
	End,
	Eager,
	Preregister
};

struct PinPipelineOptions
{
	int device = 0;
	std::size_t step_bytes = 32u << 20;
	ReleasePolicy release = ReleasePolicy::End;
};

class PinPipelineCopy final
{
public:
	PinPipelineCopy();
	~PinPipelineCopy();
	PinPipelineCopy(const PinPipelineCopy&) = delete;
	PinPipelineCopy& operator=(const PinPipelineCopy&) = delete;
	[[nodiscard]] Result<void> Setup(const PinPipelineOptions&, unsigned char* host, std::size_t bytes,
									 const Stream&);
	[[nodiscard]] Result<double> Copy(Dir direction, unsigned char* device, std::uint32_t reps);
	void Teardown() noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
} // namespace runtime
} // namespace copying
} // namespace cmt
