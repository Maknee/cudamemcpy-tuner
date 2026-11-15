#include <atomic>
#include <benchmark/benchmark.h>
#include <cstring>
#include <thread>
#include <vector>
#include "evaluation/microbenchmarks/thread_pool.h"

// Pool round-trip cost including the minimum supported copy, not an empty
// synthetic notification. Idle/busy variants expose scheduling sensitivity.

namespace {
bool failed = false;

struct BusyCpu
{
	std::atomic<bool> stop{false};
	cmt::Thread thread;

	cmt::Result<void> Start(bool enabled)
	{
		if (enabled)
		{
			return thread.Start(
				[this]
				{
					// This flag carries no payload; relaxed accesses only request termination.
					while (!stop.load(std::memory_order_relaxed))
					{
						cmt::CpuRelax();
					}
				});
		}
		return {};
	}

	~BusyCpu()
	{
		stop.store(true, std::memory_order_relaxed);
		if (thread.Joinable())
		{
			auto joined = thread.Join();
			if (!joined)
			{
				cmt::Panic(joined.error().Message());
			}
		}
	}
};

void measure(benchmark::State& state, cmt::PoolProtocol protocol)
{
	const auto bytes = static_cast<std::size_t>(state.range(0));
	cmt::PoolOptions options;
	options.workers = static_cast<unsigned>(state.range(1));
	options.protocol = protocol;
	auto created = cmt::ThreadPool::Create(options);
	if (!created || (options.workers != 0 && !*created))
	{
		state.SkipWithError(created ? "requested worker pool is disabled" : created.error().Message());
		failed = true;
		return;
	}
	auto pool = std::move(*created);
	constexpr std::size_t guard = 64;
	std::vector<uint8_t> src(bytes + 2 * guard, 0xa5), dst(bytes + 2 * guard, 0xa5);
	for (std::size_t i = 0; i < bytes; ++i)
	{
		src[guard + i] = static_cast<uint8_t>((i * 31) ^ (i >> 8) ^ (i >> 16));
	}
	// Use the same CPU loop in the serial control and every pool participant.
	const cmt::CopyLoop loop{cmt::VariantId::Memcpy, {}};
	auto copy = [&]
	{
		if (pool)
		{
			return pool->Copy1D(loop, dst.data() + guard, src.data() + guard, bytes);
		}
		cmt::hwcopy::Run(loop, dst.data() + guard, src.data() + guard, bytes);
		cmt::StoreFence();
		return true;
	};
	// Validate changed input on two consecutive jobs before timing. A repeated
	// constant payload would not detect a worker that only ran its first job.
	for (unsigned round = 0; round < 2; ++round)
	{
		for (std::size_t i = 0; i < bytes; ++i)
		{
			src[guard + i] ^= 0x3d;
		}
		std::memset(dst.data() + guard, 0, bytes);
		if (!copy() || src != dst)
		{
			state.SkipWithError("copy failed byte or guard verification before timing");
			failed = true;
			return;
		}
	}
	BusyCpu busy;
	auto started = busy.Start(state.range(2) != 0);
	if (!started)
	{
		state.SkipWithError(started.error().Message());
		failed = true;
		return;
	}
	for (auto iteration : state)
	{
		(void)iteration;
		if (!copy())
		{
			state.SkipWithError("pool refused the copy");
			failed = true;
			break;
		}
		benchmark::ClobberMemory();
	}
	if (src != dst)
	{
		state.SkipWithError("pool returned before the copied bytes were visible");
		failed = true;
	}
	state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(bytes));
	state.counters["copy_threads"] = options.workers + 1;
	state.counters["abseil_mutex"] = 1;
}

} // namespace

int main(int argc, char** argv)
{
	cmt::InstallFatalAllocationHandler();
	benchmark::Initialize(&argc, argv);
	if (benchmark::ReportUnrecognizedArguments(argc, argv))
	{
		return 2;
	}
	auto* serial = benchmark::RegisterBenchmark("serial", measure, cmt::PoolProtocol::Condvar);
	serial->ArgNames({"bytes", "workers", "busy"})->UseRealTime();
	for (const std::int64_t bytes : {1 << 20, 8 << 20, 16 << 20, 64 << 20})
	{
		for (const std::int64_t busy : {0, 1})
		{
			serial->Args({bytes, 0, busy});
		}
	}
	for (const auto protocol : {cmt::PoolProtocol::Condvar, cmt::PoolProtocol::Mpmc})
	{
		auto* bench = benchmark::RegisterBenchmark(
			protocol == cmt::PoolProtocol::Condvar ? "condvar" : "mpmc", measure, protocol);
		bench->ArgNames({"bytes", "workers", "busy"})->UseRealTime();
		for (const std::int64_t bytes : {1 << 20, 8 << 20, 16 << 20, 64 << 20})
		{
			for (const std::int64_t workers : {1, 3})
			{
				for (const std::int64_t busy : {0, 1})
				{
					bench->Args({bytes, workers, busy});
				}
			}
		}
	}
	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
	return failed ? 6 : 0;
}
