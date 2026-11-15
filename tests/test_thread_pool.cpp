// test_thread_pool.cpp - split planning and both pool protocols with
// 1..16 copy threads (caller + 0..15 workers): 1D and pitched 2D/3D
// copies, byte-verified, pitch padding must stay untouched. Also a burst
// of back-to-back rounds with condition-variable parking.
#include <cstring>
#include <initializer_list>
#include <memory>
#include <limits>

#include "check.h"
#include "evaluation/microbenchmarks/thread_pool.h"
#include "copying/chunk_pipeline.h"

namespace {

using cmt::Copy2DView;
using cmt::PoolOptions;
using cmt::PoolProtocol;
using cmt::ThreadPool;

constexpr uint8_t kPad = 0x5A;

// Exercise memcpy and the CPU's default loop. A job uses one loop on every
// thread, but successive jobs on the same persistent pool may select different loops.
const cmt::CopyLoop kMemcpy{cmt::VariantId::Memcpy, {}};
const cmt::CopyLoop kDefault = cmt::hwcopy::DefaultLoop({}, std::nullopt, cmt::hwcopy::GetCpu());

const char* name(PoolProtocol p)
{
	return p == PoolProtocol::Mpmc ? "mpmc" : "condvar";
}

void check_1d(ThreadPool* pool, unsigned threads, PoolProtocol proto, std::size_t bytes, unsigned seed)
{
	check::Buffer src(bytes + 64);
	check::Buffer dst(bytes + 64, kPad);
	for (std::size_t i = 0; i < std::size(src); i++)
	{
		src[i] = check::pattern(i, seed);
	}
	const bool ran = pool != nullptr && pool->Copy1D(kMemcpy, std::data(dst) + 3, std::data(src) + 1, bytes);
	if (!ran)
	{
		CHECKF(pool == nullptr || bytes < cmt::kMinParallelCopyBytes, "%s t=%u refused %zu bytes",
			   name(proto), threads, bytes);
		cmt::hwcopy::Run(kMemcpy, std::data(dst) + 3, std::data(src) + 1, bytes);
	}
	CHECKF(std::memcmp(std::data(dst) + 3, std::data(src) + 1, bytes) == 0, "%s t=%u 1d bytes=%zu",
		   name(proto), threads, bytes);
	CHECK(dst[0] == kPad && dst[1] == kPad && dst[2] == kPad && dst[3 + bytes] == kPad);
}

// width x height rows per slice, depth slices; pitches add padding.
void check_2d(ThreadPool* pool, unsigned threads, PoolProtocol proto, uint64_t width, uint64_t height,
			  uint64_t depth, uint64_t src_pad, uint64_t dst_pad, uint64_t extra_rows)
{
	const uint64_t sp = width + src_pad;
	const uint64_t dp = width + dst_pad;
	const uint64_t sh = height + extra_rows; // slice height may exceed the copied rows
	const uint64_t dh = height + extra_rows;
	check::Buffer src(sp * sh * depth);
	check::Buffer dst(dp * dh * depth, kPad);
	for (std::size_t i = 0; i < std::size(src); i++)
	{
		src[i] = check::pattern(i, 3);
	}
	const Copy2DView c{std::data(dst), dp, dh, std::data(src), sp, sh, width, height, depth};
	const bool ran = pool != nullptr && pool->Copy2D(kDefault, c);
	if (!ran)
	{
		CHECKF(pool == nullptr || width * height * depth < cmt::kMinParallelCopyBytes, "%s t=%u 2d refused",
			   name(proto), threads);
		cmt::Run2DSerial(kMemcpy, c);
	}
	bool ok = true;
	for (uint64_t z = 0; z < depth && ok; z++)
	{
		for (uint64_t y = 0; y < dh && ok; y++)
		{
			const uint8_t* drow = std::data(dst) + z * dp * dh + y * dp;
			const uint8_t* srow = std::data(src) + z * sp * sh + y * sp;
			for (uint64_t x = 0; x < dp; x++)
			{
				const bool copied = y < height && x < width;
				if (drow[x] != (copied ? srow[x] : kPad))
				{
					ok = false;
					break;
				}
			}
		}
	}
	CHECKF(ok, "%s t=%u 2d w=%llu h=%llu d=%llu", name(proto), threads, (unsigned long long)width,
		   (unsigned long long)height, (unsigned long long)depth);
}

void test_plans()
{
	cmt::CopyShare shares[cmt::kMaxCopyParticipants];
	uint8_t a = 0;
	uint8_t b = 0;
	CHECK(!cmt::Plan1D(shares, 4, kMemcpy, &a, &b, cmt::kMinParallelCopyBytes - 1));
	CHECK(cmt::Plan1D(shares, 3, kMemcpy, &a, &b, cmt::kMinParallelCopyBytes + 100));
	uint64_t total = 0;
	for (unsigned i = 0; i < 3; i++)
	{
		CHECK(shares[i].byte_offset == total);
		CHECK(i == 2 || shares[i].byte_count % cmt::kSplitAlignmentBytes == 0);
		total += shares[i].byte_count;
	}
	CHECK(total == cmt::kMinParallelCopyBytes + 100);
	// Row split: 10 rows over 4 shares = 3,3,3,1.
	const Copy2DView c{&a, 1u << 20, 10, &b, 1u << 20, 10, 1u << 20, 10, 1};
	CHECK(cmt::Plan2D(shares, 4, kMemcpy, c));
	CHECK(shares[0].row_count == 3 && shares[3].row_count == 1 && shares[3].first_row == 9);
	// 2 rows over 4 shares: the last two shares are empty.
	const Copy2DView c2{&a, 1u << 20, 2, &b, 1u << 20, 2, 1u << 20, 2, 1};
	CHECK(cmt::Plan2D(shares, 4, kMemcpy, c2));
	CHECK(shares[2].row_count == 0 && shares[2].byte_count == 0 && shares[3].row_count == 0);
}

void test_pool(PoolProtocol proto, unsigned threads)
{
	PoolOptions opt;
	opt.workers = threads - 1;
	opt.protocol = proto;
	if (threads == 5)
	{
		cpu_set_t allowed;
		const int status = sched_getaffinity(0, sizeof(allowed), &allowed);
		CHECK(status == 0);
		if (status != 0)
		{
			return;
		}
		for (unsigned cpu = 0; cpu < CPU_SETSIZE && opt.cpus.size() < 2; ++cpu)
		{
			if (CPU_ISSET(cpu, &allowed))
			{
				opt.cpus.push_back(cpu);
			}
		}
	}
	auto created = ThreadPool::Create(opt);
	CHECK(created);
	if (!created)
	{
		return;
	}
	std::unique_ptr<ThreadPool> pool = std::move(*created);
	CHECK((pool == nullptr) == (threads == 1));

	for (std::size_t bytes :
		 {std::size_t{4096}, std::size_t{cmt::kMinParallelCopyBytes - 1},
		  std::size_t{cmt::kMinParallelCopyBytes}, std::size_t{cmt::kMinParallelCopyBytes + 4097},
		  std::size_t{7} * cmt::kMinParallelCopyBytes + 3})
	{
		check_1d(pool.get(), threads, proto, bytes, threads);
	}
	check_2d(pool.get(), threads, proto, 4096 + 13, 300, 1, 64, 7, 0);	 // row split
	check_2d(pool.get(), threads, proto, 1000, 700, 3, 24, 0, 5);		 // 3D, slice height > rows
	check_2d(pool.get(), threads, proto, (3u << 20) + 5, 1, 1, 0, 0, 0); // one row: byte split
	check_2d(pool.get(), threads, proto, 100, 5, 2, 0, 0, 0);			 // too small: serial
}

void test_burst(PoolProtocol proto)
{
	PoolOptions opt;
	opt.workers = 7;
	opt.protocol = proto;
	auto created = ThreadPool::Create(opt);
	CHECK(created);
	if (!created)
	{
		return;
	}
	std::unique_ptr<ThreadPool> pool = std::move(*created);
	const std::size_t bytes = 2u << 20;
	check::Buffer src(bytes);
	check::Buffer dst(bytes);
	bool ok = true;
	for (unsigned round = 0; round < 300 && ok; round++)
	{
		src[round * 4099 % bytes] = static_cast<uint8_t>(round);
		// Switch loops without rebuilding the pool and verify each new payload.
		const auto loop = round % 2 == 0 ? kMemcpy : kDefault;
		ok = pool->Copy1D(loop, std::data(dst), std::data(src), bytes) &&
			 std::memcmp(std::data(dst), std::data(src), bytes) == 0;
	}
	CHECKF(ok, "%s burst", name(proto));
}

void TestAffinityFailures(PoolProtocol protocol)
{
	PoolOptions options;
	options.workers = 2;
	options.protocol = protocol;
	// A later worker's invalid mask must stop and join workers already started.
	auto allowed_cpus = cmt::GetThreadAffinity();
	CHECK(allowed_cpus && !allowed_cpus->empty());
	if (!allowed_cpus || allowed_cpus->empty())
	{
		return;
	}
	options.cpus = {allowed_cpus->front(), std::numeric_limits<unsigned>::max()};
	auto rejected = ThreadPool::Create(options);
	CHECK(!rejected);
	if (!rejected)
	{
		CHECK(rejected.error().IsUsage());
	}

	options.cpus.clear();
	auto created = ThreadPool::Create(options);
	CHECK(created);
	if (!created)
	{
		return;
	}
	auto& pool = *created;
	auto bad_pin = pool->PinWorkers({std::numeric_limits<unsigned>::max()});
	CHECK(!bad_pin);
	if (!bad_pin)
	{
		CHECK(bad_pin.error().IsUsage());
	}
	CHECK(pool->PinWorkers({}));
	check_1d(pool.get(), 3, protocol, 1u << 20, 42);

	cpu_set_t allowed;
	const auto affinity_status = sched_getaffinity(0, sizeof(allowed), &allowed);
	CHECK(affinity_status == 0);
	if (affinity_status != 0)
	{
		return;
	}
	for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu)
	{
		if (CPU_ISSET(cpu, &allowed))
		{
			CHECK(pool->PinWorkers({cpu}));
			check_1d(pool.get(), 3, protocol, 1u << 20, 43);
			break;
		}
	}
}

void TestConcurrentCallers(PoolProtocol protocol)
{
	PoolOptions options;
	options.workers = 3;
	options.protocol = protocol;
	auto created = ThreadPool::Create(options);
	CHECK(created);
	if (!created)
	{
		return;
	}
	auto& pool = *created;
	Mutex gate_mutex;
	ConditionVariable gate;
	bool begin = false;
	std::atomic<unsigned> wrong{0};
	std::array<cmt::Thread, 4> callers;
	for (unsigned caller = 0; caller < callers.size(); ++caller)
	{
		auto started = callers[caller].Start(
			[&, caller]
			{
				constexpr std::size_t bytes = cmt::kMinParallelCopyBytes + 4097;
				check::Buffer source(bytes), destination(bytes);
				{
					std::unique_lock<Mutex> guard(gate_mutex);
					while (!begin)
					{
						gate.Wait(guard.mutex());
					}
				}
				for (unsigned round = 0; round < 24; ++round)
				{
					const auto value = static_cast<unsigned char>(caller * 32 + round + 1);
					std::memset(std::data(source), value, bytes);
					std::memset(std::data(destination), 0, bytes);
					const auto loop = round % 2 ? kMemcpy : kDefault;
					if (!pool->Copy1D(loop, std::data(destination), std::data(source), bytes) ||
						std::memcmp(std::data(destination), std::data(source), bytes) != 0)
					{
						wrong.fetch_add(1, std::memory_order_relaxed);
					}
				}
			});
		CHECK(started);
		if (!started)
		{
			break;
		}
	}
	{
		std::lock_guard<Mutex> guard(gate_mutex);
		begin = true;
		gate.SignalAll();
	}
	for (auto& caller : callers)
	{
		CHECK(caller.Join());
	}
	CHECK(wrong.load(std::memory_order_relaxed) == 0);
}

void TestImmediateShutdown(PoolProtocol protocol)
{
	PoolOptions options;
	options.workers = 3;
	options.protocol = protocol;
	for (unsigned round = 0; round < 16; ++round)
	{
		auto created = ThreadPool::Create(options);
		CHECK(created);
		// Destroy before a first job. Workers may still be starting or may
		// already be asleep; both must observe shutdown and join.
	}
}

void TestMappingSizeOverflow()
{
	cmt::CopySettings knobs;
	knobs.workers = 2;
	knobs.slots = 2;
	knobs.chunk_bytes = UINT64_MAX;
	CHECK(cmt::ring::SlotStride(knobs.chunk_bytes) == 0);
	CHECK(cmt::ring::RequiredPinnedBytes(knobs) == 0);
	CHECK(!cmt::ring::SettingsFit({2, 2, UINT64_MAX}, knobs));
	knobs.chunk_bytes = UINT64_MAX / 2;
	CHECK(cmt::ring::RequiredPinnedBytes(knobs) == 0);
}

} // namespace

int main(int argc, char** argv)
{
	if (argc == 2 && std::strcmp(argv[1], "shared_cpu") == 0)
	{
		// New workers inherit this mask, reproducing a caller pinned by its
		// host application. No timing threshold: verify completion and bytes.
		cpu_set_t saved, one;
		const int got = sched_getaffinity(0, sizeof(saved), &saved);
		CHECK(got == 0);
		if (got != 0)
		{
			return check::exit_code();
		}
		CPU_ZERO(&one);
		for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu)
		{
			if (CPU_ISSET(cpu, &saved))
			{
				CPU_SET(cpu, &one);
				break;
			}
		}
		const int pinned = sched_setaffinity(0, sizeof(one), &one);
		CHECK(pinned == 0);
		if (pinned != 0)
		{
			return check::exit_code();
		}
		for (PoolProtocol proto : {PoolProtocol::Condvar, PoolProtocol::Mpmc})
		{
			test_burst(proto);
			TestConcurrentCallers(proto);
			TestImmediateShutdown(proto);
		}
		CHECK(sched_setaffinity(0, sizeof(saved), &saved) == 0);
		return check::exit_code();
	}
	TestMappingSizeOverflow();
	test_plans();
	for (PoolProtocol proto : {PoolProtocol::Condvar, PoolProtocol::Mpmc})
	{
		TestAffinityFailures(proto);
		TestConcurrentCallers(proto);
		TestImmediateShutdown(proto);
		for (unsigned threads = 1; threads <= 16; threads++)
		{
			test_pool(proto, threads);
		}
		test_burst(proto);
		std::printf("ok   %s\n", name(proto));
	}
	return check::exit_code();
}
