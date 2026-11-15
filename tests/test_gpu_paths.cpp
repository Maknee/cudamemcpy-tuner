// Every registered path, every variant, on the GPU this runs on: the data arrives right (verified "ok"), a
// row is emitted per size, rates are positive and below the PCIe link's raw rate. Sizes hit the edge cases of
// each loop: one page, a size that is not a multiple of the 1 MiB slot or of the pin step, and a multi-step
// size. The generic test iterates over BenchmarkPaths(), so a new path is covered without editing this
// file. Exit 77 (skipped) without a GPU.
#include <chrono>
#include <cmath>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <cstring>
#include <string>
#include <vector>

#include "tests/check.h"
#include "measure/copy.h"
#include "measure/pin_pipeline.h"
#include "cli.h"
#include "cuda/cuda.h"
#include "machine.h"
#include "measure/metrics.h"

using namespace cmt;
using A = std::vector<std::string>;

namespace {

DeviceInfo device_info;

const DeviceInfo& dev()
{
	return device_info;
}

/// "16.0 GT/s PCIe x16" -> raw data rate in GB/s, one direction (128b/130b from 8 GT/s on, 8b/10b below);
/// 0 when sysfs did not say (a VM, an unknown format).
double LinkGbps(const std::string& link)
{
	double gts = 0;
	int width = 0;
	if (std::sscanf(link.c_str(), "%lf GT/s PCIe x%d", &gts, &width) != 2 || gts <= 0 || width <= 0)
	{
		return 0;
	}
	const double enc = gts >= 8 ? 128.0 / 130 : 0.8;
	return gts * enc * width / 8;
}

const std::string kSizes = "4096,1052720,5242896"; // 4 KiB; 1 MiB + 4 KiB + 48; 5 MiB + 16
const A kBase = {"--sizes", kSizes, "--samples", "2", "--min-bytes", "1M"};

std::vector<MeasurementRow> run_args(const A& extra)
{
	A args = kBase;
	args.insert(std::end(args), std::begin(extra), std::end(extra));
	const auto o = ParseArgs(args);
	CHECK(o);
	if (!o)
	{
		return {};
	}
	std::vector<MeasurementRow> rows;
	auto result = Run(o->cfg, dev(),
					  [&](const MeasurementRow& r) -> Result<void>
					  {
						  rows.push_back(r);
						  return {};
					  });
	CHECK(result);
	if (!result)
	{
		std::printf("  %s\n", result.error().Message());
	}
	return rows;
}

std::string label(const A& a)
{
	std::string s;
	for (const auto& x : a)
	{
		s += x + " ";
	}
	return s;
}

/// Run, then check every row: data verified (or skip where nothing can be checked), rates sane.
void expect_good(const A& extra, bool expect_skip = false)
{
	const auto rows = run_args(extra);
	const double link = LinkGbps(dev().link_max);
	for (const MeasurementRow& r : rows)
	{
		const bool ok = expect_skip ? r.verified == Verified::Skip : r.verified == Verified::Ok;
		const bool moved = r.path != "register" && r.dma;
		// T2: a copy over the link cannot beat the link's raw rate (x2 for both directions, +10% for clock
		// jitter)
		const bool too_fast = moved && link > 0 && r.gbps_best > link * (r.dir == "both" ? 2 : 1) * 1.1;
		if (!ok || !(r.gbps_best > 0) || r.gbps_min > r.gbps_best || r.samples < 2 || too_fast)
		{
			check::failures()++;
			std::printf("  %s: size %zu verified=%s best=%.3f min=%.3f samples=%zu link=%.1f\n",
						label(extra).c_str(), r.size_bytes, VerifiedName(r.verified), r.gbps_best, r.gbps_min,
						r.samples, link);
		}
	}
	if (std::empty(rows))
	{
		check::failures()++;
		std::printf("  %s: no rows\n", label(extra).c_str());
	}
}

} // namespace

TEST(every_builtin_path_with_its_defaults)
{
	for (const PathSpec& p : BenchmarkPaths())
	{
		expect_good({"--path", p.name});
	}
}

TEST(pinned_ce_every_alloc_direction_and_page_kind)
{
	for (const char* d : {"htod", "dtoh"})
	{
		for (const char* a : {"register", "hostalloc", "hostalloc_wc"})
		{
			for (const char* pg : {"4k", "thp"})
			{
				if (std::string(a) != "register" && std::string(pg) == "thp")
				{
					continue;
				}
				A args{"--path", "pinned_ce", "--dir", d, "--alloc", a};
				if (std::string(a) == "register")
				{
					args.insert(std::end(args), {"--pages", pg});
				}
				expect_good(args);
			}
		}
	}
}

TEST(bidir_with_hostalloc)
{
	for (const auto& r : run_args({"--path", "bidir_ce", "--alloc", "hostalloc"}))
	{
		CHECK(r.verified == Verified::Ok);
		CHECK(r.gbps_h2d > 0 && r.gbps_d2h > 0);
		CHECK_EQ(r.dir, std::string("both"));
	}
}

TEST(pageable_both_directions_and_pages)
{
	for (const char* d : {"htod", "dtoh"})
	{
		for (const char* pg : {"4k", "thp"})
		{
			expect_good({"--path", "pageable", "--dir", d, "--pages", pg});
		}
	}
}

TEST(staged_nt_reports_the_shared_kernel)
{
	const auto selected = hwcopy::Select("nt_store@avx2", CopyCtx{}, std::nullopt, hwcopy::GetCpu());
	if (!selected || selected->fell_back)
	{
		return;
	}
	const auto rows = run_args({"--path", "staged", "--fill", "nt_store@avx2"});
	CHECK(!rows.empty());
	for (const auto& row : rows)
	{
		CHECK_EQ(row.fill, std::string("nt_store@avx2"));
		CHECK(row.verified == Verified::Ok);
	}
}

TEST(staged_avx512_records_the_selected_kernel)
{
	for (const char* name : {"cached@avx512", "nt_store@avx512"})
	{
		const auto selected = hwcopy::Select(name, CopyCtx{}, std::nullopt, hwcopy::GetCpu());
		if (!selected || selected->fell_back)
		{
			continue;
		}
		const auto rows = run_args({"--path", "staged", "--fill", name});
		CHECK(!rows.empty());
		for (const auto& row : rows)
		{
			CHECK_EQ(row.fill, std::string(name));
			CHECK(row.verified == Verified::Ok);
		}
	}
}

TEST(staged_every_fill_threads_and_wc)
{
	for (const char* f : {"memcpy", "cached@avx2", "cached@avx512", "movsb", "nt_store@avx2",
						  "nt_store@avx512", "nt_store@sse2"})
	{
		for (const char* t : {"1", "2"})
		{
			for (bool wc : {false, true})
			{
				const auto selected = hwcopy::Select(f, CopyCtx{}, std::nullopt, hwcopy::GetCpu());
				if (!selected || selected->fell_back)
				{
					continue;
				}
				A a = {"--path", "staged", "--fill", f, "--threads", t, "--slots", "2"};
				if (wc)
				{
					a.push_back("--wc");
				}
				expect_good(a);
			}
		}
	}
	expect_good({"--path", "staged", "--fill", "none"}, true); // DMA alone: nothing to verify
	StagedBenchmarkOptions cpu_only;
	cpu_only.dma = false;
	StagedCopyBenchmark copier;
	CHECK_ERROR_KIND(copier.Setup(cpu_only), Unsupported);
}

TEST(staged_dtoh_drains_cached_and_wc_slots_across_repetitions)
{
	for (const char* name : {"memcpy", "movsb", "cached@avx2", "nt_store@avx2", "nt_store@sse2",
							 "cached@avx512", "nt_store@avx512"})
	{
		const auto selected = hwcopy::Select(name, CopyCtx{}, std::nullopt, hwcopy::GetCpu());
		if (!selected || selected->fell_back)
		{
			std::printf("  (skip DtoH %s: not supported on this CPU)\n", name);
			continue;
		}
		for (bool wc : {false, true})
		{
			for (const char* threads : {"1", "3"})
			{
				// Fewer chunks than workers, odd tails, and chunk counts not divisible
				// by workers. Multiple repetitions must not create overlapping writers.
				A args = {"--path",	   "staged",		  "--dir",		 "dtoh", "--fill",		 name,
						  "--threads", threads,			  "--slots",	 "2",	 "--slot-bytes", "64K",
						  "--sizes",   "4K,65552,196624", "--min-bytes", "1M"};
				if (wc)
				{
					args.push_back("--wc");
				}
				const auto rows = run_args(args);
				CHECK_EQ(rows.size(), std::size_t{3});
				for (const auto& row : rows)
				{
					CHECK_EQ(row.dir, std::string("dtoh"));
					CHECK_EQ(row.fill, std::string(name));
					CHECK_EQ(row.alloc, std::string(wc ? "slots_wc" : "slots"));
					CHECK(row.verified == Verified::Ok);
					CHECK(row.reps > 1);
					CHECK(row.samples >= 2);
					CHECK(std::isfinite(row.gbps_med) && row.gbps_med > 0);
				}
			}
		}
	}
	expect_good({"--path", "staged", "--dir", "dtoh", "--fill", "none"}, true);
}

TEST(staged_explicit_cpu_placement_uses_the_pre_pin_allowed_mask)
{
	auto allowed = GetThreadAffinity();
	CHECK(allowed);
	if (!allowed)
	{
		return;
	}
	const auto adjacent = std::adjacent_find(allowed->begin(), allowed->end(), [](unsigned a, unsigned b)
											 { return b == a + 1 && b <= static_cast<unsigned>(INT_MAX); });
	if (adjacent == allowed->end())
	{
		std::printf("  explicit staged CPU placement: no adjacent allowed CPUs; skipped\n");
		return;
	}
	const auto first = *adjacent;
	// Isolate intentional caller pinning in a temporary thread. The test process
	// keeps its original affinity; no caller is temporarily widened for startup.
	Result<void> outcome;
	auto exercise = [&]() -> Result<void>
	{
		CMT_CUDA(cudaSetDevice(dev().ordinal));
		CMT_RETURN_IF_ERROR(PinThread(static_cast<int>(first)));
		constexpr std::size_t bytes = 128u << 10;
		auto host = HostBuffer::Create(bytes, Pages::Small);
		if (!host)
		{
			return Failure(host.error());
		}
		auto device = DeviceBuffer::Create(bytes);
		if (!device)
		{
			return Failure(device.error());
		}
		PatternFill(host->data(), bytes, 71);
		StagedBenchmarkOptions shape{
			dev().ordinal, static_cast<int>(first), 2, 2, 64u << 10, LoopName{}, true, false, true, *allowed};
		StagedCopyBenchmark copier;
		CMT_RETURN_IF_ERROR(copier.Setup(shape));
		CMT_RETURN_IF_ERROR(copier.Copy(Dir::HtoD, host->data(), device->data(), bytes, 1));
		auto mismatches = DevicePatternCheck(device->data(), bytes, 71, nullptr);
		if (!mismatches)
		{
			return Failure(mismatches.error());
		}
		if (*mismatches != 0)
		{
			return Failure(Error::Kind::Verify, "Explicit CPU placement corrupted staged copy");
		}
		shape.worker_allowed_cpus = {first};
		auto rejected = copier.Setup(shape);
		if (rejected || !rejected.error().IsUsage())
		{
			return Failure(Error::Kind::Verify, "Staged worker escaped its saved CPU affinity");
		}
		auto caller = GetThreadAffinity();
		if (!caller)
		{
			return Failure(caller.error());
		}
		if (*caller != std::vector<unsigned>{first})
		{
			return Failure(Error::Kind::Verify, "Staged startup changed its caller CPU affinity");
		}
		return {};
	};
	Thread caller;
	auto started = caller.Start([&] { outcome = exercise(); }, ThreadPlacement{});
	CHECK(started);
	if (!started)
	{
		return;
	}
	CHECK(caller.Join());
	CHECK(outcome);
	if (!outcome)
	{
		std::printf("  %s\n", outcome.error().Message());
	}
}

// The real NVML for this GPU: found by PCI address, a nonempty set of configured CPUs, and the same answer as
// the kernel's local_cpulist (a disagreement would show in the machine record too). The first lookup pays
// dlopen and nvmlInit_v2; the time is printed because the hook pays it at a process's first staged copy.
TEST(nvml_finds_this_gpu_by_pci_address_and_agrees_with_sysfs)
{
	const auto start = std::chrono::steady_clock::now();
	const auto nvml = NvmlGpuCpus(dev().pci_bus_id);
	const auto first =
		std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	const auto again_start = std::chrono::steady_clock::now();
	const auto again = NvmlGpuCpus(dev().pci_bus_id);
	const auto second =
		std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - again_start).count();
	const auto sysfs = GpuLocalCpus(dev().pci_bus_id, {});
	std::printf("  NVML %s: %s (first lookup %.2f ms, then %.3f ms); %s: %s %s\n", dev().pci_bus_id.c_str(),
				nvml ? FormatCpuList(*nvml).c_str() : nvml.error().Message(), first, second,
				CpuSourceName(sysfs.source), FormatCpuList(sysfs.cpus).c_str(), sysfs.sysfs_error.c_str());
	if (!nvml && nvml.error().IsUnsupported())
	{
		std::printf("  no usable NVML here: the sysfs fallback is the path\n");
		return;
	}
	CHECK(nvml && again && *nvml == *again && !std::empty(*nvml));
	CHECK(std::empty(sysfs.cpus) || sysfs.cpus == *nvml);
}

// Only the record that places the tuner's children asks NVML (34 ms at its first query); a benchmark child's
// record says it did not, and both carry the sysfs answer.
TEST(only_the_placing_machine_record_asks_nvml)
{
	const auto child = check::Require(MachineReportFor(dev().ordinal, false));
	CHECK(child.text.find(R"("nvml_cpus":null,"nvml_error":"not queried")") != std::string::npos);
	const auto metadata = check::Require(MachineReportFor(dev().ordinal, false, std::nullopt, true));
	CHECK(metadata.text.find(R"("nvml_error":"not queried")") == std::string::npos);
	CHECK(child.text.find(R"("sysfs_source":)") != std::string::npos);
	CHECK(metadata.text.find(R"("used")") ==
		  std::string::npos); // nvml_cpus != null already says which applies
}

TEST(staged_gpu_local_workers_observe_affinity_and_copy_correctly)
{
	struct RestoreEnvironment
	{
		const char* names[3] = {"CMT_WORKER_NUMA", "CMT_WORKER_CPUS", "CMT_WORKER_NUMA_NODE"};
		std::optional<std::string> saved[3];

		RestoreEnvironment()
		{
			for (unsigned i = 0; i < 3; ++i)
			{
				if (const char* value = std::getenv(names[i]))
				{
					saved[i] = value;
				}
				CHECK(unsetenv(names[i]) == 0);
			}
			CHECK(setenv("CMT_WORKER_NUMA", "1", 1) == 0);
		}

		~RestoreEnvironment()
		{
			for (unsigned i = 0; i < 3; ++i)
			{
				CHECK((saved[i] ? setenv(names[i], saved[i]->c_str(), 1) : unsetenv(names[i])) == 0);
			}
		}
	} restore_environment;

	auto allowed = GetThreadAffinity();
	auto node = PciNumaNode(dev().pci_bus_id);
	CHECK(allowed && node);
	if (!allowed || !node)
	{
		return;
	}
	// A GPU node this thread may not run on (or no topology) leaves the worker on the caller's CPUs: the
	// placement never fails startup (cmt::ResolveWorkerPlacement).
	auto local = *node ? NumaNodeCpus(**node) : Result<std::vector<unsigned>>(*allowed);
	const std::vector<unsigned> expected = local ? *local : *allowed;
	std::vector<unsigned> observed;
	auto worker = copying::runtime::Worker::Create(
		[&]() -> Result<void>
		{
			auto affinity = GetThreadAffinity();
			if (!affinity)
			{
				return Failure(affinity.error());
			}
			observed = std::move(*affinity);
			return {};
		},
		dev().ordinal);
	CHECK(worker);
	if (!worker)
	{
		return;
	}
	CHECK((*worker)->Join());
	CHECK(observed == expected);
	if (*node && local)
	{
		std::printf("  GPU-local worker: PCI %s, NUMA node %d, %zu allowed CPUs\n", dev().pci_bus_id.c_str(),
					**node, observed.size());
	}
	else
	{
		std::printf("  GPU topology unknown: worker inherited %zu allowed CPUs\n", observed.size());
	}

	auto selected = CudaCheck(cudaSetDevice(dev().ordinal), "Select staged test device");
	CHECK(selected);
	if (!selected)
	{
		return;
	}
	constexpr std::size_t bytes = 128u << 10;
	auto host = HostBuffer::Create(bytes, Pages::Small);
	auto device = DeviceBuffer::Create(bytes);
	CHECK(host && device);
	if (!host || !device)
	{
		return;
	}
	PatternFill(host->data(), bytes, 83);
	StagedCopyBenchmark copier;
	auto setup = copier.Setup(
		StagedBenchmarkOptions{dev().ordinal, -1, 2, 2, 64u << 10, LoopName{}, true, false, true});
	CHECK(setup);
	if (!setup)
	{
		return;
	}
	auto uploaded = copier.Copy(Dir::HtoD, host->data(), device->data(), bytes, 1);
	CHECK(uploaded);
	if (!uploaded)
	{
		return;
	}
	auto mismatches = DevicePatternCheck(device->data(), bytes, 83, nullptr);
	CHECK(mismatches && *mismatches == 0);
	std::memset(host->data(), 0, bytes);
	auto downloaded = copier.Copy(Dir::DtoH, host->data(), device->data(), bytes, 1);
	CHECK(downloaded);
	if (downloaded)
	{
		CHECK_EQ(PatternCheck(host->data(), bytes, 83), std::size_t{0});
	}
	auto caller = GetThreadAffinity();
	CHECK(caller && *caller == *allowed);
}

TEST(staged_refreshes_slots_between_calls_without_a_warmup)
{
	constexpr std::size_t bytes = (3u << 16) + 16, guard = 16;
	for (const char* name : {"memcpy", "movsb", "cached@avx2", "nt_store@avx2", "nt_store@sse2",
							 "cached@avx512", "nt_store@avx512"})
	{
		const auto selected = hwcopy::Select(name, CopyCtx{}, std::nullopt, hwcopy::GetCpu());
		if (!selected || selected->fell_back)
		{
			continue;
		}
		for (bool wc : {false, true})
		{
			for (std::size_t threads : {1u, 3u})
			{
				auto host_result = HostBuffer::Create(bytes + 2 * guard, Pages::Small);
				CHECK(host_result);
				if (!host_result)
				{
					return;
				}
				auto host = std::move(*host_result);
				auto* const data = std::data(host) + guard;
				auto device_result = DeviceBuffer::Create(bytes);
				CHECK(device_result);
				if (!device_result)
				{
					return;
				}
				auto device = std::move(*device_result);
				auto producer_result = Stream::Create();
				CHECK(producer_result);
				if (!producer_result)
				{
					return;
				}
				auto producer = std::move(*producer_result);
				cmt::StagedCopyBenchmark copier;
				CHECK(copier.Setup(cmt::StagedBenchmarkOptions{dev().ordinal, -1, threads, 2, 64u << 10,
															   *ParseLoopName(NativeArch(), name), true, wc,
															   true}));
				CHECK_ERROR_KIND(copier.Copy(static_cast<Dir>(255), data, std::data(device), bytes, 3),
								 Usage);
				// The second call must copy B, not reuse the A bytes retained in
				// slots. No copier warmup may hide a missing CPU leg or DMA.
				for (Dir dir : {Dir::HtoD, Dir::DtoH})
				{
					for (std::uint64_t seed : {0x3141592653ull, 0x2718281828ull})
					{
						std::memset(std::data(host), 0xD3, std::size(host));
						if (dir == Dir::HtoD)
						{
							PatternFill(data, bytes, seed);
							CHECK(CudaCheck(cudaMemsetAsync(std::data(device), 0, bytes, producer.Get()),
											"test CUDA operation"));
						}
						else
						{
							CHECK(LaunchPatternFill(std::data(device), bytes, seed, producer.Get()));
						}
						CHECK(producer.Sync());
						const auto seconds_result = copier.Copy(dir, data, std::data(device), bytes, 3);
						CHECK(seconds_result);
						if (!seconds_result)
						{
							return;
						}
						const double seconds = *seconds_result;
						CHECK(seconds > 0 && std::isfinite(seconds));
						const std::size_t bad =
							dir == Dir::HtoD
								? DevicePatternCheck(std::data(device), bytes, seed, producer.Get())
									  .value_or(bytes / 8)
								: PatternCheck(data, bytes, seed);
						CHECK_EQ(bad, std::size_t{0});
						for (std::size_t i = 0; i < guard; i++)
						{
							CHECK(std::data(host)[i] == 0xD3);
							CHECK(data[bytes + i] == 0xD3);
						}
					}
				}
			}
		}
	}
}

TEST(lifecycle_measures_real_allocations_and_a_completed_transfer)
{
	for (const char* direction : {"htod", "dtoh"})
	{
		for (const char* pages : {"4k", "thp"})
		{
			const auto rows = run_args({"--path", "lifecycle", "--dir", direction, "--pages", pages});
			CHECK_EQ(std::size(rows), std::size_t{3});
			for (const auto& row : rows)
			{
				CHECK(row.verified == Verified::Ok);
				CHECK(row.setup_s > 0 && row.copy_s > 0 && row.teardown_s > 0);
				const double total = row.setup_s + row.copy_s + row.teardown_s;
				const double rate = static_cast<double>(row.size_bytes) / total / 1e9;
				CHECK(std::fabs(rate / row.gbps_med - 1) < 1e-9);
			}
		}
	}
}

TEST(pin_pipeline_rejects_invalid_inputs_before_cuda_work)
{
	using namespace cmt::copying::runtime;
	unsigned char placeholder = 0;
	Stream stream;
	PinPipelineCopy copier;
	PinPipelineOptions shape;
	shape.step_bytes = BasePageBytes();
	CHECK_ERROR_KIND(copier.Setup(shape, nullptr, 1, stream), Usage);
	CHECK_ERROR_KIND(copier.Setup(shape, &placeholder, 0, stream), Usage);
	auto invalid = shape;
	invalid.step_bytes = 0;
	CHECK_ERROR_KIND(copier.Setup(invalid, &placeholder, 1, stream), Usage);
	invalid = shape;
	invalid.release = static_cast<ReleasePolicy>(255);
	CHECK_ERROR_KIND(copier.Setup(invalid, &placeholder, 1, stream), Usage);

	// End policy setup only records borrowed ranges. Each rejected Copy must
	// fail before registering or dereferencing these deliberately tiny buffers.
	auto setup = copier.Setup(shape, &placeholder, 1, stream);
	CHECK(setup);
	if (!setup)
	{
		return;
	}
	CHECK_ERROR_KIND(copier.Copy(static_cast<Dir>(255), &placeholder, 1), Usage);
	CHECK_ERROR_KIND(copier.Copy(Dir::HtoD, nullptr, 1), Usage);
	CHECK_ERROR_KIND(copier.Copy(Dir::HtoD, &placeholder, 0), Usage);
	setup = copier.Setup(shape, &placeholder, std::numeric_limits<std::size_t>::max(), stream);
	CHECK(setup);
	if (!setup)
	{
		return;
	}
	CHECK_ERROR_KIND(copier.Copy(Dir::HtoD, &placeholder, std::numeric_limits<std::uint32_t>::max()), Usage);
}

TEST(pin_pipeline_every_unpin_policy)
{
	for (const char* direction : {"htod", "dtoh"})
	{
		for (const char* u : {"end", "eager", "prereg"})
		{
			for (const char* pg : {"4k", "thp"})
			{
				expect_good({"--path", "pin_pipeline", "--dir", direction, "--unpin", u, "--pages", pg,
							 "--pin-step", "2M"});
			}
		}
	}
}

TEST(register_emits_pin_and_unpin_rows)
{
	const auto rows = run_args({"--path", "register"});
	CHECK_EQ(std::size(rows), std::size_t{6});
	for (std::size_t i = 0; i + 1 < std::size(rows); i += 2)
	{
		CHECK_EQ(rows[i].phase, std::string("pin"));
		CHECK_EQ(rows[i + 1].phase, std::string("unpin"));
		CHECK_EQ(rows[i].dir, std::string("-"));
		CHECK(rows[i].gbps_best > 0 && rows[i + 1].gbps_best > 0);
	}
}

TEST(pin_copy_unpin_phases_add_up_to_the_rate)
{
	for (const char* d : {"htod", "dtoh"})
	{
		for (const char* pg : {"4k", "thp"})
		{
			const A a = {"--path", "pin_copy_unpin", "--dir", d, "--pages", pg};
			expect_good(a);
			for (const MeasurementRow& r : run_args(a))
			{
				CHECK(r.setup_s > 0 && r.copy_s > 0 && r.teardown_s > 0);
				// the best sample's three phases are the best sample's time: size / sum = gbps_best (rounding
				// only)
				const double g =
					static_cast<double>(r.size_bytes) / (r.setup_s + r.copy_s + r.teardown_s) / 1e9;
				if (std::fabs(g / r.gbps_best - 1) > 0.02)
				{
					check::failures()++;
					std::printf("  %s%zu: phases give %.3f GB/s, row says %.3f\n", label(a).c_str(),
								r.size_bytes, g, r.gbps_best);
				}
			}
		}
	}
}

TEST(sizes_larger_than_the_gpu_or_host_are_skipped_not_fatal)
{
	const auto rows =
		run_args({"--path", "pinned_ce", "--sizes", "4096,1024G"}); // 1 TiB: neither memory has it
	CHECK_EQ(std::size(rows), std::size_t{1});
	CHECK_EQ(rows.empty() ? 0 : rows[0].size_bytes, std::size_t{4096});
}

TEST(the_last_device_works_too)
{ // worker threads must follow --device
	auto count = CudaDeviceCount();
	CHECK(count);
	if (!count)
	{
		return;
	}
	const int n = *count;
	if (n < 2)
	{
		std::printf("  (one device: skipped)\n");
		return;
	}
	const std::string last = std::to_string(n - 1);
	const auto o = ParseArgs(
		{"--device", last, "--path", "staged", "--threads", "2", "--sizes", "1M", "--samples", "2"});
	CHECK(o);
	if (!o)
	{
		return;
	}
	const auto d = QueryDevice(n - 1);
	CHECK(d);
	if (!d)
	{
		return;
	}
	std::vector<MeasurementRow> rows;
	CHECK(Run(o->cfg, *d,
			  [&](const MeasurementRow& r) -> Result<void>
			  {
				  rows.push_back(r);
				  return {};
			  }));
	CHECK(!std::empty(rows) && rows[0].verified == Verified::Ok);
	const auto p = ParseArgs({"--device", last, "--path", "pin_pipeline", "--unpin", "eager", "--sizes", "8M",
							  "--pin-step", "2M"});
	CHECK(p);
	if (!p)
	{
		return;
	}
	rows.clear();
	CHECK(Run(p->cfg, *d,
			  [&](const MeasurementRow& r) -> Result<void>
			  {
				  rows.push_back(r);
				  return {};
			  }));
	CHECK(!std::empty(rows) && rows[0].verified == Verified::Ok);
	CHECK(CudaCheck(cudaSetDevice(0), "test CUDA operation"));
}

TEST(device_check_sees_a_single_wrong_word)
{
	// the verifier is what makes every rate trustworthy; prove it can fail. Two 64 MiB host pieces and a
	// 4 B tail that is no pattern word: the second piece must carry the host pattern's word indices.
	const std::size_t b = (std::size_t{65} << 20) + 4;
	auto d_result = DeviceBuffer::Create(b);
	CHECK(d_result);
	if (!d_result)
	{
		return;
	}
	auto d = std::move(*d_result);
	auto s_result = Stream::Create();
	CHECK(s_result);
	if (!s_result)
	{
		return;
	}
	auto s = std::move(*s_result);
	CHECK(LaunchPatternFill(std::data(d), b, 99, s.Get()));
	std::vector<unsigned char> h(b);
	CHECK(CudaCheck(cudaMemcpy(std::data(h), std::data(d), b, cudaMemcpyDeviceToHost), "test CUDA operation"));
	CHECK_EQ(PatternCheck(std::data(h), b - 4, 99), std::size_t{0});
	CHECK_EQ(
		DevicePatternCheck(std::data(d), b, 99, s.Get()).value_or(std::numeric_limits<std::size_t>::max()),
		std::size_t{0});
	CHECK(CudaCheck(cudaMemsetAsync(std::data(d) + (std::size_t{64} << 20) + 4096, 0, 8, s.Get()),
					"test CUDA operation"));
	CHECK_EQ(
		DevicePatternCheck(std::data(d), b, 99, s.Get()).value_or(std::numeric_limits<std::size_t>::max()),
		std::size_t{1});
	CHECK_EQ(
		DevicePatternCheck(std::data(d), b, 98, s.Get()).value_or(std::numeric_limits<std::size_t>::max()),
		b / 8);
}

TEST(device_info_is_filled)
{
	const DeviceInfo& d = dev();
	CHECK(!std::empty(d.name));
	CHECK(d.sms > 0);
	CHECK(d.cc_major >= 3);
	CHECK(d.memory_bytes > 0);
	CHECK(std::size(d.pci_bus_id) >= 7);
	CHECK(!std::empty(d.host_cpu));
	std::printf("  (%s, cc %d.%d, %d SMs, link %s = %.1f GB/s raw)\n", d.name.c_str(), d.cc_major, d.cc_minor,
				d.sms, d.link_max.c_str(), LinkGbps(d.link_max));
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
