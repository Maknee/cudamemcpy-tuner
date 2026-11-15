#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <tuple>

#include "config.h"
#include "tests/check.h"
#include "tests/numa_fixture.h"
#include "tuner.h"

using namespace cmt;

TEST(find_uses_half_open_ranges)
{
	Table table;
	table.rows.push_back({Dir::HtoD, 65536, 1048576, Mode::Chunks, {2, 4, 1048576, {}, false}, {}});
	CHECK(table.Find(Dir::HtoD, 65535) == nullptr);
	CHECK(table.Find(Dir::HtoD, 65536) != nullptr);
	CHECK(table.Find(Dir::HtoD, 1048576) == nullptr);
	CHECK(table.Find(Dir::DtoH, 65536) == nullptr);
}

TEST(memory_budget_sums_cached_and_wc_pool_maxima)
{
	Table table;
	CopySettings cached{2, 4, 1048576, {}, false};
	CopySettings wc{1, 2, 1048576, {}, false};
	wc.wc_slots = true;
	table.rows.push_back({Dir::HtoD, 1, 2, Mode::Chunks, cached, {}});
	table.rows.push_back({Dir::HtoD, 2, 3, Mode::Chunks, wc, {}});
	CHECK_EQ(table.MaxPinnedBytes(false), std::size_t{8u << 20});
	CHECK_EQ(table.MaxPinnedBytes(true), std::size_t{2u << 20});
	CHECK_EQ(table.MaxPinnedBytes(), std::size_t{10u << 20});
}

TEST(legacy_text_tables_are_rejected)
{
	CHECK(!ParseTuningResult("cmt-tune-table 1\nrow htod 0 inf driver\n"));
}

using namespace cmt;

namespace {
CopyRun Measured(Mode mode, unsigned round, double gbps, std::vector<double> samples = {})
{
	CopyRun run;
	run.bytes = 1048576;
	run.process_round = round;
	run.mode = mode;
	run.settings = {1, 2, 1048576, "memcpy", false};
	run.seconds = static_cast<double>(run.bytes) / (gbps * 1e9);
	run.gbps = gbps;
	run.sample_gbps = samples.empty() ? std::vector<double>{gbps} : std::move(samples);
	run.status = "complete";
	run.verified = true;
	run.execution = mode == Mode::Driver ? "bare_driver" : "hooked_copy";
	return run;
}
} // namespace

// One process with two samples is enough evidence; one process with one sample is not.
TEST(samples_not_processes_decide_selection)
{
	SelectionPolicy policy;
	std::vector<CopyRun> runs{Measured(Mode::Driver, 1, 10, {10, 10.5}), Measured(Mode::Chunks, 1, 15, {15, 16})};
	CHECK(check::Require(BuildCopyLookup(runs, policy, {})).Find(Dir::HtoD, 1048576)->mode == Mode::Chunks);
	runs[1].sample_gbps = {15};
	CHECK(check::Require(BuildCopyLookup(runs, policy, {})).Find(Dir::HtoD, 1048576)->mode == Mode::Driver);
	// The slowest staged sample must beat the fastest driver sample, whichever sample it is.
	runs[1].sample_gbps = {10.4, 30};
	CHECK(check::Require(BuildCopyLookup(runs, policy, {})).Find(Dir::HtoD, 1048576)->mode == Mode::Driver);
	// Every requested round is required, even when the samples present would suffice.
	runs[1].sample_gbps = {15, 16};
	policy.process_rounds = 2;
	CHECK(check::Require(BuildCopyLookup(runs, policy, {})).Find(Dir::HtoD, 1048576)->mode == Mode::Driver);
}

TEST(completed_process_rounds_select_a_measured_winner)
{
	SelectionPolicy policy;
	policy.process_rounds = 2;
	std::vector<CopyRun> runs{Measured(Mode::Driver, 1, 10), Measured(Mode::Driver, 2, 10),
							  Measured(Mode::Chunks, 1, 15), Measured(Mode::Chunks, 2, 15)};
	auto table = check::Require(BuildCopyLookup(runs, policy, {}));
	const Row* selected = table.Find(Dir::HtoD, 1048576);
	CHECK(selected != nullptr);
	if (selected != nullptr)
	{
		CHECK(selected->mode == Mode::Chunks);
		CHECK(selected->settings.loop.id == VariantId::Memcpy);
		CHECK(selected->settings.loop.ctx.flush_batch == 4096);
		CHECK(!selected->loop_fell_back);
	}
	runs.pop_back();
	table = check::Require(BuildCopyLookup(runs, policy, {}));
	CHECK(table.Find(Dir::HtoD, 1048576)->mode == Mode::Driver);
	// A failed requested point must use the instant base, not an older measured winner.
	auto base = Measured(Mode::Chunks, 1, 20);
	base.coverage = "instant";
	base.execution = "direct_engine";
	auto control = Measured(Mode::Driver, 1, 10);
	control.coverage = "instant";
	auto missing = Measured(Mode::Driver, 1, 10);
	missing.bytes = 2u << 20;
	missing.status = "failed";
	runs.push_back(base);
	runs.push_back(control);
	runs.push_back(missing);
	for (unsigned round : {1u, 2u})
	{
		auto later = Measured(Mode::Driver, round, 10);
		later.bytes = 4u << 20;
		later.seconds *= 4;
		runs.push_back(later);
	}
	table = check::Require(BuildCopyLookup(runs, policy, {}));
	CHECK(table.Find(Dir::HtoD, 1048576)->mode == Mode::Chunks);
	CHECK(table.Find(Dir::HtoD, 2u << 20)->mode == Mode::Chunks);
}

// Without INSTANT runs (QUICK, FULL): below the smallest measured size the driver, between sizes the
// winner of the size below, above the largest size its winner, and the driver after a failed size.
TEST(measured_rows_extend_up_from_each_size)
{
	std::vector<CopyRun> runs;
	for (std::uint64_t bytes : {std::uint64_t{1} << 20, std::uint64_t{64} << 20})
	{
		auto driver = Measured(Mode::Driver, 1, 10, {10, 10.5});
		auto staged = Measured(Mode::Chunks, 1, 15, {15, 16});
		for (CopyRun* run : {&driver, &staged})
		{
			run->bytes = bytes;
			run->seconds = static_cast<double>(bytes) / (run->gbps * 1e9);
		}
		staged.settings.workers = bytes == (std::uint64_t{1} << 20) ? 1 : 2;
		runs.push_back(driver);
		runs.push_back(staged);
	}
	auto table = check::Require(BuildCopyLookup(runs, {}, {}));
	CHECK(table.Find(Dir::HtoD, 512u << 10)->mode == Mode::Driver);
	CHECK_EQ(table.Find(Dir::HtoD, 32u << 20)->settings.workers, 1u);
	CHECK_EQ(table.Find(Dir::HtoD, 64u << 20)->settings.workers, 2u);
	CHECK_EQ(table.Find(Dir::HtoD, std::uint64_t{16} << 30)->settings.workers, 2u);
	CHECK(table.Find(Dir::HtoD, std::uint64_t{16} << 30)->mode == Mode::Chunks);
	runs.back().status = "failed";
	table = check::Require(BuildCopyLookup(runs, {}, {}));
	CHECK_EQ(table.Find(Dir::HtoD, 32u << 20)->settings.workers, 1u);
	CHECK(table.Find(Dir::HtoD, 64u << 20)->mode == Mode::Driver);
	CHECK(table.Find(Dir::HtoD, std::uint64_t{16} << 30)->mode == Mode::Driver);
}

TEST(invalid_and_duplicate_measurements_do_not_select)
{
	auto run = Measured(Mode::Driver, 1, 10);
	CHECK(!BuildCopyLookup({run, run}, {}, {}));
	run.verified = false;
	CHECK(!ValidateCopyRun(run));
	run.verified = true;
	run.gbps = 20;
	CHECK(!ValidateCopyRun(run));
}

using namespace cmt;

// One process per setting, direction and round, covering every size; the driver is one more.
TEST(each_process_times_every_size)
{
	TuneArgs args;
	args.search = tune::Search::Full;
	args.dirs = {Dir::HtoD};
	args.grid = check::Require(tune::ParseTuningParams(
		args.search, "sizes=4K,1M,1G threads=1,2 buffers=2 buffer_size=1M loops=memcpy memory=cached,wc rounds=2"));
	auto runs = check::Require(MakeRuns(args));
	CHECK_EQ(std::size(runs), std::size_t{(1 + 2 * 2) * 2});
	bool driver = false, wc = false;
	for (const auto& run : runs)
	{
		CHECK(run.sizes == (std::vector<std::uint64_t>{4096, 1u << 20, 1u << 30}));
		CHECK_EQ(run.samples, args.grid.samples); // FULL times the driver as often as each setting
		CHECK(run.copy.process_round >= 1 && run.copy.process_round <= 2);
		driver |= run.copy.mode == Mode::Driver;
		wc |= run.copy.settings.wc_slots;
	}
	CHECK(driver && wc);
	args.max_processes = runs.size() - 1;
	CHECK(!MakeRuns(args));
	args.max_processes = runs.size();
	args.max_pinned_mib = 2;
	CHECK_EQ(std::size(check::Require(MakeRuns(args))), std::size_t{(1 + 2) * 2});
}

TEST(tuning_params_reject_what_a_search_cannot_change)
{
	using tune::Search;
	const auto full = check::Require(tune::ParseTuningParams(Search::Full, ""));
	CHECK_EQ(std::size(full.threads) * std::size(full.buffers) * std::size(full.buffer_bytes), std::size_t{75});
	CHECK(full.wc == std::vector<bool>{false});
	CHECK_EQ(full.samples, 2u);
	CHECK_EQ(full.rounds, 1u);
	CHECK(check::Require(tune::ParseTuningParams(Search::Quick, "")).sizes ==
		  (std::vector<std::uint64_t>{64u << 10, 512u << 10, 1u << 20, 4u << 20, 16u << 20, 64u << 20, 512u << 20,
									  1u << 30}));
	CHECK(check::Require(tune::ParseTuningParams(Search::Instant, "")).sizes ==
		  (std::vector<std::uint64_t>{1u << 20, 4u << 20, 16u << 20, 64u << 20, 512u << 20, 1u << 30}));
	CHECK_EQ(check::Require(tune::ParseTuningParams(Search::Quick, " samples=3  rounds=2 ")).samples, 3u);
	CHECK_EQ(check::Require(tune::ParseTuningParams(Search::Quick, "tie=2%")).tie_percent, 2u);
	CHECK_EQ(full.tie_percent, 0u);
	for (const char* bad :
		 {"threads=0", "threads=17", "buffers=33", "buffer_size=3K", "sizes=0", "sizes=1M,1M", "loops=nope",
		  "memory=pinned", "samples=1", "rounds=x", "thread=2", "threads", "threads=2 threads=4",
		  "sizes=", "tie=2", "tie=0.02", "tie=1.5%", "tie=101%", "tie=%"})
	{
		CHECK(!tune::ParseTuningParams(Search::Full, bad));
	}
	CHECK(!tune::ParseTuningParams(Search::Quick, "threads=2"));
	CHECK(!tune::ParseTuningParams(Search::Instant, "sizes=1M"));
	CHECK(!tune::ParseTuningParams(Search::Instant, "tie=2%"));
}

// The estimate: the fewest threads whose summed one-core rate reaches 1.3 x the pinned rate. The first three
// are measured probes and pinned references of the hosts named on each line; the next two sit on either
// side of 1.3 (10 >= 1.3 x 7.6 = 9.88; 10 < 1.3 x 7.8 = 10.14), so 1.0 or 1.5 fails one of them.
TEST(estimate_takes_the_fewest_threads_that_reach_the_pinned_rate)
{
	CHECK_EQ(tune::EstimateThreads(23.16, 26.8, 16), 2); // 3090
	CHECK_EQ(tune::EstimateThreads(25.10, 50.7, 16), 4); // 5090 cpu-to-gpu
	CHECK_EQ(tune::EstimateThreads(5.09, 56.1, 16), 16); // Xeon 5090
	CHECK_EQ(tune::EstimateThreads(10, 7.6, 16), 1);
	CHECK_EQ(tune::EstimateThreads(10, 7.8, 16), 2);
	CHECK_EQ(tune::EstimateThreads(1, 1000, 16), 16); // never more than FULL's largest
}

// On a machine with fewer CPUs than the estimate wants, INSTANT and QUICK try no more threads than it has: the
// Xeon's 16 becomes 4 on 4 or 6 allowed CPUs, and a setting with more workers is not measured.
TEST(threads_stay_within_the_allowed_cpus)
{
	CHECK_EQ(tune::ThreadCap(1), 1);
	CHECK_EQ(tune::ThreadCap(4), 4);
	CHECK_EQ(tune::ThreadCap(6), 4);
	CHECK_EQ(tune::ThreadCap(64), 16);
	CHECK_EQ(tune::EstimateThreads(5.09, 56.1, tune::ThreadCap(6)), 4);
	CHECK_EQ(tune::EstimateThreads(23.16, 26.8, tune::ThreadCap(1)), 1);
	CHECK_EQ(tune::EstimateThreads(1, 1000, tune::ThreadCap(4)), 4); // even when no count reaches the rate
	TuneArgs args;
	const auto cpu = DetectCpuIsas();
	CHECK(!tune::Admit(args, cpu, 4, {8, 2, 1u << 20, "memcpy", false}));
	CHECK(tune::Admit(args, cpu, 4, {4, 2, 1u << 20, "memcpy", false}));
}

namespace {
std::string Key(const StagingSettings& s)
{
	return fmt::format("{}/{}/{}M/{}", s.workers, s.slots, s.slot_bytes >> 20, s.loop);
}

// Runs a walk the way MeasureSearch does: measure `next`, walk again, until it ends. `landscape` gives a
// setting its objective when it is measured, knowing how many were measured before it.
std::vector<std::string> Walk(tune::Search search, Dir direction, int threads,
							  const std::function<double(const std::string&, std::size_t)>& landscape,
							  const tune::AdmitSetting& admit = [](StagingSettings s) { return s; })
{
	std::map<std::string, double> measured;
	const tune::SettingScore score = [&](const StagingSettings& s) -> std::optional<double>
	{
		const auto found = measured.find(Key(s));
		return found == measured.end() ? std::nullopt : std::optional<double>(found->second);
	};
	for (;;)
	{
		const auto walk = tune::WalkSettings(search, direction, threads, false, score, admit);
		if (!walk.next)
		{
			std::vector<std::string> keys;
			for (const auto& s : walk.visited)
			{
				keys.push_back(Key(s));
			}
			CHECK_EQ(std::size(keys), std::size(measured)); // every measured setting is in the walk
			return keys;
		}
		CHECK(!measured.count(Key(*walk.next))); // none is measured twice
		measured[Key(*walk.next)] = landscape(Key(*walk.next), std::size(measured));
	}
}
} // namespace

// INSTANT measures its two settings per direction at the estimated threads, and nothing else. A setting the
// machine cannot run is skipped, and two names for one setting are measured once.
TEST(instant_measures_two_settings_at_the_estimate)
{
	const auto flat = [](const std::string&, std::size_t) { return 0.0; };
	CHECK(Walk(tune::Search::Instant, Dir::HtoD, 4, flat) ==
		  (std::vector<std::string>{"4/2/1M/memcpy", "4/2/1M/nt_store"}));
	CHECK(Walk(tune::Search::Instant, Dir::DtoH, 16, flat) ==
		  (std::vector<std::string>{"16/2/1M/nt_store", "16/4/2M/nt_store"}));
	const tune::AdmitSetting no_memcpy = [](StagingSettings s) -> std::optional<StagingSettings>
	{
		if (s.loop == "memcpy")
		{
			return std::nullopt;
		}
		return s;
	};
	CHECK(Walk(tune::Search::Instant, Dir::HtoD, 2, flat, no_memcpy) ==
		  (std::vector<std::string>{"2/2/1M/nt_store"}));
	const tune::AdmitSetting one_loop = [](StagingSettings s)
	{
		s.loop = "nt_store@avx512";
		return s;
	};
	CHECK(Walk(tune::Search::Instant, Dir::HtoD, 2, flat, one_loop) ==
		  (std::vector<std::string>{"2/2/1M/nt_store@avx512"}));
}

// QUICK: INSTANT's two, then the portfolio with threads relative to the estimate (x4, x2, x0.5), then the
// descent. A step needs more than selection's 2%: a threads neighbour 2.5% better moves the descent there (it
// then tries that neighbour's buffer shapes); exactly 2% or 1.5% better does not, and the walk ends after one pass.
TEST(quick_steps_only_for_more_than_two_percent)
{
	const auto with = [](double objective)
	{ return [objective](const std::string& key, std::size_t) { return key == "16/2/1M/memcpy" ? objective : 0.0; }; };
	const auto moved = Walk(tune::Search::Quick, Dir::HtoD, 2, with(std::log(1.025)));
	CHECK((std::vector<std::string>(moved.begin(), moved.begin() + 8)) ==
		  (std::vector<std::string>{"2/2/1M/memcpy", "2/2/1M/nt_store", "8/1/2M/nt_store@avx2", "4/1/1M/cached@avx2",
									"1/2/4M/memcpy", "8/1/1M/nt_store@avx2", "1/4/1M/nt_store@avx2", "1/1/1M/movsb"}));
	CHECK(std::count(moved.begin(), moved.end(), "16/4/4M/memcpy") == 1);
	for (const double objective : {std::log1p(0.02), std::log(1.015)})
	{
		const auto stayed = Walk(tune::Search::Quick, Dir::HtoD, 2, with(objective));
		CHECK(std::count(stayed.begin(), stayed.end(), "16/2/1M/memcpy") == 1);
		CHECK(std::count(stayed.begin(), stayed.end(), "16/4/4M/memcpy") == 0);
		CHECK(std::size(stayed) < std::size(moved));
	}
}

namespace {
// One process's record at one size: `gbps` per sample, or a status without samples.
CopyRun Record(std::uint64_t bytes, std::optional<StagingSettings> staged, std::vector<double> gbps,
			   const char* status = "complete")
{
	CopyRun run;
	run.bytes = bytes;
	run.mode = staged ? Mode::Chunks : Mode::Driver;
	run.settings = staged.value_or(StagingSettings{});
	run.sample_gbps = std::move(gbps);
	run.status = status;
	return run;
}
} // namespace

// QUICK's objective compares a setting with the driver at 16 MiB and up only: a setting 50% faster at 4M but 2%
// slower at 64M scores below zero. Without a size >= 16 MiB in the grid every size counts. A size the setting
// skipped for capacity compares nothing; a setting not measured yet has no score, one that failed scores -inf.
TEST(quick_objective_compares_large_copies_with_the_driver)
{
	const StagingSettings s{2, 2, 1u << 20, "memcpy", false};
	TuneArgs args;
	args.grid.sizes = {4u << 20, 64u << 20};
	const std::vector<CopyRun> runs{Record(4u << 20, std::nullopt, {10, 10}), Record(64u << 20, std::nullopt, {10, 10}),
									Record(4u << 20, s, {15, 15}), Record(64u << 20, s, {9.8, 9.8})};
	CHECK(std::abs(*tune::Score(args, runs, Dir::HtoD, s, false) - std::log(0.98)) < 1e-12);
	args.grid.sizes = {1u << 20, 4u << 20};
	const std::vector<CopyRun> small{Record(1u << 20, std::nullopt, {10, 10}), Record(4u << 20, std::nullopt, {10, 10}),
									 Record(1u << 20, s, {12, 12}), Record(4u << 20, s, {15, 15})};
	CHECK(std::abs(*tune::Score(args, small, Dir::HtoD, s, false) - (std::log(1.2) + std::log(1.5)) / 2) < 1e-12);
	args.grid.sizes = {64u << 20, 1u << 30};
	const std::vector<CopyRun> capacity{Record(64u << 20, std::nullopt, {10, 10}),
										Record(1u << 30, std::nullopt, {10, 10}), Record(64u << 20, s, {12, 12}),
										Record(1u << 30, s, {}, "capacity")};
	CHECK(std::abs(*tune::Score(args, capacity, Dir::HtoD, s, false) - std::log(1.2)) < 1e-12);
	const std::vector<CopyRun> failed{Record(64u << 20, std::nullopt, {10, 10}), Record(1u << 30, std::nullopt, {10, 10}),
									  Record(64u << 20, s, {}, "failed"), Record(1u << 30, s, {}, "failed")};
	CHECK(!tune::Score(args, failed, Dir::HtoD, s, false));
	CHECK(*tune::Score(args, failed, Dir::HtoD, s, true) == -std::numeric_limits<double>::infinity());
}

// A setting whose processes fail is measured once, then ranks last: the walk goes on past it and ends, and the
// failed setting is not among those completeness waits for (else a setting that always fails would keep the result
// partial on every --resume).
TEST(a_failing_setting_is_measured_once_and_not_awaited)
{
	for (const auto search : {tune::Search::Instant, tune::Search::Quick})
	{
		std::map<std::string, double> measured;
		std::map<std::string, std::size_t> attempts;
		const tune::ScoreSetting score = [&](const StagingSettings& s, bool failed) -> std::optional<double>
		{
			const auto found = measured.find(Key(s));
			return found != measured.end()		? std::optional<double>(found->second)
				   : failed						? std::optional<double>(-std::numeric_limits<double>::infinity())
												: std::nullopt;
		};
		const tune::MeasureSetting measure = [&](const StagingSettings& s) -> Result<int>
		{
			if (++attempts[Key(s)] > 3)
			{
				return 1; // measured again and again: stop the walk so the test reports it
			}
			if (Key(s) != "2/2/1M/memcpy") // this one's processes always fail
			{
				measured[Key(s)] = 0;
			}
			return 0;
		};
		std::vector<StagingSettings> counted;
		CHECK_EQ(check::Require(tune::RunWalk(search, Dir::HtoD, 2, false, [](StagingSettings s) { return s; }, score,
											  measure, counted)),
				 0);
		CHECK_EQ(attempts["2/2/1M/memcpy"], std::size_t{1});
		CHECK_EQ(std::size(counted), std::size(measured));
		CHECK(std::none_of(counted.begin(), counted.end(),
						   [](const StagingSettings& s) { return Key(s) == "2/2/1M/memcpy"; }));
	}
}

// A descent that always finds a better setting stops at 31 staged settings (32 processes with the driver).
// Settings the machine cannot run are skipped without spending the budget.
TEST(quick_stays_within_its_budget)
{
	const auto improving = [](const std::string&, std::size_t order) { return 0.05 * static_cast<double>(order); };
	CHECK_EQ(std::size(Walk(tune::Search::Quick, Dir::DtoH, 4, improving)), std::size_t{31});
	const tune::AdmitSetting no_avx2 = [](StagingSettings s) -> std::optional<StagingSettings>
	{
		if (s.loop.find("@avx2") != std::string::npos)
		{
			return std::nullopt;
		}
		return s;
	};
	const auto skipped = Walk(tune::Search::Quick, Dir::DtoH, 4, improving, no_avx2);
	CHECK_EQ(std::size(skipped), std::size_t{31});
	CHECK(std::none_of(skipped.begin(), skipped.end(),
					   [](const std::string& key) { return key.find("@avx2") != std::string::npos; }));
}

// A probe that wrote a wrong measurement is a Verify error, not a usage error; a probe child that exits 3 (this
// machine cannot run it) stays Unsupported, any other exit is Verify.
TEST(estimate_probe_errors_have_their_kinds)
{
	TuneArgs args;
	args.dirs = {Dir::HtoD};
	const auto read = [&](bool verified)
	{
		const auto cpu = fmt::format(R"({{"status":"complete","cpu_copy_probes":[{{"direction":"htod","status":)"
									 R"("complete","verified":true,"gbps":20,"pages_requested":"{}"}}]}})",
									 PagesName(args.pages));
		const auto pinned = fmt::format(R"([{{"direction":"htod","path":"pinned_ce","bytes":67108864,"status":)"
										R"("complete","verified":{},"gbps":26,"pages_requested":"driver"}}])",
										verified);
		JsonParser cpu_parser, pinned_parser;
		Json cpu_json, pinned_json;
		CHECK(!cpu_parser.parse(cpu).get(cpu_json) && !pinned_parser.parse(pinned).get(pinned_json));
		return tune::ReadEstimate(args, cpu_json, pinned_json);
	};
	const auto good = check::Require(read(true));
	CHECK(good.cpu_gbps.at(Dir::HtoD) == 20 && good.pinned_gbps.at(Dir::HtoD) == 26);
	CHECK_ERROR_KIND(read(false), Verify);
	CHECK(tune::ProbeFailure(3).GetKind() == Error::Kind::Unsupported);
	CHECK(tune::ProbeFailure(6).GetKind() == Error::Kind::Verify);
	CHECK(tune::ProbeFailure(1).GetKind() == Error::Kind::Verify);
}

// The staged settings follow the probe, so QUICK lists its drivers and refuses a --max-processes below its bound.
// The driver times 3x the samples (tuner.cpp kDriverSampleFactor): one slow driver sample sets every setting's bar.
TEST(quick_bounds_its_processes_before_the_probe)
{
	TuneArgs args;
	args.search = tune::Search::Quick;
	args.grid = check::Require(tune::ParseTuningParams(args.search, ""));
	const auto drivers = check::Require(MakeRuns(args));
	CHECK_EQ(std::size(drivers), std::size_t{2});
	for (const auto& run : drivers)
	{
		CHECK(run.copy.mode == Mode::Driver);
		CHECK_EQ(run.samples, 3 * args.grid.samples);
	}
	args.max_processes = 2 * 32 - 1;
	CHECK(!MakeRuns(args));
	args.max_processes = 2 * 32;
	CHECK(MakeRuns(args));
}

// The children run where the hook's default puts the workers, the GPU's CPUs (NVML's set, else its sysfs
// node's), unless that cannot be known or the user placed the workers; then they run unpinned and the record
// says why. Never an error.
TEST(measurement_children_run_on_the_gpu_cpus_or_unpinned_with_a_reason)
{
	const auto allowed = check::Require(GetThreadAffinity());
	if (std::size(allowed) < 2)
	{
		std::printf("  needs two allowed CPUs; skipped\n");
		return;
	}
	check::TwoNodeSysfs sysfs(allowed);
	const std::optional<std::string> gpu{check::kFixtureGpu};
	const auto place = [&](std::vector<std::string> env, const GpuCpus& nvml = {},
						   std::optional<std::string> pci = std::optional<std::string>{check::kFixtureGpu})
	{ return check::Require(tune::MeasurementPlacement(env, pci, nvml, sysfs.root)); };
	const auto local = place({});
	CHECK(local.cpus == sysfs.node1);
	CHECK_EQ(local.decision, "cpus=" + FormatCpuList(sysfs.node1));
	CHECK_EQ(local.detail, std::string("sysfs node cpulist, sysfs node 1"));
	const auto nvml =
		place({}, [&](const std::string&) -> Result<std::vector<unsigned>> { return sysfs.node0; });
	CHECK(nvml.cpus == sysfs.node0 && nvml.detail == "nvml, sysfs node 1");
	CHECK(place({"CMT_WORKER_NUMA=1", "CUDA_VISIBLE_DEVICES=0"}).cpus == sysfs.node1);
	for (const auto& [env, pci, why] :
		 {std::tuple{std::vector<std::string>{"CMT_WORKER_NUMA=0"}, gpu, "--numa off"},
		  std::tuple{std::vector<std::string>{"CMT_WORKER_CPUS=0"}, gpu, "places the workers"},
		  std::tuple{std::vector<std::string>{}, std::optional<std::string>{}, "PCI address is unknown"}})
	{
		const auto unpinned = place(env, {}, pci);
		CHECK(std::empty(unpinned.cpus));
		CHECK(unpinned.decision == "unpinned" && unpinned.detail.find(why) != std::string::npos);
	}
	sysfs.Write(std::string("bus/pci/devices/") + check::kFixtureGpu + "/numa_node", "-1"); // and no NVML
	CHECK(place({}).decision == "unpinned" && place({}).detail.find("the CPUs of GPU ") == 0);
	sysfs.Write("devices/system/node/has_cpu", "0");
	CHECK_EQ(place({}).decision, std::string("unpinned"));
	CHECK_EQ(place({}).detail, std::string("one NUMA node"));
}

// A child starts on ProcessCommand::cpus (its own mask, as /proc reports it); the spawning thread gets its
// mask back.
TEST(a_child_starts_on_the_requested_cpus_and_the_caller_keeps_its_own)
{
	const auto allowed = check::Require(GetThreadAffinity());
	char path[] = "/tmp/cmt-affinity-XXXXXX";
	CHECK(mkdtemp(path) != nullptr);
	ProcessCommand command;
	command.executable = "/bin/sh";
	command.arguments = {"-c", "grep Cpus_allowed_list /proc/self/status"};
	command.stdout_path = std::string(path) + "/out";
	command.cpus = {allowed.back()};
	const auto result = check::Require(RunProcess(command));
	CHECK(result.completed && result.exit_code == 0);
	const auto text = check::Require(ReadTextFile(command.stdout_path));
	CHECK_EQ(text, "Cpus_allowed_list:\t" + std::to_string(allowed.back()) + "\n");
	CHECK(check::Require(GetThreadAffinity()) == allowed);
	std::error_code error;
	std::filesystem::remove_all(path, error);
}

// A measurement child gets the three inherited variables when they are set (tune::InheritedEnvironment) and
// nothing else of the tuner's environment (inherit_environment = false, as every tune child has).
TEST(a_tuning_child_sees_only_the_inherited_device_variables)
{
	const char* names[] = {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LD_LIBRARY_PATH", "FOO"};
	std::optional<std::string> saved[4];
	for (std::size_t i = 0; i < std::size(names); i++)
	{
		if (const char* value = std::getenv(names[i]))
		{
			saved[i] = value;
		}
	}
	CHECK(setenv("CUDA_VISIBLE_DEVICES", "GPU-test", 1) == 0);
	CHECK(setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1) == 0);
	CHECK(setenv("LD_LIBRARY_PATH", "/opt/test/lib", 1) == 0);
	CHECK(setenv("FOO", "1", 1) == 0);
	const auto inherited = tune::InheritedEnvironment();
	CHECK_EQ(inherited,
			 (std::vector<std::string>{"CUDA_VISIBLE_DEVICES=GPU-test", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
									   "LD_LIBRARY_PATH=/opt/test/lib"}));
	char path[] = "/tmp/cmt-env-XXXXXX";
	CHECK(mkdtemp(path) != nullptr);
	ProcessCommand command;
	command.executable = "/usr/bin/env";
	command.environment = inherited;
	command.environment.push_back("CMT_EXPECTED_RUN=x"); // what the tuner adds itself still arrives
	command.inherit_environment = false;
	command.stdout_path = std::string(path) + "/out";
	const auto result = check::Require(RunProcess(command));
	CHECK(result.completed && result.exit_code == 0);
	CHECK_EQ(check::Require(ReadTextFile(command.stdout_path)),
			 std::string("CMT_EXPECTED_RUN=x\nCUDA_DEVICE_ORDER=PCI_BUS_ID\nCUDA_VISIBLE_DEVICES=GPU-test\n"
						 "LD_LIBRARY_PATH=/opt/test/lib\n")); // sorted by name; no FOO, PATH or HOME
	CHECK(unsetenv("CUDA_DEVICE_ORDER") == 0);
	CHECK(unsetenv("LD_LIBRARY_PATH") == 0);
	CHECK_EQ(tune::InheritedEnvironment(), (std::vector<std::string>{"CUDA_VISIBLE_DEVICES=GPU-test"}));
	for (std::size_t i = 0; i < std::size(names); i++)
	{
		CHECK((saved[i] ? setenv(names[i], saved[i]->c_str(), 1) : unsetenv(names[i])) == 0);
	}
	std::error_code error;
	std::filesystem::remove_all(path, error);
}

// The metadata child's record decides the children's placement: its NVML set when it has one, else sysfs read
// now; the record (tune.json identity child_placement) follows what sysfs shows.
TEST(the_machine_record_places_the_children)
{
	const auto allowed = check::Require(GetThreadAffinity());
	if (std::size(allowed) < 2)
	{
		std::printf("  needs two allowed CPUs; skipped\n");
		return;
	}
	check::TwoNodeSysfs sysfs(allowed);
	const auto place = [&](const std::string& nvml_cpus)
	{
		JsonParser parser;
		Json machine;
		const auto text = std::string(R"({"gpus":[{"pci_bus_id":")") + check::kFixtureGpu +
						  R"(","cpu_affinity":{"nvml_cpus":)" + nvml_cpus + "}}]}";
		CHECK(!parser.parse(text).get(machine));
		return check::Require(PlacementFor(TuneArgs{}, machine, sysfs.root));
	};
	CHECK(place("null").cpus == sysfs.node1); // NVML did not answer: sysfs node 1
	const auto nvml = place('"' + check::TwoNodeSysfs::List(sysfs.node0) + '"');
	CHECK(nvml.cpus == sysfs.node0 && nvml.detail == "nvml, sysfs node 1");
	sysfs.Write("devices/system/node/has_cpu", "0");
	CHECK_EQ(place("null").decision, std::string("unpinned"));
}

// A benchmark child starts clean (only options.env and the run's settings) on options.placement.cpus.
TEST(a_benchmark_child_is_clean_and_placed)
{
	TuneArgs options;
	options.env = {"CUDA_VISIBLE_DEVICES=GPU-test"};
	options.placement.cpus = {3, 5};
	BenchmarkRun run;
	run.copy.mode = Mode::Driver;
	run.sizes = {1u << 20};
	const auto command =
		BenchmarkCommand(options, run, "/fixture/request.json", "/fixture/result.json", "/fixture/p");
	CHECK(!command.inherit_environment);
	CHECK(command.cpus == options.placement.cpus);
	CHECK(command.environment.front() == "CUDA_VISIBLE_DEVICES=GPU-test");
}

// tune.json's request records the registration limit as selection reads it (config.cpp, RegistrationLimitOf):
// without --max-registered-mib, null and the explicit_only statement, in the request and its pinning settings;
// with it, the value. tests/test_selection.cpp builds its requests with these exact fields.
TEST(the_request_records_an_explicit_registration_limit_or_none)
{
	const auto request = [](std::optional<std::size_t> limit_mib)
	{
		TuneArgs options;
		options.search = tune::Search::FullWithKernel;
		options.grid = check::Require(tune::ParseTuningParams(options.search, ""));
		options.max_registered_mib = limit_mib;
		return check::Require(RequestJson(options));
	};
	const auto fields = [](const std::string& text)
	{
		JsonParser parser;
		Json document;
		CHECK(!parser.parse(text).get(document));
		std::string_view rule;
		CHECK(!document["registration_limit"].get(rule) && rule == "explicit_only");
		Json recorded;
		CHECK(!document["progressive_pinning_settings"].get(recorded));
		const auto settings = check::Require(ParsePinningSettings(recorded));
		std::uint64_t mib = 0;
		const bool null_mib = document["max_registered_mib"].is_null();
		CHECK(null_mib || !document["max_registered_mib"].get(mib));
		return std::make_tuple(null_mib, mib, settings);
	};
	const auto [none_is_null, none_mib, none] = fields(request(std::nullopt));
	CHECK(none_is_null && !none.max_registered_bytes);
	CHECK_EQ(none.max_chunk_bytes, std::uint64_t{64} << 20);
	const auto [set_is_null, set_mib, set] = fields(request(1024));
	CHECK(!set_is_null && set_mib == 1024 && set.max_registered_bytes == std::uint64_t{1} << 30);
	// A limit below the 64 MiB chunk lowers the chunk to it (still a valid setting).
	const auto [small_is_null, small_mib, small] = fields(request(16));
	CHECK(!small_is_null && small_mib == 16 && small.max_chunk_bytes == std::uint64_t{16} << 20);
}

// Pinned allocation (tune::PlanHostAllocation): quick and full measure it unless --host_alloc off or a
// --max-pinned-mib below the hook's 2 MiB; instant never does. Private memory only where THP allows it.
TEST(pinned_allocation_is_measured_by_quick_and_full_only)
{
	const auto plan = [](tune::Search search, bool host_alloc = true, std::size_t max_pinned_mib = 1024,
						 std::optional<std::string> thp = std::string("always [madvise] never\n"))
	{
		TuneArgs options;
		options.search = search;
		options.host_alloc = host_alloc;
		options.max_pinned_mib = max_pinned_mib;
		return tune::PlanHostAllocation(options, thp);
	};
	for (auto search : {tune::Search::Quick, tune::Search::Full, tune::Search::FullWithKernel})
	{
		const auto measured = plan(search);
		CHECK(!measured.skipped && measured.bytes == std::uint64_t{1} << 30 && measured.rounds == 2 &&
			  measured.processes == 3 && measured.thp == "madvise" && measured.private_memory);
		CHECK(plan(search, false).skipped == "host_alloc off");
	}
	CHECK(plan(tune::Search::Instant).skipped == "instant search");
	CHECK(plan(tune::Search::Instant, false).skipped == "instant search");
	// --max-pinned-mib caps the allocation; below the hook's 2 MiB there is nothing to compare.
	CHECK(!plan(tune::Search::Quick, true, 256).skipped && plan(tune::Search::Quick, true, 256).bytes == 256u << 20);
	CHECK(!plan(tune::Search::Quick, true, 2).skipped);
	CHECK(plan(tune::Search::Quick, true, 1).skipped == "max_pinned_mib below 2 MiB");
	// The THP setting, read and recorded, never changed: private memory only under madvise or always.
	CHECK(plan(tune::Search::Quick, true, 1024, "[always] madvise never\n").private_memory);
	const auto never = plan(tune::Search::Quick, true, 1024, "always madvise [never]\n");
	CHECK(!never.skipped && !never.private_memory && never.thp == "never");
	const auto unread = plan(tune::Search::Quick, true, 1024, std::nullopt);
	CHECK(!unread.skipped && !unread.private_memory && !unread.thp);
	for (const auto& skipped : {plan(tune::Search::Instant), plan(tune::Search::Quick, false),
								plan(tune::Search::Quick, true, 1)})
	{
		CHECK(std::any_of(std::begin(tuning::kHostAllocReasons), std::end(tuning::kHostAllocReasons),
						  [&](const char* reason) { return skipped.skipped == reason; }));
	}
}

// The rule `cmt run --config` turns the hook on by: the copies' (every sample faster than every baseline sample,
// and the median more than 2% faster), in seconds.
TEST(the_hook_is_used_only_when_clearly_faster)
{
	const std::vector<double> driver{0.39, 0.40, 0.41, 0.39, 0.40, 0.42};
	CHECK(tuning::HostAllocClearlyFaster({0.11, 0.12, 0.11, 0.12, 0.11, 0.13}, driver));
	CHECK(!tuning::HostAllocClearlyFaster(driver, {0.11, 0.12, 0.11, 0.12, 0.11, 0.13}));
	// One hook sample as slow as the fastest driver sample: not clearly faster, however good the median.
	CHECK(!tuning::HostAllocClearlyFaster({0.11, 0.12, 0.11, 0.12, 0.11, 0.39}, driver));
	// Every sample faster, but the medians within 2%.
	CHECK(!tuning::HostAllocClearlyFaster({1.00, 1.00}, {1.01, 1.01}));
	CHECK(tuning::HostAllocClearlyFaster({1.00, 1.00}, {1.03, 1.03}));
	// Too few samples to say.
	CHECK(!tuning::HostAllocClearlyFaster({0.1}, driver) && !tuning::HostAllocClearlyFaster({0.1, 0.1}, {0.4}));
	CHECK(!tuning::HostAllocClearlyFaster({}, {}));
}

namespace {
tuning::HostAllocation MeasuredAllocation(std::vector<double> shared_alloc, std::string private_state = "measured")
{
	tuning::HostAllocation section;
	section.state = "measured";
	section.bytes = std::uint64_t{1} << 30;
	section.threads = 8;
	section.thp = "madvise";
	section.driver = {{0.39, 0.40, 0.41}, {0.15, 0.15, 0.16}, {0, 0, 0}};
	section.shared = {std::move(shared_alloc), {0.15, 0.16, 0.15}, {0, 0, 0}};
	section.private_state = std::move(private_state);
	if (section.private_state == "measured")
	{
		section.private_memory = {{0.06, 0.06, 0.07}, {0.10, 0.11, 0.10}, {1048576, 1048576, 1046528}};
	}
	return section;
}

std::string Encoded(const tuning::HostAllocation& section)
{
	JsonWriter writer;
	tuning::WriteHostAllocation(writer, section);
	return check::Require(FinishJson(writer));
}

// `section_json` as the "host_allocation" of a document, read back.
Result<std::optional<tuning::HostAllocation>> ReadBack(const std::string& section_json)
{
	JsonParser parser;
	Json document;
	const std::string text = section_json.empty() ? std::string(R"({"schema":"cmt.tuning"})")
												  : R"({"schema":"cmt.tuning","host_allocation":)" + section_json + "}";
	if (parser.parse(text).get(document))
	{
		return UsageError("test document is not JSON");
	}
	return tuning::ReadHostAllocation(document);
}

std::string ReplacedOnce(std::string text, const std::string& from, const std::string& to)
{
	const auto at = text.find(from);
	CHECKF(at != std::string::npos, "no %s in %s", from.c_str(), text.c_str());
	return at == std::string::npos ? text : text.replace(at, from.size(), to);
}
} // namespace

// tune.json "host_allocation" reads back as written; a file without it (an older tune) reads as none; anything
// else, a stale decision or median included, is refused.
TEST(the_host_allocation_section_reads_back_exactly)
{
	const auto faster = MeasuredAllocation({0.11, 0.12, 0.11});
	const auto text = Encoded(faster);
	CHECKF(text.find(R"("decision":{"shared":true,"private":true})") != std::string::npos, "%s", text.c_str());
	CHECKF(text.find(R"("alloc_median_seconds":0.4)") != std::string::npos, "%s", text.c_str());
	const auto read = check::Require(ReadBack(text));
	CHECK(read && read->state == "measured" && read->bytes == faster.bytes && read->threads == 8 &&
		  read->thp == "madvise" && read->driver.alloc_seconds == faster.driver.alloc_seconds &&
		  read->shared.free_seconds == faster.shared.free_seconds &&
		  read->private_memory.huge_kib == faster.private_memory.huge_kib && read->private_state == "measured");
	CHECK_EQ(Encoded(*read), text);
	// Older tunes have no section: nothing measured, nothing decided.
	CHECK(!check::Require(ReadBack("")));
	// Not faster; private not allowed (null arm); private failed; THP unread.
	auto slower = MeasuredAllocation({0.39, 0.45, 0.41}, "not_allowed");
	slower.thp = std::nullopt;
	const auto slower_text = Encoded(slower);
	const auto not_allowed = R"("private_state":"not_allowed","private":null,"decision":{"shared":false,"private":false})";
	CHECKF(slower_text.find(R"("thp":null)") != std::string::npos && slower_text.find(not_allowed) != std::string::npos,
		   "%s", slower_text.c_str());
	CHECK(check::Require(ReadBack(slower_text)));
	CHECK(check::Require(ReadBack(Encoded(MeasuredAllocation({0.11, 0.12, 0.11}, "failed")))));
	// Not measured, for each reason.
	for (const char* reason : tuning::kHostAllocReasons)
	{
		tuning::HostAllocation skipped;
		skipped.reason = reason;
		const auto skipped_text = Encoded(skipped);
		CHECK_EQ(skipped_text, fmt::format(R"({{"state":"not_measured","reason":"{}"}})", reason) + "\n");
		const auto back = check::Require(ReadBack(skipped_text));
		CHECK(back && back->state == "not_measured" && back->reason == reason);
	}
	const std::vector<std::string> refused{
		ReplacedOnce(text, R"("decision":{"shared":true)", R"("decision":{"shared":false)"),
		ReplacedOnce(text, R"("private":true})", R"("private":false})"),
		ReplacedOnce(text, R"("alloc_median_seconds":0.4)", R"("alloc_median_seconds":0.3)"),
		ReplacedOnce(text, R"("alloc_seconds":[0.39,)", R"("alloc_seconds":[0.01,)"),
		ReplacedOnce(text, R"("state":"measured")", R"("state":"measured","note":"x")"),
		ReplacedOnce(text, R"("threads":8)", R"("threads":0)"),
		ReplacedOnce(text, R"("bytes":1073741824)", R"("bytes":0)"),
		ReplacedOnce(text, R"("thp":"madvise")", R"("thp":3)"),
		ReplacedOnce(text, R"("huge_kib":[0,0,0])", R"("huge_kib":[0,0])"),
		ReplacedOnce(text, R"("free_seconds":[0.15,0.15,0.16])", R"("free_seconds":[0.15,0.15,-0.16])"),
		ReplacedOnce(text, R"("private_state":"measured")", R"("private_state":"not_allowed")"),
		ReplacedOnce(slower_text, R"("private_state":"not_allowed")", R"("private_state":"measured")"),
		ReplacedOnce(text, R"("private_state":"measured")", R"("private_state":"maybe")"),
		R"({"state":"not_measured","reason":"/home/alice"})",
		R"({"state":"not_measured"})",
		R"({"state":"not_measured","reason":"instant search","bytes":1})",
		R"({"state":"skipped","reason":"instant search"})",
		R"({"reason":"instant search"})",
		R"("measured")",
		"null"};
	for (std::size_t i = 0; i < refused.size(); i++)
	{
		const auto read_refused = ReadBack(refused[i]);
		CHECKF(!read_refused && read_refused.error().IsUsage(), "variant %zu read: %s", i, refused[i].c_str());
	}
}

TEST_MAIN()
