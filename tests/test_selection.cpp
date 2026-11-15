// The selection line (config.h): the table the hook reads from line 1 is the one the full parse computes on
// this machine, or the hook parses every run.
//   test_selection                 the cases below
//   test_selection --cmt CMT       `cmt check` exit codes and --write
//   test_selection --resume CMT    `cmt tune --resume` with a tie band re-selects and measures nothing (GPU)
//   test_selection --instant CMT   `cmt tune --search instant` measures what it estimated, once (GPU)
//   test_selection TUNE_JSON...    each published result converted as `cmt check --write` does: tuning::Load,
//                                  given the line and a broken body, equals the full parse of the original file
#include "cuda/kernel_status.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "check.h"
#include "common/helpers.h"
#include "config.h"
#include "share.h"
#include "tuning_fixture.h"

using namespace cmt;

namespace {
using namespace tuning_fixture;

// INSTANT. HtoD: an unpinned nt_store layout (40 GB/s) beats two memcpy layouts; DtoH: the 128 MiB memcpy pool
// (30) beats the 2 MiB one (20). A CPU without vector ISAs cannot run nt_store, and less than 128 MiB of headroom
// cannot hold the large pool: both change the full parse's winner.
std::vector<CopyRun> InstantRuns()
{
	const StagingSettings small{1, 2, kMiB, "memcpy", false};
	const StagingSettings large{2, 4, 16 * kMiB, "memcpy", false};
	const StagingSettings vector{1, 2, kMiB, "nt_store", false};
	const auto run = [](Dir direction, Mode mode, double gbps, StagingSettings settings = {})
	{ return Run(direction, 4 * kMiB, mode, gbps, "instant", std::move(settings)); };
	return {run(Dir::HtoD, Mode::Driver, 10),		 run(Dir::HtoD, Mode::Chunks, 20, small),
			run(Dir::HtoD, Mode::Chunks, 30, large), run(Dir::HtoD, Mode::Chunks, 40, vector),
			run(Dir::DtoH, Mode::Driver, 10),		 run(Dir::DtoH, Mode::Chunks, 20, small),
			run(Dir::DtoH, Mode::Chunks, 30, large)};
}

// Measured, HtoD: at 1 MiB a staged layout wins; at 16 MiB it loses (a driver row); 32 MiB was requested but
// failed (a driver row); at 64 MiB progressive pinning (30) beats the staged layout (12), which wins wherever
// pinning is unavailable. Pinning has no limit (Document's old-default request), so its row extends.
std::vector<CopyRun> MeasuredRuns()
{
	const StagingSettings small{1, 2, kMiB, "memcpy", false};
	const auto run = [&](std::uint64_t bytes, Mode mode, double gbps)
	{ return Run(Dir::HtoD, bytes, mode, gbps, "measured", mode == Mode::Chunks ? small : StagingSettings{}); };
	auto failed = run(32 * kMiB, Mode::Driver, 10);
	failed.status = "failed";
	return {run(kMiB, Mode::Driver, 10),		run(kMiB, Mode::Chunks, 20),
			run(16 * kMiB, Mode::Driver, 10),	run(16 * kMiB, Mode::Chunks, 9),
			failed,								run(64 * kMiB, Mode::Driver, 10),
			run(64 * kMiB, Mode::Chunks, 12),	run(64 * kMiB, Mode::ProgressivePinning, 30)};
}

// `text` with a tie band recorded in its request, as `cmt tune --tuning_params="tie=N%"` writes it.
std::string WithTie(std::string text, unsigned percent)
{
	return text.insert(text.find("\"process_rounds\""), fmt::format("\"tie_percent\":{},", percent));
}

// Measured, HtoD, against a 10 GB/s driver. At 64 MiB: 4x4x1M (20, 16 MiB pool) has the best median; within 2%
// (19.6) tie 2x2x1M (19.9) and 2x1x2M (19.7) with the smallest pool and fewest workers, 4x1x1M (19.95, same pool,
// more workers) and 1x4x4M (19.8, one worker, 16 MiB); 1x1x1M (19.58, 1 MiB) is just outside. At 1 MiB: 1x1x1M
// (20) stages the copy whole; 2x2x512K (19.8) ties with a chunk below the copy size.
std::vector<CopyRun> TieRuns()
{
	const auto run =
		[](std::uint64_t bytes, double gbps, int workers = 0, int slots = 0, std::size_t chunk = 0)
	{
		return Run(Dir::HtoD, bytes, workers ? Mode::Chunks : Mode::Driver, gbps, "measured",
				   workers ? StagingSettings{workers, slots, chunk, "memcpy", false} : StagingSettings{});
	};
	return {run(64 * kMiB, 10),
			run(64 * kMiB, 20, 4, 4, kMiB),
			run(64 * kMiB, 19.9, 2, 2, kMiB),
			run(64 * kMiB, 19.95, 4, 1, kMiB),
			run(64 * kMiB, 19.7, 2, 1, 2 * kMiB),
			run(64 * kMiB, 19.58, 1, 1, kMiB),
			run(64 * kMiB, 19.8, 1, 4, 4 * kMiB),
			run(kMiB, 10),
			run(kMiB, 20, 1, 1, kMiB),
			run(kMiB, 19.8, 2, 2, kMiB / 2)};
}

// Every field the hook reads, the resolved copy loop included.
bool SameTable(const Table& a, const Table& b)
{
	if (std::size(a.rows) != std::size(b.rows) || a.max_pinned_bytes != b.max_pinned_bytes ||
		a.worker_placement.cpus != b.worker_placement.cpus ||
		a.worker_placement.gpu_numa != b.worker_placement.gpu_numa ||
		a.worker_placement.numa_node != b.worker_placement.numa_node ||
		a.source_cpu_model != b.source_cpu_model || a.source_cpu_arch != b.source_cpu_arch ||
		a.source_gpu_name != b.source_gpu_name)
	{
		return false;
	}
	for (std::size_t i = 0; i < std::size(a.rows); i++)
	{
		const Row& x = a.rows[i];
		const Row& y = b.rows[i];
		if (x.dir != y.dir || x.min_bytes != y.min_bytes || x.max_bytes != y.max_bytes || x.mode != y.mode ||
			x.settings.workers != y.settings.workers || x.settings.slots != y.settings.slots ||
			x.settings.chunk_bytes != y.settings.chunk_bytes || x.settings.wc_slots != y.settings.wc_slots ||
			x.settings.loop.id != y.settings.loop.id ||
			x.settings.loop.ctx.flush_batch != y.settings.loop.ctx.flush_batch ||
			x.settings.loop.ctx.write_back != y.settings.loop.ctx.write_back ||
			x.pinning.first_chunk_bytes != y.pinning.first_chunk_bytes ||
			x.pinning.max_chunk_bytes != y.pinning.max_chunk_bytes ||
			x.pinning.max_registered_bytes != y.pinning.max_registered_bytes ||
			x.pinning.read_only != y.pinning.read_only || x.loop_fell_back != y.loop_fell_back ||
			x.loop_name != y.loop_name)
		{
			return false;
		}
	}
	return true;
}

// What each row runs, before a CPU resolves the loop name.
std::string Choices(const Table& table)
{
	std::string text;
	for (const Row& row : table.rows)
	{
		text += fmt::format("{} {} {} {} {} {};", row.min_bytes, ModeName(row.mode), row.loop_name,
							row.settings.workers, row.settings.slots, row.settings.chunk_bytes);
	}
	return text;
}

// For every local CPU, headroom and pinning availability: an admitted line's table equals the full parse; where
// the line is refused, the full parse selects differently, so refusing was necessary. Returns {admitted, refused}.
std::pair<std::size_t, std::size_t> CheckAdmission(const std::string& text)
{
	CpuIsas none;
	none.mask = 0;
	LocalCapabilities everything;
	everything.cpu.mask = UINT16_MAX;
	everything.progressive_pinning = true;
	std::pair<std::size_t, std::size_t> counts;
	for (const CpuIsas& cpu : {DetectCpuIsas(), none})
	{
		for (std::size_t headroom :
			 {std::size_t{0}, 2 * kMiB, 8 * kMiB, 64 * kMiB, 128 * kMiB, 130 * kMiB, SIZE_MAX})
		{
			for (bool pinning : {false, true})
			{
				const LocalCapabilities local{cpu, headroom, pinning};
				const auto full = check::Require(ParseTuningResult(text, local));
				std::uint64_t file_bytes = 0;
				const auto selected = tuning::ReadSelection(text, local, file_bytes);
				(selected ? counts.first : counts.second)++;
				CHECK(selected ? SameTable(*selected, full) && file_bytes == std::size(text)
							   : selected.error().IsUnsupported() &&
									 Choices(check::Require(tuning::ReadSelection(text, everything, file_bytes))) !=
										 Choices(full));
			}
		}
	}
	return counts;
}

// tuning::Load's capabilities without CMT_OVERRIDE.
LocalCapabilities LoadCapabilities()
{
	LocalCapabilities local;
	local.max_pinned_bytes = HostMemoryAvailable(Pages::Small).value_or(0);
	local.progressive_pinning = ProgressivePinningAvailable();
	return local;
}

// The file through tuning::Load equals `expected`.
bool LoadsAs(const std::string& text, const Table& expected)
{
	check::Require(WriteTextFile("test_selection.json", text));
	const auto loaded = tuning::Load("test_selection.json");
	return loaded && SameTable(**loaded, expected);
}
} // namespace

TEST(instant_line_equals_the_full_parse_where_admitted)
{
	const auto [admitted, refused] = CheckAdmission(check::Require(Reselect(Document(InstantRuns()))));
	// This CPU with room for the 128 MiB pool (pinning either way); every other case loses a winner.
	CHECK_EQ(admitted, std::size_t{6});
	CHECK_EQ(refused, std::size_t{22});
}

TEST(measured_line_equals_the_full_parse_where_admitted)
{
	const auto text = check::Require(Reselect(Document(MeasuredRuns())));
	const auto [admitted, refused] = CheckAdmission(text);
	// Pinning available and room for the 2 MiB pool, on either CPU (memcpy runs everywhere).
	CHECK_EQ(admitted, std::size_t{12});
	CHECK_EQ(refused, std::size_t{16});
	std::uint64_t file_bytes = 0;
	const auto table = check::Require(tuning::ReadSelection(text, {DetectCpuIsas(), SIZE_MAX, true}, file_bytes));
	CHECK_EQ(Choices(table), std::string("0 driver  0 0 0;1048576 staged memcpy 1 2 1048576;16777216 driver  0 0 0;"
										 "33554432 driver  0 0 0;67108864 progressive_pinning  0 0 0;"
										 "0 driver  0 0 0;")); // then DtoH: no runs
}

namespace {
constexpr std::uint64_t kGiB = 1024 * kMiB;

// Measured, DtoH, the RTX 3090 FULL with-kernel shape (at 1 GiB: pinning 25.7, the best staged 20.3, the driver
// 18.4). At 64 MiB only pinning is compared with the driver. At
// `top` (1 GiB, that file's largest size, unless given): pinning 25.7 wins; `unpinned` are the other candidates, by
// default 2x1x8M nt_store (20, 16 MiB pool), 1x1x4M memcpy (19, 4 MiB) and 1x2x1M memcpy (18.3, under the 2% gain
// over 18). Then `above`, each size's candidates as {GB/s, settings}, a driver at 18 included.
std::vector<CopyRun> BudgetRuns(
	std::uint64_t top = kGiB,
	std::vector<std::pair<double, StagingSettings>> unpinned = {{20, {2, 1, 8 * kMiB, "nt_store", false}},
																{19, {1, 1, 4 * kMiB, "memcpy", false}},
																{18.3, {1, 2, kMiB, "memcpy", false}}},
	std::map<std::uint64_t, std::vector<std::pair<double, StagingSettings>>> above = {})
{
	const auto run = [](std::uint64_t bytes, Mode mode, double gbps, StagingSettings settings = {})
	{ return Run(Dir::DtoH, bytes, mode, gbps, "measured", std::move(settings)); };
	std::vector<CopyRun> runs{run(64 * kMiB, Mode::Driver, 18), run(64 * kMiB, Mode::ProgressivePinning, 25),
							  run(top, Mode::Driver, 18.4), run(top, Mode::ProgressivePinning, 25.7)};
	for (auto& [gbps, settings] : unpinned)
	{
		runs.push_back(run(top, Mode::Chunks, gbps, settings));
	}
	for (auto& [bytes, candidates] : above)
	{
		runs.push_back(run(bytes, Mode::Driver, 18));
		for (auto& [gbps, settings] : candidates)
		{
			runs.push_back(run(bytes, Mode::Chunks, gbps, settings));
		}
	}
	return runs;
}

// Choices' DtoH part (the HtoD part is "0 driver  0 0 0;": no HtoD runs).
std::string DtohChoices(const Table& table)
{
	const auto text = Choices(table);
	return text.substr(text.find(';') + 1);
}

LocalCapabilities Everything()
{
	LocalCapabilities everything;
	everything.cpu.mask = UINT16_MAX;
	everything.progressive_pinning = true;
	return everything;
}

// `runs` as `cmt tune --max-registered-mib 1024` writes them: the limit the budget tests below apply.
std::string Limited(const std::vector<CopyRun>& runs)
{
	return Document(runs, kMachine, Limit::Explicit1G);
}
} // namespace

// A recorded 1 GiB default limit must not end the pinning rows: on an RTX 3090 (DtoH), pinning measured
// 25.4-25.6 GB/s at 2-16 GiB against 20.3 GB/s staged at 1 GiB. Without an explicit limit, the pinning row
// extends to every larger size and records no limit for the hook.
TEST(an_old_default_limit_does_not_end_pinning_rows)
{
	for (Limit limit : {Limit::OldDefault, Limit::None})
	{
		const auto table = check::Require(ParseTuningResult(Document(BudgetRuns(), kMachine, limit), Everything()));
		CHECK_EQ(DtohChoices(table), std::string("0 driver  0 0 0;67108864 progressive_pinning  0 0 0;"
												 "1073741824 progressive_pinning  0 0 0;"));
		for (std::uint64_t bytes : {kGiB, kGiB + 1, 2 * kGiB, 16 * kGiB, std::uint64_t{UINT64_MAX - 1}})
		{
			const Row* row = table.Find(Dir::DtoH, bytes);
			CHECK(row != nullptr && row->mode == Mode::ProgressivePinning && !row->pinning.max_registered_bytes);
		}
	}
	// An explicit limit in a file without the statement (any value but 1 GiB) is kept: rows end at 512 MiB.
	const auto old_explicit = BudgetRuns(512 * kMiB, {{20, {2, 1, 8 * kMiB, "memcpy", false}}});
	const auto table =
		check::Require(ParseTuningResult(Document(old_explicit, kMachine, Limit::OldExplicit512), Everything()));
	CHECK_EQ(DtohChoices(table), std::string("0 driver  0 0 0;67108864 progressive_pinning  0 0 0;"
											 "536870912 progressive_pinning  0 0 0;"
											 "536870913 staged memcpy 2 1 8388608;"));
	CHECK(*table.Find(Dir::DtoH, 512 * kMiB)->pinning.max_registered_bytes == 512 * kMiB);
}

// The request and every pinning run must agree on the limit, and the statement must be exactly explicit_only:
// otherwise the file is refused whole (Usage) rather than read with a guessed limit.
TEST(a_request_that_misstates_its_registration_limit_is_refused)
{
	const auto refused = [](const std::string& text)
	{
		const auto table = ParseTuningResult(text, Everything());
		return !table && table.error().IsUsage();
	};
	const auto none = Document(BudgetRuns(), kMachine, Limit::None);
	const auto replaced = [](std::string text, const std::string& from, const std::string& to)
	{
		const auto at = text.find(from);
		CHECK(at != std::string::npos);
		return text.replace(at, std::size(from), to);
	};
	CHECK(!refused(none));
	CHECK(refused(replaced(none, R"("registration_limit":"explicit_only")", R"("registration_limit":"default")")));
	// null without the statement: no `cmt tune` writes it.
	CHECK(refused(replaced(none, R"("registration_limit":"explicit_only",)", "")));
	// The request says none, a run records 1 GiB; or the request's settings disagree with its max_registered_mib.
	CHECK(refused(replaced(none, R"("max_registered_bytes":null,"read_only":false},"sample_count")",
						   R"("max_registered_bytes":1073741824,"read_only":false},"sample_count")")));
	CHECK(refused(replaced(none, R"("max_registered_mib":null)", R"("max_registered_mib":1024)")));
	// The field is required either way: a missing limit is not read as none.
	CHECK(refused(replaced(none, R"("max_chunk_bytes":67108864,"max_registered_bytes":null,"read_only":false}})",
						   R"("max_chunk_bytes":67108864,"read_only":false}})")));
}

// ParsePinningSettings, which reads runs, requests, benchmark requests and selection rows alike.
TEST(pinning_settings_read_a_limit_or_null)
{
	const auto parse = [](const char* limit)
	{
		JsonParser parser;
		Json settings;
		const auto text = fmt::format(R"({{"method":"progressive_pinning","first_chunk_bytes":4194304,)"
									  R"("max_chunk_bytes":67108864{},"read_only":false}})",
									  limit);
		CHECK(!parser.parse(text).get(settings));
		return ParsePinningSettings(settings);
	};
	CHECK(!check::Require(parse(R"(,"max_registered_bytes":null)")).max_registered_bytes);
	CHECK(*check::Require(parse(R"(,"max_registered_bytes":1073741824)")).max_registered_bytes == kGiB);
	CHECK_ERROR_KIND(parse(""), Usage);
	CHECK_ERROR_KIND(parse(R"(,"max_registered_bytes":"none")"), Usage);
	CHECK_ERROR_KIND(parse(R"(,"max_registered_bytes":33554432)"), Usage); // below the 64 MiB chunk
	CHECK_ERROR_KIND(parse(R"(,"max_registered_bytes":2199023255552)"), Usage); // 2 TiB, above kMaxBytes
}

// The line of a file with pinning rows records the limit as the full parse applies it (null for the old default),
// says so (registration_limit), and equals the full parse wherever it is admitted.
TEST(a_pinning_line_records_the_limit_it_applies)
{
	const auto text = check::Require(Reselect(Document(BudgetRuns(), kMachine, Limit::OldDefault)));
	const auto line = text.substr(0, *SelectionLineBytes(text));
	CHECK(line.find(R"(,"registration_limit":"explicit_only"})") != std::string::npos);
	// The rows (the line also echoes the request, which keeps its recorded 1 GiB).
	const auto rows = line.substr(line.find(R"("rows":)"));
	CHECK(rows.find(R"("max_registered_bytes":null)") != std::string::npos);
	CHECK(rows.find(R"("max_registered_bytes":1073741824)") == std::string::npos);
	const auto [admitted, refused] = CheckAdmission(text);
	CHECK_EQ(admitted, std::size_t{14});
	CHECK_EQ(refused, std::size_t{14});
	// Without the statement (pinning rows ending at 1 GiB) the line is not used; the full parse is, so the hook
	// runs the rebuilt rows. A wrong statement is an invalid line.
	auto stale = text;
	const std::string field = R"(,"registration_limit":"explicit_only")";
	stale.erase(stale.find(field), std::size(field));
	std::uint64_t file_bytes = 0;
	const auto from_stale = tuning::ReadSelection(stale, Everything(), file_bytes);
	CHECK(!from_stale && from_stale.error().IsUnsupported());
	auto invalid = text;
	invalid.replace(invalid.find(field), std::size(field), R"(,"registration_limit":"default")");
	const auto from_invalid = tuning::ReadSelection(invalid, Everything(), file_bytes);
	CHECK(!from_invalid && from_invalid.error().IsUsage());
	// Lines without a pinning row are written as before.
	CHECK(check::Require(Reselect(Document(InstantRuns()))).find("registration_limit") == std::string::npos);
}

// With an explicit limit (`cmt tune --max-registered-mib`), above a pinning row's registration budget (1 GiB, also
// a measured size) the copy cannot be pinned, and the row is the best of the other settings measured at that size,
// not the driver.
TEST(above_a_pinning_budget_the_best_unpinned_setting_runs)
{
	const auto text = Limited(BudgetRuns());
	const auto table = check::Require(ParseTuningResult(text, Everything()));
	CHECK_EQ(DtohChoices(table), std::string("0 driver  0 0 0;67108864 progressive_pinning  0 0 0;"
											 "1073741824 progressive_pinning  0 0 0;"
											 "1073741825 staged nt_store 2 1 8388608;"));
	CHECK(table.Find(Dir::DtoH, kGiB)->mode == Mode::ProgressivePinning);
	CHECK(table.Find(Dir::DtoH, kGiB + 1)->mode == Mode::Chunks);
	CHECK(table.Find(Dir::DtoH, 16 * kGiB)->mode == Mode::Chunks);
	CHECK_EQ(table.Find(Dir::DtoH, 16 * kGiB)->settings.workers, 2u);
	// Without pinning here the same size selects the same staged setting, from 1 GiB up: the rule above the
	// budget is that selection.
	LocalCapabilities unpinned = Everything();
	unpinned.progressive_pinning = false;
	CHECK_EQ(DtohChoices(check::Require(ParseTuningResult(text, unpinned))),
			 std::string("0 driver  0 0 0;67108864 driver  0 0 0;1073741824 staged nt_store 2 1 8388608;"));
}

// The other candidates face every rule the winner faces, among themselves: the 2% gain, the CPU and pinned-budget
// filters, and the tie band anchored at their own best median (20: 19 ties at 10%, which against pinning's 25.7
// it would not).
TEST(above_a_pinning_budget_the_selection_rules_apply_to_the_other_settings)
{
	const auto text = Limited(BudgetRuns());
	const auto above = [&](const std::string& document, LocalCapabilities local)
	{
		const auto choices = DtohChoices(check::Require(ParseTuningResult(document, local)));
		return choices.substr(choices.rfind(';', std::size(choices) - 2) + 1);
	};
	CHECK_EQ(above(text, Everything()), std::string("1073741825 staged nt_store 2 1 8388608;"));
	CHECK_EQ(above(WithTie(text, 10), Everything()), std::string("1073741825 staged memcpy 1 1 4194304;"));
	CHECK_EQ(above(WithTie(text, 4), Everything()), std::string("1073741825 staged nt_store 2 1 8388608;"));
	LocalCapabilities scalar = Everything();
	scalar.cpu.mask = 0; // nt_store needs a vector ISA
	CHECK_EQ(above(text, scalar), std::string("1073741825 staged memcpy 1 1 4194304;"));
	LocalCapabilities small = Everything();
	small.max_pinned_bytes = 8 * kMiB; // holds the 4 MiB pool, not the 16 MiB one
	CHECK_EQ(above(text, small), std::string("1073741825 staged memcpy 1 1 4194304;"));
	// Only the setting under the 2% gain is left: the driver, as at any size where no candidate qualifies.
	const auto under_gain = Limited(BudgetRuns(kGiB, {{18.3, {1, 2, kMiB, "memcpy", false}}}));
	CHECK_EQ(above(under_gain, Everything()), std::string("1073741825 driver  0 0 0;"));
}

// Pinning the only candidate compared at the size: above the budget this measurement says nothing, as before the
// rule: the driver, or the INSTANT row where INSTANT selected one. Other candidates compared but none qualified:
// a measured driver row, which replaces the INSTANT row as every measured driver row does.
TEST(above_a_pinning_budget_with_nothing_else_compared)
{
	const auto pinning_only = BudgetRuns(kGiB, {});
	CHECK_EQ(DtohChoices(check::Require(ParseTuningResult(Limited(pinning_only), Everything()))),
			 std::string("0 driver  0 0 0;67108864 progressive_pinning  0 0 0;"
						 "1073741824 progressive_pinning  0 0 0;1073741825 driver  0 0 0;"));
	const StagingSettings instant_choice{1, 2, kMiB, "memcpy", false};
	const auto with_instant = [&](std::vector<CopyRun> runs)
	{
		runs.push_back(Run(Dir::DtoH, 4 * kMiB, Mode::Driver, 10, "instant"));
		runs.push_back(Run(Dir::DtoH, 4 * kMiB, Mode::Chunks, 20, "instant", instant_choice));
		const auto choices = DtohChoices(check::Require(ParseTuningResult(Limited(runs), Everything())));
		return choices.substr(choices.rfind(';', std::size(choices) - 2) + 1);
	};
	CHECK_EQ(with_instant(pinning_only), std::string("1073741825 staged memcpy 1 2 1048576;"));
	CHECK_EQ(with_instant(BudgetRuns(kGiB, {{18.3, {1, 2, kMiB, "memcpy", false}}})),
			 std::string("1073741825 driver  0 0 0;"));
}

// A pinning winner below the budget: the copies between the budget and the next measured size take that size's
// choice without pinning; the next size selects on its own.
TEST(between_a_pinning_budget_and_the_next_size)
{
	const auto runs = BudgetRuns(512 * kMiB, {{20, {2, 1, 8 * kMiB, "memcpy", false}}},
								 {{2 * kGiB, {{21, {4, 1, 4 * kMiB, "memcpy", false}}}}});
	const auto table = check::Require(ParseTuningResult(Limited(runs), Everything()));
	CHECK_EQ(DtohChoices(table), std::string("0 driver  0 0 0;67108864 progressive_pinning  0 0 0;"
											 "536870912 progressive_pinning  0 0 0;"
											 "1073741825 staged memcpy 2 1 8388608;"
											 "2147483648 staged memcpy 4 1 4194304;"));
	CHECK(table.Find(Dir::DtoH, kGiB)->mode == Mode::ProgressivePinning);
	CHECK_EQ(table.Find(Dir::DtoH, 2 * kGiB - 1)->settings.workers, 2u);
}

// The line records the rows above the budget and says so; a line with a pinning row written before the rule (no
// above_pinning_budget, the driver above the budget) is not used, and the hook parses every run instead.
TEST(a_pinning_line_states_its_rows_above_the_budget)
{
	const std::string field = R"(,"above_pinning_budget":"unpinned")";
	const auto original = Limited(BudgetRuns());
	const auto text = check::Require(Reselect(original));
	CHECK(text.find(field + R"(,"registration_limit":"explicit_only"})") < *SelectionLineBytes(text));
	// Admitted with pinning, a vector ISA (this CPU) and 64 MiB of headroom or more, which holds the 16 MiB pool
	// above the budget.
	const auto [admitted, refused] = CheckAdmission(text);
	CHECK_EQ(admitted, std::size_t{4});
	CHECK_EQ(refused, std::size_t{24});
	// Lines without a pinning row are written as before.
	CHECK(check::Require(Reselect(Document(InstantRuns()))).find("above_pinning_budget") ==
		  std::string::npos);
	// As an older cmt wrote it: without the field, the driver above the budget.
	auto stale = text;
	stale.erase(stale.find(field), std::size(field));
	const auto staged = stale.find(R"({"method":"staged","cpu_loop":"nt_store")");
	CHECK(staged < *SelectionLineBytes(stale));
	stale.replace(staged, stale.find('}', staged) + 1 - staged, R"({"method":"driver"})");
	std::uint64_t file_bytes = 0;
	const auto from_stale = tuning::ReadSelection(stale, Everything(), file_bytes);
	CHECK(!from_stale && from_stale.error().IsUnsupported());
	const auto full = check::Require(ParseTuningResult(original, LoadCapabilities()));
	CHECK(LoadsAs(stale, full));
	// The line is not used: with a broken body, nothing loads.
	CHECK(!LoadsAs(stale.substr(0, *SelectionLineBytes(stale)) + "broken body", full));
	auto invalid = text;
	invalid.replace(invalid.find(field), std::size(field), R"(,"above_pinning_budget":"driver")");
	const auto from_invalid = tuning::ReadSelection(invalid, Everything(), file_bytes);
	CHECK(!from_invalid && from_invalid.error().IsUsage());
	std::remove("test_selection.json");
}

// The tie order: a chunk below the copy size, then the smallest pool, fewer workers, the higher median; 0 (and a
// file without the band) keeps the best median.
TEST(tie_band_prefers_small_chunks_then_small_pools)
{
	LocalCapabilities everything;
	everything.cpu.mask = UINT16_MAX;
	const auto choices = [&](const std::string& text)
	{ return Choices(check::Require(ParseTuningResult(text, everything))); };
	const auto rows = [](const char* at_1m, const char* at_64m)
	{
		return fmt::format(
			"0 driver  0 0 0;1048576 staged memcpy {};67108864 staged memcpy {};0 driver  0 0 0;", at_1m,
			at_64m);
	};
	const auto text = Document(TieRuns());
	CHECK_EQ(choices(text), rows("1 1 1048576", "4 4 1048576"));
	CHECK_EQ(choices(WithTie(text, 0)), choices(text));
	CHECK_EQ(choices(WithTie(text, 2)), rows("2 2 524288", "2 2 1048576"));
	CHECK_EQ(choices(WithTie(text, 3)), rows("2 2 524288", "1 1 1048576"));
	CHECK(!ParseTuningResult(WithTie(text, 101), everything));
}

// With 8 MiB of headroom the line's own rows fit, but its 16 MiB reference does not: here the best median is
// 4x1x1M, whose band admits 1x1x1M, so the line must be refused (tie_reference_rows).
TEST(tie_line_equals_the_full_parse_where_admitted)
{
	const auto [admitted, refused] =
		CheckAdmission(check::Require(Reselect(WithTie(Document(TieRuns()), 2))));
	CHECK_EQ(admitted, std::size_t{16});
	CHECK_EQ(refused, std::size_t{12});
}

// Both paths check the recorded placement with one function: a range larger than this machine fails both.
TEST(placement_is_checked_on_both_paths)
{
	auto text = check::Require(Reselect(Document(InstantRuns())));
	for (auto at = text.find("=0,2"); at != std::string::npos; at = text.find("=0,2"))
	{
		text.replace(at, 4, "=0-99999999");
	}
	std::uint64_t file_bytes = 0;
	CHECK(!tuning::ReadSelection(text, {}, file_bytes) && !ParseTuningResult(text));
}

// tuning::Load reads only line 1 when it is admitted, and every run otherwise, with the same table; a garbled
// line 1 falls back to every run, or, where the document itself is invalid, to an error (the driver).
TEST(load_reads_the_line_or_every_run)
{
	const auto old = Document(InstantRuns());
	const auto text = check::Require(Reselect(old));
	const auto bytes = *SelectionLineBytes(text);
	const auto value = text.substr(0, bytes - 2); // without ",\n"
	const auto rest = text.substr(bytes);
	const auto full = check::Require(ParseTuningResult(old, LoadCapabilities()));
	// The line is admitted here only with room for its 128 MiB pool.
	const bool admitted = LoadCapabilities().max_pinned_bytes >= 128 * kMiB;
	CHECK(LoadsAs(text, full));
	CHECK(LoadsAs(old, full));
	CHECK(LoadsAs(value + ",\n" + "broken body", full) == admitted); // only line 1 was read (size: a warning)
	for (const auto& garbled : {"{\n" + text.substr(1), value + ",\r\n" + rest, value + "\n," + rest,
								value + "," + rest})
	{
		CHECK(!SelectionLineBytes(garbled) && LoadsAs(garbled, full));
	}
	auto invalid = text;
	invalid.replace(invalid.find("test CPU"), 8, "test \xff CPU");
	check::Require(WriteTextFile("test_selection.json", invalid));
	CHECK(!tuning::Load("test_selection.json")); // not UTF-8: the line and the document are refused
	// A line of exactly kMaxSelectionLineBytes is read; one byte more is not, and the hook parses every run.
	const auto padded = [&](std::size_t line_bytes)
	{ return value + std::string(line_bytes - std::size(value) - 2, ' ') + ",\n"; };
	CHECK(LoadsAs(padded(kMaxSelectionLineBytes) + "broken body", full) == admitted);
	CHECK(LoadsAs(padded(kMaxSelectionLineBytes + 1) + rest, full));
	CHECK(!LoadsAs(padded(kMaxSelectionLineBytes + 1) + "broken body", full));
	CHECK(!tuning::Load("test_selection_missing.json"));
	std::remove("test_selection.json");
}

// Reselect is what `cmt check` compares with: a row edited after tuning and a changed size differ from it, and
// it restores the line. Mixed INSTANT and measured runs, and a line the hook would not read, get no line.
TEST(reselect_recomputes_the_line)
{
	const auto text = check::Require(Reselect(Document(InstantRuns())));
	CHECK(check::Require(Reselect(text)) == text);
	auto edited = text;
	const auto at = edited.find("\"slots_per_worker\":4");
	CHECK(at < *SelectionLineBytes(text));
	edited[at + std::size("\"slots_per_worker\":4") - 2] = '3';
	CHECK(check::Require(Reselect(edited)) == text);
	CHECK(check::Require(Reselect(text + " ")) != text + " ");
	auto runs = InstantRuns();
	runs.push_back(MeasuredRuns().front());
	const auto big_machine = std::string(R"({"host":{"cpu":{"model":")") + std::string(kMaxSelectionLineBytes, 'x') +
							 R"(","arch":"test"}},"gpus":[{"name":"test GPU"}]})";
	for (const auto& document : {Document(runs), Document(InstantRuns(), big_machine)})
	{
		const auto written = check::Require(Reselect(document));
		CHECK(!SelectionLineBytes(written) && check::Require(Reselect(written)) == written);
	}
}

// `cmt check`: 0 and OK for a line that matches, 6 for one that differs, and --write fixes it.
int CheckCommand(const char* cmt)
{
	const auto text = check::Require(Reselect(Document(InstantRuns())));
	const auto run_check = [&](const std::string& file, const char* flags)
	{
		check::Require(WriteTextFile("test_selection_check.json", file));
		const int status =
			std::system(fmt::format("{} check {} test_selection_check.json > /dev/null 2>&1", cmt, flags).c_str());
		return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
	};
	CHECK_EQ(run_check(text, ""), 0);
	CHECK_EQ(run_check(text + " ", ""), 6);
	CHECK_EQ(run_check(Document(InstantRuns()), ""), 6);
	CHECK_EQ(run_check(Document(InstantRuns()), "--write"), 0);
	CHECK(check::Require(ReadTextFile("test_selection_check.json")) == text);
	CHECK_EQ(run_check(check::Require(Reselect(WithTie(Document(TieRuns()), 2))), ""), 0);
	// A pinning line as written before the rows above the budget were the unpinned choice, or before limits were
	// explicit only (the old default's 1 GiB in its rows, no statement): each differs, and --write fixes it.
	for (Limit limit : {Limit::Explicit1G, Limit::OldDefault})
	{
		const auto pinning = check::Require(Reselect(Document(BudgetRuns(), kMachine, limit)));
		CHECK_EQ(run_check(pinning, ""), 0);
		auto stale = pinning;
		const std::string field = limit == Limit::Explicit1G ? R"(,"above_pinning_budget":"unpinned")"
															 : R"(,"registration_limit":"explicit_only")";
		stale.erase(stale.find(field), std::size(field));
		CHECK_EQ(run_check(stale, ""), 6);
		CHECK_EQ(run_check(stale, "--write"), 0);
		CHECK(check::Require(ReadTextFile("test_selection_check.json")) == pinning);
	}
	// `cmt tune --preview` (no GPU needed): without --max-registered-mib pinning is measured at every size from its
	// first chunk up; with it, only up to the limit.
	const auto preview = [&](const char* limit)
	{
		const auto out = std::string("test_selection_preview.txt");
		const int status = std::system(
			fmt::format("{} tune --out test_selection_preview --search full_with_kernel --preview {} "
						"--tuning_params=\"sizes=1M,1G,4G,16G threads=1 buffers=1 buffer_size=1M loops=memcpy\" "
						"> {} 2>&1",
						cmt, limit, out)
				.c_str());
		CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
		const auto printed = check::Require(ReadTextFile(out));
		std::remove(out.c_str());
		return printed;
	};
	CHECK(preview("").find("; sizes 1G,4G,16G.") != std::string::npos);
	CHECK(preview("--max-registered-mib 1024").find("; sizes 1G.") != std::string::npos);
	std::remove("test_selection_check.json");
	return check::exit_code();
}

// The GPU tests run `cmt tune` for real. They skip (77) only when `cmt bench --info` sees no CUDA device; with one,
// a tune that fails, a broken probe included, fails the test.
bool NoGpu(const char* cmt)
{
	const int status = std::system(fmt::format("{} bench --info > /dev/null 2>&1", cmt).c_str());
	return WIFEXITED(status) && WEXITSTATUS(status) == 77;
}

// A finished FULL resumed with tie=2% re-selects from its saved runs: no child measures (the attempts stay), line
// 1 changes as the band predicts, and `cmt check` agrees. The measured rates are rewritten first (every sample of a
// setting equal) so the band, not noise, decides: 2x2 (20) wins without it; 1x2 (19.8) and 2x1 (19.7) tie with
// the smallest pool, and 1x2 has fewer workers.
int ResumeCommand(const char* cmt)
{
	if (NoGpu(cmt))
	{
		std::printf("SKIP: no CUDA device visible\n");
		return 77;
	}
	const std::string out = "test_selection_resume";
	const auto run = [&](const std::string& command)
	{
		const int status = std::system(fmt::format("{} {} >> {}.log 2>&1", cmt, command, out).c_str());
		return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
	};
	const auto tune = [&](const char* tie, const char* resume)
	{
		return run(fmt::format(
			"tune --out {} --search full --direction cpu-to-gpu --progress_bar off --tuning_params="
			"\"sizes=64M threads=1,2 buffers=1,2 buffer_size=1M loops=memcpy{}\" {}",
			out, tie, resume));
	};
	std::filesystem::remove_all(out);
	std::filesystem::remove(out + ".log");
	if (tune("", "") != 0)
	{
		std::printf("FAIL: cmt tune failed; see %s.log\n", out.c_str());
		return 1;
	}
	const std::map<std::pair<std::string, std::string>, double> rates{
		{{"1", "1"}, 15}, {{"1", "2"}, 19.8}, {{"2", "1"}, 19.7}, {{"2", "2"}, 20}};
	const std::regex layout(R"re("pipeline_workers":(\d+),"slots_per_worker":(\d+))re");
	auto text = check::Require(ReadTextFile(out + "/tune.json"));
	for (auto at = text.find("{\"direction\":", text.find("\"runs\":[")); at != std::string::npos;
		 at = text.find("{\"direction\":", at + 1))
	{
		auto end = at;
		for (int depth = 0; end == at || depth > 0; end++)
		{
			depth += text[end] == '{' ? 1 : text[end] == '}' ? -1 : 0;
		}
		auto record = text.substr(at, end - at);
		std::smatch match;
		const double gbps = std::regex_search(record, match, layout) ? rates.at({match[1], match[2]}) : 10;
		CHECK(record.find("\"copies_per_sample\":1,") != std::string::npos); // elapsed_seconds == seconds
		record = std::regex_replace(record, std::regex(R"re("(elapsed_)?seconds":[^,}]+)re"),
									fmt::format("\"$1seconds\":{}", 64.0 * kMiB / (gbps * 1e9)));
		record =
			std::regex_replace(record, std::regex(R"re("gbps":[^,}]+)re"), fmt::format("\"gbps\":{}", gbps));
		text.replace(at, end - at, record);
	}
	check::Require(WriteTextFile(out + "/tune.json", text));
	CHECK_EQ(run(fmt::format("check --write {}/tune.json", out)), 0);
	LocalCapabilities everything;
	everything.cpu.mask = UINT16_MAX;
	// The selected rows and the number of recorded child processes.
	const auto state = [&]
	{
		const auto saved = check::Require(ReadTextFile(out + "/tune.json"));
		std::uint64_t file_bytes = 0;
		JsonParser parser;
		simdjson::dom::array attempts;
		CHECK(!parser.parse(saved)["attempts"].get(attempts));
		return std::pair{Choices(check::Require(tuning::ReadSelection(saved, everything, file_bytes))),
						 std::size(attempts)};
	};
	const auto rows = [](const char* shape)
	{ return fmt::format("0 driver  0 0 0;67108864 staged memcpy {} 1048576;0 driver  0 0 0;", shape); };
	const auto [before, measured] = state();
	CHECK_EQ(before, rows("2 2"));
	CHECK_EQ(tune(" tie=2%", "--resume"), 0);
	const auto [after, remeasured] = state();
	CHECK_EQ(after, rows("1 2"));
	CHECK_EQ(remeasured, measured);
	CHECK_EQ(run(fmt::format("check {}/tune.json", out)), 0);
	std::filesystem::remove_all(out);
	return check::exit_code();
}

// INSTANT measures, in each direction, the driver and its two settings at the threads it estimated, each at all six
// sizes, even when both directions estimate the same setting; --resume then measures nothing.
int InstantCommand(const char* cmt)
{
	if (NoGpu(cmt))
	{
		std::printf("SKIP: no CUDA device visible\n");
		return 77;
	}
	const std::string out = "test_selection_instant";
	// cmt tune hands this process's CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER to its children itself.
	const auto tune = [&](const char* resume)
	{
		const int status =
			std::system(fmt::format("{} tune --out {} --search instant --progress_bar off {} >> {}.log 2>&1",
									cmt, out, resume, out)
							.c_str());
		return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
	};
	std::filesystem::remove_all(out);
	std::filesystem::remove(out + ".log");
	// Fresh tuning launches through the same hook/config path as cmt run and
	// returns the application's status. No extra GPU search is needed for this check.
	const int first = tune("-- sh -c 'test -n \"$CMT_CONFIG\" && test -n \"$LD_PRELOAD\" || exit 99; exit 17'");
	const auto saved = ReadTextFile(out + "/tune.json");
	if (first != 17 || !saved)
	{
		std::printf("FAIL: cmt tune --search instant exited %d; see %s.log\n", first, out.c_str());
		return 1;
	}
	const auto output = check::Require(ReadTextFile(out + ".log"));
	CHECK(output.find("Saved tuned file:") != std::string::npos);
	CHECK(output.find("Loaded tuned file:") != std::string::npos);
	for (const char* detail : {"Grid (", "GB/s:", "process 1/", "Kernel:", "derived intervals"})
	{
		CHECK(output.find(detail) == std::string::npos);
	}
	JsonParser parser;
	Json document;
	simdjson::dom::array runs, attempts;
	CHECK(!parser.parse(*saved).get(document) && !document["runs"].get(runs) && !document["attempts"].get(attempts));
	// The children's placement is part of the identity (a resume that would place them elsewhere is refused),
	// and the record shows both CPU sources (nvml_cpus, when not null, is the one that applies).
	std::string_view placement;
	CHECK(!document.at_pointer("/identity/child_placement").get(placement));
	CHECK(placement == "unpinned" || placement.rfind("cpus=", 0) == 0); // the decision only, never its reason
	std::string_view note;
	CHECK(!document.at_pointer("/invocations/0/child_placement").get(note) && note.rfind(placement, 0) == 0);
	Json sysfs_cpus;
	CHECK(!document.at_pointer("/source_machine/gpus/0/cpu_affinity/sysfs_cpus").get(sysfs_cpus));
	// Nothing in the file names this host, its GPU or its directories (share.h): no hostname, UUID, PCI address or
	// absolute path. (Other kinds the share scan knows have legitimate look-alikes here: executed_loop's '@'.)
	auto hostname = ReadTextFile("/proc/sys/kernel/hostname");
	if (hostname && !hostname->empty())
	{
		hostname->erase(hostname->find_last_not_of("\n") + 1);
		CHECKF(saved->find("\"" + *hostname + "\"") == std::string::npos, "tune.json names the host %s",
			   hostname->c_str());
	}
	for (const auto& finding : share::FindSensitive(document))
	{
		const std::string_view kind(finding.kind);
		// /identity's keys are JSON pointers into /source_machine ("/driver", "/gpus/0/name"): the scan reads each
		// key as an absolute path. Accept exactly those keys; any value, or a key that points nowhere, still fails.
		Json pointed;
		if (finding.pointer == "/identity" && kind == "an absolute or home path" &&
			!document.at_pointer("/source_machine" + finding.value).get(pointed))
		{
			continue;
		}
		CHECKF(kind == "an e-mail address or user@host" || kind == "an IPv4 address" ||
				   kind == "an IPv6 or MAC address",
			   "tune.json %s is %s: %s", finding.pointer.c_str(), finding.kind, finding.value.c_str());
	}
	for (const char* direction : {"htod", "dtoh"})
	{
		std::int64_t threads = 0;
		CHECK(!document.at_pointer(fmt::format("/estimate/threads/{}", direction)).get(threads));
		std::map<std::string, std::size_t> sizes; // per setting
		for (Json run : runs)
		{
			std::string_view name;
			Json settings;
			std::int64_t workers = 0;
			CHECK(!run["direction"].get(name) && !run["settings"].get(settings));
			if (name == direction)
			{
				sizes[simdjson::to_string(settings)]++;
				CHECK(settings["pipeline_workers"].get(workers) || workers == threads);
			}
		}
		CHECK_EQ(std::size(sizes), std::size_t{3});
		for (const auto& [settings, count] : sizes)
		{
			CHECK_EQ(count, std::size_t{6});
		}
	}
	const auto measured = std::size(attempts);
	CHECK_EQ(tune("--resume"), 0);
	CHECK(!parser.parse(check::Require(ReadTextFile(out + "/tune.json")))["attempts"].get(attempts));
	CHECK_EQ(std::size(attempts), measured);
	std::filesystem::remove_all(out);
	return check::exit_code();
}

int main(int argc, char** argv)
{
	if (argc == 1)
	{
		return check::run_all();
	}
	if (argc == 3 && std::string_view(argv[1]) == "--cmt")
	{
		return CheckCommand(argv[2]);
	}
	if (argc == 3 && std::string_view(argv[1]) == "--resume")
	{
		return ResumeCommand(argv[2]);
	}
	if (argc == 3 && std::string_view(argv[1]) == "--instant")
	{
		return InstantCommand(argv[2]);
	}
	for (int i = 1; i < argc; i++)
	{
		const auto original = check::Require(ReadTextFile(argv[i]));
		const auto text = check::Require(Reselect(original));
		const auto line = SelectionLineBytes(text);
		CHECK(line.has_value());
		if (!line)
		{
			continue;
		}
		const auto full = check::Require(ParseTuningResult(original, LoadCapabilities()));
		std::uint64_t file_bytes = 0;
		const auto admitted = tuning::ReadSelection(text, LoadCapabilities(), file_bytes);
		// A refused line cannot load with a broken body; then the converted file must load through every run.
		const bool same = LoadsAs(admitted ? text.substr(0, *line) + "broken body" : text, full);
		CHECK(same);
		std::printf("%s: %zu rows, line %zu bytes, file %zu -> %zu bytes, %s, %s\n", argv[i], std::size(full.rows),
					*line, std::size(original), std::size(text), admitted ? "line read" : admitted.error().Message(),
					same ? "same table" : "DIFFERENT");
	}
	std::remove("test_selection.json");
	return check::exit_code();
}
