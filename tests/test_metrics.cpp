#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>
#include "common/aliases.h"
#include "common/error.h"
#include "measure/metrics.h"
#include "tests/check.h"

using namespace cmt;
using V = std::vector<std::size_t>;

TEST(parse_size_suffixes)
{
	CHECK_EQ(ParseSize("4096").value_or(0), std::size_t{4096});
	CHECK_EQ(ParseSize("4K").value_or(0), std::size_t{4096});
	CHECK_EQ(ParseSize("4k").value_or(0), std::size_t{4096});
	CHECK_EQ(ParseSize("4KiB").value_or(0), std::size_t{4096});
	CHECK_EQ(ParseSize("4KB").value_or(0), std::size_t{4096});
	CHECK_EQ(ParseSize("1M").value_or(0), std::size_t{1} << 20);
	CHECK_EQ(ParseSize("16G").value_or(0), std::size_t{16} << 30);
	CHECK_EQ(ParseSize(" 2M ").value_or(0), std::size_t{2} << 20);
	CHECK_EQ(ParseSize("100B").value_or(0), std::size_t{100});
}

TEST(parse_size_rejects)
{
	CHECK_ERROR_KIND(ParseSize(""), Usage);
	CHECK_ERROR_KIND(ParseSize("K"), Usage);
	CHECK_ERROR_KIND(ParseSize("0"), Usage);
	CHECK_ERROR_KIND(ParseSize("4T"), Usage);
	CHECK_ERROR_KIND(ParseSize("4KX"), Usage);
	CHECK_ERROR_KIND(ParseSize("-4K"), Usage);
	CHECK_ERROR_KIND(ParseSize("99999999999999999999"), Usage);
	CHECK_ERROR_KIND(ParseSize("17179869184G"), Usage); // 2^34 GiB = 2^64 bytes: overflows size_t
	CHECK_ERROR_KIND(ParseSize("1025G"), Usage);		// above the 1 TiB limit
	CHECK_EQ(ParseSize("1024G").value_or(0), std::size_t{1} << 40);
}

TEST(ranges_are_inclusive_powers_of_two)
{
	CHECK_EQ(ParseSizes("4K..32K").value_or(V{}), (V{4096, 8192, 16384, 32768}));
	CHECK_EQ(ParseSizes("1M..1M").value_or(V{}), (V{1u << 20}));
	CHECK_EQ(ParseSizes("64K,1M..4M").value_or(V{}), (V{65536, 1u << 20, 2u << 20, 4u << 20}));
	CHECK_EQ(ParseSizes("4K..16G").value_or(V{}).size(), std::size_t{23}); // 2^12 .. 2^34
	CHECK_ERROR_KIND(ParseSizes("3K..12K"), Usage);						   // not powers of two
	CHECK_ERROR_KIND(ParseSizes("1M..4K"), Usage);						   // reversed
	CHECK_ERROR_KIND(ParseSizes("1M,,2M"), Usage);						   // empty item
	CHECK_ERROR_KIND(ParseSizes(""), Usage);
}

TEST(human_size_uses_largest_exact_unit)
{
	CHECK_EQ(HumanSize(4096), std::string("4 KiB"));
	CHECK_EQ(HumanSize(1536 * 1024), std::string("1536 KiB"));
	CHECK_EQ(HumanSize(std::size_t{16} << 30), std::string("16 GiB"));
	CHECK_EQ(HumanSize(100), std::string("100 B"));
}

TEST(reps_make_a_sample_at_least_min_bytes)
{
	CHECK_EQ(RepsFor(4096, 64u << 20), 16384u);
	CHECK_EQ(RepsFor(64u << 20, 64u << 20), 1u);
	CHECK_EQ(RepsFor(std::size_t{1} << 30, 64u << 20), 1u); // never zero
	CHECK_EQ(RepsFor(3u << 20, 64u << 20), 21u);			// floor: 21 x 3 MiB = 63 MiB
	CHECK_PANICS(RepsFor(0, 1));
	CHECK_EQ(RepsFor(16, std::size_t{1} << 40), 1u << 20); // capped at kMaxReps
}

TEST(capacity_coverage_thresholds_mixed_and_unknown_host)
{
	const std::size_t margin = 256u << 20;
	const auto mixed = ComputeSizeCoverage({100, 101, 201}, margin + 400, margin + 230, 2, 2);
	CHECK_EQ(mixed.device_usable, std::size_t{400});
	CHECK_EQ(*mixed.host_usable, std::size_t{207});
	CHECK_EQ(mixed.Eligible(), (V{100, 101}));
	CHECK(mixed.sizes[2].limit == CapacityLimit::Device);
	const auto host = ComputeSizeCoverage({103, 104}, margin + 1000, margin + 230, 1, 2);
	CHECK_EQ(host.Eligible(), (V{103}));
	CHECK(host.sizes[1].limit == CapacityLimit::Host);
	const auto unknown = ComputeSizeCoverage({400, 401}, margin + 400, std::nullopt, 1, 100);
	CHECK(unknown.Eligible().empty());
	CHECK(!unknown.host_available);
	CHECK(!unknown.host_usable);
	CHECK(SizeCoverageJson(unknown).find("\"available_bytes\":null") != std::string::npos);
}

TEST(capacity_coverage_empty_budgets_all_skipped_and_multiplication_overflow)
{
	const auto small = ComputeSizeCoverage({1, 2}, 0, 0, 1, 1);
	CHECK(small.Eligible().empty());
	CHECK(SizeCoverageJson(small).find("all_capacity_skipped") != std::string::npos);
	const auto maximum = std::numeric_limits<std::size_t>::max();
	const auto large = ComputeSizeCoverage({maximum / 2, maximum}, maximum, std::nullopt, 3, 0);
	CHECK(large.Eligible().empty()); // multiplying either requested size by three would overflow
	const auto no_buffers = ComputeSizeCoverage({maximum}, 0, 0, 0, 0);
	CHECK_EQ(no_buffers.Eligible(), (V{maximum}));
	CHECK(SizeCoverageJson(no_buffers).find("\"reason\":null") != std::string::npos);
}

TEST(capacity_skips_use_google_records_without_measurement_csv_rows)
{
	char name[] = "capacity-host-test";
	char* argv[] = {name, nullptr};
	int argc = 1;
	benchmark::Initialize(&argc, argv);
	benchmark::ClearRegisteredBenchmarks();
	const auto coverage = ComputeSizeCoverage({4096, 8192}, 0, std::nullopt, 1, 1);
	benchmark::AddCustomContext("cmt_size_coverage_schema", "1");
	benchmark::AddCustomContext("cmt_size_coverage", SizeCoverageJson(coverage));
	int callbacks = 0, formatted_measurements = 0;
	for (const auto& size : coverage.sizes)
	{
		benchmark::RegisterBenchmark(("capacity-host-test/" + std::to_string(size.bytes)).c_str(),
									 [&](benchmark::State& state)
									 {
										 ++callbacks;
										 state.SkipWithMessage("capacity_skipped: device_capacity");
									 })
			->UseManualTime()
			->Iterations(1)
			->Repetitions(1);
	}
	std::ostringstream json_output, csv_output;
	benchmark::JSONReporter json_reporter;
	json_reporter.SetOutputStream(&json_output);
	cmt::bench::SchemaReporter csv_reporter(
		[&](const std::string&, const std::vector<SampleOut>&, Verified)
		{
			++formatted_measurements;
			return std::vector<MeasurementRow>{};
		});
	csv_reporter.SetOutputStream(&csv_output);

	// Exercise both real reporters from the same Google callbacks, without a
	// file-output flag or any GPU path/runner/allocation.
	struct Reporters : benchmark::BenchmarkReporter
	{
		benchmark::BenchmarkReporter &a, &b;

		Reporters(benchmark::BenchmarkReporter& first, benchmark::BenchmarkReporter& second)
			: a(first), b(second)
		{
		}

		bool ReportContext(const Context& context) override
		{
			return a.ReportContext(context) && b.ReportContext(context);
		}

		void ReportRuns(const std::vector<Run>& runs) override
		{
			a.ReportRuns(runs);
			b.ReportRuns(runs);
		}

		void Finalize() override
		{
			a.Finalize();
			b.Finalize();
		}
	} reporters(json_reporter, csv_reporter);

	const auto registered = benchmark::RunSpecifiedBenchmarks(&reporters, "^capacity-host-test/");
	benchmark::ClearRegisteredBenchmarks();
	benchmark::Shutdown();
	CHECK_EQ(registered, std::size_t{2});
	CHECK_EQ(callbacks, 2);
	CHECK_EQ(formatted_measurements, 0);
	CHECK_EQ(csv_output.str(), CsvHeader() + "\n");
	JsonParser parser;
	simdjson::dom::element document;
	const auto parsed = parser.parse(json_output.str()).get(document);
	CHECK(parsed == simdjson::SUCCESS);
	if (parsed != simdjson::SUCCESS)
	{
		return;
	}
	simdjson::dom::array records;
	const auto array_error = document["benchmarks"].get_array().get(records);
	CHECK(array_error == simdjson::SUCCESS);
	if (array_error != simdjson::SUCCESS)
	{
		return;
	}
	CHECK_EQ(records.size(), std::size_t{2});
	for (const auto record : records)
	{
		bool skipped = false;
		std::string_view message;
		CHECK(record["skipped"].get_bool().get(skipped) == simdjson::SUCCESS);
		CHECK(skipped);
		CHECK(record["skip_message"].get_string().get(message) == simdjson::SUCCESS);
		CHECK(message == "capacity_skipped: device_capacity");
		CHECK(record["bytes_per_second"].error() == simdjson::NO_SUCH_FIELD);
	}
	std::string_view coverage_json;
	const auto coverage_error = document["context"]["cmt_size_coverage"].get_string().get(coverage_json);
	CHECK(coverage_error == simdjson::SUCCESS);
	if (coverage_error != simdjson::SUCCESS)
	{
		return;
	}
	JsonParser coverage_parser;
	simdjson::dom::element raw;
	const auto coverage_parsed = coverage_parser.parse(coverage_json.data(), coverage_json.size()).get(raw);
	CHECK(coverage_parsed == simdjson::SUCCESS);
	if (coverage_parsed != simdjson::SUCCESS)
	{
		return;
	}
	simdjson::dom::array sizes;
	const auto sizes_error = raw["requested_sizes"].get_array().get(sizes);
	CHECK(sizes_error == simdjson::SUCCESS);
	if (sizes_error != simdjson::SUCCESS)
	{
		return;
	}
	V requested;
	for (const auto size : sizes)
	{
		std::uint64_t bytes = 0;
		CHECK(size.get_uint64().get(bytes) == simdjson::SUCCESS);
		requested.push_back(bytes);
	}
	CHECK_EQ(requested, (V{4096, 8192}));
	std::string_view outcome;
	CHECK(raw["outcome"].get_string().get(outcome) == simdjson::SUCCESS);
	CHECK(outcome == "all_capacity_skipped");
	simdjson::dom::element available;
	const auto available_error = raw["host"]["available_bytes"].get(available);
	CHECK(available_error == simdjson::SUCCESS);
	if (available_error == simdjson::SUCCESS)
	{
		CHECK(available.is_null());
	}
}

using namespace cmt;

TEST(empty_input_gives_zero_summary)
{
	const Summary s = Summarize({});
	CHECK_EQ(s.n, std::size_t{0});
	CHECK_EQ(s.best, 0.0);
}

TEST(best_min_and_upper_median)
{
	const Summary s = Summarize({3.0, 1.0, 2.0});
	CHECK_EQ(s.best, 3.0);
	CHECK_EQ(s.min, 1.0);
	CHECK_EQ(s.median, 2.0);
	// even count: the upper median (measure/metrics.h)
	CHECK_EQ(Summarize({4.0, 1.0, 3.0, 2.0}).median, 3.0);
	CHECK_EQ(Summarize({7.0}).median, 7.0);
}

TEST(gbps_is_decimal)
{
	CHECK_EQ(Gbps(1e9, 1.0), 1.0);
	CHECK_EQ(Gbps(double(1u << 30), 1.0), 1.073741824);
	CHECK_EQ(Gbps(1e9, 0.0), 0.0); // never divide by zero
}

using namespace cmt;

TEST(fill_then_check_is_clean)
{
	for (std::size_t bytes : {8u, 64u, 4096u, (1u << 20) + 8u, 3u << 20})
	{
		std::vector<std::uint64_t> v(bytes / 8);
		PatternFill(std::data(v), bytes, 42);
		CHECK_EQ(PatternCheck(std::data(v), bytes, 42), std::size_t{0});
	}
}

TEST(check_counts_every_wrong_word)
{
	const std::size_t bytes = 1u << 20;
	std::vector<std::uint64_t> v(bytes / 8);
	PatternFill(std::data(v), bytes, 7);
	v[0] ^= 1;				 // first word
	v[std::size(v) / 2] = 0; // middle (a different thread's range)
	v.back() = v.back() + 1; // last word
	CHECK_EQ(PatternCheck(std::data(v), bytes, 7), std::size_t{3});
}

TEST(other_seed_never_verifies)
{
	// a stale buffer from another size or direction must fail, word for word
	const std::size_t bytes = 64u << 10;
	std::vector<std::uint64_t> v(bytes / 8);
	PatternFill(std::data(v), bytes, 1);
	CHECK_EQ(PatternCheck(std::data(v), bytes, 2), bytes / 8);
}

TEST(words_are_distinct_so_misplaced_chunks_fail)
{
	// a chunk copied to the wrong offset: the check sees every moved word
	const std::size_t bytes = 8192;
	std::vector<std::uint64_t> v(bytes / 8);
	PatternFill(std::data(v), bytes, 3);
	std::set<std::uint64_t> seen(std::begin(v), std::end(v));
	CHECK_EQ(std::size(seen), std::size(v));
	std::memmove(std::data(v) + 64, std::data(v), 64 * 8); // 512 B chunk lands 512 B too far
	CHECK_EQ(PatternCheck(std::data(v), bytes, 3), std::size_t{64});
}

namespace {

/// Everything `f` writes to stderr, through a temporary file (a pipe could fill before it is read).
template <class F>
std::string CapturedStderr(F&& f)
{
	char path[] = "/tmp/cmt-pattern-report-XXXXXX";
	const int file = mkstemp(path);
	CHECK(file >= 0);
	std::fflush(stderr);
	const int saved = dup(STDERR_FILENO);
	CHECK(saved >= 0 && dup2(file, STDERR_FILENO) >= 0);
	f();
	std::fflush(stderr);
	CHECK(dup2(saved, STDERR_FILENO) >= 0);
	close(saved);
	close(file);
	auto text = ReadTextFile(path);
	unlink(path);
	CHECK(static_cast<bool>(text));
	return text ? *text : std::string{};
}

/// From each "pattern check: word J ... it is this seed's word K" line, J -> K; a line that names no index
/// in range (or does not parse) maps J to SIZE_MAX. Lines in the order printed.
std::vector<std::pair<std::size_t, std::size_t>> ReportedWords(const std::string& report)
{
	std::vector<std::pair<std::size_t, std::size_t>> words;
	std::istringstream lines(report);
	for (std::string line; std::getline(lines, line);)
	{
		std::size_t word = 0;
		if (std::sscanf(line.c_str(), "pattern check: word %zu ", &word) != 1)
		{
			continue;
		}
		constexpr std::string_view kNames = "it is this seed's word ";
		const auto at = line.find(kNames);
		std::size_t names = std::numeric_limits<std::size_t>::max();
		if (at != std::string::npos)
		{
			std::sscanf(line.c_str() + at + std::size(kNames), "%zu", &names);
		}
		words.emplace_back(word, names);
	}
	return words;
}

} // namespace

// A misplaced word (index J holding index K's word) must be reported as K's: that is what tells a stale
// or misplaced chunk from a flipped bit. A wrong inverse constant in PatternCheck names another index.
TEST(the_report_names_whose_word_a_wrong_word_is)
{
	constexpr std::uint64_t seed = 0x5eed;
	// 64 KiB: one thread checks it all, so the report is the first wrong words in index order.
	{
		const std::size_t bytes = 64u << 10;
		std::vector<std::uint64_t> v(bytes / 8);
		PatternFill(std::data(v), bytes, seed);
		v[100] = PatternWord(7, seed);
		v[5000] = PatternWord(4000, seed);
		v[8191] = PatternWord(0, seed);
		std::size_t bad = 0;
		const auto report = CapturedStderr([&] { bad = PatternCheck(std::data(v), bytes, seed); });
		CHECK_EQ(bad, std::size_t{3});
		V flat; // J0, K0, J1, K1, ...: check::show prints vectors of numbers
		for (const auto& [word, names] : ReportedWords(report))
		{
			flat.insert(std::end(flat), {word, names});
		}
		CHECK_EQ(flat, (V{100, 7, 5000, 4000, 8191, 0}));
	}
	// 4 MiB (several threads) with 64 planted words: exactly 8 reported, sorted, each a planted word named
	// by its source; the count is still all 64.
	{
		const std::size_t bytes = 4u << 20;
		const std::size_t words = bytes / 8;
		std::vector<std::uint64_t> v(words);
		PatternFill(std::data(v), bytes, seed);
		HashMap<std::size_t, std::size_t> planted;
		for (std::size_t n = 0; n < 64; n++)
		{
			const std::size_t at = (n * 8191 + 3) % words;
			const std::size_t from = (at + 1 + n * 977) % words; // never `at` itself
			v[at] = PatternWord(from, seed);
			planted[at] = from;
		}
		std::size_t bad = 0;
		const auto report = CapturedStderr([&] { bad = PatternCheck(std::data(v), bytes, seed); });
		CHECK_EQ(bad, std::size_t{64});
		const auto reported = ReportedWords(report);
		CHECK_EQ(std::size(reported), std::size_t{8});
		for (std::size_t i = 0; i < std::size(reported); i++)
		{
			const auto [word, names] = reported[i];
			CHECK(planted.count(word) == 1);
			CHECK_EQ(names, planted[word]);
			CHECK(i == 0 || reported[i - 1].first < word);
		}
	}
}

TEST(sizes_must_be_whole_words)
{
	std::vector<std::uint64_t> v(2);
	CHECK_PANICS(PatternFill(std::data(v), 12, 0));
	CHECK_PANICS(PatternCheck(std::data(v), 12, 0));
}

TEST(thread_ownership_moves_and_joins_exactly_once)
{
	std::atomic<unsigned> completed{0};
	cmt::Thread original;
	auto started = original.Start([&] { completed.fetch_add(1); });
	CHECK(started);
	if (!started)
	{
		return;
	}
	cmt::Thread moved(std::move(original));
	CHECK(!original.Joinable());
	CHECK(moved.Joinable());
	CHECK(moved.Join());
	CHECK(!moved.Joinable());
	CHECK(moved.Join());
	CHECK_EQ(completed.load(), 1u);
	{
		cmt::Thread destructor_join;
		CHECK(destructor_join.Start([&] { completed.fetch_add(1); }));
	}
	CHECK_EQ(completed.load(), 2u);
}

using namespace cmt;

namespace {
std::size_t fields(const std::string& s)
{
	std::size_t n = 1;
	for (char c : s)
	{
		n += c == ',';
	}
	return n;
}

std::string field(const std::string& s, std::size_t k)
{
	std::size_t a = 0;
	for (std::size_t i = 0; i < k; i++)
	{
		a = s.find(',', a) + 1;
	}
	return s.substr(a, s.find(',', a) - a);
}
} // namespace

TEST(row_has_as_many_fields_as_header)
{
	MeasurementRow r;
	CHECK_EQ(fields(CsvRow(r)), fields(CsvHeader()));
}

TEST(text_fields_preserve_bytes_with_csv_quoting)
{
	MeasurementRow r;
	r.gpu = "NVIDIA GeForce RTX 3090, rev A";
	r.tag = "a,\"b\"\nc";
	const std::string s = CsvRow(r);
	const std::string prefix = fmt::format("{},\"a,\"\"b\"\"\nc\",\"NVIDIA GeForce RTX 3090, rev A\",", kCsvSchema);
	CHECK_EQ(s.substr(0, prefix.size()), prefix);
}

TEST(schema_first_and_values_in_place)
{
	MeasurementRow r;
	r.path = "pinned_ce";
	r.size_bytes = 4096;
	r.gbps_best = 26.8934;
	r.verified = Verified::Ok;
	r.copy_s = 0.00125;
	r.huge_kib = 2048;
	const std::string s = CsvRow(r), h = CsvHeader();
	CHECK_EQ(field(s, 0), std::to_string(kCsvSchema));
	// look values up by header name, as a reader does
	for (std::size_t k = 0; k < fields(h); k++)
	{
		const std::string name = field(h, k);
		if (name == "path")
		{
			CHECK_EQ(field(s, k), std::string("pinned_ce"));
		}
		if (name == "size_bytes")
		{
			CHECK_EQ(field(s, k), std::string("4096"));
		}
		if (name == "gbps_best")
		{
			CHECK_EQ(field(s, k), std::string("26.893"));
		}
		if (name == "verified")
		{
			CHECK_EQ(field(s, k), std::string("ok"));
		}
		if (name == "dma")
		{
			CHECK_EQ(field(s, k), std::string("1"));
		}
		if (name == "copy_s")
		{
			CHECK_EQ(field(s, k), std::string("0.00125"));
		}
		if (name == "huge_kib")
		{
			CHECK_EQ(field(s, k), std::string("2048"));
		}
		if (name == "phase")
		{
			CHECK_EQ(field(s, k), std::string("-"));
		}
	}
}

TEST(schema_3_only_appends)
{
	// readers of schema 1 and 2 find every old column at the same position
	const std::string h = CsvHeader();
	CHECK_EQ(kCsvSchema, 3);
	CHECK_EQ(fields(h), std::size_t{29});
	CHECK_EQ(field(h, 23), std::string("verified"));   // last schema-1 column, unmoved
	CHECK_EQ(field(h, 27), std::string("teardown_s")); // last schema-2 column, unmoved
	CHECK_EQ(field(h, 28), std::string("phase"));
}

TEST(unknown_huge_pages_print_empty_and_seconds_keep_precision)
{
	MeasurementRow r;
	r.setup_s = 1.234567891e-5; // a 4 KiB register: %.6f would print 0.000012
	const std::string s = CsvRow(r), h = CsvHeader();
	for (std::size_t k = 0; k < fields(h); k++)
	{
		if (field(h, k) == "huge_kib")
		{
			CHECK_EQ(field(s, k), std::string(""));
		}
		if (field(h, k) == "setup_s")
		{
			CHECK_EQ(field(s, k), std::string("1.23456789e-05"));
		}
	}
}

// fmt's "{:.3f}" / "{:.9g}" must write the same bytes as printf's "%.3f" / "%.9g", so CSV files written by
// either compare, including zero, tiny and huge values and rounding edges.
TEST(fmt_writes_the_same_numbers_as_printf)
{
	const double vs[] = {0.0,		 1.0,  0.0005,			 0.0015,		 2.5, 26.8975,	 26.8965,
						 123456.789, 1e-9, 3.25e-7,			 1.234567891e-3, 0.1, 1.0 / 3.0, 12345678912.0,
						 1e300,		 -0.0, 0.000123456789012};
	for (double v : vs)
	{
		char a[512],
			b[512]; // %.3f of 1e300 is 305 characters: a 40-byte buffer would cut it
		std::snprintf(a, sizeof a, "%.3f", v);
		std::snprintf(b, sizeof b, "%.9g", v);
		CHECK_EQ(fmt::format("{:.3f}", v), std::string(a));
		CHECK_EQ(fmt::format("{:.9g}", v), std::string(b));
	}
}

namespace {
using Reporter = cmt::bench::SchemaReporter;

Reporter::Run sample(double seconds)
{
	Reporter::Run run;
	run.run_name.function_name = "test/htod/1024";
	run.real_accumulated_time = seconds;
	run.counters = {{"verified", 1}, {"has_phases", 1},	  {"has_directions", 1}, {"setup_s", 0.1},
					{"copy_s", 0.2}, {"teardown_s", 0.3}, {"htod_gbps", 17},	 {"dtoh_gbps", 19}};
	return run;
}
} // namespace

TEST(lifecycle_phase_counters_round_trip_without_changing_total_time)
{
	auto run = sample(2.8);
	run.counters["has_lifecycle"] = 1;
	for (std::size_t i = 0; i < cmt::kLifecycleCounters.size(); i++)
	{
		run.counters[cmt::kLifecycleCounters[i]] = static_cast<double>(i + 1) / 10;
	}
	bool seen = false;
	Reporter reporter(
		[&](const std::string&, const std::vector<cmt::SampleOut>& samples, cmt::Verified)
		{
			CHECK_EQ(samples.size(), 1u);
			CHECK(samples[0].lifecycle.has_value());
			double sum = 0;
			for (std::size_t i = 0; i < cmt::kLifecycleCounters.size(); i++)
			{
				CHECK_EQ((*samples[0].lifecycle)[i], static_cast<double>(i + 1) / 10);
				sum += (*samples[0].lifecycle)[i];
			}
			CHECK(std::fabs(sum - samples[0].seconds) < 1e-12);
			seen = true;
			return std::vector<cmt::MeasurementRow>{};
		});
	reporter.ReportRuns({run});
	reporter.Finalize();
	CHECK(seen);
}

TEST(raw_repetitions_preserve_phases_and_ignore_google_aggregates)
{
	std::size_t calls = 0;
	Reporter reporter(
		[&](const std::string& name, const std::vector<cmt::SampleOut>& samples, cmt::Verified verified)
		{
			calls++;
			CHECK_EQ(name, std::string("test/htod/1024"));
			CHECK_EQ(std::size(samples), 2u);
			CHECK_EQ(samples[0].seconds, 1.0);
			CHECK_EQ(samples[1].seconds, 2.0);
			CHECK_EQ(samples[0].phases->setup, 0.1);
			CHECK_EQ(samples[0].phases->copy, 0.2);
			CHECK_EQ(samples[0].phases->teardown, 0.3);
			CHECK_EQ(samples[0].directions->htod_gbps, 17.0);
			CHECK_EQ(samples[0].directions->dtoh_gbps, 19.0);
			CHECK(verified == cmt::Verified::Ok);
			cmt::MeasurementRow row;
			row.size_bytes = 1000000000;
			row.reps = 1;
			cmt::SummarizeResult(row, samples, 1);
			CHECK_EQ(row.samples, 2u);
			CHECK_EQ(row.gbps_best, 1.0);
			CHECK_EQ(row.gbps_med, 1.0); // legacy upper median of rates, not inverse mean time
			CHECK_EQ(row.gbps_min, 0.5);
			return std::vector<cmt::MeasurementRow>{row};
		});
	std::ostringstream output;
	reporter.SetOutputStream(&output);
	CHECK(reporter.ReportContext(Reporter::Context{}));
	reporter.ReportRuns({sample(1)});
	auto aggregate = sample(1.5);
	aggregate.run_type = Reporter::Run::RT_Aggregate;
	reporter.ReportRuns({sample(2), aggregate});
	reporter.Finalize();
	CHECK_EQ(calls, 1u);
	CHECK(output.str().find(cmt::CsvHeader()) == 0);
}

TEST(a_failed_repetition_invalidates_the_whole_group)
{
	std::size_t calls = 0;
	Reporter reporter(
		[&](const auto&, const auto&, auto)
		{
			calls++;
			return std::vector<cmt::MeasurementRow>{};
		});
	auto failed = sample(0);
	failed.skipped = benchmark::internal::SkippedWithError;
	reporter.ReportRuns({sample(1), failed});
	reporter.Finalize();
	CHECK_EQ(calls, 0u);
}

TEST(unverified_measurements_never_acquire_an_ok_label)
{
	bool called = false;
	Reporter reporter(
		[&](const auto&, const auto&, cmt::Verified verified)
		{
			called = true;
			CHECK(verified == cmt::Verified::Skip);
			return std::vector<cmt::MeasurementRow>{};
		});
	auto skipped = sample(1);
	skipped.counters["verified"] = 0;
	reporter.ReportRuns({sample(1), skipped});
	reporter.Finalize();
	CHECK(called);
}

namespace {

/// The `cache_policy` gpu_copy saves for a flushed, complete 4 MiB pageable measurement whose
/// samples each time `copies` copies; empty when the file could not be written or read back.
std::string SavedCachePolicy(std::uint32_t copies)
{
	PathSpec pageable;
	pageable.name = "pageable";
	Measurement measurement;
	measurement.config.path = &pageable;
	measurement.config.flush = true;
	measurement.bytes = 4u << 20;
	measurement.copies_per_sample = copies;
	measurement.status = "complete";
	measurement.reason.clear();
	for (std::size_t i = 0; i < 2; i++)
	{
		SampleOut sample;
		sample.seconds = 1e-3 * copies;
		measurement.samples.push_back(sample);
	}
	char path[] = "/tmp/cmt-cache-policy-XXXXXX";
	const int descriptor = mkstemp(path);
	if (descriptor < 0)
	{
		return {};
	}
	close(descriptor);
	const bool written = static_cast<bool>(WriteMeasurements(path, "{}", {measurement}));
	const auto text = ReadTextFile(path);
	unlink(path);
	if (!written || !text)
	{
		return {};
	}
	JsonParser parser;
	Json document;
	std::string_view policy;
	if (parser.parse(*text).get(document) || document["runs"].at(0)["cache_policy"].get(policy))
	{
		return {};
	}
	return std::string(policy);
}

} // namespace

// gpu_copy flushes once per sample. With 16 copies of one 4 MiB buffer per sample (its default
// --min-bytes 64M) only the first copy starts cold, so the file must not claim the sample was.
TEST(only_a_one_copy_sample_is_labelled_evicted_before_sample)
{
	if (!FlushSupported())
	{
		CHECK_EQ(SavedCachePolicy(1), std::string("warm")); // nothing is flushed here
		return;
	}
	CHECK_EQ(SavedCachePolicy(1), std::string("evicted_before_sample"));
	CHECK_EQ(SavedCachePolicy(16), std::string("evicted_before_first_copy"));
}

TEST_MAIN()
