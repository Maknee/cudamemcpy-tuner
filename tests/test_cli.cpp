#include <cstdio>
#include <string>
#include <vector>

#include "tests/check.h"
#include "cli.h"
#include "common/error.h"
#include "measure/metrics.h"
#include "runner.h"

using namespace cmt;
using A = std::vector<std::string>;

namespace {
bool is(const Config& c, const char* name)
{
	return c.path && c.path->name == name;
}

// Capture only this synchronous parser call; restore stderr before assertions
// about its diagnostic. No pipe capacity or child-process lifetime is involved.
template <class F>
std::string CaptureStderr(F action)
{
	FILE* output = std::tmpfile();
	CHECK(output != nullptr);
	if (!output)
	{
		return {};
	}
	std::fflush(stderr);
	const int saved = dup(STDERR_FILENO);
	CHECK(saved >= 0);
	if (saved < 0)
	{
		std::fclose(output);
		return {};
	}
	const int redirected = dup2(fileno(output), STDERR_FILENO);
	if (redirected < 0)
	{
		close(saved);
		std::fclose(output);
		CHECK(redirected >= 0);
		return {};
	}
	action();
	std::fflush(stderr);
	const int restored = dup2(saved, STDERR_FILENO);
	close(saved);
	CHECK(restored >= 0);
	if (restored < 0)
	{
		std::fclose(output);
		std::abort(); // Do not continue the suite with stderr redirected.
	}
	std::rewind(output);
	std::string text;
	char buffer[256];
	while (const auto bytes = std::fread(buffer, 1, sizeof(buffer), output))
	{
		text.append(buffer, bytes);
	}
	CHECK(!std::ferror(output));
	std::fclose(output);
	return text;
}
} // namespace

TEST(defaults)
{
	const Options o = check::Require(ParseArgs({}));
	CHECK(is(o.cfg, "pinned_ce"));
	CHECK(o.cfg.dir == Dir::HtoD);
	CHECK(o.cfg.flush);
	CHECK(o.cfg.verify);
	CHECK_EQ(o.cfg.samples, 3);
	CHECK_EQ(std::size(o.cfg.sizes), std::size_t{19}); // 4K..1G
	CHECK(!o.header);
	CHECK(o.cfg.worker_allowed_cpus.empty());
}

TEST(every_flag_reaches_the_config)
{
	const std::string exact_tag = " comma, quote\" slash\\ tab\t newline\n UTF-8: \xCE\xBB ";
	const Options o = check::Require(ParseArgs(
		A{"--path",		  "staged",		"--dir",		 "htod",	  "--sizes",  "1M..4M",		 "--pages",
		  "thp",		  "--fill",		"nt_store@sse2", "--threads", "1",		  "--slots",	 "8",
		  "--slot-bytes", "4M",			"--wc",			 "--samples", "5",		  "--min-bytes", "128M",
		  "--budget-s",	  "1.5",		"--cpu",		 "0",		  "--device", "1",			 "--tag",
		  exact_tag,	  "--no-flush", "--no-verify",	 "--header"}));
	const Config& c = o.cfg;
	CHECK(is(c, "staged"));
	CHECK(c.pages == Pages::Thp);
	CHECK(c.cpu_copy && LoopNameText(c.cpu_loop) == "nt_store@sse2");
	CHECK_EQ(c.threads, 1);
	CHECK_EQ(c.slots, 8);
	CHECK_EQ(c.slot_bytes, std::size_t{4u << 20});
	CHECK(c.wc_slots);
	CHECK_EQ(c.samples, 5);
	CHECK_EQ(c.min_bytes, std::size_t{128u << 20});
	CHECK_EQ(c.budget_s, 1.5);
	CHECK_EQ(c.cpu, 0);
	CHECK_EQ(c.device, 1);
	CHECK_EQ(c.tag, exact_tag);

	// Serialize the parsed tag without running or inventing a timed measurement.
	char temporary[] = "/tmp/cmt-tag-roundtrip-XXXXXX";
	const int descriptor = mkstemp(temporary);
	CHECK(descriptor >= 0);
	if (descriptor >= 0)
	{
		close(descriptor);

		struct RemoveOutput
		{
			const char* path;

			~RemoveOutput()
			{
				unlink(path);
			}
		} remove_output{temporary};

		Measurement measurement;
		measurement.config = c;
		measurement.bytes = c.sizes.front();
		measurement.status = "unsupported";
		measurement.reason = "serialization fixture; no measurement";
		const auto written = WriteMeasurements(temporary, "{}", {measurement});
		CHECK(written);
		if (written)
		{
			const auto text = ReadTextFile(temporary);
			CHECK(text);
			if (text)
			{
				JsonParser parser;
				Json document;
				const auto parsed = parser.parse(*text).get(document);
				CHECK(parsed == simdjson::SUCCESS);
				if (parsed == simdjson::SUCCESS)
				{
					std::string_view saved_tag;
					const auto read = document["runs"].at(0)["tag"].get(saved_tag);
					CHECK(read == simdjson::SUCCESS);
					if (read == simdjson::SUCCESS)
					{
						CHECK_EQ(std::string(saved_tag), exact_tag);
					}
				}
			}
		}
	}
	CHECK(!c.flush);
	CHECK(!c.verify);
	CHECK(o.header);
	CHECK_EQ(std::size(c.sizes), std::size_t{3});
	CHECK(c.set_knobs.Has(Knob::Fill) && c.set_knobs.Has(Knob::Wc) && !c.set_knobs.Has(Knob::Unpin));
}

TEST(no_dma_and_pin_copy_unpin)
{
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "staged", "--no-dma"}), Usage);
	CHECK(check::Require(ParseArgs(A{"--path", "staged"})).cfg.dma);
	CHECK(is(check::Require(ParseArgs(A{"--path", "pin_copy_unpin", "--dir", "dtoh", "--pages", "thp"})).cfg,
			 "pin_copy_unpin"));
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "pinned_ce", "--no-dma"}),
					 Usage); // CPU-only staging is not supported
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "staged", "--no-dma", "--fill", "none"}),
					 Usage); // measures nothing
}

TEST(flags_the_path_ignores_are_refused)
{
	// Accepting these would silently measure something other than what was asked for.
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "pinned_ce", "--fill", "nt_store@avx2"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "pageable", "--threads", "2"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "staged", "--unpin", "eager"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "register", "--dir", "dtoh"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "bidir_ce", "--dir", "htod"}), Usage);
	const auto failure = ParseArgs(A{"--path", "pageable", "--wc"});
	CHECK(!failure);
	if (!failure)
	{
		CHECK(failure.error().IsUsage());
		CHECK_EQ(failure.error().GetCode(), 0);
		CHECK(std::string(failure.error().Message())
				  .find("specified option has no effect on the selected --path") != std::string::npos);
	}
}

TEST(a_repeated_flag_takes_its_last_value)
{
	const Options o =
		check::Require(ParseArgs(A{"--sizes", "4K", "--samples", "2", "--sizes", "1M", "--samples", "5"}));
	CHECK_EQ(o.cfg.sizes, (std::vector<std::size_t>{1u << 20}));
	CHECK_EQ(o.cfg.samples, 5);
	const auto overwritten = check::Require(
		ParseArgs({"--sizes", "BAD", "--sizes", "1M", "--min-bytes", "1Q", "--min-bytes", "64M"}));
	CHECK_EQ(overwritten.cfg.sizes, (std::vector<std::size_t>{1u << 20}));
	CHECK_EQ(overwritten.cfg.min_bytes, std::size_t{64u << 20});
	const auto tuned = check::Require(ParseTuneArgs({"--out", "unused", "--tuning_params", "sizes=BAD",
													 "--tuning_params", "sizes=1M", "--min-bytes", "1Q",
													 "--min-bytes", "64M"}));
	CHECK(tuned.grid.sizes == (std::vector<std::uint64_t>{1u << 20}));
	CHECK_EQ(tuned.min_bytes, std::size_t{64u << 20});
	CHECK_EQ(check::Require(ParseArgs({"--path", "staged", "--slot-bytes", "BAD", "--slot-bytes", "1M"}))
				 .cfg.slot_bytes,
			 std::size_t{1u << 20});
	CHECK_EQ(check::Require(ParseArgs({"--path", "pin_pipeline", "--pin-step", "BAD", "--pin-step", "32M"}))
				 .cfg.pin_step,
			 std::size_t{32u << 20});

	CHECK_ERROR_KIND(ParseArgs({"--sizes", "1M", "--sizes", "4K,12XB"}), Usage);
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "unused", "--min-bytes", "64M", "--min-bytes", "1Q"}), Usage);
}

TEST(equals_form)
{
	const Options o =
		check::Require(ParseArgs(A{"--path=pin_pipeline", "--unpin=eager", "--pin-step=8M", "--sizes=64M"}));
	CHECK(is(o.cfg, "pin_pipeline"));
	CHECK(o.cfg.unpin == Unpin::Eager);
	CHECK_EQ(o.cfg.pin_step, std::size_t{8u << 20});
}

TEST(bad_input_is_a_usage_error)
{
	const auto sizes_diagnostic =
		CaptureStderr([] { CHECK_ERROR_KIND(ParseArgs({"--sizes", "4K,12XB"}), Usage); });
	for (const char* text : {"--sizes", "12XB", "byte 5 (zero-based)"})
	{
		CHECK(sizes_diagnostic.find(text) != std::string::npos);
	}
	const auto min_bytes_diagnostic = CaptureStderr(
		[]
		{
			const auto malformed = ParseTuneArgs({"--out", "unused", "--min-bytes", "1Q"});
			CHECK(!malformed);
			if (!malformed)
			{
				CHECK(malformed.error().IsUsage());
				CHECK_EQ(malformed.error().GetCode(), 0);
			}
		});
	for (const char* text : {"--min-bytes", "1Q", "byte 1 (zero-based)"})
	{
		CHECK(min_bytes_diagnostic.find(text) != std::string::npos);
	}

	CHECK_ERROR_KIND(ParseArgs(A{"--path", "zerocopy"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--frobnicate", "1"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--samples"}), Usage); // missing value
	CHECK_ERROR_KIND(ParseArgs(A{"--samples", "three"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--samples", "0"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"stray"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--sizes", "100"}), Usage);	// not a multiple of 16
	CHECK_ERROR_KIND(ParseArgs(A{"--budget-s", "nan"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--budget-s", "inf"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--device", "-1"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--cpu", "-2"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--cpu", "100000"}), Usage); // outside any affinity mask
}

TEST(parser_reports_integer_overflow_and_empty_required_values)
{
	CHECK_ERROR_KIND(ParseArgs({"--threads", "9999999999999999999999999"}), Usage);
	CHECK_ERROR_KIND(ParseArgs({"--path", "staged", "--cpu", "2147483647", "--threads", "2147483647"}),
					 Usage);
	CHECK_ERROR_KIND(ParseTraceArgs({"--out=", "--", "true"}), Usage);
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "x", "--tuning_params=samples=1.5"}), Usage);
}

TEST(run_requires_a_config_and_application_separator)
{
	CHECK_ERROR_KIND(ParseRunArgs({"--config", "tune.json", "python"}), Usage);
	CHECK_ERROR_KIND(ParseRunArgs({"--", "python"}), Usage);
	auto options = check::Require(ParseRunArgs({"--config", "tune.json", "--", "python", "--help"}));
	CHECK_EQ(options.command, (A{"python", "--help"}));
	CHECK(options.overrides.empty());
}

// --remote_config takes an optional address: alone, `=URL` or ` URL`, and never a word after `--`.
TEST(remote_config_takes_an_optional_address)
{
	const auto remote = [](A args) { return check::Require(ParseRunArgs(std::move(args))).remote; };
	CHECK(remote({"--remote_config", "--", "app"}) == std::optional<std::string>("https://cudamemcpy.com"));
	CHECK(remote({"--remote_config=http://localhost:8080", "--", "app"}) ==
		  std::optional<std::string>("http://localhost:8080"));
	CHECK(remote({"--remote_config", "https://mirror.example", "--", "app"}) ==
		  std::optional<std::string>("https://mirror.example"));
	CHECK(remote({"--config", "t.json", "--", "app"}) == std::nullopt);
	const auto options = check::Require(ParseRunArgs({"--remote_config", "--", "app", "--remote_config=x"}));
	CHECK(options.remote == std::optional<std::string>("https://cudamemcpy.com") &&
		  options.command == (A{"app", "--remote_config=x"}));
	CHECK_ERROR_KIND(ParseRunArgs({"--config", "t.json", "--remote_config", "--", "app"}), Usage);
	CHECK_ERROR_KIND(ParseRunArgs({"--remote_config", "a", "b", "--", "app"}), Usage);
}

// `cmt run --numa` reaches the hook as the numa= run override (tests/test_tune_table_e2e.cpp loads it).
TEST(run_numa_switch_is_a_run_override)
{
	for (const char* value : {"on", "off"})
	{
		const auto options =
			check::Require(ParseRunArgs({"--config", "t.json", "--numa", value, "--", "app"}));
		CHECK_EQ(options.overrides, (A{std::string("numa=") + value}));
	}
	CHECK_ERROR_KIND(ParseRunArgs({"--config", "t.json", "--numa", "maybe", "--", "app"}), Usage);
	CHECK_ERROR_KIND(ParseRunOverrides("numa=1"), Usage);
	CHECK(check::Require(ParseRunOverrides("numa=off")).numa == false);
	CHECK(ParseRunArgs({"--help-advanced"})->help.find("--numa") != std::string::npos);
	CHECK(check::Require(ParseTuneArgs({"--help-advanced"})).help.find("--numa") != std::string::npos);
}

// `cmt run --cuda_host_alloc_hook on|off|private`: an explicit value, unset when absent (the tune decides then;
// tests/test_share.cpp checks what reaches the application).
TEST(run_cuda_host_alloc_hook_takes_on_off_or_private)
{
	const auto hook = [](A args) { return check::Require(ParseRunArgs(std::move(args))).cuda_host_alloc_hook; };
	CHECK(!hook({"--config", "t.json", "--", "app"}));
	CHECK(hook({"--config", "t.json", "--cuda_host_alloc_hook", "off", "--", "app"}) == "off");
	CHECK(hook({"--config", "t.json", "--cuda_host_alloc_hook", "on", "--", "app"}) == "on");
	CHECK(hook({"--config", "t.json", "--cuda_host_alloc_hook=on", "--", "app"}) == "on");
	CHECK(hook({"--config", "t.json", "--cuda_host_alloc_hook", "private", "--", "app"}) == "private");
	CHECK(hook({"--remote_config", "--cuda_host_alloc_hook", "on", "--", "app"}) == "on");
	const auto options =
		check::Require(ParseRunArgs({"--config", "t.json", "--", "app", "--cuda_host_alloc_hook", "on"}));
	CHECK(!options.cuda_host_alloc_hook && options.overrides.empty());
	CHECK_ERROR_KIND(ParseRunArgs({"--config", "t.json", "--cuda_host_alloc_hook", "--", "app"}), Usage);
	CHECK_ERROR_KIND(ParseRunArgs({"--config", "t.json", "--cuda_host_alloc_hook", "1", "--", "app"}), Usage);
	CHECK_ERROR_KIND(ParseRunArgs({"--config", "t.json", "--cuda_host_alloc_hook", "maybe", "--", "app"}), Usage);
	const auto help = ParseRunArgs({"--help-advanced"})->help;
	CHECK(help.find("--cuda-host-alloc-hook") != std::string::npos && help.find("fork()") != std::string::npos);
}

// `cmt tune --host_alloc on|off`: on by default (quick and full time pinned allocation; tune::PlanHostAllocation).
TEST(tune_host_alloc_takes_on_or_off)
{
	CHECK(check::Require(ParseTuneArgs({"--out", "o"})).host_alloc);
	CHECK(check::Require(ParseTuneArgs({"--out", "o", "--host_alloc", "on"})).host_alloc);
	CHECK(!check::Require(ParseTuneArgs({"--out", "o", "--host_alloc", "off"})).host_alloc);
	CHECK(!check::Require(ParseTuneArgs({"--out", "o", "--host_alloc=off", "--search", "full"})).host_alloc);
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "o", "--host_alloc", "private"}), Usage);
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "o", "--host_alloc"}), Usage);
	const auto help = check::Require(ParseTuneArgs({"--help-advanced"})).help;
	CHECK(help.find("--host-alloc") != std::string::npos && help.find("fork()") != std::string::npos);
}

TEST(tuning_retains_last_scalar_values_and_has_no_env_option)
{
	const auto tune = check::Require(ParseTuneArgs({"--out", "first", "--out", "last"}));
	CHECK_EQ(tune.out_dir, std::string("last"));
	CHECK(tune.env.empty());
	// The children's environment is fixed (tune::kInheritedVariables); there is no --env.
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "x", "--env", "STAGING_THREADS=1"}), Usage);
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "x", "--env", "CUDA_VISIBLE_DEVICES=0"}), Usage);
}

TEST(nonsense_combinations_are_refused)
{
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "staged", "--threads", "0"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "pin_pipeline", "--pin-step", "1000"}), Usage);
}

TEST(pin_pipeline_accepts_both_directions)
{
	for (const auto* pages : {"4k", "thp"})
	{
		CHECK(check::Require(ParseArgs(A{"--path", "pin_pipeline", "--dir", "dtoh", "--pages", pages}))
				  .cfg.dir == Dir::DtoH);
		CHECK(check::Require(ParseArgs(A{"--path", "pin_pipeline", "--dir", "htod", "--pages", pages}))
				  .cfg.dir == Dir::HtoD);
	}
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "pin_pipeline", "--dir", "dtoh", "--pin-step", "1000"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "pin_pipeline", "--dir", "dtoh", "--samples", "0"}), Usage);
	CHECK_ERROR_KIND(ParseArgs(A{"--path", "pin_pipeline", "--dir", "invalid"}), Usage);
}

TEST(staged_dtoh_accepts_cached_and_wc_slots)
{
	const auto cached = check::Require(ParseArgs(A{"--path", "staged", "--dir", "dtoh"})).cfg;
	const auto wc = check::Require(ParseArgs(A{"--path", "staged", "--dir", "dtoh", "--wc"})).cfg;
	CHECK(cached.dir == Dir::DtoH);
	CHECK(!cached.wc_slots);
	CHECK(wc.dir == Dir::DtoH);
	CHECK(wc.wc_slots);
}

TEST(help_info_and_header_skip_validation)
{
	CHECK(check::Require(ParseArgs(A{"--help"})).help);
	CHECK(check::Require(ParseArgs(A{"--help", "--sizes", "garbage"}))
			  .help); // T6: --help works whatever else is on the line
	CHECK(check::Require(ParseArgs(A{"--info", "--path", "staged", "--dir", "dtoh"})).info);
	const auto info = check::Require(ParseArgs(A{"--info", "--device", "3"}));
	CHECK(info.info);
	CHECK(info.cfg.device == 3);
	CHECK_ERROR_KIND(ParseArgs(A{"--info", "--device", "-1"}), Usage);
	CHECK(check::Require(ParseArgs(A{"--header"})).HeaderOnly());
	CHECK(!check::Require(ParseArgs(A{"--header", "--sizes", "4K"})).HeaderOnly());
	const std::string u = UsageText();
	for (const PathSpec& p : BenchmarkPaths())
	{
		CHECK(u.find(p.name) != std::string::npos); // generated from the built-in table
	}
}

TEST(names_round_trip)
{
	for (Alloc a : {Alloc::Register, Alloc::HostAlloc, Alloc::HostAllocWc})
	{
		CHECK(check::Require(ParseAlloc(AllocName(a))) == a);
	}
	for (Unpin u : {Unpin::End, Unpin::Eager, Unpin::Prereg})
	{
		CHECK(check::Require(ParseUnpin(UnpinName(u))) == u);
	}
	CHECK(check::Require(ParseDir("h2d")) == Dir::HtoD);
	CHECK(check::Require(ParseDir("d2h")) == Dir::DtoH);
	CHECK_EQ(std::size(BenchmarkPaths()), std::size_t{10});
	for (const PathSpec& p : BenchmarkPaths())
	{
		CHECK(check::Require(PathNamed(p.name)) == &p);
	}
}

TEST(page_policy_cannot_be_silently_ignored_by_cuda_allocations)
{
	for (const char* alloc : {"hostalloc", "hostalloc_wc"})
	{
		for (const char* pages : {"4k", "thp", "hugetlb"})
		{
			CHECK_ERROR_KIND(ParseArgs(A{"--path", "pinned_ce", "--alloc", alloc, "--pages", pages}), Usage);
		}
	}
	CHECK(check::Require(ParseArgs(A{"--path", "pinned_ce", "--alloc", "hostalloc_wc"})).cfg.alloc ==
		  Alloc::HostAllocWc);
	CHECK(check::Require(ParseArgs(A{"--path", "staged", "--wc", "--pages", "thp"})).cfg.wc_slots);
}

TEST(tune_can_forward_an_application_without_parsing_its_flags)
{
	const auto a = check::Require(ParseTuneArgs({"--", "python", "app.py", "--help", "--out", "app-results"}));
	CHECK(a.help.empty());
	CHECK_EQ(a.out_dir, check::Require(CacheDirectoryPath()));
	CHECK_EQ(a.command, (A{"python", "app.py", "--help", "--out", "app-results"}));
	CHECK(!a.measurement_options && !a.verbose);
	const auto verbose = check::Require(ParseTuneArgs({"--verbose", "--debug", "--", "app"}));
	CHECK(verbose.verbose && verbose.debug && !verbose.measurement_options);
	CHECK(check::Require(ParseRunArgs({"--config", "t.json", "--debug", "--", "app"})).debug);
	CHECK(check::Require(ParseRunArgs({"--config", "t.json", "--verbose", "--", "app"})).verbose);
	const auto custom = check::Require(ParseTuneArgs({"--out", "t", "--search", "instant", "--", "app"}));
	CHECK_EQ(custom.out_dir, std::string("t"));
	CHECK(custom.measurement_options && custom.search == cmt::tune::Search::Instant);
	CHECK(check::Require(ParseTuneArgs({"--out", "t"})).command.empty());
	CHECK_EQ(check::Require(ParseTuneArgs({})).out_dir, a.out_dir);
	CHECK_ERROR_KIND(ParseTuneArgs({"--"}), Usage);
	CHECK_ERROR_KIND(ParseTuneArgs({"--", ""}), Usage);
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "t", "python", "app.py"}), Usage);
	CHECK(!check::Require(ParseTuneArgs({"--help", "--", "app"})).help.empty());
}

TEST(tune_sampling_defaults_and_explicit_controls)
{
	const A base{"--hook", "/fixture/hook.so", "--out", "/fixture/tune"};
	const auto defaults = check::Require(ParseTuneArgs(base));
	CHECK_EQ(defaults.grid.samples, 2u);
	CHECK_EQ(defaults.grid.rounds, 1u);
	CHECK_EQ(defaults.max_processes, std::size_t{4096});
	CHECK_EQ(defaults.min_bytes, std::size_t{1});
	auto explicit_args = base;
	explicit_args.insert(explicit_args.end(), {"--tuning_params", "samples=5 rounds=2", "--min-bytes", "128MiB",
											   "--max-processes", "512"});
	const auto specified = check::Require(ParseTuneArgs(explicit_args));
	CHECK_EQ(specified.grid.samples, 5u);
	CHECK_EQ(specified.grid.rounds, 2u);
	CHECK_EQ(specified.tuning_params, std::string("samples=5 rounds=2"));
	CHECK_EQ(specified.max_processes, std::size_t{512});
	CHECK_EQ(specified.min_bytes, std::size_t{128u << 20});
}

TEST(tune_sampling_rejects_invalid_or_missing_values)
{
	const A base{"--hook", "/fixture/hook.so", "--out", "/fixture/tune"};
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "x", "--max-processes", "0"}), Usage);
	for (const auto& value : {"", "0", "-1", "1.5", "bad", "99999999999999999999"})
	{
		auto arguments = base;
		arguments.insert(arguments.end(), {"--tuning_params", std::string("samples=") + value});
		CHECK_ERROR_KIND(ParseTuneArgs(arguments), Usage);
	}
	for (const auto& value : {"", "0", "-1", "bad", "1T", "99999999999999999999G"})
	{
		auto arguments = base;
		arguments.insert(arguments.end(), {"--min-bytes", value});
		CHECK_ERROR_KIND(ParseTuneArgs(arguments), Usage);
	}
	for (const auto& flag : {"--tuning_params", "--min-bytes"})
	{
		auto arguments = base;
		arguments.push_back(flag);
		CHECK_ERROR_KIND(ParseTuneArgs(arguments), Usage);
	}
	// Replaced by --tuning_params.
	for (const auto& flag : {"--samples", "--rounds", "--sizes", "--max-workers", "--candidates", "--chunk-sizes"})
	{
		auto arguments = base;
		arguments.insert(arguments.end(), {flag, "1"});
		CHECK_ERROR_KIND(ParseTuneArgs(arguments), Usage);
	}
}

TEST(tune_help_shows_each_search_default_grid_without_predicting_runtime)
{
	const auto help = check::Require(ParseTuneArgs({"--help-advanced"})).help;
	for (const char* text :
		 {"--tuning-params", "sizes=64K,512K,1M,4M,16M,64M,512M,1G", "threads=1,2,4,8,16", "buffer_size=1M,2M,4M,8M,16M",
		  "full_with_kernel", "--tuning-params=\"threads=2,4 buffer_size=1M,4M\""})
	{
		CHECK(help.find(text) != std::string::npos);
	}
	CHECK(help.find("hours") == std::string::npos);
	CHECK(help.find("minutes") == std::string::npos);
}

// full_with_kernel times THP caller buffers without being told; asked for small pages it refuses and says why.
TEST(full_with_kernel_times_thp_caller_pages)
{
	CHECK(check::Require(ParseTuneArgs({"--out", "x", "--search", "full_with_kernel"})).pages == Pages::Thp);
	CHECK(check::Require(ParseTuneArgs({"--out", "x", "--search", "full_with_kernel", "--pages", "thp"}))
			  .pages == Pages::Thp);
	CHECK(check::Require(ParseTuneArgs({"--out", "x", "--search", "full"})).pages == Pages::Small);
	const auto small = ParseTuneArgs({"--out", "x", "--search", "full_with_kernel", "--pages", "small"});
	CHECK_ERROR_KIND(small, Usage);
	CHECK(!small && std::string(small.error().Message()).find("THP-backed") != std::string::npos);
}

TEST(tune_normal_command_leaves_tool_paths_for_main)
{
	const auto a = check::Require(ParseTuneArgs({"--out", "/fixture/quick"}));
	CHECK(a.hook.empty());
	CHECK(a.benchmark_exe.empty());
	CHECK(a.search == cmt::tune::Search::Quick);
	CHECK(a.grid.sizes ==
		  (std::vector<std::uint64_t>{64u << 10, 512u << 10, 1u << 20, 4u << 20, 16u << 20, 64u << 20, 512u << 20,
									  1u << 30}));
	CHECK(!a.preview && !a.dry_run);
	CHECK(check::Require(ParseTuneArgs({})).command.empty());
}

TEST(tune_uses_canonical_direction_option)
{
	const auto args = check::Require(ParseTuneArgs({"--out", "/fixture/one", "--direction", "cpu-to-gpu"}));
	CHECK(args.dirs == (std::vector<cmt::Dir>{cmt::Dir::HtoD}));
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "/fixture/copy", "--dir", "htod"}), Usage);
	CHECK(check::Require(ParseTuneArgs({"--out", "/fixture/copy", "--direction", "both"})).dirs ==
		  (std::vector<cmt::Dir>{cmt::Dir::HtoD, cmt::Dir::DtoH}));
	CHECK(check::Require(ParseTuneArgs({"--out", "/fixture/copy", "--direction", "gpu-to-cpu"})).dirs ==
		  (std::vector<cmt::Dir>{cmt::Dir::DtoH}));
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "/fixture/copy", "--direction", "sideways"}), Usage);
}

TEST(tune_full_grid_is_the_declared_default)
{
	const auto a = check::Require(ParseTuneArgs({"--out", "/fixture/full", "--search", "full", "--preview"}));
	CHECK(a.preview);
	CHECK(a.search == cmt::tune::Search::Full);
	CHECK(a.grid.sizes == (std::vector<std::uint64_t>{4u << 10, 16u << 10, 64u << 10, 128u << 10, 512u << 10,
													  1u << 20, 2u << 20, 4u << 20, 8u << 20, 16u << 20,
													  64u << 20, 128u << 20, 512u << 20, 1u << 30}));
	CHECK(a.grid.buffers == (std::vector<int>{1, 2, 4}));
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "/fixture/full", "--max-registered-mib", "64"}), Usage);
}

TEST(tune_help_separates_normal_and_advanced_controls)
{
	const auto normal = check::Require(ParseTuneArgs({"--help"})).help;
	CHECK(normal.find("Search:") < normal.find("Measurements:"));
	CHECK(normal.find("Measurements:") < normal.find("Output:"));
	for (const auto* flag : {"--direction", "--tuning-params", "--preview", "--out"})
	{
		CHECK(normal.find(flag) != std::string::npos);
	}
	for (const auto* flag : {"--hook", "--benchmark-exe", "--min-bytes", "--dry-run"})
	{
		CHECK(normal.find(flag) == std::string::npos);
	}
	CHECK(normal.find("Examples:") != std::string::npos);
	CHECK(normal.find("threads=1,2,4") == std::string::npos);
	CHECK(normal.find("--help-advanced") != std::string::npos);
	CHECK(normal.size() < 2200);
	const auto run_help = check::Require(ParseRunArgs({"--help"})).help;
	CHECK(run_help.find("--config") != std::string::npos);
	CHECK(run_help.find("--help-advanced") != std::string::npos);
	CHECK(run_help.size() < 1400);
	CHECK(normal.find("1024") != std::string::npos);
	CHECK(normal.find("w2_n4_") == std::string::npos);
	const auto advanced = check::Require(ParseTuneArgs({"--help-advanced"})).help;
	CHECK(advanced.find("Advanced:") != std::string::npos);
	for (const auto* flag : {"--hook", "--benchmark-exe", "--min-bytes", "--dry-run"})
	{
		CHECK(advanced.find(flag) != std::string::npos);
	}
	CHECK(advanced.find("--env") == std::string::npos);
}

TEST(tune_hidden_options_remain_accepted)
{
	const auto a = check::Require(
		ParseTuneArgs({"--out", "/fixture/advanced", "--hook", "/fixture/hook.so", "--benchmark-exe",
					   "/fixture/gpu_copy", "--min-bytes", "1M", "--dry-run"}));
	CHECK_EQ(a.hook, std::string("/fixture/hook.so"));
	CHECK_EQ(a.benchmark_exe, std::string("/fixture/gpu_copy"));
	CHECK_EQ(a.min_bytes, std::size_t{1u << 20});
	CHECK(a.dry_run && !a.preview);
}

TEST(worker_cpu_placement_flags_have_explicit_precedence)
{
	const auto defaults = check::Require(ParseArgs({}));
	CHECK(!defaults.worker_numa.has_value());
	CHECK(!defaults.worker_cpus.has_value());
	const auto selected = check::Require(ParseArgs({"--numa", "on", "--worker-cpus", "0-0"}));
	CHECK(selected.worker_numa.value_or(false));
	CHECK_EQ(selected.worker_cpus.value_or("missing"), std::string("0-0"));
	const auto disabled = check::Require(ParseArgs({"--numa", "off"}));
	CHECK(disabled.worker_numa.has_value());
	CHECK(!disabled.worker_numa.value_or(true));
	CHECK_ERROR_KIND(ParseArgs({"--cpu", "0", "--numa", "on"}), Usage);
	CHECK_ERROR_KIND(ParseArgs({"--cpu", "0", "--worker-cpus", "0"}), Usage);
	CHECK_ERROR_KIND(ParseArgs({"--numa", "maybe"}), Usage);
	CHECK_ERROR_KIND(ParseArgs({"--worker-numa"}), Usage); // not a second spelling of --numa
}

TEST(worker_cpu_lists_are_validated_by_all_running_actions)
{
	for (const auto* value : {"", ",1", "1,", "-1", "3-1", "a", "1--2", "1 2"})
	{
		CHECK_ERROR_KIND(ParseArgs({"--worker-cpus", value}), Usage);
		CHECK_ERROR_KIND(ParseTuneArgs({"--out", "/fixture/tune", "--worker-cpus", value}), Usage);
	}
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "/fixture/tune", "--numa", "maybe"}), Usage);
	CHECK_ERROR_KIND(ParseTraceArgs({"--out", "trace.jsonl", "--numa", "off", "--", "true"}), Usage);
}

TEST(worker_placement_is_recorded_in_tuning_request)
{
	const auto args =
		check::Require(ParseTuneArgs({"--out", "/fixture/tune", "--hook", "/fixture/hook.so", "--direction",
									  "htod", "--numa", "on", "--worker-cpus", "0"}));
	CHECK_EQ(args.env, (A{"CMT_WORKER_CPUS=0"})); // on is the default: nothing recorded
	CHECK(check::Require(ParseTuneArgs({"--out", "/fixture/tune", "--numa", "on"})).env.empty());
	CHECK_EQ(check::Require(ParseTuneArgs({"--out", "/fixture/tune", "--numa", "off"})).env,
			 (A{"CMT_WORKER_NUMA=0"}));
	CHECK(check::Require(ParseTuneArgs({"--out", "/fixture/tune"})).env.empty()); // on: the default
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "/fixture/tune", "--worker-numa"}), Usage);
}

TEST(top_level_help_lists_public_commands)
{
	const auto text = TopLevelUsage();
	for (const char* command : {"tune", "run", "export", "trace", "patch", "unpatch", "bench"})
	{
		CHECK(text.find(std::string("\n  ") + command + " ") != std::string::npos);
	}
	for (const char* name : {"check", "kernel"})
	{
		CHECK((text.find(std::string("\n  ") + name + " ") != std::string::npos) == InternalCommandsEnabled());
	}
	CHECK(text.find("cmt COMMAND --help") != std::string::npos);
	const auto bench = UsageText();
	CHECK(bench.find("cmt bench") != std::string::npos);
	CHECK(bench.find("cmt tune --help") == std::string::npos); // the command list is top-level only
	CHECK(CheckUsage().find("cmt check [--write] TUNE_JSON") != std::string::npos);
	CHECK(text.find("cmt --version") != std::string::npos);
	CHECK(text.find("Start with: cmt tune -- APP") != std::string::npos);
}

// Every multi-word flag is canonical with dashes, and its underscore spelling is the same flag: both forms, with a
// separate or an `=` value, give the same result; help shows only the dashed one.
TEST(multi_word_flags_take_dashes_or_underscores)
{
	for (const auto& [dashed, underscored] :
		 {std::pair{A{"--out", "o", "--tuning-params", "samples=3", "--host-alloc", "off", "--progress-bar=off",
					  "--max-pinned-mib", "64", "--max-processes=9", "--min-bytes", "1M", "--dry-run",
					  "--benchmark-exe", "x"},
					A{"--out", "o", "--tuning_params", "samples=3", "--host_alloc", "off", "--progress_bar=off",
					  "--max_pinned_mib", "64", "--max_processes=9", "--min_bytes", "1M", "--dry_run",
					  "--benchmark_exe", "x"}}})
	{
		const auto a = check::Require(ParseTuneArgs(dashed));
		const auto b = check::Require(ParseTuneArgs(underscored));
		CHECK(a.tuning_params == "samples=3" && b.tuning_params == "samples=3");
		CHECK(!a.host_alloc && !b.host_alloc && !a.progress_bar && !b.progress_bar);
		CHECK(a.max_pinned_mib == 64 && b.max_pinned_mib == 64 && a.max_processes == 9 && b.max_processes == 9);
		CHECK(a.min_bytes == b.min_bytes && a.dry_run && b.dry_run && b.benchmark_exe == "x");
	}
	const auto run = [](A args) { return check::Require(ParseRunArgs(std::move(args))); };
	CHECK(run({"--remote-config", "--", "app"}).remote == run({"--remote_config", "--", "app"}).remote);
	CHECK(run({"--remote_config=http://localhost:1", "--", "app"}).remote ==
		  std::optional<std::string>("http://localhost:1"));
	CHECK(run({"--config", "t", "--cuda_host_alloc_hook=on", "--", "app"}).cuda_host_alloc_hook ==
		  run({"--config", "t", "--cuda-host-alloc-hook", "on", "--", "app"}).cuda_host_alloc_hook);
	// The application's words are never rewritten.
	CHECK((run({"--config", "t", "--", "app", "--host_alloc", "--remote_config"}).command ==
		   A{"app", "--host_alloc", "--remote_config"}));
	CHECK_EQ(check::Require(ParseArgs({"--slot_bytes", "2M", "--path", "staged"})).cfg.slot_bytes,
			 std::size_t{2u << 20});
	CHECK(check::Require(ParseArgs({"--worker_cpus", "0"})).worker_cpus == std::optional<std::string>("0"));
	CHECK(!check::Require(ParseTuneArgs({"--help_advanced"})).help.empty());
	// A word that is no flag with either spelling stays an error under its own name.
	CHECK_ERROR_KIND(ParseTuneArgs({"--out", "o", "--no_such_flag", "1"}), Usage);
	for (const auto& help : {check::Require(ParseTuneArgs({"--help-advanced"})).help,
							 check::Require(ParseRunArgs({"--help"})).help, UsageText()})
	{
		for (const char* old : {"--tuning_params", "--host_alloc", "--progress_bar", "--remote_config",
								"--cuda_host_alloc_hook", "--max_pinned_mib"})
		{
			CHECKF(help.find(old) == std::string::npos, "%s in help", old);
		}
	}
}

TEST(trace_arguments_preserve_the_application_command)
{
	const auto a = check::Require(
		ParseTraceArgs({"--out", "copy sizes.jsonl", "--", "python", "train.py", "--out", "model dir"}));
	CHECK_EQ(a.out, std::string("copy sizes.jsonl"));
	CHECK_EQ(a.command, (std::vector<std::string>{"python", "train.py", "--out", "model dir"}));
	CHECK(check::Require(ParseTraceArgs({"--help"})).help.find("requested bytes") != std::string::npos);
	CHECK(check::Require(ParseTraceArgs({"--help"})).help.find("-- COMMAND [ARGS...]") != std::string::npos);
	CHECK(check::Require(ParseTraceArgs({"--help"})).help.find("125") != std::string::npos);
	CHECK(check::Require(ParseRunArgs({"--help"})).help.find("-- COMMAND [ARGS...]") != std::string::npos);
	CHECK_ERROR_KIND(ParseTraceArgs({"--out", "x"}), Usage);
	CHECK_ERROR_KIND(ParseTraceArgs({"--out", "x", "--"}), Usage);
	CHECK_ERROR_KIND(ParseTraceArgs({"--", "true"}), Usage);
	CHECK_ERROR_KIND(ParseTraceArgs({"--out", "-", "--", "true"}), Usage);
	CHECK_ERROR_KIND(ParseTraceArgs({"--out", "x", "--bad", "--", "true"}), Usage);
}

TEST(children_drop_hook_variables_but_keep_the_evaluation_runners)
{
	for (const char* name : {"LD_PRELOAD", "CMT_CONFIG", "CMT_OVERRIDE", "CMT_WORKER_NUMA", "STAGING_X",
							 "MEMCPY_HOOK_X", "CUDAMEMCPY_TUNER_TRACE_X"})
	{
		CHECK(HookVariable(name));
	}
	// A workload's stock and hooked runs must see the same environment folder.
	for (const char* name : {"CMT_ENV_ROOT", "CMT_WORKLOAD", "CMT_BIN_DIR", "CUDA_VISIBLE_DEVICES", "PATH"})
	{
		CHECK(!HookVariable(name));
	}
}

TEST_MAIN()
