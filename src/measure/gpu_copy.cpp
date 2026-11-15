#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include "cli.h"
#include "config.h"
#include "copying/chunk_pipeline.h"
#include "measure/copy.h"
#include "measure/inline_policy.h"
#include "measure/metrics.h"
#include "cuda/cuda.h"
#include "common/helpers.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "log.h"

namespace {
using cmt::Result;
constexpr uint64_t kMaxExactInteger = uint64_t{1} << 53;
int exit_code = 0;

struct RunnerOwner
{
	std::size_t max_bytes;
	std::unique_ptr<cmt::PathRunner> runner;
};

struct Series
{
	std::string name;
	cmt::Config cfg;
	cmt::DeviceInfo dev;
	std::size_t bytes;
	RunnerOwner* owner;
	cmt::MeasurementRow identity;
	cmt::CapacityLimit capacity_limit;
	cmt::Measurement measurement;
	bool terminal_failure = false;
};

bool ExactJobIntegers(const cmt::ring::CopyObservation& job)
{
	for (const auto value : {job.completed, job.requested_workers, job.actual_workers, job.slots,
							 job.max_slots_used, job.slot_bytes, job.variant, job.write_back, job.flush_bytes,
							 job.bench_dma, job.wc_slots, job.pace_quantum})
	{
		if (value > kMaxExactInteger)
		{
			return false;
		}
	}
	return true;
}

const char* OutcomeName(cmt::tune_audit::Outcome outcome)
{
	constexpr const char* names[] = {
		"Unknown",	   "Chunks",	   "DriverRow",		"Fork",		  "Capture", "ProgressivePinning",
		"NonPageable", "NoPipeline",   "KnobsMismatch", "NotStarted", "Env",	 "Driver",
		"NotReady",	   "LoopFallback", "Nested"};
	static_assert(std::size(names) == static_cast<unsigned>(cmt::tune_audit::Outcome::Count));
	const auto index = static_cast<unsigned>(outcome);
	return index < std::size(names) ? names[index] : "Invalid";
}

class TuneAudit
{
public:
	TuneAudit(benchmark::State& benchmark_state, const cmt::Config& cfg, std::size_t transfer_bytes)
		: state(benchmark_state), bytes(transfer_bytes), direction(cfg.dir),
		  thread(std::this_thread::get_id())
	{
	}

	Result<void> Setup(const cmt::Config& cfg)
	{
		const char* requested = std::getenv("CMT_TUNE_AUDIT");
		if (!requested || std::strcmp(requested, "1") != 0)
		{
			return {};
		}
		enabled = true;
		CMT_RETURN_IF_ERROR(Require(state.threads() == 1, "requires a single benchmark caller thread"));
		CMT_RETURN_IF_ERROR(Require(cfg.path->name == "pageable" && cfg.verify && cfg.dma,
									"requires verified, completed pageable copies"));
		const char* path = cmt::SecureExecution() ? nullptr : std::getenv("CMT_EXPECTED_RUN");
		CMT_RETURN_IF_ERROR(Require(path && *path, "requires CMT_EXPECTED_RUN"));
		auto text = cmt::ReadTextFile(path, cmt::tuning::kMaxConfigBytes);
		if (!text)
		{
			return cmt::Failure(text.error());
		}
		JsonParser parser;
		Json request;
		Json settings;
		std::string_view method;
		CMT_RETURN_IF_ERROR(Require(!parser.parse(*text).get(request) && !request["settings"].get(settings) &&
										!settings["method"].get(method) &&
										(method == "staged" || method == "progressive_pinning"),
									"audit requires requested staged or progressive-pinning settings"));
		if (method == "progressive_pinning")
		{
			auto pinning = cmt::ParsePinningSettings(settings);
			if (!pinning)
			{
				return cmt::Failure(pinning.error());
			}
			expected_pinning = *pinning;
			mode = cmt::Mode::ProgressivePinning;
		}
		else
		{
			auto chunks = cmt::ParseChunksSettings(settings);
			if (!chunks)
			{
				return cmt::Failure(chunks.error());
			}
			expected_settings = *chunks;
		}
		const auto hook_config = cmt::ParseConfig([](const char* name) { return std::getenv(name); },
												  cmt::hwcopy::GetCpu());
		if (!hook_config)
		{
			return cmt::Failure(hook_config.error());
		}
		CMT_RETURN_IF_ERROR(
			Require(!hook_config->debug, "debug logging must be disabled for timed evidence"));
		CMT_RETURN_IF_ERROR(
			Require(hook_config->tune_table == nullptr, "fixed run must not load a tuning result"));
		if (mode == cmt::Mode::Chunks)
		{
			cmt::CopyCtx ctx;
			const auto selected = cmt::hwcopy::Select(expected_settings.loop, ctx, hook_config->copy_isa,
													  cmt::hwcopy::GetCpu());
			CMT_RETURN_IF_ERROR(
				Require(selected && !selected->fell_back, "requested loop/ISA cannot execute"));
			loop = selected->loop;
		}
		void* begin_symbol = dlsym(RTLD_DEFAULT, "cmt_tune_audit_begin");
		void* end_symbol = dlsym(RTLD_DEFAULT, "cmt_tune_audit_end");
		static_assert(sizeof(begin) == sizeof(begin_symbol) && sizeof(end) == sizeof(end_symbol),
					  "function pointer ABI");
		std::memcpy(&begin, &begin_symbol, sizeof(begin));
		std::memcpy(&end, &end_symbol, sizeof(end));
		CMT_RETURN_IF_ERROR(Require(begin && end, "hook observation symbols missing"));
		state.counters["tune_audit_version"] = cmt::tune_audit::kVersion;
		state.counters["tune_audit_valid"] = 0;
		state.counters["tune_audit_row_mode"] = static_cast<unsigned>(mode);

		return {};
	}

	TuneAudit(const TuneAudit&) = delete;
	TuneAudit& operator=(const TuneAudit&) = delete;

	~TuneAudit()
	{
		if (audit_open)
		{
			cmt::tune_audit::Snapshot ignored;
			(void)end(&ignored, sizeof(ignored));
		}
	}

	Result<void> Begin()
	{
		if (!enabled)
		{
			return {};
		}
		CMT_RETURN_IF_ERROR(Require(thread == std::this_thread::get_id(), "caller thread changed"));
		CMT_RETURN_IF_ERROR(Require(begin(cmt::tune_audit::kVersion, sizeof(cmt::tune_audit::Snapshot)) == 1,
									"begin ABI/version mismatch or overlapping observation"));
		audit_open = true;

		return {};
	}

	Result<void> End(uint64_t expected_calls, bool warmup)
	{
		if (!enabled)
		{
			return {};
		}
		CMT_RETURN_IF_ERROR(
			Require(bytes > 0 && bytes <= kMaxExactInteger && expected_calls <= kMaxExactInteger / bytes,
					"sample accounting exceeds exact JSON integer range"));
		cmt::tune_audit::Snapshot s;
		CMT_RETURN_IF_ERROR(Require(thread == std::this_thread::get_id(), "caller thread changed"));
		const int ok = end(&s, sizeof(s));
		audit_open = false;
		CMT_RETURN_IF_ERROR(Require(ok == 1 && s.version == cmt::tune_audit::kVersion && s.size == sizeof(s),
									"end ABI/version mismatch"));
		const auto outcome = mode == cmt::Mode::ProgressivePinning
								 ? cmt::tune_audit::Outcome::ProgressivePinning
								 : cmt::tune_audit::Outcome::Env;
		const auto accepted = s.outcomes[static_cast<unsigned>(outcome)];
		const std::string prefix = warmup ? "tune_audit_warmup_" : "tune_audit_measured_";
		CMT_RETURN_IF_ERROR(Add(prefix + "calls", s.calls));
		CMT_RETURN_IF_ERROR(Add(prefix + "bytes", s.bytes));
		CMT_RETURN_IF_ERROR(Add(prefix + "accepted", accepted));
		CMT_RETURN_IF_ERROR(Add(prefix + "sync_calls", s.sync_calls));
		CMT_RETURN_IF_ERROR(Add(prefix + "async_calls", s.async_calls));
		CMT_RETURN_IF_ERROR(Add(prefix + "rejected", s.calls - accepted));
		CMT_RETURN_IF_ERROR(Add(prefix + "errors", s.errors));
		CMT_RETURN_IF_ERROR(Add(prefix + "mixed", s.mixed));
		for (unsigned i = 0; i < static_cast<unsigned>(cmt::tune_audit::Outcome::Count); ++i)
		{
			CMT_RETURN_IF_ERROR(Add(prefix + "outcome" + std::to_string(i), s.outcomes[i]));
		}
		CMT_RETURN_IF_ERROR(Require(s.row_min <= kMaxExactInteger &&
										(s.row_max == UINT64_MAX || s.row_max <= kMaxExactInteger),
									"observed tuple exceeds exact JSON integer range"));
		state.counters["tune_audit_row_min_bytes"] = static_cast<double>(s.row_min);
		state.counters["tune_audit_row_max_bytes"] =
			s.row_max == UINT64_MAX ? -1.0 : static_cast<double>(s.row_max);
		std::string unexpected = s.calls == 0 ? "no intercepted copy" : "none";
		for (unsigned i = 0; i < static_cast<unsigned>(cmt::tune_audit::Outcome::Count); ++i)
		{
			if (i != static_cast<unsigned>(outcome) && s.outcomes[i] != 0)
			{
				unexpected = fmt::format("{} count={}", OutcomeName(static_cast<cmt::tune_audit::Outcome>(i)),
										 s.outcomes[i]);
				break;
			}
		}
		const bool matches =
			s.calls == expected_calls && accepted == expected_calls && s.bytes == bytes * expected_calls &&
			s.direction == static_cast<uint64_t>(direction) && s.copy_bytes == bytes && s.row_min == 0 &&
			s.row_max == 0 && s.errors == 0 && s.mixed == 0 && s.sync_calls + s.async_calls == s.calls;
		if (!matches)
		{
			spdlog::error("path/count mismatch: expected {} calls, observed {}, accepted {}, errors {}, "
						  "mixed {}; first unexpected {}",
						  expected_calls, s.calls, accepted, s.errors, s.mixed, unexpected);
			return cmt::Failure(cmt::Error::Kind::Verify, "path/count mismatch");
		}
		if (mode == cmt::Mode::ProgressivePinning)
		{
			return CheckPinning(s.pinning);
		}
		else
		{
			const auto& j = s.job;
			CMT_RETURN_IF_ERROR(
				Require(ExactJobIntegers(j), "observed job exceeds exact JSON integer range"));
			state.counters["tune_audit_requested_workers"] = static_cast<double>(j.requested_workers);
			state.counters["tune_audit_actual_workers"] = static_cast<double>(j.actual_workers);
			state.counters["tune_audit_slots"] = static_cast<double>(j.slots);
			state.counters["tune_audit_max_slots_used"] = static_cast<double>(j.max_slots_used);
			state.counters["tune_audit_slot_bytes"] = static_cast<double>(j.slot_bytes);
			state.counters["tune_audit_variant"] = static_cast<double>(j.variant);
			state.counters["tune_audit_write_back"] = static_cast<double>(j.write_back);
			state.counters["tune_audit_flush_bytes"] = static_cast<double>(j.flush_bytes);
			state.counters["tune_audit_bench_dma"] = static_cast<double>(j.bench_dma);
			state.counters["tune_audit_wc_slots"] = static_cast<double>(j.wc_slots);
			state.counters["tune_audit_pace_gbps"] = j.pace_gbps;
			state.counters["tune_audit_pace_quantum"] = static_cast<double>(j.pace_quantum);
			const uint64_t chunks = (bytes + expected_settings.slot_bytes - 1) / expected_settings.slot_bytes;
			const uint64_t workers = std::min<uint64_t>(expected_settings.workers, chunks);
			CMT_RETURN_IF_ERROR(Require(
				j.completed == 1 && j.requested_workers == static_cast<uint64_t>(expected_settings.workers) &&
					j.actual_workers == workers &&
					j.slots == static_cast<uint64_t>(expected_settings.slots) &&
					j.max_slots_used == std::min<uint64_t>(j.slots, (chunks + workers - 1) / workers) &&
					j.slot_bytes == expected_settings.slot_bytes &&
					j.variant == static_cast<uint64_t>(loop.id) &&
					j.write_back == static_cast<uint64_t>(loop.ctx.write_back) &&
					j.flush_bytes == loop.ctx.flush_batch && j.bench_dma == 0 && j.pace_gbps == 0 &&
					j.wc_slots == static_cast<uint64_t>(expected_settings.wc_slots),
				"actual job differs from requested loop/ISA/shape, or CPU leg was bypassed/paced"));
			const auto variant = cmt::hwcopy::VariantOf(loop.id);
			state.counters["tune_audit_method"] = static_cast<unsigned>(variant.method);
			state.counters["tune_audit_isa"] = static_cast<unsigned>(variant.isa);
		}

		return {};
	}

	Result<void> Complete(uint64_t expected_calls)
	{
		if (!enabled)
		{
			return {};
		}
		CMT_RETURN_IF_ERROR(
			Require(state.counters["tune_audit_measured_calls"].value == static_cast<double>(expected_calls),
					"aggregate measured call count differs from iterations * copies_per_sample"));
		state.counters["tune_audit_single_caller"] = 1;
		state.counters["tune_audit_valid"] = 1;

		return {};
	}

private:
	Result<void> CheckPinning(const cmt::pinning::Observation& observed)
	{
		for (const auto value : {observed.completed, observed.registered_bytes, observed.released_bytes,
								 observed.registered_dma_bytes, observed.pageable_dma_bytes,
								 observed.borrowed_bytes, observed.chunks, observed.first_chunk_bytes,
								 observed.max_chunk_bytes, observed.max_registered_bytes, observed.read_only})
		{
			CMT_RETURN_IF_ERROR(
				Require(value <= kMaxExactInteger, "observed pinning exceeds exact JSON integer range"));
		}
		state.counters["tune_audit_pinning_completed"] = static_cast<double>(observed.completed);
		state.counters["tune_audit_pinning_registered_bytes"] =
			static_cast<double>(observed.registered_bytes);
		state.counters["tune_audit_pinning_released_bytes"] = static_cast<double>(observed.released_bytes);
		state.counters["tune_audit_pinning_registered_dma_bytes"] =
			static_cast<double>(observed.registered_dma_bytes);
		state.counters["tune_audit_pinning_pageable_dma_bytes"] =
			static_cast<double>(observed.pageable_dma_bytes);
		state.counters["tune_audit_pinning_borrowed_bytes"] = static_cast<double>(observed.borrowed_bytes);
		state.counters["tune_audit_pinning_chunks"] = static_cast<double>(observed.chunks);
		state.counters["tune_audit_pinning_first_chunk_bytes"] =
			static_cast<double>(observed.first_chunk_bytes);
		state.counters["tune_audit_pinning_max_chunk_bytes"] = static_cast<double>(observed.max_chunk_bytes);
		state.counters["tune_audit_pinning_max_registered_bytes"] =
			static_cast<double>(observed.max_registered_bytes);
		state.counters["tune_audit_pinning_read_only"] = static_cast<double>(observed.read_only);

		const auto rounded_bytes = cmt::AlignUp(bytes, cmt::BasePageBytes());
		CMT_RETURN_IF_ERROR(
			Require(rounded_bytes && observed.completed == 1 && observed.registered_bytes >= *rounded_bytes &&
						(!expected_pinning.max_registered_bytes ||
						 observed.registered_bytes <= *expected_pinning.max_registered_bytes) &&
						observed.released_bytes == observed.registered_bytes &&
						observed.registered_dma_bytes == bytes && observed.pageable_dma_bytes == 0 &&
						observed.borrowed_bytes == 0 && observed.chunks > 0,
					"pinning did not register, copy through completion and release the full transfer within "
					"its budget"));
		CMT_RETURN_IF_ERROR(
			Require(observed.first_chunk_bytes == expected_pinning.first_chunk_bytes &&
						observed.max_chunk_bytes == expected_pinning.max_chunk_bytes &&
						observed.max_registered_bytes == expected_pinning.max_registered_bytes.value_or(0) &&
						observed.read_only == static_cast<uint64_t>(expected_pinning.read_only),
					"actual pinning settings differ from the requested settings"));
		return {};
	}

	template <std::size_t N>
	static Result<void> Require(bool ok, const char (&message)[N])
	{
		if (!ok)
		{
			return cmt::Failure(cmt::Error::Kind::Verify, message);
		}

		return {};
	}

	Result<void> Add(const std::string& key, uint64_t value)
	{
		auto& counter = state.counters[key].value;
		CMT_RETURN_IF_ERROR(
			Require(value <= kMaxExactInteger && counter <= static_cast<double>(kMaxExactInteger - value),
					"aggregate accounting exceeds exact JSON integer range"));
		counter += static_cast<double>(value);

		return {};
	}

	benchmark::State& state;
	std::size_t bytes;
	cmt::Dir direction;
	std::thread::id thread;
	cmt::Mode mode = cmt::Mode::Chunks;
	cmt::StagingSettings expected_settings;
	cmt::PinningSettings expected_pinning;
	cmt::CopyLoop loop;
	cmt::tune_audit::Begin begin = nullptr;
	cmt::tune_audit::End end = nullptr;
	bool enabled = false;
	bool audit_open = false;
};

Result<void> Measure(benchmark::State& state, Series* series)
{
	if (series->capacity_limit != cmt::CapacityLimit::None)
	{
		state.SkipWithMessage(fmt::format("capacity_skipped: {}", series->measurement.reason).c_str());
		series->measurement.status = "capacity";
		return {};
	}
	const auto& cfg = series->cfg;
	const auto& dev = series->dev;
	const auto bytes = series->bytes;
	{
		const auto& path = *cfg.path;
		auto& runner = series->owner->runner;
		if (!runner)
		{
			auto created = path.make(cmt::RunContext{cfg, dev, series->owner->max_bytes});
			if (!created)
			{
				return cmt::Failure(std::move(created.error()));
			}
			runner = std::move(*created);
		}
		series->measurement.allocation = runner->Alloc();
		const auto reps = cmt::RepsFor(bytes, cfg.min_bytes);
		TuneAudit audit(state, cfg, bytes);
		CMT_RETURN_IF_ERROR(audit.Setup(cfg));
		CMT_RETURN_IF_ERROR(runner->Prepare(bytes));
		CMT_RETURN_IF_ERROR(audit.Begin());
		CMT_RETURN_IF_ERROR(runner->Sample(bytes, 1));
		CMT_RETURN_IF_ERROR(audit.End(1, true));
		CMT_RETURN_IF_ERROR(
			runner->ClearDestination(bytes));
		cmt::SampleOut sample;
		for (auto iteration : state)
		{
			(void)iteration;
			if (cfg.flush && cmt::FlushSupported() && path.dirs != cmt::PathSpec::Dirs::None)
			{
				CMT_RETURN_IF_ERROR(runner->Flush(bytes));
			}
			CMT_RETURN_IF_ERROR(audit.Begin());
			auto measured = runner->Sample(bytes, reps);
			if (!measured)
			{
				return cmt::Failure(std::move(measured.error()));
			}
			sample = std::move(*measured);
			series->measurement.samples.push_back(sample);
			CMT_RETURN_IF_ERROR(audit.End(reps, false));
			state.SetIterationTime(sample.seconds);
		}
		std::optional<std::size_t> wrong;
		if (cfg.verify)
		{
			auto checked = runner->Verify(bytes);
			if (!checked)
			{
				return cmt::Failure(std::move(checked.error()));
			}
			wrong = *checked;
			if (wrong && *wrong != 0)
			{
				auto again = runner->Verify(bytes);
				fmt::print(stderr, "verification: {} bytes, {} wrong 8-byte words, {} on a second check\n", bytes,
						   *wrong, again && *again ? fmt::to_string(**again) : "unknown");
				return cmt::Failure(cmt::Error::Kind::Verify, "copy verification failed");
			}
		}
		CMT_RETURN_IF_ERROR(audit.Complete(static_cast<uint64_t>(state.iterations()) * reps));
		state.counters["verified"] = wrong ? 1.0 : 0.0;
		state.counters["has_phases"] = sample.phases ? 1.0 : 0.0;
		state.counters["has_directions"] = sample.directions ? 1.0 : 0.0;
		state.counters["has_lifecycle"] = sample.lifecycle ? 1.0 : 0.0;
		if (sample.lifecycle)
		{
			for (std::size_t i = 0; i < cmt::kLifecycleCounters.size(); i++)
			{
				state.counters[cmt::kLifecycleCounters[i]] = (*sample.lifecycle)[i];
			}
		}
		state.counters["setup_s"] = sample.phases ? sample.phases->setup : 0;
		state.counters["copy_s"] = sample.phases ? sample.phases->copy : 0;
		state.counters["teardown_s"] = sample.phases ? sample.phases->teardown : 0;
		state.counters["htod_gbps"] = sample.directions ? sample.directions->htod_gbps : 0;
		state.counters["dtoh_gbps"] = sample.directions ? sample.directions->dtoh_gbps : 0;
		series->identity =
			cmt::ResultIdentity(cfg, dev, *runner, bytes, reps,
								cfg.flush && cmt::FlushSupported() && path.dirs != cmt::PathSpec::Dirs::None,
								wrong ? cmt::Verified::Ok : cmt::Verified::Skip);
		state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(bytes) * reps *
								path.moved_bytes);
		state.counters["copies_per_sample"] = reps;
		state.counters["verification_enabled"] = cfg.verify ? 1.0 : 0.0;
		series->measurement.copies_per_sample = reps;
		series->measurement.verified = wrong.has_value();
		series->measurement.verification = path.name == "register" || path.name == "host_allocation"
											   ? "host_contents_unchanged"
											   : "byte_equality";
		series->measurement.status = "complete";
		series->measurement.reason.clear();
		series->measurement.huge_kib = runner->HugeKiB();
		for (const auto& [key, value] : state.counters)
		{
			if (key.rfind("tune_audit_", 0) == 0)
			{
				series->measurement.observations[key] = value.value;
			}
		}
		const auto huge = runner->HugeKiB();
		state.counters["huge_kib"] = huge ? static_cast<double>(*huge) : -1.0;
		const bool driver_pages = path.accepts.Has(cmt::Knob::Alloc) && cfg.alloc != cmt::Alloc::Register;
		state.SetLabel(fmt::format("pages={} alloc={} wc_slots={}",
								   driver_pages ? "driver" : cmt::PagesName(cfg.pages), runner->Alloc(),
								   cfg.wc_slots));
	}
	return {};
}

void MeasureCallback(benchmark::State& state, Series* series)
{
	if (series->terminal_failure)
	{
		state.SkipWithError(series->measurement.reason.c_str());
		return;
	}
	auto result = Measure(state, series);
	if (!result)
	{
		series->terminal_failure = true;
		series->measurement.status = result.error().IsUnsupported() ? "unsupported" : "failed";
		series->measurement.verified = false;
		series->measurement.reason = result.error().Message();
		state.SkipWithError(result.error().Message());
		exit_code = std::max(exit_code, result.error().ExitCode());
	}
}
} // namespace

Result<int> RunMain(int argc, char** argv)
{
	{
		std::vector<std::string> copy_args;
		for (int i = 1; i < argc; i++)
		{
			if (std::string(argv[i]) != "--")
			{
				continue;
			}
			copy_args.assign(argv + i + 1, argv + argc);
			argc = i;
			break;
		}
		std::string schema_file;
		std::string output_file;
		bool rows_only = false;
		for (int i = 1; i < argc; ++i)
		{
			const std::string arg(argv[i]);
			int consumed = 1;
			if (arg == "--out")
			{
				if (i + 1 == argc)
				{
					return cmt::UsageError("--out requires a file");
				}
				output_file = argv[i + 1];
				consumed = 2;
			}
			else if (arg.rfind("--out=", 0) == 0)
			{
				output_file = arg.substr(6);
			}
			else if (arg == "--rows")
			{
				rows_only = true;
				schema_file = "-";
			}
			else if (arg.rfind("--csv=", 0) == 0)
			{
				schema_file = arg.substr(6);
			}
			else
			{
				continue;
			}
			if ((arg.rfind("--out", 0) == 0 && output_file.empty()) ||
				(arg.rfind("--csv", 0) == 0 && schema_file.empty()))
			{
				return cmt::UsageError("output file must not be empty");
			}
			for (int j = i; j + consumed < argc; ++j)
			{
				argv[j] = argv[j + consumed];
			}
			argc -= consumed;
			--i;
		}
		auto parsed = cmt::ParseArgs(copy_args);
		if (!parsed)
		{
			return cmt::Failure(std::move(parsed.error()));
		}
		auto options = std::move(*parsed);
		if (options.help)
		{
			CMT_RETURN_IF_ERROR(cmt::WriteStdout(fmt::format(
				"gpu_copy [--out FILE] [--csv=FILE|-] [Google Benchmark flags] -- [copy options]\n"
				"Capacity-skipped sizes emit Google Benchmark skip records, with no measurement "
				"CSV rows.\n{}",
				cmt::UsageText())));
			return 0;
		}
		if (options.header)
		{
			CMT_RETURN_IF_ERROR(cmt::WriteStdout(cmt::CsvHeader() + "\n"));
			return 0;
		}
		if (options.cfg.budget_s != 0)
		{
			return cmt::UsageError(
				"gpu_copy: --budget-s is not supported; use --samples to bound repetitions");
		}
		benchmark::Initialize(&argc, argv);
		if (benchmark::ReportUnrecognizedArguments(argc, argv))
		{
			return 2;
		}
		CMT_RETURN_IF_ERROR(options.ApplyWorkerPlacement());
		const auto inline_context = cmt::bench::ConfigureInlinePolicy(options.cfg);
		if (!inline_context)
		{
			return cmt::Failure(inline_context.error());
		}
		for (const auto& [key, value] : *inline_context)
		{
			benchmark::AddCustomContext(key, value);
		}
		auto prepared = cmt::PrepareBenchmark(options.cfg);
		if (!prepared)
		{
			return cmt::Failure(prepared.error());
		}
		if (!prepared->has_value())
		{
			return 77;
		}
		const auto dev = std::move(prepared->value());
		auto machine = cmt::MachineReportFor(options.cfg.device, false);
		if (!machine)
		{
			return cmt::Failure(machine.error());
		}
		if (options.info)
		{
			CMT_RETURN_IF_ERROR(cmt::WriteStdout(cmt::Describe(dev)));
			return 0;
		}
		CMT_CUDA(cudaSetDevice(dev.ordinal));
		cmt::SizeCoverage coverage;
		auto fitting = cmt::SizesThatFit(options.cfg, *options.cfg.path, &coverage);
		if (!fitting)
		{
			return cmt::Failure(std::move(fitting.error()));
		}
		const auto sizes = std::move(*fitting);
		if (const char* audit = std::getenv("CMT_TUNE_AUDIT"); audit && std::strcmp(audit, "1") == 0)
		{
			benchmark::AddCustomContext("cmt_tune_audit_schema", fmt::to_string(cmt::tune_audit::kVersion));
		}
		benchmark::AddCustomContext("cmt_tag", options.cfg.tag);
		benchmark::AddCustomContext("cmt_path", options.cfg.path->name);
		benchmark::AddCustomContext("cmt_direction", cmt::DirName(options.cfg.dir));
		benchmark::AddCustomContext("cmt_samples", fmt::to_string(options.cfg.samples));
		benchmark::AddCustomContext("cmt_min_bytes", fmt::to_string(options.cfg.min_bytes));
		benchmark::AddCustomContext("cmt_size_coverage_schema", "1");
		benchmark::AddCustomContext("cmt_size_coverage", cmt::SizeCoverageJson(coverage));
		RunnerOwner owner{sizes.empty() ? 0 : *std::max_element(std::begin(sizes), std::end(sizes)), {}};
		const char* expected_run = cmt::SecureExecution() ? nullptr : std::getenv("CMT_EXPECTED_RUN");
		std::string expected_settings;
		std::string_view expected_method;
		uint64_t expected_round = 0;
		JsonParser expected_parser;
		if (expected_run)
		{
			auto expected_text = cmt::ReadTextFile(expected_run, cmt::tuning::kMaxConfigBytes);
			if (!expected_text)
			{
				return cmt::Failure(expected_text.error());
			}
			Json request;
			Json settings;
			simdjson::dom::array requested_sizes;
			std::string_view schema, direction;
			uint64_t version = 0;
			if (expected_parser.parse(*expected_text).get(request) || request["schema"].get(schema) ||
				schema != "cmt.benchmark-run" || request["version"].get(version) || version != 2 ||
				request["direction"].get(direction) || direction != cmt::DirName(options.cfg.dir) ||
				request["sizes_bytes"].get(requested_sizes) || requested_sizes.size() != coverage.sizes.size() ||
				request["process_round"].get(expected_round) || expected_round == 0 || expected_round > UINT_MAX ||
				request["settings"].get(settings) || settings["method"].get(expected_method) ||
				(expected_method != "staged" && expected_method != "progressive_pinning" &&
				 expected_method != "driver"))
			{
				return cmt::UsageError("expected run does not match this benchmark");
			}
			std::size_t index = 0;
			for (Json value : requested_sizes)
			{
				uint64_t bytes = 0;
				if (value.get(bytes) || bytes != coverage.sizes[index++].bytes)
				{
					return cmt::UsageError("expected run sizes differ from --sizes");
				}
			}
			expected_settings = simdjson::to_string(settings);
			if (expected_method != "driver")
			{
				const char* audit = std::getenv("CMT_TUNE_AUDIT");
				if (!audit || std::strcmp(audit, "1") != 0)
				{
					return cmt::UsageError("fixed hooked run requires CMT_TUNE_AUDIT=1");
				}
			}
			else if (std::getenv("LD_PRELOAD"))
			{
				return cmt::UsageError("bare driver control must have no preload");
			}
		}
		std::vector<std::unique_ptr<Series>> series;
		for (const auto& size : coverage.sizes)
		{
			const auto bytes = size.bytes;
			const std::string name =
				fmt::format("{}/{}/{}", options.cfg.path->name, cmt::DirName(options.cfg.dir), bytes);
			series.push_back(
				std::make_unique<Series>(Series{name, options.cfg, dev, bytes, &owner, {}, size.limit, {}}));
			auto& measurement = series.back()->measurement;
			measurement.config = options.cfg;
			measurement.bytes = bytes;
			if (options.cfg.path->name == "pinned_ce")
			{
				measurement.allocation = cmt::AllocName(options.cfg.alloc);
			}
			if (size.reason != nullptr)
			{
				measurement.reason = size.reason;
			}
			const char* cuda_trace = std::getenv("CUDAMEMCPY_TUNER_TRACE_ENABLE");
			measurement.measurement_class = "primary";
			if (cuda_trace && std::strcmp(cuda_trace, "1") == 0)
			{
				measurement.measurement_class = "cuda_trace";
			}
			measurement.experiment = *inline_context;
			measurement.execution =
				(options.cfg.path->name == "staged" || options.cfg.path->name == "staged_lifecycle")
					? "direct_engine"
					: (std::getenv("LD_PRELOAD") ? "hook_unverified" : "bare_driver");
			if (expected_run)
			{
				measurement.process_round = static_cast<unsigned>(expected_round);
				measurement.settings_json = expected_settings;
				measurement.execution = expected_method == "driver" ? "bare_driver" : "hooked_copy";
			}
			benchmark::RegisterBenchmark(name.c_str(), MeasureCallback, series.back().get())
				->UseManualTime()
				->Iterations(1)
				->Repetitions(size.limit == cmt::CapacityLimit::None ? options.cfg.samples : 1)
				->ReportAggregatesOnly(false);
		}
		std::ofstream csv;
		cmt::bench::SchemaReporter reporter(
			[&](const std::string& name, const std::vector<cmt::SampleOut>& samples, cmt::Verified verified)
			{
				for (const auto& item : series)
				{
					if (item->name != name)
					{
						continue;
					}
					auto row = item->identity;
					row.verified = verified;
					cmt::SummarizeResult(row, samples, item->cfg.path->moved_bytes);
					return item->owner->runner->Rows(row, samples);
				}
				cmt::Panic("Google Benchmark reported an unregistered series");
			},
			!rows_only);
		if (!std::empty(schema_file) && schema_file != "-")
		{
			csv.open(schema_file);
			if (!csv)
			{
				return cmt::Failure(cmt::Error::Kind::Os, "cannot open schema CSV");
			}
			reporter.SetOutputStream(&csv);
		}
		benchmark::RunSpecifiedBenchmarks(std::empty(schema_file) ? nullptr : &reporter);
		if (!std::empty(schema_file) && !reporter.GetOutputStream())
		{
			return cmt::Failure(cmt::Error::Kind::Os, "writing schema CSV failed");
		}
		if (csv.is_open())
		{
			csv.close();
			if (!csv)
			{
				return cmt::Failure(cmt::Error::Kind::Os, "closing schema CSV failed");
			}
		}
		benchmark::Shutdown();
		owner.runner.reset();
		if (!output_file.empty())
		{
			std::vector<cmt::Measurement> measurements;
			for (auto& item : series)
			{
				measurements.push_back(std::move(item->measurement));
			}
			CMT_RETURN_IF_ERROR(cmt::WriteMeasurements(output_file, machine->text, measurements));
		}
		return exit_code;
	}
}

int main(int argc, char** argv)
{
	cmt::InstallFatalAllocationHandler();
	auto result = RunMain(argc, argv);
	if (result)
	{
		return *result;
	}
	fmt::print(stderr, "gpu_copy: {}\n", result.error().Message());
	return result.error().ExitCode();
}
