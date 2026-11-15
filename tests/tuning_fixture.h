// tune.json fixtures for the selection and sharing tests: measured runs as `cmt tune` records them, in a
// cmt.tuning document without a selection line (Reselect adds one).
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "check.h"
#include "common/helpers.h"
#include "config.h"

namespace tuning_fixture {
using namespace cmt;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;
constexpr char kMachine[] = R"({"host":{"cpu":{"model":"test CPU","arch":"test"}},"gpus":[{"name":"test GPU"}]})";
constexpr char kRequest[] = R"({"process_rounds":1,"max_pinned_mib":1024,"environment":["CMT_WORKER_CPUS=0,2"])";

// How a FULL with-kernel request records the registration limit (PinningSettings; tune_table.cpp, RegistrationLimitOf).
enum class Limit : std::uint8_t
{
	OldDefault,		// as `cmt tune` wrote every request before limits were explicit only: 1024 MiB, no statement
	OldExplicit512, // the same era with --max-registered-mib 512
	None,			// `cmt tune` now, without --max-registered-mib: null, "registration_limit":"explicit_only"
	Explicit1G		// `cmt tune` now with --max-registered-mib 1024
};

// What the request and each of its pinning runs record for `limit`.
inline std::optional<std::uint64_t> Recorded(Limit limit)
{
	switch (limit)
	{
	case Limit::OldDefault:
	case Limit::Explicit1G:
		return 1024 * kMiB;
	case Limit::OldExplicit512:
		return 512 * kMiB;
	case Limit::None:
		return std::nullopt;
	}
	return std::nullopt;
}

// The FULL with-kernel request that pinning runs need (SelectTable), with PinningSettings' chunk defaults, as
// RequestJson (tuner.cpp) writes it; test_tuner checks those fields of RequestJson itself.
inline std::string PinningRequest(Limit limit)
{
	const auto recorded = Recorded(limit);
	const bool old = limit == Limit::OldDefault || limit == Limit::OldExplicit512;
	return fmt::format(R"(,"progressive_pinning":true,"search":"full_with_kernel","max_registered_mib":{},{})"
					   R"("progressive_pinning_settings":{{"method":"progressive_pinning","first_chunk_bytes":4194304,)"
					   R"("max_chunk_bytes":67108864,"max_registered_bytes":{},"read_only":false}})",
					   recorded ? std::to_string(*recorded / kMiB) : "null",
					   old ? "" : R"("registration_limit":"explicit_only",)",
					   recorded ? std::to_string(*recorded) : "null");
}

inline CopyRun Run(Dir direction, std::uint64_t bytes, Mode mode, double gbps, const char* coverage,
			StagingSettings settings = {})
{
	CopyRun run;
	run.direction = direction;
	run.bytes = bytes;
	run.mode = mode;
	run.settings = std::move(settings);
	run.gbps = gbps;
	run.seconds = static_cast<double>(bytes) / (gbps * 1e9);
	run.status = "complete";
	run.verified = true;
	run.coverage = coverage;
	run.execution = mode == Mode::Driver						 ? "bare_driver"
					: std::string_view(coverage) == "instant" ? "direct_engine"
															 : "hooked_copy";
	return run;
}

// The hook's accepted-call observations that ReadCopyRuns requires of a completed hooked copy.
inline std::string Audits(const CopyRun& run)
{
	if (run.execution != "hooked_copy" || run.status != "complete")
	{
		return "";
	}
	auto text = fmt::format(R"(,"tune_audit_valid":1,"tune_audit_measured_rejected":0,"tune_audit_measured_errors":0,)"
							R"("tune_audit_measured_mixed":0,"tune_audit_measured_calls":1,)"
							R"("tune_audit_measured_accepted":1,"tune_audit_measured_bytes":{})",
							run.bytes);
	if (run.mode == Mode::Chunks)
	{
		return text + fmt::format(R"(,"tune_audit_write_back":0,"tune_audit_flush_bytes":{},)"
								  R"("tune_audit_requested_workers":{},"tune_audit_slots":{},"tune_audit_slot_bytes":{},)"
								  R"("tune_audit_wc_slots":{},"tune_audit_bench_dma":0,"tune_audit_pace_gbps":0)",
								  CopyCtx{}.flush_batch, run.settings.workers, run.settings.slots,
								  run.settings.slot_bytes, run.settings.wc_slots ? 1 : 0);
	}
	const PinningSettings& pinning = run.pinning;
	return text + fmt::format(
					  R"(,"tune_audit_version":3,"tune_audit_row_mode":{4},"tune_audit_row_min_bytes":0,)"
					  R"("tune_audit_row_max_bytes":0,"tune_audit_pinning_completed":1,)"
					  R"("tune_audit_pinning_registered_bytes":{0},"tune_audit_pinning_released_bytes":{0},)"
					  R"("tune_audit_pinning_registered_dma_bytes":{0},"tune_audit_pinning_pageable_dma_bytes":0,)"
					  R"("tune_audit_pinning_borrowed_bytes":0,"tune_audit_pinning_chunks":1,)"
					  R"("tune_audit_pinning_first_chunk_bytes":{1},"tune_audit_pinning_max_chunk_bytes":{2},)"
					  R"("tune_audit_pinning_max_registered_bytes":{3},"tune_audit_pinning_read_only":0)",
					  run.bytes, pinning.first_chunk_bytes, pinning.max_chunk_bytes,
					  pinning.max_registered_bytes.value_or(0), // the hook's observation of no limit: 0
					  static_cast<unsigned>(Mode::ProgressivePinning));
}

// A published result without a selection line, as written before the line existed. Pinning runs record the
// request's `limit`, as `cmt tune` writes them.
inline std::string Document(std::vector<CopyRun> runs, const std::string& machine = kMachine,
							Limit limit = Limit::OldDefault, const std::string& request = kRequest)
{
	const bool pinning = std::any_of(runs.begin(), runs.end(),
									 [](const CopyRun& run) { return run.mode == Mode::ProgressivePinning; });
	for (auto& run : runs)
	{
		run.pinning.max_registered_bytes = run.mode == Mode::ProgressivePinning ? Recorded(limit) : std::nullopt;
	}
	std::string text = R"({"schema":"cmt.tuning","version":1,"state":"complete","selection_policy":")" +
					   std::string(kSelectionPolicyVersion) + R"(","request":)" + request +
					   (pinning ? PinningRequest(limit) : "") + R"(},"source_machine":)" + machine + R"(,"runs":[)";
	for (std::size_t i = 0; i < std::size(runs); i++)
	{
		JsonWriter record;
		WriteCopyRun(record, runs[i]);
		auto json = check::Require(FinishJson(record));
		json.resize(json.rfind('}'));
		if (runs[i].status == "complete")
		{
			// Two equal samples: enough for a measured comparison, and the first is the median.
			const auto sample = fmt::format(R"({{"seconds":{0},"copies_per_sample":1,"elapsed_seconds":{0}}})",
											runs[i].seconds);
			json += fmt::format(R"(,"sample_count":2,"median_sample_index":0,"samples":[{0},{0}])", sample);
		}
		text += (i == 0 ? "" : ",") + json + Audits(runs[i]) + "}";
	}
	return text + "]}\n";
}

} // namespace tuning_fixture
