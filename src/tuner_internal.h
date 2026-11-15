#pragma once
#include "tuner.h"

namespace cmt::tune_detail {
inline constexpr std::size_t kKiB = 1024;
inline constexpr std::size_t kMiB = 1024 * kKiB;
inline constexpr double kEstimateHeadroom = 1.3;
inline constexpr std::size_t kPinnedProbeBytes = 64 * kMiB;
inline constexpr std::size_t kQuickSettings =
	31;
inline constexpr std::size_t kInstantSettings = 2;

struct TuningData
{
	Pages pages = Pages::Small;
	std::string request;
	std::string source_machine;
	std::string out_dir;
	std::map<std::string, std::string> identity;
	std::vector<CopyRun> runs;
	std::vector<std::string> raw_runs;
	std::vector<std::string> attempts;
	std::size_t next_artifact_id = 0;
	std::vector<std::string> previous_invocations;
	Clock::time_point invocation_start = Clock::now();
	double metadata_child_wall = 0;
	double probe_child_wall = 0;
	double measured_children_wall = 0;
	std::string metadata_timings = "{}";
	std::string probe_timings = "{}";
	std::string state = "partial";
	std::string matrix_state = "pending";
	std::optional<tune::Estimate> estimate;
	std::optional<tuning::HostAllocation> host_allocation;
	std::string placement_note;
};

Result<void> ParseDocument(JsonParser& parser, const std::string& text, Json& value,
						   const std::string& source_path);
Result<void> AddRuns(TuningData& data, Json document, const BenchmarkRun* expected,
					 unsigned expected_samples = 0);
Result<std::string> ChildTimings(Json document, const std::string& source_path);
Result<void> RecordAttempt(TuningData& data, const ProcessCommand& command, const ProcessResult& result,
						   const std::vector<std::string>& artifacts);
Result<void> SaveTuning(const std::string& directory, const TuningData& data);
bool SameSettings(const CopyRun& a, const CopyRun& b);

std::string ShortSize(std::uint64_t bytes);
bool FitsPinnedBudget(const TuneArgs& options, const StagingSettings& settings);

template <typename T, typename Format>
std::string JoinList(const std::vector<T>& values, Format format)
{
	std::string text;
	for (const auto& value : values)
	{
		text += (text.empty() ? "" : ",") + format(value);
	}
	return text;
}

} // namespace cmt::tune_detail
