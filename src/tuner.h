#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include "common/helpers.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "measure/metrics.h"
#include "config.h"
#include "runner.h"

namespace cmt {
namespace tune {
constexpr int kMaxProcessRounds = 20;
constexpr std::size_t kMaxStagingMiB = 32 * 1024;
constexpr std::size_t kMinRegistrationMiB = 4;

enum class Search : std::uint8_t
{
	Instant,
	Quick,
	Full,
	FullWithKernel
};
const char* SearchName(Search search);
bool IsFullSearch(Search search);

struct Grid
{
	std::vector<std::uint64_t> sizes;
	std::vector<int> threads;
	std::vector<int> buffers;
	std::vector<std::size_t> buffer_bytes;
	std::vector<std::string> loops;
	std::vector<bool> wc;
	unsigned samples = 2;
	unsigned rounds = 1;
	unsigned tie_percent = 0;
};

Result<Grid> ParseTuningParams(Search search, std::string_view text);
std::string FormatGrid(Search search, const Grid& grid);

int ThreadCap(std::size_t allowed_cpus);

inline constexpr const char* kInheritedVariables[] = {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
													  "LD_LIBRARY_PATH"};
std::vector<std::string> InheritedEnvironment();

struct ChildPlacement
{
	std::vector<unsigned> cpus;
	std::string decision;
	std::string detail;
};

Result<ChildPlacement> MeasurementPlacement(const std::vector<std::string>& env,
											const std::optional<std::string>& gpu_pci, const GpuCpus& nvml,
											std::string_view sysfs = "/sys");

int EstimateThreads(double cpu_gbps, double pinned_gbps, int max_threads);

struct Walk
{
	std::vector<StagingSettings> visited;
	std::optional<StagingSettings> next;
};
using SettingScore = std::function<std::optional<double>(const StagingSettings&)>;
using AdmitSetting = std::function<std::optional<StagingSettings>(StagingSettings)>;
Walk WalkSettings(Search search, Dir direction, int threads, bool wc, const SettingScore& score,
				  const AdmitSetting& admit);

using ScoreSetting = std::function<std::optional<double>(const StagingSettings&, bool failed)>;
using MeasureSetting = std::function<Result<int>(const StagingSettings&)>;
Result<int> RunWalk(Search search, Dir direction, int threads, bool wc, const AdmitSetting& admit,
					const ScoreSetting& score, const MeasureSetting& measure, std::vector<StagingSettings>& counted);
} // namespace tune

struct TuneArgs
{
	std::string hook;
	std::string executable;
	std::string benchmark_exe;
	std::string out_dir;
	std::vector<std::string> command;
	bool measurement_options = false;
	tune::Search search = tune::Search::Quick;
	std::string tuning_params;
	tune::Grid grid;
	std::vector<Dir> dirs{Dir::HtoD, Dir::DtoH};
	Pages pages = Pages::Small;
	std::size_t min_bytes = 1;
	std::size_t max_pinned_mib = 1024;
	std::optional<std::size_t> max_registered_mib;
	std::size_t max_processes = 4096;
	std::vector<std::string> env;
	tune::ChildPlacement placement;
	std::string identity_salt;
	int device = 0;
	bool host_alloc = true;
	bool dry_run = false;
	bool preview = false;
	bool resume = false;
	bool progress_bar = true;
	bool verbose = false;
	bool debug = false;
	std::string help;
};

namespace tune {
struct HostAllocPlan
{
	std::optional<std::string> skipped;
	std::uint64_t bytes = 0;
	unsigned rounds = 0;
	unsigned processes = 0;
	std::optional<std::string> thp;
	bool private_memory = false;
};
HostAllocPlan PlanHostAllocation(const TuneArgs& options, const std::optional<std::string>& thp_enabled);
} // namespace tune

struct BenchmarkRun
{
	std::string name;
	CopyRun copy;
	std::vector<std::uint64_t> sizes;
	unsigned samples = 0;
};

Result<tune::ChildPlacement> PlacementFor(const TuneArgs& options, Json machine,
										  std::string_view sysfs = "/sys");
ProcessCommand BenchmarkCommand(const TuneArgs& options, const BenchmarkRun& run, const std::string& request,
								const std::string& result, const std::string& prefix);
Result<std::string> RequestJson(const TuneArgs& options);

Result<std::vector<BenchmarkRun>> MakeRuns(const TuneArgs& options);
namespace tune {
Result<std::map<std::string, std::string>> ResumeIdentity(const TuneArgs& options, Json machine);
Result<std::string> SavedSalt(const std::string& tune_json);
} // namespace tune
Result<int> RunTuning(const TuneArgs& options);

namespace tune {
struct Estimate
{
	std::map<Dir, double> cpu_gbps;
	std::map<Dir, double> pinned_gbps;
	std::string cpu_copies;
	std::string pinned_copies;
	int max_threads = 1;
};
Result<Estimate> ReadEstimate(const TuneArgs& options, Json cpu_copies, Json pinned_copies);
Error ProbeFailure(int exit_code);

std::optional<StagingSettings> Admit(const TuneArgs& options, const CpuIsas& cpu, int max_threads,
									 StagingSettings settings);
std::optional<double> Score(const TuneArgs& options, const std::vector<CopyRun>& runs, Dir direction,
							const StagingSettings& settings, bool failed);
} // namespace tune

} // namespace cmt
