#include "cuda/kernel_status.h"
#include "tuner_internal.h"
#include "share.h"

#include <indicators/progress_bar.hpp>
#include <iostream>

namespace cmt {
using namespace tune_detail;

namespace {
constexpr std::size_t kProgressBarWidth = 16;
constexpr std::size_t kEstimatedRunBytes = 4 * kKiB;
constexpr std::size_t kEstimatedSampleBytes = 512;
constexpr std::size_t kEstimatedMetadataBytes = 2 * kMiB;
constexpr unsigned kProbeSamples = 3;
constexpr unsigned kDriverSampleFactor = 3;

using tune::Search;

struct Layout
{
	std::string name;
	StagingSettings settings;
};

Layout MakeLayout(int workers, int slots, std::size_t chunk_bytes, const std::string& loop, bool wc)
{
	Layout layout;
	layout.name = fmt::format("w{}_s{}_c{}_{}_{}", workers, slots, chunk_bytes, loop, wc ? "wc" : "cached");
	layout.settings = {workers, slots, chunk_bytes, loop, wc};
	return layout;
}

std::vector<Layout> Layouts(const TuneArgs& options)
{
	std::vector<Layout> layouts;
	for (bool wc : options.grid.wc)
	{
		for (int threads : options.grid.threads)
		{
			for (int buffers : options.grid.buffers)
			{
				for (std::size_t bytes : options.grid.buffer_bytes)
				{
					for (const auto& loop : options.grid.loops)
					{
						layouts.push_back(MakeLayout(threads, buffers, bytes, loop, wc));
					}
				}
			}
		}
	}
	return layouts;
}

PinningSettings RequestedPinning(const TuneArgs& options)
{
	PinningSettings settings;
	if (options.max_registered_mib)
	{
		settings.max_registered_bytes = *options.max_registered_mib * kMiB;
		settings.max_chunk_bytes = std::min(settings.max_chunk_bytes, *settings.max_registered_bytes);
	}
	return settings;
}

BenchmarkRun ProcessFor(const TuneArgs& options, Dir direction, const std::optional<StagingSettings>& staged,
						unsigned round)
{
	BenchmarkRun run;
	run.name = "driver";
	run.copy.direction = direction;
	run.copy.process_round = round;
	run.sizes = options.grid.sizes;
	run.samples =
		options.grid.samples * (staged || tune::IsFullSearch(options.search) ? 1 : kDriverSampleFactor);
	if (staged)
	{
		run.name =
			MakeLayout(staged->workers, staged->slots, staged->slot_bytes, staged->loop, staged->wc_slots)
				.name;
		run.copy.mode = Mode::Chunks;
		run.copy.settings = *staged;
	}
	return run;
}

} // namespace

Result<std::string> RequestJson(const TuneArgs& options)
{
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "search", tune::SearchName(options.search), true);
	JsonField(writer, "tuning_params", options.tuning_params);
	JsonField(writer, "grid", tune::FormatGrid(options.search, options.grid));
	JsonField(writer, "pages", options.pages == Pages::Thp ? "thp" : "small");
	JsonField(writer, "process_rounds", options.grid.rounds);
	if (options.grid.tie_percent > 0)
	{
		JsonField(writer, "tie_percent", options.grid.tie_percent);
	}
	JsonField(writer, "samples", options.grid.samples);
	JsonField(writer, "min_bytes", options.min_bytes);
	JsonField(writer, "max_pinned_mib", options.max_pinned_mib);
	const bool pinning = options.search == Search::FullWithKernel;
	JsonField(writer, "progressive_pinning", pinning);
	JsonField(writer, "max_registered_mib", options.max_registered_mib);
	JsonField(writer, kRegistrationLimitKey, kExplicitOnly);
	if (pinning)
	{
		JsonKey(writer, "progressive_pinning_settings");
		WriteSettings(writer, Mode::ProgressivePinning, {}, RequestedPinning(options));
	}
	JsonField(writer, "max_processes", options.max_processes);
	JsonField(writer, "device", options.device);
	JsonKey(writer, "directions");
	writer.start_array();
	bool first = true;
	for (const auto direction : options.dirs)
	{
		if (!first)
		{
			writer.append_comma();
		}
		first = false;
		writer.escape_and_append_with_quotes(DirName(direction));
	}
	writer.end_array();
	JsonKey(writer, "environment");
	WriteJsonStringArray(writer, share::ShareableEnvironment(options.env));
	writer.end_object();
	return FinishJson(writer);
}

namespace {

Result<std::string> ProbeRequest(const TuneArgs& options, bool estimate)
{
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "schema", "cmt.probe", true);
	JsonField(writer, "version", 2);
	JsonField(writer, "device", options.device);
	JsonField(writer, "metadata_only", !estimate);
	JsonField(writer, "pages", options.pages == Pages::Thp ? "thp" : "small");
	JsonField(writer, "samples", kProbeSamples);
	JsonField(writer, "pinned_copy_bytes", kPinnedProbeBytes);
	JsonKey(writer, "directions");
	writer.start_array();
	for (std::size_t i = 0; i < std::size(options.dirs); i++)
	{
		writer.append_raw(i == 0 ? "" : ",");
		writer.escape_and_append_with_quotes(DirName(options.dirs[i]));
	}
	writer.end_array();
	JsonKey(writer, "cpu_copies");
	writer.start_array();
	for (std::size_t i = 0; estimate && i < std::size(options.dirs); i++)
	{
		const bool htod = options.dirs[i] == Dir::HtoD;
		writer.append_raw(i == 0 ? "" : ",");
		writer.start_object();
		JsonField(writer, "direction", DirName(options.dirs[i]), true);
		JsonField(writer, "loop", htod ? (NativeArch() == Arch::X86_64 ? "movsb" : "memcpy") : "nt_store");
		JsonField(writer, "bytes", htod ? 2 * kMiB : 16 * kMiB);
		writer.end_object();
	}
	writer.end_array();
	JsonKey(writer, "environment");
	WriteJsonStringArray(writer, options.env);
	writer.end_object();
	return FinishJson(writer);
}

} // namespace

Result<tune::ChildPlacement> PlacementFor(const TuneArgs& options, Json machine, std::string_view sysfs)
{
	std::string_view bus;
	std::string_view
		nvml_text;
	const bool has_bus = machine.at_pointer("/gpus/0/pci_bus_id").get(bus) == simdjson::SUCCESS;
	const bool has_nvml =
		machine.at_pointer("/gpus/0/cpu_affinity/nvml_cpus").get(nvml_text) == simdjson::SUCCESS;
	const GpuCpus recorded_nvml = [&](const std::string&) -> Result<std::vector<unsigned>>
	{
		if (!has_nvml)
		{
			return Failure(Error::Kind::Unsupported, "the machine record has no NVML CPU set");
		}
		return ParseThreadCpus(nvml_text);
	};
	return tune::MeasurementPlacement(options.env,
									  has_bus ? std::optional<std::string>{std::string(bus)} : std::nullopt,
									  recorded_nvml, sysfs);
}

Error tune::ProbeFailure(int exit_code)
{
	return exit_code == Error(Error::Kind::Unsupported, "").ExitCode()
			   ? Error(Error::Kind::Unsupported, "this machine cannot run the estimate probe", exit_code)
			   : Error(Error::Kind::Verify, "estimate probe failed", exit_code);
}

namespace {

bool FromProcess(const CopyRun& saved, const BenchmarkRun& process)
{
	return saved.coverage == "measured" && saved.direction == process.copy.direction &&
		   saved.process_round == process.copy.process_round && SameSettings(saved, process.copy) &&
		   std::find(process.sizes.begin(), process.sizes.end(), saved.bytes) != process.sizes.end();
}

bool Finished(const BenchmarkRun& process, const std::vector<CopyRun>& saved)
{
	return std::all_of(
		process.sizes.begin(), process.sizes.end(),
		[&](std::uint64_t bytes)
		{
			return std::any_of(
				saved.begin(), saved.end(), [&](const CopyRun& run)
				{ return run.bytes == bytes && run.status != "failed" && FromProcess(run, process); });
		});
}

Result<std::string> ExpectedRunJson(const BenchmarkRun& run)
{
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "schema", "cmt.benchmark-run", true);
	JsonField(writer, "version", 2);
	JsonField(writer, "direction", DirName(run.copy.direction));
	JsonKey(writer, "sizes_bytes");
	writer.start_array();
	for (std::size_t i = 0; i < std::size(run.sizes); i++)
	{
		if (i != 0)
		{
			writer.append_comma();
		}
		writer.append_raw(std::to_string(run.sizes[i]));
	}
	writer.end_array();
	JsonField(writer, "process_round", run.copy.process_round);
	JsonKey(writer, "settings");
	WriteSettings(writer, run.copy.mode, run.copy.settings, run.copy.pinning);
	writer.end_object();
	return FinishJson(writer);
}

} // namespace

ProcessCommand BenchmarkCommand(const TuneArgs& options, const BenchmarkRun& run, const std::string& request,
								const std::string& result, const std::string& prefix)
{
	ProcessCommand command;
	command.executable = options.benchmark_exe;
	command.stdout_path = prefix + ".stdout";
	command.stderr_path = prefix + ".stderr";
	command.arguments = {"--out",
						 result,
						 "--",
						 "--path",
						 "pageable",
						 "--pages",
						 PagesName(options.pages),
						 "--dir",
						 DirName(run.copy.direction),
						 "--sizes",
						 JoinList(run.sizes, [](std::uint64_t bytes) { return std::to_string(bytes); }),
						 "--samples",
						 std::to_string(run.samples),
						 "--min-bytes",
						 std::to_string(options.min_bytes),
						 "--device",
						 std::to_string(options.device)};
	command.environment = options.env;
	command.inherit_environment = false;
	command.cpus = options.placement.cpus;
	command.environment.push_back("CMT_EXPECTED_RUN=" + request);
	if (run.copy.mode != Mode::Driver)
	{
		command.environment.push_back("LD_PRELOAD=" + options.hook);
		command.environment.push_back("CMT_TUNE_AUDIT=1");
	}
	if (run.copy.mode == Mode::Chunks)
	{
		command.environment.push_back("STAGING_PIPELINE_WORKERS=" +
									  std::to_string(run.copy.settings.workers));
		command.environment.push_back("STAGING_THREADS_COPY_NUM_BUFFERS=" +
									  std::to_string(run.copy.settings.slots));
		command.environment.push_back("STAGING_BUFFER_SIZE=" +
									  std::to_string(run.copy.settings.slot_bytes / 1024));
		command.environment.push_back("STAGING_COPY_NT_IMPL_H2D=" + run.copy.settings.loop);
		command.environment.push_back("STAGING_COPY_NT_IMPL_D2H=" + run.copy.settings.loop);
		command.environment.push_back(std::string("STAGING_SLOT_ALLOC=") +
									  (run.copy.settings.wc_slots ? "hostalloc_wc" : "hostalloc"));
	}
	return command;
}

namespace {

struct SearchCost
{
	std::vector<BenchmarkRun> round;
	std::map<Dir, std::size_t> staged;
	std::set<std::string> over_budget;
	std::vector<std::uint64_t> pinning_sizes;
	std::size_t processes = 0;
	std::size_t estimated_bytes = 0;
};

Result<SearchCost> CountRuns(const TuneArgs& options)
{
	const std::set<Dir> directions(options.dirs.begin(), options.dirs.end());
	if (options.pages != Pages::Small && options.pages != Pages::Thp)
	{
		return UsageError("tuning caller pages must be small or thp");
	}
	if (options.max_pinned_mib < 1 || options.max_pinned_mib > tune::kMaxStagingMiB ||
		options.max_processes < 1 || options.dirs.empty() || directions.size() != options.dirs.size() ||
		options.grid.sizes.empty() || options.grid.wc.empty() || options.grid.rounds < 1 ||
		options.grid.samples < 1)
	{
		return UsageError("invalid tuning search limits");
	}
	if (options.max_registered_mib && (*options.max_registered_mib < tune::kMinRegistrationMiB ||
									   *options.max_registered_mib > kMaxBytes / kMiB))
	{
		return UsageError("--max-registered-mib must be at least 4 MiB and within the supported byte limit");
	}
	SearchCost cost;
	if (options.search == Search::FullWithKernel)
	{
		const auto pinning = RequestedPinning(options);
		for (const auto bytes : options.grid.sizes)
		{
			if (bytes >= pinning.first_chunk_bytes &&
				(!pinning.max_registered_bytes || bytes <= *pinning.max_registered_bytes))
			{
				cost.pinning_sizes.push_back(bytes);
			}
		}
		if (cost.pinning_sizes.empty())
		{
			return UsageError("progressive pinning requires a size of at least 4 MiB (and at most "
							  "--max-registered-mib when given)");
		}
	}
	const auto& grid = options.grid;
	if (tune::IsFullSearch(options.search) && grid.threads.size() * grid.buffers.size() *
													  grid.buffer_bytes.size() * grid.loops.size() *
													  grid.wc.size() >
												  options.max_processes)
	{
		return UsageError("search exceeds --max-processes; use --preview and narrow --tuning-params");
	}
	for (Dir direction : options.dirs)
	{
		BenchmarkRun process = ProcessFor(options, direction, std::nullopt, 1);
		cost.round.push_back(process);
		if (!tune::IsFullSearch(options.search))
		{
			cost.staged[direction] =
				(options.search == Search::Instant ? kInstantSettings : kQuickSettings) * grid.wc.size();
			continue;
		}
		for (const auto& layout : Layouts(options))
		{
			if (!FitsPinnedBudget(options, layout.settings))
			{
				cost.over_budget.insert(fmt::format("{} threads x {} buffers x {}", layout.settings.workers,
													layout.settings.slots,
													ShortSize(layout.settings.slot_bytes)));
				continue;
			}
			process.name = layout.name;
			process.samples = grid.samples;
			process.copy.mode = Mode::Chunks;
			process.copy.settings = layout.settings;
			cost.round.push_back(process);
			cost.staged[direction]++;
		}
		if (!cost.pinning_sizes.empty())
		{
			process.name = "progressive_pinning";
			process.copy.mode = Mode::ProgressivePinning;
			process.copy.settings = {};
			process.copy.pinning = RequestedPinning(options);
			process.sizes = cost.pinning_sizes;
			cost.round.push_back(process);
		}
	}
	cost.processes = cost.round.size() * grid.rounds;
	for (const auto& [direction, staged] : cost.staged)
	{
		cost.processes += tune::IsFullSearch(options.search) ? 0 : staged * grid.rounds;
	}
	cost.estimated_bytes =
		kEstimatedMetadataBytes +
		cost.processes * grid.sizes.size() * (kEstimatedRunBytes + grid.samples * kEstimatedSampleBytes);
	return cost;
}

void PrintCost(const TuneArgs& options, const SearchCost& cost)
{
	const char* search = tune::SearchName(options.search);
	const bool searched = !tune::IsFullSearch(options.search);
	if (searched)
	{
		std::fprintf(
			stdout,
			"%s: first one probe process times one core's CPU copy and a 64 MiB pinned cudaMemcpy per "
			"direction; the copying threads are the fewest of 1, 2, 4, 8, 16 whose summed CPU rate "
			"reaches 1.3 x the pinned rate. %s\n",
			search,
			options.search == Search::Instant
				? "Two staged settings per direction at those threads are measured."
				: "Instant's two staged settings and six more per direction are measured, then a "
				  "coordinate descent (threads, buffers x buffer size, loop) steps while a step is more "
				  "than 2% faster at 16 MiB and up.");
	}
	std::fprintf(stdout,
				 "%s: the stock driver against the real LD_PRELOAD hook. Each process is fresh, runs one "
				 "setting at every size, warms up untimed before each sample and evicts the caller buffer "
				 "from the CPU caches (no sudo) before each timed sample.\n"
				 "Grid (change it with --tuning-params; defaults in --help-advanced):\n%s",
				 search, tune::FormatGrid(options.search, options.grid).c_str());
	std::fprintf(stdout,
				 options.min_bytes <= 1
					 ? "Caller pages: %s. Each timed sample is one cold copy.\n"
					 : "Caller pages: %s. A timed sample repeats a copy until it moves %s; "
					   "only its first copy is cold.\n",
				 options.pages == Pages::Thp ? "thp" : "small", HumanSize(options.min_bytes).c_str());
	for (const auto& over : cost.over_budget)
	{
		std::fprintf(stdout, "Skipped over --max-pinned-mib %zu: %s (every loop and memory)\n",
					 options.max_pinned_mib, over.c_str());
	}
	std::size_t per_direction = 0;
	for (Dir direction : options.dirs)
	{
		const auto staged = cost.staged.count(direction) ? cost.staged.at(direction) : 0;
		per_direction = 1 + staged + (cost.pinning_sizes.empty() ? 0 : 1);
		std::fprintf(stdout, "%s: 1 driver + %s%zu staged settings%s = %s%zu processes per round.\n",
					 direction == Dir::HtoD ? "cpu-to-gpu" : "gpu-to-cpu", searched ? "at most " : "", staged,
					 cost.pinning_sizes.empty() ? "" : " + 1 progressive pinning", searched ? "at most " : "",
					 per_direction);
	}
	if (!cost.pinning_sizes.empty())
	{
		const auto pinning = RequestedPinning(options);
		const auto limit =
			options.max_registered_mib
				? fmt::format("limit {} MiB, --max-registered-mib", *options.max_registered_mib)
				: std::string("no limit set: up to the copy's size, within the GPU's memory");
		std::fprintf(stdout,
					 "Progressive pinning registers %s..%s chunks during a copy (%s) and releases "
					 "them at copy end; sizes %s.\n",
					 HumanSize(pinning.first_chunk_bytes).c_str(), HumanSize(pinning.max_chunk_bytes).c_str(),
					 limit.c_str(), JoinList(cost.pinning_sizes, ShortSize).c_str());
	}
	std::fprintf(stdout,
				 "Processes: %s%zu (rounds=%zu; limit --max-processes %zu), plus 1 metadata child%s. "
				 "Estimated output %.1f MiB.\n",
				 searched ? "at most " : "", cost.processes, static_cast<std::size_t>(options.grid.rounds),
				 options.max_processes, searched ? " and 1 probe" : "",
				 static_cast<double>(cost.estimated_bytes) / kMiB);
	auto thp = ReadTextFile("/sys/kernel/mm/transparent_hugepage/enabled", kKiB);
	const auto plan =
		tune::PlanHostAllocation(options, thp ? std::optional<std::string>(*thp) : std::nullopt);
	if (plan.skipped)
	{
		std::fprintf(stdout, "Pinned allocation: not measured (%s).\n", plan.skipped->c_str());
	}
	else
	{
		const std::string private_arm =
			plan.private_memory ? ", its private memory (THP " + *plan.thp + "; measured, never chosen)" : "";
		std::fprintf(
			stdout,
			"Pinned allocation: %u processes per arm time one %s cudaHostAlloc and cudaFreeHost each: the "
			"driver, the hook's shared memory%s (--host-alloc off skips it).\n",
			plan.rounds * plan.processes, HumanSize(plan.bytes).c_str(), private_arm.c_str());
	}
}

std::vector<BenchmarkRun> ExpandRuns(const TuneArgs& options, const std::vector<BenchmarkRun>& round_runs)
{
	std::vector<BenchmarkRun> runs;
	for (unsigned round = 1; round <= options.grid.rounds; round++)
	{
		for (const auto& selected : round_runs)
		{
			BenchmarkRun run = selected;
			run.copy.process_round = round;
			runs.push_back(std::move(run));
		}
	}
	return runs;
}

Result<std::vector<BenchmarkRun>> CheckedRuns(const TuneArgs& options, const SearchCost& cost)
{
	if (cost.processes > options.max_processes)
	{
		return UsageError("search exceeds --max-processes; use --preview and narrow --tuning-params");
	}
	if (cost.estimated_bytes > tuning::kMaxConfigBytes)
	{
		return UsageError("estimated search output exceeds 256 MiB; narrow --tuning-params");
	}
	return ExpandRuns(options, cost.round);
}

} // namespace

Result<std::vector<BenchmarkRun>> MakeRuns(const TuneArgs& options)
{
	auto cost = CountRuns(options);
	if (!cost)
	{
		return Failure(cost.error());
	}
	return CheckedRuns(options, *cost);
}

namespace {

struct OutputOwner
{
	int fd = -1;

	~OutputOwner()
	{
		if (fd >= 0)
		{
			close(fd);
		}
	}
};

Result<void> CheckKernelSearch()
{
	const auto status = nv::ReadHostRegistrationStatus();
	if (!status.active || status.large_system_pages != true)
	{
		std::fprintf(stderr,
					 "cmt tune: full_with_kernel needs the verified host-registration modules (A+B+D) "
					 "loaded: %s; large-page mapping opt-in %s. Load them with cmt patch first.\n",
					 status.evidence, status.large_system_pages == true ? "enabled" : "unverified or off");
		return Failure(
			Error::Kind::Unsupported,
			"full_with_kernel requires loaded host-registration modules; tune does not change modules");
	}
	auto thp = ReadTextFile("/sys/kernel/mm/transparent_hugepage/enabled", kKiB);
	if (!thp || thp->find("[never]") != std::string::npos)
	{
		std::fprintf(stderr, "cmt tune: full_with_kernel needs transparent huge pages not [never] in "
							 "/sys/kernel/mm/transparent_hugepage/enabled\n");
		return Failure(Error::Kind::Unsupported, "full_with_kernel requires THP available");
	}
	return {};
}

Result<bool> PrepareSearch(TuneArgs& options, std::vector<BenchmarkRun>& planned)
{
	auto self = SelfExe();
	if (!self)
	{
		return Failure(self.error());
	}
	options.executable = *self;
	if (options.benchmark_exe.empty())
	{
		options.benchmark_exe = (std::filesystem::path(*self).parent_path() / "gpu_copy").string();
	}
	for (auto& entry : tune::InheritedEnvironment())
	{
		options.env.push_back(std::move(entry));
	}
	auto cost = CountRuns(options);
	if (!cost)
	{
		return Failure(cost.error());
	}
	if (options.verbose || options.preview || options.dry_run)
	{
		PrintCost(options, *cost);
	}
	if (options.preview)
	{
		return false;
	}
	auto generated = CheckedRuns(options, *cost);
	if (!generated)
	{
		return Failure(generated.error());
	}
	if (options.dry_run)
	{
		for (const auto& run : *generated)
		{
			std::fprintf(stdout, "%s %s %zu sizes round %u\n", run.name.c_str(), DirName(run.copy.direction),
						 run.sizes.size(), run.copy.process_round);
		}
		return false;
	}
	if (options.search == Search::FullWithKernel)
	{
		CMT_RETURN_IF_ERROR(CheckKernelSearch());
	}
	auto hook = ResolvePreloadLibrary(options.hook);
	if (!hook)
	{
		return Failure(hook.error());
	}
	options.hook = std::move(*hook);
	planned = std::move(*generated);
	return true;
}

Result<void> ChooseIdentitySalt(TuneArgs& options)
{
	auto existing = options.resume ? ReadTextFile(options.out_dir + "/tune.json", tuning::kMaxConfigBytes)
								   : Result<std::string>();
	if (!options.resume || (!existing && existing.error().GetCode() == ENOENT))
	{
		auto salt = share::NewSalt();
		if (!salt)
		{
			return Failure(salt.error());
		}
		options.identity_salt = std::move(*salt);
		return {};
	}
	if (!existing)
	{
		return Failure(existing.error());
	}
	auto salt = tune::SavedSalt(*existing);
	if (!salt)
	{
		std::fprintf(stderr,
					 "cmt tune: %s/tune.json was written by an older cmt (its identity records the GPU UUID "
					 "and PCI addresses); it cannot be resumed, start a new --out\n",
					 options.out_dir.c_str());
		return Failure(salt.error());
	}
	options.identity_salt = std::move(*salt);
	return {};
}

Result<TuningData> CollectIdentity(TuneArgs& options, Clock::time_point invocation_start,
								   const MeasurementLock& lock)
{
	auto request = RequestJson(options);
	if (!request)
	{
		return Failure(request.error());
	}
	const char* tmpdir = std::getenv("TMPDIR");
	std::string temporary = std::string(tmpdir && tmpdir[0] == '/' ? tmpdir : "/tmp") + "/cmt-tune-XXXXXX";
	if (!mkdtemp(temporary.data()))
	{
		return OsError("create tuning preparation directory");
	}

	struct Temporary
	{
		std::string path;
		bool keep = false;

		~Temporary()
		{
			if (!keep)
			{
				std::error_code error;
				std::filesystem::remove_all(path, error);
			}
		}
	} preparation{temporary};

	const std::string probe_request = preparation.path + "/probe-request.json";
	const std::string probe_result = preparation.path + "/probe-result.json";
	auto metadata_request = ProbeRequest(options, false);
	if (!metadata_request)
	{
		return Failure(metadata_request.error());
	}
	CMT_RETURN_IF_ERROR(WriteTextFile(probe_request, *metadata_request));
	ProcessCommand probe;
	probe.executable = options.executable;
	probe.arguments = {"--tune-probe", probe_request, "--out", probe_result};
	probe.environment = options.env;
	probe.inherit_environment = false;
	lock.Apply(probe);
	probe.stderr_path = preparation.path + "/probe.stderr";
	const auto metadata_start = Clock::now();
	auto process = RunProcess(probe);
	const double metadata_child_wall = std::chrono::duration<double>(Clock::now() - metadata_start).count();
	if (!process)
	{
		return Failure(process.error());
	}
	if (process->interrupted_by_signal != 0)
	{
		preparation.keep = true;
		std::fprintf(stderr,
					 "cmt tune: interrupted metadata child exited %d (signal %d); evidence retained in %s\n",
					 process->exit_code, process->signal, preparation.path.c_str());
		return Failure(Error::Kind::Interrupted, "machine collection interrupted",
					   process->interrupted_by_signal);
	}
	if (!process->completed || process->exit_code != 0)
	{
		auto diagnostic = ReadTextFile(probe.stderr_path);
		if (diagnostic)
		{
			if (std::fwrite(diagnostic->data(), 1, diagnostic->size(), stderr) != diagnostic->size())
			{
				std::fprintf(stderr, "cmt tune: could not forward the metadata child's stderr\n");
			}
			if (!diagnostic->empty() && diagnostic->back() != '\n')
			{
				std::fputc('\n', stderr);
			}
		}
		else
		{
			std::fprintf(stderr, "cmt tune: could not read %s: %s (code %d)\n", probe.stderr_path.c_str(),
						 diagnostic.error().Message(), diagnostic.error().GetCode());
		}
		if (process->signal != 0)
		{
			return Failure(Error::Kind::Interrupted, "machine collection terminated by signal",
						   process->signal);
		}
		if (process->exit_code == 77 || process->exit_code == Error(Error::Kind::Unsupported, "").ExitCode())
		{
			return Failure(Error::Kind::Unsupported, "machine collection found no usable GPU",
						   process->exit_code);
		}
		return Failure(Error::Kind::Verify, "machine collection failed before tuning", process->exit_code);
	}
	auto metadata = ReadTextFile(probe_result, tuning::kMaxConfigBytes);
	if (!metadata)
	{
		return Failure(metadata.error());
	}
	JsonParser metadata_parser;
	Json metadata_document;
	CMT_RETURN_IF_ERROR(ParseDocument(metadata_parser, *metadata, metadata_document, probe_result));
	Json machine;
	if (metadata_document["source_machine"].get(machine) != simdjson::SUCCESS)
	{
		return UsageError("probe omitted source_machine");
	}
	std::string_view registration_status;
	bool pinning_available = false;
	if (!machine["driver"]["host_registration"]["status"].get(registration_status) &&
		registration_status == "active")
	{
		bool large_pages = false;
		const auto large_page_error =
			machine["driver"]["host_registration"]["large_system_pages_opt_in"].get(large_pages);
		pinning_available = large_page_error == simdjson::SUCCESS && large_pages;
		if (options.verbose)
		{
			std::fprintf(stdout,
						 "Kernel: host-memory registration optimizations active (batched pin/unpin; "
						 "large GPU mappings %s, actual mapping size not measured).\n",
						 large_page_error ? "unknown"
						 : large_pages	  ? "enabled for eligible memory"
										  : "disabled");
		}
	}
	else if (options.verbose)
	{
		std::fprintf(stdout,
					 "Kernel: no verified active host-registration bundle; tuning the loaded driver.\n");
	}
	if (options.search == Search::FullWithKernel && !pinning_available)
	{
		return Failure(Error::Kind::Unsupported,
					   "machine collection did not verify the required progressive-pinning kernel support");
	}
	auto placement = PlacementFor(options, machine);
	if (!placement)
	{
		return Failure(placement.error());
	}
	if (options.verbose)
	{
		std::fprintf(stdout, "Measurement processes: %s (%s).\n", placement->decision.c_str(),
					 placement->detail.c_str());
	}
	options.placement = std::move(*placement);
	auto identity = tune::ResumeIdentity(options, machine);
	if (!identity)
	{
		return Failure(identity.error());
	}
	TuningData data;
	std::string_view bus;
	data.placement_note = options.placement.decision + " (" +
						  share::WithoutValue(options.placement.detail,
											  machine.at_pointer("/gpus/0/pci_bus_id").get(bus) ? "" : bus) +
						  ")";
	data.pages = options.pages;
	data.invocation_start = invocation_start;
	data.metadata_child_wall = metadata_child_wall;
	auto metadata_timings = ChildTimings(metadata_document, probe_result);
	if (!metadata_timings)
	{
		return Failure(metadata_timings.error());
	}
	data.metadata_timings = *metadata_timings;
	auto shareable = share::ShareableMachine(machine);
	if (!shareable)
	{
		return Failure(shareable.error());
	}
	data.request = *request;
	data.source_machine = std::move(*shareable);
	data.out_dir = options.out_dir;
	data.identity = *identity;
	return data;
}

Result<void> OpenOutput(const TuneArgs& options, TuningData& data, OutputOwner& owner)
{
	std::error_code error;
	const bool exists = std::filesystem::exists(options.out_dir, error);
	if (error)
	{
		return IoFailure("inspect tuning output", error);
	}
	if (!exists && options.resume)
	{
		return UsageError("cannot resume a missing output directory");
	}
	if (!exists)
	{
		CMT_RETURN_IF_ERROR(MakeDirectories(options.out_dir));
	}
	const int directory_fd = open(options.out_dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	if (directory_fd < 0)
	{
		return OsError("open tuning output directory");
	}

	owner.fd = directory_fd;

	if (flock(directory_fd, LOCK_EX | LOCK_NB) != 0)
	{
		return OsError("another tuner owns this output directory");
	}
	auto existing = ReadTextFile(options.out_dir + "/tune.json", tuning::kMaxConfigBytes);
	if (existing)
	{
		if (!options.resume)
		{
			return UsageError("output already contains tuning results; choose a new directory or --resume");
		}
		JsonParser parser;
		Json document;
		CMT_RETURN_IF_ERROR(ParseDocument(parser, *existing, document, options.out_dir + "/tune.json"));
		simdjson::dom::object old_identity;
		if (document["identity"].get(old_identity) != simdjson::SUCCESS)
		{
			return UsageError("saved tuning identity is missing");
		}
		std::map<std::string, std::string> old;
		for (auto field : old_identity)
		{
			std::string_view value;
			if (field.value.get(value) != simdjson::SUCCESS ||
				!old.emplace(std::string(field.key), std::string(value)).second)
			{
				return UsageError("invalid saved tuning identity");
			}
		}
		if (old != data.identity)
		{
			for (const auto& [key, value] : data.identity)
			{
				if (!old.count(key) || old.at(key) != value)
				{
					std::fprintf(stderr, "resume refused: changed %s; existing output preserved\n",
								 key.c_str());
				}
			}
			return UsageError("resume identity changed; start new results instead");
		}
		CMT_RETURN_IF_ERROR(AddRuns(data, document, nullptr));
		std::string_view saved_matrix_state;
		if (document["matrix_state"].get(saved_matrix_state) ||
			(saved_matrix_state != "pending" && saved_matrix_state != "partial" &&
			 saved_matrix_state != "complete"))
		{
			return UsageError("saved tuning matrix state is invalid");
		}
		data.matrix_state = saved_matrix_state;
		Json cpu_copies;
		Json pinned_copies;
		if (!document.at_pointer("/estimate/cpu_copies").get(cpu_copies) &&
			!document.at_pointer("/estimate/pinned_copies").get(pinned_copies))
		{
			auto saved = tune::ReadEstimate(options, cpu_copies, pinned_copies);
			if (!saved)
			{
				return Failure(saved.error());
			}
			data.estimate = std::move(*saved);
		}
		auto host_allocation = tuning::ReadHostAllocation(document);
		if (!host_allocation)
		{
			return Failure(host_allocation.error());
		}
		if (*host_allocation && (*host_allocation)->state == "measured")
		{
			data.host_allocation = std::move(*host_allocation);
		}
		Json old_invocations;
		if (document["invocations"].get(old_invocations) == simdjson::SUCCESS)
		{
			simdjson::dom::array invocations;
			if (old_invocations.get(invocations) != simdjson::SUCCESS)
			{
				return UsageError("invalid invocation history");
			}
			for (Json invocation : invocations)
			{
				data.previous_invocations.push_back(simdjson::to_string(invocation));
			}
		}
		simdjson::dom::array attempts;
		if (document["attempts"].get(attempts) != simdjson::SUCCESS)
		{
			return UsageError("saved attempts missing");
		}
		for (Json attempt : attempts)
		{
			data.attempts.push_back(simdjson::to_string(attempt));
		}
	}
	else if (existing.error().GetCode() != ENOENT)
	{
		return Failure(existing.error());
	}
	else if (options.resume)
	{
		return UsageError("resume requires tune.json");
	}
	const auto artifacts = options.out_dir + "/artifacts";
	CMT_RETURN_IF_ERROR(MakeDirectories(artifacts));
	data.next_artifact_id = data.attempts.size();
	std::filesystem::directory_iterator entry(artifacts, error);
	const std::filesystem::directory_iterator end;
	while (!error && entry != end)
	{
		const auto name = entry->path().filename().string();
		constexpr std::string_view run_prefix = "run-";
		constexpr std::string_view probe_prefix = "probe-";
		constexpr std::string_view host_alloc_prefix = "host-alloc-";
		const auto prefix = name.rfind(run_prefix, 0) == 0			? run_prefix
							: name.rfind(probe_prefix, 0) == 0		? probe_prefix
							: name.rfind(host_alloc_prefix, 0) == 0 ? host_alloc_prefix
																	: std::string_view{};
		if (!prefix.empty())
		{
			std::size_t id = 0;
			const auto parsed = std::from_chars(name.data() + prefix.size(), name.data() + name.size(), id);
			if (parsed.ec == std::errc::result_out_of_range ||
				(parsed.ec == std::errc{} && id == std::numeric_limits<std::size_t>::max()))
			{
				return UsageError("tuning artifact number is too large");
			}
			if (parsed.ec == std::errc{})
			{
				data.next_artifact_id = std::max(data.next_artifact_id, id + 1);
			}
		}
		entry.increment(error);
	}
	if (error)
	{
		return IoFailure("scan tuning artifacts", error);
	}
	return {};
}

Result<void> EnsureEstimate(const TuneArgs& options, TuningData& data, const MeasurementLock& lock)
{
	if (data.estimate)
	{
		if (options.verbose)
		{
			std::fprintf(stderr, "estimate: reuse the saved CPU and pinned copy rates\n");
		}
		return {};
	}
	if (options.verbose)
	{
		std::fprintf(stderr, "estimate: time one core's CPU copy and a 64 MiB pinned copy per direction\n");
	}
	auto request = ProbeRequest(options, true);
	if (!request)
	{
		return Failure(request.error());
	}
	if (data.next_artifact_id == std::numeric_limits<std::size_t>::max())
	{
		return UsageError("tuning artifact numbers exhausted");
	}
	const auto prefix = options.out_dir + "/artifacts/probe-" + std::to_string(data.next_artifact_id++);
	const auto saved_request = prefix + "-request.json";
	const auto saved_result = prefix + "-result.json";
	CMT_RETURN_IF_ERROR(WriteTextFile(saved_request, *request));
	ProcessCommand probe;
	probe.executable = options.executable;
	probe.environment = options.env;
	probe.inherit_environment = false;
	probe.cpus = options.placement.cpus;
	lock.Apply(probe);
	probe.arguments = {"--tune-probe", saved_request, "--out", saved_result};
	probe.stderr_path = prefix + ".stderr";
	const auto probe_start = Clock::now();
	auto process = RunProcess(probe);
	data.probe_child_wall = std::chrono::duration<double>(Clock::now() - probe_start).count();
	if (!process)
	{
		return Failure(process.error());
	}
	CMT_RETURN_IF_ERROR(
		RecordAttempt(data, probe, *process, {saved_request, saved_result, probe.stderr_path}));
	CMT_RETURN_IF_ERROR(SaveTuning(options.out_dir, data));
	if (process->interrupted_by_signal != 0)
	{
		return Failure(Error::Kind::Interrupted, "estimate probe interrupted",
					   process->interrupted_by_signal);
	}
	if (!process->completed || process->exit_code != 0)
	{
		if (process->signal != 0)
		{
			return Failure(Error::Kind::Interrupted, "estimate probe terminated by signal", process->signal);
		}
		std::fprintf(stderr, "cmt tune: the estimate probe failed; its stderr is %s\n",
					 probe.stderr_path.c_str());
		return Failure(tune::ProbeFailure(process->exit_code));
	}
	auto result = ReadTextFile(saved_result, tuning::kMaxConfigBytes);
	if (!result)
	{
		return Failure(result.error());
	}
	JsonParser parser;
	Json document;
	CMT_RETURN_IF_ERROR(ParseDocument(parser, *result, document, saved_result));
	auto timings = ChildTimings(document, saved_result);
	if (!timings)
	{
		return Failure(timings.error());
	}
	data.probe_timings = *timings;
	Json cpu_copies;
	Json pinned_copies;
	if (document["diagnostics"].get(cpu_copies) || document["runs"].get(pinned_copies))
	{
		return Failure(Error::Kind::Verify, "estimate probe omitted its CPU or pinned copies");
	}
	auto estimate = tune::ReadEstimate(options, cpu_copies, pinned_copies);
	if (!estimate)
	{
		return Failure(estimate.error());
	}
	data.estimate = std::move(*estimate);
	return SaveTuning(options.out_dir, data);
}

struct HostAllocSample
{
	std::string served_by;
	double alloc_seconds = 0;
	double free_seconds = 0;
	std::uint64_t huge_kib = 0;
};

Result<HostAllocSample> ReadHostAllocSample(const std::string& path, std::string_view arm,
											std::uint64_t bytes)
{
	auto text = ReadTextFile(path, tuning::kMaxConfigBytes);
	if (!text)
	{
		return Failure(text.error());
	}
	JsonParser parser;
	Json document;
	CMT_RETURN_IF_ERROR(ParseDocument(parser, *text, document, path));
	HostAllocSample sample;
	std::string_view schema, recorded_arm, served_by;
	std::uint64_t version = 0;
	std::uint64_t recorded_bytes = 0;
	bool verified = false;
	if (document["schema"].get(schema) || schema != "cmt.host_alloc_result" ||
		document["version"].get(version) || version != 1 || document["arm"].get(recorded_arm) ||
		recorded_arm != arm || document["bytes"].get(recorded_bytes) || recorded_bytes != bytes ||
		document["served_by"].get(served_by) || document["alloc_seconds"].get(sample.alloc_seconds) ||
		!std::isfinite(sample.alloc_seconds) || sample.alloc_seconds < 0 ||
		document["free_seconds"].get(sample.free_seconds) || !std::isfinite(sample.free_seconds) ||
		sample.free_seconds < 0 || document["huge_kib"].get(sample.huge_kib) ||
		document["verified"].get(verified) || !verified)
	{
		return Failure(Error::Kind::Verify,
					   "the host allocation child wrote an incomplete or unverified result");
	}
	sample.served_by = served_by;
	return sample;
}

Result<int> MeasureHostAllocArm(const TuneArgs& options, TuningData& data, const MeasurementLock& lock,
								const std::string& arm, std::uint64_t bytes,
								std::optional<HostAllocSample>& sample)
{
	sample.reset();
	if (data.next_artifact_id == std::numeric_limits<std::size_t>::max())
	{
		return UsageError("tuning artifact numbers exhausted");
	}
	const auto prefix = options.out_dir + "/artifacts/host-alloc-" + std::to_string(data.next_artifact_id++);
	const auto saved_request = prefix + "-request.json";
	const auto saved_result = prefix + "-result.json";
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "schema", "cmt.host_alloc_probe", true);
	JsonField(writer, "version", 1);
	JsonField(writer, "device", options.device);
	JsonField(writer, "bytes", bytes);
	JsonField(writer, "arm", arm);
	writer.end_object();
	auto request = FinishJson(writer);
	if (!request)
	{
		return Failure(request.error());
	}
	CMT_RETURN_IF_ERROR(WriteTextFile(saved_request, *request));
	ProcessCommand child;
	child.executable = options.executable;
	child.environment = options.env;
	child.inherit_environment = false;
	child.cpus = options.placement.cpus;
	if (arm != "driver")
	{
		child.environment.push_back("LD_PRELOAD=" + options.hook);
		child.environment.push_back("CMT_CUDA_HOST_ALLOC_HOOK=1");
		if (arm == "private")
		{
			child.environment.push_back("CMT_CUDA_HOST_ALLOC_PRIVATE=1");
		}
	}
	lock.Apply(child);
	child.arguments = {"--tune-host-alloc", saved_request, "--out", saved_result};
	child.stderr_path = prefix + ".stderr";
	const auto started = Clock::now();
	auto process = RunProcess(child);
	data.measured_children_wall += std::chrono::duration<double>(Clock::now() - started).count();
	if (!process)
	{
		return Failure(process.error());
	}
	CMT_RETURN_IF_ERROR(
		RecordAttempt(data, child, *process, {saved_request, saved_result, child.stderr_path}));
	CMT_RETURN_IF_ERROR(SaveTuning(options.out_dir, data));
	if (process->interrupted_by_signal != 0)
	{
		return kSignalExitBase + process->interrupted_by_signal;
	}
	if (process->signal != 0)
	{
		return kSignalExitBase + process->signal;
	}
	auto result =
		process->completed && process->exit_code == 0
			? ReadHostAllocSample(saved_result, arm, bytes)
			: Result<HostAllocSample>(Failure(Error::Kind::Verify, "it exited nonzero", process->exit_code));
	if (!result)
	{
		std::fprintf(stderr, "cmt tune: a pinned allocation process (%s) failed: %s; its stderr is %s\n",
					 arm.c_str(), result.error().Message(), child.stderr_path.c_str());
		return 0;
	}
	if (result->served_by != arm)
	{
		std::fprintf(stderr,
					 "cmt tune: a pinned allocation process (%s) was served by the %s; its stderr is %s\n",
					 arm.c_str(), result->served_by == "driver" ? "driver (the hook fell back)" : "hook",
					 child.stderr_path.c_str());
		return 0;
	}
	sample = std::move(*result);
	return 0;
}

Result<int> MeasureHostAllocation(const TuneArgs& options, TuningData& data, const MeasurementLock& lock)
{
	if (data.host_allocation)
	{
		if (options.verbose)
		{
			std::fprintf(stderr, "pinned allocation: reuse the saved measurement\n");
		}
		return 0;
	}
	auto thp = ReadTextFile("/sys/kernel/mm/transparent_hugepage/enabled", kKiB);
	const auto plan =
		tune::PlanHostAllocation(options, thp ? std::optional<std::string>(*thp) : std::nullopt);
	tuning::HostAllocation section;
	if (plan.skipped)
	{
		section.reason = *plan.skipped;
		data.host_allocation = std::move(section);
		return 0;
	}
	section.state = "measured";
	section.bytes = plan.bytes;
	section.threads = HookConfig{}.cuda_host_alloc_threads;
	section.thp = plan.thp;
	section.private_state = plan.private_memory ? "measured" : "not_allowed";
	std::vector<std::string> arms{"driver", "shared"};
	if (plan.private_memory)
	{
		arms.push_back("private");
	}
	if (options.verbose)
	{
		std::fprintf(stderr,
					 "pinned allocation: time one %s cudaHostAlloc and its cudaFreeHost in each of %zu fresh "
					 "processes: the driver, the hook's shared memory%s\n",
					 HumanSize(plan.bytes).c_str(), std::size(arms) * plan.rounds * plan.processes,
					 plan.private_memory ? " and its private memory" : "");
	}
	if (options.progress_bar && !options.verbose)
	{
		std::fprintf(stderr, "cmt tune: Tuning pinned allocations...\n");
	}
	bool failed = false;
	for (unsigned round = 0; round < plan.rounds && !failed; round++)
	{
		for (unsigned process = 0; process < plan.processes && !failed; process++)
		{
			for (const auto& arm : arms)
			{
				if (arm == "private" && section.private_state != "measured")
				{
					continue;
				}
				std::optional<HostAllocSample> sample;
				auto measured = MeasureHostAllocArm(options, data, lock, arm, plan.bytes, sample);
				if (!measured || *measured != 0)
				{
					return measured;
				}
				if (!sample)
				{
					if (arm == "private")
					{
						section.private_state = "failed";
						continue;
					}
					failed = true;
					break;
				}
				auto& into = arm == "driver"   ? section.driver
							 : arm == "shared" ? section.shared
											   : section.private_memory;
				into.alloc_seconds.push_back(sample->alloc_seconds);
				into.free_seconds.push_back(sample->free_seconds);
				into.huge_kib.push_back(sample->huge_kib);
			}
		}
	}
	if (failed)
	{
		section = tuning::HostAllocation{};
		section.reason = "measurement failed";
	}
	data.host_allocation = std::move(section);
	CMT_RETURN_IF_ERROR(SaveTuning(options.out_dir, data));
	return 0;
}

void PrintHostAllocation(const tuning::HostAllocation& section)
{
	if (section.state != "measured")
	{
		std::fprintf(stdout,
					 "  pinned allocation: not measured (%s); `cmt run --config` keeps the driver's\n",
					 section.reason.c_str());
		return;
	}
	if (std::empty(section.driver.alloc_seconds) || std::empty(section.shared.alloc_seconds))
	{
		std::fprintf(stdout, "  pinned allocation: no samples; `cmt run --config` keeps the driver's\n");
		return;
	}
	const bool shared =
		tuning::HostAllocClearlyFaster(section.shared.alloc_seconds, section.driver.alloc_seconds);
	std::fprintf(
		stdout,
		"  pinned allocation, %s: driver %.3f s, hook %.3f s (median of %zu); `cmt run --config` %s\n",
		HumanSize(section.bytes).c_str(), Median(section.driver.alloc_seconds),
		Median(section.shared.alloc_seconds), std::size(section.shared.alloc_seconds),
		shared ? "uses the hook (--cuda-host-alloc-hook off keeps the driver's)"
			   : "keeps the driver's: the hook was not clearly faster");
	if (section.private_state == "measured" &&
		tuning::HostAllocClearlyFaster(section.private_memory.alloc_seconds, section.driver.alloc_seconds))
	{
		const double private_median = Median(section.private_memory.alloc_seconds);
		const double driver_median = Median(section.driver.alloc_seconds);
		const auto speedup = private_median > 0
								 ? fmt::format("{:.1f}x as fast as", driver_median / private_median)
								 : std::string("faster than");
		std::fprintf(
			stdout,
			"  private huge-page memory: %.3f s, %s the driver; never chosen for you: opt in "
			"with `cmt run --cuda-host-alloc-hook private` if the application does not rely on fork() "
			"children sharing its pinned memory\n",
			private_median, speedup.c_str());
	}
}

class TuningProgress
{
public:
	TuningProgress(bool show_progress, bool show_details, std::size_t planned, std::size_t finished,
				   bool at_most)
		: total(planned), enabled(show_progress), verbose(show_details), bounded(at_most),
		  interactive(enabled && isatty(STDERR_FILENO)),
		  bar(indicators::option::Stream{std::cerr}, indicators::option::PrefixText{"Tuning "},
			  indicators::option::BarWidth{kProgressBarWidth}, indicators::option::MaxProgress{total},
			  indicators::option::ShowElapsedTime{true}),
		  completed(finished), reused(finished)
	{
		if (verbose && at_most)
		{
			std::fprintf(stderr, "processes: at most %zu; the search may stop sooner\n", total);
		}
		else if (verbose)
		{
			std::fprintf(stderr, "processes: %zu planned; %zu finished earlier, %zu remaining\n", total,
						 reused, total - reused);
		}
		Draw();
	}

	~TuningProgress()
	{
		Draw(true);
		FinishLine();
		if (verbose)
		{
			std::fprintf(stderr, "processes: %zu/%zu done (%zu finished earlier); %zu with a failed size\n",
						 completed, total, reused, failed);
		}
	}

	void Begin(const BenchmarkRun& run)
	{
		if (!verbose)
		{
			return;
		}
		FinishLine();
		const auto& settings = run.copy.settings;
		std::string method = "original driver";
		if (run.copy.mode == Mode::Chunks)
		{
			method =
				fmt::format("staged; {} thread{}; {} buffers/thread; {} buffers; {}; {}", settings.workers,
							settings.workers == 1 ? "" : "s", settings.slots, HumanSize(settings.slot_bytes),
							settings.loop, settings.wc_slots ? "WC pinned" : "cached pinned");
		}
		else if (run.copy.mode == Mode::ProgressivePinning)
		{
			const auto& limit = run.copy.pinning.max_registered_bytes;
			method = fmt::format("progressive pinning; chunks {}..{}; registration limit {}; release at end",
								 HumanSize(run.copy.pinning.first_chunk_bytes),
								 HumanSize(run.copy.pinning.max_chunk_bytes),
								 limit ? HumanSize(*limit) : std::string("none (the GPU's memory)"));
		}
		const char* direction = run.copy.direction == Dir::HtoD ? "CPU -> GPU" : "GPU -> CPU";
		std::fprintf(stderr, "process %zu/%zu: %s; %s; %zu sizes; round %u\n", completed + 1, total,
					 direction, method.c_str(), run.sizes.size(), run.copy.process_round);
		if (interactive)
		{
			Draw();
		}
	}

	void Complete(const std::vector<CopyRun>& runs)
	{
		completed++;
		std::string rates;
		bool any_failed = false;
		for (const auto& run : runs)
		{
			any_failed |= run.status == "failed";
			rates += fmt::format(" {} {}", ShortSize(run.bytes),
								 run.status == "complete" ? fmt::format("{:.1f}", run.gbps) : run.status);
			if (!run.reason.empty())
			{
				rates += " (" + run.reason + ")";
			}
		}
		failed += any_failed;
		Draw();
		if (verbose || any_failed)
		{
			FinishLine();
			std::fprintf(stderr, "%s GB/s:%s\n", any_failed ? "cmt tune: failed measurement;" : " ",
						 rates.c_str());
		}
	}

	void FinishLine()
	{
		if (line_open)
		{
			std::fputc('\n', stderr);
			line_open = false;
		}
	}

private:
	void Draw(bool final = false)
	{
		if (!enabled || (final && drawn && completed == last_reported))
		{
			return;
		}
		const auto text = fmt::format("{}/{}{} processes", completed, bounded ? "<=" : "", total);
		if (interactive)
		{
			bar.set_option(indicators::option::PostfixText{text});
			bar.set_progress(completed);
			line_open = !bar.is_completed();
		}
		else if (!drawn || final || completed == total ||
				 completed - last_reported >= std::max<std::size_t>(1, total / 10))
		{
			std::fprintf(stderr, "cmt tune: Tuning... %s\n", text.c_str());
		}
		else
		{
			return;
		}
		drawn = true;
		last_reported = completed;
	}

	std::size_t total;
	bool enabled;
	bool verbose;
	bool bounded;
	bool interactive;
	indicators::ProgressBar bar;
	std::size_t completed;
	std::size_t reused;
	std::size_t failed = 0;
	bool line_open = false;
	bool drawn = false;
	std::size_t last_reported = 0;
};

Result<void> AdmitBenchmarkResult(const TuneArgs& options, TuningData& data, const BenchmarkRun& expected,
								  const ProcessResult& measured, const std::string& result_path)
{
	auto exists = IsRegularFile(result_path);
	if (!exists)
	{
		return Failure(exists.error());
	}
	if (measured.interrupted_by_signal != 0 || !measured.completed || !*exists)
	{
		for (const auto bytes : expected.sizes)
		{
			CopyRun failed = expected.copy;
			failed.bytes = bytes;
			failed.execution = expected.copy.mode == Mode::Driver ? "bare_driver" : "hooked_copy";
			failed.reason =
				measured.completed ? "benchmark process wrote no result" : "benchmark process interrupted";
			JsonWriter writer;
			WriteCopyRun(writer, failed, data.pages);
			auto raw = FinishJson(writer);
			if (!raw)
			{
				return Failure(raw.error());
			}
			data.runs.push_back(std::move(failed));
			data.raw_runs.push_back(*raw);
		}
		return {};
	}
	auto result = ReadTextFile(result_path, tuning::kMaxConfigBytes);
	if (!result)
	{
		return Failure(result.error());
	}
	JsonParser parser;
	Json document;
	CMT_RETURN_IF_ERROR(ParseDocument(parser, *result, document, result_path));
	Json machine;
	if (document["source_machine"].get(machine))
	{
		return UsageError("benchmark omitted source_machine");
	}
	auto identity = tune::ResumeIdentity(options, machine);
	if (!identity)
	{
		return Failure(identity.error());
	}
	if (*identity != data.identity)
	{
		return UsageError("benchmark machine or build changed after tuning preparation");
	}
	return AddRuns(data, document, &expected, expected.samples);
}

Result<int> MeasureProcess(const TuneArgs& options, const BenchmarkRun& run, TuningData& data,
						   const MeasurementLock& lock, TuningProgress& progress)
{
	for (std::size_t i = data.runs.size(); i > 0; i--)
	{
		if (FromProcess(data.runs[i - 1], run))
		{
			data.runs.erase(data.runs.begin() + static_cast<std::ptrdiff_t>(i - 1));
			data.raw_runs.erase(data.raw_runs.begin() + static_cast<std::ptrdiff_t>(i - 1));
		}
	}
	if (data.next_artifact_id == std::numeric_limits<std::size_t>::max())
	{
		return UsageError("tuning artifact numbers exhausted");
	}
	const auto prefix = options.out_dir + "/artifacts/run-" + std::to_string(data.next_artifact_id++);
	const auto expected_path = prefix + ".request.json";
	const auto result_path = prefix + ".json";
	auto expected = ExpectedRunJson(run);
	if (!expected)
	{
		return Failure(expected.error());
	}
	CMT_RETURN_IF_ERROR(WriteTextFile(expected_path, *expected));
	auto command = BenchmarkCommand(options, run, expected_path, result_path, prefix);
	lock.Apply(command);
	progress.Begin(run);
	const auto measured_start = Clock::now();
	auto measured = RunProcess(command);
	data.measured_children_wall += std::chrono::duration<double>(Clock::now() - measured_start).count();
	if (!measured)
	{
		return Failure(measured.error());
	}
	CMT_RETURN_IF_ERROR(RecordAttempt(
		data, command, *measured, {expected_path, result_path, command.stdout_path, command.stderr_path}));
	const auto first = data.runs.size();
	const auto admitted = AdmitBenchmarkResult(options, data, run, *measured, result_path);
	CMT_RETURN_IF_ERROR(SaveTuning(options.out_dir, data));
	if (!admitted)
	{
		return Failure(admitted.error());
	}
	progress.Complete({data.runs.begin() + static_cast<std::ptrdiff_t>(first), data.runs.end()});
	if (measured->interrupted_by_signal != 0)
	{
		return kSignalExitBase + measured->interrupted_by_signal;
	}
	if (measured->signal != 0)
	{
		return kSignalExitBase + measured->signal;
	}
	return 0;
}

Result<int> MeasureRemaining(const TuneArgs& options, const std::vector<BenchmarkRun>& planned,
							 TuningData& data, const MeasurementLock& lock)
{
	if (planned.empty())
	{
		return 0;
	}
	const auto reused = static_cast<std::size_t>(std::count_if(
		planned.begin(), planned.end(), [&](const BenchmarkRun& run) { return Finished(run, data.runs); }));
	TuningProgress progress(options.progress_bar, options.verbose, planned.size(), reused, false);
	for (const auto& run : planned)
	{
		if (!Finished(run, data.runs))
		{
			auto measured = MeasureProcess(options, run, data, lock, progress);
			if (!measured || *measured != 0)
			{
				return measured;
			}
		}
	}
	return 0;
}

} // namespace

std::optional<double> tune::Score(const TuneArgs& options, const std::vector<CopyRun>& runs, Dir direction,
								  const StagingSettings& settings, bool failed)
{
	constexpr double kFailed = -std::numeric_limits<double>::infinity();
	std::vector<BenchmarkRun> staged, driver;
	for (unsigned round = 1; round <= options.grid.rounds; round++)
	{
		staged.push_back(ProcessFor(options, direction, settings, round));
		driver.push_back(ProcessFor(options, direction, std::nullopt, round));
	}
	if (!std::all_of(staged.begin(), staged.end(),
					 [&](const BenchmarkRun& run) { return Finished(run, runs); }))
	{
		return failed ? std::optional<double>(kFailed) : std::nullopt;
	}
	const auto rates = [&](const std::vector<BenchmarkRun>& processes, std::uint64_t bytes)
	{
		std::vector<double> gbps;
		for (const auto& run : runs)
		{
			if (run.bytes == bytes && run.status == "complete" &&
				std::any_of(processes.begin(), processes.end(),
							[&](const BenchmarkRun& process) { return FromProcess(run, process); }))
			{
				gbps.insert(gbps.end(), run.sample_gbps.begin(), run.sample_gbps.end());
			}
		}
		return gbps;
	};
	const auto& sizes = options.grid.sizes;
	const bool any_large =
		std::any_of(sizes.begin(), sizes.end(), [](std::uint64_t bytes) { return bytes >= 16 * kMiB; });
	double sum = 0;
	std::size_t compared = 0;
	for (const auto bytes : sizes)
	{
		const auto stock = rates(driver, bytes);
		if ((any_large && bytes < 16 * kMiB) || stock.empty())
		{
			continue;
		}
		const auto ours = rates(staged, bytes);
		if (!ours.empty())
		{
			sum += std::log(Median(ours) / Median(stock));
			compared++;
		}
	}
	return compared ? sum / static_cast<double>(compared) : kFailed;
}

namespace {

Result<int> MeasureSearch(const TuneArgs& options, TuningData& data, const MeasurementLock& lock,
						  std::vector<BenchmarkRun>& planned)
{
	auto cost = CountRuns(options);
	if (!cost)
	{
		return Failure(cost.error());
	}
	TuningProgress progress(options.progress_bar, options.verbose, cost->processes, 0, true);
	const auto cpu = DetectCpuIsas();
	const auto measure = [&](Dir direction, const std::optional<StagingSettings>& settings) -> Result<int>
	{
		for (unsigned round = 1; round <= options.grid.rounds; round++)
		{
			const auto run = ProcessFor(options, direction, settings, round);
			if (!Finished(run, data.runs))
			{
				auto measured = MeasureProcess(options, run, data, lock, progress);
				if (!measured || *measured != 0)
				{
					return measured;
				}
			}
		}
		return 0;
	};
	const auto plan = [&](Dir direction, const std::optional<StagingSettings>& settings)
	{
		for (unsigned round = 1; round <= options.grid.rounds; round++)
		{
			planned.push_back(ProcessFor(options, direction, settings, round));
		}
	};
	for (Dir direction : options.dirs)
	{
		auto measured = measure(direction, std::nullopt);
		if (!measured || *measured != 0)
		{
			return measured;
		}
		plan(direction, std::nullopt);
		const int threads =
			tune::EstimateThreads(data.estimate->cpu_gbps.at(direction),
								  data.estimate->pinned_gbps.at(direction), data.estimate->max_threads);
		const tune::ScoreSetting score = [&](const StagingSettings& settings, bool failed)
		{ return tune::Score(options, data.runs, direction, settings, failed); };
		const tune::AdmitSetting admit = [&](StagingSettings settings)
		{ return tune::Admit(options, cpu, data.estimate->max_threads, std::move(settings)); };
		const tune::MeasureSetting staged = [&](const StagingSettings& settings)
		{ return measure(direction, settings); };
		for (bool wc : options.grid.wc)
		{
			std::vector<StagingSettings> counted;
			measured = tune::RunWalk(options.search, direction, threads, wc, admit, score, staged, counted);
			if (!measured || *measured != 0)
			{
				return measured;
			}
			for (const auto& settings : counted)
			{
				plan(direction, settings);
			}
		}
	}
	return 0;
}

bool MatrixComplete(const std::vector<BenchmarkRun>& planned, const TuningData& data)
{
	return std::any_of(data.runs.begin(), data.runs.end(),
					   [](const auto& run) { return run.status == "complete"; }) &&
		   std::all_of(planned.begin(), planned.end(),
					   [&](const auto& run) { return Finished(run, data.runs); });
}

Result<int> FinishTuning(const TuneArgs& options, const std::vector<BenchmarkRun>& planned, TuningData& data)
{
	const bool complete = MatrixComplete(planned, data);
	data.matrix_state = complete ? "complete" : "partial";
	data.state = data.matrix_state;
	SelectionPolicy policy;
	policy.process_rounds = options.grid.rounds;
	policy.tie_band = options.grid.tie_percent / 100.0;
	LocalCapabilities capabilities;
	capabilities.max_pinned_bytes = options.max_pinned_mib * kMiB;
	capabilities.progressive_pinning = ProgressivePinningAvailable();
	auto lookup = BuildCopyLookup(data.runs, policy, capabilities);
	if (!lookup)
	{
		return Failure(lookup.error());
	}
	CMT_RETURN_IF_ERROR(SaveTuning(options.out_dir, data));
	if (!options.verbose)
	{
		std::fprintf(stderr, "cmt tune: %s: %s/tune.json\n",
					 complete ? "Saved tuned file" : "Incomplete results", options.out_dir.c_str());
		return complete ? 0 : 1;
	}
	std::fprintf(stdout, "%s: %zu recorded copy runs, %zu derived intervals; %s/tune.json\n",
				 data.state.c_str(), data.runs.size(), lookup->rows.size(), options.out_dir.c_str());
	for (const Row& row : lookup->rows)
	{
		if (row.mode != Mode::Driver)
		{
			const auto& settings = row.settings;
			std::fprintf(stdout, "  %s %s..%s: %s\n", DirName(row.dir), ShortSize(row.min_bytes).c_str(),
						 row.max_bytes == UINT64_MAX ? "inf" : ShortSize(row.max_bytes).c_str(),
						 row.mode == Mode::ProgressivePinning
							 ? "progressive pinning"
							 : fmt::format("{} thread{} x {} buffers x {}; {}{}", settings.workers,
										   settings.workers == 1 ? "" : "s", settings.slots,
										   HumanSize(settings.chunk_bytes), hwcopy::Describe(settings.loop),
										   settings.wc_slots ? "; WC" : "")
								   .c_str());
		}
	}
	if (data.host_allocation)
	{
		PrintHostAllocation(*data.host_allocation);
	}
	return complete ? 0 : 1;
}

} // namespace

Result<int> RunTuning(const TuneArgs& supplied)
{
	const auto invocation_start = Clock::now();
	TuneArgs options = supplied;
	std::vector<BenchmarkRun> planned;
	auto prepared = PrepareSearch(options, planned);
	if (!prepared)
	{
		return Failure(prepared.error());
	}
	if (!*prepared)
	{
		return 0;
	}
	CMT_RETURN_IF_ERROR(
		ChooseIdentitySalt(options));
	auto lock = MeasurementLock::Acquire();
	if (!lock)
	{
		return Failure(lock.error());
	}
	if (options.verbose)
	{
		std::fprintf(stderr, "metadata: identify the CPU, GPU, driver and build\n");
	}
	auto data = CollectIdentity(options, invocation_start, *lock);
	if (!data)
	{
		return Failure(data.error());
	}
	OutputOwner owner;
	if (options.verbose)
	{
		std::fprintf(stderr, "%s\n",
					 options.resume ? "resume: validate identity and skip finished processes"
									: "output: prepare the tuning directory");
	}
	CMT_RETURN_IF_ERROR(OpenOutput(options, *data, owner));
	const bool searched = !tune::IsFullSearch(options.search);
	data->matrix_state = !searched && MatrixComplete(planned, *data) ? "complete" : "partial";
	if (searched)
	{
		CMT_RETURN_IF_ERROR(EnsureEstimate(options, *data, *lock));
		planned.clear();
	}
	auto measured = searched ? MeasureSearch(options, *data, *lock, planned)
							 : MeasureRemaining(options, planned, *data, *lock);
	if (!measured)
	{
		return Failure(measured.error());
	}
	if (*measured != 0)
	{
		return *measured;
	}
	auto allocated = MeasureHostAllocation(options, *data, *lock);
	if (!allocated)
	{
		return Failure(allocated.error());
	}
	if (*allocated != 0)
	{
		return *allocated;
	}
	if (options.verbose)
	{
		std::fprintf(stderr, "selection: compare verified results and publish tune.json\n");
	}
	return FinishTuning(options, planned, *data);
}
} // namespace cmt
