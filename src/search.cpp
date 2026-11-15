#include "tuner_internal.h"

namespace cmt {
using namespace tune_detail;

namespace {
constexpr unsigned kMaxSamples = 100;

constexpr std::array<int, 5> kSearchThreads{1, 2, 4, 8, 16};

constexpr std::size_t kHostAllocBytes = 1024 * kMiB;
constexpr std::size_t kHostAllocMinBytes = 2 * kMiB;
constexpr unsigned kHostAllocRounds = 2;
constexpr unsigned kHostAllocProcesses = 3;

struct Relative
{
	int shift;
	int slots;
	std::size_t slot_mib;
	const char* loop;
};

constexpr Relative kSeeds[2][kInstantSettings] = {{{0, 2, 1, "memcpy"}, {0, 2, 1, "nt_store"}},
												  {{0, 2, 1, "nt_store"}, {0, 4, 2, "nt_store"}}};
constexpr Relative kPortfolio[2][6] = {{{2, 1, 2, "nt_store@avx2"},
										{1, 1, 1, "cached@avx2"},
										{-1, 2, 4, "memcpy"},
										{2, 1, 1, "nt_store@avx2"},
										{-1, 4, 1, "nt_store@avx2"},
										{-1, 1, 1, "movsb"}},
									   {{2, 2, 4, "nt_store@avx2"},
										{0, 1, 1, "nt_store@avx2"},
										{2, 4, 1, "movsb"},
										{2, 4, 2, "nt_store@avx512"},
										{2, 4, 1, "nt_store@avx2"},
										{1, 2, 8, "nt_store@avx512"}}};
constexpr std::pair<int, std::size_t> kSlotShapes[] = {{1, 1}, {2, 1}, {4, 1}, {2, 2},
													   {4, 2}, {2, 4}, {4, 4}};

std::vector<std::string> SupportedLoops()
{
	std::vector<std::string> loops;
	const auto cpu = DetectCpuIsas();
	for (const Variant& variant : Variants(NativeArch()))
	{
		if (cpu.Has(variant.isa) && variant.method != Method::NtStorePrefetch)
		{
			loops.push_back(VariantText(variant));
		}
	}
	return loops;
}

} // namespace

namespace tune {
const char* SearchName(Search search)
{
	switch (search)
	{
	case Search::Instant:
		return "instant";
	case Search::Quick:
		return "quick";
	case Search::Full:
		return "full";
	case Search::FullWithKernel:
		return "full_with_kernel";
	}
	return "unknown";
}

bool IsFullSearch(Search search)
{
	return search == Search::Full || search == Search::FullWithKernel;
}

HostAllocPlan PlanHostAllocation(const TuneArgs& options, const std::optional<std::string>& thp_enabled)
{
	HostAllocPlan plan;
	plan.bytes = std::min(kHostAllocBytes, options.max_pinned_mib * kMiB);
	plan.rounds = kHostAllocRounds;
	plan.processes = kHostAllocProcesses;
	const auto open = thp_enabled ? thp_enabled->find('[') : std::string::npos;
	const auto close = open != std::string::npos ? thp_enabled->find(']', open) : std::string::npos;
	if (close != std::string::npos)
	{
		plan.thp = thp_enabled->substr(open + 1, close - open - 1);
	}
	plan.private_memory = plan.thp == "madvise" || plan.thp == "always";
	if (options.search == Search::Instant)
	{
		plan.skipped = "instant search";
	}
	else if (!options.host_alloc)
	{
		plan.skipped = "host_alloc off";
	}
	else if (plan.bytes < kHostAllocMinBytes)
	{
		plan.skipped = "max_pinned_mib below 2 MiB";
	}
	return plan;
}

namespace {
Grid DefaultGrid(Search search)
{
	Grid grid;
	grid.wc = {false};
	switch (search)
	{
	case Search::Instant:
		grid.sizes = {1 * kMiB, 4 * kMiB, 16 * kMiB, 64 * kMiB, 512 * kMiB, 1024 * kMiB};
		break;
	case Search::Quick:
		grid.sizes = {64 * kKiB, 512 * kKiB, 1 * kMiB,	 4 * kMiB,
					  16 * kMiB, 64 * kMiB,	 512 * kMiB, 1024 * kMiB};
		break;
	case Search::Full:
	case Search::FullWithKernel:
		grid.sizes = {4 * kKiB, 16 * kKiB, 64 * kKiB, 128 * kKiB, 512 * kKiB, 1 * kMiB,	  2 * kMiB,
					  4 * kMiB, 8 * kMiB,  16 * kMiB, 64 * kMiB,  128 * kMiB, 512 * kMiB, 1024 * kMiB};
		grid.threads = {1, 2, 4, 8, 16};
		grid.buffers = {1, 2, 4};
		grid.buffer_bytes = {1 * kMiB, 2 * kMiB, 4 * kMiB, 8 * kMiB, 16 * kMiB};
		grid.loops = SupportedLoops();
		break;
	}
	return grid;
}

bool AcceptsKey(Search search, std::string_view key)
{
	return search != Search::Instant && (IsFullSearch(search) || key == "sizes" || key == "memory" ||
										 key == "samples" || key == "rounds" || key == "tie");
}

Result<std::vector<std::string_view>> SplitList(std::string_view key, std::string_view value)
{
	std::vector<std::string_view> items;
	for (std::size_t start = 0;;)
	{
		const auto end = value.find(',', start);
		const auto item = value.substr(start, end == std::string_view::npos ? end : end - start);
		if (item.empty() || std::find(items.begin(), items.end(), item) != items.end())
		{
			std::fprintf(stderr, "cmt tune: --tuning-params %.*s: empty or repeated list entry\n",
						 static_cast<int>(key.size()), key.data());
			return UsageError("invalid --tuning-params list");
		}
		items.push_back(item);
		if (end == std::string_view::npos)
		{
			return items;
		}
		start = end + 1;
	}
}

std::optional<unsigned> Count(std::string_view text, unsigned min, unsigned max)
{
	unsigned value = 0;
	const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
	if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value < min || value > max)
	{
		return std::nullopt;
	}
	return value;
}

std::optional<std::uint64_t> Bytes(std::string_view text)
{
	auto bytes = ParseSize(std::string(text));
	return bytes && *bytes > 0 ? std::optional<std::uint64_t>(*bytes) : std::nullopt;
}

bool ParseValue(Grid& grid, std::string_view key, const std::vector<std::string_view>& items)
{
	const bool single = items.size() == 1;
	if (key == "samples" || key == "rounds")
	{
		const auto count =
			single ? Count(items[0], 1, key == "samples" ? kMaxSamples : kMaxProcessRounds) : std::nullopt;
		(key == "samples" ? grid.samples : grid.rounds) = count.value_or(0);
		return count.has_value();
	}
	if (key == "tie")
	{
		const auto percent = single && items[0].back() == '%'
								 ? Count(items[0].substr(0, items[0].size() - 1), 0, 100)
								 : std::nullopt;
		grid.tie_percent = percent.value_or(0);
		return percent.has_value();
	}
	if (key == "memory")
	{
		grid.wc.clear();
		for (const auto item : items)
		{
			if (item != "cached" && item != "wc")
			{
				return false;
			}
			grid.wc.push_back(item == "wc");
		}
		return true;
	}
	if (key == "loops")
	{
		if (single && items[0] == "all")
		{
			grid.loops = SupportedLoops();
			return true;
		}
		grid.loops.clear();
		for (const auto item : items)
		{
			const auto loop = ParseLoopName(NativeArch(), item);
			if (!loop || !Resolve(NativeArch(), *loop, DetectCpuIsas(), std::nullopt))
			{
				return false;
			}
			grid.loops.emplace_back(item);
		}
		return true;
	}
	if (key == "threads" || key == "buffers")
	{
		auto& counts = key == "threads" ? grid.threads : grid.buffers;
		counts.clear();
		for (const auto item : items)
		{
			const auto count = Count(item, 1, key == "threads" ? kMaxPipelineWorkers : kMaxStagingBuffers);
			if (!count)
			{
				return false;
			}
			counts.push_back(static_cast<int>(*count));
		}
		return true;
	}
	auto& sizes = key == "sizes" ? grid.sizes : grid.buffer_bytes;
	sizes.clear();
	for (const auto item : items)
	{
		const auto bytes = Bytes(item);
		if (!bytes || (key == "buffer_size" && (*bytes < kMinSlotBytes || *bytes > kMaxSlotBytes ||
												*bytes % kChunkSizeGranularity != 0)))
		{
			return false;
		}
		sizes.push_back(*bytes);
	}
	return true;
}
} // namespace

Result<Grid> ParseTuningParams(Search search, std::string_view text)
{
	static const std::map<std::string_view, const char*> accepted{
		{"sizes", "copy sizes such as 4K,1M,1G"},
		{"threads", "copying threads 1..16, caller included"},
		{"buffers", "pinned buffers per thread, 1..32"},
		{"buffer_size", "pinned buffer bytes, 64K..64M in 4K steps"},
		{"loops", "all, or CPU loop names this CPU supports"},
		{"memory", "cached, wc or cached,wc"},
		{"samples", "timed copies per size per process, 1..100"},
		{"rounds", "fresh processes per setting, 1..20"},
		{"tie", "a selection tie band 0%..100%, such as 2%"}};
	Grid grid = DefaultGrid(search);
	std::set<std::string_view> seen;
	for (std::size_t start = 0; start < text.size();)
	{
		const auto end = std::min(text.find(' ', start), text.size());
		const auto entry = text.substr(start, end - start);
		start = end + 1;
		if (entry.empty())
		{
			continue;
		}
		const auto separator = entry.find('=');
		const auto key = entry.substr(0, separator);
		const auto found = accepted.find(key);
		if (separator == std::string_view::npos || found == accepted.end())
		{
			std::fprintf(stderr,
						 "cmt tune: --tuning-params: unknown entry '%.*s'; keys are key=value with key "
						 "one of sizes, threads, buffers, buffer_size, loops, memory, samples, rounds, tie\n",
						 static_cast<int>(entry.size()), entry.data());
			return UsageError("unknown --tuning-params key");
		}
		if (!AcceptsKey(search, key) || !seen.insert(key).second)
		{
			std::fprintf(stderr, "cmt tune: --tuning-params %.*s: %s\n", static_cast<int>(key.size()),
						 key.data(),
						 AcceptsKey(search, key) ? "given twice"
						 : search == Search::Instant
							 ? "instant has fixed sizes and settings; use --search quick or full"
							 : "quick chooses its settings; use --search full to change them");
			return UsageError("--tuning-params key is repeated or not accepted by this search");
		}
		auto items = SplitList(key, entry.substr(separator + 1));
		if (!items)
		{
			return Failure(items.error());
		}
		if (!ParseValue(grid, key, *items))
		{
			std::fprintf(stderr, "cmt tune: --tuning-params %.*s: expected %s\n",
						 static_cast<int>(entry.size()), entry.data(), found->second);
			return UsageError("invalid --tuning-params value");
		}
	}
	if (grid.samples * grid.rounds < kMinMeasuredSamples)
	{
		std::fprintf(stderr,
					 "cmt tune: --tuning-params samples x rounds must be at least %u: selection "
					 "compares the ranges of all timed samples\n",
					 kMinMeasuredSamples);
		return UsageError("--tuning-params gives too few samples to select");
	}
	return grid;
}

std::string FormatGrid(Search search, const Grid& grid)
{
	std::string text = fmt::format("  sizes={}\n", JoinList(grid.sizes, ShortSize));
	if (search == Search::Instant)
	{
		return text + "  layouts=estimated threads; cpu-to-gpu 2 x 1M memcpy and nt_store, gpu-to-cpu 2 x 1M "
					  "and 4 x 2M nt_store\n  samples=2 rounds=1 (no keys accepted)\n";
	}
	if (search == Search::Quick)
	{
		text +=
			"  layouts=searched: instant's 2, 6 more, then coordinate descent; at most 31 per direction\n";
	}
	else
	{
		const auto number = [](int value) { return std::to_string(value); };
		const auto name = [](const std::string& value) { return value; };
		text += fmt::format("  threads={}\n  buffers={}\n  buffer_size={}\n  loops={}\n",
							JoinList(grid.threads, number), JoinList(grid.buffers, number),
							JoinList(grid.buffer_bytes, ShortSize), JoinList(grid.loops, name));
	}
	const auto memory = [](bool wc) { return std::string(wc ? "wc" : "cached"); };
	text += fmt::format("  memory={}\n  samples={}\n  rounds={}\n", JoinList(grid.wc, memory), grid.samples,
						grid.rounds);
	return grid.tie_percent ? text + fmt::format("  tie={}%\n", grid.tie_percent) : text;
}

int ThreadCap(std::size_t allowed_cpus)
{
	int cap = kSearchThreads.front();
	for (int threads : kSearchThreads)
	{
		cap = static_cast<std::size_t>(threads) <= allowed_cpus ? threads : cap;
	}
	return cap;
}

Result<ChildPlacement> MeasurementPlacement(const std::vector<std::string>& env,
											const std::optional<std::string>& gpu_pci, const GpuCpus& nvml,
											std::string_view sysfs)
{
	WorkerPlacement request;
	for (const auto& entry : env)
	{
		const auto separator = entry.find('=');
		if (separator != std::string::npos)
		{
			CMT_RETURN_IF_ERROR(ParseWorkerPlacement(std::string_view(entry).substr(0, separator),
													 std::string_view(entry).substr(separator + 1), request));
		}
	}
	const auto unpinned = [](std::string why) { return ChildPlacement{{}, "unpinned", std::move(why)}; };
	if (!std::empty(request.cpus) || request.numa_node)
	{
		return unpinned("CMT_WORKER_CPUS or CMT_WORKER_NUMA_NODE places the workers");
	}
	if (request.gpu_numa == false)
	{
		return unpinned("--numa off");
	}
	if (CpuNodeCount(sysfs) < 2)
	{
		return unpinned("one NUMA node");
	}
	if (!gpu_pci)
	{
		return unpinned("the GPU's PCI address is unknown");
	}
	const auto local = GpuLocalPlacement(*gpu_pci, nvml, sysfs);
	if (std::empty(local.cpus))
	{
		return unpinned(local.reason);
	}
	return ChildPlacement{local.cpus, "cpus=" + FormatCpuList(local.cpus), local.reason};
}

std::vector<std::string> InheritedEnvironment()
{
	std::vector<std::string> entries;
	for (const char* name : kInheritedVariables)
	{
		if (const char* value = std::getenv(name))
		{
			entries.push_back(std::string(name) + "=" + value);
		}
	}
	return entries;
}

int EstimateThreads(double cpu_gbps, double pinned_gbps, int max_threads)
{
	for (int threads : kSearchThreads)
	{
		if (threads * cpu_gbps >= kEstimateHeadroom * pinned_gbps)
		{
			return std::min(threads, max_threads);
		}
	}
	return max_threads;
}

std::optional<StagingSettings> Admit(const TuneArgs& options, const CpuIsas& cpu, int max_threads,
									 StagingSettings settings)
{
	const auto name = ParseLoopName(NativeArch(), settings.loop);
	const auto variant = name ? Resolve(NativeArch(), *name, cpu, std::nullopt) : std::nullopt;
	if (!variant || settings.workers > max_threads || !FitsPinnedBudget(options, settings))
	{
		return std::nullopt;
	}
	settings.loop = VariantText(*variant);
	return settings;
}

Walk WalkSettings(Search search, Dir direction, int threads, bool wc, const SettingScore& score,
				  const AdmitSetting& admit)
{
	const std::size_t row = direction == Dir::HtoD ? 0 : 1;
	const std::size_t budget = search == Search::Instant ? std::size(kSeeds[row]) : kQuickSettings;
	Walk walk;
	const auto visit = [&](StagingSettings setting)
	{
		auto admitted = admit(std::move(setting));
		if (!admitted || std::find(walk.visited.begin(), walk.visited.end(), *admitted) != walk.visited.end())
		{
			return false;
		}
		if (!score(*admitted))
		{
			walk.next = std::move(*admitted);
			return true;
		}
		walk.visited.push_back(std::move(*admitted));
		return std::size(walk.visited) >= budget;
	};
	const auto relative = [&](const Relative& r)
	{
		const int scaled = r.shift >= 0 ? threads << r.shift : threads >> -r.shift;
		return StagingSettings{std::clamp(scaled, kSearchThreads.front(), kSearchThreads.back()), r.slots,
							   r.slot_mib * kMiB, r.loop, wc};
	};
	for (const auto& seed : kSeeds[row])
	{
		if (visit(relative(seed)))
		{
			return walk;
		}
	}
	if (search == Search::Instant)
	{
		return walk;
	}
	for (const auto& member : kPortfolio[row])
	{
		if (visit(relative(member)))
		{
			return walk;
		}
	}
	if (walk.visited.empty())
	{
		return walk;
	}
	StagingSettings best = walk.visited.front();
	double best_score = *score(best);
	for (const auto& setting : walk.visited)
	{
		if (*score(setting) > best_score)
		{
			best = setting;
			best_score = *score(setting);
		}
	}
	const std::vector<const char*> loops =
		direction == Dir::HtoD ? std::vector<const char*>{"memcpy", "nt_store", "movsb", "cached", "nt_load"}
							   : std::vector<const char*>{"nt_store", "movsb", "memcpy", "nt_store@avx2"};
	const double step = std::log1p(SelectionPolicy{}.minimum_gain);
	for (bool moved = true; moved;)
	{
		moved = false;
		for (std::size_t coordinate = 0; coordinate < 3; coordinate++)
		{
			std::vector<StagingSettings> neighbours;
			for (std::size_t i = 0; i < (coordinate == 0   ? std::size(kSearchThreads)
										 : coordinate == 1 ? std::size(kSlotShapes)
														   : std::size(loops));
				 i++)
			{
				StagingSettings neighbour = best;
				if (coordinate == 0)
				{
					neighbour.workers = kSearchThreads[i];
				}
				else if (coordinate == 1)
				{
					neighbour.slots = kSlotShapes[i].first;
					neighbour.slot_bytes = kSlotShapes[i].second * kMiB;
				}
				else
				{
					neighbour.loop = loops[i];
				}
				if (visit(neighbour))
				{
					return walk;
				}
				neighbours.push_back(std::move(neighbour));
			}
			std::optional<StagingSettings> next;
			double next_score = -std::numeric_limits<double>::infinity();
			for (auto& neighbour : neighbours)
			{
				auto admitted = admit(std::move(neighbour));
				if (admitted && *score(*admitted) > next_score)
				{
					next_score = *score(*admitted);
					next = std::move(admitted);
				}
			}
			if (next && next_score > best_score + step)
			{
				best = std::move(*next);
				best_score = next_score;
				moved = true;
			}
		}
	}
	return walk;
}

Result<int> RunWalk(Search search, Dir direction, int threads, bool wc, const AdmitSetting& admit,
					const ScoreSetting& score, const MeasureSetting& measure,
					std::vector<StagingSettings>& counted)
{
	std::vector<StagingSettings> failed;
	const SettingScore known = [&](const StagingSettings& settings)
	{ return score(settings, std::find(failed.begin(), failed.end(), settings) != failed.end()); };
	for (;;)
	{
		const auto walk = WalkSettings(search, direction, threads, wc, known, admit);
		if (!walk.next)
		{
			for (const auto& settings : walk.visited)
			{
				if (std::find(failed.begin(), failed.end(), settings) == failed.end())
				{
					counted.push_back(settings);
				}
			}
			return 0;
		}
		auto measured = measure(*walk.next);
		if (!measured || *measured != 0)
		{
			return measured;
		}
		if (!score(*walk.next, false))
		{
			failed.push_back(*walk.next);
		}
	}
}
} // namespace tune

std::string tune_detail::ShortSize(std::uint64_t bytes)
{
	for (const auto& [unit, scale] :
		 {std::pair{'G', kMiB * kKiB}, std::pair{'M', kMiB}, std::pair{'K', kKiB}})
	{
		if (bytes >= scale && bytes % scale == 0)
		{
			return fmt::format("{}{}", bytes / scale, unit);
		}
	}
	return std::to_string(bytes);
}

bool tune_detail::FitsPinnedBudget(const TuneArgs& options, const StagingSettings& settings)
{
	return static_cast<std::size_t>(settings.workers) * settings.slots * settings.slot_bytes <=
		   options.max_pinned_mib * kMiB;
}

Result<tune::Estimate> tune::ReadEstimate(const TuneArgs& options, Json cpu_copies, Json pinned_copies)
{
	std::string_view status;
	simdjson::dom::array probes;
	simdjson::dom::array pinned;
	if (cpu_copies["status"].get(status) || status != "complete" ||
		cpu_copies["cpu_copy_probes"].get(probes) || pinned_copies.get(pinned))
	{
		return Failure(Error::Kind::Verify,
					   "estimate probe requires completed CPU-copy diagnostics and pinned-copy runs");
	}
	Estimate estimate;
	const auto add = [&](Json record, std::map<Dir, double>& rates, std::string_view pages) -> Result<void>
	{
		std::string_view direction, requested_pages;
		bool verified = false;
		double gbps = 0;
		if (record["direction"].get(direction) || record["status"].get(status) || status != "complete" ||
			record["verified"].get(verified) || !verified || record["gbps"].get(gbps) ||
			!std::isfinite(gbps) || gbps <= 0 || record["pages_requested"].get(requested_pages) ||
			requested_pages != pages)
		{
			return Failure(Error::Kind::Verify, "estimate probe returned an incomplete or unverified copy");
		}
		const auto dir = direction == "htod" ? Dir::HtoD : Dir::DtoH;
		if ((direction != "htod" && direction != "dtoh") ||
			std::find(options.dirs.begin(), options.dirs.end(), dir) == options.dirs.end() ||
			!rates.emplace(dir, gbps).second)
		{
			return Failure(Error::Kind::Verify,
						   "estimate probe returned an unrequested or repeated direction");
		}
		return {};
	};
	for (Json probe : probes)
	{
		CMT_RETURN_IF_ERROR(add(probe, estimate.cpu_gbps, PagesName(options.pages)));
	}
	for (Json run : pinned)
	{
		std::string_view path;
		std::uint64_t bytes = 0;
		if (run["path"].get(path) || path != "pinned_ce" || run["bytes"].get(bytes) ||
			bytes != kPinnedProbeBytes)
		{
			return Failure(Error::Kind::Verify, "estimate probe returned an unexpected pinned copy");
		}
		CMT_RETURN_IF_ERROR(add(run, estimate.pinned_gbps, "driver"));
	}
	if (std::size(estimate.cpu_gbps) != std::size(options.dirs) ||
		std::size(estimate.pinned_gbps) != std::size(options.dirs))
	{
		return Failure(Error::Kind::Verify, "estimate probe omitted a direction");
	}
	estimate.cpu_copies = simdjson::to_string(cpu_copies);
	estimate.pinned_copies = simdjson::to_string(pinned_copies);
	auto allowed = std::empty(options.placement.cpus) ? GetThreadAffinity()
													  : Result<std::vector<unsigned>>{options.placement.cpus};
	if (!allowed)
	{
		return Failure(allowed.error());
	}
	estimate.max_threads = tune::ThreadCap(std::size(*allowed));
	return estimate;
}

} // namespace cmt
