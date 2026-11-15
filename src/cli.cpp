#include "cli.h"
#include "common/aliases.h"
#include "common/error.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "measure/metrics.h"

#include <CLI/CLI.hpp>
#include <new>

namespace cmt {
namespace {

bool IsAdvancedHelp(const std::string& word)
{
	return word == "--help-advanced" || word == "--help_advanced";
}

bool WantsHelp(const std::vector<std::string>& args)
{
	return std::any_of(args.begin(), args.end(), [](const std::string& word)
					   { return word == "-h" || word == "--help" || IsAdvancedHelp(word); });
}

std::string CanonicalFlag(const CLI::App& app, const std::string& word)
{
	if (word.rfind("--", 0) != 0)
	{
		return word;
	}
	const auto equals = word.find('=');
	std::string name = word.substr(0, equals);
	if (name.find('_') == std::string::npos)
	{
		return word;
	}
	std::replace(name.begin(), name.end(), '_', '-');
	if (app.get_option_no_throw(name) == nullptr)
	{
		return word;
	}
	return equals == std::string::npos ? name : name + word.substr(equals);
}

void ParseCli(CLI::App& app, const std::vector<std::string>& args)
{
	// CLI11's vector overload consumes arguments in reverse order.
	std::vector<std::string> reversed;
	reversed.reserve(args.size());
	for (auto word = args.rbegin(); word != args.rend(); ++word)
	{
		reversed.push_back(CanonicalFlag(app, *word));
	}
	app.parse(reversed);
}

Result<void> CheckWorkerCpus(const std::string& list)
{
	auto cpus = ParseThreadCpus(list);
	if (!cpus)
	{
		std::fprintf(stderr, "cmt: --worker-cpus %s: %s\n", list.c_str(), cpus.error().Message());
		return Failure(cpus.error());
	}
	return {};
}

MaybeError CliFailure(const CLI::Error& error)
{
	std::fprintf(stderr, "cmt: %s\n", error.what());
	return UsageError("invalid command line; see the preceding diagnostic");
}

CLI::Validator SizeArgument(bool list, bool allow_standard = false)
{
	return CLI::Validator(
			   [list, allow_standard](std::string& text) -> std::string
			   {
				   if (allow_standard && text == "standard")
				   {
					   return {};
				   }
				   std::size_t error_offset = 0;
				   const char* reason = nullptr;
				   if (list)
				   {
					   const auto parsed = ParseSizes(text, &error_offset);
					   if (!parsed)
					   {
						   reason = parsed.error().Message();
					   }
				   }
				   else
				   {
					   const auto parsed = ParseSize(text, &error_offset);
					   if (!parsed)
					   {
						   reason = parsed.error().Message();
					   }
				   }
				   return reason ? fmt::format("byte {} (zero-based) in {:?}: {}", error_offset, text, reason)
								 : std::string{};
			   },
			   "SIZE")
		.application_index(0);
}

struct Flags
{
	CLI::App app{"Host <-> device bandwidth of one copy path, as a bytes-vs-GB/s sweep (nvbandwidth style)",
				 "cmt bench"};
	std::string path;
	std::string dir;
	std::string sizes = "4K..1G";
	std::string pages;
	std::string alloc;
	std::string fill;
	std::string slot_bytes;
	std::string pin_step;
	std::string unpin;
	std::string min_bytes;
	std::string tag;
	int threads = 1;
	int slots = 4;
	int samples = 3;
	int cpu = -1;
	int device = 0;
	double budget_s = 0;
	std::string numa;
	std::string worker_cpus;
	CLI::Option* worker_numa_option = nullptr;
	CLI::Option* worker_cpus_option = nullptr;
	bool header = false;
	bool help = false;
	bool info = false;
	bool no_flush = false;
	bool no_verify = false;
	bool wc = false;
	cmt::HashMap<Knob, std::vector<CLI::Option*>>
		knob_opts;

	Flags()
	{
		app.set_help_flag();
		app.option_defaults()->take_last();
		std::string paths;
		for (const PathSpec& p : BenchmarkPaths())
		{
			paths += "\n      " + p.name + ": " + p.summary;
		}
		app.add_option("--path", path, "copy path (default pinned_ce):" + paths);
		AddKnob(Knob::Direction, app.add_option("--dir", dir, "htod | dtoh (default htod)"));
		app.add_option("--sizes", sizes,
					   "e.g. 4K..16G or 64K,1M,1G; ranges are powers of two (default 4K..1G)")
			->check(SizeArgument(true));
		AddKnob(Knob::Pages, app.add_option("--pages", pages,
											"small | thp | hugetlb: mmap host-page policy "
											"(default base pages; 4k aliases small)"));
		AddKnob(Knob::Alloc,
				app.add_option("--alloc", alloc, "register | hostalloc | hostalloc_wc (default register)"));
		std::string copy_loops;
		for (const Variant& variant : Variants(NativeArch()))
		{
			copy_loops += VariantText(variant) + " | ";
		}
		AddKnob(Knob::Fill, app.add_option("--fill", fill,
										   "staged CPU loop: " + copy_loops +
											   "none (DMA only). ISA must be supported by this CPU; "
											   "omit @ISA to select its widest variant (default memcpy)"));
		AddKnob(Knob::Threads, app.add_option("--threads", threads, "staged: copier threads (default 1)"));
		AddKnob(Knob::Slots, app.add_option("--slots", slots, "staged: slots per thread (default 4)"));
		AddKnob(Knob::SlotBytes,
				app.add_option("--slot-bytes", slot_bytes, "staged: bytes per slot (default 1M)")
					->check(SizeArgument(false)));
		AddKnob(Knob::Wc, app.add_flag("--wc", wc, "staged: write-combined slots (DtoH CPU-read control)"));
		AddKnob(Knob::PinStep, app.add_option("--pin-step", pin_step,
											  "pin_pipeline: bytes per cudaHostRegister step (default 32M)")
								   ->check(SizeArgument(false)));
		AddKnob(Knob::Unpin,
				app.add_option("--unpin", unpin, "pin_pipeline: end | eager | prereg (default end)"));
		app.add_option("--samples", samples, "samples per size (default 3)");
		app.add_option("--min-bytes", min_bytes,
					   "a sample copies at least this many bytes, repeating the copy, at most 2^20 copies per "
					   "sample (default 64M)")
			->check(SizeArgument(false));
		app.add_option("--budget-s", budget_s,
					   "stop a size after S seconds, once 2 samples ran (default 0: no limit)");
		app.add_option("--cpu", cpu,
					   "pin the main thread to this CPU; staged copier t runs on CPU + t (default: none)");
		app.add_option("--device", device, "CUDA device (default 0)")->check(CLI::NonNegativeNumber);
		worker_numa_option =
			app.add_option("--numa", numa,
						   "on or off: staged workers and their pool on the GPU's NUMA node; "
						   "caller unchanged; conflicts with --cpu (default: on with 2+ "
						   "NUMA nodes)")
				->check(CLI::IsMember({"on", "off"}));
		worker_cpus_option =
			app.add_option("--worker-cpus", worker_cpus,
						   "Worker CPU IDs/ranges, e.g. 2,4-7; overrides automatic NUMA CPU placement");
		app.add_option("--tag", tag, "label preserved in measurement JSON; also the first CSV column");
		app.add_flag("--no-flush", no_flush,
					 "do not evict the host buffer from CPU caches before each sample");
		app.add_flag("--no-verify", no_verify, "do not check the data after each size");
		app.add_flag("--header", header,
					 "print the CSV header first (alone: the header only, no GPU needed)");
		app.add_flag("--info", info, "print the GPU, link and host, then exit");
		app.add_flag("-h,--help", help, "this text");
		app.footer("lifecycle: sum of allocation/first-touch, pin, completed copy, unpin and frees; "
				   "preparation/verification excluded. Use --min-bytes 1 for one lifecycle per sample.\n\n"
				   "A flag that the chosen path does not read is an error, not ignored.\n"
				   "exit codes: 0 ok, 1 failed measurement / internal error, 2 usage, 3 unsupported "
				   "here, 4 OS error,\n"
				   "5 CUDA error, 6 wrong data, 77 no GPU");
	}

	void AddKnob(Knob k, CLI::Option* o)
	{
		knob_opts[k].push_back(o);
	}
};

} // namespace

std::string UsageText()
try
{
	Flags f;
	return f.app.help();
}
catch (const CLI::Error& error)
{
	std::fprintf(stderr, "cmt: %s\n", error.what());
	Panic("CLI help construction failed");
}
catch (const std::bad_alloc&)
{
	Panic("CLI allocation failed");
}

bool InternalCommandsEnabled()
{
	const char* value = std::getenv("CMT_INTERNAL_ENABLE");
	return value && std::string_view(value) == "1";
}

std::string TopLevelUsage()
{
	std::string text = "cmt: tune CPU <-> GPU copies per size, then apply the result to an unchanged application\n\n"
		   "Usage: cmt COMMAND [OPTIONS]\n"
		   "       cmt --version\n\n"
		   "Commands:\n"
		   "  tune     measure copy strategies; optionally load/tune and launch with cmt tune -- APP\n"
		   "  run      run an application with a tune.json (cmt run --config tune.json -- python train.py)\n"
		   "           or with the matching shared tune (cmt run --remote-config -- python train.py)\n"
		   "  export   write a shareable copy of tune.json: its selection and a summary, nothing identifying\n"
		   "  trace    record an application's CUDA copy sizes and memory types\n"
		   "  patch    build and load the host-registration NVIDIA kernel modules (root)\n"
		   "  unpatch  restore the kernel modules `cmt patch` replaced (root)\n"
		   "  bench    measure one copy path's bandwidth over a size sweep\n"
		   "  help     print this text\n";
	if (InternalCommandsEnabled())
	{
		text += "\nInternal commands (CMT_INTERNAL_ENABLE=1):\n"
				"  check    validate results or rebuild their selection\n"
				"  kernel   inspect, build, run and restore explicit kernel module sets\n";
	}
	return text + "\n`cmt COMMAND --help` prints a command's options. Start with: cmt tune -- APP\n"
		   "Flags are spelled with dashes (--max-pinned-mib); the same words with underscores work too.\n";
}

std::string CheckUsage()
{
	return "Recompute tune.json's selection line (line 1) from the runs below it\n"
		   "Usage: cmt check [--write] TUNE_JSON\n"
		   "       cmt check URL\n\n"
		   "Prints OK when line 1 matches the runs; exits 6 when it differs.\n"
		   "--write replaces a differing line 1 (atomically) instead of failing.\n"
		   "A shared tune (`cmt export`, cmt.tune-share/1, a file or an https URL) and a tune from\n"
		   "`cmt run --remote-config` hold no runs: their form is validated, and their rows cannot be re-derived.\n";
}

Result<Options> ParseArgs(const std::vector<std::string>& args)
try
{
	Flags f;
	if (WantsHelp(args))
	{
		Options o;
		o.help = true;
		auto path = PathNamed("pinned_ce");
		if (!path)
		{
			return cmt::Failure(path.error());
		}
		o.cfg.path = *path;
		return o;
	}
	ParseCli(f.app, args);

	Options o;
	Config& c = o.cfg;
	o.header = f.header, o.help = f.help, o.info = f.info;
	o.only_header = std::size(args) == 1 && args[0] == "--header";
	auto path = PathNamed(std::empty(f.path) ? "pinned_ce" : f.path);
	if (!path)
	{
		return cmt::Failure(path.error());
	}
	c.path = *path;
	c.device = f.device;
	if (o.help || o.info || o.only_header)
	{
		return o;
	}

	if (f.worker_numa_option->count())
	{
		o.worker_numa = f.numa == "on";
	}
	if (f.worker_cpus_option->count())
	{
		CMT_RETURN_IF_ERROR(CheckWorkerCpus(f.worker_cpus));
		o.worker_cpus = f.worker_cpus;
	}
	if ((o.worker_numa.has_value() || o.worker_cpus.has_value()) && f.cpu != -1)
	{
		return Failure(Error::Kind::Usage, "--cpu cannot be combined with worker placement flags");
	}

	for (const auto& [k, opts] : f.knob_opts)
	{
		for (const CLI::Option* op : opts)
		{
			if (op->count() > 0)
			{
				c.set_knobs.Set(k);
			}
		}
	}
	if (!std::empty(f.dir))
	{
		auto value = ParseDir(f.dir);
		if (!value)
		{
			return cmt::Failure(value.error());
		}
		c.dir = std::move(*value);
	}
	if (!std::empty(f.pages))
	{
		auto value = ParsePages(f.pages);
		if (!value)
		{
			return cmt::Failure(value.error());
		}
		c.pages = std::move(*value);
	}
	if (!std::empty(f.alloc))
	{
		auto value = ParseAlloc(f.alloc);
		if (!value)
		{
			return cmt::Failure(value.error());
		}
		c.alloc = std::move(*value);
	}
	if (!std::empty(f.fill))
	{
		c.cpu_copy = f.fill != "none";
		if (c.cpu_copy)
		{
			auto loop = ParseLoopName(NativeArch(), f.fill);
			if (!loop)
			{
				return UsageError("unknown CPU copy loop; see --fill in --help");
			}
			c.cpu_loop = *loop;
		}
	}
	if (!std::empty(f.slot_bytes))
	{
		auto value = ParseSize(f.slot_bytes);
		if (!value)
		{
			return cmt::Failure(value.error());
		}
		c.slot_bytes = std::move(*value);
	}
	if (!std::empty(f.pin_step))
	{
		auto value = ParseSize(f.pin_step);
		if (!value)
		{
			return cmt::Failure(value.error());
		}
		c.pin_step = std::move(*value);
	}
	if (!std::empty(f.unpin))
	{
		auto value = ParseUnpin(f.unpin);
		if (!value)
		{
			return cmt::Failure(value.error());
		}
		c.unpin = std::move(*value);
	}
	if (!std::empty(f.min_bytes))
	{
		auto value = ParseSize(f.min_bytes);
		if (!value)
		{
			return cmt::Failure(value.error());
		}
		c.min_bytes = std::move(*value);
	}
	c.threads = f.threads, c.slots = f.slots, c.samples = f.samples, c.cpu = f.cpu;
	c.budget_s = f.budget_s, c.tag = f.tag;
	c.flush = !f.no_flush, c.verify = !f.no_verify, c.wc_slots = f.wc, c.dma = true;
	{
		auto value = ParseSizes(f.sizes);
		if (!value)
		{
			return cmt::Failure(value.error());
		}
		c.sizes = std::move(*value);
	}
	CMT_RETURN_IF_ERROR(c.Validate());
	return o;
}
catch (const CLI::Error& error)
{
	return CliFailure(error);
}
catch (const std::bad_alloc&)
{
	Panic("CLI allocation failed");
}

Result<void> Options::ApplyWorkerPlacement() const
{
	if (worker_numa && setenv("CMT_WORKER_NUMA", *worker_numa ? "1" : "0", 1) != 0)
	{
		return OsError("cannot set worker NUMA CPU placement");
	}
	if (worker_cpus && setenv("CMT_WORKER_CPUS", worker_cpus->c_str(), 1) != 0)
	{
		return OsError("cannot set explicit worker CPU placement");
	}
	return {};
}

Result<TuneArgs> ParseTuneArgs(const std::vector<std::string>& args)
try
{
	CLI::App app{"Compare copy strategies for each size and direction; write measured tune.json", "cmt tune"};
	app.usage("Usage: cmt tune [OPTIONS] [-- COMMAND [ARGS...]]");
	app.set_help_flag();
	app.option_defaults()->take_last();
	TuneArgs a;
	const auto separator = std::find(args.begin(), args.end(), "--");
	const std::vector<std::string> flags(args.begin(), separator);
	std::string search = "quick", dir = "both", min_bytes;
	std::string progress_bar = "on";
	std::string host_alloc = "on";
	std::string pages = "small";
	std::string numa;
	std::string worker_cpus;
	std::vector<CLI::Option*> advanced;
	app.add_option("--search", search, "instant, quick, full or full_with_kernel (default: quick)")
		->group("Search")
		->check(CLI::IsMember({"instant", "quick", "full", "full_with_kernel"}));
	app.add_option("--tuning-params", a.tuning_params,
				   "Override the search grid: \"key=value ...\"; see the defaults and examples below")
		->group("Search");
	app.add_option("--direction", dir, "cpu-to-gpu (or htod), gpu-to-cpu (or dtoh), or both (default: both)")
		->group("Search")
		->check(CLI::IsMember({"cpu-to-gpu", "gpu-to-cpu", "htod", "dtoh", "both"}));
	app.add_option("--max-pinned-mib", a.max_pinned_mib,
				   "Skip layouts whose threads x buffers x buffer_size exceed this (default: 1024)")
		->group("Search")
		->check(CLI::Range(1, static_cast<int>(tune::kMaxStagingMiB)));
	std::size_t registration_limit_mib = 0;
	auto* registration_cap_option =
		app.add_option("--max-registered-mib", registration_limit_mib,
					   "full_with_kernel: limit on the application memory progressive pinning registers at once, "
					   "in MiB, measured and then applied by `cmt run` (default: none; a copy pins up to its own "
					   "size, within its GPU's memory, and releases it before returning)")
			->check(CLI::Range(static_cast<int>(tune::kMinRegistrationMiB),
							   static_cast<int>(kMaxBytes / (1024 * 1024))))
			->group("Search");
	app.add_option("--max-processes", a.max_processes,
				   "Maximum benchmark child processes allowed before starting (default: 4096)")
		->group("Search")
		->check(CLI::PositiveNumber);
	app.add_option("--numa", numa,
				   "on or off: run each measurement process on the GPU's NUMA node "
				   "(caller, buffers, staging pool and workers); tune.json records it and "
				   "`cmt run` applies the same (default: on with two or more NUMA nodes)")
		->check(CLI::IsMember({"on", "off"}))
		->group("Measurements");
	auto* worker_cpus_option =
		app.add_option("--worker-cpus", worker_cpus,
					   "Worker CPU IDs/ranges, e.g. 2,4-7; overrides automatic NUMA CPU placement")
			->group("Measurements");
	auto* pages_option = app.add_option("--pages", pages,
										"Caller buffer: small or thp (default: small; full_with_kernel: thp "
										"only); THP is a request, not guaranteed backing. CUDAHostAlloc "
										"staging buffers are unchanged")
							 ->check(CLI::IsMember({"small", "thp"}))
							 ->group("Measurements");
	app.add_option("--device", a.device, "CUDA device index (default: 0)")
		->group("Measurements")
		->check(CLI::Range(0, 1024));
	app.add_option("--host-alloc", host_alloc,
				   "on or off: quick and full also time one 1 GiB cudaHostAlloc (at most --max-pinned-mib) in "
				   "fresh processes, the driver's against the hook's (`cmt run --cuda-host-alloc-hook`), about "
				   "10 s; `cmt run --config` then uses the hook only if it was clearly faster. Where THP is "
				   "madvise or always the hook's private huge-page memory is timed too, but never chosen: a fork() "
				   "child gets a copy of it, not a share (default: on; instant never measures it)")
		->check(CLI::IsMember({"on", "off"}))
		->group("Measurements");
	app.add_option("--out", a.out_dir, "Results directory (default: $XDG_CACHE_HOME/cmt or ~/.cache/cmt)")
		->group("Output");
	app.add_flag("--verbose", a.verbose, "Show measurement details and launch decisions")
		->group("Output");
	app.add_flag("--debug", a.debug, "Log the launched application's hook decisions")->group("Output");
	app.add_flag("--preview", a.preview, "Print the exact grid and process count; run nothing, create no files")
		->group("Output");
	app.add_flag("--resume", a.resume, "Continue the same search in --out; finished processes are not rerun")
		->group("Output");
	app.add_option("--progress-bar", progress_bar,
				   "Show tuning progress: on or off (default: on). Redirected output uses plain lines")
		->group("Output")
		->check(CLI::IsMember({"on", "off"}));
	advanced.push_back(
		app.add_option("--hook", a.hook, "Hook library (default: beside cmt)")->group("Advanced"));
	advanced.push_back(
		app.add_option("--benchmark-exe", a.benchmark_exe,
						   "The gpu_copy program (built on Google Benchmark) that times each setting (default: "
						   "gpu_copy beside cmt)")
			->group("Advanced"));
	auto* min_bytes_option =
		app.add_option("--min-bytes", min_bytes,
					   "A sample copies at least this many bytes, repeating the copy, at most 2^20 copies per "
					   "sample (default: 1, one copy)")
			->check(SizeArgument(false))
			->group("Advanced");
	advanced.push_back(min_bytes_option);
	advanced.push_back(app.add_flag("--dry-run", a.dry_run, "Print every planned process without running it")
						   ->group("Advanced"));
	bool help = false, advanced_help = false;
	app.add_flag("-h,--help", help, "Show concise tuning help")->group("Output");
	app.add_flag("--help-advanced", advanced_help, "Show help including advanced controls")->group("Output");
	const auto defaults = [](tune::Search kind)
	{
		auto grid = tune::ParseTuningParams(kind, "");
		if (!grid)
		{
			Panic("default tuning grid is invalid");
		}
		return tune::FormatGrid(kind, *grid);
	};
	app.footer(fmt::format(
		"Load a complete saved tune or tune first; -- COMMAND then launches through cmt run.\n"
		"Explicit measurement controls or --resume validate/resume existing measurements.\n"
		"Preview/dry-run never launch the application.\n\n"
		"Searches and their default --tuning-params:\n"
		"instant: a probe estimates the copying threads; two staged settings per direction at them.\n{}"
		"quick: instant's probe and settings, six more, then a coordinate descent.\n{}"
		"full: every layout of the grid, each in its own fresh process that times all sizes.\n{}"
		"full_with_kernel: full plus progressive pinning (register chunks during a copy, release them at "
		"copy end), timing THP caller buffers. Needs THP enabled and the verified host-registration modules "
		"loaded with `cmt patch`; tune never switches modules. No limit by default: a copy pins up to its "
		"own size of host memory until it returns; --max-registered-mib sets one.\n\n"
		"--tuning-params=\"key=value key=value\": lists are comma-separated; sizes use K/M/G (powers of "
		"1024). Keys: sizes, threads (copying threads, caller included), buffers (per thread), buffer_size, "
		"loops (all or names), memory (cached, wc or cached,wc), samples (per size per process), rounds "
		"(fresh processes per setting), tie (a selection band such as 2%: among settings within it of the "
		"best median, the smallest pinned pool wins; --resume with another tie re-selects, measuring "
		"nothing). quick accepts sizes, memory, samples, rounds and tie; instant accepts none. Selection "
		"needs samples x rounds >= 2.\n\n"
		"Examples:\n"
		"  cmt tune --out tune-quick\n"
		"  cmt tune --out tune-full --search full --preview\n"
		"  cmt tune --out tune-full --search full --tuning-params=\"threads=2,4 buffer_size=1M,4M\"\n"
		"  cmt tune --out tune-big --search full --tuning-params=\"sizes=64M,1G loops=movsb,memcpy samples=3\"\n"
		"  cmt tune --out tune-quick-wc --tuning-params=\"sizes=1M,1G memory=cached,wc rounds=2\"\n"
		"  cmt tune --out tune-full --search full --resume\n"
		"  cmt tune --out tune-full --search full --resume --tuning-params=\"tie=2%\"",
		defaults(tune::Search::Instant), defaults(tune::Search::Quick), defaults(tune::Search::Full)));
	if (WantsHelp(flags))
	{
		if (std::none_of(flags.begin(), flags.end(), IsAdvancedHelp))
		{
			for (auto* option : advanced)
			{
				option->group("");
			}
			for (const char* name : {"--max-registered-mib", "--max-processes", "--numa", "--worker-cpus",
								 "--pages", "--host-alloc", "--progress-bar"})
			{
				app.get_option(name)->group("");
			}
			app.get_option("--tuning-params")->description("Override grid settings; see --help-advanced");
			app.footer("Examples:\n"
					   "  cmt tune -- python app.py\n"
					   "  cmt tune --out tune\n"
					   "  cmt tune --out tune --resume\n"
					   "  cmt tune --out tune -- python app.py\n\n"
					   "Load a complete saved tune or tune first; -- COMMAND then launches it.\n"
					   "Explicit measurement controls or --resume validate/resume saved results.\n"
					   "Preview/dry-run never launch the application.\n"
					   "Use --help-advanced for grids, placement, allocation and measurement controls.");
		}
		a.help = app.help();
		return a;
	}
	ParseCli(app, flags);
	if (separator != args.end())
	{
		if (separator + 1 == args.end() || (separator + 1)->empty())
		{
			return UsageError("tune: expected -- followed by the application command");
		}
		a.command.assign(separator + 1, args.end());
	}
	if (a.out_dir.empty())
	{
		auto cache = CacheDirectoryPath();
		if (!cache)
		{
			return Failure(cache.error());
		}
		a.out_dir = *cache;
	}
	for (const char* name : {"--search", "--tuning-params", "--direction", "--max-pinned-mib",
							 "--max-registered-mib", "--max-processes", "--numa", "--worker-cpus", "--pages",
							 "--device", "--host-alloc", "--benchmark-exe", "--min-bytes"})
	{
		a.measurement_options |= app.get_option(name)->count() != 0;
	}
	if (numa == "off")
	{
		a.env.push_back("CMT_WORKER_NUMA=0");
	}
	if (worker_cpus_option->count())
	{
		CMT_RETURN_IF_ERROR(CheckWorkerCpus(worker_cpus));
		a.env.push_back("CMT_WORKER_CPUS=" + worker_cpus);
	}
	if (min_bytes_option->count() != 0)
	{
		auto value = ParseSize(min_bytes);
		if (!value)
		{
			return cmt::Failure(value.error());
		}
		a.min_bytes = std::move(*value);
	}
	a.progress_bar = progress_bar == "on";
	a.host_alloc = host_alloc == "on";
	a.search = search == "instant" ? tune::Search::Instant
			   : search == "quick" ? tune::Search::Quick
			   : search == "full"  ? tune::Search::Full
								   : tune::Search::FullWithKernel;
	if (a.search == tune::Search::FullWithKernel)
	{
		if (pages_option->count() != 0 && pages != "thp")
		{
			return UsageError(
				"full_with_kernel times only THP caller buffers (the patched modules' gain is in "
				"pinning THP-backed memory); drop --pages small");
		}
		pages = "thp";
	}
	a.pages = pages == "thp" ? Pages::Thp : Pages::Small;
	auto grid = tune::ParseTuningParams(a.search, a.tuning_params);
	if (!grid)
	{
		return Failure(grid.error());
	}
	a.grid = std::move(*grid);
	if (registration_cap_option->count() != 0 && a.search != tune::Search::FullWithKernel)
	{
		return UsageError("--max-registered-mib requires --search full_with_kernel");
	}
	if (registration_cap_option->count() != 0)
	{
		a.max_registered_mib = registration_limit_mib;
	}
	if (dir == "htod" || dir == "cpu-to-gpu")
	{
		a.dirs = {cmt::Dir::HtoD};
	}
	else if (dir == "dtoh" || dir == "gpu-to-cpu")
	{
		a.dirs = {cmt::Dir::DtoH};
	}
	return a;
}
catch (const CLI::Error& error)
{
	return CliFailure(error);
}
catch (const std::bad_alloc&)
{
	Panic("CLI allocation failed");
}

Result<TraceArgs> ParseTraceArgs(const std::vector<std::string>& args)
try
{
	TraceArgs a;
	const auto separator = std::find(args.begin(), args.end(), "--");
	CLI::App app{"Record CUDA copy sizes and pointer memory types without changing copy strategies",
				 "cmt trace"};
	app.usage("Usage: cmt trace --out FILE [--hook LIBRARY] -- COMMAND [ARGS...]");
	app.set_help_flag();
	app.add_option("--out", a.out, "New JSONL output file; existing files are refused")->required();
	app.add_option("--hook", a.hook, "Trace library (default: libcmt_trace.so beside cmt)");
	app.footer("Example: cmt trace --out copies.jsonl -- python train.py --model example\n"
			   "Tracing is opt-in and changes timing. Byte counts are requested bytes; failed API calls\n"
			   "are not successful transfers. Memory types describe pointer endpoints, not every page.\n"
			   "Coverage: supported 1D runtime/driver APIs; not 2D/3D, peer/batch/array APIs, graphics, "
			   "or graph replay.\n"
			   "Only the launched process is traced; fork/exec child processes are excluded.\n"
			   "No records does not prove there were no GPU copies.\n"
			   "Exit status: the application's; 125 when cmt trace itself fails (a usage error, an existing\n"
			   "--out, or a trace that did not complete although the application exited 0).");
	bool help = false;
	app.add_flag("-h,--help", help, "Show trace help");
	std::vector<std::string> options(args.begin(), separator);
	if (WantsHelp(options))
	{
		a.help = app.help();
		return a;
	}
	ParseCli(app, options);
	if (separator == args.end() || separator + 1 == args.end())
	{
		return Failure(Error::Kind::Usage, "trace: expected -- followed by the application command");
	}
	if (a.out.empty() || a.out == "-")
	{
		return Failure(Error::Kind::Usage, "trace: --out must name a new regular file");
	}
	a.command.assign(separator + 1, args.end());
	if (a.command.front().empty())
	{
		return Failure(Error::Kind::Usage, "trace: the application command is empty");
	}
	return a;
}
catch (const CLI::Error& error)
{
	return CliFailure(error);
}
catch (const std::bad_alloc&)
{
	Panic("CLI allocation failed");
}

Result<KernelArgs> ParseKernelArgs(const std::vector<std::string>& args)
try
{
	KernelArgs options;
	const auto action = std::find_if(args.begin(), args.end(),
									 [](const auto& word)
									 {
										 return word == "patch" || word == "unpatch" || word == "inspect" ||
												word == "build" || word == "run" || word == "restore";
									 });
	CLI::App app{"Explicit kernel module workflow; never selected by tuning", "cmt kernel"};
	app.set_help_flag();
	app.add_option("--script", options.script, "Reviewed kmod.sh location (must precede the action)");
	bool help = false;
	app.add_flag("-h,--help", help, "Show kernel workflow help");
	app.footer(
		"cmt patch --check\n"
		"cmt patch --build-only --work-dir NEW_DIRECTORY\n"
		"sudo ./cmt patch [--jobs N] [--work-dir DIRECTORY] [--state FILE]\n"
		"sudo ./cmt unpatch [--state FILE] [--measurement-lock LOCK]\n"
		"cmt kernel inspect\n"
		"cmt kernel build --measurement-lock LOCK --source SOURCE --out NEW_DIRECTORY\n"
		"cmt kernel run --measurement-lock LOCK --set SET --restore-set ORIGINAL_SET --state NEW_STATE.json "
		"-- COMMAND\n"
		"cmt kernel run --measurement-lock LOCK --set HOST_REGISTRATION_SET --restore-set ORIGINAL_SET "
		"--state "
		"NEW_STATE.json --large-system-pages "
		"-- COMMAND\n"
		"cmt kernel restore --state STATE.json\n"
		"Loading/restoring requires root, idle NVIDIA devices, and verified original module artifacts.\n"
		"patch --jobs: parallel compile jobs (default 4). --work-dir: the NVIDIA source and build cache\n"
		"(default ~/.cache/cmt/kernel, and /var/cache/cmt/kernel as root). --state: the recovery journal\n"
		"patch writes and unpatch reads to restore the original modules (default\n"
		"/var/lib/cmt/kernel/state.json). --measurement-lock: the lock `cmt tune` holds while measuring\n"
		"(default /run/user/UID/cmt-measurements.lock, or /tmp/cmt-UID/cmt-measurements.lock in a private\n"
		"directory without /run/user/UID; UID is SUDO_UID under sudo; CMT_MEASUREMENT_LOCK_PATH overrides it).\n"
		"sudo's secure_path does not search the current directory: run ./cmt, or its full path.\n"
		"patch matches the installed NVIDIA driver release, not the CUDA toolkit. No process is killed.\n"
		"Use ACTION --help for kernel-owner options and supported releases.");
	std::vector<std::string> flags(args.begin(), action);
	if (WantsHelp(flags))
	{
		options.help = app.help();
		return options;
	}
	ParseCli(app, flags);
	if (action == args.end())
	{
		return UsageError("kernel requires patch, unpatch, inspect, build, run or restore");
	}
	options.arguments.assign(action, args.end());
	return options;
}
catch (const CLI::Error& error)
{
	return CliFailure(error);
}
catch (const std::bad_alloc&)
{
	Panic("CLI allocation failed");
}

Result<ExportArgs> ParseExportArgs(const std::vector<std::string>& args)
try
{
	ExportArgs options;
	CLI::App app{"Write a shareable copy of a tune.json: the selection, a machine key and a per-size summary",
				 "cmt export"};
	app.set_help_flag();
	app.add_option("tune_json", options.tune, "A complete tune.json whose line 1 matches its runs")->required();
	app.add_option("--out", options.out, "The shared tune to write; an existing file is not replaced")->required();
	bool help = false;
	app.add_flag("-h,--help", help, "Show export help");
	app.footer("Shared: the selection rows, the request fields selection reads, CPU/GPU/link/driver model facts, "
			   "per-size driver and selected GB/s, the tool version, the source file's sha256.\n"
			   "Not shared: runs, hostname, kernel command line, GPU UUID, PCI addresses, paths, environment.\n"
			   "A string that looks like a UUID, an e-mail or user@host, an IPv4/IPv6 or MAC address, a PCI\n"
			   "address, or a home or other absolute path refuses the export.\n"
			   "Format and contribution instructions: tunes/README.md.");
	if (WantsHelp(args))
	{
		options.help = app.help();
		return options;
	}
	ParseCli(app, args);
	return options;
}
catch (const CLI::Error& error)
{
	return CliFailure(error);
}
catch (const std::bad_alloc&)
{
	Panic("CLI allocation failed");
}

Result<RunArgs> ParseRunArgs(const std::vector<std::string>& args)
try
{
	RunArgs options;
	const auto separator = std::find(args.begin(), args.end(), "--");
	CLI::App app{"Run an application with a measured copy configuration", "cmt run"};
	app.usage("Usage: cmt run (--config FILE|URL | --remote-config [URL]) [OPTIONS] -- COMMAND [ARGS...]");
	app.set_help_flag();
	auto* config = app.add_option("--config", options.config,
								  "tune.json measured on this machine, or an https URL of a shared tune")
					   ->type_name("<path|URL>");
	std::vector<std::string> remote;
	auto* remote_option =
		app.add_option(
			   "--remote-config", remote,
			   "Use a static website tune matching this CPU and GPU model; default: cudamemcpy.com")
			->expected(0, 1)
			->type_name("[URL]");
	config->excludes(remote_option);
	app.add_option("--hook", options.hook, "Hook library (default: beside cmt)");
	app.add_flag("--debug", options.debug, "Log the hook's configuration and selected copy paths");
	app.add_flag("--verbose", options.verbose, "Explain decisions from the saved tune");
	std::string numa;
	app.add_option(
		   "--numa", numa,
		   "on or off: the staged copy workers and their pool on the GPU's NUMA node (default: on with "
		   "two or more NUMA nodes, or as tune.json recorded). The application's threads and buffers "
		   "stay where it put them")
		->check(CLI::IsMember({"on", "off"}));
	std::string host_alloc;
	auto* host_alloc_option =
		app.add_option("--cuda-host-alloc-hook", host_alloc,
					   "on, off or private: on serves the application's cudaHostAlloc / cudaMallocHost of 2 MiB or "
					   "more with shared memory that several threads fault in and the driver then registers, "
					   "instead of the driver's single-threaded allocation. Write-combined requests and any failure "
					   "use the driver's allocation. private does the same with private memory, which gets huge "
					   "pages where THP is madvise or always, but which a fork() child gets a copy of, made at "
					   "fork(), instead of sharing it as it shares the driver's: only for applications that do not "
					   "fork with pinned memory they share. Default: on when the --config tune measured the hook "
					   "clearly faster (`cmt tune --host-alloc`), else off; private is never chosen for you")
			->check(CLI::IsMember({"on", "off", "private"}));
	app.add_option("--override", options.overrides,
				   "Manual run settings: KEY=VALUE[,KEY=VALUE]; repeatable; tune.json is unchanged")
		->take_all()
		->expected(1)
		->allow_extra_args(false);
	app.footer(
		"Staging overrides change existing staged selections only.\n"
		"Keys: pipeline_workers, slots_per_worker, chunk_bytes (decimal bytes), cpu_loop,\n"
		"      staging_memory=cached_pinned|wc_pinned, progressive_pinning=auto|on|off,\n"
		"      pinning_max_mib=N.\n"
		"progressive_pinning=auto uses measured winning pinning selections.\n"
		"on requests pinning explicitly, subject to verified kernel support and registration limits.\n"
		"off disables new registrations. Existing registration obligations still complete safely.\n"
		"pinning_max_mib limits the application memory pinned at once (as CMT_PINNING_MAX_MIB). Without it\n"
		"a pinned copy may pin up to its own size, within its GPU's memory; that host memory cannot be\n"
		"swapped or reclaimed until the copy returns.\n"
		"Duplicate settings are errors.\n"
		"pipeline_workers counts all copying threads, including the caller.\n"
		"Combined staging pools per CUDA context must fit the original tune.json max_pinned_mib budget.\n"
		"Overrides are manual and unmeasured; no setting file is written.\n"
		"Example: cmt run --config tune.json --override pipeline_workers=2,chunk_bytes=1048576 "
		"--override staging_memory=wc_pinned -- python train.py");
	bool help = false;
	app.add_flag("-h,--help", help, "Show application launch help");
	bool advanced_help = false;
	app.add_flag("--help-advanced", advanced_help, "Show allocation, placement and manual overrides");
	std::vector<std::string> flags(args.begin(), separator);
	if (WantsHelp(flags))
	{
		if (std::none_of(flags.begin(), flags.end(), IsAdvancedHelp))
		{
			for (const char* name : {"--hook", "--numa", "--cuda-host-alloc-hook", "--override"})
			{
				app.get_option(name)->group("");
			}
			remote_option->description("Download the matching shared tune (default server: cudamemcpy.com)");
			app.footer("Example: cmt run --config tune/tune.json -- python app.py\n"
					   "Use --help-advanced for allocation, placement and manual overrides.");
		}
		options.help = app.help();
		return options;
	}
	ParseCli(app, flags);
	if (remote_option->count() > 0)
	{
		const bool bare = std::empty(remote) || std::empty(remote.front());
		options.remote = bare ? std::string("https://cudamemcpy.com") : remote.front();
	}
	else if (std::empty(options.config))
	{
		return UsageError("run: --config FILE|URL or --remote-config [URL] is required");
	}
	if (!std::empty(numa))
	{
		options.overrides.push_back("numa=" + numa);
	}
	if (host_alloc_option->count() != 0)
	{
		options.cuda_host_alloc_hook = host_alloc;
	}
	if (separator == args.end() || separator + 1 == args.end() || (separator + 1)->empty())
	{
		return UsageError("run: expected -- followed by the application command");
	}
	options.command.assign(separator + 1, args.end());
	return options;
}
catch (const CLI::Error& error)
{
	return CliFailure(error);
}
catch (const std::bad_alloc&)
{
	Panic("CLI allocation failed");
}

} // namespace cmt
