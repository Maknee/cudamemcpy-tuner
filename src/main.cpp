#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string_view>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "cli.h"
#include "common/error.h"
#include "common/helpers.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "measure/copy.h"
#include "measure/metrics.h"
#include "log.h"
#include "machine.h"
#include "remote.h"
#include "runner.h"
#include "share.h"

namespace {

constexpr int kTraceOwnFailure = 125;

cmt::Result<int> TraceMain(const std::vector<std::string>& args)
{
	using namespace cmt;
	auto parsed_options = ParseTraceArgs(args);
	if (!parsed_options)
	{
		return cmt::Failure(parsed_options.error());
	}
	auto& options = *parsed_options;
	if (!options.help.empty())
	{
		CMT_RETURN_IF_ERROR(cmt::WriteStdout(options.help));
		return 0;
	}
	return RunTrace(options);
}

cmt::Result<int> TuneMain(const std::vector<std::string>& args)
{
	auto options = cmt::ParseTuneArgs(args);
	if (!options)
	{
		return cmt::Failure(options.error());
	}
	if (!options->help.empty())
	{
		CMT_RETURN_IF_ERROR(cmt::WriteStdout(options->help));
		return 0;
	}
	const bool launch = !options->command.empty();
	const auto config = (std::filesystem::path(options->out_dir) / "tune.json").string();
	const auto run_application = [&]() -> cmt::Result<int>
	{
		CMT_RETURN_IF_ERROR(cmt::WriteStdout(""));
		cmt::RunArgs run;
		run.config = config;
		run.hook = options->hook;
		run.command = options->command;
		run.debug = options->debug;
		run.verbose = options->verbose;
		return cmt::RunConfiguredApplication(run);
	};
	if (options->preview || options->dry_run)
	{
		if (launch)
		{
			std::fprintf(stderr, "cmt tune: preview only; application will not be launched\n");
		}
		return cmt::RunTuning(*options);
	}
	{
		auto saved = cmt::ReadTextFile(config, cmt::tuning::kMaxConfigBytes);
		if (saved)
		{
			JsonParser parser;
			Json document;
			std::string_view state;
			if (parser.parse(*saved).get(document) || document["state"].get(state))
			{
				std::fprintf(stderr, "cmt tune: cannot read saved tuning results: %s\n", config.c_str());
				return cmt::Failure(cmt::Error::Kind::Verify, "invalid saved tune; choose another --out");
			}
			if (state == "complete" && !options->resume && !options->measurement_options)
			{
				return run_application();
			}
			options->resume = true;
		}
		else if (saved.error().GetCode() != ENOENT)
		{
			std::fprintf(stderr, "cmt tune: cannot read %s\n", config.c_str());
			return cmt::Failure(saved.error());
		}
	}
	auto machine = cmt::DetectThisMachine();
	if (!machine)
	{
		return cmt::Failure(machine.error());
	}
	std::fprintf(stderr, "cmt tune: Tuning%s\n",
				 options->resume ? "... (resuming saved measurements)" : "...");
	if (launch && options->verbose)
	{
		std::fprintf(stderr, "cmt tune: will launch %s after tuning succeeds\n", options->command.front().c_str());
	}
	auto tuned = cmt::RunTuning(*options);
	if (!tuned || *tuned != 0 || !launch)
	{
		return tuned;
	}
	return run_application();
}

cmt::Result<int> RunMain(const std::vector<std::string>& args)
{
	using namespace cmt;
	auto options = ParseRunArgs(args);
	if (!options)
	{
		return Failure(options.error());
	}
	if (!options->help.empty())
	{
		CMT_RETURN_IF_ERROR(WriteStdout(options->help));
		return 0;
	}
	if (!options->remote && !IsUrl(options->config))
	{
		return RunConfiguredApplication(*options);
	}
	auto here = DetectThisMachine();
	if (!here)
	{
		return Failure(here.error());
	}
	RunArgs local = *options;
	if (options->remote)
	{
		auto tune = DownloadClosest(*options->remote, here->cpu, here->gpus.front());
		if (!tune)
		{
			return Failure(tune.error());
		}
		std::fprintf(stderr, "cmt run: using tune from %s / %s (exact match)\n", tune->gpu.c_str(),
					 tune->cpu.c_str());
		local.config = std::move(tune->path);
	}
	else
	{
		auto path = DownloadShare(options->config);
		if (!path)
		{
			return Failure(path.error());
		}
		local.config = std::move(*path);
	}
	return RunConfiguredApplication(local, std::move(*here));
}

cmt::Result<int> CheckMain(const std::vector<std::string>& args)
{
	using namespace cmt;
	if (std::find_if(args.begin(), args.end(), [](const std::string& word)
					 { return word == "-h" || word == "--help"; }) != args.end())
	{
		CMT_RETURN_IF_ERROR(WriteStdout(CheckUsage()));
		return 0;
	}
	const bool write = !args.empty() && args.front() == "--write";
	if (args.size() != (write ? 2u : 1u))
	{
		return UsageError("usage: cmt check [--write] TUNE_JSON");
	}
	if (IsUrl(args.back()))
	{
		if (write)
		{
			return UsageError("--write changes a local file; a URL is checked only");
		}
		auto path = DownloadShare(args.back());
		if (!path)
		{
			return Failure(path.error());
		}
		CMT_RETURN_IF_ERROR(WriteStdout(fmt::format("OK: a shared tune, validated and saved as {}\n", *path)));
		return 0;
	}
	auto text = ReadTextFile(args.back(), tuning::kMaxConfigBytes);
	if (text && share::IsSelectionOnly(*text))
	{
		if (write)
		{
			return UsageError("a selection-only tune holds no runs; --write cannot recompute its selection");
		}
		auto info = share::ValidateSelectionOnly(*text);
		if (!info)
		{
			return Failure(info.error());
		}
		CMT_RETURN_IF_ERROR(WriteStdout(fmt::format(
			"OK: selection only (from {}) of {} / {}; its rows can't be re-derived: it holds no runs\n", info->source,
			info->key.gpu, info->key.cpu)));
		return 0;
	}
	if (text && share::IsShare(*text))
	{
		if (write)
		{
			return UsageError("a shared tune holds no runs; --write cannot recompute its selection");
		}
		auto info = share::Validate(*text);
		if (!info)
		{
			return Failure(info.error());
		}
		CMT_RETURN_IF_ERROR(WriteStdout(fmt::format(
			"OK: a shared tune (cmt.tune-share/1) of {} / {}, driver {}, {}; its rows cannot be re-derived: a "
			"shared tune holds no runs\n",
			info->key.gpu, info->key.cpu, info->key.driver, info->tier)));
		return 0;
	}
	auto expected = text ? Reselect(*text) : Failure(text.error());
	if (!expected)
	{
		return Failure(expected.error());
	}
	JsonParser parser;
	Json document;
	if (parser.parse(*text).get(document) || !tuning::ReadHostAllocation(document))
	{
		return Failure(Error::Kind::Verify, "tune.json's host_allocation is not as cmt writes it");
	}
	if (*expected != *text && !write)
	{
		return Failure(Error::Kind::Verify, "line 1 differs from the runs; run `cmt check --write`");
	}
	if (*expected != *text)
	{
		CMT_RETURN_IF_ERROR(WriteTextFileAtomic(args.back(), *expected));
	}
	CMT_RETURN_IF_ERROR(WriteStdout("OK\n"));
	return 0;
}

cmt::Result<int> ExportMain(const std::vector<std::string>& args)
{
	using namespace cmt;
	auto options = ParseExportArgs(args);
	if (!options)
	{
		return Failure(options.error());
	}
	if (!options->help.empty())
	{
		CMT_RETURN_IF_ERROR(WriteStdout(options->help));
		return 0;
	}
	const auto exists = [&]
	{
		std::fprintf(stderr, "cmt export: %s exists; remove it or choose another --out\n", options->out.c_str());
		return UsageError("--out names an existing file");
	};
	std::error_code status_error;
	if (std::filesystem::exists(std::filesystem::symlink_status(options->out, status_error)))
	{
		return exists();
	}
	auto text = ReadTextFile(options->tune, tuning::kMaxConfigBytes);
	auto shared = text ? share::Export(*text) : Failure(text.error());
	if (!shared)
	{
		std::fprintf(stderr, "cmt export: cannot share %s\n", options->tune.c_str());
		return Failure(shared.error());
	}
	auto info = share::Validate(*shared);
	if (!info)
	{
		return Failure(info.error());
	}
	if (auto written = WriteNewTextFileAtomic(options->out, *shared); !written)
	{
		if (written.error().GetKind() == Error::Kind::Os && written.error().GetCode() == EEXIST)
		{
			return exists();
		}
		return Failure(written.error());
	}
	CMT_RETURN_IF_ERROR(WriteStdout(fmt::format("wrote {} ({} bytes): {} / {}, driver {}, {}\n"
												"to contribute it, add it to the repository as tunes/{}\n",
												options->out, std::size(*shared), info->key.gpu, info->key.cpu,
												info->key.driver, info->tier,
												share::RepositoryPath(*info, *shared))));
	return 0;
}

void WarnIfOthersCanChange(const std::string& script)
{
	std::error_code error;
	auto path = std::filesystem::absolute(script, error).lexically_normal();
	if (error)
	{
		return;
	}
	for (bool file = true;; file = false)
	{
		struct stat status{};
		if (lstat(path.c_str(), &status) == 0)
		{
			const bool sticky_directory = !file && S_ISDIR(status.st_mode) && (status.st_mode & S_ISVTX);
			if (status.st_uid != 0 || (!sticky_directory && (status.st_mode & (S_IWGRP | S_IWOTH))))
			{
				const auto why = status.st_uid != 0
									 ? "owned by uid " + std::to_string(status.st_uid) + ", not root"
									 : std::string("writable by group or others");
				std::fprintf(stderr,
							 "cmt: note: running %s as root, but %s is %s: whoever can change it can change what "
							 "root runs. A root-owned install (e.g. /opt/cmt) rules that out\n",
							 script.c_str(), path.c_str(), why.c_str());
				return;
			}
		}
		if (!path.has_parent_path() || path == path.parent_path())
		{
			return;
		}
		path = path.parent_path();
	}
}

cmt::Result<int> KernelMain(const std::vector<std::string>& args)
{
	using namespace cmt;
	auto options = ParseKernelArgs(args);
	if (!options)
	{
		return Failure(options.error());
	}
	if (!options->help.empty())
	{
		CMT_RETURN_IF_ERROR(WriteStdout(options->help));
		return 0;
	}
	if (options->script.empty())
	{
		auto executable = SelfExe();
		if (!executable)
		{
			return Failure(executable.error());
		}
		options->script =
			(std::filesystem::path(*executable).parent_path() / "kernel_patches/nvidia-open/kmod.sh")
				.string();
	}
	auto exists = IsRegularFile(options->script);
	if (!exists)
	{
		return Failure(exists.error());
	}
	if (!*exists)
	{
		return UsageError("kernel script does not exist");
	}
	if (geteuid() == 0)
	{
		WarnIfOthersCanChange(options->script);
	}
	ProcessCommand command;
	command.executable = "/bin/bash";
	command.arguments = {options->script};
	command.arguments.insert(command.arguments.end(), options->arguments.begin(), options->arguments.end());
	return RunApplication(command);
}

int ReportError(const cmt::Error& error, std::string_view command = {})
{
	if (error.GetCode() != 0)
	{
		if (error.GetKind() == cmt::Error::Kind::Os)
		{
			spdlog::error("cmt: {} (OS error {}: {})", error.Message(), error.GetCode(),
						  std::strerror(error.GetCode()));
		}
		else
		{
			spdlog::error("cmt: {} (code {})", error.Message(), error.GetCode());
		}
	}
	else
	{
		spdlog::error("cmt: {}", error.Message());
	}
	if (error.IsUsage())
	{
		const std::initializer_list<std::string_view> commands = {
			"tune", "run", "check", "export", "trace", "bench", "patch", "unpatch", "kernel"};
		if (std::find(commands.begin(), commands.end(), command) != commands.end())
		{
			spdlog::error("(see cmt {} --help)", command);
		}
		else
		{
			spdlog::error("(see cmt --help)");
		}
	}
	return error.ExitCode();
}

int Report(const cmt::Result<int>& result, std::string_view command)
{
	return result ? *result : ReportError(result.error(), command);
}

} // namespace

static cmt::Result<void> PutLine(const std::string& line)
{
	return cmt::WriteStdout(line + "\n");
}

static cmt::Result<int> BenchMain(const std::vector<std::string>& args)
{
	using namespace cmt;
	auto options = ParseArgs(args);
	if (!options)
	{
		return cmt::Failure(options.error());
	}
	auto& o = *options;
	if (o.help)
	{
		CMT_RETURN_IF_ERROR(cmt::WriteStdout(UsageText()));
		return 0;
	}
	if (o.HeaderOnly())
	{
		CMT_RETURN_IF_ERROR(PutLine(CsvHeader()));
		return 0;
	}
	CMT_RETURN_IF_ERROR(o.ApplyWorkerPlacement());
	const auto prepared = PrepareBenchmark(o.cfg);
	if (!prepared)
	{
		return Failure(prepared.error());
	}
	if (!prepared->has_value())
	{
		spdlog::warn("cmt: no CUDA device visible");
		return 77;
	}
	const DeviceInfo& device = prepared->value();
	if (o.info)
	{
		CMT_RETURN_IF_ERROR(cmt::WriteStdout(Describe(device)));
		return 0;
	}
	if (o.header)
	{
		CMT_RETURN_IF_ERROR(PutLine(CsvHeader()));
	}
	CMT_RETURN_IF_ERROR(Run(o.cfg, device, [](const MeasurementRow& row) { return PutLine(CsvRow(row)); }));
	return 0;
}

static cmt::Result<std::string> VersionText()
{
	using namespace cmt;
	JsonWriter writer;
	WriteToolJson(writer);
	auto text = FinishJson(writer);
	if (!text)
	{
		return Failure(text.error());
	}
	JsonParser parser;
	Json tool;
	std::string_view version, commit;
	if (parser.parse(*text).get(tool) || tool["version"].get(version))
	{
		return Failure(Error::Kind::Verify, "the build information is malformed");
	}
	bool dirty = false;
	const bool known_commit = !tool["commit"].get(commit) && !std::empty(commit) && commit != "none";
	const bool known_dirty = !tool["dirty"].get(dirty);
	return fmt::format("cmt {} (commit {}{})\n", version, known_commit ? commit : "unknown",
					   known_dirty && dirty ? ", with uncommitted changes" : "");
}

static cmt::Result<int> Main(const std::vector<std::string>& args)
{
	using namespace cmt;
	if (args.empty())
	{
		std::fputs(TopLevelUsage().c_str(), stderr);
		return 2;
	}
	const std::string& command = args.front();
	const std::vector<std::string> remaining(args.begin() + 1, args.end());
	if (command == "--tune-probe")
	{
		if (args.size() != 4 || args[2] != "--out")
		{
			return UsageError("invalid internal probe arguments");
		}
		return RunTuneProbe(args[1], args[3]);
	}
	if (command == "--tune-host-alloc")
	{
		if (args.size() != 4 || args[2] != "--out")
		{
			return UsageError("invalid internal host allocation arguments");
		}
		return RunTuneHostAlloc(args[1], args[3]);
	}
	if (command == "help" && args.size() > 1)
	{
		if (args.size() != 2)
		{
			return UsageError("usage: cmt help COMMAND");
		}
		return Main({args[1], "--help"});
	}
	if (command == "-h" || command == "--help" || command == "help")
	{
		CMT_RETURN_IF_ERROR(WriteStdout(TopLevelUsage()));
		return 0;
	}
	if (command == "--version")
	{
		auto version = VersionText();
		if (!version)
		{
			return Failure(version.error());
		}
		CMT_RETURN_IF_ERROR(WriteStdout(*version));
		return 0;
	}
	if (command == "trace")
	{
		auto traced = TraceMain(remaining);
		if (!traced)
		{
			ReportError(traced.error(), "trace");
			return kTraceOwnFailure;
		}
		return traced;
	}
	if ((command == "check" || command == "kernel") && !InternalCommandsEnabled())
	{
		return UsageError("internal commands require CMT_INTERNAL_ENABLE=1");
	}
	if (command == "patch" || command == "unpatch")
	{
		return KernelMain(args);
	}
	if (command == "kernel")
	{
		return KernelMain(remaining);
	}
	if (command == "run")
	{
		return RunMain(remaining);
	}
	if (command == "tune")
	{
		return TuneMain(remaining);
	}
	if (command == "check")
	{
		return CheckMain(remaining);
	}
	if (command == "export")
	{
		return ExportMain(remaining);
	}
	if (command == "bench")
	{
		return BenchMain(remaining);
	}
	if (!command.empty() && command.front() == '-')
	{
		std::fprintf(stderr, "cmt: %s is not a command; the bandwidth benchmark's options go after `cmt bench`\n",
					 command.c_str());
	}
	else
	{
		std::fprintf(stderr, "cmt: unknown command %s\n", command.c_str());
	}
	return UsageError("unknown command");
}

int main(int argc, char** argv)
{
	cmt::InstallFatalAllocationHandler();
	const std::vector<std::string> args(argv + 1, argv + argc);
	return Report(Main(args), args.empty() ? std::string_view{} : args.front());
}
