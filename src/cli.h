#pragma once

#include <climits>
#include <limits>
#include <optional>
#include <algorithm>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <vector>
#include "common/aliases.h"
#include "measure/metrics.h"
#include "runner.h"
#include "tuner.h"

namespace cmt {

struct KernelArgs
{
	std::string script;
	std::string help;
	std::vector<std::string> arguments;
};

Result<KernelArgs> ParseKernelArgs(const std::vector<std::string>& args);

Result<RunArgs> ParseRunArgs(const std::vector<std::string>& args);

struct ExportArgs
{
	std::string tune;
	std::string out;
	std::string help;
};

Result<ExportArgs> ParseExportArgs(const std::vector<std::string>& args);

Result<TraceArgs> ParseTraceArgs(const std::vector<std::string>& args);

struct Options
{
	Config cfg;
	std::optional<bool> worker_numa;
	std::optional<std::string> worker_cpus;
	[[nodiscard]] Result<void> ApplyWorkerPlacement() const;
	bool header = false;
	bool help = false;
	bool info = false;
	bool only_header = false;

	bool HeaderOnly() const
	{
		return only_header;
	}
};

Result<Options> ParseArgs(const std::vector<std::string>& args);

std::string UsageText();

bool InternalCommandsEnabled();

std::string TopLevelUsage();

std::string CheckUsage();

Result<TuneArgs> ParseTuneArgs(const std::vector<std::string>& args);

} // namespace cmt
