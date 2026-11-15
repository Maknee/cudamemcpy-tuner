#pragma once
#include "common/defines.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <fcntl.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "common/error.h"
#include "common/helpers.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "config.h"

namespace cmt {

Result<std::string> ResolvePreloadLibrary(const std::string& requested,
										  const char* default_library = "libcmt_hook.so");

struct RunArgs
{
	std::string config;
	std::optional<std::string> remote;
	std::optional<std::string> cuda_host_alloc_hook;
	std::string hook;
	std::string help;
	bool debug = false;
	bool verbose = false;
	std::vector<std::string> overrides;
	std::vector<std::string> command;
};

struct ThisMachine
{
	std::string arch;
	std::string cpu;
	std::vector<std::string> gpus;
};
Result<ThisMachine> DetectThisMachine();

Result<int> RunConfiguredApplication(const RunArgs& options, std::optional<ThisMachine> here = std::nullopt);

struct TraceArgs
{
	std::string out;
	std::string hook;
	std::string help;
	std::vector<std::string> command;
};

Result<int> RunTrace(const TraceArgs& options);

bool HookVariable(const std::string& name);

struct ProcessCommand
{
	std::string executable;
	std::vector<std::string> arguments;
	std::vector<std::string> environment;
	std::string stdout_path;
	std::string stderr_path;
	bool clean_hook_environment = true;
	bool inherit_environment = true;
	std::chrono::seconds cancellation_grace{5};
	std::vector<unsigned> cpus;
};

std::string ComposePreload(const std::string& hook, const char* inherited);

uid_t MeasurementUid();

struct MeasurementLockPlaces
{
	std::string runtime_base = "/run/user";
	std::string fallback_base = "/tmp";
	uid_t uid = MeasurementUid();
};

std::string DefaultMeasurementLockPath(const MeasurementLockPlaces& places = {});

class MeasurementLock
{
public:
	static Result<MeasurementLock> Acquire(const std::string& path = {},
										   const MeasurementLockPlaces& places = {});

	MeasurementLock(const MeasurementLock&) = delete;
	MeasurementLock& operator=(const MeasurementLock&) = delete;
	MeasurementLock(MeasurementLock&& other) noexcept;
	MeasurementLock& operator=(MeasurementLock&&) = delete;
	~MeasurementLock();

	void Apply(ProcessCommand& command) const;

private:
	MeasurementLock(int descriptor, std::string filename);

	int fd = -1;
	std::string path;
};

struct ProcessResult
{
	int exit_code = 0;
	int signal = 0;
	int interrupted_by_signal = 0;
	bool completed = false;
};

Result<ProcessResult> RunProcess(const ProcessCommand& command);

Result<int> RunApplication(const ProcessCommand& command);

} // namespace cmt
