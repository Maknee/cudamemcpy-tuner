#include "runner.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <utility>
#include <dlfcn.h>
#include <sys/file.h>
#include <sys/stat.h>

extern char** environ;

namespace cmt {

namespace {

constexpr char kMeasurementLockName[] = "cmt-measurements.lock";

std::string FallbackLockDirectory(const MeasurementLockPlaces& places)
{
	return places.fallback_base + "/cmt-" + std::to_string(places.uid);
}

Result<int> OpenPrivateLockDirectory(const MeasurementLockPlaces& places)
{
	const std::string path = FallbackLockDirectory(places);
	const std::string name = "cmt-" + std::to_string(places.uid);
	const int base = open(places.fallback_base.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	if (base < 0)
	{
		return OsError("open the measurement lock's fallback base directory");
	}
	const bool created = mkdirat(base, name.c_str(), 0700) == 0;
	if (!created && errno != EEXIST)
	{
		const int code = errno;
		close(base);
		return OsError("create the private measurement lock directory", code);
	}
	const int directory = openat(base, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
	const int code = errno;
	close(base);
	if (directory < 0)
	{
		std::fprintf(stderr,
					 "cmt: the measurement lock directory %s is not a directory or is a symbolic link (%s); "
					 "inspect and remove it, or set CMT_MEASUREMENT_LOCK_PATH\n",
					 path.c_str(), std::strerror(code));
		return OsError("open the private measurement lock directory without following a link", code);
	}
	if (created)
	{
		if (fchmod(directory, 0700) != 0 ||
			(geteuid() == 0 && places.uid != 0 && fchown(directory, places.uid, static_cast<gid_t>(-1)) != 0))
		{
			const int failure = errno;
			close(directory);
			return OsError("prepare the private measurement lock directory", failure);
		}
	}
	struct stat info{};
	if (fstat(directory, &info) != 0)
	{
		const int failure = errno;
		close(directory);
		return OsError("inspect the private measurement lock directory", failure);
	}
	if (!S_ISDIR(info.st_mode) || info.st_uid != places.uid || (info.st_mode & 07777) != 0700)
	{
		close(directory);
		std::fprintf(
			stderr,
			"cmt: the measurement lock directory %s must be a directory owned by uid %u with mode 0700 "
			"(it is owned by uid %u, mode %04o); inspect and remove it, or set CMT_MEASUREMENT_LOCK_PATH\n",
			path.c_str(), static_cast<unsigned>(places.uid), static_cast<unsigned>(info.st_uid),
			static_cast<unsigned>(info.st_mode & 07777));
		return UsageError("the private measurement lock directory is not private");
	}
	return directory;
}

} // namespace

MeasurementLock::MeasurementLock(int descriptor, std::string filename)
	: fd(descriptor), path(std::move(filename))
{
}

MeasurementLock::MeasurementLock(MeasurementLock&& other) noexcept
	: fd(std::exchange(other.fd, -1)), path(std::move(other.path))
{
}

MeasurementLock::~MeasurementLock()
{
	if (fd >= 0)
	{
		close(fd);
	}
}

Result<MeasurementLock> MeasurementLock::Acquire(const std::string& requested,
												 const MeasurementLockPlaces& places)
{
	const char* inherited_fd = std::getenv("CMT_MEASUREMENT_LOCK_FD");
	const char* inherited_path = std::getenv("CMT_MEASUREMENT_LOCK_PATH");
	if ((inherited_fd == nullptr) != (inherited_path == nullptr))
	{
		return UsageError("inherited measurement lock requires both descriptor and path");
	}
	std::string filename = requested;
	const bool default_path = filename.empty() && inherited_path == nullptr;
	if (filename.empty())
	{
		filename = inherited_path ? inherited_path : DefaultMeasurementLockPath(places);
	}
	if (filename.empty())
	{
		return UsageError("measurement lock path is empty");
	}
	std::error_code error;
	const auto absolute = std::filesystem::absolute(filename, error);
	if (error)
	{
		return IoFailure("resolve measurement lock", error);
	}
	if (inherited_fd)
	{
		int descriptor = -1;
		const auto end = inherited_fd + std::strlen(inherited_fd);
		const auto parsed = std::from_chars(inherited_fd, end, descriptor);
		if (parsed.ec != std::errc{} || parsed.ptr != end || descriptor <= STDERR_FILENO)
		{
			return UsageError("invalid inherited measurement lock descriptor");
		}
		struct stat actual{};
		struct stat expected{};
		struct stat declared{};
		if (fstat(descriptor, &actual) != 0 || stat(absolute.c_str(), &expected) != 0 ||
			stat(inherited_path, &declared) != 0)
		{
			return OsError("inspect inherited measurement lock");
		}
		if (!S_ISREG(actual.st_mode) || actual.st_dev != expected.st_dev ||
			actual.st_ino != expected.st_ino || actual.st_dev != declared.st_dev ||
			actual.st_ino != declared.st_ino)
		{
			return UsageError("inherited measurement lock path and descriptor differ");
		}
		const int flags = fcntl(descriptor, F_GETFD);
		if (flags < 0)
		{
			return OsError("inspect measurement lock descriptor flags");
		}
		if (flags & FD_CLOEXEC)
		{
			return UsageError("inherited measurement lock must survive exec");
		}
		constexpr std::size_t kMaxLockInfoBytes = 64 * 1024;
		auto info = ReadTextFile("/proc/self/fdinfo/" + std::to_string(descriptor), kMaxLockInfoBytes);
		if (!info)
		{
			return Failure(info.error());
		}
		bool exclusive = false;
		std::istringstream lines(*info);
		std::string line;
		while (std::getline(lines, line))
		{
			std::istringstream words(line);
			std::string key;
			std::string id;
			std::string kind;
			std::string advisory;
			std::string access;
			if (words >> key >> id >> kind >> advisory >> access && key == "lock:" && kind == "FLOCK" &&
				advisory == "ADVISORY" && access == "WRITE")
			{
				exclusive = true;
			}
		}
		if (!exclusive)
		{
			return UsageError("inherited descriptor has no exclusive measurement flock proof");
		}
		const int duplicate = fcntl(descriptor, F_DUPFD, STDERR_FILENO + 1);
		if (duplicate < 0)
		{
			return OsError("retain inherited measurement lock");
		}
		return MeasurementLock(duplicate, absolute.string());
	}
	const auto parent = absolute.parent_path();
	int directory = -1;
	if (parent.lexically_normal() == std::filesystem::path(FallbackLockDirectory(places)).lexically_normal())
	{
		auto opened = OpenPrivateLockDirectory(places);
		if (!opened)
		{
			return Failure(opened.error());
		}
		directory = *opened;
	}
	else if (default_path)
	{
		directory = open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
		if (directory < 0)
		{
			const int code = errno;
			std::fprintf(
				stderr, "cmt: cannot open the measurement lock's directory %s without following a link: %s\n",
				parent.c_str(), std::strerror(code));
			return OsError("open the measurement lock's runtime directory", code);
		}
	}
	else
	{
		CMT_RETURN_IF_ERROR(MakeDirectories(parent.string()));
		directory = open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
		if (directory < 0)
		{
			return OsError("open the measurement lock's directory");
		}
	}
	const auto name = absolute.filename();
	const uid_t effective = geteuid();
	bool created = false;
	int descriptor = openat(directory, name.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
	if (descriptor < 0 && errno == ENOENT)
	{
		descriptor =
			openat(directory, name.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
		created = descriptor >= 0;
		if (descriptor < 0 && errno == EEXIST)
		{
			descriptor = openat(directory, name.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
		}
	}
	const int open_error = errno;
	close(directory);
	if (descriptor < 0)
	{
		if (open_error == ELOOP)
		{
			std::fprintf(stderr, "cmt: the measurement lock %s is a symbolic link; refusing to follow it\n",
						 absolute.c_str());
		}
		return OsError("open measurement lock", open_error);
	}
	if (descriptor <= STDERR_FILENO)
	{
		const int duplicate = fcntl(descriptor, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
		const int code = errno;
		close(descriptor);
		if (duplicate < 0)
		{
			return OsError("reserve measurement lock descriptor above stdio", code);
		}
		descriptor = duplicate;
	}
	MeasurementLock owner(descriptor, absolute.string());
	if (created && effective == 0 && places.uid != 0 &&
		fchown(descriptor, places.uid, static_cast<gid_t>(-1)) != 0)
	{
		return OsError("give the measurement lock to the invoking user");
	}
	struct stat file{};
	if (fstat(descriptor, &file) != 0)
	{
		return OsError("inspect measurement lock");
	}
	if (!S_ISREG(file.st_mode))
	{
		return UsageError("measurement lock must be a regular file");
	}
	if (file.st_uid != places.uid && file.st_uid != 0)
	{
		std::fprintf(stderr, "cmt: the measurement lock %s belongs to uid %u, not to uid %u or root\n",
					 absolute.c_str(), static_cast<unsigned>(file.st_uid), static_cast<unsigned>(places.uid));
		return UsageError("measurement lock is owned by another user");
	}
	if (flock(descriptor, LOCK_EX | LOCK_NB) != 0)
	{
		return OsError("another process owns the measurement lock");
	}
	if (fcntl(descriptor, F_SETFD, 0) != 0)
	{
		return OsError("allow children to inherit measurement ownership");
	}
	return owner;
}

uid_t MeasurementUid()
{
	const uid_t effective = geteuid();
	const char* sudo_uid = effective == 0 ? secure_getenv("SUDO_UID") : nullptr;
	if (sudo_uid == nullptr || *sudo_uid == '\0')
	{
		return effective;
	}
	std::uint32_t uid = 0;
	const auto end = sudo_uid + std::strlen(sudo_uid);
	const auto parsed = std::from_chars(sudo_uid, end, uid);
	if (parsed.ec != std::errc{} || parsed.ptr != end || uid == static_cast<std::uint32_t>(-1))
	{
		return effective;
	}
	return static_cast<uid_t>(uid);
}

std::string DefaultMeasurementLockPath(const MeasurementLockPlaces& places)
{
	const std::string runtime = places.runtime_base + "/" + std::to_string(places.uid);
	struct stat directory{};
	if (lstat(runtime.c_str(), &directory) != 0 && errno == ENOENT)
	{
		return FallbackLockDirectory(places) + "/" + kMeasurementLockName;
	}
	return runtime + "/" + kMeasurementLockName;
}

std::string ComposePreload(const std::string& hook, const char* inherited)
{
	std::string preload = hook;
	std::string_view rest = inherited != nullptr ? inherited : "";
	while (!rest.empty())
	{
		const auto end = rest.find_first_of(" :");
		const std::string_view entry = rest.substr(0, end);
		rest = end == std::string_view::npos ? std::string_view{} : rest.substr(end + 1);
		const auto slash = entry.rfind('/');
		const std::string_view base = slash == std::string_view::npos ? entry : entry.substr(slash + 1);
		if (entry.empty() || base.rfind("libcmt_", 0) == 0)
		{
			continue;
		}
		preload += ' ';
		preload += entry;
	}
	return preload;
}

void MeasurementLock::Apply(ProcessCommand& command) const
{
	command.environment.push_back("CMT_MEASUREMENT_LOCK_FD=" + std::to_string(fd));
	command.environment.push_back("CMT_MEASUREMENT_LOCK_PATH=" + path);
}

Result<std::string> ResolvePreloadLibrary(const std::string& requested, const char* default_library)
{
	std::filesystem::path library(requested);
	if (requested.empty())
	{
		const auto executable = SelfExe();
		if (!executable)
		{
			return Failure(executable.error());
		}
		library = std::filesystem::path(*executable).parent_path() / default_library;
	}
	std::error_code error;
	auto absolute = std::filesystem::absolute(library, error).string();
	if (error)
	{
		return IoFailure("resolve hook library", error);
	}
	auto regular = IsRegularFile(absolute);
	if (!regular)
	{
		return Failure(regular.error());
	}
	if (absolute.find_first_of(" :\t\n") != std::string::npos)
	{
		return UsageError(
			"preload library path contains separators; use a directory without spaces or colons");
	}
	if (!*regular)
	{
		return UsageError("preload library does not exist or is not a regular file");
	}
	return absolute;
}

namespace {

volatile sig_atomic_t trace_child = 0;

void TraceSignal(int signal)
{
	const int saved = errno;
	if (trace_child > 0)
	{
		kill(trace_child, signal);
	}
	errno = saved;
}

constexpr auto kChildPollInterval = std::chrono::milliseconds(20);
volatile sig_atomic_t child_interruption = 0;

void RecordChildInterruption(int signal)
{
	child_interruption = signal;
}

Result<void> SignalOwnedProcessGroup(pid_t child, int signal)
{
	if (kill(-child, signal) != 0 && errno != ESRCH)
	{
		return OsError("signal benchmark process group");
	}
	return {};
}

Result<std::vector<std::string>> ChildEnvironment(const ProcessCommand& command)
{
	std::map<std::string, std::string> values;
	for (char** entry = command.inherit_environment ? environ : nullptr; entry && *entry; ++entry)
	{
		std::string value(*entry);
		const auto at = value.find('=');
		if (at == std::string::npos)
		{
			continue;
		}
		const auto name = value.substr(0, at);
		if (command.clean_hook_environment && HookVariable(name))
		{
			continue;
		}
		values[name] = value.substr(at + 1);
	}
	for (const auto& value : command.environment)
	{
		const auto at = value.find('=');
		if (at == std::string::npos || at == 0 || value.find('\0') != std::string::npos)
		{
			return UsageError("process environment requires NAME=VALUE");
		}
		values[value.substr(0, at)] = value.substr(at + 1);
	}
	std::vector<std::string> result;
	for (const auto& [name, value] : values)
	{
		result.push_back(name + "=" + value);
	}
	return result;
}

std::vector<char*> Pointers(std::vector<std::string>& words)
{
	std::vector<char*> result;
	for (auto& word : words)
	{
		result.push_back(word.data());
	}
	result.push_back(nullptr);
	return result;
}
} // namespace

Result<ProcessResult> RunProcess(const ProcessCommand& command)
{
	if (command.executable.empty())
	{
		return UsageError("empty process executable");
	}
	if (command.cancellation_grace <= std::chrono::seconds::zero() ||
		command.cancellation_grace > std::chrono::minutes(1))
	{
		return UsageError("child cancellation grace must be positive and at most one minute");
	}
	auto environment = ChildEnvironment(command);
	if (!environment)
	{
		return Failure(environment.error());
	}
	auto env = Pointers(*environment);
	std::vector<std::string> words{command.executable};
	words.insert(words.end(), command.arguments.begin(), command.arguments.end());
	auto argv = Pointers(words);
	posix_spawn_file_actions_t actions;
	int code = posix_spawn_file_actions_init(&actions);
	if (code)
	{
		return OsError("initialize process file actions", code);
	}

	struct ActionsGuard
	{
		posix_spawn_file_actions_t* actions;

		~ActionsGuard()
		{
			posix_spawn_file_actions_destroy(actions);
		}
	} actions_guard{&actions};

	code = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
	if (code)
	{
		return OsError("configure process input", code);
	}
	for (const auto& [path, descriptor] :
		 {std::pair{command.stdout_path, STDOUT_FILENO}, std::pair{command.stderr_path, STDERR_FILENO}})
	{
		if (path.empty())
		{
			continue;
		}
		code = posix_spawn_file_actions_addopen(&actions, descriptor, path.c_str(),
												O_WRONLY | O_CREAT | O_TRUNC, 0644);
		if (code)
		{
			return OsError("configure process output", code);
		}
	}
	posix_spawnattr_t attributes;
	code = posix_spawnattr_init(&attributes);
	if (code)
	{
		return OsError("initialize process attributes", code);
	}

	struct AttributeGuard
	{
		posix_spawnattr_t* attributes;

		~AttributeGuard()
		{
			posix_spawnattr_destroy(attributes);
		}
	} attribute_guard{&attributes};

	code = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
	if (!code)
	{
		code = posix_spawnattr_setpgroup(&attributes, 0);
	}
	if (code)
	{
		return OsError("configure child process group", code);
	}
	child_interruption = 0;
	struct sigaction handler{};
	handler.sa_handler = RecordChildInterruption;
	sigemptyset(&handler.sa_mask);
	struct sigaction old_int{}, old_term{};
	if (sigaction(SIGINT, &handler, &old_int))
	{
		return OsError("install child cancellation handler");
	}
	if (sigaction(SIGTERM, &handler, &old_term))
	{
		const int saved = errno;
		sigaction(SIGINT, &old_int, nullptr);
		return OsError("install child cancellation handler", saved);
	}

	struct SignalGuard
	{
		struct sigaction old_int;
		struct sigaction old_term;
		struct sigaction old_child{};
		bool have_child = false;

		~SignalGuard()
		{
			if (have_child)
			{
				sigaction(SIGCHLD, &old_child, nullptr);
			}
			sigaction(SIGINT, &old_int, nullptr);
			sigaction(SIGTERM, &old_term, nullptr);
		}
	} signals{old_int, old_term};
	struct sigaction child_handler{};
	child_handler.sa_handler = SIG_DFL;
	sigemptyset(&child_handler.sa_mask);
	if (sigaction(SIGCHLD, &child_handler, &signals.old_child) != 0)
	{
		return OsError("preserve waitable benchmark child");
	}
	signals.have_child = true;

	std::vector<unsigned> own_cpus;
	if (!std::empty(command.cpus))
	{
		auto saved = GetThreadAffinity();
		if (!saved)
		{
			return Failure(saved.error());
		}
		own_cpus = std::move(*saved);
		CMT_RETURN_IF_ERROR(SetThreadAffinity(pthread_self(), command.cpus));
	}
	pid_t child = 0;
	code = posix_spawnp(&child, command.executable.c_str(), &actions, &attributes, argv.data(), env.data());
	const auto restored =
		std::empty(own_cpus) ? Result<void>{} : RestoreThreadAffinity(pthread_self(), own_cpus);
	if (code)
	{
		CMT_RETURN_IF_ERROR(restored);
		return OsError("spawn benchmark process", code);
	}
	std::optional<Clock::time_point> interrupted_at;
	int forwarded_signal = 0;
	bool forced_stop = false;
	Result<void> observed;
	for (;;)
	{
		const int interruption = child_interruption;
		if (interruption != 0 && forwarded_signal != interruption)
		{
			observed = SignalOwnedProcessGroup(child, interruption);
			if (!observed)
			{
				break;
			}
			forwarded_signal = interruption;
			if (!interrupted_at)
			{
				interrupted_at = Clock::now();
			}
		}
		if (interrupted_at && !forced_stop && Clock::now() - *interrupted_at >= command.cancellation_grace)
		{
			observed = SignalOwnedProcessGroup(child, SIGKILL);
			if (!observed)
			{
				break;
			}
			forced_stop = true;
		}

		// Keep the leader waitable to prevent PID reuse before process-group cleanup.
		siginfo_t information{};
		if (waitid(P_PID, child, &information, WEXITED | WNOHANG | WNOWAIT) != 0)
		{
			if (errno == EINTR)
			{
				continue;
			}
			observed = OsError("observe benchmark process completion");
			break;
		}
		if (information.si_pid == child)
		{
			break;
		}
		const timespec pause{
			0, std::chrono::duration_cast<std::chrono::nanoseconds>(kChildPollInterval).count()};
		nanosleep(&pause, nullptr);
	}
	if (observed || observed.error().GetCode() != ECHILD)
	{
		auto stopped = SignalOwnedProcessGroup(child, SIGKILL);
		if (!stopped && observed)
		{
			observed = Failure(stopped.error());
		}
	}
	int status = 0;
	while (waitpid(child, &status, 0) < 0)
	{
		if (errno == EINTR)
		{
			continue;
		}
		return OsError("reap benchmark process");
	}
	CMT_RETURN_IF_ERROR(observed);
	CMT_RETURN_IF_ERROR(restored);
	ProcessResult result;
	result.completed = WIFEXITED(status);
	result.signal = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
	result.exit_code = result.completed ? WEXITSTATUS(status) : kSignalExitBase + result.signal;
	result.interrupted_by_signal = child_interruption;
	return result;
}

Result<ThisMachine> DetectThisMachine()
{
	ThisMachine machine;
#if defined(__x86_64__)
	machine.arch = "x86_64";
#elif defined(__aarch64__)
	machine.arch = "aarch64";
#else
	machine.arch = "unknown";
#endif
	auto model = CpuModelName();
	if (!model)
	{
		return Failure(model.error());
	}
	machine.cpu = std::move(*model);
	void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!driver)
	{
		std::fprintf(stderr, "cmt: libcuda.so.1 does not load (%s)\n", dlerror());
		return Unsupported("no CUDA device visible");
	}
	const auto init = reinterpret_cast<int (*)(unsigned)>(dlsym(driver, "cuInit"));
	const auto count = reinterpret_cast<int (*)(int*)>(dlsym(driver, "cuDeviceGetCount"));
	const auto get = reinterpret_cast<int (*)(int*, int)>(dlsym(driver, "cuDeviceGet"));
	const auto name = reinterpret_cast<int (*)(char*, int, int)>(dlsym(driver, "cuDeviceGetName"));
	if (!init || !count || !get || !name)
	{
		return Failure(Error::Kind::Cuda, "libcuda.so.1 lacks cuInit, cuDeviceGetCount, cuDeviceGet or cuDeviceGetName");
	}
	constexpr int kNoDevice = 100;
	const int status = init(0);
	if (status == kNoDevice)
	{
		return Unsupported("no CUDA device visible");
	}
	if (status != 0)
	{
		return Failure(Error::Kind::Cuda, "cuInit failed", status);
	}
	int devices = 0;
	if (const int failed = count(&devices))
	{
		return Failure(Error::Kind::Cuda, "cuDeviceGetCount failed", failed);
	}
	if (devices <= 0)
	{
		return Unsupported("no CUDA device visible");
	}
	for (int ordinal = 0; ordinal < devices; ordinal++)
	{
		int device = -1;
		char text[256] = {};
		const int failed = get(&device, ordinal);
		if (failed || name(text, sizeof(text) - 1, device))
		{
			return Failure(Error::Kind::Cuda, "cuDeviceGet or cuDeviceGetName failed", failed);
		}
		machine.gpus.emplace_back(text);
	}
	return machine;
}

namespace {

template <class Samples>
std::string SpeedThan(const Samples& driver, const Samples& faster)
{
	const double denominator = Median(faster);
	if (!(denominator > 0.0))
	{
		return "faster than";
	}
	char text[64];
	std::snprintf(text, sizeof(text), "%.2fx as fast as", Median(driver) / denominator);
	return text;
}

Result<bool> TunedHostAllocation(const std::string& path, bool verbose)
{
	auto text = ReadTextFile(path, tuning::kMaxConfigBytes);
	if (!text)
	{
		return Failure(text.error());
	}
	JsonParser parser;
	Json document;
	if (parser.parse(*text).get(document))
	{
		return UsageError("tune.json is not JSON");
	}
	auto section = tuning::ReadHostAllocation(document);
	if (!section)
	{
		return Failure(section.error());
	}
	if (!*section || (*section)->state != "measured")
	{
		return false;
	}
	const auto& measured = **section;
	const bool shared = tuning::HostAllocClearlyFaster(measured.shared.alloc_seconds, measured.driver.alloc_seconds);
	if (verbose && shared)
	{
		const std::string speed = SpeedThan(measured.driver.alloc_seconds, measured.shared.alloc_seconds);
		std::fprintf(stderr,
					 "cmt run: pinned allocations use the hook (the tune timed it %s the driver); "
					 "--cuda-host-alloc-hook off keeps the driver's\n",
					 speed.c_str());
	}
	if (verbose && measured.private_state == "measured" &&
		tuning::HostAllocClearlyFaster(measured.private_memory.alloc_seconds, measured.driver.alloc_seconds))
	{
		const std::string speed = SpeedThan(measured.driver.alloc_seconds, measured.private_memory.alloc_seconds);
		std::fprintf(stderr,
					 "cmt run: the tune timed private huge-page memory %s the driver; it is not "
					 "used unless you ask: --cuda-host-alloc-hook private (a fork() child then gets a copy of the "
					 "pinned memory, not a share)\n",
					 speed.c_str());
	}
	return shared;
}

} // namespace

Result<int> RunConfiguredApplication(const RunArgs& options, std::optional<ThisMachine> here)
{
	std::string assignments;
	for (const auto& entry : options.overrides)
	{
		if (entry.empty())
		{
			return UsageError("--override requires nonempty KEY=VALUE assignments");
		}
		if (!assignments.empty())
		{
			assignments += ',';
		}
		assignments += entry;
	}
	RunOverrides overrides;
	if (!assignments.empty())
	{
		auto parsed = ParseRunOverrides(assignments);
		if (!parsed)
		{
			return Failure(parsed.error());
		}
		overrides = std::move(*parsed);
	}
	auto table = tuning::Load(options.config.c_str(), overrides);
	if (!table)
	{
		std::fprintf(stderr, "cmt run: cannot use --config %s\n", options.config.c_str());
		return Failure(table.error());
	}
	if (!here)
	{
		auto detected = DetectThisMachine();
		if (!detected)
		{
			return Failure(detected.error());
		}
		here = std::move(*detected);
	}
	const Table& source = **table;
	const bool measured_here =
		here->arch == source.source_cpu_arch && here->cpu == source.source_cpu_model &&
		std::find(std::begin(here->gpus), std::end(here->gpus), source.source_gpu_name) != std::end(here->gpus);
	if (!measured_here && !options.remote)
	{
		std::fprintf(stderr,
					 "cmt run: %s was measured on %s / %s (%s), not on this %s / %s (%s), so the hook would ignore "
					 "it.\nMeasure this machine: cmt tune --search quick\n",
					 options.config.c_str(), source.source_gpu_name.c_str(), source.source_cpu_model.c_str(),
					 source.source_cpu_arch.c_str(), here->gpus.front().c_str(), here->cpu.c_str(),
					 here->arch.c_str());
		return Unsupported("the tune is for another machine");
	}
	std::error_code path_error;
	const auto config = std::filesystem::absolute(options.config, path_error).string();
	if (path_error)
	{
		return IoFailure("resolve tuning configuration", path_error);
	}
	if (options.command.empty())
	{
		std::fprintf(stderr, "cmt tune: Loaded tuned file: %s\n", config.c_str());
		return 0;
	}
	auto hook = ResolvePreloadLibrary(options.hook);
	if (!hook)
	{
		return Failure(hook.error());
	}
	ProcessCommand command;
	command.executable = options.command.front();
	command.arguments.assign(options.command.begin() + 1, options.command.end());
	command.environment = {"LD_PRELOAD=" + ComposePreload(*hook, std::getenv("LD_PRELOAD")), "CMT_CONFIG=" + config};
	if (options.debug)
	{
		command.environment.push_back("MEMCPY_HOOK_DEBUG=1");
	}
	if (options.remote)
	{
		command.environment.push_back("CMT_REMOTE_CONFIG=1");
	}
	std::string host_alloc = options.cuda_host_alloc_hook.value_or("off");
	if (!options.cuda_host_alloc_hook && !options.remote)
	{
		auto decided = TunedHostAllocation(config, options.verbose);
		if (!decided)
		{
			std::fprintf(stderr, "cmt run: cannot use --config %s\n", options.config.c_str());
			return Failure(decided.error());
		}
		host_alloc = *decided ? "on" : "off";
	}
	if (host_alloc != "off")
	{
		command.environment.push_back("CMT_CUDA_HOST_ALLOC_HOOK=1");
	}
	if (host_alloc == "private")
	{
		command.environment.push_back("CMT_CUDA_HOST_ALLOC_PRIVATE=1");
	}
	if (overrides.numa && !overrides.HasStaging() && !overrides.progressive_pinning)
	{
		command.environment.push_back("CMT_OVERRIDE=" + FormatRunOverrides(overrides));
	}
	else if (!overrides.Empty())
	{
		const auto normalized = FormatRunOverrides(overrides);
		std::fprintf(
			stderr,
			"cmt run: manual, unmeasured overrides: %s\n"
			"Staging settings affect staged rows only. Progressive pinning is a separate copy policy.\n"
			"Per-context staging pools: %zu / %zu bytes.\n",
			normalized.c_str(), (*table)->MaxPinnedBytes(), (*table)->max_pinned_bytes);
		command.environment.push_back("CMT_OVERRIDE=" + normalized);
	}
	std::fprintf(stderr, "cmt run: Loaded tuned file: %s\ncmt run: launching %s\n", config.c_str(),
				 command.executable.c_str());
	return RunApplication(command);
}

Result<int> RunApplication(const ProcessCommand& command)
{
	auto environment = ChildEnvironment(command);
	if (!environment)
	{
		return Failure(environment.error());
	}
	auto env = Pointers(*environment);
	std::vector<std::string> words{command.executable};
	words.insert(words.end(), command.arguments.begin(), command.arguments.end());
	auto argv = Pointers(words);
	execvpe(command.executable.c_str(), argv.data(), env.data());
	return OsError("launch application");
}

Result<int> RunTrace(const TraceArgs& options)
{
	auto resolved = ResolvePreloadLibrary(options.hook, "libcmt_trace.so");
	if (!resolved)
	{
		return Failure(resolved.error());
	}
	const auto& hook = *resolved;

	struct Files
	{
		int out = -1;
		int status[2] = {-1, -1};

		~Files()
		{
			for (int fd : {out, status[0], status[1]})
			{
				if (fd >= 0)
				{
					close(fd);
				}
			}
		}
	} files;

	files.out = open(options.out.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_APPEND, 0600);
	if (files.out < 0)
	{
		return IoFailure("trace: cannot create output file", errno);
	}
	if (socketpair(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK, 0, files.status) != 0)
	{
		return IoFailure("trace: status socket", errno);
	}
	for (int* fd : {&files.out, &files.status[0], &files.status[1]})
	{
		if (*fd <= STDERR_FILENO)
		{
			const int moved = fcntl(*fd, F_DUPFD, STDERR_FILENO + 1);
			if (moved < 0)
			{
				return IoFailure("trace: reserve descriptor", errno);
			}
			close(*fd);
			*fd = moved;
		}
	}

	std::vector<std::string> environment;
	std::string preload = hook;
	for (char** entry = environ; entry && *entry; ++entry)
	{
		const std::string value(*entry), name = value.substr(0, value.find('='));
		if (name == "LD_PRELOAD")
		{
			if (value.find("libcmt_hook") != std::string::npos)
			{
				return Failure(Error::Kind::Usage,
							   "trace: remove libcmt_hook from LD_PRELOAD; use only the trace library");
			}
			if (value.size() > name.size() + 1)
			{
				preload += " " + value.substr(name.size() + 1);
			}
		}
		else if (name.rfind("CUDAMEMCPY_TUNER_TRACE_", 0) != 0)
		{
			environment.push_back(value);
		}
	}
	environment.push_back("LD_PRELOAD=" + preload);
	environment.push_back("CUDAMEMCPY_TUNER_TRACE_ENABLE=1");
	environment.push_back("CUDAMEMCPY_TUNER_TRACE_FD=" + std::to_string(files.out));
	environment.push_back("CUDAMEMCPY_TUNER_TRACE_STATUS_FD=" + std::to_string(files.status[1]));
	environment.push_back("CUDAMEMCPY_TUNER_TRACE_PARENT_PID=" + std::to_string(getpid()));
	std::vector<char*> argv, envp;
	for (const auto& word : options.command)
	{
		argv.push_back(const_cast<char*>(word.c_str()));
	}
	argv.push_back(nullptr);
	for (auto& word : environment)
	{
		envp.push_back(word.data());
	}
	envp.push_back(nullptr);

	struct Signals
	{
		sigset_t previous{};
		struct sigaction old_int{};
		struct sigaction old_term{};
		bool masked = false;
		bool have_int = false;
		bool have_term = false;

		~Signals()
		{
			trace_child = 0;
			if (have_int)
			{
				sigaction(SIGINT, &old_int, nullptr);
			}
			if (have_term)
			{
				sigaction(SIGTERM, &old_term, nullptr);
			}
			if (masked)
			{
				sigprocmask(SIG_SETMASK, &previous, nullptr);
			}
		}
	} signals;

	sigset_t blocked;
	sigemptyset(&blocked);
	sigaddset(&blocked, SIGINT);
	sigaddset(&blocked, SIGTERM);
	if (sigprocmask(SIG_BLOCK, &blocked, &signals.previous) != 0)
	{
		return IoFailure("trace: block signals during spawn", errno);
	}
	signals.masked = true;
	struct sigaction handler{};
	handler.sa_handler = TraceSignal;
	sigemptyset(&handler.sa_mask);
	signals.have_int = sigaction(SIGINT, &handler, &signals.old_int) == 0;
	signals.have_term = sigaction(SIGTERM, &handler, &signals.old_term) == 0;
	if (!signals.have_int || !signals.have_term)
	{
		return IoFailure("trace: install signal forwarding", errno);
	}

	posix_spawn_file_actions_t actions;
	int rc = posix_spawn_file_actions_init(&actions);
	if (rc != 0)
	{
		return IoFailure("trace: prepare spawn", rc);
	}
	posix_spawnattr_t attributes;
	rc = posix_spawnattr_init(&attributes);
	if (rc != 0)
	{
		posix_spawn_file_actions_destroy(&actions);
		return IoFailure("trace: prepare child signals", rc);
	}
	rc = posix_spawn_file_actions_addclose(&actions, files.status[0]);
	if (rc == 0)
	{
		rc = posix_spawnattr_setsigmask(&attributes, &signals.previous);
	}
	if (rc == 0)
	{
		rc = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSIGMASK);
	}
	pid_t child = -1;
	if (rc == 0)
	{
		rc = posix_spawnp(&child, argv.front(), &actions, &attributes, argv.data(), envp.data());
	}
	posix_spawn_file_actions_destroy(&actions);
	posix_spawnattr_destroy(&attributes);
	if (rc != 0)
	{
		return IoFailure("trace: cannot start application", rc);
	}
	close(files.status[1]);
	files.status[1] = -1;
	trace_child = child;
	sigprocmask(SIG_SETMASK, &signals.previous, nullptr);
	signals.masked = false;
	int status = 0;
	pid_t waited;
	do
	{
		waited = waitpid(child, &status, 0);
	} while (waited < 0 && errno == EINTR);
	const int wait_error = errno;
	trace_child = 0;
	if (waited < 0)
	{
		return IoFailure("trace: wait for application", wait_error);
	}

	bool started = false, failed = false;
	char last = 0, events[128];
	ssize_t count;
	for (;;)
	{
		count = read(files.status[0], events, sizeof(events));
		if (count < 0 && errno == EINTR)
		{
			continue;
		}
		if (count <= 0)
		{
			if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
			{
				failed = true;
			}
			break;
		}
		for (ssize_t i = 0; i < count; ++i)
		{
			started |= events[i] == 'S';
			failed |= events[i] == 'F';
			last = events[i];
		}
	}
	const int child_status = WIFEXITED(status) ? WEXITSTATUS(status) : kSignalExitBase + WTERMSIG(status);
	const bool complete = started && !failed && last == 'E';
	std::fprintf(stderr, "cmt trace: %s; %s\n", options.out.c_str(),
				 complete ? "trace closed; zero records does not prove zero copies"
						  : "INCOMPLETE trace (initialization, write failure, or abnormal application exit)");
	if (child_status != 0)
	{
		return child_status;
	}
	if (!complete)
	{
		return Failure(Error::Kind::Verify, "CUDA trace did not complete");
	}
	return 0;
}
bool HookVariable(const std::string& name)
{
	if (name == "CMT_ENV_ROOT" || name == "CMT_WORKLOAD" || name == "CMT_BIN_DIR")
	{
		return false;
	}
	return name == "LD_PRELOAD" || name.rfind("CMT_", 0) == 0 || name.rfind("STAGING_", 0) == 0 ||
		   name.rfind("MEMCPY_HOOK_", 0) == 0 || name.rfind("CUDAMEMCPY_TUNER_TRACE_", 0) == 0;
}

} // namespace cmt
