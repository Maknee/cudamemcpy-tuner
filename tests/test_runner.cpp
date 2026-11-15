// test_runner.cpp - child processes and the measurement lock (src/runner.cpp): what a `cmt run` application,
// a measurement child and a lock holder get. No GPU, no hook library.
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "tests/check.h"
#include "common/helpers.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "runner.h"

extern char** environ;

using namespace cmt;

namespace {

// A fresh directory under the test's working directory (ctest runs it in the build tree), removed on exit.
struct ScratchDirectory
{
	std::string path;

	ScratchDirectory()
	{
		char name[] = "runner-test-XXXXXX";
		path = std::filesystem::absolute(mkdtemp(name) != nullptr ? name : "").string();
	}

	~ScratchDirectory()
	{
		std::error_code error;
		std::filesystem::remove_all(path, error);
	}
};

std::string RunAndRead(ProcessCommand command, const std::string& output)
{
	command.stdout_path = output;
	const auto result = check::Require(RunProcess(command));
	CHECK(result.completed && result.exit_code == 0);
	return check::Require(ReadTextFile(output));
}

TEST(cmt_run_keeps_the_users_other_preloads_after_the_hook)
{
	CHECK_EQ(ComposePreload("/opt/cmt/libcmt_hook.so", nullptr), std::string("/opt/cmt/libcmt_hook.so"));
	CHECK_EQ(ComposePreload("/opt/cmt/libcmt_hook.so", ""), std::string("/opt/cmt/libcmt_hook.so"));
	CHECK_EQ(ComposePreload("/h/libcmt_hook.so", "/usr/lib/libjemalloc.so.2"),
			 std::string("/h/libcmt_hook.so /usr/lib/libjemalloc.so.2"));
	// Spaces and colons both separate (ld.so); empty fields vanish; order is kept.
	CHECK_EQ(ComposePreload("/h/libcmt_hook.so", "a.so:b.so  c.so:"), std::string("/h/libcmt_hook.so a.so b.so c.so"));
	// Earlier cmt libraries go (another hook, the trace library), whatever their directory; others stay.
	CHECK_EQ(ComposePreload("/h/libcmt_hook.so", "/old/libcmt_hook.so:/x/libcmt_trace.so libtcmalloc.so"),
			 std::string("/h/libcmt_hook.so libtcmalloc.so"));
	CHECK_EQ(ComposePreload("/h/libcmt_hook.so", "/x/mylibcmt_other.so"),
			 std::string("/h/libcmt_hook.so /x/mylibcmt_other.so"));
}

TEST(a_child_reads_dev_null_not_the_terminal)
{
	ScratchDirectory scratch;
	ProcessCommand command;
	command.executable = "/bin/sh";
	command.arguments = {"-c", "readlink /proc/self/fd/0"};
	CHECK_EQ(RunAndRead(command, scratch.path + "/out"), std::string("/dev/null\n"));
}

TEST(an_environment_entry_without_equals_is_not_passed_on)
{
	ScratchDirectory scratch;
	// A raw environ entry without '=' (execve accepts one). The child must not get it, neither as itself nor as
	// "NOEQUALS_ENTRY=NOEQUALS_ENTRY".
	std::vector<char*> entries;
	for (char** entry = environ; *entry != nullptr; ++entry)
	{
		entries.push_back(*entry);
	}
	char raw[] = "NOEQUALS_ENTRY";
	char kept[] = "RUNNER_TEST_KEPT=1";
	entries.push_back(raw);
	entries.push_back(kept);
	entries.push_back(nullptr);
	char** saved = environ;
	environ = std::data(entries);
	ProcessCommand command;
	command.executable = "/usr/bin/env";
	const std::string out = RunAndRead(command, scratch.path + "/out");
	environ = saved;
	CHECK(out.find("NOEQUALS_ENTRY") == std::string::npos);
	CHECK(out.find("RUNNER_TEST_KEPT=1\n") != std::string::npos);
}

bool InheritedLockSet()
{
	if (std::getenv("CMT_MEASUREMENT_LOCK_FD") != nullptr ||
		std::getenv("CMT_MEASUREMENT_LOCK_PATH") != nullptr)
	{
		std::puts("skip: an inherited measurement lock is set");
		return true;
	}
	return false;
}

// Scratch stand-ins for /run/user and /tmp: production never changes them, tests cannot.
MeasurementLockPlaces ScratchPlaces(const ScratchDirectory& scratch, uid_t uid = geteuid())
{
	MeasurementLockPlaces places;
	places.runtime_base = scratch.path + "/run-user";
	places.fallback_base = scratch.path + "/tmp";
	places.uid = uid;
	std::filesystem::create_directories(places.runtime_base);
	std::filesystem::create_directories(places.fallback_base);
	return places;
}

unsigned ModeOf(const std::string& path)
{
	struct stat info{};
	return lstat(path.c_str(), &info) == 0 ? static_cast<unsigned>(info.st_mode & 07777) : 0xffffu;
}

TEST(the_production_lock_places_are_fixed)
{
	const MeasurementLockPlaces places;
	CHECK_EQ(places.runtime_base, std::string("/run/user"));
	CHECK_EQ(places.fallback_base, std::string("/tmp"));
	CHECK_EQ(places.uid, MeasurementUid());
	const std::string runtime = "/run/user/" + std::to_string(MeasurementUid());
	const std::string expected =
		access(runtime.c_str(), F_OK) == 0
			? runtime + "/cmt-measurements.lock"
			: "/tmp/cmt-" + std::to_string(MeasurementUid()) + "/cmt-measurements.lock";
	CHECK_EQ(DefaultMeasurementLockPath(), expected);
}

TEST(the_default_lock_is_in_the_users_runtime_directory_when_it_exists)
{
	if (InheritedLockSet())
	{
		return;
	}
	ScratchDirectory scratch;
	const auto places = ScratchPlaces(scratch);
	const std::string runtime = places.runtime_base + "/" + std::to_string(places.uid);
	CHECK(mkdir(runtime.c_str(), 0700) == 0);
	CHECK_EQ(DefaultMeasurementLockPath(places), runtime + "/cmt-measurements.lock");
	{
		auto lock = MeasurementLock::Acquire({}, places);
		CHECK(lock.has_value());
		CHECK_EQ(ModeOf(runtime + "/cmt-measurements.lock"), 0600u);
	}
	// No fallback directory is made while the runtime directory exists.
	CHECK(access((places.fallback_base + "/cmt-" + std::to_string(places.uid)).c_str(), F_OK) != 0);
	// A runtime directory that is a symbolic link is refused, not followed.
	ScratchDirectory other;
	auto linked = ScratchPlaces(other);
	const std::string real = other.path + "/real";
	CHECK(mkdir(real.c_str(), 0700) == 0);
	CHECK(symlink(real.c_str(), (linked.runtime_base + "/" + std::to_string(linked.uid)).c_str()) == 0);
	CHECK(!MeasurementLock::Acquire({}, linked).has_value());
	CHECK(access((real + "/cmt-measurements.lock").c_str(), F_OK) != 0);
}

TEST(without_a_runtime_directory_the_lock_is_in_a_private_tmp_directory)
{
	if (InheritedLockSet())
	{
		return;
	}
	ScratchDirectory scratch;
	const auto places = ScratchPlaces(scratch);
	const std::string directory = places.fallback_base + "/cmt-" + std::to_string(places.uid);
	CHECK_EQ(DefaultMeasurementLockPath(places), directory + "/cmt-measurements.lock");
	const mode_t mask = umask(0077); // mkdir's mode is masked; the directory must still end up 0700
	{
		auto lock = MeasurementLock::Acquire({}, places);
		CHECK(lock.has_value());
		CHECK(!MeasurementLock::Acquire({}, places).has_value()); // held
	}
	umask(mask);
	CHECK_EQ(ModeOf(directory), 0700u);
	CHECK_EQ(ModeOf(directory + "/cmt-measurements.lock"), 0600u);
	CHECK(MeasurementLock::Acquire({}, places).has_value()); // the existing private directory is reused
	// Named explicitly, the same directory gets the same checks.
	CHECK(MeasurementLock::Acquire(directory + "/cmt-measurements.lock", places).has_value());
}

TEST(a_fallback_directory_owned_by_another_user_is_refused)
{
	if (InheritedLockSet())
	{
		return;
	}
	ScratchDirectory scratch;
	// The test cannot make another user's directory; it asks for another user's lock instead.
	const auto places = ScratchPlaces(scratch, geteuid() + 1);
	const std::string directory = places.fallback_base + "/cmt-" + std::to_string(places.uid);
	CHECK(mkdir(directory.c_str(), 0700) == 0);
	CHECK(!MeasurementLock::Acquire({}, places).has_value());
	CHECK(access((directory + "/cmt-measurements.lock").c_str(), F_OK) != 0);
}

TEST(a_fallback_directory_others_can_read_is_refused)
{
	if (InheritedLockSet())
	{
		return;
	}
	ScratchDirectory scratch;
	const auto places = ScratchPlaces(scratch);
	const std::string directory = places.fallback_base + "/cmt-" + std::to_string(places.uid);
	CHECK(mkdir(directory.c_str(), 0700) == 0 && chmod(directory.c_str(), 0755) == 0);
	CHECK(!MeasurementLock::Acquire({}, places).has_value());
	CHECK(access((directory + "/cmt-measurements.lock").c_str(), F_OK) != 0);
	CHECK_EQ(ModeOf(directory), 0755u); // reported, never repaired
}

TEST(a_symlinked_fallback_directory_is_refused)
{
	if (InheritedLockSet())
	{
		return;
	}
	ScratchDirectory scratch;
	const auto places = ScratchPlaces(scratch);
	const std::string real = scratch.path + "/real";
	CHECK(mkdir(real.c_str(), 0700) == 0);
	CHECK(symlink(real.c_str(), (places.fallback_base + "/cmt-" + std::to_string(places.uid)).c_str()) == 0);
	CHECK(!MeasurementLock::Acquire({}, places).has_value());
	CHECK(access((real + "/cmt-measurements.lock").c_str(), F_OK) != 0);
}

TEST(a_symlinked_lock_in_the_fallback_directory_is_refused)
{
	if (InheritedLockSet())
	{
		return;
	}
	ScratchDirectory scratch;
	const auto places = ScratchPlaces(scratch);
	const std::string directory = places.fallback_base + "/cmt-" + std::to_string(places.uid);
	CHECK(mkdir(directory.c_str(), 0700) == 0);
	const std::string target = scratch.path + "/target";
	CHECK(symlink(target.c_str(), (directory + "/cmt-measurements.lock").c_str()) == 0);
	CHECK(!MeasurementLock::Acquire({}, places).has_value());
	CHECK(access(target.c_str(), F_OK) != 0);
}

TEST(an_explicit_lock_is_exclusive_and_never_follows_a_link)
{
	if (InheritedLockSet())
	{
		return;
	}
	ScratchDirectory scratch;
	const std::string path = scratch.path + "/sub/cmt-measurements.lock";
	{
		auto first = MeasurementLock::Acquire(path);
		CHECK(first.has_value());
		struct stat file{};
		CHECK(stat(path.c_str(), &file) == 0 && S_ISREG(file.st_mode) && (file.st_mode & 0777) == 0600);
		CHECK(!MeasurementLock::Acquire(path).has_value()); // held
	}
	CHECK(MeasurementLock::Acquire(path).has_value()); // released with its owner
	// A symbolic link where the lock should be: refused, and its target is neither created nor locked.
	const std::string target = scratch.path + "/target";
	const std::string link = scratch.path + "/link.lock";
	CHECK(symlink(target.c_str(), link.c_str()) == 0);
	CHECK(!MeasurementLock::Acquire(link).has_value());
	CHECK(access(target.c_str(), F_OK) != 0);
	// A directory is not a lock.
	CHECK(!MeasurementLock::Acquire(scratch.path).has_value());
}

} // namespace

TEST_MAIN()
