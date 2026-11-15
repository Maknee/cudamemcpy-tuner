#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <unistd.h>
#include "common/error.h"
#include "common/log.h"
#include "log.h"
#include "tests/check.h"

// The low-level logger's observable contract: bounded complete lines, exact
// formatting and truncation, without requiring the hook's configuration state.

namespace cmt {
namespace {
bool test_debug = false;
} // namespace

bool DebugLoggingEnabled()
{
	errno = EIO; // even configuration inspection must not leak errno through logging
	return test_debug;
}
} // namespace cmt

namespace {
template <class F>
std::string CaptureLine(F write)
{
	int fds[2];
	if (pipe(fds) != 0)
	{
		CHECK(false);
		return {};
	}
	// Each line is <=1024 bytes, below Linux's pipe capacity, so the writer cannot
	// block waiting for this reader. write_line is noexcept, including bad output.
	write(fds[1]);
	close(fds[1]);
	char text[cmt::logging::kLineBytes + 1];
	const ssize_t bytes = read(fds[0], text, sizeof text);
	close(fds[0]);
	if (bytes < 0)
	{
		CHECK(false);
		return {};
	}
	return std::string(text, static_cast<std::size_t>(bytes));
}
} // namespace

TEST(prefix_and_arguments_are_formatted_into_one_line)
{
	const auto line = CaptureLine(
		[](int fd) { cmt::logging::WriteLine(fd, "[copy] ", "bytes={} mode={}", 42u, "cached"); });
	CHECK_EQ(line, std::string("[copy] bytes=42 mode=cached\n"));
}

TEST(truncation_is_bounded_and_marked)
{
	const std::string message(4096, 'x');
	const auto line = CaptureLine([&](int fd) { cmt::logging::WriteLine(fd, "[copy] ", "{}", message); });
	CHECK_EQ(std::size(line), cmt::logging::kLineBytes);
	CHECK_EQ(line.substr(std::size(line) - 4), std::string("...\n"));
}

TEST(empty_and_oversized_prefixes_cannot_overrun_the_buffer)
{
	CHECK_EQ(CaptureLine([](int fd) { cmt::logging::WriteLine(fd, {}, "{}", 17); }), std::string("17\n"));
	const std::string prefix(4096, 'p');
	const auto line = CaptureLine([&](int fd) { cmt::logging::WriteLine(fd, prefix, "message"); });
	CHECK(std::size(line) <= cmt::logging::kLineBytes);
	CHECK_EQ(line.substr(std::size(line) - 4), std::string("...\n"));
}

TEST(logging_failure_preserves_application_errno)
{
	errno = ENOMEM;
	cmt::logging::WriteLine(-1, "[copy] ", "{}", 17);
	CHECK_EQ(errno, ENOMEM);
	errno = ERANGE;
	cmt::LogDebug("disabled");
	CHECK_EQ(errno, ERANGE);
}

TEST(debug_logging_is_runtime_gated_even_in_release)
{
	bool preserved = false;
	const auto line = CaptureLine(
		[&](int fd)
		{
			const int saved = dup(STDERR_FILENO);
			if (saved < 0 || dup2(fd, STDERR_FILENO) < 0)
			{
				CHECK(false);
				if (saved >= 0)
				{
					close(saved);
				}
				return;
			}
			cmt::test_debug = true;
			errno = ENOMEM;
			cmt::LogDebug("bytes={}", 42);
			preserved = errno == ENOMEM;
			cmt::test_debug = false;
			dup2(saved, STDERR_FILENO);
			close(saved);
		});
	CHECK_EQ(line, std::string("[cmt] bytes=42\n"));
	CHECK(preserved);
}

// src/log.h: what a diagnostic looks like on the wire. Tests and scripts match cmt's and
// cmt-sigscan's stderr text exactly, and cmt's stdout is CSV, so the property is: every level writes exactly
// "<message>\n" to stderr, nothing to stdout, before main() has done anything.

namespace {

std::string slurp(const std::string& path)
{
	std::ifstream f(path);
	std::stringstream s;
	s << f.rdbuf();
	return s.str();
}

/// Run `fn` with fd 1 and fd 2 redirected to files; returns {stdout text, stderr text}.
template <class F>
std::pair<std::string, std::string> CaptureStreams(F fn)
{
	const std::string out = std::string(CMT_TEST_TMP) + "/log_stdout",
					  err = std::string(CMT_TEST_TMP) + "/log_stderr";
	std::fflush(stdout);
	std::fflush(stderr);
	const int saved_out = dup(STDOUT_FILENO), saved_err = dup(STDERR_FILENO);
	const int o = open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644),
			  e = open(err.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
	CHECK(saved_out >= 0 && saved_err >= 0 && o >= 0 && e >= 0);
	dup2(o, STDOUT_FILENO);
	dup2(e, STDERR_FILENO);
	fn();
	std::fflush(stdout);
	std::fflush(stderr);
	dup2(saved_out, STDOUT_FILENO);
	dup2(saved_err, STDERR_FILENO);
	for (const int fd : {o, e, saved_out, saved_err})
	{
		close(fd);
	}
	return {slurp(out), slurp(err)};
}

} // namespace

TEST(every_level_writes_the_message_alone_to_stderr)
{
	// no call to InstallStderrLogger() here: the static initializer must already have done it
	const auto [out, err] = CaptureStreams(
		[]
		{
			spdlog::info("cmt: skip {}: needs {} x that in device memory, {:.1f} GiB free", "4 GiB", 2u,
						 12.34);
			spdlog::warn("cmt tune: ignoring inherited {} (the runs get only the device variables)",
						 "LD_PRELOAD");
			spdlog::error("cmt-sigscan: {}: not an ELF file", "x");
			spdlog::info("\n{:<5} {:>10}|", "dir", "size"); // a leading newline is kept, no prefix before it
		});
	CHECK_EQ(out, std::string(""));
	CHECK_EQ(err,
			 std::string("cmt: skip 4 GiB: needs 2 x that in device memory, 12.3 GiB free\n"
						 "cmt tune: ignoring inherited LD_PRELOAD (the runs get only the device variables)\n"
						 "cmt-sigscan: x: not an ELF file\n"
						 "\ndir         size|\n"));
}

TEST(install_is_idempotent)
{
	cmt::InstallStderrLogger();
	const auto [out, err] = CaptureStreams([] { spdlog::info("again"); });
	CHECK_EQ(out, std::string(""));
	CHECK_EQ(err, std::string("again\n"));
}

TEST_MAIN()
