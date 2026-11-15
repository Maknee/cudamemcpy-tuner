#include "tests/check.h"
#include "common/helpers.h"
#include "common/memory.h"
#include "common/cpu.h"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>

#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace cmt;

namespace {
struct Scratch
{
	std::string path = std::string(CMT_TEST_TMP) + "/io_XXXXXX";

	Scratch()
	{
		if (!mkdtemp(path.data()))
		{
			path.clear();
		}
	}

	~Scratch()
	{
		if (!path.empty())
		{
			std::error_code error;
			fs::remove_all(path, error);
		}
	}
};
} // namespace

TEST(alignment_checks_zero_overflow_and_non_power_of_two)
{
	CHECK_EQ(AlignUp(0, 4096).value_or(1), std::uint64_t{0});
	CHECK_EQ(AlignUp(4096, 4096).value_or(0), std::uint64_t{4096});
	CHECK_EQ(AlignUp(4097, 4096).value_or(0), std::uint64_t{8192});
	CHECK_EQ(AlignUp(13, 6).value_or(0), std::uint64_t{18});
	CHECK(!AlignUp(1, 0));
	CHECK(!AlignUp(UINT64_MAX, 4096));
	CHECK_EQ(AlignUp(UINT64_MAX, 1).value_or(0), UINT64_MAX);
}

TEST(error_propagation_reuses_static_message_storage)
{
	static constexpr char message[] = "native operation failed";
	const Error error(Error::Kind::Os, message, EIO);
	const Result<int> value = Failure(error);
	const Result<void> status = Failure(value.error());
	CHECK(status.error().Message() == message);
	CHECK_EQ(status.error().GetCode(), EIO);
	static_assert(std::is_trivially_copyable_v<Error>);
	static_assert(!std::is_constructible_v<Error, Error::Kind, std::string>);
	static_assert(!std::is_constructible_v<Error, Error::Kind, std::string_view>);
}

TEST(missing_empty_and_unreadable_are_distinct)
{
	Scratch dir;
	CHECK(!dir.path.empty());
	if (dir.path.empty())
	{
		return;
	}
	const auto missing = ReadTextFile(dir.path + "/missing");
	CHECK(!missing);
	if (!missing)
	{
		CHECK_EQ(missing.error().GetCode(), ENOENT);
	}
	CHECK(WriteTextFile(dir.path + "/empty", ""));
	const auto empty = ReadTextFile(dir.path + "/empty");
	CHECK(empty && empty->empty());
	const auto directory = ReadTextFile(dir.path);
	CHECK(!directory);
	if (!directory)
	{
		CHECK(directory.error().GetKind() == Error::Kind::Os);
		CHECK_EQ(directory.error().GetCode(), EISDIR);
	}
}

TEST(common_paths_distinguish_missing_files_and_report_the_executable)
{
	Scratch dir;
	CHECK(!dir.path.empty());
	if (dir.path.empty())
	{
		return;
	}
	const auto missing = IsRegularFile(dir.path + "/missing");
	CHECK(missing && !*missing);
	const auto directory = IsRegularFile(dir.path);
	CHECK(directory && !*directory);
	CHECK(WriteTextFile(dir.path + "/file", "contents"));
	const auto regular = IsRegularFile(dir.path + "/file");
	CHECK(regular && *regular);
	const auto executable = SelfExe();
	CHECK(executable && !executable->empty());
	if (executable)
	{
		std::error_code error;
		CHECK(fs::equivalent(*executable, "/proc/self/exe", error));
		CHECK(!error);
	}
}

TEST(generic_json_helpers_preserve_nulls_order_and_final_newline)
{
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "known", std::optional<bool>(false), true);
	JsonField(writer, "zero", std::optional<int>(0));
	JsonField(writer, "unknown", std::optional<int>{});
	JsonKey(writer, "names");
	WriteJsonStringArray(writer, {"a", "b\n"});
	writer.end_object();
	const auto text = FinishJson(writer);
	CHECK(text);
	if (text)
	{
		CHECK_EQ(*text,
				 std::string("{\"known\":false,\"zero\":0,\"unknown\":null,\"names\":[\"a\",\"b\\n\"]}\n"));
	}
}

TEST(directory_scans_are_sorted_and_distinguish_missing_from_empty)
{
	Scratch dir;
	CHECK(!dir.path.empty());
	if (dir.path.empty())
	{
		return;
	}
	auto empty = ListDirectory(dir.path);
	CHECK(empty && empty->empty());
	CHECK(WriteTextFile(dir.path + "/z.txt", ""));
	CHECK(MakeDirectories(dir.path + "/a"));
	CHECK(WriteTextFile(dir.path + "/m.txt", ""));
	auto names = ListDirectory(dir.path);
	CHECK(names);
	if (names)
	{
		CHECK(*names == (std::vector<std::string>{"a", "m.txt", "z.txt"}));
	}
	auto missing = ListDirectory(dir.path + "/missing");
	CHECK(!missing);
	if (!missing)
	{
		CHECK(missing.error().GetKind() == Error::Kind::Os);
		CHECK_EQ(missing.error().GetCode(), ENOENT);
	}
	auto file = ListDirectory(dir.path + "/z.txt");
	CHECK(!file);
	if (!file)
	{
		CHECK_EQ(file.error().GetCode(), ENOTDIR);
	}
}

TEST(buffered_and_large_writes_cannot_hide_a_full_device)
{
	for (std::size_t size : {std::size_t{8}, std::size_t{65536}})
	{
		const auto result = WriteTextFile("/dev/full", std::string(size, 'x'));
		CHECK(!result);
		if (!result)
		{
			CHECK(result.error().GetKind() == Error::Kind::Os);
			CHECK_EQ(result.error().GetCode(), ENOSPC);
			CHECK_EQ(result.error().ExitCode(), 4);
		}
	}
}

TEST(failed_atomic_write_preserves_the_old_file)
{
	Scratch dir;
	CHECK(!dir.path.empty());
	if (dir.path.empty())
	{
		return;
	}
	const auto path = dir.path + "/table";
	CHECK(WriteTextFile(path, "old verified table\n"));
	// A directory nobody may create in: the temporary file cannot be made (root may, so not checked as root).
	if (geteuid() != 0)
	{
		fs::permissions(dir.path, fs::perms::owner_read | fs::perms::owner_exec);
		const auto result = WriteTextFileAtomic(path, "new table\n");
		fs::permissions(dir.path, fs::perms::owner_all);
		CHECK(!result);
		if (!result)
		{
			CHECK_EQ(result.error().GetCode(), EACCES);
		}
		CHECK_EQ(ReadTextFile(path).value_or("<read failed>"), std::string("old verified table\n"));
	}
	// A failed rename (a directory in the way) leaves no temporary file behind.
	CHECK(MakeDirectories(dir.path + "/in-the-way"));
	CHECK(!WriteTextFileAtomic(dir.path + "/in-the-way", "new table\n"));
	CHECK(ListDirectory(dir.path).value_or(std::vector<std::string>{}) ==
		  (std::vector<std::string>{"in-the-way", "table"}));
}

// The old fixed `PATH.part` name followed a link planted there: now the temporary name is unique and created
// with O_EXCL | O_NOFOLLOW, so a planted `.part` is neither written through nor used.
TEST(atomic_write_ignores_a_planted_part_file)
{
	Scratch dir;
	CHECK(!dir.path.empty());
	if (dir.path.empty())
	{
		return;
	}
	const auto path = dir.path + "/table";
	const auto victim = dir.path + "/victim";
	CHECK(WriteTextFile(path, "old\n"));
	CHECK(WriteTextFile(victim, "victim\n"));
	std::error_code error;
	fs::create_symlink(victim, path + ".part", error);
	CHECK(!error);
	CHECK(WriteTextFileAtomic(path, "new\n"));
	CHECK_EQ(ReadTextFile(path).value_or(""), std::string("new\n"));
	CHECK_EQ(ReadTextFile(victim).value_or(""), std::string("victim\n"));
	CHECK(fs::is_symlink(path + ".part", error));
	CHECK(ListDirectory(dir.path).value_or(std::vector<std::string>{}) ==
		  (std::vector<std::string>{"table", "table.part", "victim"}));
}

TEST(atomic_write_keeps_the_mode_and_writes_through_a_link)
{
	Scratch dir;
	CHECK(!dir.path.empty());
	if (dir.path.empty())
	{
		return;
	}
	const auto mode = [](const std::string& path)
	{
		struct stat status{};
		return lstat(path.c_str(), &status) == 0 ? status.st_mode & 07777 : 07777u;
	};
	// A replaced file keeps its permission bits.
	const auto path = dir.path + "/table";
	CHECK(WriteTextFile(path, "old\n"));
	CHECK(chmod(path.c_str(), 0640) == 0);
	CHECK(WriteTextFileAtomic(path, "new\n"));
	CHECK_EQ(ReadTextFile(path).value_or(""), std::string("new\n"));
	CHECK_EQ(mode(path), 0640u);
	// A new file gets 0666 less the umask, as fopen gives it.
	const mode_t mask = umask(022);
	CHECK(WriteTextFileAtomic(dir.path + "/fresh", "x"));
	umask(mask);
	CHECK_EQ(mode(dir.path + "/fresh"), 0644u);
	// Through a symbolic link: the file it names is replaced, the link stays a link.
	CHECK(MakeDirectories(dir.path + "/real"));
	const auto target = dir.path + "/real/tune.json";
	CHECK(WriteTextFile(target, "old\n"));
	std::error_code error;
	fs::create_symlink("real/tune.json", dir.path + "/link.json", error);
	CHECK(!error);
	CHECK(WriteTextFileAtomic(dir.path + "/link.json", "new\n"));
	CHECK(fs::is_symlink(dir.path + "/link.json", error));
	CHECK_EQ(ReadTextFile(target).value_or(""), std::string("new\n"));
	CHECK(ListDirectory(dir.path + "/real").value_or(std::vector<std::string>{}) ==
		  (std::vector<std::string>{"tune.json"}));
}

TEST(a_new_file_never_replaces_anything)
{
	Scratch dir;
	CHECK(!dir.path.empty());
	if (dir.path.empty())
	{
		return;
	}
	const auto path = dir.path + "/share.json";
	CHECK(WriteNewTextFileAtomic(path, "first\n"));
	CHECK_EQ(ReadTextFile(path).value_or(""), std::string("first\n"));
	const auto again = WriteNewTextFileAtomic(path, "second\n");
	CHECK(!again);
	if (!again)
	{
		CHECK(again.error().GetKind() == Error::Kind::Os);
		CHECK_EQ(again.error().GetCode(), EEXIST);
	}
	CHECK_EQ(ReadTextFile(path).value_or(""), std::string("first\n"));
	// A link, even a dangling one, is something: it is neither followed nor replaced.
	std::error_code error;
	fs::create_symlink(dir.path + "/elsewhere", dir.path + "/dangling", error);
	CHECK(!error);
	const auto dangling = WriteNewTextFileAtomic(dir.path + "/dangling", "x");
	CHECK(!dangling && dangling.error().GetCode() == EEXIST);
	CHECK(!fs::exists(dir.path + "/elsewhere"));
	CHECK(ListDirectory(dir.path).value_or(std::vector<std::string>{}) ==
		  (std::vector<std::string>{"dangling", "share.json"}));
}

TEST(json_refuses_numbers_it_cannot_represent)
{
	for (const double value : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
							   -std::numeric_limits<double>::infinity()})
	{
		JsonWriter writer;
		writer.start_object();
		JsonField(writer, "ok", 1.5, true);
		JsonField(writer, "rate", std::optional<double>(value));
		writer.end_object();
		const auto text = FinishJson(writer);
		CHECK(!text);
		if (!text)
		{
			CHECK(text.error().GetKind() == Error::Kind::Verify);
		}
	}
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "rate", 2.5, true);
	JsonKey(writer, "samples");
	writer.start_array();
	JsonValue(writer, 0.25);
	writer.end_array();
	JsonField(writer, "text", std::string_view("\x01 escaped"));
	writer.end_object();
	CHECK_EQ(FinishJson(writer).value_or(""), std::string("{\"rate\":2.5,\"samples\":[0.25],\"text\":\"\\u0001 escaped\"}\n"));
}

TEST(reading_a_device_by_mistake_stops_at_the_default_limit)
{
	const auto zero = ReadTextFile("/dev/zero");
	CHECK(!zero);
	if (!zero)
	{
		CHECK(zero.error().IsUsage());
	}
}

TEST(filesystem_error_uses_its_code_instead_of_errno)
{
	errno = EDOM;
	const auto error = IoFailure("test operation", std::make_error_code(std::errc::permission_denied));
	CHECK_EQ(error.value().GetCode(), EACCES);
	Scratch dir;
	CHECK(!dir.path.empty());
	if (dir.path.empty())
	{
		return;
	}
	std::error_code filesystem_error;
	fs::create_directory(dir.path + "/target", filesystem_error);
	CHECK(!filesystem_error);
	if (filesystem_error)
	{
		return;
	}
	errno = EDOM;
	const auto renamed = WriteTextFileAtomic(dir.path + "/target", "contents");
	CHECK(!renamed);
	if (!renamed)
	{
		CHECK_EQ(renamed.error().GetCode(), EISDIR);
		CHECK(std::string(renamed.error().Message()).find("rename ") == 0);
	}
	CHECK(fs::is_directory(dir.path + "/target", filesystem_error));
}

TEST(recoverable_failure_keeps_the_same_error_through_propagation)
{
	const auto original = Failure(Error::Kind::Cuda, "copy: original cause", 123);
	const Result<std::string> first = original;
	const Result<void> second = cmt::Failure(first.error());
	CHECK(!second);
	CHECK_EQ(second.error().GetCode(), 123);
	CHECK_EQ(second.error().ExitCode(), 5);
	CHECK_EQ(std::string(second.error().Message()), std::string("copy: original cause"));
}

TEST(failed_io_does_not_leak_file_descriptors)
{
	const auto count_fds = []() -> Result<std::size_t>
	{
		std::error_code error;
		std::filesystem::directory_iterator it("/proc/self/fd", error), end;
		std::size_t count = 0;
		for (; !error && it != end; it.increment(error))
		{
			++count;
		}
		if (error)
		{
			return IoFailure("count descriptors", error);
		}
		return count;
	};
	auto before = count_fds();
	CHECK(before);
	if (!before)
	{
		return;
	}
	for (std::size_t i = 0; i < 64; i++)
	{
		CHECK(!WriteTextFile("/dev/full", "buffered"));
		CHECK(!ReadTextFile("/proc/self"));
	}
	const auto after = count_fds();
	CHECK(after);
	if (after)
	{
		CHECK_EQ(*before, *after);
	}
}

TEST_MAIN()
