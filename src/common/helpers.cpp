#include "helpers.h"

#include <fcntl.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

namespace cmt {

Result<std::string> CacheDirectoryPath()
{
	const char* xdg = std::getenv("XDG_CACHE_HOME");
	const char* home = std::getenv("HOME");
	if (xdg && xdg[0] == '/')
	{
		return std::string(xdg) + "/cmt";
	}
	if (home && home[0] == '/')
	{
		return std::string(home) + "/.cache/cmt";
	}
	return UsageError("set XDG_CACHE_HOME or HOME, or choose an explicit output directory");
}

Result<std::string> SelfExe()
{
	constexpr std::size_t kInitialPathBytes = 256;
	std::string path(kInitialPathBytes, '\0');
	for (;;)
	{
		const ssize_t bytes = readlink("/proc/self/exe", path.data(), path.size());
		if (bytes < 0)
		{
			return OsError("readlink /proc/self/exe");
		}
		if (static_cast<std::size_t>(bytes) < path.size())
		{
			path.resize(static_cast<std::size_t>(bytes));
			return path;
		}
		if (path.size() > path.max_size() / 2)
		{
			return OsError("readlink /proc/self/exe", ENAMETOOLONG);
		}
		path.resize(path.size() * 2);
	}
}

void WriteSortedJson(JsonWriter& writer, Json value)
{
	if (value.is_object())
	{
		std::vector<std::pair<std::string_view, Json>> fields;
		const auto object = value.get_object().value_unsafe();
		for (auto field : object)
		{
			fields.emplace_back(field.key, field.value);
		}
		std::stable_sort(fields.begin(), fields.end(),
						 [](const auto& left, const auto& right) { return left.first < right.first; });
		writer.start_object();
		bool first = true;
		for (const auto& [key, field] : fields)
		{
			JsonKey(writer, key, first);
			WriteSortedJson(writer, field);
			first = false;
		}
		writer.end_object();
	}
	else if (value.is_array())
	{
		writer.start_array();
		bool first = true;
		const auto array = value.get_array().value_unsafe();
		for (Json element : array)
		{
			if (!first)
			{
				writer.append_comma();
			}
			WriteSortedJson(writer, element);
			first = false;
		}
		writer.end_array();
	}
	else
	{
		writer.append_raw(simdjson::to_string(value));
	}
}

void WriteJsonStringArray(JsonWriter& writer, const std::vector<std::string>& values)
{
	writer.start_array();
	bool first = true;
	for (const auto& value : values)
	{
		if (!first)
		{
			writer.append_comma();
		}
		writer.escape_and_append_with_quotes(value);
		first = false;
	}
	writer.end_array();
}

Result<std::string> FinishJson(const JsonWriter& writer)
{
	std::string_view text;
	if (const auto error = writer.view().get(text))
	{
		if (error == simdjson::MEMALLOC || error == simdjson::OUT_OF_CAPACITY)
		{
			Panic("cannot allocate JSON output");
		}
		return Failure(Error::Kind::Verify, "JSON writer failed", static_cast<int>(error));
	}
	if (!writer.validate_unicode())
	{
		return Failure(Error::Kind::Verify, "JSON contains invalid UTF-8; refusing to replace bytes");
	}
	if (text.find(kNonFiniteJsonMarker[0]) != std::string_view::npos)
	{
		return Failure(Error::Kind::Verify, "JSON number is not finite (NaN or infinity); refusing to write it");
	}
	return std::string(text) + "\n";
}

Result<std::string> ReadTextFile(const std::string& path, std::size_t max_bytes)
{
	std::unique_ptr<std::FILE, int (*)(std::FILE*)> file(std::fopen(path.c_str(), "rb"), &std::fclose);
	if (!file)
	{
		const int code = errno;
		return IoFailure("open file", code);
	}
	std::string text;
	constexpr std::size_t kReadBufferBytes = 16 * 1024;
	char buffer[kReadBufferBytes];
	for (;;)
	{
		const auto remaining = max_bytes - text.size();
		const auto read_bytes = remaining < sizeof buffer ? remaining + 1 : sizeof buffer;
		errno = 0;
		const auto count = std::fread(buffer, 1, read_bytes, file.get());
		const int code = errno;
		if (std::ferror(file.get()))
		{
			return IoFailure("read file", code);
		}
		if (count > remaining)
		{
			return UsageError("file exceeds input size limit");
		}
		text.append(buffer, count);
		if (count < read_bytes)
		{
			break;
		}
	}
	errno = 0;
	if (std::fclose(file.release()) != 0)
	{
		const int code = errno;
		return IoFailure("close file", code);
	}
	return text;
}

Result<std::vector<std::string>> ListDirectory(const std::string& path)
{
	std::error_code code;
	std::filesystem::directory_iterator it(path, code), end;
	if (code)
	{
		return IoFailure("open directory", code);
	}
	std::vector<std::string> names;
	for (; !code && it != end; it.increment(code))
	{
		names.push_back(it->path().filename().string());
	}
	if (code)
	{
		return IoFailure("read directory", code);
	}
	std::sort(names.begin(), names.end());
	return names;
}

Result<bool> IsRegularFile(const std::string& path)
{
	std::error_code code;
	const auto status = std::filesystem::status(path, code);
	if (code == std::errc::no_such_file_or_directory || code == std::errc::not_a_directory)
	{
		return false;
	}
	if (code)
	{
		return IoFailure("inspect file", code);
	}
	return std::filesystem::is_regular_file(status);
}

Result<void> WriteTextFile(const std::string& path, const std::string& text)
{
	std::unique_ptr<std::FILE, int (*)(std::FILE*)> file(std::fopen(path.c_str(), "wb"), &std::fclose);
	if (!file)
	{
		const int code = errno;
		return IoFailure("open file", code);
	}
	errno = 0;
	const auto count = std::fwrite(text.data(), 1, text.size(), file.get());
	const int code = errno;
	if (count != text.size() || std::ferror(file.get()))
	{
		return IoFailure("write file", code);
	}
	errno = 0;
	if (std::fclose(file.release()) != 0)
	{
		const int close_code = errno;
		return IoFailure("close file", close_code);
	}
	return {};
}

namespace {

struct Descriptor
{
	int fd = -1;
	Descriptor() = default;
	explicit Descriptor(int value) : fd(value)
	{
	}
	Descriptor(const Descriptor&) = delete;
	Descriptor& operator=(const Descriptor&) = delete;
	~Descriptor()
	{
		if (fd >= 0)
		{
			close(fd);
		}
	}
};

Result<void> WriteAll(int fd, std::string_view text)
{
	while (!text.empty())
	{
		const ssize_t wrote = write(fd, text.data(), text.size());
		if (wrote < 0 && errno == EINTR)
		{
			continue;
		}
		if (wrote <= 0)
		{
			return IoFailure("write file", wrote < 0 ? errno : EIO);
		}
		text.remove_prefix(static_cast<std::size_t>(wrote));
	}
	return {};
}

Result<std::pair<int, std::string>> CreateTemporary(int directory, const std::string& name, mode_t mode)
{
	static constexpr char kDigits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
	for (int attempt = 0; attempt < 128; attempt++)
	{
		unsigned char random[6];
		if (getrandom(random, sizeof random, 0) != static_cast<ssize_t>(sizeof random))
		{
			return OsError("create a temporary file name");
		}
		std::string candidate = "." + name + ".";
		for (unsigned char byte : random)
		{
			candidate += kDigits[byte % (sizeof kDigits - 1)];
		}
		const int fd =
			openat(directory, candidate.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode);
		if (fd >= 0)
		{
			return std::pair{fd, std::move(candidate)};
		}
		if (errno != EEXIST)
		{
			return OsError("create temporary file");
		}
	}
	return OsError("create temporary file", EEXIST);
}

Result<void> WriteAtomically(const std::string& path, const std::string& text, bool replace)
{
	std::filesystem::path target(path);
	std::error_code code;
	if (replace && std::filesystem::is_symlink(target, code))
	{
		target = std::filesystem::weakly_canonical(target, code);
		if (code)
		{
			return IoFailure("resolve symbolic link", code);
		}
	}
	const auto name = target.filename().string();
	if (name.empty() || name == "." || name == "..")
	{
		return IoFailure("rename file", EISDIR);
	}
	const auto parent = target.parent_path();
	Descriptor directory(open(parent.empty() ? "." : parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
	if (directory.fd < 0)
	{
		return OsError("open directory");
	}
	struct stat existing{};
	const bool replaces =
		replace && fstatat(directory.fd, name.c_str(), &existing, AT_SYMLINK_NOFOLLOW) == 0 && S_ISREG(existing.st_mode);
	auto created = CreateTemporary(directory.fd, name, replaces ? 0600 : 0666);
	if (!created)
	{
		return Failure(created.error());
	}
	const std::string temporary = std::move(created->second);
	Descriptor file(created->first);
	const auto fail = [&](const Error& error) -> Result<void>
	{
		unlinkat(directory.fd, temporary.c_str(), 0);
		return Failure(error);
	};
	if (replaces)
	{
		if (geteuid() == 0 && fchown(file.fd, existing.st_uid, existing.st_gid) != 0)
		{
			std::fprintf(stderr, "cmt: could not keep the owner of %s\n", path.c_str());
		}
		if (fchmod(file.fd, existing.st_mode & 0777) != 0)
		{
			return fail(OsError("set file permissions").value());
		}
	}
	if (auto written = WriteAll(file.fd, text); !written)
	{
		return fail(written.error());
	}
	if (fsync(file.fd) != 0)
	{
		return fail(OsError("sync file").value());
	}
	const int closed = close(file.fd);
	file.fd = -1;
	if (closed != 0)
	{
		return fail(OsError("close file").value());
	}
	if (replace)
	{
		if (renameat(directory.fd, temporary.c_str(), directory.fd, name.c_str()) != 0)
		{
			return fail(OsError("rename file").value());
		}
	}
	else if (renameat2(directory.fd, temporary.c_str(), directory.fd, name.c_str(), RENAME_NOREPLACE) != 0)
	{
		if (errno != EINVAL && errno != ENOSYS)
		{
			return fail(OsError("rename file").value());
		}
		if (linkat(directory.fd, temporary.c_str(), directory.fd, name.c_str(), 0) != 0)
		{
			return fail(OsError("rename file").value());
		}
		unlinkat(directory.fd, temporary.c_str(), 0);
	}
	fsync(directory.fd);
	return {};
}

} // namespace

Result<void> WriteTextFileAtomic(const std::string& path, const std::string& text)
{
	return WriteAtomically(path, text, true);
}

Result<void> WriteNewTextFileAtomic(const std::string& path, const std::string& text)
{
	return WriteAtomically(path, text, false);
}

Result<void> MakeDirectories(const std::string& path)
{
	std::error_code code;
	std::filesystem::create_directories(path, code);
	if (code)
	{
		return IoFailure("create directories", code);
	}
	return {};
}

Result<void> WriteStdout(const std::string& text)
{
	errno = 0;
	const auto written = std::fwrite(text.data(), 1, text.size(), stdout);
	if (written != text.size() || std::fflush(stdout) != 0 || std::ferror(stdout))
	{
		const int code = errno;
		return IoFailure("write stdout", code);
	}
	return {};
}

} // namespace cmt
