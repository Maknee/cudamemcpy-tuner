#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <string_view>
#include <utility>

#include <fmt/format.h>
#include <unistd.h>

namespace cmt {
namespace logging {

struct PreserveErrno
{
	const int value = errno;

	~PreserveErrno()
	{
		errno = value;
	}
};

constexpr std::size_t kLineBytes = 1024; // < PIPE_BUF (4096 on Linux): one atomic write

inline void WriteAll(int fd, const char* p, std::size_t n) noexcept
{
	const PreserveErrno preserve;
	while (n > 0)
	{
		const ssize_t w = ::write(fd, p, n);
		if (w <= 0)
		{
			if (w < 0 && errno == EINTR)
			{
				continue;
			}
			return;
		}
		p += w;
		n -= static_cast<std::size_t>(w);
	}
}

template <class... T>
void WriteLine(int fd, std::string_view prefix_text, fmt::format_string<T...> f, T&&... args) noexcept
{
	const PreserveErrno preserve;
	char buf[kLineBytes];
	const std::size_t prefix = std::min(std::size(prefix_text), kLineBytes - 5);
	constexpr char kCut[] = "...";
	if (prefix != 0)
	{
		std::memcpy(buf, std::data(prefix_text), prefix);
	}
	const std::size_t room = kLineBytes - prefix - (sizeof(kCut) - 1) - 1;
	std::size_t len = prefix;
	const auto result = fmt::format_to_n(buf + prefix, room, f, std::forward<T>(args)...);
	if (result.size <= room)
	{
		len += result.size;
	}
	else
	{
		len += room;
		std::memcpy(buf + len, kCut, sizeof(kCut) - 1);
		len += sizeof(kCut) - 1;
	}
	buf[len++] = '\n';
	WriteAll(fd, buf, len);
}

} // namespace logging
} // namespace cmt

namespace cmt {

inline constexpr const char* kHookLogPrefix = "[cmt] ";

bool DebugLoggingEnabled();

template <class... T>
void LogTo(int fd, fmt::format_string<T...> format, T&&... args) noexcept
{
	cmt::logging::WriteLine(fd, kHookLogPrefix, format, std::forward<T>(args)...);
}

template <class... T>
void LogLine(fmt::format_string<T...> format, T&&... args) noexcept
{
	LogTo(STDERR_FILENO, format, std::forward<T>(args)...);
}

template <class... T>
void LogDebug(fmt::format_string<T...> format, T&&... args) noexcept
{
	const cmt::logging::PreserveErrno preserve;
	if (DebugLoggingEnabled())
	{
		LogLine(format, std::forward<T>(args)...);
	}
}

} // namespace cmt
