#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#include <sys/auxv.h>

#include "aliases.h"
#include "error.h"

namespace cmt {

[[nodiscard]] Result<std::string> SelfExe();

[[nodiscard]] Result<std::string> CacheDirectoryPath();

inline bool SecureExecution() noexcept
{
	return getauxval(AT_SECURE) != 0;
}

inline void JsonKey(JsonWriter& writer, std::string_view key, bool first = false)
{
	if (!first)
	{
		writer.append_comma();
	}
	writer.escape_and_append_with_quotes(key);
	writer.append_colon();
}

inline constexpr char kNonFiniteJsonMarker[] = "\x01non-finite";

template <class T>
void JsonValue(JsonWriter& writer, const T& value)
{
	if constexpr (std::is_floating_point_v<T>)
	{
		if (!std::isfinite(value))
		{
			writer.append_raw(kNonFiniteJsonMarker);
			return;
		}
	}
	writer.append(value);
}

template <class T>
void JsonField(JsonWriter& writer, std::string_view key, const T& value, bool first = false)
{
	if constexpr (std::is_floating_point_v<T>)
	{
		JsonKey(writer, key, first);
		JsonValue(writer, value);
	}
	else
	{
		if (!first)
		{
			writer.append_comma();
		}
		writer.append_key_value(key, value);
	}
}

template <class T>
void JsonField(JsonWriter& writer, std::string_view key, const std::optional<T>& value, bool first = false)
{
	if (value)
	{
		JsonField(writer, key, *value, first);
	}
	else
	{
		JsonField(writer, key, nullptr, first);
	}
}

void WriteJsonStringArray(JsonWriter& writer, const std::vector<std::string>& values);

void WriteSortedJson(JsonWriter& writer, Json value);

[[nodiscard]] Result<std::string> FinishJson(const JsonWriter& writer);

constexpr std::optional<std::uint64_t> AlignUp(std::uint64_t bytes, std::uint64_t alignment) noexcept
{
	if (alignment == 0)
	{
		return std::nullopt;
	}
	const auto remainder = bytes % alignment;
	const auto padding = remainder == 0 ? 0 : alignment - remainder;
	if (bytes > std::numeric_limits<std::uint64_t>::max() - padding)
	{
		return std::nullopt;
	}
	return bytes + padding;
}

template <std::size_t N>
[[nodiscard]] inline MaybeError IoFailure(const char (&operation)[N], int code) noexcept
{
	return OsError(operation, code ? code : EIO);
}

template <std::size_t N>
[[nodiscard]] inline MaybeError IoFailure(const char (&operation)[N], const std::error_code& code) noexcept
{
	return OsError(operation, code.value());
}

[[nodiscard]] Result<bool> IsRegularFile(const std::string& path);
inline constexpr std::size_t kDefaultMaxReadBytes = std::size_t{256} << 20;
[[nodiscard]] Result<std::string> ReadTextFile(const std::string& path, std::size_t max_bytes = kDefaultMaxReadBytes);
[[nodiscard]] Result<std::vector<std::string>> ListDirectory(const std::string& path);

[[nodiscard]] Result<void> WriteTextFile(const std::string& path, const std::string& text);
[[nodiscard]] Result<void> WriteTextFileAtomic(const std::string& path, const std::string& text);
[[nodiscard]] Result<void> WriteNewTextFileAtomic(const std::string& path, const std::string& text);
[[nodiscard]] Result<void> MakeDirectories(const std::string& path);
[[nodiscard]] Result<void> WriteStdout(const std::string& text);

} // namespace cmt
