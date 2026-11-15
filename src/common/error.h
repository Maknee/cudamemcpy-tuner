#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <new>
#include <string_view>
#include <type_traits>
#include <utility>

#include <tl/expected.hpp>
#include <unistd.h>

namespace cmt {

constexpr int kSignalExitBase = 128;

inline void InstallFatalAllocationHandler() noexcept
{
	std::set_new_handler(&std::abort);
}

// Error messages borrow static storage.
class Error
{
public:
	enum class Kind : std::uint8_t
	{
		Usage,
		Unsupported,
		Os,
		Cuda,
		Verify,
		Interrupted
	};

	template <std::size_t N>
	constexpr Error(Kind error_kind, const char (&text)[N], int error_code = 0) noexcept
		: kind(error_kind), message(text, N - 1), code(error_code)
	{
	}

	Kind GetKind() const noexcept
	{
		return kind;
	}

	int GetCode() const noexcept
	{
		return code;
	}

	const char* Message() const noexcept
	{
		return message.data();
	}

	int ExitCode() const noexcept
	{
		switch (kind)
		{
		case Kind::Usage:
			return 2;
		case Kind::Unsupported:
			return 3;
		case Kind::Os:
			return 4;
		case Kind::Cuda:
			return 5;
		case Kind::Verify:
			return 6;
		case Kind::Interrupted:
			return kSignalExitBase + code;
		}
		return 1;
	}

	bool IsUsage() const noexcept
	{
		return kind == Kind::Usage;
	}

	bool IsUnsupported() const noexcept
	{
		return kind == Kind::Unsupported;
	}

private:
	Kind kind;
	std::string_view message;
	int code;
};

static_assert(std::is_trivially_copyable_v<Error>, "Error must not own or copy message storage");

template <class T>
using Result = tl::expected<T, Error>;

using MaybeError = tl::unexpected<Error>;

[[nodiscard]] inline MaybeError Failure(Error error) noexcept
{
	return tl::make_unexpected(error);
}

template <std::size_t N>
[[nodiscard]] inline MaybeError Failure(Error::Kind kind, const char (&message)[N], int code = 0) noexcept
{
	return Failure(Error(kind, message, code));
}

[[noreturn]] inline void Panic(std::string_view message) noexcept
{
	constexpr std::string_view prefix = "cmt panic: ";
	for (auto text : {prefix, message, std::string_view{"\n"}})
	{
		while (!text.empty())
		{
			const auto written = ::write(STDERR_FILENO, text.data(), text.size());
			if (written > 0)
			{
				text.remove_prefix(static_cast<std::size_t>(written));
			}
			else if (written < 0 && errno == EINTR)
			{
				continue;
			}
			else
			{
				break;
			}
		}
	}
	std::abort();
}

template <std::size_t N>
[[nodiscard]] inline MaybeError UsageError(const char (&message)[N]) noexcept
{
	return Failure(Error::Kind::Usage, message);
}

template <std::size_t N>
[[nodiscard]] inline MaybeError Unsupported(const char (&message)[N]) noexcept
{
	return Failure(Error::Kind::Unsupported, message);
}

template <std::size_t N>
[[nodiscard]] inline MaybeError OsError(const char (&operation)[N], int code = errno) noexcept
{
	return Failure(Error::Kind::Os, operation, code);
}

} // namespace cmt

#define CMT_RETURN_IF_ERROR(expression)                                                                      \
	do                                                                                                       \
	{                                                                                                        \
		auto cmt_status_result = (expression);                                                               \
		if (!cmt_status_result)                                                                              \
		{                                                                                                    \
			return cmt::Failure(std::move(cmt_status_result.error()));                                       \
		}                                                                                                    \
	} while (false)
