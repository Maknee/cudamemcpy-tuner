// A minimal test harness: no dependency to download or vendor, so the tests build anywhere the tool
// builds (the A100 VM had no network access to fetch one). Each test file is one executable; ctest
// runs them. A failed CHECK prints the expression and keeps going, so one run reports every failure.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <functional>
#include <string>
#include <vector>
#include <type_traits>
#include <utility>

namespace check {

// Fixture setup must succeed before a test can inspect its value. An unexpected
// error fails this test process; it is not an exception or an unchecked value().
template <class Result>
auto Require(Result result)
{
	if (!result)
	{
		std::fprintf(stderr, "test setup failed: %s\n", result.error().Message());
		std::abort();
	}
	if constexpr (!std::is_void_v<typename Result::value_type>)
	{
		return std::move(*result);
	}
}

inline std::size_t& failures()
{
	static std::size_t f = 0;
	return f;
}

struct Case
{
	const char* name;
	std::function<void()> fn;
};

inline std::vector<Case>& cases()
{
	static std::vector<Case> c;
	return c;
}

struct Reg
{
	Reg(const char* n, std::function<void()> f)
	{
		cases().push_back({n, std::move(f)});
	}
};

inline int run_all()
{
	for (auto& c : cases())
	{
		const std::size_t before = failures();
		c.fn();
		std::printf("%s %s\n", failures() == before ? "ok  " : "FAIL", c.name);
	}
	std::printf("%zu cases, %zu failed checks\n", std::size(cases()), failures());
	return failures() ? 1 : 0;
}

inline int exit_code()
{
	if (failures() != 0)
	{
		std::fprintf(stderr, "FAILED: %zu check(s)\n", failures());
		return 1;
	}
	std::printf("all checks passed\n");
	return 0;
}

// A zeroed, 4 KiB aligned byte buffer.
class Buffer
{
public:
	explicit Buffer(std::size_t n, uint8_t fill = 0) : size_(n)
	{
		if (posix_memalign(reinterpret_cast<void**>(&data_), 4096, n == 0 ? 1 : n) != 0)
		{
			std::abort();
		}
		std::memset(data_, fill, n);
	}

	~Buffer()
	{
		std::free(data_);
	}

	Buffer(const Buffer&) = delete;
	Buffer& operator=(const Buffer&) = delete;

	uint8_t* data()
	{
		return data_;
	}

	std::size_t size() const
	{
		return size_;
	}

	uint8_t& operator[](std::size_t i)
	{
		return data_[i];
	}

private:
	uint8_t* data_ = nullptr;
	std::size_t size_;
};

// Non-periodic byte pattern (a 64-bit mix of index and seed), so a
// chunk copied to the wrong offset never matches by accident.
inline uint8_t pattern(std::size_t i, unsigned seed = 0)
{
	uint64_t x = static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull + seed * 0xD1B54A32D192ED03ull + 1;
	x ^= x >> 31;
	x *= 0xBF58476D1CE4E5B9ull;
	x ^= x >> 29;
	return static_cast<uint8_t>(x >> 56);
}

} // namespace check

#define CHECK_CAT2(a, b) a##b
#define CHECK_CAT(a, b) CHECK_CAT2(a, b)
#define TEST(name)                                                                                           \
	static void name();                                                                                      \
	static check::Reg CHECK_CAT(reg_, name)(#name, name);                                                    \
	static void name()

#define CHECK(cond)                                                                                          \
	do                                                                                                       \
	{                                                                                                        \
		if (!(cond))                                                                                         \
		{                                                                                                    \
			check::failures()++;                                                                             \
			std::printf("  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond);                           \
		}                                                                                                    \
	} while (0)

#define CHECK_EQ(a, b)                                                                                       \
	do                                                                                                       \
	{                                                                                                        \
		const auto va_ = (a);                                                                                \
		const auto vb_ = (b);                                                                                \
		if (!(va_ == vb_))                                                                                   \
		{                                                                                                    \
			check::failures()++;                                                                             \
			std::printf("  %s:%d: CHECK_EQ(%s, %s) failed: %s != %s\n", __FILE__, __LINE__, #a, #b,          \
						check::show(va_).c_str(), check::show(vb_).c_str());                                 \
		}                                                                                                    \
	} while (0)

// Recoverable errors are returned as values; the expression is evaluated once.
#define CHECK_ERROR_KIND(expression, kind)                                                                   \
	do                                                                                                       \
	{                                                                                                        \
		auto checked_result = (expression);                                                                  \
		CHECK(!checked_result);                                                                              \
		if (!checked_result)                                                                                 \
			CHECK(checked_result.error().GetKind() == cmt::Error::Kind::kind);                               \
	} while (false)

// Panic tests run in a child so an invariant failure cannot end the test suite.
// Only SIGABRT counts as the expected failure; a segfault must not pass.
#define CHECK_PANICS(statement)                                                                              \
	do                                                                                                       \
	{                                                                                                        \
		const pid_t child = fork();                                                                          \
		CHECK(child >= 0);                                                                                   \
		if (child == 0)                                                                                      \
		{                                                                                                    \
			const rlimit no_core{0, 0};                                                                      \
			setrlimit(RLIMIT_CORE, &no_core);                                                                \
			statement;                                                                                       \
			_exit(0);                                                                                        \
		}                                                                                                    \
		if (child > 0)                                                                                       \
		{                                                                                                    \
			int status = 0;                                                                                  \
			pid_t reaped;                                                                                    \
			do                                                                                               \
			{                                                                                                \
				reaped = waitpid(child, &status, 0);                                                         \
			} while (reaped < 0 && errno == EINTR);                                                          \
			CHECK(reaped == child && WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);                    \
		}                                                                                                    \
	} while (false)

namespace check {
inline std::string show(const std::string& s)
{
	return "\"" + s + "\"";
}

inline std::string show(const char* s)
{
	return show(std::string(s));
}

inline std::string show(bool b)
{
	return b ? "true" : "false";
}

template <class T>
std::string show(const T& v)
{
	return std::to_string(v);
}

template <class T>
std::string show(const std::vector<T>& v)
{
	std::string s = "[";
	for (std::size_t i = 0; i < std::size(v); i++)
	{
		s += (i ? "," : "") + show(v[i]);
	}
	return s + "]";
}
} // namespace check

#define TEST_MAIN()                                                                                          \
	int main()                                                                                               \
	{                                                                                                        \
		return check::run_all();                                                                             \
	}

// CHECK with a printf-style context message on failure.
#define CHECKF(cond, ...)                                                                                    \
	do                                                                                                       \
	{                                                                                                        \
		if (!(cond))                                                                                         \
		{                                                                                                    \
			++check::failures();                                                                             \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s: ", __FILE__, __LINE__, #cond);                    \
			std::fprintf(stderr, __VA_ARGS__);                                                               \
			std::fprintf(stderr, "\n");                                                                      \
		}                                                                                                    \
	} while (0)
