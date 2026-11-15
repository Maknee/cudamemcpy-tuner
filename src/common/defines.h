#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cmt {
using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
								 std::chrono::high_resolution_clock, std::chrono::steady_clock>;
} // namespace cmt

namespace cmt {

constexpr std::size_t kMaxBytes = std::size_t{1} << 40;
constexpr std::uint32_t kMaxReps = std::uint32_t{1} << 20;

inline std::uint64_t PatternWord(std::uint64_t i, std::uint64_t seed)
{
	return (i * 0x9E3779B97F4A7C15ull) ^ seed;
}

inline double NowSeconds()
{
	return std::chrono::duration<double>(cmt::Clock::now().time_since_epoch()).count();
}

} // namespace cmt

namespace cmt {

enum class Dir : std::uint8_t
{
	HtoD,
	DtoH
};

inline const char* DirName(Dir direction)
{
	return direction == Dir::HtoD ? "htod" : "dtoh";
}

struct CopyPhases
{
	double setup = 0;
	double copy = 0;
	double teardown = 0;
};

} // namespace cmt
