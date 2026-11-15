#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "arch/copy.h"
#include "common/aliases.h"
#include "config.h"

namespace cmt {

struct Copy2DView
{
	uint8_t* dst;
	uint64_t dst_pitch;
	uint64_t dst_height;
	const uint8_t* src;
	uint64_t src_pitch;
	uint64_t src_height;
	uint64_t width;
	uint64_t height;
	uint64_t depth;
};

inline void Run2DSerial(CopyLoop loop, const Copy2DView& copy)
{
	hwcopy::WithLoop(loop.id,
					 [&](auto implementation)
					 {
						 using Loop = decltype(implementation);
						 for (uint64_t slice = 0; slice < copy.depth; ++slice)
						 {
							 uint8_t* dst_row = copy.dst + slice * copy.dst_pitch * copy.dst_height;
							 const uint8_t* src_row = copy.src + slice * copy.src_pitch * copy.src_height;
							 for (uint64_t row = 0; row < copy.height; ++row)
							 {
								 Loop::Copy(loop.ctx, dst_row, src_row, static_cast<std::size_t>(copy.width));
								 dst_row += copy.dst_pitch;
								 src_row += copy.src_pitch;
							 }
						 }
					 });
}

enum class Direction : std::uint8_t
{
	Unknown,
	ToStaging,
	FromStaging
};

struct HostCopyLoops
{
	CopyLoop global;
	std::optional<CopyLoop> h2d;
	std::optional<CopyLoop> d2h;
};

HostCopyLoops SelectCopyLoops(const HookConfig& config);

CopyLoop CopyLoopFor(Direction direction);

void CopySerial(Direction dir, void* dst, const void* src, std::size_t bytes);

void Copy2D(Direction dir, const Copy2DView& copy);

} // namespace cmt
