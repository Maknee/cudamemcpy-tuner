#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "arch/disasm/disasm.h"
#include "arch/disasm/plan.h"

namespace cmt {
namespace patch {
namespace x86 {

constexpr std::size_t kAbsJumpBytes = 14;

void EncodeAbsoluteJump(uint8_t* out, uint64_t target);

struct Steal
{
	bool ok = false;
	const char* why = "";
	std::size_t stolen = 0;
	uint8_t rel_at[kMaxRelocations] = {};
	unsigned rel_count = 0;
};

Steal DecodeStolenInstructions(const uint8_t* code, std::size_t avail, uint64_t addr, std::size_t min_steal,
							   std::size_t max_steal);

EntryPlan PlanFullReplace(uint64_t site, uint64_t target);
WrapPlan PlanWrap(const uint8_t* code, std::size_t avail, uint64_t site, uint64_t tramp, uint64_t target);

} // namespace x86
} // namespace patch
} // namespace cmt
