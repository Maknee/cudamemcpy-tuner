#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "arch/disasm/disasm.h"
#include "arch/disasm/plan.h"

namespace cmt {
namespace patch {
namespace a64 {

constexpr uint32_t kNtGnuPropertyType0 = 5;
constexpr uint32_t kGnuPropertyAarch64Feature1And = 0xc0000000;
constexpr uint32_t kFeature1Bti = 1u << 0;
constexpr uint32_t kFeature1Pac = 1u << 1;

enum class NoteVerdict : uint8_t
{
	NoBti,
	Bti,
	Malformed,
};

NoteVerdict BtiFromNotes(const uint8_t* notes, std::size_t size, std::size_t align);

constexpr std::size_t kAbsBranchBytes = 16;
constexpr uint32_t kLdrX16Literal8 = 0x58000050;
constexpr uint32_t kBrX16 = 0xd61f0200;

void EncodeAbsoluteBranch(uint8_t* out, uint64_t target);

const char* Refusal(const uint8_t* code, std::size_t avail, uint64_t addr);

EntryPlan PlanFullReplace(uint64_t site, uint64_t target);
WrapPlan PlanWrap(const uint8_t* code, std::size_t avail, uint64_t site, uint64_t tramp, uint64_t target);

} // namespace a64
} // namespace patch
} // namespace cmt
