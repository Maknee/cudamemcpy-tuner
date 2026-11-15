#pragma once

#include <cstddef>
#include <cstdint>

#include "arch/copy.h"
#include "common/error.h"

namespace cmt {
namespace patch {

struct DriverVersion;

struct InlineThresholdPlan
{
	bool ok = false;
	const char* why = "not planned";
	uint32_t paged_limit_va = 0;
	uint8_t paged_limit[4] = {};
	uint32_t pinned_fetch_va = 0;
	uint8_t pinned_fetch[11] = {};
};

InlineThresholdPlan PlanInlineThresholds(Arch arch, const DriverVersion& row, const uint8_t* current,
										 std::size_t available, uint64_t paged_limit, uint64_t pinned_limit);

constexpr std::size_t kVerifiedBytes = 16;

constexpr std::size_t kMaxTrampolineBytes = 64;
constexpr std::size_t kMaxRelocations = 4;

struct EntryPlan
{
	bool ok = false;
	const char* why = "not planned";
	uint8_t bytes[kVerifiedBytes] = {};
	std::size_t len = 0;
};

struct WrapPlan
{
	bool ok = false;
	const char* why = "not planned";
	std::size_t stolen = 0;
	uint8_t entry[kVerifiedBytes] = {};
	std::size_t entry_len = 0;
	uint8_t tramp[kMaxTrampolineBytes] = {};
	std::size_t tramp_len = 0;
	uint8_t rel_at[kMaxRelocations] = {};
	unsigned rel_count = 0;
};

EntryPlan PlanFullReplace(Arch arch, uint64_t site, uint64_t target);

WrapPlan PlanWrap(Arch arch, const uint8_t* code, std::size_t avail, uint64_t site, uint64_t tramp,
				  uint64_t target);

namespace a64 {
enum class NoteVerdict : uint8_t;
} // namespace a64

bool VerifyBytes(uintptr_t va, const uint8_t* expected, std::size_t len);

bool IsLibraryPatchable(uintptr_t base, const char** why);

a64::NoteVerdict LoadedObjectBti(uintptr_t addr);

[[nodiscard]] cmt::Result<void> InstallFullReplacement(uintptr_t va, void* new_fn, const char* what);
[[nodiscard]] cmt::Result<void> InstallWrapper(uintptr_t va, void* new_fn, void** orig_out, const char* what);

} // namespace patch
} // namespace cmt
