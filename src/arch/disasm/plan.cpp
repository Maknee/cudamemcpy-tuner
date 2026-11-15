#include "arch/disasm/plan.h"
#include "cuda/versions.h"
#include "patch_instructions.h"

#include <cerrno>
#include <cstring>
#include <limits>
#include <link.h>
#include <sys/mman.h>
#include <unistd.h>

#include "common/log.h"
#include "arch/disasm/aarch64/relocate.h"
#include "arch/disasm/x86_64/relocate.h"

namespace cmt {
namespace patch {

InlineThresholdPlan PlanInlineThresholds(Arch arch, const DriverVersion& row, const uint8_t* current,
										 std::size_t available, uint64_t paged_limit, uint64_t pinned_limit)
{
	InlineThresholdPlan plan;
	const auto start = row.inline_paged_check.offset;
	if (arch != Arch::X86_64 || start == 0)
	{
		plan.why = "no scanned x86-64 inline threshold gate";
		return plan;
	}
	if (start > std::numeric_limits<uint32_t>::max() - kInlineGateBytes)
	{
		plan.why = "inline threshold gate offset overflows";
		return plan;
	}
	if (row.inline_paged_value.offset != start + 7 || row.inline_pinned_value.offset != start + 13 ||
		row.inline_pinned_check.offset != start + 24)
	{
		plan.why = "inline value/check offsets do not describe the scanned layout";
		return plan;
	}
	if (!current || available < kInlineGateBytes ||
		std::memcmp(current, row.inline_paged_check.original, sizeof(row.inline_paged_check.original)) != 0 ||
		std::memcmp(current + 7, row.inline_paged_value.original, sizeof(row.inline_paged_value.original)) !=
			0 ||
		std::memcmp(current + 13, row.inline_pinned_value.original,
					sizeof(row.inline_pinned_value.original)) != 0 ||
		std::memcmp(current + 24, row.inline_pinned_check.original,
					sizeof(row.inline_pinned_check.original)) != 0)
	{
		plan.why = "current inline gate differs from the complete scanned bytes";
		return plan;
	}
	if (paged_limit > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
	{
		plan.why = "pageable CMP sign-extends imm32; limit must be <= INT32_MAX";
		return plan;
	}
	if (current[0] != 0x48 || current[1] != 0x81 || current[2] != 0xbb || current[11] != 0x76 ||
		current[13] != 0xbf || current[18] != 0xff || (current[19] != 0x91 && current[19] != 0x92) ||
		current[24] != 0x48 || current[25] != 0x3b || current[26] != 0x83 || current[31] != 0x73 ||
		std::memcmp(current + 3, current + 27, 4) != 0)
	{
		plan.why = "unsupported inline threshold gate encoding";
		return plan;
	}
	static_assert(sizeof(instructions::kX86InlineLimit) == sizeof(plan.pinned_fetch));
	std::memcpy(plan.pinned_fetch, instructions::kX86InlineLimit, sizeof(plan.pinned_fetch));
	for (unsigned i = 0; i < 4; ++i)
	{
		plan.paged_limit[i] = static_cast<uint8_t>(paged_limit >> (i * 8));
	}
	for (unsigned i = 0; i < 8; ++i)
	{
		plan.pinned_fetch[2 + i] = static_cast<uint8_t>(pinned_limit >> (i * 8));
	}
	plan.paged_limit_va = row.inline_paged_value.offset;
	plan.pinned_fetch_va = row.inline_pinned_value.offset;
	plan.why = "";
	plan.ok = true;
	return plan;
}

EntryPlan PlanFullReplace(Arch arch, uint64_t site, uint64_t target)
{
	return arch == Arch::X86_64 ? x86::PlanFullReplace(site, target) : a64::PlanFullReplace(site, target);
}

WrapPlan PlanWrap(Arch arch, const uint8_t* code, std::size_t avail, uint64_t site, uint64_t tramp,
				  uint64_t target)
{
	return arch == Arch::X86_64 ? x86::PlanWrap(code, avail, site, tramp, target)
								: a64::PlanWrap(code, avail, site, tramp, target);
}

namespace {

constexpr std::size_t kTrampolinePageBytes =
	4096;

uintptr_t PageOf(const void* addr)
{
	return reinterpret_cast<uintptr_t>(addr) & ~static_cast<uintptr_t>(getpagesize() - 1);
}

int Protect(void* addr, std::size_t len, int prot)
{
	const uintptr_t page = PageOf(addr);
	const std::size_t total = (reinterpret_cast<uintptr_t>(addr) - page) + len;
	return mprotect(reinterpret_cast<void*>(page), total, prot);
}

void FlushInstructionCache(void* begin, std::size_t len)
{
	char* b = static_cast<char*>(begin);
	__builtin___clear_cache(b, b + len);
}

cmt::Result<void> WriteSite(void* site, const uint8_t* bytes, std::size_t len)
{
	uint8_t original[64];
	if (len > sizeof(original))
	{
		return cmt::Failure(cmt::Error::Kind::Usage, "patch longer than the site buffer");
	}
	if (Protect(site, len, PROT_READ | PROT_WRITE | PROT_EXEC) != 0)
	{
		const int code = errno;
		return cmt::Failure(cmt::Error::Kind::Os, "cannot make patch site writable", code);
	}
	std::memcpy(original, site, len);
	std::memcpy(site, bytes, len);
	FlushInstructionCache(site, len);
	if (Protect(site, len, PROT_READ | PROT_EXEC) == 0)
	{
		return {};
	}
	const int code = errno;
	std::memcpy(site, original, len);
	FlushInstructionCache(site, len);
	if (Protect(site, len, PROT_READ | PROT_EXEC) != 0)
	{
		LogLine("error: patch site {} stays writable and executable (original code restored)", site);
	}
	return cmt::Failure(cmt::Error::Kind::Os, "cannot restore R-X protection; patch reverted", code);
}

constexpr uint32_t kPtGnuProperty = 0x6474e553;

struct NoteSearch
{
	uintptr_t addr;
	bool found = false;
	a64::NoteVerdict verdict = a64::NoteVerdict::NoBti;
};

bool InsideLoad(const dl_phdr_info* info, ElfW(Addr) vaddr, std::size_t size)
{
	for (ElfW(Half) i = 0; i < info->dlpi_phnum; i++)
	{
		const ElfW(Phdr)& ph = info->dlpi_phdr[i];
		if (ph.p_type == PT_LOAD && vaddr >= ph.p_vaddr && size <= ph.p_filesz &&
			vaddr - ph.p_vaddr <= ph.p_filesz - size)
		{
			return true;
		}
	}
	return false;
}

int FindNotes(dl_phdr_info* info, std::size_t /*size*/, void* data)
{
	auto* s = static_cast<NoteSearch*>(data);
	bool contains = false;
	for (ElfW(Half) i = 0; i < info->dlpi_phnum && !contains; i++)
	{
		const ElfW(Phdr)& ph = info->dlpi_phdr[i];
		const uintptr_t lo = info->dlpi_addr + ph.p_vaddr;
		contains = ph.p_type == PT_LOAD && s->addr >= lo && s->addr - lo < ph.p_memsz;
	}
	if (!contains)
	{
		return 0;
	}
	s->found = true;
	for (ElfW(Half) i = 0; i < info->dlpi_phnum; i++)
	{
		const ElfW(Phdr)& ph = info->dlpi_phdr[i];
		if (ph.p_type != PT_NOTE && ph.p_type != kPtGnuProperty)
		{
			continue;
		}
		if (!InsideLoad(info, ph.p_vaddr, ph.p_filesz))
		{
			s->verdict = a64::NoteVerdict::Malformed;
			return 1;
		}
		const auto* notes = reinterpret_cast<const uint8_t*>(info->dlpi_addr + ph.p_vaddr);
		const a64::NoteVerdict v = a64::BtiFromNotes(notes, ph.p_filesz, ph.p_align < 4 ? 4 : ph.p_align);
		if (v != a64::NoteVerdict::NoBti)
		{
			s->verdict = v;
			return 1;
		}
	}
	return 1;
}

} // namespace

a64::NoteVerdict LoadedObjectBti(uintptr_t addr)
{
	NoteSearch s{addr};
	dl_iterate_phdr(&FindNotes, &s);
	return s.found ? s.verdict : a64::NoteVerdict::Malformed;
}

bool IsLibraryPatchable(uintptr_t base, const char** why)
{
	if (kNativeArch != Arch::Aarch64)
	{
		*why = "";
		return true;
	}
	switch (LoadedObjectBti(base))
	{
	case a64::NoteVerdict::NoBti:
		*why = "";
		return true;
	case a64::NoteVerdict::Bti:
		*why = "its GNU property note asks for BTI-guarded pages, which the aarch64 patcher does not "
			   "support";
		return false;
	case a64::NoteVerdict::Malformed:
		*why = "its program headers or notes could not be read";
		return false;
	}
	*why = "unknown";
	return false;
}

bool VerifyBytes(uintptr_t va, const uint8_t* expected, std::size_t len)
{
	return std::memcmp(reinterpret_cast<const void*>(va), expected, len) == 0;
}

cmt::Result<void> InstallFullReplacement(uintptr_t va, void* new_fn, const char* what)
{
	const EntryPlan plan = PlanFullReplace(kNativeArch, va, reinterpret_cast<uintptr_t>(new_fn));
	if (!plan.ok)
	{
		LogDebug("Cannot plan full replacement for {}: {}", what, plan.why);
		return cmt::Failure(cmt::Error::Kind::Unsupported, "Cannot plan full function replacement");
	}
	void* site = reinterpret_cast<void*>(va);
	CMT_RETURN_IF_ERROR(WriteSite(site, plan.bytes, plan.len));
	LogDebug("patched {} at {:x} -> {}", what, static_cast<unsigned long>(va), fmt::ptr(new_fn));
	return {};
}

cmt::Result<void> InstallWrapper(uintptr_t va, void* new_fn, void** orig_out, const char* what)
{
	void* page =
		mmap(nullptr, kTrampolinePageBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (page == MAP_FAILED)
	{
		const int code = errno;
		return cmt::Failure(cmt::Error::Kind::Os, "cannot allocate trampoline page", code);
	}
	const WrapPlan plan = PlanWrap(kNativeArch, reinterpret_cast<const uint8_t*>(va), 2 * kVerifiedBytes, va,
								   reinterpret_cast<uintptr_t>(page), reinterpret_cast<uintptr_t>(new_fn));
	if (!plan.ok)
	{
		munmap(page, kTrampolinePageBytes);
		LogDebug("Cannot plan wrapper for {}: {}", what, plan.why);
		return cmt::Failure(cmt::Error::Kind::Unsupported, "Cannot relocate function prologue into wrapper");
	}
	std::memcpy(page, plan.tramp, plan.tramp_len);
	if (mprotect(page, kTrampolinePageBytes, PROT_READ | PROT_EXEC) != 0)
	{
		const int code = errno;
		munmap(page, kTrampolinePageBytes);
		return cmt::Failure(cmt::Error::Kind::Os, "cannot make trampoline executable", code);
	}
	FlushInstructionCache(page, plan.tramp_len);
	void* site = reinterpret_cast<void*>(va);
	*orig_out = page;
	CMT_RETURN_IF_ERROR(WriteSite(site, plan.entry, plan.entry_len));
	LogDebug("wrapped {} at {:x} -> {} (original via trampoline {}, {} bytes moved, {} re-pointed)", what,
			 static_cast<unsigned long>(va), fmt::ptr(new_fn), fmt::ptr(page), plan.stolen, plan.rel_count);
	return {};
}

} // namespace patch
} // namespace cmt
