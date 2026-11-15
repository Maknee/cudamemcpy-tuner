#include "arch/disasm/aarch64/relocate.h"
#include "patch_instructions.h"

namespace cmt {
namespace patch {
namespace a64 {
namespace {

uint32_t ReadU32(const uint8_t* p)
{
	uint32_t v;
	std::memcpy(&v, p, 4);
	return v;
}

std::size_t Pad(std::size_t n, std::size_t a)
{
	return (n + a - 1) & ~(a - 1);
}

bool IsScratch(unsigned reg)
{
	return reg == ARM64_REG_X16 || reg == ARM64_REG_X17 || reg == ARM64_REG_W16 || reg == ARM64_REG_W17;
}

enum class Kind : std::uint8_t
{
	Refused,
	Plain,
	Load
};

Kind KindOf(unsigned id)
{
	switch (id)
	{
	case ARM64_INS_STP:
	case ARM64_INS_STR:
	case ARM64_INS_STRB:
	case ARM64_INS_STRH:
	case ARM64_INS_STUR:
	case ARM64_INS_STURB:
	case ARM64_INS_STURH:
	case ARM64_INS_ADD:
	case ARM64_INS_SUB:
	case ARM64_INS_MOV:
	case ARM64_INS_MOVZ:
	case ARM64_INS_MOVK:
	case ARM64_INS_MOVN:
	case ARM64_INS_ORR:
	case ARM64_INS_MRS:
	case ARM64_INS_NOP:
	case ARM64_INS_HINT:
	case ARM64_INS_BTI:
	case ARM64_INS_PACIASP:
	case ARM64_INS_PACIBSP:
		return Kind::Plain;
	case ARM64_INS_LDP:
	case ARM64_INS_LDR:
	case ARM64_INS_LDRB:
	case ARM64_INS_LDRH:
	case ARM64_INS_LDRSW:
	case ARM64_INS_LDUR:
		return Kind::Load;
	default:
		return Kind::Refused;
	}
}

const char* RefusalOf(disasm::Session& cs, const cs_insn* insn)
{
	if (insn == nullptr)
	{
		return "undecodable or truncated instruction";
	}
	if (cs.InGroup(insn, CS_GRP_JUMP) || cs.InGroup(insn, CS_GRP_CALL) || cs.InGroup(insn, CS_GRP_RET) ||
		cs.InGroup(insn, CS_GRP_INT) || cs.InGroup(insn, CS_GRP_IRET) ||
		cs.InGroup(insn, CS_GRP_BRANCH_RELATIVE) || cs.InGroup(insn, CS_GRP_PRIVILEGE))
	{
		return "branch, call, return or exception instruction in the window";
	}
	if (insn->id == ARM64_INS_ADR || insn->id == ARM64_INS_ADRP)
	{
		return "PC-relative address (adr/adrp) in the window";
	}
	const Kind kind = KindOf(insn->id);
	if (kind == Kind::Refused)
	{
		return "instruction outside the relocatable prologue list";
	}
	const cs_arm64& a = insn->detail->arm64;
	bool has_mem = false;
	for (uint8_t i = 0; i < a.op_count; i++)
	{
		const cs_arm64_op& op = a.operands[i];
		if (op.type == ARM64_OP_REG && IsScratch(op.reg))
		{
			return "uses x16/x17, which the branch back clobbers";
		}
		if (op.type == ARM64_OP_MEM)
		{
			has_mem = true;
			if (IsScratch(op.mem.base) || IsScratch(op.mem.index))
			{
				return "uses x16/x17, which the branch back clobbers";
			}
			if (op.mem.base == ARM64_REG_INVALID)
			{
				return "memory operand without a base register";
			}
		}
	}
	if (kind == Kind::Load && !has_mem)
	{
		return "PC-relative load (literal) in the window";
	}
	return nullptr;
}

} // namespace

NoteVerdict BtiFromNotes(const uint8_t* notes, std::size_t size, std::size_t align)
{
	if (align != 4 && align != 8)
	{
		return NoteVerdict::Malformed;
	}
	std::size_t off = 0;
	bool bti = false;
	while (off < size)
	{
		if (size - off < 12)
		{
			return NoteVerdict::Malformed;
		}
		const uint32_t namesz = ReadU32(notes + off);
		const uint32_t descsz = ReadU32(notes + off + 4);
		const uint32_t type = ReadU32(notes + off + 8);
		const std::size_t name_at = off + 12;
		const std::size_t desc_at =
			off + Pad(12 + std::size_t{namesz}, align);
		if (namesz > size || descsz > size || desc_at > size || descsz > size - desc_at)
		{
			return NoteVerdict::Malformed;
		}
		const bool gnu = namesz == 4 && std::memcmp(notes + name_at, "GNU", 4) == 0;
		if (gnu && type == kNtGnuPropertyType0)
		{
			std::size_t p = desc_at;
			const std::size_t end = desc_at + descsz;
			while (p < end)
			{
				if (end - p < 8)
				{
					return NoteVerdict::Malformed;
				}
				const uint32_t pr_type = ReadU32(notes + p);
				const uint32_t pr_datasz = ReadU32(notes + p + 4);
				if (pr_datasz > end - p - 8)
				{
					return NoteVerdict::Malformed;
				}
				if (pr_type == kGnuPropertyAarch64Feature1And)
				{
					if (pr_datasz < 4)
					{
						return NoteVerdict::Malformed;
					}
					bti = bti || (ReadU32(notes + p + 8) & kFeature1Bti) != 0;
				}
				p += 8 + Pad(pr_datasz, 8);
			}
		}
		off = off + Pad(desc_at - off + descsz, align);
	}
	return bti ? NoteVerdict::Bti : NoteVerdict::NoBti;
}

void EncodeAbsoluteBranch(uint8_t* out, uint64_t target)
{
	static_assert(sizeof(instructions::kA64Branch) == 8);
	std::memcpy(out, instructions::kA64Branch, sizeof(instructions::kA64Branch));
	std::memcpy(out + 8, &target, 8);
}

const char* Refusal(const uint8_t* code, std::size_t avail, uint64_t addr)
{
	disasm::Session cs(Arch::Aarch64);
	if (!cs.IsValid())
	{
		return "Capstone unavailable (arena busy or exhausted)";
	}
	return RefusalOf(cs, cs.Next(&code, &avail, &addr));
}

EntryPlan PlanFullReplace(uint64_t site, uint64_t target)
{
	EntryPlan e;
	if (site % 8 != 0)
	{
		e.why = "site not 8-byte aligned: the branch's literal would be misaligned";
		return e;
	}
	EncodeAbsoluteBranch(e.bytes, target);
	e.len = kAbsBranchBytes;
	e.ok = true;
	e.why = "";
	return e;
}

WrapPlan PlanWrap(const uint8_t* code, std::size_t avail, uint64_t site, uint64_t tramp, uint64_t target)
{
	WrapPlan w;
	if (site % 8 != 0 || tramp % 8 != 0)
	{
		w.why = "site or trampoline not 8-byte aligned: a branch literal would be misaligned";
		return w;
	}
	if (avail < kAbsBranchBytes)
	{
		w.why = "fewer than 16 prologue bytes";
		return w;
	}
	{
		disasm::Session cs(Arch::Aarch64);
		if (!cs.IsValid())
		{
			w.why = "Capstone unavailable (arena busy or exhausted)";
			return w;
		}
		const uint8_t* p = code;
		std::size_t left = kAbsBranchBytes;
		uint64_t pc = site;
		for (std::size_t off = 0; off < kAbsBranchBytes; off += 4)
		{
			const char* why = RefusalOf(cs, cs.Next(&p, &left, &pc));
			if (why != nullptr)
			{
				w.why = why;
				return w;
			}
		}
	}
	std::memcpy(w.tramp, code, kAbsBranchBytes);
	EncodeAbsoluteBranch(w.tramp + kAbsBranchBytes, site + kAbsBranchBytes);
	w.tramp_len = 2 * kAbsBranchBytes;
	EncodeAbsoluteBranch(w.entry, target);
	w.entry_len = kAbsBranchBytes;
	w.stolen = kAbsBranchBytes;
	w.ok = true;
	w.why = "";
	return w;
}

} // namespace a64
} // namespace patch
} // namespace cmt
