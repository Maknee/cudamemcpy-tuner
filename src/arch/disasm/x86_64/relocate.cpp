#include "arch/disasm/x86_64/relocate.h"
#include "patch_instructions.h"

namespace cmt {
namespace patch {
namespace x86 {

void EncodeAbsoluteJump(uint8_t* out, uint64_t target)
{
	static_assert(sizeof(instructions::kX86Jump) == 6);
	std::memcpy(out, instructions::kX86Jump, sizeof(instructions::kX86Jump));
	std::memcpy(out + 6, &target, 8);
}

Steal DecodeStolenInstructions(const uint8_t* code, std::size_t avail, uint64_t addr, std::size_t min_steal,
							   std::size_t max_steal)
{
	Steal s;
	disasm::Session cs(Arch::X86_64);
	if (!cs.IsValid())
	{
		s.why = "Capstone unavailable (arena busy or exhausted)";
		return s;
	}
	const uint8_t* p = code;
	std::size_t left = avail;
	uint64_t pc = addr;
	while (s.stolen < min_steal)
	{
		const cs_insn* insn = cs.Next(&p, &left, &pc);
		if (insn == nullptr)
		{
			s.why = "undecodable or truncated instruction";
			return s;
		}
		if (cs.InGroup(insn, CS_GRP_JUMP) || cs.InGroup(insn, CS_GRP_CALL) || cs.InGroup(insn, CS_GRP_RET) ||
			cs.InGroup(insn, CS_GRP_INT) || cs.InGroup(insn, CS_GRP_IRET) ||
			cs.InGroup(insn, CS_GRP_BRANCH_RELATIVE) || cs.InGroup(insn, CS_GRP_PRIVILEGE))
		{
			s.why = "branch, call, return, interrupt or privileged instruction in the window";
			return s;
		}
		const cs_x86& x = insn->detail->x86;
		for (uint8_t i = 0; i < x.op_count; i++)
		{
			const cs_x86_op& op = x.operands[i];
			const bool rip_mem =
				op.type == X86_OP_MEM && (op.mem.base == X86_REG_RIP || op.mem.index == X86_REG_RIP);
			const bool rip_reg = op.type == X86_OP_REG && op.reg == X86_REG_RIP;
			if (rip_reg || (rip_mem && op.mem.index == X86_REG_RIP))
			{
				s.why = "RIP used other than as a memory base";
				return s;
			}
			if (!rip_mem)
			{
				continue;
			}
			if (x.encoding.disp_size != 4 || x.encoding.disp_offset == 0 ||
				x.encoding.disp_offset + 4u > insn->size || s.rel_count == kMaxRelocations)
			{
				s.why = "RIP-relative operand without a locatable disp32";
				return s;
			}
			s.rel_at[s.rel_count++] = static_cast<uint8_t>(s.stolen + x.encoding.disp_offset);
		}
		s.stolen += insn->size;
		if (s.stolen > max_steal)
		{
			s.why = "whole instructions would exceed the verified prologue";
			return s;
		}
	}
	s.ok = true;
	return s;
}

EntryPlan PlanFullReplace(uint64_t /*site*/, uint64_t target)
{
	EntryPlan e;
	EncodeAbsoluteJump(e.bytes, target);
	e.len = kAbsJumpBytes;
	e.ok = true;
	e.why = "";
	return e;
}

WrapPlan PlanWrap(const uint8_t* code, std::size_t avail, uint64_t site, uint64_t tramp, uint64_t target)
{
	WrapPlan w;
	const Steal s = DecodeStolenInstructions(code, avail, site, kAbsJumpBytes, kVerifiedBytes);
	if (!s.ok)
	{
		w.why = s.why;
		return w;
	}
	if (s.stolen + kAbsJumpBytes > kMaxTrampolineBytes)
	{
		w.why = "trampoline too long";
		return w;
	}
	std::memcpy(w.tramp, code, s.stolen);
	for (unsigned i = 0; i < s.rel_count; i++)
	{
		const std::size_t at = s.rel_at[i];
		int32_t old_disp;
		std::memcpy(&old_disp, code + at, 4);
		const int64_t delta = static_cast<int64_t>(tramp - site);
		const int64_t new_disp = static_cast<int64_t>(old_disp) - delta;
		if (new_disp != static_cast<int32_t>(new_disp))
		{
			w.why = "RIP-relative target out of rel32 range of the trampoline";
			return w;
		}
		const int32_t d = static_cast<int32_t>(new_disp);
		std::memcpy(w.tramp + at, &d, 4);
		w.rel_at[i] = s.rel_at[i];
	}
	w.rel_count = s.rel_count;
	EncodeAbsoluteJump(w.tramp + s.stolen, site + s.stolen);
	w.tramp_len = s.stolen + kAbsJumpBytes;

	EncodeAbsoluteJump(w.entry, target);
	for (std::size_t i = kAbsJumpBytes; i < s.stolen; i++)
	{
		w.entry[i] = instructions::kX86Trap[0];
	}
	w.entry_len = s.stolen;
	w.stolen = s.stolen;
	w.ok = true;
	w.why = "";
	return w;
}

} // namespace x86
} // namespace patch
} // namespace cmt
