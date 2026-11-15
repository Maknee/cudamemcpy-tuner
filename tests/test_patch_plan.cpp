// test_patch_plan.cpp - the patch planners, both tables and the row selection, on any host.
//
// Planning is pure (arch/disasm/plan.h), so this runs the aarch64 planner on the aarch64 table's real
// prologues on an x86-64 machine, and the x86-64 planner on an aarch64 one. What it pins:
//   - x86-64: every row's wrapped site (buffer_init) moves exactly 14 bytes with no relocation, the values
//     x86lde (the decoder Capstone replaced) computed on the same bytes of the real driver files.
//     RIP-relative operands are re-pointed so they keep their target, and refused when the trampoline is out
//     of rel32 range.
//   - x86-64 refusals: a branch or call in the window, bytes Capstone cannot decode, truncated bytes, a window
//     longer than the 16 verified bytes.
//   - aarch64: the branch is "ldr x16, #8; br x16; .quad target" (decoded back with Capstone), every row's
//     sites plan, internal_memcpy is absent, the descriptor equals x86-64's for the same version; every
//     PC-relative or scratch-register instruction is refused, the prologue instructions compilers emit are
//     accepted; misaligned sites are refused.
//   - aarch64 BTI preflight: the GNU property note parser on synthetic notes, and on the objects of this
//     process.
//   - rows: the aarch64 table is unverified, so select_row withholds its rows unless allowed.
//   - Capstone's arena: reset after every plan, peak well below its size, one session at a time.
#include <capstone/capstone.h>

#include <array>
#include <cstring>
#include <memory>
#include <sys/mman.h>
#include <utility>
#include <vector>

#include "check.h"
#include "common/memory.h"
#include "common/cpu.h"
#include "arch/disasm/aarch64/relocate.h"
#include "arch/disasm/disasm.h"
#include "hook.h"
#include "arch/disasm/plan.h"
#include "cuda/versions.h"
#include "arch/disasm/x86_64/relocate.h"

namespace {

using cmt::Arch;
using cmt::patch::DriverVersion;
using cmt::patch::VersionTable;
using cmt::patch::WrapPlan;

uint64_t rd64(const uint8_t* p)
{
	uint64_t v;
	std::memcpy(&v, p, 8);
	return v;
}

uint32_t rd32(const uint8_t* p)
{
	uint32_t v;
	std::memcpy(&v, p, 4);
	return v;
}

std::vector<uint8_t> words(std::initializer_list<uint32_t> insns)
{
	std::vector<uint8_t> out(std::size(insns) * 4);
	std::size_t i = 0;
	for (uint32_t w : insns)
	{
		std::memcpy(std::data(out) + 4 * i++, &w, 4);
	}
	return out;
}

// ---------------------------------------------------------------------------------------------------------------
// x86-64
// ---------------------------------------------------------------------------------------------------------------

void test_assembled_templates()
{
	// Instructions come from Keystone, while the full-width target is data.
	// Pin both encodings independently, including addresses an assembler might
	// otherwise shorten, sign-extend or try to encode as a branch immediate.
	for (const uint64_t target : {uint64_t{0}, uint64_t{0x123456789abcdef0}, UINT64_MAX})
	{
		uint8_t out[20];
		std::memset(out, 0xa5, sizeof(out));
		cmt::patch::x86::EncodeAbsoluteJump(out, target);
		const uint8_t prefix[] = {0xff, 0x25, 0, 0, 0, 0};
		CHECK(std::memcmp(out, prefix, sizeof(prefix)) == 0 && rd64(out + 6) == target);
		CHECK(out[14] == 0xa5 && out[19] == 0xa5);
		std::memset(out, 0xa5, sizeof(out));
		cmt::patch::a64::EncodeAbsoluteBranch(out, target);
		CHECK(rd32(out) == cmt::patch::a64::kLdrX16Literal8);
		CHECK(rd32(out + 4) == cmt::patch::a64::kBrX16 && rd64(out + 8) == target);
		CHECK(out[16] == 0xa5 && out[19] == 0xa5);
	}
}

void test_x86_table()
{
	const VersionTable& t = cmt::patch::VersionTableFor(Arch::X86_64);
	CHECK(t.arch == Arch::X86_64 && t.count > 0 && t.enabled_by_default);
	for (std::size_t i = 0; i < t.count; i++)
	{
		const DriverVersion& vi = t.rows[i];
		const uint64_t site = 0x7f0000000000ull + vi.buffer_init_va;
		const WrapPlan w = cmt::patch::PlanWrap(Arch::X86_64, vi.buffer_init_prologue, 16, site,
												site + 0x100000, 0x7f1000000000ull);
		// Pinned: the stolen length and relocations x86lde computed on the driver bytes (equivalence run).
		CHECKF(w.ok && w.stolen == 14 && w.rel_count == 0, "%s buffer_init: %s, stolen %zu", vi.version,
			   w.why, w.stolen);
		CHECK(w.entry_len == 14 && w.entry[0] == 0xff && w.entry[1] == 0x25 &&
			  rd64(w.entry + 6) == 0x7f1000000000ull);
		CHECK(std::memcmp(w.tramp, vi.buffer_init_prologue, 14) == 0 && rd64(w.tramp + 14 + 6) == site + 14);
		// The engine's first whole instructions end at byte 17, past the verified 16: never wrapped (it is
		// only fully replaced, which needs no decoding).
		const WrapPlan h = cmt::patch::PlanWrap(Arch::X86_64, vi.h2h_prologue, 16, site, site + 0x1000, 0);
		CHECKF(!h.ok, "%s h2h must not be wrappable within 16 bytes", vi.version);
		CHECK(cmt::patch::FindVersion(t, vi.version) == &vi);
	}
	CHECK(cmt::patch::FindVersion(t, "1") == nullptr && cmt::patch::FindVersion(t, "") == nullptr);
}

// push rbp; mov rbp,rsp; mov rax,[rip+d32]; add eax,edi; push r15 | ...
void test_x86_rip_relative()
{
	const uint8_t code[] = {0x55, 0x48, 0x89, 0xe5, 0x48, 0x8b, 0x05, 0x10, 0x20, 0x00,
							0x00, 0x01, 0xf8, 0x41, 0x57, 0x41, 0x5f, 0x5d, 0xc3};
	const uint64_t site = 0x500000000ull;
	const uint64_t target_of_load = site + 11 + 0x2010; // end of the mov + disp
	const uint64_t tramp = site - 0x40000000ull;		// 1 GiB below: within rel32
	const WrapPlan w = cmt::patch::PlanWrap(Arch::X86_64, code, sizeof(code), site, tramp, 0x1234);
	CHECKF(w.ok && w.stolen == 15 && w.rel_count == 1 && w.rel_at[0] == 7, "%s", w.why);
	int32_t disp;
	std::memcpy(&disp, w.tramp + 7, 4);
	CHECK(tramp + 11 + static_cast<int64_t>(disp) == target_of_load); // same target from the trampoline
	CHECK(w.entry[14] == 0xcc);										  // 15 stolen: one int3 pad byte
	CHECK(rd64(w.tramp + 15 + 6) == site + 15);						  // jump back after the moved bytes
	// The same function with its trampoline 3 GiB away: the disp32 cannot reach; refused.
	const WrapPlan far = cmt::patch::PlanWrap(Arch::X86_64, code, sizeof(code), site, site + (3ull << 30), 0);
	CHECK(!far.ok);
}

void test_x86_refusals()
{
	struct Case
	{
		const char* what;
		std::vector<uint8_t> bytes;
		std::size_t avail;
	};

	const Case cases[] = {
		{"jne rel8 in the window",
		 {0x55, 0x48, 0x89, 0xe5, 0x75, 0x10, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90},
		 16},
		{"call rel32 first",
		 {0xe8, 0, 0, 0, 0, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90},
		 16},
		{"ret in the window",
		 {0x55, 0xc3, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90},
		 16},
		{"jmp *rax in the window",
		 {0x55, 0xff, 0xe0, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90},
		 16},
		// 0x06 (push es) does not exist in 64-bit mode.
		{"undecodable opcode",
		 {0x55, 0x06, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90},
		 16},
		// mov rax,[rip+d32] cut after 5 of its 7 bytes.
		{"truncated instruction", {0x55, 0x48, 0x89, 0xe5, 0x48, 0x8b, 0x05, 0x10, 0x20}, 9},
		// 1 + 3 + 2 x 7: the third instruction ends at byte 18.
		{"window past 16 bytes",
		 {0x55, 0x48, 0x89, 0xe5, 0x48, 0x8b, 0x05, 1, 0, 0, 0, 0x48, 0x8b, 0x0d, 1, 0, 0, 0},
		 18},
	};
	for (const Case& c : cases)
	{
		const WrapPlan w =
			cmt::patch::PlanWrap(Arch::X86_64, std::data(c.bytes), c.avail, 0x400000, 0x401000, 0x1000);
		CHECKF(!w.ok, "x86 %s must be refused", c.what);
		std::printf("x86 refused (%s): %s\n", c.what, w.why);
	}
	const cmt::patch::EntryPlan e = cmt::patch::PlanFullReplace(Arch::X86_64, 0x400003, 0xdeadbeef00ull);
	CHECK(e.ok && e.len == 14 && rd64(e.bytes + 6) == 0xdeadbeef00ull);
}

// ---------------------------------------------------------------------------------------------------------------
// aarch64
// ---------------------------------------------------------------------------------------------------------------

// Decodes the planned branch with Capstone: ldr x16, <site + 8>; br x16; and the literal holds the target.
void check_a64_branch(const uint8_t* at, uint64_t addr, uint64_t target)
{
	csh h;
	CHECK(cs_open(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN, &h) == CS_ERR_OK);
	cs_option(h, CS_OPT_DETAIL, CS_OPT_ON);
	cs_insn* insn = nullptr;
	const std::size_t n = cs_disasm(h, at, 8, addr, 0, &insn);
	CHECK(n == 2);
	if (n == 2)
	{
		CHECK(insn[0].id == ARM64_INS_LDR && insn[0].detail->arm64.op_count == 2 &&
			  insn[0].detail->arm64.operands[0].reg == ARM64_REG_X16 &&
			  insn[0].detail->arm64.operands[1].type == ARM64_OP_IMM &&
			  static_cast<uint64_t>(insn[0].detail->arm64.operands[1].imm) == addr + 8);
		CHECK(insn[1].id == ARM64_INS_BR && insn[1].detail->arm64.operands[0].reg == ARM64_REG_X16);
	}
	cs_free(insn, n);
	cs_close(&h);
	CHECK(rd64(at + 8) == target);
}

void test_a64_table()
{
	const VersionTable& t = cmt::patch::VersionTableFor(Arch::Aarch64);
	const VersionTable& x = cmt::patch::VersionTableFor(Arch::X86_64);
	CHECK(t.arch == Arch::Aarch64 && t.count == 4 && !t.enabled_by_default);
	for (std::size_t i = 0; i < t.count; i++)
	{
		const DriverVersion& vi = t.rows[i];
		CHECKF(vi.internal_memcpy_va == 0, "%s: aarch64 has no internal_memcpy (memcpy@plt)", vi.version);
		CHECKF(vi.h2h_va % 8 == 0 && vi.buffer_init_va % 8 == 0, "%s: sites 8-aligned", vi.version);
		const uint64_t base = 0xffff80000000ull;
		for (const auto& site : {std::make_pair(vi.h2h_va, vi.h2h_prologue),
								 std::make_pair(vi.buffer_init_va, vi.buffer_init_prologue)})
		{
			const uint64_t va = base + site.first;
			const uint64_t tramp = 0xffff90000000ull;
			const WrapPlan w = cmt::patch::PlanWrap(Arch::Aarch64, site.second, 16, va, tramp, 0xaaaa0000ull);
			CHECKF(w.ok && w.stolen == 16 && w.entry_len == 16 && w.tramp_len == 32, "%s %x: %s", vi.version,
				   site.first, w.why);
			CHECK(std::memcmp(w.tramp, site.second, 16) == 0); // moved verbatim
			check_a64_branch(w.tramp + 16, tramp + 16, va + 16);
			check_a64_branch(w.entry, va, 0xaaaa0000ull);
		}
		// The descriptor layout is the x86-64 one for the same version (cmt-sigscan: sigscan_cross_<v>).
		const DriverVersion* xv = cmt::patch::FindVersion(x, vi.version);
		CHECKF(xv != nullptr, "%s: an x86-64 row exists", vi.version);
		if (xv != nullptr)
		{
			CHECKF(xv->op_stride == vi.op_stride && xv->op_kind == vi.op_kind &&
					   xv->op_some_pointer1 == vi.op_some_pointer1 &&
					   xv->op_some_pointer1_off == vi.op_some_pointer1_off &&
					   xv->op_some_pointer1_width == vi.op_some_pointer1_width &&
					   xv->op_some_pointer1_height == vi.op_some_pointer1_height &&
					   xv->op_some_pointer2 == vi.op_some_pointer2 &&
					   xv->op_some_pointer2_width == vi.op_some_pointer2_width &&
					   xv->op_some_pointer2_height == vi.op_some_pointer2_height &&
					   xv->op_some_pointer3 == vi.op_some_pointer3 &&
					   xv->op_some_pointer3_width == vi.op_some_pointer3_width &&
					   xv->op_some_pointer3_height == vi.op_some_pointer3_height &&
					   xv->op_unkx == vi.op_unkx && xv->op_unky == vi.op_unky && xv->op_unkz == vi.op_unkz &&
					   xv->copying_width_bytes == vi.copying_width_bytes &&
					   xv->copying_height_bytes == vi.copying_height_bytes &&
					   xv->copying_depth_bytes == vi.copying_depth_bytes &&
					   xv->op_some_pointer1_address == vi.op_some_pointer1_address &&
					   xv->op_kind_1 == vi.op_kind_1 && xv->op_kind_2 == vi.op_kind_2 &&
					   xv->op_kind_3 == vi.op_kind_3,
				   "%s: descriptor differs from x86-64's", vi.version);
		}
	}
}

void test_a64_classification()
{
	struct Case
	{
		const char* what;
		uint32_t insn; // encodings checked with the cross assembler (aarch64-conda-linux-gnu-as / objdump)
		bool movable;
	};

	const Case cases[] = {
		{"adr x0, .", 0x10000000, false},
		{"adrp x0, 0", 0x90000000, false},
		{"ldr x0, .+8 (literal)", 0x58000040, false},
		{"ldrsw x0, .+8 (literal)", 0x98000040, false},
		{"prfm pldl1keep, .+8 (literal)", 0xd8000040, false},
		{"b .+8", 0x14000002, false},
		{"bl .+8", 0x94000002, false},
		{"b.eq .+8", 0x54000040, false},
		{"cbz x0, .+8", 0xb4000040, false},
		{"cbnz x0, .+8", 0xb5000040, false},
		{"tbz w0, #0, .+8", 0x36000040, false},
		{"tbnz w0, #0, .+8", 0x37000040, false},
		{"br x0", 0xd61f0000, false},
		{"blr x0", 0xd63f0000, false},
		{"ret", 0xd65f03c0, false},
		{"svc #0", 0xd4000001, false},
		{"udf #0 (undefined)", 0x00000000, false},
		{"mov x16, x0 (scratch)", 0xaa0003f0, false},
		{"ldr x0, [x17] (scratch base)", 0xf9400220, false},
		{"add x0, x16, #1 (scratch source)", 0x91000600, false},
		{"stp x29, x30, [sp, #-16]!", 0xa9bf7bfd, true},
		{"mov x29, sp", 0x910003fd, true},
		{"sub sp, sp, #0x40", 0xd10103ff, true},
		{"str x19, [sp, #16]", 0xf9000bf3, true},
		{"ldr x0, [x1]", 0xf9400020, true},
		{"bti c", 0xd503245f, true},
		{"paciasp", 0xd503233f, true},
		{"nop", 0xd503201f, true},
		{"mov x0, #1", 0xd2800020, true},
		{"stp q0, q1, [sp, #32]", 0xad0107e0, true},
	};
	for (const Case& c : cases)
	{
		uint8_t b[4];
		std::memcpy(b, &c.insn, 4);
		const char* why = cmt::patch::a64::Refusal(b, 4, 0x400000);
		CHECKF((why == nullptr) == c.movable, "aarch64 %s: expected %s, got %s", c.what,
			   c.movable ? "movable" : "refused", why == nullptr ? "movable" : why);
		// The same instruction as the third of a wrapped prologue: the whole wrap follows the verdict.
		const std::vector<uint8_t> p = words({0xa9bf7bfd, 0x910003fd, c.insn, 0xd503201f});
		const WrapPlan w =
			cmt::patch::PlanWrap(Arch::Aarch64, std::data(p), std::size(p), 0x400000, 0x500000, 0x600000);
		CHECKF(w.ok == c.movable, "aarch64 wrap with %s", c.what);
	}
	uint8_t two[2] = {0x1f, 0x20};
	CHECK(cmt::patch::a64::Refusal(two, 2, 0) != nullptr); // truncated
	const std::vector<uint8_t> ok = words({0xa9bf7bfd, 0x910003fd, 0xd503201f, 0xd503201f});
	CHECK(!cmt::patch::PlanWrap(Arch::Aarch64, std::data(ok), 12, 0x400000, 0x500000, 0)
			   .ok); // fewer than 16 bytes
	CHECK(
		!cmt::patch::PlanWrap(Arch::Aarch64, std::data(ok), 16, 0x400004, 0x500000, 0).ok); // site misaligned
	CHECK(!cmt::patch::PlanWrap(Arch::Aarch64, std::data(ok), 16, 0x400000, 0x500004, 0)
			   .ok); // trampoline misaligned
	CHECK(!cmt::patch::PlanFullReplace(Arch::Aarch64, 0x400004, 0x1000).ok);
	const cmt::patch::EntryPlan e = cmt::patch::PlanFullReplace(Arch::Aarch64, 0x400008, 0x123456789abcull);
	CHECK(e.ok && e.len == 16 && rd32(e.bytes) == cmt::patch::a64::kLdrX16Literal8 &&
		  rd32(e.bytes + 4) == cmt::patch::a64::kBrX16);
	check_a64_branch(e.bytes, 0x400008, 0x123456789abcull);
}

// ---------------------------------------------------------------------------------------------------------------
// BTI preflight
// ---------------------------------------------------------------------------------------------------------------

std::vector<uint8_t> note(uint32_t type, const char* name, const std::vector<uint8_t>& desc,
						  std::size_t align)
{
	std::vector<uint8_t> out(12);
	const uint32_t namesz = static_cast<uint32_t>(std::strlen(name) + 1);
	const uint32_t descsz = static_cast<uint32_t>(std::size(desc));
	std::memcpy(std::data(out), &namesz, 4);
	std::memcpy(std::data(out) + 4, &descsz, 4);
	std::memcpy(std::data(out) + 8, &type, 4);
	out.insert(std::end(out), name, name + namesz);
	while (std::size(out) % align != 0)
	{
		out.push_back(0);
	}
	out.insert(std::end(out), std::begin(desc), std::end(desc));
	while (std::size(out) % align != 0)
	{
		out.push_back(0);
	}
	return out;
}

std::vector<uint8_t> property(uint32_t pr_type, uint32_t value)
{
	std::vector<uint8_t> d(16, 0);
	const uint32_t datasz = 4;
	std::memcpy(std::data(d), &pr_type, 4);
	std::memcpy(std::data(d) + 4, &datasz, 4);
	std::memcpy(std::data(d) + 8, &value, 4);
	return d;
}

void test_bti_notes()
{
	using cmt::patch::a64::BtiFromNotes;
	using cmt::patch::a64::NoteVerdict;
	namespace a = cmt::patch::a64;
	const uint32_t and_type = a::kGnuPropertyAarch64Feature1And;
	const auto bti = note(a::kNtGnuPropertyType0, "GNU", property(and_type, a::kFeature1Bti), 8);
	const auto pac = note(a::kNtGnuPropertyType0, "GNU", property(and_type, a::kFeature1Pac), 8);
	const auto x86 = note(a::kNtGnuPropertyType0, "GNU", property(0xc0000002, 3), 8); // x86 IBT | SHSTK
	const auto build_id = note(3, "GNU", std::vector<uint8_t>(20, 0xab), 4);
	CHECK(BtiFromNotes(std::data(bti), std::size(bti), 8) == NoteVerdict::Bti);
	CHECK(BtiFromNotes(std::data(pac), std::size(pac), 8) == NoteVerdict::NoBti);
	CHECK(BtiFromNotes(std::data(x86), std::size(x86), 8) == NoteVerdict::NoBti);
	CHECK(BtiFromNotes(std::data(build_id), std::size(build_id), 4) == NoteVerdict::NoBti);
	std::vector<uint8_t> both = x86;
	both.insert(std::end(both), std::begin(bti), std::end(bti));
	CHECK(BtiFromNotes(std::data(both), std::size(both), 8) == NoteVerdict::Bti); // the second note counts
	CHECK(BtiFromNotes(std::data(bti), std::size(bti) - 3, 8) == NoteVerdict::Malformed); // truncated
	std::vector<uint8_t> trailing = pac;
	trailing.insert(std::end(trailing), 5,
					0); // a well-formed note, then 5 bytes that cannot be a note header
	CHECK(BtiFromNotes(std::data(trailing), std::size(trailing), 8) == NoteVerdict::Malformed);
	CHECK(BtiFromNotes(std::data(bti), std::size(bti), 2) == NoteVerdict::Malformed); // impossible alignment
	CHECK(BtiFromNotes(nullptr, 0, 8) == NoteVerdict::NoBti);						  // no notes at all
	// This process: the test binary's own notes read through dl_iterate_phdr (well formed, no aarch64 BTI on
	// an x86-64 host; a native aarch64 test binary built without -mbranch-protection has none either), and an
	// address no object contains.
	CHECK(cmt::patch::LoadedObjectBti(reinterpret_cast<uintptr_t>(&test_bti_notes)) == NoteVerdict::NoBti);
	CHECK(cmt::patch::LoadedObjectBti(0x10) == NoteVerdict::Malformed);
}

// ---------------------------------------------------------------------------------------------------------------
// Rows and the arena
// ---------------------------------------------------------------------------------------------------------------

void test_select_row()
{
	const char* why = nullptr;
	const VersionTable& x = cmt::patch::VersionTableFor(Arch::X86_64);
	const VersionTable& a = cmt::patch::VersionTableFor(Arch::Aarch64);
	CHECK(cmt::patch::SelectRow(x, "615.71.09", false, &why) != nullptr && *why == '\0');
	CHECK(cmt::patch::SelectRow(a, "615.71.09", false, &why) == nullptr &&
		  std::strstr(why, "MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS") != nullptr);
	const DriverVersion* r = cmt::patch::SelectRow(a, "615.71.09", true, &why);
	CHECK(r != nullptr && r->h2h_va == 0x332580 && std::strstr(why, "UNVERIFIED") != nullptr);
	CHECK(cmt::patch::SelectRow(a, "999.1.1", true, &why) == nullptr &&
		  std::strstr(why, "no row") != nullptr);
	CHECK(cmt::patch::SelectRow(x, "615.71.09", false, &why) !=
		  cmt::patch::SelectRow(a, "615.71.09", true, &why));
	// A scanned-only row in an enabled table (580.178.04, manifest purpose `scanned`) needs the same opt-in.
	const DriverVersion* scanned = cmt::patch::FindVersion(x, "580.178.04");
	CHECK(scanned != nullptr && scanned->scanned_only);
	CHECK(cmt::patch::SelectRow(x, "580.178.04", false, &why) == nullptr &&
		  std::strstr(why, "MEMCPY_HOOK_ALLOW_UNVERIFIED_ROWS") != nullptr);
	CHECK(cmt::patch::SelectRow(x, "580.178.04", true, &why) == scanned && std::strstr(why, "UNVERIFIED") != nullptr);
	CHECK(&cmt::patch::NativeVersionTable() == &cmt::patch::VersionTableFor(cmt::kNativeArch));
}

std::array<uint8_t, 33> original_inline_gate(const cmt::patch::DriverVersion& row)
{
	std::array<uint8_t, 33> bytes{};
	// SAFETY: the fixed extents 13 + 11 + 9 cover this owned array exactly.
	std::memcpy(bytes.data(), row.inline_paged_check.original, 13);
	std::memcpy(bytes.data() + 13, row.inline_pinned_value.original, 11);
	std::memcpy(bytes.data() + 24, row.inline_pinned_check.original, 9);
	return bytes;
}

void test_inline_thresholds()
{
	const auto& table = cmt::patch::VersionTableFor(Arch::X86_64);
	for (std::size_t i = 0; i < table.count; ++i)
	{
		const auto& row = table.rows[i];
		CHECK(row.inline_paged_check.offset != 0);
		const auto gate_bytes = original_inline_gate(row);
		const auto* original = gate_bytes.data();
		CHECK(rd32(original + 7) == 65536);
		CHECK(rd32(original + 3) == row.copying_width_bytes &&
			  rd32(original + 27) == row.copying_width_bytes);
		for (uint64_t limit : {uint64_t{0}, uint64_t{1}, uint64_t{0x7fffffff}, uint64_t{0x80000000},
							   uint64_t{0xffffffff}, uint64_t{0x123456789abcdef0}, UINT64_MAX})
		{
			auto plan = cmt::patch::PlanInlineThresholds(Arch::X86_64, row, original, 33, 0x7fffffff, limit);
			CHECK(plan.ok && *plan.why == '\0');
			if (!plan.ok)
			{
				return; // Never decode an uninitialized replacement after a failed assertion.
			}
			CHECK(plan.paged_limit_va == row.inline_paged_value.offset);
			CHECK(plan.paged_limit_va == row.inline_paged_check.offset + 7);
			CHECK(plan.pinned_fetch_va == row.inline_pinned_value.offset);
			CHECK(plan.pinned_fetch_va == row.inline_paged_check.offset + 13);
			CHECK(rd32(plan.paged_limit) == 0x7fffffff);
			CHECK(rd64(plan.pinned_fetch + 2) == limit);
			// Decode replacement plus original comparison/branch independently.
			uint8_t patched[33];
			std::memcpy(patched, original, sizeof(patched));
			std::memcpy(patched + 7, plan.paged_limit, 4);
			std::memcpy(patched + 13, plan.pinned_fetch, 11);
			cmt::patch::disasm::Session cs(Arch::X86_64);
			CHECK(cs.IsValid());
			const uint8_t* pc = patched;
			std::size_t left = sizeof(patched);
			uint64_t address = row.inline_paged_check.offset;
			const unsigned expected[] = {X86_INS_CMP, X86_INS_JBE, X86_INS_MOVABS,
										 X86_INS_NOP, X86_INS_CMP, X86_INS_JAE};
			for (unsigned id : expected)
			{
				const cs_insn* insn = cs.Next(&pc, &left, &address);
				CHECK(insn && insn->id == id);
				if (insn && id == X86_INS_MOVABS)
				{
					const auto& x = insn->detail->x86;
					CHECK(x.op_count == 2 && x.operands[0].type == X86_OP_REG &&
						  x.operands[0].reg == X86_REG_RAX && x.operands[0].size == 8);
					CHECK(x.operands[1].type == X86_OP_IMM &&
						  static_cast<uint64_t>(x.operands[1].imm) == limit);
				}
			}
			CHECK(left == 0);
			for (std::size_t b = 0; b < sizeof(patched); ++b)
			{
				if (!(b >= 7 && b < 11) && !(b >= 13 && b < 24))
				{
					CHECK(patched[b] == original[b]);
				}
			}
		}
		CHECK(cmt::patch::PlanInlineThresholds(Arch::X86_64, row, original, 33, 0, 0).ok);
		// Every original byte participates in the guard, including both jumps.
		for (std::size_t b = 0; b < 33; ++b)
		{
			uint8_t altered[33];
			std::memcpy(altered, original, 33);
			altered[b] ^= 1;
			CHECK(!cmt::patch::PlanInlineThresholds(Arch::X86_64, row, altered, 33, 1, 2).ok);
		}
		for (std::size_t length = 0; length < 33; ++length)
		{
			CHECK(!cmt::patch::PlanInlineThresholds(Arch::X86_64, row, original, length, 1, 2).ok);
		}
		CHECK(!cmt::patch::PlanInlineThresholds(Arch::X86_64, row, nullptr, 33, 1, 2).ok);
		CHECK(!cmt::patch::PlanInlineThresholds(Arch::Aarch64, row, original, 33, 1, 2).ok);
		for (uint64_t limit : {uint64_t{0x80000000}, UINT64_MAX})
		{
			const auto rejected = cmt::patch::PlanInlineThresholds(Arch::X86_64, row, original, 33, limit, 2);
			CHECK(!rejected.ok && rejected.paged_limit_va == 0 && rejected.pinned_fetch_va == 0);
		}
		auto changed = row;
		const auto reject_changed_site = [&](auto& site)
		{
			site.offset ^= 1;
			CHECK(!cmt::patch::PlanInlineThresholds(Arch::X86_64, changed, original, 33, 1, 2).ok);
			site.offset ^= 1;
			for (std::size_t b = 0; b < sizeof(site.original); ++b)
			{
				site.original[b] ^= 1;
				CHECK(!cmt::patch::PlanInlineThresholds(Arch::X86_64, changed, original, 33, 1, 2).ok);
				site.original[b] ^= 1;
			}
		};
		reject_changed_site(changed.inline_paged_value);
		reject_changed_site(changed.inline_paged_check);
		reject_changed_site(changed.inline_pinned_value);
		reject_changed_site(changed.inline_pinned_check);
		auto invalid = row;
		invalid.inline_paged_check.offset = UINT32_MAX;
		CHECK(!cmt::patch::PlanInlineThresholds(Arch::X86_64, invalid, original, 33, 1, 2).ok);
		invalid = row;
		invalid.inline_paged_check.offset = 0;
		CHECK(!cmt::patch::PlanInlineThresholds(Arch::X86_64, invalid, original, 33, 1, 2).ok);
	}
	const auto& arm = cmt::patch::VersionTableFor(Arch::Aarch64);
	for (std::size_t i = 0; i < arm.count; ++i)
	{
		CHECK(arm.rows[i].inline_paged_value.offset == 0);
		CHECK(arm.rows[i].inline_paged_check.offset == 0);
		CHECK(arm.rows[i].inline_pinned_value.offset == 0);
		CHECK(arm.rows[i].inline_pinned_check.offset == 0);
	}
}

// Execute the generated instructions in an isolated CPU fixture, never libcuda.
// The original gate's branch destinations are redirected to return 1/2; the
// untaken path returns 0. This exercises CMP's sign extension and unsigned Jcc
// boundaries, including values whose high 32 bits would be lost by mov eax.
void test_inline_execution()
{
#if defined(__x86_64__)
	const auto& table = cmt::patch::VersionTableFor(Arch::X86_64);
	// SAFETY: mmap owns a private 4096-byte region; MAP_FAILED is checked
	// before access. The guard below unmaps it after every normal/early return.
	auto* memory = static_cast<uint8_t*>(
		mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
	CHECK(memory != MAP_FAILED);
	if (memory == MAP_FAILED)
	{
		return;
	}
	const auto unmap = [](uint8_t* p) { CHECK(munmap(p, 4096) == 0); };
	const std::unique_ptr<uint8_t, decltype(unmap)> mapping(memory, unmap);
	for (std::size_t i = 0; i < table.count; ++i)
	{
		const auto& row = table.rows[i];
		for (uint64_t paged : {uint64_t{0}, uint64_t{1}, uint64_t{0x7fffffff}})
		{
			for (uint64_t pinned :
				 {uint64_t{0}, uint64_t{1}, uint64_t{0x80000000}, uint64_t{0x100000001}, UINT64_MAX})
			{
				auto plan = cmt::patch::PlanInlineThresholds(
					Arch::X86_64, row, original_inline_gate(row).data(), 33, paged, pinned);
				CHECK(plan.ok);
				if (!plan.ok)
				{
					return; // CHECK records failure but does not stop execution.
				}
				// SAFETY: the guard owns the writable page, and each of these
				// constant-size copies fits in its first 58 bytes. No code runs
				// while this thread builds or rewrites the fixture.
				std::memset(memory, 0x90, 4096);
				const uint8_t entry[] = {0x53, 0x48, 0x89, 0xfb}; // push rbx; mov rbx,rdi
				std::memcpy(memory, entry, sizeof(entry));
				auto* gate = memory + sizeof(entry);
				std::memcpy(gate, original_inline_gate(row).data(), 33);
				std::memcpy(gate + 7, plan.paged_limit, 4);
				std::memcpy(gate + 13, plan.pinned_fetch, 11);
				gate[12] = 40 - 13; // pageable -> return 1
				gate[32] = 47 - 33; // pinned -> return 2
				for (unsigned arm = 0; arm < 3; ++arm)
				{
					const uint8_t result[] = {0xb8, static_cast<uint8_t>(arm), 0, 0, 0, 0x5b, 0xc3};
					std::memcpy(gate + 33 + arm * 7, result, sizeof(result));
				}
				// SAFETY: memory is page-aligned and covers the owned mapping.
				// All writes are finished before RX; no RWX interval is used.
				const int executable = mprotect(memory, 4096, PROT_READ | PROT_EXEC);
				CHECK(executable == 0);
				if (executable != 0)
				{
					return;
				}
				// SAFETY: this Linux x86-64 fixture implements the SysV ABI:
				// RDI points to descriptor bytes; RBX is saved/restored; every
				// exit pops RBX and returns its unsigned result in EAX. It makes
				// no calls and clobbers no other callee-saved register. RX was
				// established above. The mapping outlives every call, and no
				// other thread can modify it. x86 instruction-cache coherence
				// permits execution after mprotect without a cache flush.
				using Gate = unsigned (*)(const void*);
				const auto run = reinterpret_cast<Gate>(memory);
				std::vector<uint8_t> descriptor(static_cast<std::size_t>(row.copying_width_bytes) + 8);
				std::vector<uint64_t> sizes{0, 1, paged, paged + 1, pinned, UINT64_MAX};
				if (paged)
				{
					sizes.push_back(paged - 1);
				}
				if (pinned)
				{
					sizes.push_back(pinned - 1);
				}
				if (pinned != UINT64_MAX)
				{
					sizes.push_back(pinned + 1);
				}
				for (uint64_t size : sizes)
				{
					// SAFETY: descriptor owns copying_width_bytes+8 bytes; memcpy permits
					// the byte buffer's alignment and writes exactly its tail.
					std::memcpy(descriptor.data() + row.copying_width_bytes, &size, 8);
					CHECK(run(descriptor.data()) == (size <= paged ? 1u : size <= pinned ? 2u : 0u));
				}
				// SAFETY: the last call has returned and no pointer escapes;
				// the page can become writable for the next fixture.
				const int writable = mprotect(memory, 4096, PROT_READ | PROT_WRITE);
				CHECK(writable == 0);
				if (writable != 0)
				{
					return;
				}
			}
		}
	}
#else
	std::puts("SKIP: inline execution fixture requires x86-64; planner checks still run");
#endif
}

void test_arena()
{
	namespace d = cmt::patch::disasm;
	CHECK(d::ArenaUsed() == 0); // every plan above closed its session
	std::printf("Capstone arena: peak %zu of %zu bytes\n", d::ArenaPeak(), d::kArenaBytes);
	CHECK(d::ArenaPeak() > 0 && d::ArenaPeak() < d::kArenaBytes / 4);
	{
		d::Session first(Arch::X86_64);
		CHECK(first.IsValid() && d::ArenaUsed() > 0);
		d::Session second(Arch::Aarch64);
		CHECK(!second.IsValid()); // one session at a time: refused, not shared
		// A plan while the arena is taken fails closed.
		const uint8_t p[16] = {0x55, 0x41, 0x89, 0xc9, 0x48, 0x89, 0xf1,
							   0x31, 0xc0, 0x48, 0x89, 0xe5, 0x41, 0x57};
		CHECK(!cmt::patch::PlanWrap(Arch::X86_64, p, 16, 0x400000, 0x401000, 0).ok);
	}
	CHECK(d::ArenaUsed() == 0);
}

} // namespace

int main()
{
	test_assembled_templates();
	test_x86_table();
	test_x86_rip_relative();
	test_x86_refusals();
	test_a64_table();
	test_a64_classification();
	test_bti_notes();
	test_select_row();
	test_inline_thresholds();
	test_inline_execution();
	test_arena();
	return check::exit_code();
}
