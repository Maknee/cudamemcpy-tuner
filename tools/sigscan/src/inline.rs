//! x86-64 inline-copy threshold sites. Byte patterns only nominate candidates;
//! decoded instructions, descriptor fields and control flow establish a match.
//!
//! The pageable gate compares an unsigned size with a sign-extended imm32.
//! Its taken branch checks pageable eligibility; it is NOT the copy body.
//! The pinned gate fetches a threshold into RAX, then tests RAX >= size.
//! Replacing only the fetch preserves the comparison, flags and branch polarity.
//! No version numbers or known site addresses participate in matching.

use crate::{
    image::{Arch, Image},
    scan::{Descriptor, Status},
};
use anyhow::{bail, ensure, Context, Result};
use capstone::{
    arch::{
        x86::{X86Insn::*, X86Operand, X86OperandType, X86Reg::*},
        BuildsCapstone, DetailsArchInsn,
    },
    Capstone, Insn, RegId,
};
use std::{collections::BTreeMap, fmt::Write, num::NonZeroU32};

/// Candidate bytes (not sufficient evidence to emit a patch location):
/// ```text
/// +00 48 81 BB <disp32> <imm32>   cmp qword ptr [rbx+size], LIMIT
/// +11 76 <rel8>                  jbe pageable_eligibility
/// +13 BF <imm32>                 mov edi, query_id
/// +18 FF 9? <disp32>             call qword ptr [base+slot] ; returns limit in RAX
/// +24 48 3B 83 <disp32>          cmp rax, qword ptr [rbx+size]
/// +31 73 <rel8>                  jae pinned_eligibility
/// ```
/// LIMIT occupies +7..+11. Nonnegative replacements must fit INT32_MAX:
/// this instruction sign-extends imm32 to 64 bits before the unsigned comparison.
/// `FF 9?` is deliberately only a prefilter: /3 far calls and SIB forms are rejected.
pub const INLINE_PAGED: &str =
    "48 81 BB ?? ?? ?? ?? ?? ?? ?? ?? 76 ?? BF ?? ?? ?? ?? FF 9? ?? ?? ?? ?? 48 3B 83 ?? ?? ?? ?? 73 ??";

/// The pinned fetch begins at paged+13. Replace exactly its first 11 bytes:
/// `48 B8 <imm64 little-endian> 90` = `movabs rax, LIMIT; nop`.
/// This supports the full u64 range. The following 7-byte compare and 2-byte
/// JAE remain byte-for-byte unchanged; do not reverse the compare or flip JAE.
/// Removing the call also removes its effects: static matching proves the
/// instruction/control-flow shape, not that the original callee has no side effects.
pub const INLINE_PINNED: &str = "BF ?? ?? ?? ?? FF 9? ?? ?? ?? ?? 48 3B 83 ?? ?? ?? ?? 73 ??";

/// One parsed token: haystack byte b matches iff (b & mask) == value.
#[derive(Clone, Copy)]
struct Token {
    value: u8,
    mask: u8,
}

struct Pattern {
    tokens: Vec<Token>,
}

impl Pattern {
    /// "48 81 BB ?? ..." -> tokens; `?x`/`x?` keep one nibble fixed.
    fn parse(text: &str) -> Result<Self> {
        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            ensure!(word.len() == 2, "token '{word}' is not two hex digits");
            let hi = nibble(word.as_bytes()[0])?;
            let lo = nibble(word.as_bytes()[1])?;
            tokens.push(match (hi, lo) {
                (Some(h), Some(l)) => Token {
                    value: h << 4 | l,
                    mask: 0xFF,
                },
                (Some(h), None) => Token {
                    value: h << 4,
                    mask: 0xF0,
                },
                (None, Some(l)) => Token {
                    value: l,
                    mask: 0x0F,
                },
                (None, None) => Token {
                    value: 0,
                    mask: 0x00,
                },
            });
        }
        ensure!(!tokens.is_empty(), "empty pattern");
        Ok(Self { tokens })
    }

    /// Every offset where the pattern matches; `base` turns offsets into VAs.
    fn find_all(&self, hay: &[u8], base: u64) -> Vec<u64> {
        let n = self.tokens.len();
        if hay.len() < n {
            return Vec::new();
        }
        (0..=hay.len() - n)
            .filter(|&i| {
                self.tokens
                    .iter()
                    .enumerate()
                    .all(|(j, t)| hay[i + j] & t.mask == t.value)
            })
            .map(|i| base + i as u64)
            .collect()
    }
}

fn nibble(c: u8) -> Result<Option<u8>> {
    Ok(match c {
        b'?' => None,
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => bail!("bad nibble '{}'", c as char),
    })
}

pub const GATE_BYTES: usize = 33;
const PINNED_OFFSET: u64 = 13;

/// Only `validate` constructs this type. Every instance owns all 33 bytes and
/// a finalized, nonzero image offset; there is no later `finish` step. Captures
/// are read from those bytes so they cannot drift from the generated guard.
///
/// This establishes the gate's static shape, not that removing the query call
/// preserves its side effects or other live registers. Live installation is a
/// separate operation; this scanner never authorizes a live installation.
#[derive(Debug)]
pub struct ValidatedGate {
    address: u64,
    offset: NonZeroU32,
    original: [u8; GATE_BYTES],
    evidence: Vec<String>,
}
impl ValidatedGate {
    pub fn address(&self) -> u64 {
        self.address
    }
    pub fn offset(&self) -> u32 {
        self.offset.get()
    }
    pub fn original(&self) -> &[u8; GATE_BYTES] {
        &self.original
    }
    pub fn pinned_address(&self) -> u64 {
        self.address + PINNED_OFFSET
    }
    pub fn paged_limit(&self) -> u32 {
        self.read_u32(7)
    }
    pub fn call_arg(&self) -> u32 {
        self.read_u32(14)
    }
    pub fn call_disp(&self) -> u32 {
        self.read_u32(20)
    }
    pub fn size_disp(&self) -> u32 {
        self.read_u32(3)
    }
    fn read_u32(&self, at: usize) -> u32 {
        // All callers above are fixed fields inside the owned 33-byte gate.
        u32::from_le_bytes(
            self.original[at..at + 4]
                .try_into()
                .expect("fixed gate field"),
        )
    }
}

/// Failure states contain diagnostics, never usable patch bytes. Unsupported
/// is distinct from a missing x86 signature: only the former emits a zero gate.
#[derive(Debug)]
pub enum InlineCopy {
    Found(ValidatedGate),
    Unsupported,
    NotFound {
        rejected: Vec<String>,
    },
    Ambiguous {
        candidates: Vec<u64>,
        rejected: Vec<String>,
    },
}
impl InlineCopy {
    pub fn status(&self) -> Status {
        match self {
            Self::Found(_) => Status::Found,
            Self::Unsupported => Status::Absent,
            Self::NotFound { .. } => Status::NotFound,
            Self::Ambiguous { .. } => Status::Ambiguous,
        }
    }
    #[cfg(test)]
    pub fn gate(&self) -> Option<&ValidatedGate> {
        if let Self::Found(gate) = self {
            Some(gate)
        } else {
            None
        }
    }
    pub fn complete(&self) -> bool {
        matches!(self.status(), Status::Found | Status::Absent)
    }

    pub fn explain(&self) -> String {
        match self {
            Self::Found(gate) => {
                let mut out = format!(
                    "-- inline thresholds: found at {:#x}, offset {:#x}\n",
                    gate.address(),
                    gate.offset()
                );
                writeln!(out, "   pinned fetch {:#x}; paged limit {:#x}; query {:#x}; slot {:#x}; width {:#x}",
                    gate.pinned_address(), gate.paged_limit(), gate.call_arg(), gate.call_disp(), gate.size_disp()).unwrap();
                for evidence in &gate.evidence {
                    writeln!(out, "   {evidence}").unwrap();
                }
                out
            }
            Self::Unsupported => "-- inline thresholds: unsupported on AArch64\n".into(),
            Self::NotFound { rejected } => {
                format!("-- inline thresholds: not found\n{}", rejected.join("\n"))
            }
            Self::Ambiguous {
                candidates,
                rejected,
            } => format!(
                "-- inline thresholds: ambiguous: {candidates:#x?}\n{}",
                rejected.join("\n")
            ),
        }
    }

    /// Appends to DriverVersion in declaration order. Validate the outcome before
    /// touching `out`: missing/ambiguous gates must never look like unsupported
    /// architectures, even if a caller forgets to check Scan::complete().
    pub fn write_fields(&self, out: &mut String) -> Result<()> {
        let (offset, original): (u32, &[u8]) = match self {
            Self::Found(gate) => (gate.offset(), gate.original()),
            Self::Unsupported => (0, &[]),
            _ => anyhow::bail!(
                "cannot emit unresolved inline thresholds: {}",
                self.explain()
            ),
        };
        // Offsets and byte views come from the same validated 33-byte gate.
        // The pageable value is embedded in its check; exposing both makes the
        // four roles usable without callers inventing their own byte offsets.
        for (name, start, length) in [
            ("inline_paged_value", 7usize, 4usize),
            ("inline_paged_check", 0, 13),
            ("inline_pinned_value", 13, 11),
            ("inline_pinned_check", 24, 9),
        ] {
            let (site_offset, bytes) = if original.is_empty() {
                (0, &[][..]) // Unsupported architecture, not a failed scan.
            } else {
                (offset + start as u32, &original[start..start + length])
            };
            let bytes = bytes
                .iter()
                .map(|b| format!("0x{b:02x}"))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(
                out,
                "  .{name} = {{.offset = {site_offset:#x}, .original = {{{bytes}}}}},"
            )
            .unwrap();
        }
        Ok(())
    }
}

fn operands(cs: &Capstone, insn: &Insn<'_>) -> Result<Vec<X86Operand>> {
    let detail = cs.insn_detail(insn)?;
    let arch = detail.arch_detail();
    Ok(arch
        .x86()
        .context("not an x86 instruction")?
        .operands()
        .collect())
}
fn reg(op: &X86Operand, expected: u32, width: u8) -> bool {
    op.size == width && matches!(op.op_type, X86OperandType::Reg(r) if r.0 == expected as u16)
}
fn imm(op: &X86Operand) -> Option<i64> {
    if let X86OperandType::Imm(value) = op.op_type {
        Some(value)
    } else {
        None
    }
}
fn mem(op: &X86Operand, base: u32, disp: u64, width: u8) -> bool {
    op.size == width
        && matches!(op.op_type, X86OperandType::Mem(m)
        if m.base().0 == base as u16 && m.index() == RegId(0) && m.segment() == RegId(0)
        && m.disp() >= 0 && m.disp() as u64 == disp)
}
fn target(cs: &Capstone, insn: &Insn<'_>) -> Result<Option<u64>> {
    // A failed decode is not an indirect branch: the caller must not silently
    // omit that edge from the patch-interior audit.
    let ops = operands(cs, insn)?;
    Ok((ops.len() == 1)
        .then(|| imm(&ops[0]))
        .flatten()
        .map(|v| v as u64))
}

fn validate(
    image: &Image,
    descriptor: &Descriptor,
    cs: &Capstone,
    address: u64,
) -> Result<ValidatedGate> {
    let end = address
        .checked_add(GATE_BYTES as u64)
        .context("gate address overflow")?;
    ensure!(
        address > image.image_base
            && end
                .checked_sub(image.image_base)
                .is_some_and(|offset| u32::try_from(offset).is_ok()),
        "gate must have a nonzero image offset and fit wholly in the table's u32 range"
    );
    let offset =
        NonZeroU32::new(u32::try_from(address - image.image_base)?).context("zero gate offset")?;
    let owners: Vec<_> = image
        .functions
        .iter()
        .filter(|f| f.start <= address && end <= f.end())
        .collect();
    ensure!(
        owners.len() == 1,
        "gate must belong to exactly one unwind function"
    );
    let function = owners[0];
    let code = image
        .at(function.start)
        .and_then(|b| b.get(..function.size as usize))
        .context("function is not wholly file-backed in one segment")?;
    ensure!(
        image.executable(function.start) && image.executable(function.end() - 1),
        "function is not executable"
    );
    let insns = cs.disasm_all(code, function.start)?;
    ensure!(
        insns
            .last()
            .is_some_and(|i| i.address() + i.bytes().len() as u64 == function.end()),
        "function does not decode completely from its unwind start"
    );
    let boundaries: BTreeMap<_, _> = insns.iter().map(|i| (i.address(), i)).collect();
    let at = |pc: u64| {
        boundaries
            .get(&pc)
            .copied()
            .context("not an instruction boundary")
    };
    let gate = [
        at(address)?,
        at(address + 11)?,
        at(address + 13)?,
        at(address + 18)?,
        at(address + 24)?,
        at(address + 31)?,
    ];
    for ((instruction, id), size) in gate
        .iter()
        .zip([
            X86_INS_CMP,
            X86_INS_JBE,
            X86_INS_MOV,
            X86_INS_CALL,
            X86_INS_CMP,
            X86_INS_JAE,
        ])
        .zip([11, 2, 5, 6, 7, 2])
    {
        ensure!(
            instruction.id().0 == id as u32 && instruction.bytes().len() == size,
            "unexpected gate instruction or encoding length"
        );
    }
    // Check ModRM as well as Capstone's classification: /3 far-call bytes
    // must never be accepted as the /2 near call this replacement removes.
    ensure!(
        matches!(gate[3].bytes(), [0xff, 0x91 | 0x92, _, _, _, _]),
        "query must encode a near call through RCX or RDX with disp32"
    );
    let paged = operands(cs, gate[0])?;
    let arg = operands(cs, gate[2])?;
    let call = operands(cs, gate[3])?;
    let pinned = operands(cs, gate[4])?;
    ensure!(
        descriptor.copying_width_bytes != 0
            && paged.len() == 2
            && pinned.len() == 2
            && mem(&paged[0], X86_REG_RBX, descriptor.copying_width_bytes, 8)
            && reg(&pinned[0], X86_REG_RAX, 8)
            && mem(&pinned[1], X86_REG_RBX, descriptor.copying_width_bytes, 8),
        "both comparisons must use the copy engine's width field"
    );
    let limit = imm(&paged[1]).context("paged limit is not immediate")?;
    ensure!(
        (0..=i32::MAX as i64).contains(&limit),
        "paged limit is negative after sign extension"
    );
    ensure!(
        arg.len() == 2 && reg(&arg[0], X86_REG_EDI, 4),
        "query argument is not EDI"
    );
    let call_arg = u32::try_from(imm(&arg[1]).context("query argument is not immediate")?)?;
    ensure!(
        call.len() == 1 && call[0].size == 8,
        "query is not a near indirect qword call"
    );
    let X86OperandType::Mem(slot) = call[0].op_type else {
        anyhow::bail!("query is not a memory call")
    };
    // RCX and RDX are the two observed dispatch-table registers. Broadening this
    // requires checking how the new base survives the EDI argument assignment.
    ensure!(
        [X86_REG_RCX as u16, X86_REG_RDX as u16].contains(&slot.base().0)
            && slot.index() == RegId(0)
            && slot.segment() == RegId(0)
            && slot.disp() > 0
            && slot.disp() % 8 == 0,
        "unsupported dispatch-table addressing"
    );
    let call_disp = u32::try_from(slot.disp())?;
    let paged_target = target(cs, gate[1])?.context("paged branch has no direct target")?;
    let pinned_target = target(cs, gate[5])?.context("pinned branch has no direct target")?;
    ensure!(
        paged_target >= end && pinned_target > paged_target,
        "eligibility branches must go forward to distinct blocks"
    );
    let pg = at(paged_target)?;
    let pg_ops = operands(cs, pg)?;
    ensure!(
        pg.id().0 == X86_INS_CMP as u32
            && pg_ops.len() == 2
            && mem(&pg_ops[0], X86_REG_RBX, descriptor.op_kind, 4)
            && imm(&pg_ops[1]) == Some(descriptor.op_kind_3 as i64),
        "paged target does not check the descriptor's op_kind_3"
    );
    let pg_branch = at(paged_target + pg.bytes().len() as u64)?;
    ensure!(
        pg_branch.id().0 == X86_INS_JNE as u32
            && target(cs, pg_branch)? == Some(address + PINNED_OFFSET),
        "non-op_kind_3 path does not rejoin at the start of the pinned fetch"
    );
    let pin = at(pinned_target)?;
    let pin_ops = operands(cs, pin)?;
    ensure!(
        pin.id().0 == X86_INS_MOV as u32
            && pin_ops.len() == 2
            && reg(&pin_ops[0], X86_REG_RDI, 8)
            && reg(&pin_ops[1], X86_REG_RBX, 8),
        "pinned target does not pass the descriptor to its eligibility helper"
    );
    let helper = at(pinned_target + pin.bytes().len() as u64)?;
    ensure!(
        helper.id().0 == X86_INS_CALL as u32
            && target(cs, helper)?.is_some_and(
                |t| image.functions.iter().any(|f| f.start == t) && image.executable(t)
            ),
        "pinned eligibility helper is not a direct call to a known function"
    );
    // Changing two instructions into movabs+nop invalidates their interior
    // entry points. Also reject any direct edge into the middle of the paged cmp.
    // Entries at +13 and +24 remain valid. This audits this function's direct
    // branches/calls; it cannot prove absence of computed or external entries.
    for insn in insns.iter() {
        let details = cs.insn_detail(insn)?;
        let branch = details.groups().iter().any(|g| {
            g.0 == capstone::InsnGroupType::CS_GRP_JUMP as u8
                || g.0 == capstone::InsnGroupType::CS_GRP_CALL as u8
        });
        if branch {
            if let Some(t) = target(cs, insn)? {
                ensure!(
                    !(address < t && t < address + 11) && !(address + 13 < t && t < address + 24),
                    "direct control-flow entry into patch interior at {t:#x}"
                );
            }
        }
    }
    let original: [u8; GATE_BYTES] = image
        .at(address)
        .and_then(|b| b.get(..GATE_BYTES))
        .context("truncated gate")?
        .try_into()
        .context("gate byte count")?;
    Ok(ValidatedGate {
        address,
        offset,
        original,
        evidence: vec![
            format!(
                "unwind function {:#x}..{:#x}; decoded from entry",
                function.start,
                function.end()
            ),
            format!(
                "width [rbx+{:#x}] agrees with copy engine; original LIMIT {limit:#x}",
                descriptor.copying_width_bytes
            ),
            format!(
                "LIMIT imm32 at {:#x}; allowed replacement 0..=0x7fffffff",
                address + 7
            ),
            format!("JBE -> {paged_target:#x}: op_kind_3 check; other kinds -> pinned fetch"),
            format!(
                "query {call_arg:#x}, table {}+{call_disp:#x}; result consumed as RAX",
                cs.reg_name(slot.base()).context("dispatch register name")?
            ),
            format!(
                "replace 11 bytes at {:#x} with movabs rax, imm64; nop",
                address + PINNED_OFFSET
            ),
            format!(
                "preserve cmp rax, [rbx+{:#x}] and JAE -> {pinned_target:#x}",
                descriptor.copying_width_bytes
            ),
            "no direct branch/call in the containing function enters either patch interior".into(),
        ],
    })
}

pub fn scan(image: &Image, descriptor: &Descriptor) -> Result<InlineCopy> {
    if image.arch != Arch::X86_64 {
        return Ok(InlineCopy::Unsupported);
    }
    let cs = Capstone::new()
        .x86()
        .mode(capstone::arch::x86::ArchMode::Mode64)
        .detail(true)
        .build()?;
    let pattern = Pattern::parse(INLINE_PAGED)?;
    let pinned_pattern = Pattern::parse(INLINE_PINNED)?;
    let mut candidates = Vec::new();
    for segment in image.segments.iter().filter(|s| s.executable) {
        if let Some(bytes) = image
            .at(segment.vaddr)
            .and_then(|b| b.get(..segment.file_size as usize))
        {
            candidates.extend(pattern.find_all(bytes, segment.vaddr));
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    let mut matches = Vec::new();
    let mut rejected = Vec::new();
    for address in candidates {
        let result = (|| {
            let bytes = image
                .at(address + PINNED_OFFSET)
                .context("missing pinned window")?;
            ensure!(
                pinned_pattern.find_all(&bytes[..(GATE_BYTES - 13).min(bytes.len())], address + 13)
                    == [address + 13],
                "pinned pattern mismatch"
            );
            validate(image, descriptor, &cs, address)
        })();
        match result {
            Ok(site) => matches.push(site),
            // Decoder failures invalidate the audit, rather than merely making
            // one candidate a non-match while another is allowed to win.
            Err(error) if error.is::<capstone::Error>() => {
                return Err(error).context("decode inline gate")
            }
            Err(error) => rejected.push(format!("rejected {address:#x}: {error:#}")),
        }
    }
    if matches.len() == 1 {
        let mut found = matches.remove(0);
        found.evidence.extend(rejected);
        return Ok(InlineCopy::Found(found));
    }
    Ok(if matches.is_empty() {
        InlineCopy::NotFound { rejected }
    } else {
        InlineCopy::Ambiguous {
            candidates: matches.iter().map(|m| m.address()).collect(),
            rejected,
        }
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::image::{Function, Segment};

    // Independent hand-encoded control-flow fixture. Its layout/values differ
    // from the corpus, so matching a known offset or version cannot pass.
    pub(crate) fn fixture() -> (Image, Descriptor) {
        let base = 0x1000;
        let mut bytes = vec![0x90; 0x101];
        let gate = [
            0x48, 0x81, 0xbb, 0xd0, 1, 0, 0, 0x34, 0x12, 0, 0, // cmp [rbx+0x1d0], 0x1234
            0x76, 0x33, // jbe base+0x50
            0xbf, 0x42, 0, 0, 0, // mov edi, 0x42
            0xff, 0x91, 0x80, 2, 0, 0, // call [rcx+0x280]
            0x48, 0x3b, 0x83, 0xd0, 1, 0, 0, // cmp rax, [rbx+0x1d0]
            0x73, 0x3f, // jae base+0x70
        ];
        bytes[0x10..0x31].copy_from_slice(&gate);
        bytes[0x50..0x55].copy_from_slice(&[0x83, 0x3b, 3, 0x75, 0xc8]); // kind != op_kind_3 -> base+0x1d
        bytes[0x70..0x78].copy_from_slice(&[0x48, 0x89, 0xdf, 0xe8, 0x88, 0, 0, 0]); // helper at base+0x100
        bytes[0x8f] = 0xc3;
        bytes[0x100] = 0xc3;
        let image = Image {
            arch: Arch::X86_64,
            version: "0.0.0".into(),
            image_base: base,
            segments: vec![Segment {
                vaddr: base,
                file_offset: 0,
                file_size: bytes.len() as u64,
                executable: true,
            }],
            functions: vec![
                Function {
                    start: base,
                    size: 0x90,
                },
                Function {
                    start: base + 0x100,
                    size: 1,
                },
            ],
            bytes,
            plt: vec![],
            got_symbols: BTreeMap::new(),
        };
        let desc = Descriptor {
            copying_width_bytes: 0x1d0,
            op_kind_3: 3,
            ..Descriptor::default()
        };
        (image, desc)
    }
    fn found(image: &Image, desc: &Descriptor) -> ValidatedGate {
        match scan(image, desc).unwrap() {
            InlineCopy::Found(gate) => gate,
            other => panic!("{}", other.explain()),
        }
    }
    fn rejected(image: &Image, desc: &Descriptor) {
        let result = scan(image, desc).unwrap();
        assert_eq!(result.status(), Status::NotFound);
        assert!(result.gate().is_none());
        let mut output = "sentinel".to_owned();
        assert!(result.write_fields(&mut output).is_err());
        assert_eq!(output, "sentinel");
    }
    #[test]
    fn serialization_needs_no_finalization_and_errors_do_not_write() {
        let (image, desc) = fixture();
        let result = scan(&image, &desc).unwrap();
        let gate = result.gate().unwrap();
        assert_eq!(gate.offset(), 0x10); // image base 0x1000, address 0x1010
        let mut fields = String::new();
        result.write_fields(&mut fields).unwrap();
        assert_eq!(fields, concat!(
            "  .inline_paged_value = {.offset = 0x17, .original = {0x34,0x12,0x00,0x00}},\n",
            "  .inline_paged_check = {.offset = 0x10, .original = {0x48,0x81,0xbb,0xd0,0x01,0x00,0x00,0x34,0x12,0x00,0x00,0x76,0x33}},\n",
            "  .inline_pinned_value = {.offset = 0x1d, .original = {0xbf,0x42,0x00,0x00,0x00,0xff,0x91,0x80,0x02,0x00,0x00}},\n",
            "  .inline_pinned_check = {.offset = 0x28, .original = {0x48,0x3b,0x83,0xd0,0x01,0x00,0x00,0x73,0x3f}},\n",
        ));
        for rejected in [
            InlineCopy::NotFound { rejected: vec![] },
            InlineCopy::Ambiguous {
                candidates: vec![0x1010, 0x1210],
                rejected: vec![],
            },
        ] {
            let mut output = "unchanged".to_owned();
            assert!(rejected.write_fields(&mut output).is_err());
            assert_eq!(output, "unchanged");
        }
    }

    #[test]
    fn branch_detail_errors_are_not_indirect_edges() {
        let (image, _) = fixture();
        // A decoder without details makes insn_detail fail deterministically.
        let cs = Capstone::new()
            .x86()
            .mode(capstone::arch::x86::ArchMode::Mode64)
            .build()
            .unwrap();
        let instructions = cs.disasm_all(&image.bytes[0x1b..0x1d], 0x101b).unwrap();
        assert!(target(&cs, instructions.iter().next().unwrap()).is_err());
    }

    #[test]
    fn captures_and_dispatch_registers() {
        let (mut image, desc) = fixture();
        for modrm in [0x91, 0x92] {
            image.bytes[0x23] = modrm;
            let result = found(&image, &desc);
            assert_eq!(result.address(), 0x1010);
            assert_eq!(result.pinned_address(), 0x101d);
            assert_eq!(
                (
                    result.paged_limit(),
                    result.call_arg(),
                    result.call_disp(),
                    result.size_disp()
                ),
                (0x1234, 0x42, 0x280, 0x1d0)
            );
            assert_eq!(result.original, image.bytes[0x10..0x31]);
        }
    }
    #[test]
    fn reject_wrong_operands_branches_and_fields() {
        // Includes far call /3, SIB, base overwritten by EDI, sign-extended
        // negative LIMIT, wrong compare operand/width and branch into an operand.
        for (offset, value) in [
            (3, 0xd8),
            (7, 0xff),
            (10, 0x80),
            (11, 0x77),
            (12, 0x34),
            (13, 0xbe),
            (19, 0x99),
            (19, 0x94),
            (19, 0x97),
            (20, 0x81),
            (24, 0x40),
            (26, 0x8b),
            (27, 0xd8),
            (31, 0x76),
            (32, 0x40),
        ] {
            let (mut image, desc) = fixture();
            image.bytes[0x10 + offset] = value;
            // A positive changed limit is valid; this case tests that variability.
            if offset == 7 {
                found(&image, &desc);
            } else {
                rejected(&image, &desc);
            }
        }
        for (offset, value) in [
            (0x52, 2),
            (0x53, 0x74),
            (0x54, 0xcd),
            (0x54, 0xd3), // rejoin at CMP instead of query: aligned, but semantically wrong
            (0x72, 0xde),
            (0x73, 0xe9),
            (0x74, 0x89),
        ] {
            let (mut image, desc) = fixture();
            image.bytes[offset] = value;
            rejected(&image, &desc);
        }
        let (image, mut desc) = fixture();
        desc.copying_width_bytes += 8;
        rejected(&image, &desc);
    }
    #[test]
    fn reject_interiors_truncation_and_non_code() {
        for destination in [0x11u8, 0x17, 0x1e, 0x22] {
            let (mut image, desc) = fixture();
            image.bytes[0..2].copy_from_slice(&[0xeb, destination - 2]);
            rejected(&image, &desc);
        }
        let (mut image, desc) = fixture();
        image.bytes[0xe..0x10].copy_from_slice(&[0x48, 0xb8]); // gate starts inside movabs immediate
        rejected(&image, &desc);
        let (mut image, desc) = fixture();
        image.functions.clear();
        rejected(&image, &desc);
        let (mut image, desc) = fixture();
        image.functions[0].size = 0x30;
        rejected(&image, &desc);
        let (mut image, desc) = fixture();
        image.segments[0].executable = false;
        rejected(&image, &desc);
        for length in 0x10..0x31 {
            let (mut image, desc) = fixture();
            image.bytes.truncate(length);
            image.segments[0].file_size = length as u64;
            rejected(&image, &desc);
        }
    }
    #[test]
    fn decoys_do_not_hide_ambiguity() {
        let (mut image, desc) = fixture();
        let mut other = image.bytes.clone();
        image.bytes.resize(0x200, 0x90);
        image.bytes.append(&mut other);
        image.segments[0].file_size = image.bytes.len() as u64;
        // This raw byte match is outside any unwind function and is rejected.
        found(&image, &desc);
        image.functions.extend([
            Function {
                start: 0x1200,
                size: 0x90,
            },
            Function {
                start: 0x1300,
                size: 1,
            },
        ]);
        let result = scan(&image, &desc).unwrap();
        assert_eq!(result.status(), Status::Ambiguous);
        let InlineCopy::Ambiguous { candidates, .. } = &result else {
            panic!("expected ambiguous")
        };
        assert_eq!(candidates, &[0x1010, 0x1210]);
        assert!(result.gate().is_none());
        let mut output = "sentinel".to_owned();
        assert!(result.write_fields(&mut output).is_err());
        assert_eq!(output, "sentinel");
    }
    #[test]
    fn reject_unrepresentable_table_offsets() {
        let (mut image, desc) = fixture();
        image.image_base = 0x1010; // zero is the generated table's unsupported sentinel
        rejected(&image, &desc);
        let (mut image, desc) = fixture();
        image.image_base = 0;
        image.segments[0].vaddr += u32::MAX as u64;
        for function in &mut image.functions {
            function.start += u32::MAX as u64;
        }
        rejected(&image, &desc);
    }

    #[test]
    fn aarch64_explicitly_unsupported() {
        let (mut image, desc) = fixture();
        image.arch = Arch::Aarch64;
        let result = scan(&image, &desc).unwrap();
        assert_eq!(result.status(), Status::Absent);
        let mut fields = String::new();
        result.write_fields(&mut fields).unwrap();
        assert_eq!(
            fields,
            concat!(
                "  .inline_paged_value = {.offset = 0x0, .original = {}},\n",
                "  .inline_paged_check = {.offset = 0x0, .original = {}},\n",
                "  .inline_pinned_value = {.offset = 0x0, .original = {}},\n",
                "  .inline_pinned_check = {.offset = 0x0, .original = {}},\n",
            )
        );
    }
}
