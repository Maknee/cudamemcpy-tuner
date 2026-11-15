//! Instruction effects needed by the copy patterns, plus their small expression model.
//!
//! Capstone owns instruction decoding. We track only register/memory operations
//! used by the patterns. An unsupported write makes that value unknown.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use capstone::arch::{arm64, x86, BuildsCapstone, DetailsArchInsn};
use capstone::{Capstone, RegId};

use crate::image::{Arch, Image};

pub type AtomId = usize;

/// Coefficients wrap at 64 bits, as the machine does. Factors are sorted so
/// `z * height * pitch` compares equally on x86 and AArch64.
#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct Expr(pub BTreeMap<Vec<AtomId>, u64>);

impl Expr {
    pub fn constant(value: u64) -> Self {
        if value == 0 {
            Self::default()
        } else {
            Self(BTreeMap::from([(Vec::new(), value)]))
        }
    }
    pub fn atom(id: AtomId) -> Self {
        Self(BTreeMap::from([(vec![id], 1)]))
    }
    fn add_term(&mut self, factors: Vec<AtomId>, coefficient: u64) {
        let value = self
            .0
            .get(&factors)
            .copied()
            .unwrap_or(0)
            .wrapping_add(coefficient);
        if value == 0 {
            self.0.remove(&factors);
        } else {
            self.0.insert(factors, value);
        }
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut out = self.clone();
        for (factors, coefficient) in &other.0 {
            out.add_term(factors.clone(), *coefficient);
        }
        out
    }
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.scaled(u64::MAX))
    }
    pub fn scaled(&self, scale: u64) -> Self {
        let mut out = Self::default();
        for (factors, coefficient) in &self.0 {
            out.add_term(factors.clone(), coefficient.wrapping_mul(scale));
        }
        out
    }
    pub fn mul(&self, other: &Self) -> Self {
        let mut out = Self::default();
        for (left, a) in &self.0 {
            for (right, b) in &other.0 {
                let mut factors = left.clone();
                factors.extend_from_slice(right);
                factors.sort_unstable();
                out.add_term(factors, a.wrapping_mul(*b));
            }
        }
        out
    }
    pub fn degree(&self) -> usize {
        self.0.keys().map(Vec::len).max().unwrap_or(0)
    }
    pub fn as_constant(&self) -> Option<u64> {
        if self.0.is_empty() {
            Some(0)
        } else if self.0.len() == 1 {
            self.0.get(&Vec::new()).copied()
        } else {
            None
        }
    }
    pub fn constant_term(&self) -> u64 {
        self.0.get(&Vec::new()).copied().unwrap_or(0)
    }
    pub fn without_constant(&self) -> Self {
        let mut out = self.clone();
        out.0.remove(&Vec::new());
        out
    }
    pub fn as_atom_plus_const(&self) -> Option<(AtomId, u64)> {
        let mut nonconstant = self.0.iter().filter(|(factors, _)| !factors.is_empty());
        let (factors, coefficient) = nonconstant.next()?;
        (factors.len() == 1 && *coefficient == 1 && nonconstant.next().is_none())
            .then(|| (factors[0], self.constant_term()))
    }
    pub fn as_atom(&self) -> Option<AtomId> {
        self.as_atom_plus_const()
            .and_then(|(id, offset)| (offset == 0).then_some(id))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Atom {
    Init(usize),
    Load { address: Expr, width: u8 },
    Unknown { pc: u64, reg: usize },
    Return { pc: u64, reg: usize },
}

#[derive(Default)]
pub struct Atoms {
    values: Vec<Atom>,
    ids: BTreeMap<Atom, AtomId>,
}

impl Atoms {
    fn intern(&mut self, atom: Atom) -> AtomId {
        if let Some(id) = self.ids.get(&atom) {
            return *id;
        }
        let id = self.values.len();
        self.values.push(atom.clone());
        self.ids.insert(atom, id);
        id
    }
    pub fn init(&mut self, reg: usize) -> AtomId {
        self.intern(Atom::Init(reg))
    }
    pub fn load(&mut self, address: &Expr, width: u8) -> AtomId {
        self.intern(Atom::Load {
            address: address.clone(),
            width,
        })
    }
    pub fn unknown(&mut self, pc: u64, reg: usize) -> AtomId {
        self.intern(Atom::Unknown { pc, reg })
    }
    pub fn ret(&mut self, pc: u64, reg: usize) -> AtomId {
        self.intern(Atom::Return { pc, reg })
    }
    pub fn get(&self, id: AtomId) -> &Atom {
        &self.values[id]
    }
    pub fn field_of(&self, id: AtomId, base: AtomId, width: u8) -> Option<u64> {
        let Atom::Load {
            address,
            width: actual,
        } = self.values.get(id)?
        else {
            return None;
        };
        if width != 0 && width != *actual {
            return None;
        }
        let (root, offset) = address.as_atom_plus_const()?;
        (root == base).then_some(offset)
    }
    pub fn deref_of(&self, id: AtomId, base: AtomId) -> Option<(u64, u64)> {
        let Atom::Load { address, width: 8 } = self.values.get(id)? else {
            return None;
        };
        let (root, offset) = address.as_atom_plus_const()?;
        Some((self.field_of(root, base, 8)?, offset))
    }
    pub fn describe(
        &self,
        expression: &Expr,
        reg_name: impl Fn(usize) -> String + Copy,
        base: Option<AtomId>,
    ) -> String {
        fn atom(
            atoms: &Atoms,
            id: AtomId,
            names: impl Fn(usize) -> String + Copy,
            base: Option<AtomId>,
        ) -> String {
            if base == Some(id) {
                return "d".into();
            }
            match atoms.get(id) {
                Atom::Init(reg) => format!("{}@entry", names(*reg)),
                Atom::Unknown { pc, reg } => format!("?{}@{pc:#x}", names(*reg)),
                Atom::Return { pc, reg } => format!("ret{}@{pc:#x}", names(*reg)),
                Atom::Load { address, width } => {
                    let value = if let Some((id, offset)) = address.as_atom_plus_const() {
                        format!("{}[{offset:#x}]", atom(atoms, id, names, base))
                    } else {
                        format!("mem({})", atoms.describe(address, names, base))
                    };
                    if *width == 8 {
                        value
                    } else {
                        format!("{value}:{width}")
                    }
                }
            }
        }
        if expression.0.is_empty() {
            return "0".into();
        }
        let mut terms: Vec<_> = expression.0.iter().collect();
        terms.sort_by_key(|(factors, _)| std::cmp::Reverse(factors.len()));
        terms
            .into_iter()
            .map(|(factors, coefficient)| {
                let mut parts = Vec::new();
                if *coefficient != 1 || factors.is_empty() {
                    parts.push(format!("{coefficient:#x}"));
                }
                parts.extend(factors.iter().map(|id| atom(self, *id, reg_name, base)));
                parts.join("*")
            })
            .collect::<Vec<_>>()
            .join(" + ")
    }
}

#[derive(Clone, Debug, Default)]
pub struct Compare {
    pub lhs: Expr,
    pub rhs: Expr,
    pub valid: bool,
}

#[derive(Clone, Debug)]
pub struct State {
    pub regs: [Expr; 34],
    pub mem: BTreeMap<(Expr, u8), Expr>,
    pub cmp: Compare,
    pub call_args: [Expr; 4],
    pub log_stores: bool,
    pub store_log: Vec<(Expr, u8)>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            regs: std::array::from_fn(|_| Expr::default()),
            mem: BTreeMap::new(),
            cmp: Compare::default(),
            call_args: std::array::from_fn(|_| Expr::default()),
            log_stores: false,
            store_log: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FlowKind {
    #[default]
    Next,
    Jump,
    CondJump,
    Call,
    IndirectCall,
    Ret,
    IndirectJump,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Cond {
    Eq,
    Ne,
    #[default]
    Other,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Flow {
    pub kind: FlowKind,
    pub target: u64,
    pub cond: Cond,
}

#[derive(Clone, Copy, Debug)]
struct Register {
    raw: u16,
    canonical: Option<usize>,
    width: u8,
}
#[derive(Clone, Debug)]
struct Memory {
    base: Option<Register>,
    index: Option<Register>,
    scale: u64,
    displacement: i64,
    segmented: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Shift {
    None,
    Left(u32),
    Other,
}
#[derive(Clone, Debug)]
enum Value {
    Register(Register),
    Immediate(i64),
    Memory(Memory),
    Other,
}
#[derive(Clone, Debug)]
struct Operand {
    value: Value,
    width: u8,
    write: bool,
    shift: Shift,
}

/// Capstone details are copied into a small owned form. Pattern matching never
/// depends on a disassembler allocation's lifetime.
#[derive(Clone, Debug)]
pub struct Instruction {
    pub address: u64,
    pub size: u64,
    mnemonic: String,
    operands_text: String,
    operands: Vec<Operand>,
    writes: Vec<Register>,
    flags_written: bool,
    writeback: bool,
    post_index: bool,
    condition: Cond,
    conditional: bool,
}
impl Instruction {
    pub fn text(&self) -> String {
        format!(
            "{:#x}  {} {}",
            self.address, self.mnemonic, self.operands_text
        )
    }
    fn immediate(&self, index: usize) -> Option<u64> {
        match &self.operands.get(index)?.value {
            Value::Immediate(value) => Some(*value as u64),
            _ => None,
        }
    }
    fn register(&self, index: usize) -> Option<Register> {
        match self.operands.get(index)?.value {
            Value::Register(reg) => Some(reg),
            _ => None,
        }
    }
}

pub struct Decoder<'a> {
    pub image: &'a Image,
    pub atoms: Atoms,
    capstone: Capstone,
    registers: BTreeMap<u16, Register>,
    flag_registers: BTreeSet<u16>,
}

impl<'a> Decoder<'a> {
    pub fn new(image: &'a Image) -> Result<Self> {
        let capstone = match image.arch {
            Arch::X86_64 => Capstone::new()
                .x86()
                .mode(x86::ArchMode::Mode64)
                .detail(true)
                .build()?,
            Arch::Aarch64 => Capstone::new()
                .arm64()
                .mode(arm64::ArchMode::Arm)
                .detail(true)
                .build()?,
        };
        let ending = match image.arch {
            Arch::X86_64 => x86::X86Reg::X86_REG_ENDING as u16,
            Arch::Aarch64 => arm64::Arm64Reg::ARM64_REG_ENDING as u16,
        };
        let mut registers = BTreeMap::new();
        let mut flag_registers = BTreeSet::new();
        for raw in 1..ending {
            if let Some(name) = capstone.reg_name(RegId(raw)) {
                if matches!(name.as_str(), "rflags" | "eflags" | "nzcv") {
                    flag_registers.insert(raw);
                }
                registers.insert(raw, register_info(image.arch, raw, &name));
            }
        }
        Ok(Self {
            image,
            atoms: Atoms::default(),
            capstone,
            registers,
            flag_registers,
        })
    }
    fn register_info(&self, id: RegId) -> Register {
        self.registers.get(&id.0).copied().unwrap_or(Register {
            raw: id.0,
            canonical: None,
            width: 8,
        })
    }
    fn optional_register(&self, id: RegId) -> Option<Register> {
        (id.0 != 0).then(|| self.register_info(id))
    }
    pub fn decode(&self, address: u64) -> Option<Instruction> {
        if !self.image.executable(address) {
            return None;
        }
        let bytes = self.image.at(address)?;
        let decoded = self
            .capstone
            .disasm_count(&bytes[..bytes.len().min(16)], address, 1)
            .ok()?;
        self.normalize(decoded.iter().next()?)
    }
    fn normalize(&self, instruction: &capstone::Insn<'_>) -> Option<Instruction> {
        let details = self.capstone.insn_detail(instruction).ok()?;
        let arch = details.arch_detail();
        let mut out = Instruction {
            address: instruction.address(),
            size: instruction.bytes().len() as u64,
            mnemonic: instruction.mnemonic()?.into(),
            operands_text: instruction.op_str().unwrap_or("").into(),
            operands: Vec::new(),
            writes: details
                .regs_write()
                .iter()
                .map(|r| self.register_info(*r))
                .collect(),
            flags_written: details
                .regs_write()
                .iter()
                .any(|r| self.flag_registers.contains(&r.0)),
            writeback: false,
            post_index: false,
            condition: Cond::Other,
            conditional: false,
        };
        match self.image.arch {
            Arch::X86_64 => {
                for operand in arch.x86()?.operands() {
                    let value = match operand.op_type {
                        x86::X86OperandType::Reg(reg) => Value::Register(self.register_info(reg)),
                        x86::X86OperandType::Imm(value) => Value::Immediate(value),
                        x86::X86OperandType::Mem(mem) => Value::Memory(Memory {
                            base: self.optional_register(mem.base()),
                            index: self.optional_register(mem.index()),
                            scale: mem.scale() as u64,
                            displacement: mem.disp(),
                            segmented: mem.segment().0 != 0,
                        }),
                        _ => Value::Other,
                    };
                    out.operands.push(Operand {
                        value,
                        width: operand.size,
                        write: operand.access.is_some_and(|a| a.is_writable()),
                        shift: Shift::None,
                    });
                }
            }
            Arch::Aarch64 => {
                let arm = arch.arm64()?;
                out.writeback = arm.writeback();
                out.flags_written |= arm.update_flags();
                out.condition = match arm.cc() {
                    arm64::Arm64CC::ARM64_CC_EQ => Cond::Eq,
                    arm64::Arm64CC::ARM64_CC_NE => Cond::Ne,
                    _ => Cond::Other,
                };
                out.conditional = !matches!(
                    arm.cc(),
                    arm64::Arm64CC::ARM64_CC_INVALID
                        | arm64::Arm64CC::ARM64_CC_AL
                        | arm64::Arm64CC::ARM64_CC_NV
                );
                // capstone-rust 0.14 does not expose the post_index field. The
                // address mode is part of the decoded AArch64 instruction word.
                // Only supported single/pair loads and stores use this value.
                let word = u32::from_le_bytes(instruction.bytes().try_into().ok()?);
                out.post_index = out.writeback
                    && if matches!(
                        out.mnemonic.as_str(),
                        "ldp" | "ldnp" | "ldpsw" | "stp" | "stnp"
                    ) {
                        (word >> 23) & 3 == 1
                    } else {
                        (word >> 10) & 3 == 1
                    };
                for operand in arm.operands() {
                    let shift = match operand.shift {
                        arm64::Arm64Shift::Invalid => Shift::None,
                        arm64::Arm64Shift::Lsl(n) => Shift::Left(n),
                        _ => Shift::Other,
                    };
                    let value = match operand.op_type {
                        arm64::Arm64OperandType::Reg(reg) => {
                            Value::Register(self.register_info(reg))
                        }
                        arm64::Arm64OperandType::Imm(value)
                        | arm64::Arm64OperandType::Cimm(value) => Value::Immediate(value),
                        arm64::Arm64OperandType::Mem(mem) => Value::Memory(Memory {
                            base: self.optional_register(mem.base()),
                            index: self.optional_register(mem.index()),
                            scale: 1,
                            displacement: mem.disp() as i64,
                            segmented: false,
                        }),
                        _ => Value::Other,
                    };
                    let width = match value {
                        Value::Register(reg) => reg.width,
                        _ => 8,
                    };
                    out.operands.push(Operand {
                        value,
                        width,
                        write: operand.access.is_some_and(|a| a.is_writable()),
                        shift,
                    });
                }
            }
        }
        Some(out)
    }
    pub fn for_each(&self, start: u64, end: u64, mut visit: impl FnMut(&Instruction) -> bool) {
        if end <= start || !self.image.executable(start) {
            return;
        }
        let Some(bytes) = self.image.at(start) else {
            return;
        };
        let length = bytes
            .len()
            .min(usize::try_from(end - start).unwrap_or(usize::MAX));
        // Decode a function at a time to keep memory bounded by its size.
        let Ok(instructions) = self.capstone.disasm_all(&bytes[..length], start) else {
            return;
        };
        for raw in instructions.iter() {
            let Some(instruction) = self.normalize(raw) else {
                return;
            };
            if !visit(&instruction) {
                return;
            }
        }
    }
    pub fn eval_cfg(
        &mut self,
        entry: &State,
        start: u64,
        end: u64,
        mut visit: impl FnMut(&mut Self, &Instruction, Flow, &mut State) -> bool,
    ) {
        let mut pending = BTreeMap::from([(start, entry.clone())]);
        let mut visited = BTreeSet::new();
        while let Some((mut pc, mut state)) = pending.pop_first() {
            while pc >= start && pc < end && visited.insert(pc) {
                let Some(instruction) = self.decode(pc) else {
                    break;
                };
                let flow = self.step(&mut state, &instruction);
                if !visit(self, &instruction, flow, &mut state) {
                    return;
                }
                if flow.kind == FlowKind::CondJump
                    && flow.target >= start
                    && flow.target < end
                    && !visited.contains(&flow.target)
                {
                    pending.entry(flow.target).or_insert_with(|| state.clone());
                }
                match flow.kind {
                    FlowKind::Jump => pc = flow.target,
                    FlowKind::Ret | FlowKind::IndirectJump => break,
                    _ => {
                        let Some(next) = pc.checked_add(instruction.size) else {
                            break;
                        };
                        pc = next;
                    }
                }
            }
        }
    }
    pub fn entry_state(&mut self) -> State {
        State {
            regs: std::array::from_fn(|reg| Expr::atom(self.atoms.init(reg))),
            ..State::default()
        }
    }
    pub fn arg_reg(&self, index: usize) -> usize {
        if self.image.arch == Arch::X86_64 {
            [7, 6, 2, 1, 8, 9][index]
        } else {
            index
        }
    }
    pub fn sp_reg(&self) -> usize {
        if self.image.arch == Arch::X86_64 {
            4
        } else {
            31
        }
    }
    pub fn reg_name(&self, reg: usize) -> String {
        if self.image.arch == Arch::X86_64 {
            const NAMES: [&str; 17] = [
                "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11",
                "r12", "r13", "r14", "r15", "rip",
            ];
            NAMES
                .get(reg)
                .map_or_else(|| format!("r?{reg}"), |n| (*n).into())
        } else {
            match reg {
                31 => "sp".into(),
                32 => "xzr".into(),
                _ => format!("x{reg}"),
            }
        }
    }
    pub fn direct_call(&self, instruction: &Instruction) -> Option<u64> {
        if matches!(instruction.mnemonic.as_str(), "call" | "bl") {
            instruction.immediate(0)
        } else {
            None
        }
    }
    pub fn is_nt_store(&self, instruction: &Instruction) -> bool {
        matches!(
            instruction.mnemonic.as_str(),
            "movntps"
                | "movntpd"
                | "movntdq"
                | "movnti"
                | "movntq"
                | "vmovntps"
                | "vmovntpd"
                | "vmovntdq"
                | "stnp"
        )
    }
    pub fn cmp_small(&self, instruction: &Instruction, low: i64, high: i64) -> bool {
        instruction.mnemonic == "cmp"
            && instruction.operands.len() == 2
            && instruction.operands[1].shift == Shift::None
            && matches!(instruction.operands[1].value, Value::Immediate(n) if n >= low && n <= high)
    }
    pub fn const_write(&self, instruction: &Instruction, reg: usize) -> (bool, Option<u64>) {
        let writes = instruction.writes.iter().any(|r| r.canonical == Some(reg))
            || instruction.operands.iter().any(|o| match &o.value {
                Value::Register(r) => o.write && r.canonical == Some(reg),
                Value::Memory(mem) => {
                    instruction.writeback && mem.base.is_some_and(|r| r.canonical == Some(reg))
                }
                _ => false,
            });
        if let Some(first) = instruction.operands.first() {
            if first.write
                && instruction
                    .register(0)
                    .is_some_and(|r| r.canonical == Some(reg))
            {
                let mnemonic = instruction.mnemonic.as_str();
                if matches!(mnemonic, "mov" | "movz") {
                    if let Some(value) = instruction.immediate(1) {
                        let shift = match instruction.operands[1].shift {
                            Shift::None => 0,
                            Shift::Left(n) if n < 64 => n,
                            _ => return (true, None),
                        };
                        return (true, Some(value << shift));
                    }
                    if self.image.arch == Arch::Aarch64
                        && instruction
                            .register(1)
                            .is_some_and(|r| r.canonical == Some(32))
                    {
                        return (true, Some(0));
                    }
                }
                if mnemonic == "xor"
                    && instruction
                        .register(1)
                        .is_some_and(|r| r.canonical == Some(reg))
                {
                    return (true, Some(0));
                }
                return (true, None);
            }
        }
        (writes, None)
    }
    pub fn plt_symbol(&self, address: u64) -> Option<String> {
        if !self.image.in_plt(address) {
            return None;
        }
        let mut pc = address;
        let mut page = None;
        for _ in 0..6 {
            let instruction = self.decode(pc)?;
            if self.image.arch == Arch::X86_64
                && matches!(instruction.mnemonic.as_str(), "jmp" | "bnd jmp")
            {
                if let Some(Operand {
                    value: Value::Memory(memory),
                    ..
                }) = instruction.operands.first()
                {
                    if memory.base.is_some_and(|r| r.canonical == Some(16)) {
                        let slot = pc
                            .wrapping_add(instruction.size)
                            .wrapping_add(memory.displacement as u64);
                        return self.image.got_symbols.get(&slot).cloned();
                    }
                }
            } else if self.image.arch == Arch::Aarch64 {
                if instruction.mnemonic == "adrp" {
                    page = Some((
                        instruction.register(0)?.canonical?,
                        instruction.immediate(1)?,
                    ));
                } else if instruction.mnemonic == "ldr" {
                    if let (
                        Some((reg, base)),
                        Some(Operand {
                            value: Value::Memory(memory),
                            ..
                        }),
                    ) = (page, instruction.operands.get(1))
                    {
                        if memory.base.is_some_and(|r| r.canonical == Some(reg)) {
                            return self
                                .image
                                .got_symbols
                                .get(&base.wrapping_add(memory.displacement as u64))
                                .cloned();
                        }
                    }
                }
            }
            pc = pc.checked_add(instruction.size)?;
        }
        None
    }
    fn unknown(&mut self, pc: u64, reg: usize) -> Expr {
        Expr::atom(self.atoms.unknown(pc, reg))
    }
    fn bounded(&mut self, value: Expr, pc: u64, reg: usize) -> Expr {
        if value.degree() > 4 || value.0.len() > 128 {
            self.unknown(pc, reg)
        } else {
            value
        }
    }
    fn read_memory(&mut self, state: &State, address: &Expr, width: u8) -> Expr {
        state
            .mem
            .get(&(address.clone(), width))
            .cloned()
            .unwrap_or_else(|| Expr::atom(self.atoms.load(address, width)))
    }
    fn write_memory(state: &mut State, address: Expr, width: u8, value: Expr) {
        if state.log_stores {
            state.store_log.push((address.clone(), width));
        }
        for old_width in [1, 2, 4, 8, 16] {
            state.mem.remove(&(address.clone(), old_width));
        }
        state.mem.insert((address, width), value);
    }
    fn clobber_call(&mut self, state: &mut State, pc: u64) {
        state.call_args = std::array::from_fn(|i| state.regs[self.arg_reg(i)].clone());
        let saved: &[usize] = if self.image.arch == Arch::X86_64 {
            &[0, 1, 2, 6, 7, 8, 9, 10, 11]
        } else {
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 30,
            ]
        };
        for reg in saved {
            state.regs[*reg] = Expr::atom(self.atoms.ret(pc, *reg));
        }
        state.mem.retain(|(address, _), _| {
            address.as_atom_plus_const().is_some_and(
                |(id, _)| matches!(self.atoms.get(id), Atom::Init(reg) if *reg == self.sp_reg()),
            )
        });
        state.cmp.valid = false;
    }
    fn clobber_written(&mut self, state: &mut State, instruction: &Instruction) {
        for register in &instruction.writes {
            if let Some(reg) = register.canonical.filter(|r| *r < state.regs.len()) {
                state.regs[reg] = self.unknown(instruction.address, reg);
            }
        }
        if instruction.flags_written {
            state.cmp.valid = false;
        }
    }
    fn register_value(
        &mut self,
        state: &State,
        instruction: &Instruction,
        register: Register,
    ) -> Expr {
        match register.canonical {
            Some(16) if self.image.arch == Arch::X86_64 => {
                Expr::constant(instruction.address.wrapping_add(instruction.size))
            }
            Some(32) if self.image.arch == Arch::Aarch64 => Expr::default(),
            Some(reg) if reg < state.regs.len() => state.regs[reg].clone(),
            _ => self.unknown(instruction.address, 100 + register.raw as usize),
        }
    }
    fn set_register(
        &mut self,
        state: &mut State,
        instruction: &Instruction,
        operand: &Operand,
        value: Expr,
    ) {
        let Value::Register(register) = operand.value else {
            return;
        };
        let Some(reg) = register.canonical.filter(|r| *r < state.regs.len()) else {
            return;
        };
        if (self.image.arch == Arch::Aarch64 && reg == 32)
            || (self.image.arch == Arch::X86_64 && reg == 16)
        {
            return;
        }
        state.regs[reg] = if self.image.arch == Arch::X86_64 && operand.width < 4 {
            self.unknown(instruction.address, reg)
        } else {
            self.bounded(value, instruction.address, reg)
        };
    }
    fn x86_address(&mut self, state: &State, instruction: &Instruction, memory: &Memory) -> Expr {
        if memory.segmented {
            return self.unknown(instruction.address, 98);
        }
        let mut address = Expr::constant(memory.displacement as u64);
        if let Some(base) = memory.base {
            address = address.add(&self.register_value(state, instruction, base));
        }
        if let Some(index) = memory.index {
            address = address.add(
                &self
                    .register_value(state, instruction, index)
                    .scaled(memory.scale),
            );
        }
        address
    }
    fn x86_value(&mut self, state: &State, instruction: &Instruction, operand: &Operand) -> Expr {
        match &operand.value {
            Value::Register(reg) => self.register_value(state, instruction, *reg),
            Value::Immediate(value) => Expr::constant(*value as u64),
            Value::Memory(memory) => {
                let address = self.x86_address(state, instruction, memory);
                self.read_memory(state, &address, operand.width)
            }
            Value::Other => self.unknown(instruction.address, 97),
        }
    }
    fn arm_value(&mut self, state: &State, instruction: &Instruction, operand: &Operand) -> Expr {
        let value = match operand.value {
            Value::Register(reg) => self.register_value(state, instruction, reg),
            Value::Immediate(value) => Expr::constant(value as u64),
            _ => return self.unknown(instruction.address, 96),
        };
        match operand.shift {
            Shift::None => value,
            Shift::Left(n) if n < 64 => value.scaled(1 << n),
            _ => self.unknown(instruction.address, 95),
        }
    }
    pub fn step(&mut self, state: &mut State, instruction: &Instruction) -> Flow {
        match self.image.arch {
            Arch::X86_64 => self.step_x86(state, instruction),
            Arch::Aarch64 => self.step_arm(state, instruction),
        }
    }
    fn step_x86(&mut self, state: &mut State, instruction: &Instruction) -> Flow {
        let operands = &instruction.operands;
        let mnemonic = instruction.mnemonic.as_str();
        let next = Flow::default();
        match mnemonic {
            "nop" | "endbr64" | "endbr32" => return next,
            "mov" | "movabs" | "movzx" | "movsx" | "movsxd" if operands.len() == 2 => {
                let value = self.x86_value(state, instruction, &operands[1]);
                match &operands[0].value {
                    Value::Register(_) => {
                        self.set_register(state, instruction, &operands[0], value)
                    }
                    Value::Memory(memory) => {
                        let address = self.x86_address(state, instruction, memory);
                        Self::write_memory(state, address, operands[0].width, value);
                    }
                    _ => {}
                }
                return next;
            }
            "lea" if operands.len() == 2 => {
                if let Value::Memory(memory) = &operands[1].value {
                    let address = self.x86_address(state, instruction, memory);
                    self.set_register(state, instruction, &operands[0], address);
                    return next;
                }
            }
            "add" | "sub" if operands.len() == 2 => {
                let left = self.x86_value(state, instruction, &operands[0]);
                let right = self.x86_value(state, instruction, &operands[1]);
                let value = if mnemonic == "add" {
                    left.add(&right)
                } else {
                    left.sub(&right)
                };
                match &operands[0].value {
                    Value::Register(_) => {
                        self.set_register(state, instruction, &operands[0], value)
                    }
                    Value::Memory(memory) => {
                        let address = self.x86_address(state, instruction, memory);
                        Self::write_memory(state, address, operands[0].width, value);
                    }
                    _ => {}
                }
                state.cmp.valid = false;
                return next;
            }
            "imul" if matches!(operands.len(), 2 | 3) && instruction.register(0).is_some() => {
                let (left, right) = if operands.len() == 2 { (0, 1) } else { (1, 2) };
                let a = self.x86_value(state, instruction, &operands[left]);
                let b = self.x86_value(state, instruction, &operands[right]);
                self.set_register(state, instruction, &operands[0], a.mul(&b));
                state.cmp.valid = false;
                return next;
            }
            "xor" if operands.len() == 2 => {
                if let (Some(a), Some(b)) = (instruction.register(0), instruction.register(1)) {
                    if a.canonical.is_some() && a.canonical == b.canonical {
                        self.set_register(state, instruction, &operands[0], Expr::default());
                        state.cmp.valid = false;
                        return next;
                    }
                }
            }
            "push" if operands.len() == 1 => {
                state.regs[4] = state.regs[4].sub(&Expr::constant(8));
                let address = state.regs[4].clone();
                let value = self.x86_value(state, instruction, &operands[0]);
                Self::write_memory(state, address, 8, value);
                return next;
            }
            "pop" if operands.len() == 1 => {
                let value = self.read_memory(state, &state.regs[4], 8);
                self.set_register(state, instruction, &operands[0], value);
                state.regs[4] = state.regs[4].add(&Expr::constant(8));
                return next;
            }
            "leave" => {
                state.regs[4] = state.regs[5].clone();
                state.regs[5] = self.read_memory(state, &state.regs[4], 8);
                state.regs[4] = state.regs[4].add(&Expr::constant(8));
                return next;
            }
            "cmp" if operands.len() == 2 => {
                state.cmp = Compare {
                    lhs: self.x86_value(state, instruction, &operands[0]),
                    rhs: self.x86_value(state, instruction, &operands[1]),
                    valid: true,
                };
                return next;
            }
            "test" if operands.len() == 2 => {
                if let (Some(a), Some(b)) = (instruction.register(0), instruction.register(1)) {
                    if a.canonical.is_some() && a.canonical == b.canonical {
                        state.cmp = Compare {
                            lhs: self.x86_value(state, instruction, &operands[0]),
                            rhs: Expr::default(),
                            valid: true,
                        };
                        return next;
                    }
                }
                state.cmp.valid = false;
                return next;
            }
            "call" => {
                self.clobber_call(state, instruction.address);
                return match instruction.immediate(0) {
                    Some(target) => Flow {
                        kind: FlowKind::Call,
                        target,
                        ..next
                    },
                    None => Flow {
                        kind: FlowKind::IndirectCall,
                        ..next
                    },
                };
            }
            "jmp" | "bnd jmp" => {
                return match instruction.immediate(0) {
                    Some(target) => Flow {
                        kind: FlowKind::Jump,
                        target,
                        ..next
                    },
                    None => Flow {
                        kind: FlowKind::IndirectJump,
                        ..next
                    },
                }
            }
            "ret" | "retf" | "ud2" | "hlt" | "int3" => {
                return Flow {
                    kind: FlowKind::Ret,
                    ..next
                }
            }
            "je" | "jne" | "ja" | "jae" | "jb" | "jbe" | "jg" | "jge" | "jl" | "jle" | "jo"
            | "jno" | "jp" | "jnp" | "js" | "jns" | "jrcxz" | "jecxz" => {
                if let Some(target) = instruction.immediate(0) {
                    let cond = match mnemonic {
                        "je" => Cond::Eq,
                        "jne" => Cond::Ne,
                        _ => Cond::Other,
                    };
                    return Flow {
                        kind: FlowKind::CondJump,
                        target,
                        cond,
                    };
                }
            }
            _ => {}
        }
        // Unknown stores still invalidate the old memory value and report their
        // extent to the staging-buffer pattern.
        for operand in operands {
            if let Value::Memory(memory) = &operand.value {
                if operand.write {
                    let address = self.x86_address(state, instruction, memory);
                    let value = self.unknown(instruction.address, 93);
                    Self::write_memory(state, address, operand.width, value);
                }
            }
        }
        self.clobber_written(state, instruction);
        next
    }
    fn step_arm(&mut self, state: &mut State, instruction: &Instruction) -> Flow {
        let operands = &instruction.operands;
        let mnemonic = instruction.mnemonic.as_str();
        let next = Flow::default();
        match mnemonic {
            "nop" | "bti" | "hint" | "paciasp" | "autiasp" => return next,
            "mov" | "movz" if operands.len() == 2 => {
                let value = self.arm_value(state, instruction, &operands[1]);
                self.set_register(state, instruction, &operands[0], value);
                return next;
            }
            "movn" if operands.len() == 2 => {
                if let Some(value) = instruction.immediate(1) {
                    let shift = match operands[1].shift {
                        Shift::None => Some(0),
                        Shift::Left(n) if n < 64 => Some(n),
                        _ => None,
                    };
                    if let Some(shift) = shift {
                        let mut value = !(value << shift);
                        if instruction.register(0).is_some_and(|r| r.width == 4) {
                            value &= 0xffff_ffff;
                        }
                        self.set_register(state, instruction, &operands[0], Expr::constant(value));
                        return next;
                    }
                }
            }
            "movk" if operands.len() == 2 => {
                let old = self
                    .arm_value(state, instruction, &operands[0])
                    .as_constant();
                let shift = match operands[1].shift {
                    Shift::None => Some(0),
                    Shift::Left(n) if n < 64 => Some(n),
                    _ => None,
                };
                if let (Some(old), Some(value), Some(shift)) =
                    (old, instruction.immediate(1), shift)
                {
                    let updated = (old & !(0xffff_u64 << shift)) | (value << shift);
                    self.set_register(state, instruction, &operands[0], Expr::constant(updated));
                    return next;
                }
            }
            "add" | "sub" | "adds" | "subs" if operands.len() == 3 => {
                let left = self.arm_value(state, instruction, &operands[1]);
                let right = self.arm_value(state, instruction, &operands[2]);
                let value = if matches!(mnemonic, "add" | "adds") {
                    left.add(&right)
                } else {
                    left.sub(&right)
                };
                self.set_register(state, instruction, &operands[0], value);
                if mnemonic == "subs" {
                    state.cmp = Compare {
                        lhs: left,
                        rhs: right,
                        valid: true,
                    };
                } else if mnemonic == "adds" {
                    state.cmp.valid = false;
                }
                return next;
            }
            "neg" if operands.len() == 2 => {
                let value = self
                    .arm_value(state, instruction, &operands[1])
                    .scaled(u64::MAX);
                self.set_register(state, instruction, &operands[0], value);
                return next;
            }
            "cmp" | "cmn" if operands.len() == 2 => {
                let lhs = self.arm_value(state, instruction, &operands[0]);
                let value = self.arm_value(state, instruction, &operands[1]);
                state.cmp = Compare {
                    lhs,
                    rhs: if mnemonic == "cmn" {
                        value.scaled(u64::MAX)
                    } else {
                        value
                    },
                    valid: true,
                };
                return next;
            }
            "mul" | "umull" | "smull" if operands.len() == 3 => {
                let left = self.arm_value(state, instruction, &operands[1]);
                let right = self.arm_value(state, instruction, &operands[2]);
                self.set_register(state, instruction, &operands[0], left.mul(&right));
                return next;
            }
            "madd" | "umaddl" | "smaddl" | "msub" if operands.len() == 4 => {
                let left = self.arm_value(state, instruction, &operands[1]);
                let right = self.arm_value(state, instruction, &operands[2]);
                let addend = self.arm_value(state, instruction, &operands[3]);
                let product = left.mul(&right);
                let value = if mnemonic == "msub" {
                    addend.sub(&product)
                } else {
                    product.add(&addend)
                };
                self.set_register(state, instruction, &operands[0], value);
                return next;
            }
            "lsl" if operands.len() == 3 => {
                if let Some(shift @ 0..=63) = instruction.immediate(2) {
                    let value = self
                        .arm_value(state, instruction, &operands[1])
                        .scaled(1 << shift);
                    self.set_register(state, instruction, &operands[0], value);
                    return next;
                }
            }
            "adrp" | "adr" if operands.len() == 2 => {
                if let Some(value) = instruction.immediate(1) {
                    self.set_register(state, instruction, &operands[0], Expr::constant(value));
                    return next;
                }
            }
            "cbz" | "cbnz" if operands.len() == 2 => {
                if let Some(target) = instruction.immediate(1) {
                    state.cmp = Compare {
                        lhs: self.arm_value(state, instruction, &operands[0]),
                        rhs: Expr::default(),
                        valid: true,
                    };
                    return Flow {
                        kind: FlowKind::CondJump,
                        target,
                        cond: if mnemonic == "cbz" {
                            Cond::Eq
                        } else {
                            Cond::Ne
                        },
                    };
                }
            }
            "tbz" | "tbnz" if operands.len() == 3 => {
                if let Some(target) = instruction.immediate(2) {
                    return Flow {
                        kind: FlowKind::CondJump,
                        target,
                        ..next
                    };
                }
            }
            "b" | "b.eq" | "b.ne" | "b.hs" | "b.lo" | "b.mi" | "b.pl" | "b.vs" | "b.vc"
            | "b.hi" | "b.ls" | "b.ge" | "b.lt" | "b.gt" | "b.le" | "b.al" | "b.nv" => {
                if let Some(target) = instruction.immediate(0) {
                    return Flow {
                        kind: if instruction.conditional {
                            FlowKind::CondJump
                        } else {
                            FlowKind::Jump
                        },
                        target,
                        cond: instruction.condition,
                    };
                }
            }
            "bl" | "blr" | "blraa" | "blraaz" | "blrab" | "blrabz" => {
                self.clobber_call(state, instruction.address);
                return if mnemonic == "bl" {
                    instruction.immediate(0).map_or(
                        Flow {
                            kind: FlowKind::IndirectCall,
                            ..next
                        },
                        |target| Flow {
                            kind: FlowKind::Call,
                            target,
                            ..next
                        },
                    )
                } else {
                    Flow {
                        kind: FlowKind::IndirectCall,
                        ..next
                    }
                };
            }
            "br" | "braa" | "braaz" | "brab" | "brabz" => {
                return Flow {
                    kind: FlowKind::IndirectJump,
                    ..next
                }
            }
            "ret" | "retaa" | "retab" | "brk" | "hlt" => {
                return Flow {
                    kind: FlowKind::Ret,
                    ..next
                }
            }
            _ => {}
        }
        if !self.arm_load_store(state, instruction) {
            self.clobber_written(state, instruction);
        }
        next
    }
    fn arm_load_store(&mut self, state: &mut State, instruction: &Instruction) -> bool {
        let mnemonic = instruction.mnemonic.as_str();
        let single_load = matches!(
            mnemonic,
            "ldr"
                | "ldur"
                | "ldrb"
                | "ldurb"
                | "ldrh"
                | "ldurh"
                | "ldrsb"
                | "ldursb"
                | "ldrsh"
                | "ldursh"
                | "ldrsw"
                | "ldursw"
                | "ldar"
                | "ldarb"
                | "ldarh"
                | "ldapr"
                | "ldaprb"
                | "ldaprh"
        );
        let pair_load = matches!(mnemonic, "ldp" | "ldnp" | "ldpsw");
        let single_store = matches!(
            mnemonic,
            "str" | "stur" | "strb" | "sturb" | "strh" | "sturh" | "stlr" | "stlrb" | "stlrh"
        );
        let pair_store = matches!(mnemonic, "stp" | "stnp");
        if !(single_load || pair_load || single_store || pair_store) {
            return false;
        }
        let count = if pair_load || pair_store { 2 } else { 1 };
        let Some(memory) = instruction.operands.get(count) else {
            return false;
        };
        let mut update_base = None;
        let address = match &memory.value {
            Value::Immediate(value) if single_load => Expr::constant(*value as u64),
            Value::Memory(memory_value) => {
                let Some(base_register) = memory_value.base else {
                    return false;
                };
                let base = self.register_value(state, instruction, base_register);
                let mut index = memory_value
                    .index
                    .map(|r| self.register_value(state, instruction, r))
                    .unwrap_or_default();
                match memory.shift {
                    Shift::Left(n) if n < 64 => index = index.scaled(1 << n),
                    Shift::None => {}
                    _ => return false,
                }
                let displacement = Expr::constant(memory_value.displacement as u64);
                let address = if instruction.writeback && instruction.post_index {
                    base.add(&index)
                } else {
                    base.add(&index).add(&displacement)
                };
                if instruction.writeback {
                    let updated = if instruction.post_index {
                        base.add(
                            &instruction
                                .immediate(count + 1)
                                .map(Expr::constant)
                                .unwrap_or(displacement),
                        )
                    } else {
                        address.clone()
                    };
                    if let Some(reg) = base_register.canonical.filter(|r| *r < 34 && *r != 32) {
                        update_base = Some((reg, updated));
                    }
                }
                address
            }
            _ => return false,
        };
        for i in 0..count {
            let Some(register) = instruction.register(i) else {
                return false;
            };
            let width = if matches!(mnemonic, "ldrsw" | "ldursw" | "ldpsw") {
                4
            } else if !matches!(mnemonic, "ldp" | "stp") && mnemonic.ends_with('b') {
                1
            } else if !matches!(mnemonic, "ldp" | "stp") && mnemonic.ends_with('h') {
                2
            } else {
                register.width
            };
            let at = address.add(&Expr::constant(i as u64 * width as u64));
            if single_load || pair_load {
                if let Some(reg) = register.canonical.filter(|r| *r < 34 && *r != 32) {
                    state.regs[reg] = self.read_memory(state, &at, width);
                }
            } else {
                let value = if register.canonical.is_some() {
                    self.register_value(state, instruction, register)
                } else {
                    self.unknown(instruction.address, 94)
                };
                Self::write_memory(state, at, width, value);
            }
        }
        if let Some((reg, value)) = update_base {
            state.regs[reg] = value;
        }
        true
    }
}

fn register_info(arch: Arch, raw: u16, name: &str) -> Register {
    if arch == Arch::X86_64 {
        const FAMILIES: [[&str; 5]; 16] = [
            ["rax", "eax", "ax", "al", "ah"],
            ["rcx", "ecx", "cx", "cl", "ch"],
            ["rdx", "edx", "dx", "dl", "dh"],
            ["rbx", "ebx", "bx", "bl", "bh"],
            ["rsp", "esp", "sp", "spl", ""],
            ["rbp", "ebp", "bp", "bpl", ""],
            ["rsi", "esi", "si", "sil", ""],
            ["rdi", "edi", "di", "dil", ""],
            ["r8", "r8d", "r8w", "r8b", ""],
            ["r9", "r9d", "r9w", "r9b", ""],
            ["r10", "r10d", "r10w", "r10b", ""],
            ["r11", "r11d", "r11w", "r11b", ""],
            ["r12", "r12d", "r12w", "r12b", ""],
            ["r13", "r13d", "r13w", "r13b", ""],
            ["r14", "r14d", "r14w", "r14b", ""],
            ["r15", "r15d", "r15w", "r15b", ""],
        ];
        if name == "rip" {
            return Register {
                raw,
                canonical: Some(16),
                width: 8,
            };
        }
        for (canonical, names) in FAMILIES.iter().enumerate() {
            if let Some(width_index) = names.iter().position(|n| !n.is_empty() && *n == name) {
                return Register {
                    raw,
                    canonical: Some(canonical),
                    width: [8, 4, 2, 1, 1][width_index],
                };
            }
        }
        Register {
            raw,
            canonical: None,
            width: if name.starts_with("xmm") {
                16
            } else if name.starts_with("ymm") {
                32
            } else if name.starts_with("zmm") {
                64
            } else {
                8
            },
        }
    } else {
        let canonical = match name {
            "sp" | "wsp" => Some(31),
            "xzr" | "wzr" => Some(32),
            "fp" => Some(29),
            "lr" => Some(30),
            _ => name
                .strip_prefix('x')
                .or_else(|| name.strip_prefix('w'))
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|n| *n <= 30),
        };
        let width = match name.as_bytes().first() {
            Some(b'w' | b's') => 4,
            Some(b'q') => 16,
            Some(b'h') => 2,
            Some(b'b') => 1,
            _ => 8,
        };
        Register {
            raw,
            canonical,
            width,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Function, Segment};

    fn image(arch: Arch, bytes: &[u8]) -> Image {
        Image {
            arch,
            version: "synthetic".into(),
            image_base: 0,
            bytes: bytes.to_vec(),
            segments: vec![Segment {
                vaddr: 0x1000,
                file_offset: 0,
                file_size: bytes.len() as u64,
                executable: true,
            }],
            functions: vec![Function {
                start: 0x1000,
                size: bytes.len() as u64,
            }],
            plt: Vec::new(),
            got_symbols: BTreeMap::new(),
        }
    }

    fn run_all(decoder: &mut Decoder<'_>, state: &mut State) {
        let end = decoder.image.functions[0].end();
        let mut pc = 0x1000;
        while pc < end {
            let instruction = decoder.decode(pc).expect("fixture instruction must decode");
            decoder.step(state, &instruction);
            pc += instruction.size;
        }
        assert_eq!(pc, end);
    }

    #[test]
    fn polynomial_operations_preserve_machine_arithmetic() {
        let a = Expr::atom(0);
        let b = Expr::atom(1);
        let c = Expr::atom(2);
        assert_eq!(a.mul(&b).mul(&c), c.mul(&a.mul(&b)));
        assert_eq!(a.add(&b).sub(&a), b);
        assert_eq!(
            Expr::constant(u64::MAX).add(&Expr::constant(1)),
            Expr::default()
        );
        assert_eq!(
            a.add(&Expr::constant(16)).as_atom_plus_const(),
            Some((0, 16))
        );
        assert_eq!(a.mul(&b).as_atom_plus_const(), None);
    }

    #[test]
    fn field_matching_checks_width_and_indirection() {
        let mut atoms = Atoms::default();
        let root = atoms.init(7);
        let field = atoms.load(&Expr::atom(root).add(&Expr::constant(24)), 8);
        let nested = atoms.load(&Expr::atom(field).add(&Expr::constant(16)), 8);
        assert_eq!(atoms.field_of(field, root, 8), Some(24));
        assert_eq!(atoms.field_of(field, root, 4), None);
        assert_eq!(atoms.deref_of(nested, root), Some((24, 16)));
        assert_eq!(atoms.field_of(usize::MAX, root, 8), None);
    }

    // These byte sequences and expected expressions also appear in the original
    // C++ scanner's independent unit tests. They exercise effects, not offsets.
    #[test]
    fn x86_stack_and_copy_address_arithmetic() {
        let image = image(
            Arch::X86_64,
            &[
                0x55, 0x48, 0x89, 0xe5, 0x48, 0x89, 0x7d, 0xb0, 0x48, 0x8b, 0x45, 0xb0, 0xc9, 0x48,
                0x0f, 0xaf, 0x47, 0x10, 0x48, 0x8d, 0x4c, 0xd0, 0x20,
            ],
        );
        let mut decoder = Decoder::new(&image).unwrap();
        let mut state = decoder.entry_state();
        let entry = state.clone();
        run_all(&mut decoder, &mut state);
        let field = Expr::atom(
            decoder
                .atoms
                .load(&entry.regs[7].add(&Expr::constant(16)), 8),
        );
        let product = entry.regs[7].mul(&field);
        assert_eq!(state.regs[0], product);
        assert_eq!(state.regs[4], entry.regs[4]);
        assert_eq!(state.regs[5], entry.regs[5]);
        assert_eq!(
            state.regs[1],
            product
                .add(&entry.regs[2].scaled(8))
                .add(&Expr::constant(32))
        );
    }

    #[test]
    fn arm_stack_writeback_and_copy_address_arithmetic() {
        let image = image(
            Arch::Aarch64,
            &[
                0xfd, 0x7b, 0xb9, 0xa9, 0xfd, 0x7b, 0xc7, 0xa8, 0xbf, 0x86, 0x00, 0xf8, 0x41, 0x0c,
                0x41, 0xf8, 0x81, 0x0b, 0x01, 0x9b, 0x00, 0x20, 0x40, 0x91, 0x83, 0x02, 0xc0, 0xd2,
            ],
        );
        let mut decoder = Decoder::new(&image).unwrap();
        let mut state = decoder.entry_state();
        let entry = state.clone();
        run_all(&mut decoder, &mut state);
        for reg in [29, 30, 31] {
            assert_eq!(state.regs[reg], entry.regs[reg]);
        }
        assert_eq!(state.regs[21], entry.regs[21].add(&Expr::constant(8)));
        assert_eq!(
            state.mem.get(&(entry.regs[21].clone(), 8)),
            Some(&Expr::default())
        );
        let updated_base = entry.regs[2].add(&Expr::constant(16));
        let loaded = Expr::atom(decoder.atoms.load(&updated_base, 8));
        assert_eq!(state.regs[2], updated_base);
        assert_eq!(
            state.regs[1],
            entry.regs[28].mul(&loaded).add(&updated_base)
        );
        assert_eq!(state.regs[0], entry.regs[0].add(&Expr::constant(0x8000)));
        assert_eq!(state.regs[3].as_constant(), Some(0x14_u64 << 32));
    }

    #[test]
    fn unsupported_writes_do_not_retain_old_facts() {
        // cmp rax, 1; rol rax, 1; mov al, 7
        let image = image(
            Arch::X86_64,
            &[0x48, 0x83, 0xf8, 1, 0x48, 0xd1, 0xc0, 0xb0, 7],
        );
        let mut decoder = Decoder::new(&image).unwrap();
        let mut state = decoder.entry_state();
        decoder.step(&mut state, &decoder.decode(0x1000).unwrap());
        assert!(state.cmp.valid);
        decoder.step(&mut state, &decoder.decode(0x1004).unwrap());
        assert!(!state.cmp.valid);
        assert!(matches!(
            decoder.atoms.get(state.regs[0].as_atom().unwrap()),
            Atom::Unknown { pc: 0x1004, reg: 0 }
        ));
        decoder.step(&mut state, &decoder.decode(0x1007).unwrap());
        assert!(matches!(
            decoder.atoms.get(state.regs[0].as_atom().unwrap()),
            Atom::Unknown { pc: 0x1007, reg: 0 }
        ));
    }

    #[test]
    fn calls_keep_stack_values_and_capture_arguments() {
        let image = image(Arch::X86_64, &[0xe8, 0, 0, 0, 0]);
        let mut decoder = Decoder::new(&image).unwrap();
        let mut state = decoder.entry_state();
        let entry = state.clone();
        let stack = entry.regs[4].sub(&Expr::constant(16));
        state.mem.insert((stack.clone(), 8), Expr::constant(7));
        state
            .mem
            .insert((entry.regs[7].clone(), 8), Expr::constant(9));
        let instruction = decoder.decode(0x1000).unwrap();
        assert_eq!(decoder.step(&mut state, &instruction).kind, FlowKind::Call);
        assert_eq!(state.call_args[0], entry.regs[7]);
        assert_eq!(state.mem.len(), 1);
        assert_eq!(state.mem.get(&(stack, 8)), Some(&Expr::constant(7)));
        assert_eq!(state.regs[3], entry.regs[3]);
        assert!(matches!(
            decoder.atoms.get(state.regs[0].as_atom().unwrap()),
            Atom::Return { pc: 0x1000, reg: 0 }
        ));
    }

    #[test]
    fn truncated_or_nonexecutable_instructions_are_rejected() {
        for arch in [Arch::X86_64, Arch::Aarch64] {
            let mut image = image(arch, &[0x0f]);
            assert!(Decoder::new(&image).unwrap().decode(0x1000).is_none());
            image.segments[0].executable = false;
            assert!(Decoder::new(&image).unwrap().decode(0x1000).is_none());
        }
    }
}
