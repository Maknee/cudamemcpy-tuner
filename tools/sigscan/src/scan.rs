//! Three function signatures shared by x86-64 and AArch64, plus the
//! x86-64 inline threshold gate implemented in `inline`.
//!
//! The copy engine computes `base + x + y*pitch + z*height*pitch` for each
//! operand type. We recover the fields from that calculation, then require the
//! source/destination layouts to agree. Registers and instruction scheduling
//! may change without changing this signature. A missing or ambiguous match
//! produces no usable patch offset.

use crate::{
    drivers,
    image::{Arch, Image},
    inline::{self, InlineCopy},
    instructions::{AtomId, Atoms, Cond, Decoder, Expr, Flow, FlowKind, State},
};
use anyhow::{bail, ensure, Context, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write,
    path::Path,
};

const WALK_LIMIT: usize = 400;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Found,
    Absent,
    NotFound,
    Ambiguous,
}
impl Status {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Found => "found",
            Self::Absent => "absent",
            Self::NotFound => "not found",
            Self::Ambiguous => "ambiguous",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Site {
    pub status: Status,
    pub address: u64,
    pub offset: u64,
    pub prologue: Vec<u8>,
    pub reason: String,
    pub candidates: Vec<u64>,
    pub evidence: Vec<String>,
}
impl Site {
    pub(crate) fn missing(reason: impl Into<String>) -> Self {
        Self {
            status: Status::NotFound,
            address: 0,
            offset: 0,
            prologue: Vec::new(),
            reason: reason.into(),
            candidates: Vec::new(),
            evidence: Vec::new(),
        }
    }
    fn finish(&mut self, image: &Image) {
        if self.status != Status::Found {
            self.address = 0;
            return;
        }
        let offset = self.address.checked_sub(image.image_base);
        let bytes = image.at(self.address).and_then(|b| b.get(..16));
        match (offset, bytes) {
            (Some(offset), Some(bytes)) if u32::try_from(offset).is_ok() => {
                self.offset = offset;
                self.prologue = bytes.to_vec();
            }
            _ => {
                self.status = Status::NotFound;
                self.reason =
                    "site has no 16-byte prologue or offset does not fit the hook's u32".into();
                self.address = 0;
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Descriptor {
    pub op_stride: u64,
    pub op_kind: u64,
    pub op_some_pointer1: u64,
    pub op_some_pointer1_off: u64,
    pub op_some_pointer1_width: u64,
    pub op_some_pointer1_height: u64,
    pub op_some_pointer2: u64,
    pub op_some_pointer2_width: u64,
    pub op_some_pointer2_height: u64,
    pub op_some_pointer3: u64,
    pub op_some_pointer3_width: u64,
    pub op_some_pointer3_height: u64,
    pub op_unkx: u64,
    pub op_unky: u64,
    pub op_unkz: u64,
    pub copying_width_bytes: u64,
    pub copying_height_bytes: u64,
    pub copying_depth_bytes: u64,
    pub op_some_pointer1_address: u64,
    pub op_kind_1: u64,
    pub op_kind_2: u64,
    pub op_kind_3: u64,
}
impl Descriptor {
    pub fn fields(&self) -> [(&'static str, u64); 19] {
        [
            ("op_stride", self.op_stride),
            ("op_kind", self.op_kind),
            ("op_some_pointer1", self.op_some_pointer1),
            ("op_some_pointer1_off", self.op_some_pointer1_off),
            ("op_some_pointer1_width", self.op_some_pointer1_width),
            ("op_some_pointer1_height", self.op_some_pointer1_height),
            ("op_some_pointer2", self.op_some_pointer2),
            ("op_some_pointer2_width", self.op_some_pointer2_width),
            ("op_some_pointer2_height", self.op_some_pointer2_height),
            ("op_some_pointer3", self.op_some_pointer3),
            ("op_some_pointer3_width", self.op_some_pointer3_width),
            ("op_some_pointer3_height", self.op_some_pointer3_height),
            ("op_unkx", self.op_unkx),
            ("op_unky", self.op_unky),
            ("op_unkz", self.op_unkz),
            ("copying_width_bytes", self.copying_width_bytes),
            ("copying_height_bytes", self.copying_height_bytes),
            ("copying_depth_bytes", self.copying_depth_bytes),
            ("op_some_pointer1_address", self.op_some_pointer1_address),
        ]
    }
}

pub struct Scan {
    pub arch: Arch,
    pub version: String,
    pub image_base: u64,
    pub engine: Site,
    pub row_copy: Site,
    pub staging: Site,
    pub inline: InlineCopy,
    pub descriptor: Descriptor,
    pub function_count: usize,
    pub instruction_count: usize,
}
impl Scan {
    pub fn complete(&self) -> bool {
        [&self.engine, &self.row_copy, &self.staging]
            .into_iter()
            .all(|s| matches!(s.status, Status::Found | Status::Absent))
            && self.inline.complete()
    }
    pub fn explain(&self) -> String {
        let mut out = format!(
            "== {} libcuda {} (base {:#x})\n{} functions; {} decoded instructions\n",
            self.arch, self.version, self.image_base, self.function_count, self.instruction_count
        );
        for (name, site) in [
            ("copy engine", &self.engine),
            ("row copy", &self.row_copy),
            ("staging initializer", &self.staging),
        ] {
            writeln!(
                out,
                "-- {name}: {} at {:#x}, offset {:#x}: {}",
                site.status.name(),
                site.address,
                site.offset,
                site.reason
            )
            .unwrap();
            for address in &site.candidates {
                writeln!(out, "   candidate {address:#x}").unwrap();
            }
            for evidence in &site.evidence {
                writeln!(out, "   {evidence}").unwrap();
            }
        }
        if self.engine.status == Status::Found {
            out.push_str("-- descriptor fields recovered from the instruction paths:\n");
            for (name, offset) in self.descriptor.fields() {
                writeln!(out, "   {name}: {offset:#x}").unwrap();
            }
        }
        out.push_str(&self.inline.explain());
        out
    }
}

pub fn file(path: &Path, arch: Option<Arch>, explain: bool) -> Result<Scan> {
    let image = Image::load(path)?;
    ensure!(
        arch.is_none_or(|arch| arch == image.arch),
        "{}: architecture differs from requested table",
        path.display()
    );
    if let Some(version) = path
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("libcuda.so."))
    {
        // libcuda.so.1 is the ordinary SONAME symlink, not a full driver version.
        ensure!(
            !drivers::valid_version(version) || version == image.version,
            "{}: filename version differs from binary banner {}",
            path.display(),
            image.version
        );
    }
    let scan = scan(&image).with_context(|| format!("scan {}", path.display()))?;
    if explain || !scan.complete() {
        eprint!("{}", scan.explain());
    }
    ensure!(
        scan.complete(),
        "{}: a required signature is missing or ambiguous; no table emitted",
        path.display()
    );
    Ok(scan)
}

struct Call {
    target: u64,
    direction_constant: Option<u64>,
}
struct FunctionFacts {
    start: u64,
    end: u64,
    calls: Vec<Call>,
    type_comparisons: usize,
    nt_stores: usize,
}
struct Prepass {
    functions: Vec<FunctionFacts>,
    by_start: BTreeMap<u64, usize>,
    instructions: usize,
}

fn prepass(decoder: &Decoder<'_>) -> Prepass {
    let mut result = Prepass {
        functions: Vec::new(),
        by_start: BTreeMap::new(),
        instructions: 0,
    };
    for function in &decoder.image.functions {
        if !decoder.image.executable(function.start) {
            continue;
        }
        let mut facts = FunctionFacts {
            start: function.start,
            end: function.end(),
            calls: Vec::new(),
            type_comparisons: 0,
            nt_stores: 0,
        };
        let mut direction = None;
        decoder.for_each(function.start, function.end(), |instruction| {
            result.instructions += 1;
            if let Some(target) = decoder.direct_call(instruction) {
                facts.calls.push(Call {
                    target,
                    direction_constant: direction,
                });
                direction = None;
                return true;
            }
            facts.type_comparisons += usize::from(decoder.cmp_small(instruction, 1, 3));
            facts.nt_stores += usize::from(decoder.is_nt_store(instruction));
            let (writes, value) = decoder.const_write(instruction, decoder.arg_reg(3));
            if writes {
                direction = value;
            }
            true
        });
        result.by_start.insert(facts.start, result.functions.len());
        result.functions.push(facts);
    }
    result
}

#[derive(Clone)]
struct Dispatch {
    kind: u64,
    target: u64,
    text: String,
    state: State,
}
struct Group {
    field: Option<u64>,
    events: Vec<Dispatch>,
}
fn ends_path(flow: FlowKind) -> bool {
    matches!(flow, FlowKind::Ret | FlowKind::IndirectJump)
}

fn as_dispatch(atoms: &Atoms, descriptor: AtomId, state: &State, flow: Flow) -> Option<(u64, u64)> {
    if flow.kind != FlowKind::CondJump || flow.cond != Cond::Eq || !state.cmp.valid {
        return None;
    }
    let kind = state.cmp.rhs.as_constant().filter(|k| *k <= 15)?;
    let lhs = state.cmp.lhs.as_atom()?;
    Some((atoms.field_of(lhs, descriptor, 4)?, kind))
}

fn collect_group(
    decoder: &mut Decoder<'_>,
    mut state: State,
    mut pc: u64,
    descriptor: AtomId,
    ignore: Option<u64>,
) -> Group {
    let mut group = Group {
        field: None,
        events: Vec::new(),
    };
    for _ in 0..WALK_LIMIT {
        let Some(instruction) = decoder.decode(pc) else {
            break;
        };
        let flow = decoder.step(&mut state, &instruction);
        if let Some((field, kind)) = as_dispatch(&decoder.atoms, descriptor, &state, flow) {
            if Some(field) == ignore {
            } else if group.field.is_none() || group.field == Some(field) {
                group.field = Some(field);
                group.events.push(Dispatch {
                    kind,
                    target: flow.target,
                    text: instruction.text(),
                    state: state.clone(),
                });
            } else {
                break;
            }
        }
        if matches!(flow.kind, FlowKind::Call | FlowKind::IndirectCall) || ends_path(flow.kind) {
            break;
        }
        pc = if flow.kind == FlowKind::Jump {
            flow.target
        } else {
            pc.wrapping_add(instruction.size)
        };
    }
    group
}

fn kinds_123(group: &Group) -> Result<BTreeMap<u64, &Dispatch>> {
    let mut kinds = BTreeMap::new();
    for event in &group.events {
        if event.kind == 0 {
            continue;
        }
        ensure!(
            event.kind <= 3 && kinds.insert(event.kind, event).is_none(),
            "invalid or repeated operand type"
        );
    }
    ensure!(kinds.len() == 3, "no type dispatch {{1,2,3}}");
    Ok(kinds)
}

struct RowCall {
    callee: u64,
    text: String,
    arguments: [Expr; 4],
    zero_tested: BTreeSet<u64>,
}
fn walk_to_call(
    decoder: &mut Decoder<'_>,
    mut state: State,
    mut pc: u64,
    descriptor: AtomId,
) -> Option<RowCall> {
    let mut zero_tested = BTreeSet::new();
    for _ in 0..WALK_LIMIT {
        let instruction = decoder.decode(pc)?;
        let flow = decoder.step(&mut state, &instruction);
        if flow.kind == FlowKind::CondJump
            && state.cmp.valid
            && state.cmp.rhs.as_constant() == Some(0)
        {
            if let Some(field) = state
                .cmp
                .lhs
                .as_atom()
                .and_then(|atom| decoder.atoms.field_of(atom, descriptor, 8))
            {
                zero_tested.insert(field);
            }
        }
        if flow.kind == FlowKind::Call {
            return Some(RowCall {
                callee: flow.target,
                text: instruction.text(),
                arguments: state.call_args,
                zero_tested,
            });
        }
        if flow.kind == FlowKind::IndirectCall || ends_path(flow.kind) {
            return None;
        }
        pc = if flow.kind == FlowKind::Jump {
            flow.target
        } else {
            pc.wrapping_add(instruction.size)
        };
    }
    None
}

struct Payload {
    base: Vec<AtomId>,
    pitch: AtomId,
    height: AtomId,
}
struct Layout {
    unkx: AtomId,
    unky: AtomId,
    unkz: AtomId,
    kinds: BTreeMap<u64, Payload>,
}

fn decompose(addresses: &BTreeMap<u64, Expr>) -> Result<Layout> {
    let mut splits = BTreeMap::new();
    for (&kind, expression) in addresses {
        let mut split: [Vec<AtomId>; 3] = std::array::from_fn(|_| Vec::new());
        for (atoms, &coefficient) in &expression.0 {
            ensure!(
                coefficient == 1,
                "type {kind}: address coefficient is not one"
            );
            match atoms.len() {
                1 => split[0].push(atoms[0]),
                2 | 3 => {
                    ensure!(
                        split[atoms.len() - 1].is_empty(),
                        "type {kind}: repeated address term"
                    );
                    split[atoms.len() - 1] = atoms.clone();
                }
                _ => bail!(
                    "type {kind}: unexpected degree-{} address term",
                    atoms.len()
                ),
            }
        }
        ensure!(
            split[2].len() == 3 && split[1].len() == 2 && split[0].len() >= 2,
            "type {kind}: address is not z*height*pitch + y*pitch + x + base"
        );
        splits.insert(kind, split);
    }
    let common = |index: usize| -> Result<AtomId> {
        let mut sets = splits
            .values()
            .map(|s| s[index].iter().copied().collect::<BTreeSet<_>>());
        let mut shared = sets.next().context("no operand addresses")?;
        for set in sets {
            shared = shared.intersection(&set).copied().collect();
        }
        ensure!(
            shared.len() == 1,
            "x/y/z field is not unique across operand types"
        );
        Ok(*shared.first().unwrap())
    };
    let (unkx, unky, unkz) = (common(0)?, common(1)?, common(2)?);
    let mut kinds = BTreeMap::new();
    for (kind, mut split) in splits {
        let pitch_position = split[1]
            .iter()
            .position(|&a| a == unky)
            .context("missing y field")?;
        split[1].remove(pitch_position);
        let pitch = split[1][0];
        let z = split[2]
            .iter()
            .position(|&a| a == unkz)
            .context("missing z field")?;
        split[2].remove(z);
        let p = split[2]
            .iter()
            .position(|&a| a == pitch)
            .context("z term does not contain pitch")?;
        split[2].remove(p);
        let height = split[2][0];
        let base = split[0].iter().copied().filter(|&a| a != unkx).collect();
        kinds.insert(
            kind,
            Payload {
                base,
                pitch,
                height,
            },
        );
    }
    Ok(Layout {
        unkx,
        unky,
        unkz,
        kinds,
    })
}

#[derive(Default)]
struct OperandKind {
    indirect: bool,
    ptr: u64,
    pointer: u64,
    address: u64,
    pitch: u64,
    height: u64,
}
struct Operand {
    unkx: u64,
    unky: u64,
    unkz: u64,
    kinds: BTreeMap<u64, OperandKind>,
}
fn to_offsets(layout: &Layout, atoms: &Atoms, descriptor: AtomId) -> Result<Operand> {
    let field = |atom| {
        atoms
            .field_of(atom, descriptor, 8)
            .context("value is not an 8-byte descriptor load")
    };
    let mut kinds = BTreeMap::new();
    for (&kind, payload) in &layout.kinds {
        let mut output = OperandKind {
            pitch: field(payload.pitch)?,
            height: field(payload.height)?,
            ..Default::default()
        };
        match payload.base.as_slice() {
            [pointer] => output.ptr = field(*pointer)?,
            [_, _] => {
                let mut offset = None;
                let mut dereference = None;
                for &atom in &payload.base {
                    if let Some(value) = atoms.deref_of(atom, descriptor) {
                        dereference = Some(value);
                    } else if let Some(value) = atoms.field_of(atom, descriptor, 8) {
                        offset = Some(value);
                    }
                }
                output.ptr = offset.context("base has no offset field")?;
                (output.pointer, output.address) =
                    dereference.context("base is not offset + pointer->address")?;
                output.indirect = true;
            }
            _ => bail!("type {kind}: invalid base terms"),
        }
        kinds.insert(kind, output);
    }
    Ok(Operand {
        unkx: field(layout.unkx)?,
        unky: field(layout.unky)?,
        unkz: field(layout.unkz)?,
        kinds,
    })
}

struct Engine {
    site: Site,
    descriptor: Descriptor,
    callee: u64,
}
fn analyze_engine(decoder: &mut Decoder<'_>, entry: u64) -> Result<Engine> {
    let state = decoder.entry_state();
    let descriptor = decoder.atoms.init(decoder.arg_reg(0));
    let source_group = collect_group(decoder, state, entry, descriptor, None);
    let source_arms = kinds_123(&source_group)?;
    let source_field = source_group.field.context("missing source dispatch")?;
    let mut calls = BTreeMap::new();
    let mut destination_field = None;
    for (&source_type, event) in source_arms.iter() {
        let destination_group = collect_group(
            decoder,
            event.state.clone(),
            event.target,
            descriptor,
            Some(source_field),
        );
        let arms = kinds_123(&destination_group).context("no second operand dispatch")?;
        ensure!(
            destination_field.is_none() || destination_field == destination_group.field,
            "source paths reach different destination dispatches"
        );
        destination_field = destination_group.field;
        for (&destination_type, event) in arms.iter() {
            let call = walk_to_call(decoder, event.state.clone(), event.target, descriptor)
                .context("no direct row-copy call")?;
            calls.insert((source_type, destination_type), call);
        }
    }
    let first = calls.values().next().context("no row-copy calls")?;
    let mut sources = BTreeMap::new();
    let mut destinations = BTreeMap::new();
    for (&(source_type, destination_type), call) in &calls {
        ensure!(
            call.callee == first.callee
                && call.arguments[2] == first.arguments[2]
                && call.zero_tested == first.zero_tested,
            "nine type paths disagree on call, size or loop bounds"
        );
        let source = sources
            .entry(source_type)
            .or_insert_with(|| call.arguments[1].clone());
        let destination = destinations
            .entry(destination_type)
            .or_insert_with(|| call.arguments[0].clone());
        ensure!(
            *source == call.arguments[1] && *destination == call.arguments[0],
            "source address depends on destination type, or reverse"
        );
    }
    let source = to_offsets(&decompose(&sources)?, &decoder.atoms, descriptor)?;
    let destination = to_offsets(&decompose(&destinations)?, &decoder.atoms, descriptor)?;
    let stride = destination_field
        .and_then(|f| f.checked_sub(source_field))
        .filter(|&n| n > 0)
        .context("nonpositive operand stride")?;
    ensure!(
        destination.unkx.checked_sub(source.unkx) == Some(stride)
            && destination.unky.checked_sub(source.unky) == Some(stride)
            && destination.unkz.checked_sub(source.unkz) == Some(stride),
        "x/y/z offsets disagree on stride"
    );
    for (kind, s) in &source.kinds {
        let d = &destination.kinds[kind];
        ensure!(
            s.indirect == d.indirect
                && d.ptr.checked_sub(s.ptr) == Some(stride)
                && d.pitch.checked_sub(s.pitch) == Some(stride)
                && d.height.checked_sub(s.height) == Some(stride)
                && (!s.indirect
                    || (d.pointer.checked_sub(s.pointer) == Some(stride)
                        && d.address == s.address)),
            "type {kind}: destination payload is not source plus stride"
        );
    }
    let indirect: Vec<_> = source
        .kinds
        .iter()
        .filter(|(_, k)| k.indirect)
        .map(|(&k, _)| k)
        .collect();
    ensure!(indirect == [1], "indirect operand must be exactly type 1");
    let width = first.arguments[2]
        .as_atom()
        .and_then(|a| decoder.atoms.field_of(a, descriptor, 8))
        .context("copy width is not a descriptor field")?;
    let height = width.checked_add(8).context("height offset overflow")?;
    let depth = width.checked_add(16).context("depth offset overflow")?;
    ensure!(
        first.zero_tested == BTreeSet::from([height, depth]),
        "loop bounds are not width+8 and width+16"
    );
    let (kind_1, kind_2, kind_3) = (&source.kinds[&1], &source.kinds[&2], &source.kinds[&3]);
    let layout = Descriptor {
        op_stride: stride,
        op_kind: source_field,
        op_unkx: source.unkx,
        op_unky: source.unky,
        op_unkz: source.unkz,
        op_some_pointer1: kind_1.pointer,
        op_some_pointer1_off: kind_1.ptr,
        op_some_pointer1_width: kind_1.pitch,
        op_some_pointer1_height: kind_1.height,
        op_some_pointer1_address: kind_1.address,
        op_some_pointer3: kind_2.ptr,
        op_some_pointer3_width: kind_2.pitch,
        op_some_pointer3_height: kind_2.height,
        op_some_pointer2: kind_3.ptr,
        op_some_pointer2_width: kind_3.pitch,
        op_some_pointer2_height: kind_3.height,
        copying_width_bytes: width,
        copying_height_bytes: height,
        copying_depth_bytes: depth,
        op_kind_1: 1,
        op_kind_2: 2,
        op_kind_3: 3,
    };
    let mut site = Site::missing("");
    site.status = Status::Found;
    site.address = entry;
    site.evidence.push(format!(
        "source dispatch at descriptor+{source_field:#x}; destination stride {stride:#x}"
    ));
    for event in source_group.events {
        site.evidence.push(event.text);
    }
    site.evidence.push(format!(
        "all nine paths call {}; width {width:#x}, loop bounds {height:#x}/{depth:#x}",
        first.text
    ));
    for (label, addresses) in [("source", &sources), ("destination", &destinations)] {
        for (kind, address) in addresses {
            let expression =
                decoder
                    .atoms
                    .describe(address, |reg| decoder.reg_name(reg), Some(descriptor));
            site.evidence
                .push(format!("{label} type {kind}: {expression}"));
        }
    }
    site.evidence.push(
        "source/destination x/y/z and all payload fields independently agree on one stride".into(),
    );
    Ok(Engine {
        site,
        descriptor: layout,
        callee: first.callee,
    })
}

fn match_engine(decoder: &mut Decoder<'_>, prepass: &Prepass) -> Engine {
    let mut found = Vec::new();
    let mut tried = 0;
    let mut rejections = Vec::new();
    for function in &prepass.functions {
        if function.type_comparisons < 6 || function.calls.is_empty() {
            continue;
        }
        tried += 1;
        match analyze_engine(decoder, function.start) {
            Ok(engine) => found.push(engine),
            Err(error) => {
                let reason = error.to_string();
                if !reason.starts_with("no type dispatch") {
                    rejections.push(format!("rejected {:#x}: {reason}", function.start));
                }
            }
        }
    }
    if found.len() == 1 {
        let mut engine = found.pop().unwrap();
        engine.site.reason =
            format!("unique among {tried} candidates with two operand dispatches and a call");
        engine.site.evidence.extend(rejections);
        return engine;
    }
    let mut site = Site::missing(format!(
        "none of {tried} candidates has the copy engine's dispatch and address calculation"
    ));
    site.evidence = rejections;
    if !found.is_empty() {
        site.status = Status::Ambiguous;
        site.reason = format!("{} functions match the copy engine", found.len());
        site.candidates = found.iter().map(|e| e.site.address).collect();
    }
    Engine {
        site,
        descriptor: Descriptor::default(),
        callee: 0,
    }
}

fn match_row_copy(decoder: &Decoder<'_>, prepass: &Prepass, engine: &Engine) -> Site {
    if engine.site.status != Status::Found {
        return Site::missing("copy engine was not found uniquely");
    }
    if decoder.image.in_plt(engine.callee) {
        let symbol = decoder.plt_symbol(engine.callee).unwrap_or_default();
        let mut site = Site::missing(format!("copy engine calls unrecognized PLT entry {symbol}"));
        if symbol == "memcpy" {
            site.status = Status::Absent;
            site.reason = "the engine copies rows with memcpy@plt: there is no in-driver copy loop to replace".into();
        }
        return site;
    }
    let Some(&index) = prepass.by_start.get(&engine.callee) else {
        return Site::missing("row-copy callee is not a function entry");
    };
    let stores = prepass.functions[index].nt_stores;
    if stores == 0 {
        return Site::missing("row-copy callee has no non-temporal stores");
    }
    let mut site = Site::missing(format!(
        "copy engine's row-copy callee; {stores} non-temporal stores"
    ));
    site.status = Status::Found;
    site.address = engine.callee;
    site
}

fn staging_arguments(arguments: &[[Expr; 4]]) -> Result<u64> {
    ensure!(arguments.len() == 4, "expected four staging calls");
    for (i, args) in arguments.iter().enumerate() {
        ensure!(
            args[3].as_constant() == Some(u64::from(i >= 2)),
            "directions are not 0,0,1,1"
        );
        ensure!(
            args[0] == arguments[0][0] && args[2] == arguments[0][2],
            "context or size differs between calls"
        );
        ensure!(
            args[1].without_constant() == arguments[0][1].without_constant(),
            "buffers have different bases"
        );
    }
    ensure!(
        arguments[0][0].as_constant().is_none() && !arguments[0][1].without_constant().0.is_empty(),
        "context or buffer base is constant"
    );
    let a0 = arguments[0][1]
        .constant_term()
        .min(arguments[1][1].constant_term());
    let a1 = arguments[0][1]
        .constant_term()
        .max(arguments[1][1].constant_term());
    let b0 = arguments[2][1]
        .constant_term()
        .min(arguments[3][1].constant_term());
    let b1 = arguments[2][1]
        .constant_term()
        .max(arguments[3][1].constant_term());
    let stride = a1 - a0;
    ensure!(
        stride > 0 && stride <= 0x1000 && b1 - b0 == stride,
        "buffers do not have one common stride"
    );
    ensure!(
        a1.checked_add(stride).is_some_and(|end| b0 >= end)
            || b1.checked_add(stride).is_some_and(|end| a0 >= end),
        "buffer pools overlap"
    );
    Ok(stride)
}

fn buffer_extent(
    decoder: &mut Decoder<'_>,
    prepass: &Prepass,
    function: u64,
) -> Option<(u64, String)> {
    let facts = prepass
        .by_start
        .get(&function)
        .map(|&i| &prepass.functions[i])?;
    let mut state = decoder.entry_state();
    state.log_stores = true;
    let buffer = decoder.atoms.init(decoder.arg_reg(1));
    let mut extent: Option<(u64, String)> = None;
    decoder.eval_cfg(
        &state,
        facts.start,
        facts.end,
        |_, instruction, _, state| {
            for (address, width) in &state.store_log {
                if let Some((atom, offset)) = address.as_atom_plus_const() {
                    if atom == buffer && (offset as i64) >= 0 {
                        if let Some(end) = offset.checked_add(u64::from(*width)) {
                            if extent.as_ref().is_none_or(|(maximum, _)| end > *maximum) {
                                extent = Some((end, instruction.text()));
                            }
                        }
                    }
                }
            }
            state.store_log.clear();
            true
        },
    );
    extent
}

fn match_staging(decoder: &mut Decoder<'_>, prepass: &Prepass) -> Site {
    let counts: BTreeMap<_, BTreeMap<u64, usize>> = prepass
        .functions
        .iter()
        .map(|f| {
            let mut counts = BTreeMap::new();
            for call in &f.calls {
                if !decoder.image.in_plt(call.target) {
                    *counts.entry(call.target).or_default() += 1;
                }
            }
            (f.start, counts)
        })
        .collect();
    let mut extents = BTreeMap::new();
    let mut matches: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    let mut pairs = 0;
    for caller in &prepass.functions {
        let direct = &counts[&caller.start];
        let mut effective = direct.clone();
        let mut via: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
        for (&helper, &n) in direct {
            if helper == caller.start {
                continue;
            }
            if let Some(callees) = counts.get(&helper) {
                for (&callee, &m) in callees {
                    if callee != helper && callee != caller.start {
                        *effective.entry(callee).or_default() += n * m;
                        via.entry(callee).or_default().insert(helper);
                    }
                }
            }
        }
        let mut wanted = BTreeSet::new();
        for (&callee, &count) in &effective {
            if count != 4 {
                continue;
            }
            let direct_calls: Vec<_> = caller.calls.iter().filter(|c| c.target == callee).collect();
            if direct_calls.len() == 4
                && direct_calls.iter().enumerate().any(|(i, c)| {
                    c.direction_constant
                        .is_some_and(|value| value != u64::from(i >= 2))
                })
            {
                continue;
            }
            let extent = extents
                .entry(callee)
                .or_insert_with(|| buffer_extent(decoder, prepass, callee));
            if extent.as_ref().is_some_and(|(end, _)| *end <= 0x1000) {
                wanted.insert(callee);
            }
        }
        if wanted.is_empty() {
            continue;
        }
        pairs += wanted.len();
        let helpers: BTreeSet<_> = wanted
            .iter()
            .filter_map(|callee| via.get(callee))
            .flatten()
            .copied()
            .collect();
        let state = decoder.entry_state();
        let mut arguments: BTreeMap<u64, Vec<[Expr; 4]>> = BTreeMap::new();
        let mut texts: BTreeMap<u64, Vec<String>> = BTreeMap::new();
        decoder.eval_cfg(
            &state,
            caller.start,
            caller.end,
            |decoder, instruction, flow, state| {
                if flow.kind != FlowKind::Call {
                    return true;
                }
                if wanted.contains(&flow.target) {
                    arguments
                        .entry(flow.target)
                        .or_default()
                        .push(state.call_args.clone());
                    texts
                        .entry(flow.target)
                        .or_default()
                        .push(instruction.text());
                }
                if helpers.contains(&flow.target) {
                    let facts = &prepass.functions[prepass.by_start[&flow.target]];
                    let mut helper_state = decoder.entry_state();
                    for (i, arg) in state.call_args.iter().enumerate() {
                        helper_state.regs[decoder.arg_reg(i)] = arg.clone();
                    }
                    helper_state.regs[decoder.sp_reg()] =
                        Expr::atom(decoder.atoms.ret(instruction.address, decoder.sp_reg()));
                    let outer = instruction.text();
                    decoder.eval_cfg(
                        &helper_state,
                        facts.start,
                        facts.end,
                        |_, inner, flow, state| {
                            if flow.kind == FlowKind::Call && wanted.contains(&flow.target) {
                                arguments
                                    .entry(flow.target)
                                    .or_default()
                                    .push(state.call_args.clone());
                                texts
                                    .entry(flow.target)
                                    .or_default()
                                    .push(format!("{} <- {outer}", inner.text()));
                            }
                            true
                        },
                    );
                }
                true
            },
        );
        for callee in wanted {
            let Some(args) = arguments.get(&callee) else {
                continue;
            };
            let Ok(stride) = staging_arguments(args) else {
                continue;
            };
            let Some((extent, store)) = extents.get(&callee).and_then(|e| e.as_ref()) else {
                continue;
            };
            if *extent != stride {
                continue;
            }
            let evidence = matches.entry(callee).or_default();
            evidence.push(format!("caller {:#x}: four calls; buffer stride {stride:#x}; highest store ends at {extent:#x}: {store}",caller.start));
            if let Some(calls) = texts.get(&callee) {
                evidence.extend(calls.iter().cloned());
            }
        }
    }
    let mut site = Site::missing(format!(
        "none of {pairs} caller/callee pairs initializes two pools of two buffers"
    ));
    if matches.len() == 1 {
        let (address, evidence) = matches.into_iter().next().unwrap();
        site.status = Status::Found;
        site.address = address;
        site.evidence = evidence;
        site.reason = format!("unique staging initializer among {pairs} caller/callee pairs");
    } else if !matches.is_empty() {
        site.status = Status::Ambiguous;
        site.reason = format!("{} functions match staging initialization", matches.len());
        site.candidates = matches.keys().copied().collect();
    }
    site
}

pub fn scan(image: &Image) -> Result<Scan> {
    let mut decoder = Decoder::new(image)?;
    let prepass = prepass(&decoder);
    let mut engine = match_engine(&mut decoder, &prepass);
    let mut row_copy = match_row_copy(&decoder, &prepass, &engine);
    let mut staging = match_staging(&mut decoder, &prepass);
    let inline = inline::scan(image, &engine.descriptor)?;
    if engine
        .descriptor
        .fields()
        .iter()
        .any(|(_, value)| u16::try_from(*value).is_err())
    {
        engine.site.status = Status::NotFound;
        engine.site.reason = "descriptor offset does not fit the hook's u16 fields".into();
    }
    engine.site.finish(image);
    row_copy.finish(image);
    staging.finish(image);
    if engine.site.status != Status::Found {
        engine.descriptor = Descriptor::default();
    }
    Ok(Scan {
        arch: image.arch,
        version: image.version.clone(),
        image_base: image.image_base,
        engine: engine.site,
        row_copy,
        staging,
        inline,
        descriptor: engine.descriptor,
        function_count: prepass.functions.len(),
        instruction_count: prepass.instructions,
    })
}
