//! Driver bytes and ELF facts. `object` reads ELF; `gimli` reads unwind entries.
//! All scanner addresses are ELF virtual addresses. Generated rows contain offsets
//! from the lowest aligned load segment, which is the base used by the hook.

use anyhow::{bail, ensure, Context, Result};
use clap::ValueEnum;
use gimli::{BaseAddresses, CieOrFde, EhFrame, LittleEndian, UnwindSection};
use object::{Object, ObjectSection, ObjectSegment, ObjectSymbol, ObjectSymbolTable};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    ops::Range,
    path::Path,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Arch {
    #[value(name = "x86_64")]
    X86_64,
    #[value(name = "aarch64")]
    Aarch64,
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
        })
    }
}

impl Arch {
    pub fn table_name(self) -> &'static str {
        match self {
            Self::X86_64 => "versions_db.inc",
            Self::Aarch64 => "versions_aarch64.inc",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Function {
    pub start: u64,
    pub size: u64,
}
impl Function {
    pub fn end(self) -> u64 {
        self.start + self.size
    }
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub vaddr: u64,
    pub file_offset: u64,
    pub file_size: u64,
    pub executable: bool,
}

#[derive(Clone, Debug)]
pub struct Image {
    pub arch: Arch,
    pub version: String,
    pub image_base: u64,
    pub bytes: Vec<u8>,
    pub segments: Vec<Segment>,
    pub functions: Vec<Function>,
    pub plt: Vec<Range<u64>>,
    pub got_symbols: BTreeMap<u64, String>,
}

impl Image {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::parse(bytes).with_context(|| format!("parse {}", path.display()))
    }

    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let elf = object::File::parse(bytes.as_slice()).context("read ELF")?;
        ensure!(
            elf.format() == object::BinaryFormat::Elf && elf.is_64() && elf.is_little_endian(),
            "only little-endian ELF64 is supported"
        );
        let arch = match elf.architecture() {
            object::Architecture::X86_64 => Arch::X86_64,
            object::Architecture::Aarch64 => Arch::Aarch64,
            other => bail!("unsupported architecture {other:?}"),
        };
        let mut image_base = u64::MAX;
        let mut segments = Vec::new();
        for s in elf.segments() {
            let (file_offset, file_size) = s.file_range();
            ensure!(
                s.address().checked_add(file_size).is_some(),
                "segment address overflow"
            );
            ensure!(
                file_offset
                    .checked_add(file_size)
                    .is_some_and(|end| end <= bytes.len() as u64),
                "segment extends past file"
            );
            let align = s.align();
            ensure!(
                align <= 1 || align.is_power_of_two(),
                "invalid segment alignment"
            );
            let aligned = if align > 1 {
                s.address() & !(align - 1)
            } else {
                s.address()
            };
            image_base = image_base.min(aligned);
            let executable = matches!(s.flags(), object::SegmentFlags::Elf { p_flags } if p_flags & object::elf::PF_X != 0);
            segments.push(Segment {
                vaddr: s.address(),
                file_offset,
                file_size,
                executable,
            });
        }
        ensure!(!segments.is_empty(), "no loadable segments");

        let section = elf
            .section_by_name(".eh_frame")
            .context("no .eh_frame function boundaries")?;
        let data = section.data().context("read .eh_frame")?;
        let mut eh = EhFrame::new(data, LittleEndian);
        eh.set_address_size(8);
        let bases = BaseAddresses::default().set_eh_frame(section.address());
        let mut entries = eh.entries(&bases);
        let mut functions = Vec::new();
        while let Some(entry) = entries.next().context("read unwind entry")? {
            if let CieOrFde::Fde(fde) = entry {
                let fde = fde
                    .parse(EhFrame::cie_from_offset)
                    .context("read function unwind entry")?;
                ensure!(
                    fde.initial_address().checked_add(fde.len()).is_some(),
                    "function range overflow"
                );
                if fde.len() != 0 && fde.initial_address() != 0 {
                    functions.push(Function {
                        start: fde.initial_address(),
                        size: fde.len(),
                    });
                }
            }
        }
        functions.sort_by_key(|f| (f.start, std::cmp::Reverse(f.size)));
        functions.dedup_by_key(|f| f.start);
        ensure!(!functions.is_empty(), "no function boundaries in .eh_frame");
        let mut plt = Vec::new();
        for name in [".plt", ".plt.sec", ".plt.got"] {
            if let Some(s) = elf.section_by_name(name) {
                let end = s
                    .address()
                    .checked_add(s.size())
                    .context("PLT range overflow")?;
                plt.push(s.address()..end);
            }
        }
        let mut got_symbols = BTreeMap::new();
        if let (Some(relocations), Some(symbols)) =
            (elf.dynamic_relocations(), elf.dynamic_symbol_table())
        {
            for (address, relocation) in relocations {
                if let object::RelocationTarget::Symbol(index) = relocation.target() {
                    let symbol = symbols
                        .symbol_by_index(index)
                        .context("read relocation symbol")?;
                    let name = symbol.name().context("read relocation symbol name")?;
                    if !name.is_empty() {
                        got_symbols.insert(address, name.to_owned());
                    }
                }
            }
        }
        let version = version_banner(&bytes)?;
        Ok(Self {
            arch,
            version,
            image_base,
            bytes,
            segments,
            functions,
            plt,
            got_symbols,
        })
    }

    pub fn at(&self, address: u64) -> Option<&[u8]> {
        for s in &self.segments {
            if let Some(delta) = address.checked_sub(s.vaddr).filter(|d| *d < s.file_size) {
                let start = usize::try_from(s.file_offset.checked_add(delta)?).ok()?;
                let end = usize::try_from(s.file_offset.checked_add(s.file_size)?).ok()?;
                return self.bytes.get(start..end);
            }
        }
        None
    }
    pub fn executable(&self, address: u64) -> bool {
        self.segments.iter().any(|s| {
            s.executable
                && address
                    .checked_sub(s.vaddr)
                    .is_some_and(|d| d < s.file_size)
        })
    }
    pub fn in_plt(&self, address: u64) -> bool {
        self.plt.iter().any(|r| r.contains(&address))
    }
}

pub fn version_banner(bytes: &[u8]) -> Result<String> {
    let regex =
        regex::bytes::Regex::new(r"Version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)\. Copyright \(C\) ")
            .unwrap();
    let versions: BTreeSet<_> = regex
        .captures_iter(bytes)
        .map(|m| {
            std::str::from_utf8(m.get(1).unwrap().as_bytes())
                .unwrap()
                .to_owned()
        })
        .collect();
    ensure!(
        versions.len() == 1,
        "expected one unique driver version banner, found {}",
        versions.len()
    );
    Ok(versions.into_iter().next().unwrap())
}
