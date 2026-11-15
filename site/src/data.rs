//! Shared data types and presentation helpers for the exporter and static browser app.

#[cfg(feature = "web")]
pub use crate::client::{download_run, get_breakdown, get_detail, get_points, get_summary};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct RunRow {
    pub id: String,
    pub machine: String,
    pub gpu: String,
    pub cpu: String,
    pub tier: String,
    pub status: String,
    pub note: String,
    pub pcie: Option<String>,
    pub link_gbps: Option<f64>,
    pub started_at: Option<String>,
    pub verify: String,
    pub verify_notes: Vec<String>,
    pub sha256: String,
    pub n_points: i64,
    pub n_configs: i64,
    /// curves use the tuner's own per-size selection (else best measured)
    pub has_selection: bool,
}

/// Best tuned vs baseline per size for one (run, dir). Columnar = tiny payload.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct Curve {
    pub size: Vec<i64>,
    pub best: Vec<f64>,
    pub base: Vec<Option<f64>>,
    pub cfg: Vec<String>,
}

pub type Curves = BTreeMap<String, BTreeMap<String, Curve>>; // run -> dir -> curve

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct Summary {
    pub runs: Vec<RunRow>,
    pub curves: Curves,
}

/// dir, size, config, gbps, n, cv, baseline, selected-by-tuner
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Point(pub String, pub i64, pub String, pub f64, pub Option<i32>, pub Option<f64>, pub bool, pub bool);

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct Detail {
    pub run: RunRow,
    pub meta: BTreeMap<String, String>,
    pub curves: BTreeMap<String, Curve>,
    /// top-3 configs per (dir,size)
    pub top: Vec<Point>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct Page {
    pub total: i64,
    pub page: i64,
    pub per: i64,
    pub rows: Vec<Point>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct PointsQuery {
    pub dir: String,  // "" = all
    pub size: i64,    // 0 = all
    pub q: String,    // config substring
    pub sort: String, // gbps|size|config|dir
    pub desc: bool,
    pub page: i64,
    pub per: i64,
}

/// Best GB/s per (value of one config dimension, size) for one run + direction.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct Breakdown {
    pub groups: Vec<(String, Vec<(i64, f64)>)>,
    pub base: Vec<(i64, f64)>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct BreakdownQuery {
    pub dir: String,
    /// loop | isa | threads | buffers | size
    pub by: String,
}

// ------------------------------------------------------------- formatting

pub fn fmt_size(b: i64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < 4 {
        v /= 1024.0;
        i += 1;
    }
    if v.fract() == 0.0 {
        format!("{} {}", v as i64, U[i])
    } else {
        format!("{:.1} {}", v, U[i])
    }
}

/// A config label split into the tuner's own grid dimensions.
#[derive(Clone, PartialEq, Default, Debug)]
pub struct Cfg {
    pub kind: String,
    pub isa: String,
    pub threads: String,
    pub buffers: String,
    pub size: String,
    pub extra: String,
}

/// "staged nt_store@sse2 4w×4s 4M [extra..]" -> parts; anything else (e.g. "cudaMemcpy") -> kind only
pub fn parse_cfg(label: &str) -> Cfg {
    let Some(rest) = label.strip_prefix("staged ") else {
        return Cfg { kind: label.into(), ..Default::default() };
    };
    let mut it = rest.split(' ');
    let lp = it.next().unwrap_or("");
    let (kind, isa) = lp.split_once('@').unwrap_or((lp, ""));
    let (w, s) = it.next().unwrap_or("").split_once('×').unwrap_or(("", ""));
    let chunk = it.next().unwrap_or("");
    let size = match chunk.char_indices().last() {
        Some((i, u @ ('K' | 'M' | 'G'))) => format!("{} {u}iB", &chunk[..i]),
        _ => chunk.to_string(),
    };
    Cfg {
        kind: kind.into(),
        isa: isa.into(),
        threads: w.trim_end_matches('w').into(),
        buffers: s.trim_end_matches('s').into(),
        size,
        extra: it.collect::<Vec<_>>().join(" "),
    }
}

pub fn loop_help(kind: &str) -> &'static str {
    match kind {
        "nt_store" => "non-temporal (streaming) stores into pinned staging buffers",
        "nt_load" => "non-temporal (streaming) loads from the source",
        "cached" => "ordinary cached loads and stores",
        "movsb" => "rep movsb",
        "memcpy" => "libc memcpy",
        "cudaMemcpy" => "plain cudaMemcpy, no staging",
        _ => "",
    }
}

/// One-line readable form for tight spots: "nt_store avx2 · 16 threads × 4 buffers × 1 MiB"
pub fn cfg_short(label: &str) -> String {
    let c = parse_cfg(label);
    if c.threads.is_empty() {
        return label.to_string();
    }
    format!("{} {} · {} threads × {} buffers × {}", c.kind, c.isa, c.threads, c.buffers, c.size)
}

/// "NVIDIA GeForce RTX 3090" -> "RTX 3090", "NVIDIA A100-SXM4-40GB" -> "A100"
pub fn gpu_type(gpu: &str) -> String {
    let g = gpu.trim_start_matches("NVIDIA ").trim_start_matches("GeForce ");
    let mut t = g.split([' ', '-']);
    match t.next() {
        Some(first @ ("RTX" | "GTX" | "Quadro" | "Tesla")) => {
            format!("{first} {}", t.next().unwrap_or("")).trim().to_string()
        }
        Some(first) if !first.is_empty() => first.to_string(),
        _ => "other".into(),
    }
}

/// "Intel(R) Xeon(R) Platinum 8481C" -> "Xeon 8481C", "AMD Ryzen 7 7800X3D 8-Core Processor" -> "Ryzen 7 7800X3D"
pub fn cpu_short(cpu: &str) -> String {
    let c = cpu.replace("(R)", "").replace("(TM)", "");
    let ultra = c.contains("Ultra");
    let s = c
        .split_whitespace()
        .filter(|w| !w.ends_with("-Core"))
        .filter(|w| !matches!(*w, "Intel" | "AMD" | "Platinum" | "Gold" | "Silver" | "Processor" | "CPU"))
        .filter(|w| *w != "Core" || ultra)
        .collect::<Vec<_>>()
        .join(" ")
        .replace('-', " ");
    if s.is_empty() {
        "unknown CPU".into()
    } else {
        s
    }
}

/// "rtx3090-20261107-full" -> "2026-11-07"
pub fn run_date(id: &str) -> String {
    id.split('-')
        .map(|p| p.trim_end_matches(|c: char| c.is_ascii_lowercase()))
        .find(|p| p.len() == 8 && p.bytes().all(|b| b.is_ascii_digit()))
        .map(|p| format!("{}-{}-{}", &p[..4], &p[4..6], &p[6..]))
        .unwrap_or_default()
}

/// Newest run per physical machine (same GPU + CPU). Run ids embed the date, so the highest id wins.
pub fn latest_per_machine(runs: &[RunRow]) -> std::collections::BTreeSet<String> {
    let mut best: BTreeMap<(&str, &str), &str> = BTreeMap::new();
    for r in runs {
        let e = best.entry((r.gpu.as_str(), r.cpu.as_str())).or_insert(r.id.as_str());
        if r.id.as_str() > *e {
            *e = r.id.as_str();
        }
    }
    best.into_values().map(str::to_string).collect()
}

pub fn fmt_gbps(v: f64) -> String {
    if v >= 100.0 {
        format!("{v:.0}")
    } else if v >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

pub fn dir_label(d: &str) -> &str {
    match d {
        "h2d" => "Host → Device",
        "d2h" => "Device → Host",
        "d2d" => "Device → Device",
        "h2h" => "Host → Host",
        "bidir" => "Bidirectional",
        other => other,
    }
}

pub const PALETTE: [&str; 10] =
    ["#2a6fdb", "#d1495b", "#2a9d8f", "#e9a23b", "#7b5ea7", "#3d7d3d", "#c0587e", "#5c6b7a", "#b5651d", "#1b998b"];

/// Dictionary-encoded rows, split into bounded JSON files by the offline exporter.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PackedPoints {
    pub dirs: Vec<String>,
    pub sizes: Vec<i64>,
    pub configs: Vec<String>,
    pub files: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PackedPoint(pub usize, pub usize, pub usize, pub f64, pub Option<i32>, pub Option<f64>, pub u8);
