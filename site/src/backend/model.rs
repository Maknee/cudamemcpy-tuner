//! The import format (`cudamemcpy.run/v1`) and the automated (T1) checks.
//!
//! One JSON file == one run. Rows are positional arrays to keep files small:
//!   cols: ["dir","size","config","gbps","n","cv","baseline","selected"]
//! `meta` keys outside `scrub::META` are dropped on import.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const FORMAT: &str = "cudamemcpy.run/v1";
pub const COLS: [&str; 8] = ["dir", "size", "config", "gbps", "n", "cv", "baseline", "selected"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunFile {
    pub format: String,
    pub run: RunMeta,
    pub cols: Vec<String>,
    pub rows: Vec<Row>,
    /// what cmt's selection line keeps of the `cmt-machine 1` record: CPU model + arch, GPU name, driver version
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine: Option<serde_json::Value>,
    /// the tuner's `{"selection":{...}}` object (line 1 of tune.json), reduced to what cmt writes there; served to `cmt run`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunMeta {
    pub id: String,
    pub machine: String,
    #[serde(default)]
    pub gpu: String,
    #[serde(default)]
    pub cpu: String,
    /// instant | quick | full | full-with-kernel ...
    pub tier: String,
    /// complete | partial
    #[serde(default = "complete")]
    pub status: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub pcie_gen: Option<u8>,
    #[serde(default)]
    pub pcie_width: Option<u8>,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub meta: BTreeMap<String, String>,
}
fn complete() -> String {
    "complete".into()
}

/// dir, size(bytes), config, gbps, n samples, cv, is_baseline, chosen by the tuner's selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row(
    pub String,
    pub i64,
    pub String,
    pub f64,
    pub i32,
    pub Option<f64>,
    pub bool,
    #[serde(default)] pub bool,
);

/// Theoretical unidirectional PCIe payload bandwidth (GB/s, decimal) for a link.
pub fn link_gbps(gen: Option<u8>, width: Option<u8>) -> Option<f64> {
    let per_lane = match gen? {
        3 => 0.985,
        4 => 1.969,
        5 => 3.938,
        6 => 7.563,
        _ => return None,
    };
    Some(per_lane * width.unwrap_or(16) as f64)
}

pub struct Verdict {
    pub tier: &'static str, // T0 / T1 / NC
    pub notes: Vec<String>,
}

/// Automated plausibility checks. Never deletes data: a failing run is marked
/// NC (non-compliant) with reasons, a run with warnings stays T0.
pub fn verify(f: &RunFile) -> Result<Verdict, String> {
    if f.format != FORMAT {
        return Err(format!("format must be {FORMAT:?}, got {:?}", f.format));
    }
    if f.cols.iter().map(String::as_str).ne(COLS.iter().copied()) {
        return Err(format!("cols must be {COLS:?}"));
    }
    let id = &f.run.id;
    if id.is_empty()
        || id.len() > 96
        || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
    {
        return Err("run.id must be 1-96 chars of [A-Za-z0-9._-]".into());
    }
    if f.rows.is_empty() {
        return Err("no rows".into());
    }
    let mut hard = vec![];
    let mut warn = vec![];
    let link = link_gbps(f.run.pcie_gen, f.run.pcie_width);
    for (i, Row(dir, size, _cfg, gbps, n, _cv, _, _)) in f.rows.iter().enumerate() {
        if !gbps.is_finite() || *gbps < 0.0 {
            hard.push(format!("row {i}: bad gbps {gbps}"));
        }
        if *size <= 0 {
            hard.push(format!("row {i}: bad size {size}"));
        }
        if *n < 0 {
            hard.push(format!("row {i}: bad n"));
        }
        if let (Some(l), true) = (link, dir == "h2d" || dir == "d2h") {
            if *gbps > l * 1.0 {
                hard.push(format!("row {i}: {gbps:.2} GB/s exceeds PCIe link ceiling {l:.1}"));
            } else if *gbps > l * 0.92 && *size >= 1 << 20 {
                warn.push(format!("row {i}: {gbps:.2} GB/s is >92% of link ceiling"));
            }
        }
        if hard.len() > 20 {
            break;
        }
    }
    if link.is_none() {
        warn.push("PCIe gen/width not recorded: ceiling check skipped".into());
    }
    // Noise check only on what the site reports per (dir,size): the tuner's pick, else the fastest row.
    // Losing configurations are expected to be noisy.
    let any_sel = f.rows.iter().any(|r| r.7);
    let mut best: std::collections::HashMap<(&str, i64), &Row> = std::collections::HashMap::new();
    for r in &f.rows {
        if any_sel && !r.7 {
            continue;
        }
        let e = best.entry((r.0.as_str(), r.1)).or_insert(r);
        if r.3 > e.3 {
            *e = r;
        }
    }
    let mut noisy: Vec<_> = best.values().filter(|r| r.1 >= 64 << 20 && r.5.is_some_and(|cv| cv > 0.05)).collect();
    noisy.sort_by_key(|r| (r.0.clone(), r.1));
    for r in noisy {
        warn.push(format!("{} {} B: reported config has CV {:.1}% (> 5%)", r.0, r.1, r.5.unwrap() * 100.0));
    }
    let tier = if !hard.is_empty() {
        "NC"
    } else if warn.iter().any(|w| !w.starts_with("PCIe")) || f.run.status != "complete" {
        "T0"
    } else {
        "T1"
    };
    let mut notes = hard;
    notes.truncate(20);
    warn.truncate(20);
    notes.extend(warn);
    if f.run.status != "complete" {
        notes.push(format!("run marked {}", f.run.status));
    }
    Ok(Verdict { tier, notes })
}
