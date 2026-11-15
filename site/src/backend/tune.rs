//! Converter: cmt `tune.json` (schema `cmt.tuning` v1) -> `cudamemcpy.run/v1`.
//!
//! * baseline  = `execution == "bare_driver"` rows (plain cudaMemcpy on pageable memory)
//! * selected  = rows whose (direction, size, settings) match the tuner's line-1 selection
//! * excluded  = rows that are not `status == complete` and `verified` (counted in the note)
//! * machine, selection = only what cmt keeps on a tune's selection line (`scrub::public_*`, an allowlist)
//! * the whole output then goes through `scrub` (usernames, UUIDs, paths, addresses, build stamps)

use super::model::{self, Row, RunFile, RunMeta};
use super::scrub;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub fn cmd_convert(args: &[String]) -> i32 {
    let mut input = None;
    let mut out = None;
    let mut meta = RunMeta { tier: "full".into(), ..Default::default() };
    let mut status_override = None;
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with('-') && args.get(i + 1).is_none_or(|v| v.starts_with('-')) {
            eprintln!("convert: {} needs a value", args[i]);
            return 2;
        }
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--id" => meta.id = v,
            "--machine" => meta.machine = v,
            "--tier" => meta.tier = v,
            "--status" => status_override = Some(v).filter(|s| !s.is_empty()),
            "--note" => meta.note = v,
            "-o" | "--out" => out = Some(v),
            a if !a.starts_with('-') && input.is_none() => {
                input = Some(a.to_string());
                i += 1;
                continue;
            }
            a => {
                eprintln!("convert: unknown argument {a}");
                return 2;
            }
        }
        i += 2;
    }
    let Some(input) = input else {
        eprintln!("usage: convert <tune.json> --id ID --machine LABEL [--tier full] [--status complete|partial] [--note ..] [-o out.json]");
        return 2;
    };
    match convert_file(&input, meta, status_override) {
        Ok(f) => {
            let mut v = serde_json::to_value(&f).unwrap();
            super::scrub::scrub(&mut v);
            let bytes = serde_json::to_vec(&v).unwrap();
            let n = f.rows.len();
            let res = match &out {
                Some(p) => std::fs::write(p, &bytes).map_err(|e| e.to_string()),
                None => {
                    use std::io::Write;
                    std::io::stdout().write_all(&bytes).map_err(|e| e.to_string())
                }
            };
            if let Err(e) = res {
                eprintln!("convert: write: {e}");
                return 1;
            }
            eprintln!(
                "converted {input} -> {} ({n} rows, selection: {})",
                out.as_deref().unwrap_or("stdout"),
                if f.selection.is_some() { "yes" } else { "no" }
            );
            0
        }
        Err(e) => {
            eprintln!("convert {input}: {e}");
            1
        }
    }
}

fn dir_name(d: &str) -> String {
    match d {
        "htod" => "h2d".into(),
        "dtoh" => "d2h".into(),
        other => other.into(),
    }
}

fn human_bytes(b: u64) -> String {
    if b >= 1 << 30 && b % (1 << 30) == 0 {
        format!("{}G", b >> 30)
    } else if b >= 1 << 20 && b % (1 << 20) == 0 {
        format!("{}M", b >> 20)
    } else if b >= 1 << 10 && b % (1 << 10) == 0 {
        format!("{}K", b >> 10)
    } else {
        format!("{b}")
    }
}

/// Short, stable, human-readable label for a settings object.
pub fn config_label(s: &Value) -> String {
    let get = |k: &str| s.get(k);
    let num = |k: &str| get(k).and_then(Value::as_u64);
    match get("method").and_then(Value::as_str) {
        Some("driver") => "cudaMemcpy".into(),
        Some("staged") => {
            let mut out = format!(
                "staged {} {}w×{}s {}",
                get("cpu_loop").and_then(Value::as_str).unwrap_or("?"),
                num("pipeline_workers").unwrap_or(0),
                num("slots_per_worker").unwrap_or(0),
                num("chunk_bytes").map(human_bytes).unwrap_or_default()
            );
            if let Some(m) = get("staging_memory").and_then(Value::as_str) {
                if m != "cached_pinned" {
                    out.push(' ');
                    out.push_str(m);
                }
            }
            append_rest(
                &mut out,
                s,
                &["method", "cpu_loop", "pipeline_workers", "slots_per_worker", "chunk_bytes", "staging_memory"],
            );
            out
        }
        Some(m) => {
            let mut out = m.to_string();
            append_rest(&mut out, s, &["method"]);
            out
        }
        None => s.to_string(),
    }
}

fn append_rest(out: &mut String, s: &Value, skip: &[&str]) {
    if let Some(o) = s.as_object() {
        for (k, v) in o {
            if skip.contains(&k.as_str()) {
                continue;
            }
            let v = match v {
                Value::Number(n) if n.as_u64().is_some_and(|x| x >= 1024) => human_bytes(n.as_u64().unwrap()),
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            out.push_str(&format!(" {k}={v}"));
        }
    }
}

fn link_gen(gts: f64) -> Option<u8> {
    Some(match gts as u32 {
        2 => 1,
        5 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        64 => 6,
        _ => return None,
    })
}

/// "16.0 GT/s PCIe x16" -> (gen, width)
fn parse_link(s: &str) -> (Option<u8>, Option<u8>) {
    let gts = s.split_whitespace().next().and_then(|t| t.parse::<f64>().ok());
    let width = s.rsplit('x').next().and_then(|t| t.trim().parse::<u8>().ok());
    (gts.and_then(link_gen), width)
}

fn cv_of(run: &Value) -> Option<f64> {
    let secs: Vec<f64> = run.get("samples")?.as_array()?.iter().filter_map(|s| s.get("seconds")?.as_f64()).collect();
    if secs.len() < 2 {
        return None;
    }
    let mean = secs.iter().sum::<f64>() / secs.len() as f64;
    let var = secs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (secs.len() - 1) as f64;
    Some(var.sqrt() / mean)
}

fn s(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(o) => o.to_string(),
    }
}

pub fn convert_file(path: &str, mut meta: RunMeta, status_override: Option<String>) -> Result<RunFile, String> {
    let text = std::fs::read(path).map_err(|e| e.to_string())?;
    let doc: Value = serde_json::from_slice(&text).map_err(|e| format!("parse: {e}"))?;
    if doc.get("schema").and_then(Value::as_str) != Some("cmt.tuning")
        || doc.get("version").and_then(Value::as_u64) != Some(1)
    {
        return Err("expected cmt.tuning version 1".into());
    }
    let sm = doc.get("source_machine").cloned().unwrap_or(Value::Null);
    let gpu = sm.pointer("/gpus/0").cloned().unwrap_or(Value::Null);
    let cpu = sm.pointer("/host/cpu").cloned().unwrap_or(Value::Null);

    // ---- run metadata
    if meta.id.is_empty() {
        return Err("--id is required".into());
    }
    if meta.machine.is_empty() {
        meta.machine = s(gpu.get("name")).trim_start_matches("NVIDIA ").to_string();
    }
    meta.gpu = s(gpu.get("name"));
    meta.cpu = s(cpu.get("model"));
    let link = s(gpu.pointer("/narrowest_link/link"));
    (meta.pcie_gen, meta.pcie_width) = parse_link(&link);
    let state = s(doc.get("state"));
    meta.status = status_override.unwrap_or(if state.is_empty() { "complete".into() } else { state.clone() });

    let mut m: BTreeMap<String, String> = BTreeMap::new();
    let mut put = |k: &str, v: String| {
        if !v.is_empty() && v != "null" {
            m.insert(k.into(), v);
        }
    };
    put("gpu.compute_capability", s(gpu.get("compute_capability")));
    put("gpu.sm_count", s(gpu.get("compute_units")));
    put(
        "gpu.memory_gib",
        gpu.get("memory_bytes")
            .and_then(Value::as_f64)
            .map(|b| format!("{:.0}", b / (1u64 << 30) as f64))
            .unwrap_or_default(),
    );
    put("gpu.numa_node", s(gpu.get("numa_node")));
    put("link", link.clone());
    put("link.current", s(gpu.pointer("/narrowest_link/current")));
    put("link.kind", s(gpu.pointer("/host_link/kind")));
    put("link.hops", gpu.get("pcie_path").and_then(Value::as_array).map(|a| a.len().to_string()).unwrap_or_default());
    put("cpu.vendor", s(cpu.get("vendor")));
    put("cpu.arch", s(cpu.get("arch")));
    put(
        "cpu.topology",
        format!(
            "{} sockets · {} cores · {} threads · {} NUMA nodes",
            s(cpu.get("sockets")),
            s(cpu.get("cores")),
            s(cpu.get("threads")),
            s(cpu.get("numa_nodes"))
        ),
    );
    if let Some(flags) = cpu.get("flags").and_then(Value::as_object) {
        put(
            "cpu.flags",
            flags
                .iter()
                .filter(|(_, v)| v.as_bool() == Some(true))
                .map(|(k, _)| k.as_str())
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    put(
        "host.memory_gib",
        sm.pointer("/host/memory/total_bytes")
            .and_then(Value::as_f64)
            .map(|b| format!("{:.0}", b / (1u64 << 30) as f64))
            .unwrap_or_default(),
    );
    put("host.thp", s(sm.pointer("/host/memory/thp_enabled")));
    put("host.os", s(sm.pointer("/host/os")));
    put("host.kernel", s(sm.pointer("/host/kernel/release")));
    put("host.virtualization", s(sm.pointer("/host/virtualization/type")));
    put("driver", s(sm.pointer("/driver/kernel_module/version")));
    put("cuda.driver_api", s(sm.pointer("/driver/cuda_driver_api")));
    put("cuda.runtime", s(sm.pointer("/driver/cuda_runtime")));
    put(
        "tool",
        format!(
            "{} {} ({})",
            s(sm.pointer("/tool/name")),
            s(sm.pointer("/tool/version")),
            s(sm.pointer("/tool/build_type"))
        ),
    );
    put("tool.cmt_sha256", s(doc.pointer("/identity/cmt_sha256")));
    put("tool.hook_sha256", s(doc.pointer("/identity/hook_sha256")));
    put("tuner.search", s(doc.pointer("/request/search")));
    put("tuner.samples", s(doc.pointer("/request/samples")));
    put("tuner.pages", s(doc.pointer("/request/pages")));
    put("tuner.policy", s(doc.get("selection_policy")));
    put("tuner.grid", s(doc.pointer("/request/grid")).split_whitespace().collect::<Vec<_>>().join("  "));
    put(
        "tuner.wall_hours",
        doc.pointer("/invocations/0/timings_seconds/total_before_publish")
            .and_then(Value::as_f64)
            .map(|v| format!("{:.2}", v / 3600.0))
            .unwrap_or_default(),
    );
    put("tuner.state", state);

    // ---- selection line (line 1), if the file has one
    let selection: Option<&Value> = doc.get("selection");
    // (dir, min, max, settings) for matching, from the tuner's own rows
    let sel_rows: Vec<(String, u64, u64, Value)> = selection
        .and_then(|s| s.get("rows"))
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter_map(|r| {
                    Some((
                        s(r.get("direction")),
                        r.get("min_bytes")?.as_u64()?,
                        r.get("max_bytes")?.as_u64()?,
                        r.get("settings")?.clone(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();

    // ---- measurements
    let runs = doc.get("runs").and_then(Value::as_array).ok_or("no runs[]")?;
    let mut rows = Vec::with_capacity(runs.len());
    let mut excluded = 0usize;
    let mut failed_verify = 0usize;
    for r in runs {
        let ok = s(r.get("status")) == "complete" && r.get("verified").and_then(Value::as_bool) == Some(true);
        let gbps = r.get("gbps").and_then(Value::as_f64);
        let (Some(gbps), true) = (gbps, ok) else {
            excluded += 1;
            if r.get("verified").and_then(Value::as_bool) == Some(false) {
                failed_verify += 1;
            }
            continue;
        };
        let dir_raw = s(r.get("direction"));
        let bytes = r.get("bytes").and_then(Value::as_u64).unwrap_or(0);
        let settings = r.get("settings").cloned().unwrap_or(Value::Object(Map::new()));
        let baseline = s(r.get("execution")) == "bare_driver";
        let selected =
            sel_rows.iter().any(|(d, lo, hi, st)| *d == dir_raw && *lo < bytes && bytes <= *hi && *st == settings);
        rows.push(Row(
            dir_name(&dir_raw),
            bytes as i64,
            config_label(&settings),
            (gbps * 1e4).round() / 1e4,
            r.get("sample_count").and_then(Value::as_i64).unwrap_or(0) as i32,
            cv_of(r).map(|v| (v * 1e4).round() / 1e4),
            baseline,
            selected,
        ));
    }
    // a bare_driver baseline row and the identical hooked "driver" row may both be selected; fine.
    if excluded > 0 {
        let msg = if failed_verify > 0 {
            format!("{excluded} measurements dropped ({failed_verify} failed the copy-correctness check)")
        } else {
            format!("{excluded} measurements dropped (incomplete)")
        };
        meta.note = if meta.note.is_empty() { msg } else { format!("{} — {msg}", meta.note) };
    }
    meta.meta = m;
    Ok(RunFile {
        format: model::FORMAT.into(),
        run: meta,
        cols: model::COLS.iter().map(|s| s.to_string()).collect(),
        rows,
        machine: Some(scrub::public_machine(&sm)).filter(|m| !m.is_null()),
        selection: selection.map(scrub::public_selection).filter(|s| !s.is_null()),
    })
}
