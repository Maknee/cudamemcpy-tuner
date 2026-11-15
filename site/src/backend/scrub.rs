//! Last-line privacy pass over everything we publish (converted run files, the DB, downloads).
//!
//! Runs on the whole JSON document, so new fields from future tuner versions are covered too:
//! * identifying keys are dropped (hostname, user, serials, MAC/IP addresses, argv, cwd, ...)
//! * GPU / MIG UUIDs anywhere in any string become a short one-way hash
//! * `(user@host)` build stamps and their build dates are removed (driver banner, kernel version)
//! * home-directory paths and e-mail addresses are masked
//! * IPv4 / MAC addresses are masked (except in version fields)

use regex::Regex;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

const DROP_KEYS: &[&str] = &[
    "hostname",
    "host_name",
    "nodename",
    "node_name",
    "fqdn",
    "domain",
    "user",
    "username",
    "user_name",
    "login",
    "uid",
    "gid",
    "home",
    "cwd",
    "pwd",
    "argv",
    "args",
    "command_line",
    "serial",
    "serial_number",
    "board_serial",
    "product_serial",
    "chassis_serial",
    "product_uuid",
    "machine_id",
    "boot_id",
    "mac",
    "mac_address",
    "ip",
    "ip_address",
    "ipv4",
    "ipv6",
    "email",
    "ssh",
    "token",
    "password",
    "secret",
    "api_key",
];

struct Res {
    gpu_uuid: Regex,
    stamp: Regex,
    date: Regex,
    home: Regex,
    email: Regex,
    ipv4: Regex,
    mac: Regex,
}

fn res() -> &'static Res {
    static R: OnceLock<Res> = OnceLock::new();
    R.get_or_init(|| Res {
        gpu_uuid: Regex::new(r"(?i)\b(GPU|MIG)-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b").unwrap(),
        // "(alice@builder)  Thu Sep 10 01:41:06 PM EDT 2026" -> ""
        stamp: Regex::new(r"\s*\([^()\s]+@[^()\s]+\)(?:\s+[A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d{1,2}\s+[\d:]+(?:\s+[AP]M)?(?:\s+[A-Z]{2,5})?(?:\s+\d{1,4})?)?").unwrap(),
        // build dates such as "Sun Jul 12 14:42:50 EDT 2026" (reveal time zone + build time)
        date: Regex::new(r"\s*\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+[A-Z][a-z]{2}\s+\d{1,2}\s+\d{1,2}:\d{2}:\d{2}(?:\s+[AP]M)?(?:\s+[A-Z]{2,5})?(?:\s+\d{1,4})?").unwrap(),
        home: Regex::new(r"/(home|Users)/[^/\s\x22']+").unwrap(),
        email: Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}").unwrap(),
        ipv4: Regex::new(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b").unwrap(),
        mac: Regex::new(r"(?i)\b[0-9a-f]{2}(?::[0-9a-f]{2}){5}\b").unwrap(),
    })
}

pub fn hash8(s: &str) -> String {
    Sha256::digest(s.as_bytes())[..4].iter().map(|b| format!("{b:02x}")).collect()
}

pub fn scrub(v: &mut Value) {
    walk(v, "");
}

fn walk(v: &mut Value, key: &str) {
    match v {
        Value::Object(m) => {
            m.retain(|k, val| {
                let kl = k.to_ascii_lowercase();
                // `iommu.cmdline` is an object of parsed iommu flags (harmless); a raw string cmdline is not
                !(DROP_KEYS.contains(&kl.as_str()) || (kl == "cmdline" && !val.is_object()))
            });
            for (k, val) in m.iter_mut() {
                if k.eq_ignore_ascii_case("uuid") {
                    if let Some(s) = val.as_str() {
                        if !s.starts_with("sha256:") {
                            *val = Value::String(format!("sha256:{}", hash8(s)));
                            continue;
                        }
                    }
                }
                walk(val, k);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| walk(x, key)),
        Value::String(s) => {
            let t = clean_str(s, key);
            if t != *s {
                *s = t;
            }
        }
        _ => {}
    }
}

pub fn clean_str(s: &str, key: &str) -> String {
    // fast path: the bulk of the data (config labels, numbers) has none of these markers
    if !s.contains(['@', '-', '/', ':', '.']) {
        return s.to_string();
    }
    let r = res();
    let mut t =
        r.gpu_uuid.replace_all(s, |c: &regex::Captures| format!("{}-sha256:{}", &c[1], hash8(&c[0]))).into_owned();
    t = r.stamp.replace_all(&t, "").into_owned();
    t = r.date.replace_all(&t, "").into_owned();
    t = r.home.replace_all(&t, "/$1/[user]").into_owned();
    t = r.email.replace_all(&t, "[email]").into_owned();
    t = r.mac.replace_all(&t, "[mac]").into_owned();
    let k = key.to_ascii_lowercase();
    if !(k.contains("version")
        || k.contains("release")
        || k.contains("driver")
        || k.contains("runtime")
        || k.contains("api"))
    {
        t = r.ipv4.replace_all(&t, "[ip]").into_owned();
    }
    t
}

// ---------------------------------------------------------------- allowlist
//
// `scrub` above is a denylist over whole documents. What we publish of the tuner's machine record and
// selection is instead an allowlist: exactly what cmt itself keeps on a tune's selection line, which is
// also all `cmt export` publishes of it (cmt `src/tune_table.cpp`: SelectionRequest, SelectionMachine,
// WriteSettings, EncodeSelection), in cmt's field order. A field not named here is never published,
// including fields a future tuner adds. Used on convert, on import, and again when serving.

/// The shape of a published JSON value.
enum Shape {
    /// a string, number, bool or null; any other value is dropped
    Scalar,
    /// the listed fields, in this order; any other field is dropped
    Object(&'static [(&'static str, Shape)]),
    /// every element
    Array(&'static Shape),
    /// the first element only (cmt's line names `gpus[0]` only)
    First(&'static Shape),
    /// `request.environment`: only `CMT_WORKER_*` entries (worker placement) affect a replay
    WorkerEnv,
}
use Shape::*;

/// A selected row's `settings` (cmt `WriteSettings`): staged copy, or progressive pinning.
const SETTINGS: Shape = Object(&[
    ("method", Scalar),
    ("cpu_loop", Scalar),
    ("pipeline_workers", Scalar),
    ("slots_per_worker", Scalar),
    ("chunk_bytes", Scalar),
    ("staging_memory", Scalar),
    ("first_chunk_bytes", Scalar),
    ("max_chunk_bytes", Scalar),
    ("max_registered_bytes", Scalar),
    ("read_only", Scalar),
]);

const ROW: Shape =
    Object(&[("direction", Scalar), ("min_bytes", Scalar), ("max_bytes", Scalar), ("settings", SETTINGS)]);

/// cmt `SelectionMachine`: CPU model and arch, GPU name, driver version. Nothing else of the host.
const MACHINE: Shape = Object(&[
    ("host", Object(&[("cpu", Object(&[("model", Scalar), ("arch", Scalar)]))])),
    ("gpus", First(&Object(&[("name", Scalar)]))),
    ("driver", Object(&[("kernel_module", Object(&[("version", Scalar)]))])),
]);

/// cmt `SelectionRequest`, in the order `cmt tune` writes the request.
const REQUEST: Shape = Object(&[
    ("search", Scalar),
    ("process_rounds", Scalar),
    ("tie_percent", Scalar),
    ("max_pinned_mib", Scalar),
    ("progressive_pinning", Scalar),
    ("max_registered_mib", Scalar),
    ("registration_limit", Scalar),
    ("progressive_pinning_settings", SETTINGS),
    ("environment", WorkerEnv),
]);

/// cmt `EncodeSelection`: the selection line's object.
const SELECTION: Shape = Object(&[
    ("policy", Scalar),
    ("file_bytes", Scalar),
    ("request", REQUEST),
    ("source_machine", MACHINE),
    ("rows", Array(&ROW)),
    ("tie_reference_rows", Array(&ROW)),
    ("above_pinning_budget", Scalar),
    ("registration_limit", Scalar),
]);

/// The run-detail `meta` keys the site shows (what `convert` writes, minus `tool.commit`).
const META: &[&str] = &[
    "cpu.arch",
    "cpu.flags",
    "cpu.topology",
    "cpu.vendor",
    "cuda.driver_api",
    "cuda.runtime",
    "driver",
    "gpu.compute_capability",
    "gpu.memory_gib",
    "gpu.numa_node",
    "gpu.sm_count",
    "host.kernel",
    "host.memory_gib",
    "host.os",
    "host.thp",
    "host.virtualization",
    "link",
    "link.current",
    "link.hops",
    "link.kind",
    "tool",
    "tool.cmt_sha256",
    "tool.hook_sha256",
    "tuner.grid",
    "tuner.pages",
    "tuner.policy",
    "tuner.samples",
    "tuner.search",
    "tuner.state",
    "tuner.wall_hours",
];

/// Writes the allowed part of `v` as compact JSON; false (and nothing written) when `v` has the wrong type.
fn write(v: &Value, shape: &Shape, out: &mut String) -> bool {
    match (shape, v) {
        (Scalar, Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)) => {
            out.push_str(&v.to_string());
        }
        (Object(fields), Value::Object(m)) => {
            out.push('{');
            let mut first = true;
            for (key, shape) in *fields {
                let Some(val) = m.get(*key) else { continue };
                let mut text = String::new();
                if write(val, shape, &mut text) {
                    out.push_str(if first { "\"" } else { ",\"" });
                    out.push_str(key);
                    out.push_str("\":");
                    out.push_str(&text);
                    first = false;
                }
            }
            out.push('}');
        }
        (Array(elem), Value::Array(a)) | (First(elem), Value::Array(a)) => {
            let take = if matches!(shape, First(_)) { 1 } else { a.len() };
            out.push('[');
            let mut first = true;
            for x in a.iter().take(take) {
                let mut text = String::new();
                if write(x, elem, &mut text) {
                    if !first {
                        out.push(',');
                    }
                    out.push_str(&text);
                    first = false;
                }
            }
            out.push(']');
        }
        (WorkerEnv, Value::Array(a)) => {
            let kept: Vec<&Value> =
                a.iter().filter(|e| e.as_str().is_some_and(|s| s.starts_with("CMT_WORKER_"))).collect();
            out.push_str(&Value::Array(kept.into_iter().cloned().collect()).to_string());
        }
        _ => return false,
    }
    true
}

fn allowed(v: &Value, shape: &Shape) -> Option<String> {
    let mut out = String::new();
    write(v, shape, &mut out).then_some(out)
}

/// The selection object exactly as cmt writes it on line 1 of a tune.json (field order included).
/// None when `sel` is not an object.
pub fn selection_json(sel: &Value) -> Option<String> {
    allowed(sel, &SELECTION)
}

/// The publishable part of a tuner selection; Null when there is none.
pub fn public_selection(sel: &Value) -> Value {
    selection_json(sel).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null)
}

/// The publishable part of a `cmt-machine 1` record; Null when there is none.
pub fn public_machine(m: &Value) -> Value {
    allowed(m, &MACHINE).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null)
}

/// Whether a run-detail `meta` key may be published.
pub fn public_meta_key(key: &str) -> bool {
    META.contains(&key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scrubs() {
        let mut v = json!({
            "host": {"hostname": "box", "kernel": {"version": "#2 SMP PREEMPT_DYNAMIC Sun Jul 12 14:42:50 EDT 2026"}},
            "driver": {"kernel_module": {"banner": "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  615.71.09  Release Build  (alice@builder)  Thu Sep 10 01:41:06 PM EDT 2026", "version": "615.71.09"}},
            "iommu": {"cmdline": {"iommu": "pt"}},
            "gpus": [{"uuid": "GPU-00000000-1111-2222-3333-444444444444"}],
            "env": ["CUDA_VISIBLE_DEVICES=GPU-00000000-1111-2222-3333-444444444444"],
            "p": "/home/alice/x", "e": "a@b.com", "ip": "1.2.3.4", "note": "from 10.0.0.5",
            "cfg": "staged cached@avx2 4w×4s 4M",
            "k2": {"version": "#138~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  7 13:43:15 UTC"},
            "k3": {"version": "#31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2"}
        });
        scrub(&mut v);
        let s = v.to_string();
        for bad in ["alice", "builder", "box", "00000000-1111", "a@b.com", "10.0.0.5", "EDT", "2026", "Aug", "UTC"] {
            assert!(!s.contains(bad), "{bad} left in {s}");
        }
        assert!(s.contains("615.71.09") && s.contains("cached@avx2") && s.contains("\"iommu\":\"pt\""));
        assert!(s.contains("Release Build\""), "{s}");
    }

    /// A selection as the tuner and older imports stored it: the whole request and machine record.
    fn leaky_selection() -> Value {
        json!({
            "policy": "measured_steps_v4", "file_bytes": 10159,
            "request": {
                "search": "full", "tuning_params": "", "grid": "sizes=4K", "pages": "small", "process_rounds": 1,
                "samples": 2, "min_bytes": 1, "max_pinned_mib": 1024, "progressive_pinning": false,
                "max_registered_mib": 1024, "max_processes": 4096, "device": 0, "directions": ["htod", "dtoh"],
                "environment": ["CUDA_VISIBLE_DEVICES=GPU-sha256:1dce631f", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "CMT_WORKER_CPUS=0-7"]
            },
            "source_machine": {
                "schema": "cmt-machine 1",
                "tool": {"commit": "151617a359b534bcd868264dae100fd18e2e1f5d", "dirty": true, "version": "0.1.0"},
                "host": {
                    "cpu": {"model": "AMD Ryzen 7 7800X3D 8-Core Processor", "arch": "x86_64", "cores": 8, "flags": {"avx2": true}},
                    "kernel": {"release": "6.6.36", "version": "#2 SMP PREEMPT_DYNAMIC"},
                    "container": {"evidence": "no marker file", "pid1_comm": "systemd", "type": null},
                    "iommu": {"groups": 28, "units": ["ivhd0"]},
                    "memory": {"process_available_bytes": 75248549888u64},
                    "virtualization": {"type": "none", "dmi": {"board_vendor": "ASUSTeK COMPUTER INC.", "product_name": "System Product Name"}}
                },
                "gpus": [
                    {"name": "NVIDIA GeForce RTX 3090", "ordinal": 0, "uuid": "sha256:1dce631f", "pci_bus_id": "0000:01:00.0",
                     "iommu_group": "13", "iommu_group_type": "DMA-FQ",
                     "narrowest_link": {"bdf": "0000:01:00.0", "link": "16.0 GT/s PCIe x16"},
                     "pcie_path": [{"bdf": "0000:00:01.1", "max_width": 16}],
                     "host_link": {"evidence": "integrated=0 sysfs_pcie_link=yes", "kind": "pcie"}},
                    {"name": "NVIDIA GeForce RTX 3090", "pci_bus_id": "0000:02:00.0"}
                ],
                "driver": {
                    "kernel_module": {"version": "615.71.09", "srcversion": "21362C9DF5C1F6DCBE23F50", "banner": "NVRM version: x"},
                    "uvm_module": {"srcversion": "2CA5427BFF504A8F70EEAE8"},
                    "host_registration": {"evidence": "Kernel journal is not active", "status": "unknown"}
                }
            },
            "rows": [
                {"direction": "htod", "max_bytes": 4096, "min_bytes": 0, "settings": {"method": "driver"}},
                {"direction": "dtoh", "max_bytes": 1073741824u64, "min_bytes": 4096,
                 "settings": {"chunk_bytes": 4194304, "cpu_loop": "nt_load@avx2", "method": "staged", "pipeline_workers": 2,
                              "slots_per_worker": 2, "staging_memory": "cached_pinned"}}
            ]
        })
    }

    #[test]
    fn selection_is_exactly_what_cmt_keeps() {
        // cmt's own line for the same selection (src/config.cpp EncodeSelection, field order included)
        let want = concat!(
            r#"{"policy":"measured_steps_v4","file_bytes":10159,"#,
            r#""request":{"search":"full","process_rounds":1,"max_pinned_mib":1024,"progressive_pinning":false,"max_registered_mib":1024,"environment":["CMT_WORKER_CPUS=0-7"]},"#,
            r#""source_machine":{"host":{"cpu":{"model":"AMD Ryzen 7 7800X3D 8-Core Processor","arch":"x86_64"}},"gpus":[{"name":"NVIDIA GeForce RTX 3090"}],"driver":{"kernel_module":{"version":"615.71.09"}}},"#,
            r#""rows":[{"direction":"htod","min_bytes":0,"max_bytes":4096,"settings":{"method":"driver"}},"#,
            r#"{"direction":"dtoh","min_bytes":4096,"max_bytes":1073741824,"settings":{"method":"staged","cpu_loop":"nt_load@avx2","pipeline_workers":2,"slots_per_worker":2,"chunk_bytes":4194304,"staging_memory":"cached_pinned"}}]}"#
        );
        assert_eq!(selection_json(&leaky_selection()).unwrap(), want);
        assert_eq!(public_selection(&leaky_selection()), serde_json::from_str::<Value>(want).unwrap());
        assert_eq!(
            public_machine(&leaky_selection()["source_machine"]),
            serde_json::from_str::<Value>(want).unwrap()["source_machine"]
        );
    }

    #[test]
    fn unknown_fields_are_dropped() {
        // a field a future tuner adds, at every level, never reaches the output
        fn plant(v: &mut Value) {
            match v {
                Value::Object(m) => {
                    m.values_mut().for_each(plant);
                    m.insert("future".into(), json!("GPU-00000000-1111-2222-3333-444444444444"));
                }
                Value::Array(a) => a.iter_mut().for_each(plant),
                _ => {}
            }
        }
        let mut sel = leaky_selection();
        plant(&mut sel);
        // known keys holding an unexpected shape are dropped too
        sel["request"]["search"] = json!({"user": "alice"});
        sel["source_machine"]["host"]["cpu"]["arch"] = json!(["x86_64", "/home/alice"]);
        let s = selection_json(&sel).unwrap();
        for bad in ["future", "00000000-1111", "alice", "search", "\"arch\""] {
            assert!(!s.contains(bad), "{bad} left in {s}");
        }
        assert!(s.contains("\"rows\":[{\"direction\":\"htod\""));
    }

    #[test]
    fn nothing_identifying_survives() {
        let s = selection_json(&leaky_selection()).unwrap();
        for bad in [
            "0000:",
            "bdf",
            "pci_bus_id",
            "iommu",
            "srcversion",
            "21362C9D",
            "dmi",
            "ASUSTeK",
            "container",
            "pid1_comm",
            "process_available",
            "kernel\":{\"release",
            "#2 SMP",
            "evidence",
            "uuid",
            "sha256",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "commit",
            "dirty",
            "151617a3",
            "ordinal",
        ] {
            assert!(!s.contains(bad), "{bad} left in {s}");
        }
    }

    #[test]
    fn not_an_object() {
        assert_eq!(selection_json(&json!(null)), None);
        assert_eq!(public_selection(&json!("x")), Value::Null);
        assert_eq!(public_machine(&json!([1])), Value::Null);
    }

    #[test]
    fn meta_keys() {
        assert!(public_meta_key("driver") && public_meta_key("link.current") && public_meta_key("tuner.grid"));
        assert!(!public_meta_key("tool.commit") && !public_meta_key("host.hostname") && !public_meta_key("anything"));
    }
}
