//! Inline vs DMA for small CPU->GPU copies: 72 timings.
//!
//! The driver can embed a small copy's bytes in the GPU command stream ("inline") instead of
//! asking the copy engine to read host memory. `CMT_INLINE_POLICY=stock|inline|dma` makes
//! gpu_copy patch the driver's size gate in its own process (x86-64, scanned drivers only).
use crate::{bench, labels, run_items, run_native, Ctx, Item};
use anyhow::Result;
use serde_json::json;
use std::path::Path;

const BYTES: [u64; 4] = [4 << 10, 16 << 10, 64 << 10, 128 << 10];
const PATHS: [&str; 2] = ["pageable", "pinned"];
const POLICIES: [&str; 3] = ["stock", "inline", "dma"];
/// Fresh processes per point (4 x 2 x 3 x 3 = 72 timings); the spread across them shows
/// process-to-process variation.
const PROCESSES: u32 = 3;

fn copy_options(path: &str, bytes: u64) -> Vec<String> {
    let mut options: Vec<String> = match path {
        "pageable" => ["--path", "pageable", "--pages", "small"]
            .map(String::from)
            .to_vec(),
        _ => ["--path", "pinned_ce", "--alloc", "hostalloc"]
            .map(String::from)
            .to_vec(),
    };
    options.extend([
        "--dir".into(),
        "htod".into(),
        "--sizes".into(),
        bytes.to_string(),
    ]);
    options
}

fn point_id(bytes: u64, path: &str, policy: &str) -> String {
    format!("{}k-{path}-{policy}", bytes >> 10)
}

/// The timing items. They go through `bench::gpu_copy`, so (like `cmt tune`) each sample times
/// one copy.
pub(crate) fn points(ctx: &Ctx) -> Result<Vec<Item>> {
    let mut timings = Vec::new();
    for bytes in BYTES {
        for path in PATHS {
            for policy in POLICIES {
                let options = copy_options(path, bytes);
                let options: Vec<&str> = options.iter().map(String::as_str).collect();
                let base = labels(json!({"bytes": bytes, "path": path, "policy": policy}));
                for process in 1..=PROCESSES {
                    let mut labels = base.clone();
                    labels.insert("process".into(), json!(process));
                    timings.push(Item {
                        id: format!("{}-p{process}", point_id(bytes, path, policy)),
                        labels,
                        argv: bench::gpu_copy(ctx, &options)?,
                        env: vec![("CMT_INLINE_POLICY".into(), policy.into())],
                    });
                }
            }
        }
    }
    Ok(timings)
}

pub(crate) fn run(ctx: &Ctx, dir: &Path) -> Result<()> {
    run_items(
        ctx,
        dir,
        &points(ctx)?,
        |item| &item.id,
        |item| item.inputs(ctx),
        |item, log| run_native(ctx, item, log, dir),
    )
}
