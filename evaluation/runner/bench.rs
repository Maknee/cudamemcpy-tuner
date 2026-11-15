//! Microbenchmarks: the settings each bench measures, as lists of native child commands.
//!
//! Rules shared by every bench: one fresh process per setting, all sizes inside that process,
//! 2 samples per size, and each sample times what `cmt tune` times, so a bench curve and a tuned
//! table compare. For gpu_copy that is one copy per sample (`--min-bytes 1`, see [`ONE_COPY`]) of a
//! host buffer flushed from the CPU caches just before it; the hooked pages bench uses the
//! `--min-bytes` its table was ranked with. cpu_loops (`--cpu-cache=cold`) times a batch of
//! copies between distinct buffer pairs, every pair flushed before the batch, so each of its
//! copies starts cold too. The flushes are the children's own, so no bench needs root.
//!
//! Resume: `--min-bytes` is part of each item's argv, and so of its saved inputs, so an item saved
//! without it (gpu_copy's default, 64M per sample) is measured again, not reused. A run directory
//! can then hold both; the report refuses to draw a graph from items timed with different
//! `--min-bytes` (plot.py `same_sampling`).
//!
//! `bench kernel` is the one bench that changes the machine: see [`kernel`].
use crate::{
    inline, labels, pending, process, run_items, run_native, run_native_as, tune, Ctx, Item,
    Refusal,
};
use anyhow::{bail, ensure, Context, Result};
use clap::ValueEnum;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tune::Arm;

#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
pub(crate) enum Bench {
    /// Plain driver: pageable, registered, cudaHostAlloc cached and WC; both directions; 4K/THP.
    References,
    /// CPU->GPU 4/16/64/128 KiB: stock vs forced inline vs forced DMA.
    Inline,
    /// Pinned copies forced onto the copy engine, both directions.
    Dma,
    /// Hooked pageable copies from 4K vs THP application buffers, both directions.
    Pages,
    /// Staging through cached vs write-combined pinned slots, both directions.
    Wc,
    /// CPU copy loops: pageable to/from cached/WC pinned memory, 1-16 copying threads.
    Cpu,
    /// Checks FULL's grid: the same search with in-between thread counts, to see whether FULL misses
    /// a better setting. Never used by `cmt run`.
    TuneCheck,
    /// Registration cache, stock vs patched (A+B+D) modules: pin/unpin, pin+copy+unpin,
    /// progressive pinning, the driver and pin_pipeline. Needs passwordless sudo.
    Kernel,
}

pub(crate) const SAMPLES: &str = "2";
/// nvbandwidth-style curve. Capped at 8G: pinning a 16 GiB THP buffer in the 5090's 30 GB
/// container stalled in kernel reclaim for 1.5 h, though gpu_copy's size check passed it.
const GPU_SIZES: &str = "4K..8G";
/// The FULL size grid; cpu_loops takes an explicit list.
const CPU_SIZES: &str = "4K,16K,64K,128K,512K,1M,2M,4M,8M,16M,64M,128M,512M,1G";
const DIRECTIONS: [&str; 2] = ["htod", "dtoh"];
const CPU_CONTROLS: [&str; 5] = [
    "pageable",
    "fill-cached",
    "fill-wc",
    "drain-cached",
    "drain-wc",
];
/// Thread counts for tune-check (the FULL default is 1,2,4,8,16).
const CHECK_THREADS: &str = "threads=1,2,3,4,6,8,12,16";
/// The WC bench's fixed staging layout. A mid-size guess, not a measured optimum: the FULL search
/// measures the others. Only the memory type and CPU loop vary between its items.
const WC_LAYOUT: [&str; 6] = ["--threads", "4", "--slots", "2", "--slot-bytes", "1M"];

pub(crate) fn run(
    ctx: &Ctx,
    bench: Bench,
    sizes: Option<String>,
    config: Option<PathBuf>,
    tuning_params: Option<String>,
) -> Result<()> {
    let name = bench.to_possible_value().expect("no skipped variants");
    let dir = ctx.root.join("bench").join(name.get_name());
    let gpu_sizes = sizes.clone().unwrap_or_else(|| GPU_SIZES.into());
    let items = match bench {
        Bench::References => references(ctx, &gpu_sizes)?,
        Bench::Dma => dma(ctx, &gpu_sizes)?,
        Bench::Wc => wc(ctx, &gpu_sizes)?,
        Bench::Pages => {
            let config = match config {
                Some(path) => path,
                None => ctx.default_config()?,
            };
            pages(ctx, &gpu_sizes, &std::fs::canonicalize(&config)?)?
        }
        Bench::Cpu => cpu(ctx, sizes.as_deref().unwrap_or(CPU_SIZES))?,
        Bench::Inline => return inline::run(ctx, &ctx.root.join("bench").join("inline")),
        Bench::TuneCheck => {
            let params = match tuning_params {
                Some(extra) => format!("{CHECK_THREADS} {extra}"),
                None => CHECK_THREADS.into(),
            };
            return tune::run_search(ctx, "full", &dir, Some(&params), None, None);
        }
        // It swaps modules around its own use of the lock, so it never runs under the caller's.
        Bench::Kernel => bail!("bench kernel is dispatched on its own (bench::kernel)"),
    };
    run_items(
        ctx,
        &dir,
        &items,
        |item| &item.id,
        |item| item.inputs(ctx),
        |item, log| run_native(ctx, item, log, &dir),
    )
}

/// gpu_copy's `--min-bytes` for every bench that runs it: one copy per sample, as `cmt tune` times
/// each setting (`TuneArgs::min_bytes` in src/tuner.h). gpu_copy's own default, 64M, repeats a
/// 4-16 MiB copy 16 to 4 times per sample on the same buffer with one sync: only the first copy
/// is cold, and each copy's last DMA wave overlaps the next, which overstates 2M-32M rates against
/// the tuner's (measured on RTX 3090 and B300).
pub(crate) const ONE_COPY: u64 = 1;

/// `gpu_copy --out {out} -- <copy options> --samples 2 --min-bytes 1 --device N`.
pub(crate) fn gpu_copy(ctx: &Ctx, options: &[&str]) -> Result<Vec<String>> {
    gpu_copy_with(ctx, options, ONE_COPY)
}

/// [`gpu_copy`] with another `--min-bytes`: only for the pages bench, which takes its table's.
fn gpu_copy_with(ctx: &Ctx, options: &[&str], min_bytes: u64) -> Result<Vec<String>> {
    let mut argv = vec![
        path_text(ctx.exe("gpu_copy")?)?,
        "--out".into(),
        "{out}".into(),
        "--".into(),
    ];
    argv.extend(options.iter().map(|option| option.to_string()));
    argv.extend([
        "--samples".into(),
        SAMPLES.into(),
        "--min-bytes".into(),
        min_bytes.to_string(),
        "--device".into(),
        ctx.device.to_string(),
    ]);
    Ok(argv)
}

pub(crate) fn path_text(path: PathBuf) -> Result<String> {
    path.into_os_string()
        .into_string()
        .ok()
        .context("paths passed to children must be UTF-8")
}

fn references(ctx: &Ctx, sizes: &str) -> Result<Vec<Item>> {
    // cudaHostAlloc chooses its own pages, so it has one variant ("driver"), not 4K and THP.
    let memories: [(&str, &str, &[&str]); 6] = [
        (
            "pageable",
            "4k",
            &["--path", "pageable", "--pages", "small"],
        ),
        ("pageable", "thp", &["--path", "pageable", "--pages", "thp"]),
        (
            "registered",
            "4k",
            &[
                "--path",
                "pinned_ce",
                "--alloc",
                "register",
                "--pages",
                "small",
            ],
        ),
        (
            "registered",
            "thp",
            &[
                "--path",
                "pinned_ce",
                "--alloc",
                "register",
                "--pages",
                "thp",
            ],
        ),
        (
            "hostalloc",
            "driver",
            &["--path", "pinned_ce", "--alloc", "hostalloc"],
        ),
        (
            "hostalloc_wc",
            "driver",
            &["--path", "pinned_ce", "--alloc", "hostalloc_wc"],
        ),
    ];
    let mut items = Vec::new();
    for direction in DIRECTIONS {
        for (memory, pages, options) in memories {
            let mut options = options.to_vec();
            options.extend(["--dir", direction, "--sizes", sizes]);
            items.push(Item {
                id: format!("{memory}-{pages}-{direction}"),
                labels: labels(json!({"memory": memory, "pages": pages, "direction": direction})),
                argv: gpu_copy(ctx, &options)?,
                env: Vec::new(),
            });
        }
    }
    Ok(items)
}

fn dma(ctx: &Ctx, sizes: &str) -> Result<Vec<Item>> {
    // CMT_INLINE_POLICY=dma raises no threshold, it lowers the inline limits to zero, so every
    // HtoD copy is encoded for the copy engine. DtoH never uses inline data; the policy is set for
    // symmetry. Machines without a scanned driver gate record "unsupported" (exit 3).
    DIRECTIONS
        .iter()
        .map(|direction| {
            Ok(Item {
                id: format!("dma-{direction}"),
                labels: labels(
                    json!({"memory": "hostalloc", "policy": "dma", "direction": direction}),
                ),
                argv: gpu_copy(
                    ctx,
                    &[
                        "--path",
                        "pinned_ce",
                        "--alloc",
                        "hostalloc",
                        "--dir",
                        direction,
                        "--sizes",
                        sizes,
                    ],
                )?,
                env: vec![("CMT_INLINE_POLICY".into(), "dma".into())],
            })
        })
        .collect()
}

/// The `--min-bytes` the table was ranked with, so the hooked curve times what `cmt tune` chose
/// its rows from. Refused when absent: every tune.json `cmt tune` writes records it, and guessing
/// would bring back the mismatch this bench exists to avoid.
fn table_min_bytes(config: &std::path::Path) -> Result<u64> {
    let tune: serde_json::Value = process::read_json(config)?;
    tune["request"]["min_bytes"]
        .as_u64()
        .filter(|&bytes| bytes > 0)
        .with_context(|| {
            format!(
                "{}: no positive request.min_bytes; pass a tune.json written by `cmt tune`",
                config.display()
            )
        })
}

fn pages(ctx: &Ctx, sizes: &str, config: &std::path::Path) -> Result<Vec<Item>> {
    let min_bytes = table_min_bytes(config)?;
    let config = path_text(config.to_path_buf())?;
    let mut items = Vec::new();
    for direction in DIRECTIONS {
        for (pages, flag) in [("4k", "small"), ("thp", "thp")] {
            let mut argv = vec![
                path_text(ctx.exe("cmt")?)?,
                "run".into(),
                "--config".into(),
                config.clone(),
                "--".into(),
            ];
            argv.extend(gpu_copy_with(
                ctx,
                &[
                    "--path", "pageable", "--pages", flag, "--dir", direction, "--sizes", sizes,
                ],
                min_bytes,
            )?);
            items.push(Item {
                id: format!("hook-{pages}-{direction}"),
                labels: labels(
                    json!({"pages": pages, "direction": direction, "hook": true, "config": config}),
                ),
                argv,
                env: Vec::new(),
            });
        }
    }
    Ok(items)
}

fn wc(ctx: &Ctx, sizes: &str) -> Result<Vec<Item>> {
    let mut items = Vec::new();
    for direction in DIRECTIONS {
        // Streaming stores fill WC memory fast; reading WC needs streaming loads (MOVNTDQA). The ISA
        // is named because a bare name runs the widest variant, and nt_load@avx512 reads WC memory
        // at 2.4 GB/s against 19.0 for @avx2 (4 threads, 1 GiB, drain-wc, RTX 3090 host; 2.3 vs
        // 17.9 on the RTX 5090 host).
        let streaming = if direction == "htod" {
            "nt_store@avx2"
        } else {
            "nt_load@avx2"
        };
        for memory in ["cached", "wc"] {
            for cpu_loop in ["memcpy", streaming] {
                let mut options = vec![
                    "--path", "staged", "--dir", direction, "--sizes", sizes, "--fill", cpu_loop,
                ];
                options.extend(WC_LAYOUT);
                if memory == "wc" {
                    options.push("--wc");
                }
                items.push(Item {
                    id: format!("{memory}-{}-{direction}", cpu_loop.replace('@', "_")),
                    labels: labels(
                        json!({"memory": memory, "cpu_loop": cpu_loop, "direction": direction}),
                    ),
                    argv: gpu_copy(ctx, &options)?,
                    env: Vec::new(),
                });
            }
        }
    }
    Ok(items)
}

// --- bench kernel ---------------------------------------------------------------------------

/// The progressive-pinning request `bench kernel` measures, and only this one: the hook's defaults
/// (`PinningSettings`, src/config.h: first chunk 4 MiB, max chunk 64 MiB) and an explicit 1 GiB
/// registration limit. The fixed limit keeps this bench's sizes and results comparable across
/// runs; larger copies are measured by `cmt tune` grids that include them.
const FIRST_CHUNK: u64 = 4 << 20;
const MAX_CHUNK: u64 = 64 << 20;
const MAX_REGISTERED: u64 = 1 << 30;
/// pin_pipeline registers this far ahead of the copy; a fixed step, not a tuned value.
const PIN_STEP: &str = "32M";
/// Fresh processes per setting and arm, as `cmt tune` runs them (`process_round` 1..3), so the
/// report takes the same median over processes as for every other bench.
const KERNEL_PROCESSES: u32 = 3;

/// Every power of two from the first chunk to the 1 GiB limit. Under that request progressive
/// pinning runs only there: the table lookup skips a progressive row outside
/// first_chunk..max_registered (src/config.cpp, `BuildCopyLookup`), and `cmt tune` with that limit
/// measures it only there (src/tuner.cpp, `pinning_sizes`). At another size the patched hook takes
/// another path and the audited child refuses, so there is nothing to compare, and `bench kernel`
/// does not take `--sizes`.
fn kernel_sizes() -> Vec<u64> {
    std::iter::successors(Some(FIRST_CHUNK), |&bytes| Some(bytes * 2))
        .take_while(|&bytes| bytes <= MAX_REGISTERED)
        .collect()
}

/// One `bench kernel` child.
pub(crate) struct KernelItem {
    arm: Arm,
    item: Item,
    /// Progressive child only: where its CMT_EXPECTED_RUN document goes, and the document.
    request: Option<(PathBuf, String)>,
    refusal: Refusal,
}

/// The `cmt.benchmark-run` request `cmt tune` hands its progressive-pinning child
/// (`ExpectedRunJson`, src/tuner.cpp), for one direction and process round.
fn pinning_request(direction: &str, sizes: &[u64], round: u32) -> String {
    json!({
        "schema": "cmt.benchmark-run", "version": 2, "direction": direction,
        "sizes_bytes": sizes, "process_round": round,
        "settings": {"method": "progressive_pinning", "first_chunk_bytes": FIRST_CHUNK,
                     "max_chunk_bytes": MAX_CHUNK, "max_registered_bytes": MAX_REGISTERED,
                     "read_only": false},
    })
    .to_string()
}

/// The items of one arm, each a gpu_copy child timed by the rule every bench follows
/// ([`gpu_copy`]: 2 samples, one copy per sample).
/// Process rounds are the outer loop, as in `cmt tune`, so slow drift spreads over every setting
/// instead of landing on the last one.
fn kernel_items(ctx: &Ctx, arm: Arm, dir: &Path) -> Result<Vec<KernelItem>> {
    let sizes = kernel_sizes();
    let size_list = sizes
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let hook = path_text(ctx.exe("libcmt_hook.so")?)?;
    let mut found = Vec::new();
    for process in 1..=KERNEL_PROCESSES {
        let mut push = |id: &str, labels: Value, options: &[&str], progressive: Option<&str>| {
            let mut options = options.to_vec();
            options.extend(["--sizes", size_list.as_str()]);
            let mut labels = crate::labels(labels);
            labels.insert("arm".into(), json!(arm.name()));
            labels.insert("process".into(), json!(process));
            let id = format!("{id}-p{process}");
            let (env, request, refusal) = match progressive {
                None => (Vec::new(), None, Refusal::Error),
                Some(direction) => {
                    let path = dir.join("requests").join(format!("{id}.json"));
                    let env = vec![
                        ("CMT_EXPECTED_RUN".to_owned(), path_text(path.clone())?),
                        ("LD_PRELOAD".to_owned(), hook.clone()),
                        ("CMT_TUNE_AUDIT".to_owned(), "1".to_owned()),
                    ];
                    let request = pinning_request(direction, &sizes, process);
                    // Stock modules have no registration cache, so the hook does not pin and the
                    // audit refuses (exit 6): that is the stock arm's answer, saved as `refused`.
                    let refusal = match arm {
                        Arm::Stock => Refusal::Result,
                        Arm::Patched => Refusal::Error,
                    };
                    (env, Some((path, request)), refusal)
                }
            };
            found.push(KernelItem {
                arm,
                item: Item {
                    id,
                    labels,
                    argv: gpu_copy(ctx, &options)?,
                    env,
                },
                request,
                refusal,
            });
            anyhow::Ok(())
        };
        for (pages, flag) in [("4k", "small"), ("thp", "thp")] {
            // Pin and unpin rate of the whole buffer: one row per size, both phases timed.
            push(
                &format!("register-{pages}"),
                json!({"kind": "register", "pages": pages}),
                &["--path", "register", "--pages", flag],
                None,
            )?;
            for direction in DIRECTIONS {
                push(
                    &format!("pin_copy_unpin-{pages}-{direction}"),
                    json!({"kind": "pin_copy_unpin", "pages": pages, "direction": direction}),
                    &[
                        "--path",
                        "pin_copy_unpin",
                        "--pages",
                        flag,
                        "--dir",
                        direction,
                    ],
                    None,
                )?;
            }
        }
        // The rest on THP buffers, as `cmt tune --search full_with_kernel --pages thp` measures.
        for direction in DIRECTIONS {
            let pageable = ["--path", "pageable", "--pages", "thp", "--dir", direction];
            push(
                &format!("progressive-{direction}"),
                json!({"kind": "progressive", "pages": "thp", "direction": direction}),
                &pageable,
                Some(direction),
            )?;
            push(
                &format!("driver-{direction}"),
                json!({"kind": "driver", "pages": "thp", "direction": direction}),
                &pageable,
                None,
            )?;
            // Runs on any modules: the stock stand-in for progressive pinning.
            push(
                &format!("pin_pipeline-{direction}"),
                json!({"kind": "pin_pipeline", "pages": "thp", "direction": direction}),
                &[
                    "--path",
                    "pin_pipeline",
                    "--pin-step",
                    PIN_STEP,
                    "--unpin",
                    "end",
                    "--pages",
                    "thp",
                    "--dir",
                    direction,
                ],
                None,
            )?;
        }
    }
    Ok(found)
}

/// What a `bench kernel` result depends on beyond [`Item::inputs`]: the request document, the
/// preloaded hook (named in the environment, not in argv), the running kernel, and for the
/// patched arm the `cmt` that builds and loads the modules. The loaded modules' identity is
/// recorded in each result (`modules`) but is not an input: it is known only after the patch, and
/// whether to patch at all is decided before it.
fn kernel_inputs(ctx: &Ctx, k: &KernelItem, release: &str) -> Value {
    let preloads: Vec<String> = k
        .item
        .env
        .iter()
        .filter(|(name, _)| name == "LD_PRELOAD")
        .map(|(_, value)| value.clone())
        .collect();
    let patch_tool = (k.arm == Arm::Patched)
        .then(|| process::files_identity(&[ctx.bin.join("cmt").to_string_lossy().into_owned()]));
    json!({
        "item": k.item.inputs(ctx),
        "request": k.request.as_ref().map(|(_, text)| text),
        "preload": process::files_identity(&preloads),
        "kernel": release,
        "patch_tool": patch_tool,
    })
}

/// srcversion of the loaded nvidia and nvidia_uvm modules: which modules a result was measured on.
/// Null where sysfs does not show them; it is a record, not a check.
fn loaded_modules() -> Value {
    let read = |module: &str| {
        std::fs::read_to_string(format!("/sys/module/{module}/srcversion"))
            .ok()
            .map(|text| text.trim().to_owned())
    };
    json!({"nvidia": read("nvidia"), "nvidia_uvm": read("nvidia_uvm")})
}

fn kernel_release() -> Result<String> {
    let path = "/proc/sys/kernel/osrelease";
    Ok(std::fs::read_to_string(path)
        .with_context(|| format!("read {path}"))?
        .trim()
        .to_owned())
}

/// `bench kernel`: stock arm, `cmt patch`, patched arm, restore, through
/// [`tune::with_kernel_arms`], which owns every machine change and its restore (THP `always` for
/// both arms, as full-with-kernel requires it). Results go to `bench/kernel/{stock,patched}/`; the
/// guard's state file is `bench/kernel/kernel-state.json`.
///
/// The measurement lock is held while an arm measures and released before the patch and unpatch,
/// which take it themselves (`cmt patch --measurement-lock`). Nothing on the machine changes when
/// every item of both arms is saved, and the modules are not swapped when the patched arm is.
pub(crate) fn kernel(ctx: &mut Ctx, host: &mut impl tune::Host) -> Result<()> {
    let dir = tune::bench_kernel_dir(ctx);
    let release = kernel_release()?;
    let arms = [Arm::Stock, Arm::Patched]
        .into_iter()
        .map(|arm| {
            let arm_dir = dir.join(arm.name());
            let items = kernel_items(ctx, arm, &arm_dir)?;
            Ok((arm_dir, items))
        })
        .collect::<Result<Vec<_>>>()?;
    let todo: Vec<usize> = arms
        .iter()
        .map(|(arm_dir, items)| {
            pending(
                arm_dir,
                items,
                |k| &k.item.id,
                |k| kernel_inputs(ctx, k, &release),
            )
        })
        .collect();
    let total: usize = arms.iter().map(|(_, items)| items.len()).sum();
    if todo.iter().all(|&n| n == 0) {
        note!(
            "{}: all {total} items done; nothing changed on the machine",
            dir.display()
        );
        return Ok(());
    }
    if todo[1] == 0 {
        note!(
            "{}: the patched arm is complete; the modules are not swapped",
            dir.display()
        );
    }
    tune::with_kernel_arms(host, &dir.join(tune::STATE_FILE), todo[1] > 0, |arm| {
        let (arm_dir, items) = &arms[match arm {
            Arm::Stock => 0,
            Arm::Patched => 1,
        }];
        ctx.lock = Some(process::MeasurementLock::wait(&ctx.lock_path)?);
        let measuring: &Ctx = ctx;
        let result = run_items(
            measuring,
            arm_dir,
            items,
            |k| &k.item.id,
            |k| kernel_inputs(measuring, k, &release),
            |k, log| {
                if let Some((path, text)) = &k.request {
                    std::fs::create_dir_all(path.parent().context("request has no directory")?)?;
                    std::fs::write(path, text)
                        .with_context(|| format!("write {}", path.display()))?;
                }
                let mut result = run_native_as(measuring, &k.item, log, arm_dir, k.refusal)?;
                result["modules"] = loaded_modules();
                Ok(result)
            },
        );
        // Released on every path: the guard's next step (patch or unpatch) takes this lock.
        ctx.lock = None;
        result
    })
}

/// Compiled non-prefetch loop names from `cpu_loops --cpu-help`. Explicit-prefetch loops are a
/// separate experiment, not production loops.
/// Loops the CPU cannot run still get a process; cpu_loops records them as skipped cells.
fn cpu_loop_names(help: &str) -> Vec<String> {
    help.lines()
        .skip_while(|line| !line.starts_with("Compiled native loop names"))
        .skip(1)
        .map(str::trim)
        .filter(|name| !name.is_empty() && !name.contains("prefetch"))
        .map(str::to_owned)
        .collect()
}

/// Every count to 16 (the hook's lane limit), then coarser steps to 32 for many-core hosts, up to the
/// CPUs this process may use (cgroup quota included); a 16-CPU host gets exactly 1..=16.
fn cpu_worker_counts() -> Vec<u32> {
    let usable = std::thread::available_parallelism().map_or(1, |n| n.get() as u32);
    (1..=16)
        .chain([20, 24, 28, 32])
        .filter(|&workers| workers <= usable.max(1))
        .collect()
}

fn cpu(ctx: &Ctx, sizes: &str) -> Result<Vec<Item>> {
    let exe = ctx.exe("cpu_loops")?;
    let help = std::process::Command::new(&exe)
        .arg("--cpu-help")
        .output()?;
    let loops = cpu_loop_names(&String::from_utf8_lossy(&help.stdout));
    ensure!(
        !loops.is_empty(),
        "{} --cpu-help listed no loops",
        exe.display()
    );
    let exe = path_text(exe)?;
    let mut items = Vec::new();
    for control in CPU_CONTROLS {
        for cpu_loop in &loops {
            for &workers in &cpu_worker_counts() {
                // Whole-chunk division: lane 0 is the caller, helpers own whole 1 MiB chunks, so
                // a copy smaller than `workers` chunks uses fewer lanes (recorded per row).
                let mut argv = vec![
                    exe.clone(),
                    format!("--benchmark_repetitions={SAMPLES}"),
                    "--out".into(),
                    "{out}".into(),
                    format!("--cpu-sizes={sizes}"),
                    format!("--cpu-control={control}"),
                    "--cpu-cache=cold".into(),
                    format!("--cpu-workers={workers}"),
                    format!("--cpu-loops={cpu_loop}"),
                ];
                if control != "pageable" {
                    argv.push(format!("--cpu-device={}", ctx.device));
                }
                items.push(Item {
                    id: format!("{control}-{}-t{workers:02}", cpu_loop.replace('@', "_")),
                    labels: labels(
                        json!({"control": control, "cpu_loop": cpu_loop, "workers": workers}),
                    ),
                    argv,
                    env: Vec::new(),
                });
            }
        }
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_names_skip_prefetch_and_header() {
        let help = "--cpu-help text\nCompiled native loop names (unsupported ...):\n  memcpy\n  \
                    nt_store@avx2\n  nt_store_prefetch@avx2\n";
        assert_eq!(cpu_loop_names(help), ["memcpy", "nt_store@avx2"]);
        assert!(cpu_loop_names("no list here").is_empty());
    }

    #[test]
    fn wc_items_name_the_isa_of_every_streaming_loop() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("gpu_copy"), "").unwrap();
        let items = wc(&crate::tests::ctx(dir.path()), "4K").unwrap();
        assert_eq!(items.len(), 8);
        for item in &items {
            let fill = &item.argv[item.argv.iter().position(|a| a == "--fill").unwrap() + 1];
            assert!(fill == "memcpy" || fill.ends_with("@avx2"), "{fill}");
            assert_eq!(item.labels["cpu_loop"], fill.as_str());
            assert!(!item.id.contains('@'), "{}", item.id);
        }
    }

    /// Every `--min-bytes` value passed to gpu_copy, which may run under `cmt run ... --`.
    fn min_bytes(argv: &[String]) -> Vec<&str> {
        let start = argv
            .iter()
            .position(|arg| arg.ends_with("/gpu_copy"))
            .expect("the item runs gpu_copy");
        argv[start..]
            .windows(2)
            .filter(|pair| pair[0] == "--min-bytes")
            .map(|pair| pair[1].as_str())
            .collect()
    }

    /// A temporary bin directory with (empty) cmt and gpu_copy, and a tune.json with `request`.
    fn bin_dir_with_table(request: serde_json::Value) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        for exe in ["cmt", "gpu_copy", "libcmt_hook.so"] {
            std::fs::write(dir.path().join(exe), "").unwrap();
        }
        let config = dir.path().join("tune.json");
        std::fs::write(&config, json!({ "request": request }).to_string()).unwrap();
        (dir, config)
    }

    #[test]
    fn every_gpu_copy_bench_times_one_copy_per_sample() {
        let (dir, config) = bin_dir_with_table(json!({"min_bytes": 1}));
        let ctx = crate::tests::ctx(dir.path());
        let mut items = references(&ctx, GPU_SIZES).unwrap();
        items.extend(dma(&ctx, GPU_SIZES).unwrap());
        items.extend(wc(&ctx, GPU_SIZES).unwrap());
        items.extend(pages(&ctx, GPU_SIZES, &config).unwrap());
        items.extend(inline::points(&ctx).unwrap());
        for arm in [Arm::Stock, Arm::Patched] {
            let found = kernel_items(&ctx, arm, &dir.path().join(arm.name())).unwrap();
            items.extend(found.into_iter().map(|k| k.item));
        }
        // references 12, dma 2, wc 8, pages 4, inline timings 72, kernel 2 x 36: every gpu_copy
        // item there is.
        assert_eq!(items.len(), 12 + 2 + 8 + 4 + 72 + 72);
        for item in &items {
            assert_eq!(min_bytes(&item.argv), ["1"], "{}", item.id);
        }
    }

    #[test]
    fn pages_bench_times_copies_as_its_table_was_ranked() {
        let (dir, config) = bin_dir_with_table(json!({"min_bytes": 4096}));
        let ctx = crate::tests::ctx(dir.path());
        for item in pages(&ctx, "4K", &config).unwrap() {
            assert_eq!(min_bytes(&item.argv), ["4096"], "{}", item.id);
        }
        // A table that does not say how it was timed is refused, not guessed.
        for request in [
            json!({}),
            json!({"min_bytes": 0}),
            json!({"min_bytes": "1"}),
        ] {
            std::fs::write(&config, json!({ "request": request }).to_string()).unwrap();
            let Err(error) = pages(&ctx, "4K", &config) else {
                panic!("pages accepted the request {request}");
            };
            assert!(
                format!("{error:#}").contains("request.min_bytes"),
                "{error:#}"
            );
        }
    }

    /// A machine for `bench kernel` with no GPU and no root. The fake Host (tune.rs) plays the
    /// module swap and THP; on a successful patch it writes `patched` to the file `modules`, on
    /// unpatch `stock`. A fake gpu_copy plays every child: it appends `<arm>/<item> <modules>` to
    /// `calls`, and answers as gpu_copy does: an audited progressive child (CMT_EXPECTED_RUN set)
    /// on stock modules refuses with exit 6 and rows saying "path/count mismatch" (or the reason in
    /// the file `refusal`); an item whose output path contains the text in `fail` exits 1; any
    /// other child completes.
    struct Machine {
        _dir: tempfile::TempDir,
        root: PathBuf,
        ctx: Ctx,
        fake: crate::tune::tests::Fake,
    }

    const ARM_ITEMS: usize = 36;

    impl Machine {
        fn new() -> Self {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path().to_path_buf();
            let script = format!(
                r#"#!/bin/sh
out=$2
modules=$(cat {root}/modules 2>/dev/null || echo stock)
item=${{out#{root}/bench/kernel/}}
echo "${{item%.partial.json}} $modules" >> {root}/calls
fail=$(cat {root}/fail 2>/dev/null)
if [ -n "$fail" ]; then case "$out" in *"$fail"*) exit 1;; esac; fi
if [ -n "$CMT_EXPECTED_RUN" ]; then
  [ -f "$CMT_EXPECTED_RUN" ] && [ -f "$LD_PRELOAD" ] && [ "$CMT_TUNE_AUDIT" = 1 ] || exit 2
  if [ "$modules" = stock ]; then
    reason=$(cat {root}/refusal 2>/dev/null || echo 'path/count mismatch')
    printf '{{"schema":"cmt.measurements","source_machine":{{}},"runs":[{{"status":"failed","reason":"%s"}}]}}' "$reason" > "$out"
    exit 6
  fi
fi
printf '{{"schema":"cmt.measurements","source_machine":{{}},"runs":[{{"status":"complete","gbps":1}}]}}' > "$out"
"#,
                root = root.display()
            );
            let gpu_copy = root.join("gpu_copy");
            std::fs::write(&gpu_copy, script).unwrap();
            std::fs::set_permissions(&gpu_copy, std::fs::Permissions::from_mode(0o755)).unwrap();
            std::fs::write(root.join("libcmt_hook.so"), "").unwrap();
            let ctx = crate::tests::ctx(&root);
            let mut fake = crate::tune::tests::Fake::new(&root);
            fake.state_file = tune::bench_kernel_dir(&ctx).join(tune::STATE_FILE);
            let (modules, lock) = (root.join("modules"), ctx.lock_path.clone());
            fake.on_step = Some(Box::new(move |step| match step {
                "patch" => {
                    // `cmt patch --measurement-lock` takes the lock: the runner must not hold it.
                    assert!(
                        lock_is_free(&lock),
                        "the measurement lock is held across the patch"
                    );
                    std::fs::write(&modules, "patched").unwrap();
                }
                "unpatch" => std::fs::write(&modules, "stock").unwrap(),
                _ => {}
            }));
            Machine {
                _dir: dir,
                root,
                ctx,
                fake,
            }
        }

        fn run(&mut self) -> Result<()> {
            kernel(&mut self.ctx, &mut self.fake)
        }

        fn calls(&self) -> Vec<String> {
            let text = std::fs::read_to_string(self.root.join("calls")).unwrap_or_default();
            text.lines().map(str::to_owned).collect()
        }

        fn calls_in(&self, arm: Arm) -> Vec<String> {
            let prefix = format!("{}/", arm.name());
            self.calls()
                .into_iter()
                .filter(|call| call.starts_with(&prefix))
                .collect()
        }

        fn saved(&self, arm: Arm, id: &str) -> Option<Value> {
            let dir = tune::bench_kernel_dir(&self.ctx).join(arm.name());
            process::read_json(&dir.join(format!("{id}.json"))).ok()
        }

        fn saved_count(&self, arm: Arm) -> usize {
            let dir = tune::bench_kernel_dir(&self.ctx).join(arm.name());
            std::fs::read_dir(dir)
                .unwrap()
                .filter(|entry| {
                    let name = entry.as_ref().unwrap().file_name();
                    let name = name.to_string_lossy();
                    name.ends_with(".json") && name != "machine.json"
                })
                .count()
        }

        fn restored(&self) -> bool {
            !self.fake.state_file.exists() && self.fake.thp == "madvise"
        }
    }

    /// Other tests fork concurrently, and a forked child holds a copy of a lock descriptor until it
    /// execs, so a free lock can take a moment to show as free.
    fn lock_is_free(path: &Path) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if process::MeasurementLock::acquire(path).is_ok() {
                return true;
            }
            if std::time::Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    #[test]
    fn kernel_bench_measures_stock_then_patched_and_restores() {
        let mut machine = Machine::new();
        machine.run().unwrap();
        assert_eq!(
            machine.fake.log,
            ["thp=always", "patch", "unpatch", "thp=madvise"]
        );
        assert!(machine.restored());
        // Every stock child ran before the patch on stock modules, every patched one after it.
        let calls = machine.calls();
        assert_eq!(calls.len(), 2 * ARM_ITEMS);
        let (stock, patched) = calls.split_at(ARM_ITEMS);
        assert!(
            stock
                .iter()
                .all(|c| c.starts_with("stock/") && c.ends_with(" stock")),
            "{stock:?}"
        );
        assert!(
            patched
                .iter()
                .all(|c| c.starts_with("patched/") && c.ends_with(" patched")),
            "{patched:?}"
        );
        // The stock refusal is saved as the result; the same child on patched modules completes.
        for direction in DIRECTIONS {
            for process in 1..=KERNEL_PROCESSES {
                let id = format!("progressive-{direction}-p{process}");
                let refused = machine.saved(Arm::Stock, &id).unwrap();
                assert_eq!(refused["status"], "refused", "{id}");
                assert_eq!(refused["runs"][0]["reason"], "path/count mismatch");
                assert_eq!(
                    machine.saved(Arm::Patched, &id).unwrap()["status"],
                    "complete"
                );
            }
        }
        // The child's request names the default settings, the direction and the process round.
        let request: Value = process::read_json(
            &tune::bench_kernel_dir(&machine.ctx).join("stock/requests/progressive-dtoh-p2.json"),
        )
        .unwrap();
        assert_eq!(request["direction"], "dtoh");
        assert_eq!(request["process_round"], 2);
        assert_eq!(request["sizes_bytes"], json!(kernel_sizes()));
        assert_eq!(request["settings"]["max_registered_bytes"], 1 << 30);
        assert_eq!(kernel_sizes().first(), Some(&(4 << 20)));
        assert_eq!(kernel_sizes().last(), Some(&(1 << 30)));
        // A rerun with everything saved changes nothing on the machine and runs nothing.
        machine.fake.log.clear();
        machine.run().unwrap();
        assert!(machine.fake.log.is_empty(), "{:?}", machine.fake.log);
        assert_eq!(machine.calls().len(), 2 * ARM_ITEMS);
    }

    #[test]
    fn failed_patch_runs_no_patched_item() {
        let mut machine = Machine::new();
        machine.fake.fail = "patch";
        assert!(machine.run().is_err());
        assert_eq!(machine.fake.log, ["thp=always", "patch", "thp=madvise"]);
        assert!(machine.restored());
        assert_eq!(machine.calls_in(Arm::Stock).len(), ARM_ITEMS);
        assert!(machine.calls_in(Arm::Patched).is_empty());
    }

    #[test]
    fn failed_patched_item_still_unpatches_and_is_retried_alone() {
        let mut machine = Machine::new();
        std::fs::write(machine.root.join("fail"), "patched/driver-htod-p2").unwrap();
        let error = machine.run().unwrap_err();
        assert!(
            format!("{error:#}").contains(&format!("1 of {ARM_ITEMS} items failed")),
            "{error:#}"
        );
        assert_eq!(
            machine.fake.log,
            ["thp=always", "patch", "unpatch", "thp=madvise"]
        );
        assert!(machine.restored());
        // The other patched items still ran and were saved.
        assert_eq!(machine.calls_in(Arm::Patched).len(), ARM_ITEMS);
        assert_eq!(machine.saved_count(Arm::Patched), ARM_ITEMS - 1);
        // The rerun patches again for the one missing item, and runs only it.
        std::fs::remove_file(machine.root.join("fail")).unwrap();
        machine.fake.log.clear();
        machine.run().unwrap();
        assert_eq!(
            machine.fake.log,
            ["thp=always", "patch", "unpatch", "thp=madvise"]
        );
        assert_eq!(machine.calls_in(Arm::Stock).len(), ARM_ITEMS);
        assert_eq!(machine.calls_in(Arm::Patched).len(), ARM_ITEMS + 1);
    }

    #[test]
    fn cancellation_in_either_arm_restores_the_machine() {
        use std::sync::atomic::Ordering;
        // Ctrl-C as the patch finishes: no patched child runs; modules and THP are restored.
        let mut machine = Machine::new();
        let flag = machine.ctx.cancel.signal.clone();
        let mut previous = machine.fake.on_step.take().unwrap();
        machine.fake.on_step = Some(Box::new(move |step| {
            previous(step);
            if step == "patch" {
                flag.store(2, Ordering::Relaxed);
            }
        }));
        let error = machine.run().unwrap_err();
        assert!(format!("{error:#}").contains("signal 2"), "{error:#}");
        assert_eq!(
            machine.fake.log,
            ["thp=always", "patch", "unpatch", "thp=madvise"]
        );
        assert!(machine.restored());
        assert!(machine.calls_in(Arm::Patched).is_empty());
        // Ctrl-C during the stock arm: no patch at all, THP restored.
        let mut machine = Machine::new();
        let flag = machine.ctx.cancel.signal.clone();
        machine.fake.on_step = Some(Box::new(move |step| {
            if step == "thp=always" {
                flag.store(15, Ordering::Relaxed);
            }
        }));
        let error = machine.run().unwrap_err();
        assert!(format!("{error:#}").contains("signal 15"), "{error:#}");
        assert_eq!(machine.fake.log, ["thp=always", "thp=madvise"]);
        assert!(machine.restored());
        assert!(machine.calls().is_empty());
    }

    #[test]
    fn failed_stock_arm_stops_before_the_patch() {
        let mut machine = Machine::new();
        std::fs::write(machine.root.join("fail"), "stock/register-4k-p1").unwrap();
        assert!(machine.run().is_err());
        assert_eq!(machine.fake.log, ["thp=always", "thp=madvise"]);
        assert!(machine.restored());
        assert_eq!(machine.calls_in(Arm::Stock).len(), ARM_ITEMS);
        assert!(machine.calls_in(Arm::Patched).is_empty());
    }

    #[test]
    fn only_a_path_refusal_on_stock_modules_is_a_result() {
        // Exit 6 also means wrong bytes; with any other reason it is a failure, never saved.
        let mut machine = Machine::new();
        std::fs::write(machine.root.join("refusal"), "verification failed").unwrap();
        let error = machine.run().unwrap_err();
        assert!(
            format!("{error:#}").contains(&format!("6 of {ARM_ITEMS} items failed")),
            "{error:#}"
        );
        assert!(machine.saved(Arm::Stock, "progressive-htod-p1").is_none());
        assert_eq!(machine.fake.log, ["thp=always", "thp=madvise"]);
        // On patched modules a refusal is a failure: here the "patch" leaves the modules stock.
        let mut machine = Machine::new();
        machine.fake.on_step = None;
        let error = machine.run().unwrap_err();
        assert!(
            format!("{error:#}").contains(&format!("6 of {ARM_ITEMS} items failed")),
            "{error:#}"
        );
        assert_eq!(
            machine.saved(Arm::Stock, "progressive-htod-p1").unwrap()["status"],
            "refused"
        );
        assert!(machine.saved(Arm::Patched, "progressive-htod-p1").is_none());
        assert_eq!(
            machine.fake.log,
            ["thp=always", "patch", "unpatch", "thp=madvise"]
        );
    }
}
