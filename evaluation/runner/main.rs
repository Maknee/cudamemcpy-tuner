//! `cmt-evaluate`: run CMT's evaluation on any Linux machine with an NVIDIA GPU.
//!
//! Every command writes under `results/<name>/<command>/` and is resumable: one JSON file per
//! finished item, written atomically when the item finishes. Rerunning the same command skips
//! finished items whose recorded inputs (`Ctx::identity`: argv, environment, size and mtime of
//! every file named in argv or preloaded by `cmt`, NVIDIA driver, GPU) match the current ones, and
//! reruns the others. There is no plan file and no admission
//! step: the native children (`cmt`, `gpu_copy`, `cpu_loops`) check their own bytes and capacity,
//! and this runner records what they return. Before each item it waits for a quiet machine
//! (`process::Quiet`) and records the load at the start and end of the item.

/// eprintln! that never panics: after SIGHUP or a closed pipe, stderr may be gone, and the runner
/// must still finish restoring the machine.
macro_rules! note {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), $($arg)*);
    }};
}
pub(crate) use note;

mod bench;
mod inline;
mod process;
mod tune;
mod workloads;

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use process::{Cancellation, MeasurementLock, Quiet};
use serde_json::{json, Map, Value};
use std::{
    cell::RefCell,
    fs::File,
    os::unix::process::ExitStatusExt,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
#[command(
    version,
    about = "Run CMT evaluations, resumable, one result file per finished item"
)]
struct Cli {
    #[command(flatten)]
    global: Global,
    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
struct Global {
    /// Results root; output goes to <results>/<name>/<command>/.
    #[arg(long, global = true, default_value = "results")]
    results: PathBuf,
    /// Run name; reuse it to resume. Default: the host name.
    #[arg(long, global = true)]
    name: Option<String>,
    /// Directory with cmt, gpu_copy, cpu_loops and the hook libraries.
    /// Default: $CMT_BIN_DIR, else this executable's directory.
    #[arg(long, global = true)]
    bin_dir: Option<PathBuf>,
    /// CUDA device index, as seen through CUDA_VISIBLE_DEVICES.
    #[arg(long, global = true, default_value_t = 0)]
    device: u32,
    /// Measurement lock shared with `cmt tune` and `cmt patch`.
    #[arg(long, global = true, default_value_os_t = MeasurementLock::default_path())]
    lock: PathBuf,
    /// Wait before each measured item until the 1-minute load average, minus this runner's own
    /// children, is below this (0: do not wait).
    #[arg(long, global = true, default_value_t = 1.0)]
    max_load: f64,
    /// Progress bar on a terminal; plain lines otherwise.
    #[arg(long, global = true, value_enum, default_value_t = Toggle::On)]
    progress: Toggle,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Toggle {
    On,
    Off,
}

#[derive(Subcommand)]
enum Command {
    /// Run `cmt tune`; `full` also runs the references and inline benches.
    Tune {
        /// Search tier; `cmt tune --search` defaults to quick too (src/cli.cpp).
        #[arg(value_enum, default_value_t = tune::Tier::Quick)]
        tier: tune::Tier,
        /// Passed to `cmt tune --tuning_params`, e.g. "sizes=4K,1M threads=1,2,4 samples=2".
        #[arg(long = "tuning_params", alias = "tuning-params")]
        tuning_params: Option<String>,
        /// cpu-to-gpu, gpu-to-cpu or both (cmt tune's default: both).
        #[arg(long)]
        direction: Option<String>,
    },
    /// Microbenchmarks with the native children; one process per setting, 2 samples per size.
    Bench {
        #[arg(value_enum)]
        bench: bench::Bench,
        /// Copy sizes passed to the child (gpu_copy accepts ranges such as 4K..1G).
        #[arg(long)]
        sizes: Option<String>,
        /// tune.json for the hooked benches (pages); default: newest tier under tune/.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Extra `--tuning_params` for tune-check.
        #[arg(long = "tuning_params", alias = "tuning-params")]
        tuning_params: Option<String>,
    },
    /// Application workloads from the catalog.
    Workload {
        #[command(subcommand)]
        action: workloads::Action,
    },
    /// Graphs and one HTML page from everything under results/<name>/, or from several result
    /// directories side by side (one figure per run in each section; needs --out).
    Report {
        #[arg(long, default_value = "python3")]
        python: PathBuf,
        /// Result directories to draw; default: results/<name>.
        runs: Vec<PathBuf>,
        /// Report directory; default for one run: <run>/report.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

/// What every command needs. `lock` is held while a measuring command runs.
pub(crate) struct Ctx {
    pub(crate) root: PathBuf,
    pub(crate) bin: PathBuf,
    pub(crate) device: u32,
    /// None for commands that only read results, and where nvidia-smi is not installed.
    pub(crate) gpu: Option<process::Gpu>,
    pub(crate) lock_path: PathBuf,
    pub(crate) lock: Option<MeasurementLock>,
    pub(crate) progress: bool,
    pub(crate) cancel: Cancellation,
    pub(crate) quiet: RefCell<Quiet>,
}

impl Ctx {
    pub(crate) fn exe(&self, name: &str) -> Result<PathBuf> {
        let path = self.bin.join(name);
        ensure!(
            path.is_file(),
            "{} not found; build it or pass --bin-dir",
            path.display()
        );
        Ok(path)
    }

    /// What a result made by a child with `argv` depends on beyond argv and environment: the files
    /// it names or preloads, the NVIDIA driver and the GPU. Every resume check goes through here.
    pub(crate) fn identity(&self, argv: &[String]) -> Value {
        json!({"files": process::files_identity(argv), "gpu": self.gpu})
    }

    pub(crate) fn acquire_lock(&mut self) -> Result<()> {
        if self.lock.is_none() {
            self.lock = Some(MeasurementLock::wait(&self.lock_path)?);
        }
        Ok(())
    }

    /// The tune.json of the most thorough complete stock-module tier, for hooked runs without
    /// --config. full-with-kernel is never a default: its settings apply only on patched modules.
    pub(crate) fn default_config(&self) -> Result<PathBuf> {
        for tier in ["full", "quick", "instant"] {
            let path = self.root.join("tune").join(tier).join("tune.json");
            if process::read_json::<Value>(&path).is_ok_and(|tune| tune["state"] == "complete") {
                return Ok(path);
            }
        }
        bail!(
            "no complete tune.json under {}; run `cmt-evaluate tune quick` first or pass --config",
            self.root.join("tune").display()
        )
    }
}

/// One unit of resumable work: a child command whose result is saved as `<dir>/<id>.json`.
/// `{out}` in `argv` is replaced by a fresh path the child writes its measurements to.
#[derive(Clone)]
pub(crate) struct Item {
    pub(crate) id: String,
    pub(crate) labels: Map<String, Value>,
    pub(crate) argv: Vec<String>,
    pub(crate) env: Vec<(String, String)>,
}

impl Item {
    /// What a saved result must have been made from to be reused.
    pub(crate) fn inputs(&self, ctx: &Ctx) -> Value {
        json!({"argv": self.argv, "env": self.env, "identity": ctx.identity(&self.argv)})
    }
}

pub(crate) fn labels(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => unreachable!("labels are built with json!({{...}})"),
    }
}

/// Print a line without tearing the progress bar.
pub(crate) fn say(bar: &ProgressBar, message: impl AsRef<str>) {
    if bar.is_hidden() {
        note!("{}", message.as_ref());
    } else {
        bar.println(message.as_ref());
    }
}

/// The saved result of the item `id` in `dir`, if any (made from whatever inputs).
fn saved(dir: &Path, id: &str) -> Option<Value> {
    process::read_json::<Value>(&dir.join(format!("{id}.json"))).ok()
}

/// The one resume rule: an item is done when its saved result was made from these `inputs`.
fn is_done(dir: &Path, id: &str, inputs: &Value) -> bool {
    saved(dir, id).is_some_and(|result| result["inputs"] == *inputs)
}

/// How many of `items` [`run_items`] would run: those without a saved result made from the same
/// `inputs`. For commands that change the machine only when there is something left to measure.
pub(crate) fn pending<T>(
    dir: &Path,
    items: &[T],
    id: impl Fn(&T) -> &str,
    inputs: impl Fn(&T) -> Value,
) -> usize {
    items
        .iter()
        .filter(|item| !is_done(dir, id(item), &inputs(item)))
        .count()
}

/// Run `items` in order, skipping those with a saved result made from the same `inputs`. A failed
/// item is logged and retried on the next invocation; the others still run. Cancellation stops
/// at once.
pub(crate) fn run_items<T>(
    ctx: &Ctx,
    dir: &Path,
    items: &[T],
    id: impl Fn(&T) -> &str,
    inputs: impl Fn(&T) -> Value,
    mut execute: impl FnMut(&T, &Path) -> Result<Value>,
) -> Result<()> {
    std::fs::create_dir_all(dir.join("logs"))?;
    let saved = |item: &T| saved(dir, id(item));
    let done = |item: &T| is_done(dir, id(item), &inputs(item));
    let finished = items.iter().filter(|item| done(item)).count();
    let changed = items
        .iter()
        .filter(|item| saved(item).is_some() && !done(item))
        .count();
    if changed > 0 {
        note!(
            "{}: {changed} saved items were made from different inputs; rerunning them",
            dir.display()
        );
    }
    let bar = if ctx.progress {
        ProgressBar::new(items.len() as u64).with_style(ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:28.cyan/blue}] {pos}/{len} {msg}",
        )?)
    } else {
        ProgressBar::hidden()
    };
    bar.set_position(finished as u64);
    say(
        &bar,
        format!(
            "{}: {} items, {finished} already done",
            dir.display(),
            items.len()
        ),
    );
    let mut failures = Vec::new();
    for item in items {
        ctx.cancel.check()?;
        if done(item) {
            continue;
        }
        let name = id(item);
        bar.set_message(name.to_owned());
        let log = dir.join("logs").join(format!("{name}.log"));
        ctx.quiet.borrow().wait(&ctx.cancel)?;
        let before = ctx.quiet.borrow().snapshot();
        // Taken before the run: a file rebuilt while it runs must not be saved as its input.
        let used = inputs(item);
        let started = Instant::now();
        let outcome = execute(item, &log);
        ctx.quiet.borrow_mut().child_finished(started);
        match outcome {
            Ok(mut result) => {
                result["inputs"] = used;
                result["load"] = json!({"start": before, "end": ctx.quiet.borrow().snapshot()});
                let status = result["status"].as_str().unwrap_or("complete").to_owned();
                process::write_json(&dir.join(format!("{name}.json")), &result)?;
                if status != "complete" {
                    let reason = result["reason"].as_str().unwrap_or("");
                    say(&bar, format!("{name}: {status} {reason}"));
                }
                bar.inc(1);
            }
            Err(error) => {
                ctx.cancel.check()?;
                say(&bar, format!("{name}: failed: {error:#}"));
                failures.push(name.to_owned());
            }
        }
    }
    bar.finish_and_clear();
    ensure!(
        failures.is_empty(),
        "{} of {} items failed ({}); logs in {}; rerun the same command to retry them",
        failures.len(),
        items.len(),
        failures.join(", "),
        dir.join("logs").display()
    );
    note!("{}: all {} items done", dir.display(), items.len());
    Ok(())
}

/// Run one native child that writes `cmt.measurements` JSON, and turn its outcome into the saved
/// item result. Exit 3 (unsupported here: no inline gate for this driver, an ISA the CPU lacks)
/// is a property of the machine and is saved as a final answer. Every other failure, including
/// 77 (no GPU visible right now), is an error and is retried on the next run.
pub(crate) fn run_native(ctx: &Ctx, item: &Item, log: &Path, dir: &Path) -> Result<Value> {
    run_native_as(ctx, item, log, dir, Refusal::Error)
}

/// What a gpu_copy audit refusal (exit 6 with every failed row "path/count mismatch") means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Refusal {
    /// A failure like any other: logged, not saved, retried.
    Error,
    /// The expected answer (progressive pinning on stock modules): saved as status `refused`.
    Result,
}

/// Exit code of `Error::Kind::Verify` (src/common/error.h, `Error::ExitCode`).
const EXIT_VERIFY: i32 = 6;
/// gpu_copy's reason when the hook took another path than the request named, or another number of
/// copies (`TuneAudit`, src/measure/gpu_copy.cpp). Exit 6 with any other reason (wrong
/// bytes, a failed pinning check) is never a refusal.
const PATH_REFUSAL: &str = "path/count mismatch";

/// True when the failed rows of `runs` are all audit refusals, and there is at least one.
fn only_path_refusals(runs: &Value) -> bool {
    let failed: Vec<&Value> = runs
        .as_array()
        .into_iter()
        .flatten()
        .filter(|row| row["status"] != "complete")
        .collect();
    !failed.is_empty() && failed.iter().all(|row| row["reason"] == PATH_REFUSAL)
}

/// [`run_native`], where `refusal` says whether an audit refusal is a result.
pub(crate) fn run_native_as(
    ctx: &Ctx,
    item: &Item,
    log: &Path,
    dir: &Path,
    refusal: Refusal,
) -> Result<Value> {
    let partial = dir.join(format!("{}.partial.json", item.id));
    // A previous interrupted attempt may have left a partial file; children refuse to overwrite.
    let _ = std::fs::remove_file(&partial);
    let out = partial.to_str().context("result path must be UTF-8")?;
    let argv: Vec<String> = item.argv.iter().map(|a| a.replace("{out}", out)).collect();
    let mut command = process::clean_command(Path::new(&argv[0]));
    let file = File::create(log)?;
    command
        .args(&argv[1..])
        .envs(item.env.iter().map(|(k, v)| (k, v)))
        .stdin(std::process::Stdio::null())
        .stdout(file.try_clone()?)
        .stderr(file);
    let started = Instant::now();
    let status = process::run_child(&mut command, ctx.lock.as_ref(), &ctx.cancel)?;
    let seconds = started.elapsed().as_secs_f64();
    ctx.cancel.check()?;
    let code = status.code();
    let (state, reason) = match code {
        Some(0) => ("complete", String::new()),
        Some(3) => ("unsupported", process::tail(log, 400)),
        Some(EXIT_VERIFY) if refusal == Refusal::Result => ("refused", PATH_REFUSAL.to_owned()),
        _ => bail!(
            "exit {:?} signal {:?}: {}",
            code,
            status.signal(),
            process::tail(log, 400)
        ),
    };
    let mut runs = Value::Array(Vec::new());
    if partial.is_file() {
        let mut document: Value = process::read_json(&partial)?;
        ensure!(
            document["schema"] == "cmt.measurements",
            "{} is not cmt.measurements",
            partial.display()
        );
        let machine = dir.join("machine.json");
        if !machine.is_file() {
            process::write_json(&machine, &document["source_machine"])?;
        }
        runs = document["runs"].take();
        std::fs::remove_file(&partial)?;
    } else {
        ensure!(state != "complete", "child succeeded without writing {out}");
    }
    // A refusal is saved only when the child's own rows say so; exit 6 also means wrong bytes.
    ensure!(
        state != "refused" || only_path_refusals(&runs),
        "exit 6 without only {PATH_REFUSAL:?} rows, so not a refusal: {}",
        process::tail(log, 400)
    );
    Ok(json!({
        "id": item.id, "labels": item.labels, "argv": item.argv,
        "env": item.env.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<Map<_, _>>(),
        "status": state, "reason": reason, "exit_code": code, "seconds": seconds, "runs": runs,
    }))
}

fn host_name() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|name| name.trim().to_owned())
        .ok()
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "local".into())
}

fn bin_dir(flag: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(dir) = flag.or_else(|| std::env::var_os("CMT_BIN_DIR").map(PathBuf::from)) {
        return std::fs::canonicalize(&dir).with_context(|| format!("bin dir {}", dir.display()));
    }
    let exe = std::env::current_exe()?;
    Ok(exe
        .parent()
        .context("executable has no directory")?
        .to_path_buf())
}

/// plot.py and the catalog are found at run time: `<bin>/evaluation/<name>`
/// (installed next to the binaries), else `evaluation/<name>` in the nearest ancestor of the bin
/// directory or of this executable (a source checkout: build/, evaluation/target/release/).
pub(crate) fn evaluation_file(ctx: &Ctx, name: &str) -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    ctx.bin
        .ancestors()
        .chain(exe.ancestors())
        .map(|dir| dir.join("evaluation").join(name))
        .find(|path| path.exists())
        .with_context(|| {
            format!(
                "evaluation/{name} not found next to {} or its parents",
                ctx.bin.display()
            )
        })
}

/// Relative --results and --lock resolve here, once, against the directory the runner started in:
/// workload children run in the repository root, where the same relative path names another place
/// (`cmt trace --out results/...` then fails with ENOENT).
fn context(global: Global, cancel: Cancellation) -> Result<Ctx> {
    let name = global.name.unwrap_or_else(host_name);
    ensure!(
        !name.is_empty() && !name.contains('/') && name != "." && name != "..",
        "--name must be one path component"
    );
    Ok(Ctx {
        root: std::path::absolute(global.results.join(name))?,
        bin: bin_dir(global.bin_dir)?,
        device: global.device,
        gpu: None,
        lock_path: std::path::absolute(global.lock)?,
        lock: None,
        progress: global.progress == Toggle::On,
        cancel,
        quiet: RefCell::new(Quiet::new(global.max_load)),
    })
}

/// plot.py's arguments: the given result directories, else this run's; plot.py checks them (and
/// requires --out for several), so the rules live in one place.
fn report_args(ctx: &Ctx, runs: Vec<PathBuf>, out: Option<PathBuf>) -> Vec<std::ffi::OsString> {
    let mut args: Vec<std::ffi::OsString> = if runs.is_empty() {
        vec![ctx.root.clone().into()]
    } else {
        runs.into_iter().map(Into::into).collect()
    };
    if let Some(out) = out {
        args.extend(["--out".into(), out.into()]);
    }
    args
}

fn dispatch(cli: Cli, cancel: Cancellation) -> Result<()> {
    let mut ctx = context(cli.global, cancel)?;
    let reads_only = matches!(
        cli.command,
        Command::Report { .. }
            | Command::Workload {
                action: workloads::Action::List(_)
            }
    );
    if !reads_only {
        ctx.gpu = process::gpu(ctx.device)?;
        // A full-with-kernel run that was killed may have left the machine patched.
        tune::recover(&ctx)?;
    }
    match cli.command {
        Command::Tune {
            tier,
            tuning_params,
            direction,
        } => tune::run(&mut ctx, tier, tuning_params, direction),
        Command::Bench {
            bench: bench::Bench::Kernel,
            sizes,
            config,
            tuning_params,
        } => {
            ensure!(
                sizes.is_none() && config.is_none() && tuning_params.is_none(),
                "bench kernel takes no --sizes, --config or --tuning_params: it measures the \
                 default progressive-pinning request at its sizes, 4 MiB to 1 GiB"
            );
            // Not under the lock: the arms take it, and `cmt patch`/`unpatch` take it between them.
            let mut host = tune::system_host(&ctx)?;
            bench::kernel(&mut ctx, &mut host)
        }
        Command::Bench {
            bench,
            sizes,
            config,
            tuning_params,
        } => {
            ctx.acquire_lock()?;
            bench::run(&ctx, bench, sizes, config, tuning_params)
        }
        Command::Workload { action } => workloads::run(&mut ctx, action),
        Command::Report { python, runs, out } => {
            let script = evaluation_file(&ctx, "plot.py")?;
            let mut command = std::process::Command::new(python);
            command.arg(script).args(report_args(&ctx, runs, out));
            let status = process::run_child(&mut command, None, &ctx.cancel)?;
            ensure!(status.success(), "plot.py failed: {status}");
            Ok(())
        }
    }
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let cancel = match Cancellation::install() {
        Ok(cancel) => cancel,
        Err(error) => {
            note!("cmt-evaluate: install signal handlers: {error:#}");
            return std::process::ExitCode::from(2);
        }
    };
    let signal = cancel.signal.clone();
    match dispatch(cli, cancel) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            note!("cmt-evaluate: {error:#}");
            let signal = signal.load(std::sync::atomic::Ordering::Relaxed);
            std::process::ExitCode::from(if signal != 0 { 128 + signal as u8 } else { 1 })
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn ctx(root: &Path) -> Ctx {
        Ctx {
            root: root.to_path_buf(),
            bin: root.to_path_buf(),
            device: 0,
            gpu: None,
            lock_path: root.join("lock"),
            lock: None,
            progress: false,
            cancel: Cancellation::never(),
            quiet: RefCell::new(Quiet::new(0.0)),
        }
    }

    #[test]
    fn resume_skips_saved_items_and_retries_failed_ones() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx(dir.path());
        let items = ["a", "b", "c"];
        let calls = RefCell::new(Vec::new());
        let execute = |item: &&str, _: &Path| {
            calls.borrow_mut().push(item.to_string());
            ensure!(
                *item != "b" || calls.borrow().len() > 3,
                "b fails the first time"
            );
            Ok(json!({"id": item}))
        };
        // First pass: a and c are saved, b fails, and the pass reports it.
        let inputs = |_: &&str| json!("v1");
        assert!(run_items(&ctx, dir.path(), &items, |i| i, inputs, execute).is_err());
        assert!(dir.path().join("a.json").is_file() && !dir.path().join("b.json").exists());
        // Second pass runs only b.
        run_items(&ctx, dir.path(), &items, |i| i, inputs, execute).unwrap();
        assert_eq!(*calls.borrow(), ["a", "b", "c", "b"]);
        // Changed inputs for c only: c reruns, a and b are reused.
        let changed = |i: &&str| json!(if *i == "c" { "v2" } else { "v1" });
        run_items(&ctx, dir.path(), &items, |i| i, changed, execute).unwrap();
        assert_eq!(*calls.borrow(), ["a", "b", "c", "b", "c"]);
        let saved: Value = process::read_json(&dir.path().join("c.json")).unwrap();
        assert_eq!(saved["inputs"], "v2");
        assert!(saved["load"]["start"]["load1"].is_number());
    }

    #[test]
    fn inputs_are_saved_as_they_were_when_the_item_started() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx(dir.path());
        let hook = std::cell::Cell::new(1);
        let execute = |_: &&str, _: &Path| {
            hook.set(2); // rebuilt while the item runs
            Ok(json!({}))
        };
        run_items(
            &ctx,
            dir.path(),
            &["a"],
            |i| i,
            |_| json!(hook.get()),
            execute,
        )
        .unwrap();
        let saved: Value = process::read_json(&dir.path().join("a.json")).unwrap();
        assert_eq!(saved["inputs"], 1);
    }

    #[test]
    fn item_inputs_cover_the_preloaded_hook_and_the_driver() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::write(root.join("cmt"), "cmt").unwrap();
        std::fs::write(root.join("libcmt_hook.so"), "hook").unwrap();
        let mut ctx = ctx(&root);
        let gpu = |driver: &str| process::Gpu {
            uuid: "GPU-a".into(),
            driver: driver.into(),
        };
        ctx.gpu = Some(gpu("580.1"));
        let item = Item {
            id: "a".into(),
            labels: Map::new(),
            argv: vec![root.join("cmt").to_string_lossy().into(), "run".into()],
            env: Vec::new(),
        };
        let old = item.inputs(&ctx);
        assert_eq!(old, item.inputs(&ctx), "unchanged inputs keep the identity");
        std::fs::write(root.join("libcmt_hook.so"), "hook2").unwrap();
        let rebuilt = item.inputs(&ctx);
        assert_ne!(old, rebuilt, "a rebuilt hook counts");
        ctx.gpu = Some(gpu("580.2"));
        assert_ne!(rebuilt, item.inputs(&ctx), "the driver counts");
    }

    #[test]
    fn relative_paths_resolve_where_the_runner_started() {
        let args = [
            "cmt-evaluate",
            "--results",
            "results",
            "--name",
            "n",
            "--lock",
            "lock",
        ];
        let cli =
            Cli::try_parse_from(args.into_iter().chain(["--bin-dir", ".", "report"])).unwrap();
        let ctx = context(cli.global, Cancellation::never()).unwrap();
        let cwd = std::env::current_dir().unwrap();
        // The trace path handed to a workload child, whatever directory that child runs in.
        assert_eq!(
            ctx.root.join("workload").join("traces"),
            cwd.join("results/n/workload/traces")
        );
        assert_eq!(ctx.lock_path, cwd.join("lock"));
    }

    #[test]
    fn report_draws_this_run_or_the_named_ones() {
        let args = |argv: &[&str]| {
            let cli = Cli::try_parse_from(argv).unwrap();
            let ctx = context(cli.global, Cancellation::never()).unwrap();
            let Command::Report { runs, out, .. } = cli.command else {
                panic!("not report");
            };
            let root = ctx.root.to_string_lossy().into_owned();
            let args = report_args(&ctx, runs, out);
            (
                root,
                args.into_iter()
                    .map(|a| a.into_string().unwrap())
                    .collect::<Vec<_>>(),
            )
        };
        let (root, alone) = args(&["cmt-evaluate", "--name", "n", "--bin-dir", ".", "report"]);
        assert_eq!(alone, [root]);
        let (_, several) = args(&[
            "cmt-evaluate",
            "--bin-dir",
            ".",
            "report",
            "results/a",
            "results/b",
            "--out",
            "o",
        ]);
        assert_eq!(several, ["results/a", "results/b", "--out", "o"]);
    }

    #[test]
    fn tune_without_a_tier_is_quick_as_in_cmt_tune() {
        let tier = |args: &[&str]| match Cli::try_parse_from(args).unwrap().command {
            Command::Tune { tier, .. } => tier,
            _ => panic!("not tune"),
        };
        assert_eq!(tier(&["cmt-evaluate", "tune"]), tune::Tier::Quick);
        assert_eq!(tier(&["cmt-evaluate", "tune", "full"]), tune::Tier::Full);
    }

    #[test]
    fn cancellation_stops_before_the_next_item() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx(dir.path());
        ctx.cancel
            .signal
            .store(15, std::sync::atomic::Ordering::Relaxed);
        let result = run_items(
            &ctx,
            dir.path(),
            &["a"],
            |i| i,
            |_| json!(1),
            |_, _| Ok(json!({})),
        );
        assert!(result.is_err() && !dir.path().join("a.json").exists());
    }

    #[test]
    fn cancelled_child_group_is_killed() {
        let cancel = Cancellation::never();
        let mut command = std::process::Command::new("sh");
        // The child ignores SIGTERM, so only the KILL after the grace period can stop it.
        command.args(["-c", "trap '' TERM; sleep 30 & wait"]);
        let flag = cancel.signal.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            flag.store(15, std::sync::atomic::Ordering::Relaxed);
        });
        let started = Instant::now();
        let status = process::run_child(&mut command, None, &cancel).unwrap();
        assert!(!status.success());
        assert!(started.elapsed() < std::time::Duration::from_secs(10));
    }
}
