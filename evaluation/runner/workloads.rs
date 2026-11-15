//! Application workloads: read the catalog, run setup, and run each workload in three arms.
//!
//! Catalog (contract with the workload authors; see evaluation/README.md): a JSON array of
//! entries, in one file or in every `*.json` of a directory (each file an array or one entry).
//! Commands run without a shell, with the repository root (the nearest ancestor of the catalog
//! that has an `evaluation/` directory) as working directory, because catalog paths such as
//! `evaluation/workloads/hf.py` are repository-relative. `${NAME}` in argv and env values expands
//! from the child environment, which adds CMT_ENV_ROOT (default `<repo>/tmp/envs`), CMT_WORKLOAD,
//! CMT_BIN_DIR and CUDA_VISIBLE_DEVICES (the --device GPU's UUID), and never inherits variables
//! that change copy behaviour.
//!
//! Arms: `stock` (no hook), `hook` (`cmt run --config tune.json`) and one `trace` pass
//! (`cmt trace`, which changes timing, so it never feeds the comparison). `cmt trace` records only
//! the process it launches, so an entry may give a separate `trace` command that is the copying
//! process itself. Before the timed runs, one `warmup` run of the stock command fills the page
//! cache and other first-run state; it is saved as arm `warmup` and never compared (on an RTX 3090,
//! vllm-startup's first stock run took 117 s, later runs 23 s). The arm that
//! goes first alternates per repeat (stock 1, hook 1, hook 2, stock 2, ...), so neither arm always
//! follows the other and slow drift hits both alike. Each finished run is saved at once into
//! `workload/<name>.json` with a hash of its inputs (catalog entry, tune.json contents, and
//! `Ctx::identity` of the files the entry's commands name, the Python modules a named driver
//! imports from its directory, cmt and its preload libraries); a resumed run skips a saved (arm,
//! repeat) only when that hash still matches, and moves stale runs to `stale_runs`. A resume that
//! still has a timed run to do warms up again first (its cache may be cold) and keeps the newest
//! warm-up only.
use crate::{process, say, Ctx};
use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Subcommand};
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Write as _,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Subcommand)]
pub(crate) enum Action {
    /// Print the catalog.
    List(Select),
    /// Run the idempotent setup commands of the selected workloads.
    Setup(Select),
    /// Run the selected workloads: stock, hooked and traced.
    Run {
        #[command(flatten)]
        select: Select,
        /// tune.json for the hook arm; default: the most thorough complete stock-module tier
        /// (full, quick, instant) under results/<name>/tune/.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Repeats per arm; default: the catalog's.
        #[arg(long)]
        repeats: Option<u32>,
        /// Skip the trace pass.
        #[arg(long)]
        no_trace: bool,
    },
}

#[derive(Args)]
pub(crate) struct Select {
    /// Workload names; default: all (or all in --group).
    names: Vec<String>,
    #[arg(long)]
    group: Option<String>,
    /// Catalog file or directory; default: evaluation/workloads/.
    #[arg(long)]
    catalog: Option<PathBuf>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub(crate) struct Workload {
    name: String,
    group: String,
    description: String,
    #[serde(default)]
    needs: Needs,
    #[serde(default)]
    setup: Setup,
    run: RunSpec,
    /// The command for the trace pass when `run` launches the copying process as a child.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    trace: Option<RunSpec>,
    #[serde(default = "three")]
    repeats: u32,
    metric: Metric,
}

fn three() -> u32 {
    3
}

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
struct Needs {
    #[serde(default)]
    gpu_gib: f64,
    #[serde(default)]
    host_gib: f64,
    #[serde(default)]
    disk_gib: f64,
    /// Newest GPU compute capability the entry's binaries support (e.g. 9.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_compute: Option<f64>,
}

/// One argv, or a list of argvs run in order.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(untagged)]
enum Setup {
    One(Vec<String>),
    Many(Vec<Vec<String>>),
}

impl Default for Setup {
    fn default() -> Self {
        Setup::Many(Vec::new())
    }
}

impl Setup {
    fn commands(&self) -> Vec<Vec<String>> {
        match self {
            Setup::One(argv) if argv.is_empty() => Vec::new(),
            Setup::One(argv) => vec![argv.clone()],
            Setup::Many(argvs) => argvs.clone(),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct RunSpec {
    argv: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct Metric {
    name: String,
    unit: String,
    better: Better,
}

#[derive(Deserialize, Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Better {
    Lower,
    Higher,
}

/// Saved per workload: what ran and every finished run.
#[derive(Deserialize, Serialize, Default)]
struct Record {
    workload: Option<Workload>,
    config: Option<String>,
    runs: Vec<Run>,
    /// Runs made from inputs that changed since; kept for reference, never compared.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    stale_runs: Vec<Run>,
}

#[derive(Deserialize, Serialize, Clone)]
struct Run {
    arm: String,
    repeat: u32,
    /// `inputs_hash` of what this run used.
    #[serde(default)]
    inputs: String,
    seconds: f64,
    /// The program's final JSON line: {"metric": .., "value": .., ...}; null for a trace command
    /// that prints none.
    result: Value,
    #[serde(default)]
    load: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace: Option<String>,
}

pub(crate) fn run(ctx: &mut Ctx, action: Action) -> Result<()> {
    match action {
        Action::List(select) => {
            let (_, workloads) = select.load(ctx)?;
            let mut out = std::io::stdout().lock();
            for w in workloads {
                let n = &w.needs;
                let better = if w.metric.better == Better::Lower {
                    "lower"
                } else {
                    "higher"
                };
                let _ =
                    writeln!(
                    out,
                    "{:<28} {:<10} x{} {} ({}, {better} is better) needs gpu {} GiB, host {} GiB, \
                     disk {} GiB\n    {}",
                    w.name, w.group, w.repeats, w.metric.name, w.metric.unit, n.gpu_gib,
                    n.host_gib, n.disk_gib, w.description
                );
            }
            Ok(())
        }
        Action::Setup(select) => {
            let (cwd, workloads) = select.load(ctx)?;
            let mut failed = Vec::new();
            for w in &workloads {
                // Never download for an entry this machine cannot run; `run` would skip it anyway.
                if let Some(reason) = missing_capacity(ctx, &cwd, &w.needs, false) {
                    note!("{}: setup skipped: {reason}", w.name);
                    continue;
                }
                if let Err(error) = setup(ctx, &cwd, w) {
                    failed.push(format!("{}: {error:#}", w.name));
                }
            }
            ensure!(failed.is_empty(), "setup failed: {}", failed.join(", "));
            Ok(())
        }
        Action::Run {
            select,
            config,
            repeats,
            no_trace,
        } => {
            let (cwd, workloads) = select.load(ctx)?;
            let config = match config {
                Some(path) => path,
                None => ctx.default_config()?,
            };
            let config = std::fs::canonicalize(&config)
                .with_context(|| format!("config {}", config.display()))?;
            let config_bytes = std::fs::read(&config)?;
            let config = crate::bench::path_text(config)?;
            let cmt = crate::bench::path_text(ctx.exe("cmt")?)?;
            ensure!(
                ctx.gpu.is_some(),
                "workload run needs nvidia-smi to pin the application to GPU --device {}",
                ctx.device
            );
            ctx.acquire_lock()?;
            let mut failed = Vec::new();
            for w in &workloads {
                ctx.cancel.check()?;
                if let Some(reason) = missing_capacity(ctx, &cwd, &w.needs, true) {
                    note!("{}: skipped: {reason}", w.name);
                    continue;
                }
                // Setup is idempotent (a no-op once done), so a run never needs a separate setup step. It
                // runs before any timed run of this entry and under the lock, so it never overlaps one.
                if let Err(error) = setup(ctx, &cwd, w) {
                    ctx.cancel.check()?;
                    note!("{}: setup failed: {error:#}", w.name);
                    failed.push(w.name.clone());
                    continue;
                }
                let plan = plan(repeats.unwrap_or(w.repeats), !no_trace);
                let identity = ctx.identity(&named_files(ctx, &cwd, w, &cmt));
                let inputs = inputs_hash(w, &config_bytes, &identity);
                if let Err(error) = run_workload(ctx, &cwd, w, &cmt, &config, &inputs, &plan) {
                    ctx.cancel.check()?;
                    note!("{}: failed: {error:#}", w.name);
                    failed.push(w.name.clone());
                }
            }
            ensure!(
                failed.is_empty(),
                "failed: {}; rerun to retry the missing runs",
                failed.join(", ")
            );
            Ok(())
        }
    }
}

/// Run `w`'s setup commands in order, stopping at the first that fails.
fn setup(ctx: &Ctx, cwd: &Path, w: &Workload) -> Result<()> {
    for argv in w.setup.commands() {
        ctx.cancel.check()?;
        note!("setup {}: {}", w.name, argv.join(" "));
        let env = child_env(ctx, cwd, &w.name, &BTreeMap::new());
        let argv = expand_all(&argv, &env);
        let mut command = std::process::Command::new(&argv[0]);
        command
            .args(&argv[1..])
            .current_dir(cwd)
            .env_clear()
            .envs(&env);
        let status = process::run_child(&mut command, None, &ctx.cancel)?;
        ensure!(status.success(), "{} ({status})", argv.join(" "));
    }
    Ok(())
}

impl Select {
    /// The working directory for the commands, and the selected entries.
    fn load(&self, ctx: &Ctx) -> Result<(PathBuf, Vec<Workload>)> {
        let path = match &self.catalog {
            Some(path) => path.clone(),
            None => crate::evaluation_file(ctx, "workloads")?,
        };
        let path =
            std::fs::canonicalize(&path).with_context(|| format!("catalog {}", path.display()))?;
        let all = load_catalog(&path)?;
        Ok((
            repository_root(&path),
            select(all, &self.names, self.group.as_deref())?,
        ))
    }
}

/// The nearest ancestor with an `evaluation/` directory; the catalog's directory otherwise.
fn repository_root(catalog: &Path) -> PathBuf {
    let dir = if catalog.is_dir() {
        catalog
    } else {
        catalog.parent().unwrap_or(catalog)
    };
    dir.ancestors()
        .find(|a| a.join("evaluation").is_dir())
        .unwrap_or(dir)
        .to_path_buf()
}

fn load_catalog(path: &Path) -> Result<Vec<Workload>> {
    let files = if path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<_>>()?;
        files.retain(|file| file.extension().is_some_and(|e| e == "json"));
        files.sort();
        files
    } else {
        vec![path.to_path_buf()]
    };
    let mut workloads = Vec::new();
    for file in files {
        let value: Value = process::read_json(&file)?;
        let entries = if value.is_array() {
            value
        } else {
            Value::Array(vec![value])
        };
        let parsed: Vec<Workload> = serde_json::from_value(entries).with_context(|| {
            format!(
                "{}: entries do not match the catalog contract",
                file.display()
            )
        })?;
        workloads.extend(parsed);
    }
    validate(&workloads)?;
    Ok(workloads)
}

fn validate(workloads: &[Workload]) -> Result<()> {
    let mut names = BTreeSet::new();
    for w in workloads {
        ensure!(
            !w.name.is_empty()
                && w.name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
                && !w.name.starts_with('.'),
            "workload name {:?} must be [A-Za-z0-9._-] (it names result files)",
            w.name
        );
        ensure!(names.insert(&w.name), "duplicate workload {}", w.name);
        ensure!(!w.run.argv.is_empty(), "{}: run.argv is empty", w.name);
        ensure!(
            w.trace.as_ref().is_none_or(|t| !t.argv.is_empty()),
            "{}: trace.argv is empty",
            w.name
        );
        ensure!(
            w.setup.commands().iter().all(|argv| !argv.is_empty()),
            "{}: empty setup command",
            w.name
        );
        ensure!(w.repeats >= 1, "{}: repeats must be at least 1", w.name);
        // `cmt run` removes these from the hooked run, so the two arms would differ (a CMT_SCRATCH
        // entry once wrote stock checkpoints to $CMT_ENV_ROOT and hooked ones to /tmp).
        for key in w.run.env.keys().chain(w.trace.iter().flat_map(|t| t.env.keys())) {
            ensure!(
                !process::is_hook_var(key),
                "{}: env {key} changes copy behaviour or is dropped from the hooked run; pass it as an argument",
                w.name
            );
        }
    }
    Ok(())
}

fn select(all: Vec<Workload>, names: &[String], group: Option<&str>) -> Result<Vec<Workload>> {
    for name in names {
        ensure!(
            all.iter().any(|w| &w.name == name),
            "unknown workload {name}"
        );
    }
    let chosen: Vec<Workload> = all
        .into_iter()
        .filter(|w| names.is_empty() || names.contains(&w.name))
        .filter(|w| group.is_none_or(|g| w.group == g))
        .collect();
    ensure!(!chosen.is_empty(), "no workload matches the selection");
    Ok(chosen)
}

const WARMUP: &str = "warmup";

/// Run order: the warm-up, then per repeat both arms, the first arm alternating (odd repeats stock
/// first); the trace pass comes last.
fn plan(repeats: u32, trace: bool) -> Vec<(&'static str, u32)> {
    let mut plan = vec![(WARMUP, 1)];
    plan.extend((1..=repeats).flat_map(|r| {
        if r % 2 == 1 {
            [("stock", r), ("hook", r)]
        } else {
            [("hook", r), ("stock", r)]
        }
    }));
    if trace {
        plan.push(("trace", 1));
    }
    plan
}

/// The entries of `plan` still to run: those not saved, and the warm-up only when a timed run is
/// among them (a resume with only the trace left needs no warm cache).
fn pending<'a>(plan: &[(&'a str, u32)], record: &Record) -> Vec<(&'a str, u32)> {
    let missing: Vec<_> = plan
        .iter()
        .copied()
        .filter(|&(arm, repeat)| {
            arm == WARMUP
                || !record
                    .runs
                    .iter()
                    .any(|r| r.arm == arm && r.repeat == repeat)
        })
        .collect();
    let timed = missing
        .iter()
        .any(|&(arm, _)| arm == "stock" || arm == "hook");
    missing
        .into_iter()
        .filter(|&(arm, _)| timed || arm != WARMUP)
        .collect()
}

fn inputs_hash(w: &Workload, config: &[u8], identity: &Value) -> String {
    let mut bytes = serde_json::to_vec(&(w, identity)).expect("a parsed entry serializes");
    bytes.push(0);
    bytes.extend_from_slice(config);
    process::hash_bytes(&bytes)
}

/// Absolute paths of the files the entry's commands name (catalog paths are relative to `cwd`),
/// the Python modules a named driver imports from its own directory (the first module of each
/// `import`/`from` line, one level: appenv.py), and `cmt run` and `cmt trace`, for
/// `files_identity`. One hash covers every arm, so a new hook also reruns the stock runs: stock and
/// hook repeats are compared in pairs, and a pair measured on different days would carry drift.
fn named_files(ctx: &Ctx, cwd: &Path, w: &Workload, cmt: &str) -> Vec<String> {
    let mut files = vec![cmt.into(), "run".into(), cmt.into(), "trace".into()];
    let setup = w
        .setup
        .commands()
        .into_iter()
        .map(|argv| (argv, BTreeMap::new()));
    let specs = std::iter::once(&w.run).chain(&w.trace);
    for (argv, extra) in setup.chain(specs.map(|spec| (spec.argv.clone(), spec.env.clone()))) {
        for arg in expand_all(&argv, &child_env(ctx, cwd, &w.name, &extra)) {
            let path = cwd.join(arg);
            if !path.is_file() {
                continue;
            }
            if path.extension().is_some_and(|e| e == "py") {
                let text = std::fs::read_to_string(&path).unwrap_or_default();
                let modules = text.lines().filter_map(|line| {
                    let line = line.trim_start();
                    let rest = line
                        .strip_prefix("import ")
                        .or(line.strip_prefix("from "))?;
                    rest.split(|c: char| !c.is_alphanumeric() && c != '_')
                        .next()
                });
                files.extend(
                    modules
                        .map(|module| path.with_file_name(format!("{module}.py")))
                        .filter(|module| module.is_file())
                        .map(|module| module.to_string_lossy().into_owned()),
                );
            }
            files.push(path.to_string_lossy().into_owned());
        }
    }
    files
}

fn child_env(
    ctx: &Ctx,
    cwd: &Path,
    name: &str,
    extra: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut env: BTreeMap<String, String> = std::env::vars()
        .filter(|(key, _)| !process::is_hook_var(key))
        .collect();
    let root = std::env::var("CMT_ENV_ROOT")
        .unwrap_or_else(|_| cwd.join("tmp").join("envs").to_string_lossy().into_owned());
    env.insert("CMT_ENV_ROOT".into(), root);
    env.insert("CMT_WORKLOAD".into(), name.into());
    env.insert("CMT_BIN_DIR".into(), ctx.bin.to_string_lossy().into_owned());
    if let Some(gpu) = &ctx.gpu {
        // The application sees exactly the GPU --device selects, as its device 0.
        env.insert("CUDA_VISIBLE_DEVICES".into(), gpu.uuid.clone());
    }
    let base = env.clone();
    for (key, value) in extra {
        env.insert(key.clone(), expand(value, &base));
    }
    env
}

/// Replace `${NAME}` with its value in `env`; an unknown name is left as written.
fn expand(text: &str, env: &BTreeMap<String, String>) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        match rest[start + 2..].find('}') {
            Some(end) => {
                let name = &rest[start + 2..start + 2 + end];
                match env.get(name) {
                    Some(value) => out.push_str(value),
                    None => out.push_str(&rest[start..start + 3 + end]),
                }
                rest = &rest[start + 3 + end..];
            }
            None => {
                out.push_str(&rest[start..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

fn expand_all(argv: &[String], env: &BTreeMap<String, String>) -> Vec<String> {
    argv.iter().map(|arg| expand(arg, env)).collect()
}

/// Keep runs made from `inputs`; move the others to `stale_runs` so they are measured again.
fn retire_stale(record: &mut Record, inputs: &str) -> usize {
    let (current, stale): (Vec<Run>, Vec<Run>) = std::mem::take(&mut record.runs)
        .into_iter()
        .partition(|run| run.inputs == inputs);
    record.runs = current;
    let moved = stale.len();
    record.stale_runs.extend(stale);
    moved
}

fn run_workload(
    ctx: &Ctx,
    cwd: &Path,
    w: &Workload,
    cmt: &str,
    config: &str,
    inputs: &str,
    plan: &[(&str, u32)],
) -> Result<()> {
    let out = ctx.root.join("workload");
    let path = out.join(format!("{}.json", w.name));
    let mut record: Record = if path.is_file() {
        process::read_json(&path)?
    } else {
        Record::default()
    };
    let stale = retire_stale(&mut record, inputs);
    if stale > 0 {
        note!(
            "{}: {stale} saved runs used different inputs (catalog entry, tune.json, scripts, \
             cmt, hook, driver or GPU); rerunning",
            w.name
        );
    }
    record.workload = Some(w.clone());
    record.config = Some(config.into());
    let todo = pending(plan, &record);
    let bar = if ctx.progress {
        ProgressBar::new(todo.len() as u64)
    } else {
        ProgressBar::hidden()
    };
    for (arm, repeat) in todo {
        ctx.cancel.check()?;
        let spec = if arm == "trace" {
            w.trace.as_ref().unwrap_or(&w.run)
        } else {
            &w.run
        };
        let env = child_env(ctx, cwd, &w.name, &spec.env);
        let stem = out.join("logs").join(format!("{}-{arm}-{repeat}", w.name));
        std::fs::create_dir_all(out.join("logs"))?;
        let trace = out.join("traces").join(format!("{}.jsonl", w.name));
        let mut full = match arm {
            "stock" | WARMUP => Vec::new(),
            "hook" => vec![
                cmt.into(),
                "run".into(),
                "--config".into(),
                config.into(),
                "--".into(),
            ],
            _ => {
                std::fs::create_dir_all(out.join("traces"))?;
                // cmt trace refuses an existing file; one left here is from an interrupted pass.
                let _ = std::fs::remove_file(&trace);
                let trace = trace.to_string_lossy().into_owned();
                vec![
                    cmt.into(),
                    "trace".into(),
                    "--out".into(),
                    trace,
                    "--".into(),
                ]
            }
        };
        full.extend(expand_all(&spec.argv, &env));
        bar.set_message(format!("{} {arm} {repeat}", w.name));
        let stdout = stem.with_extension("stdout");
        let stderr = stem.with_extension("stderr");
        let mut command = std::process::Command::new(&full[0]);
        command
            .args(&full[1..])
            .current_dir(cwd)
            .env_clear()
            .envs(&env)
            .stdin(std::process::Stdio::null())
            .stdout(File::create(&stdout)?)
            .stderr(File::create(&stderr)?);
        ctx.quiet.borrow().wait(&ctx.cancel)?;
        let before = ctx.quiet.borrow().snapshot();
        let started = Instant::now();
        let status = process::run_child(&mut command, None, &ctx.cancel);
        let seconds = started.elapsed().as_secs_f64();
        ctx.quiet.borrow_mut().child_finished(started);
        let status = status?;
        ctx.cancel.check()?;
        if !status.success() {
            bail!(
                "{arm} run {repeat}: {status}: {}",
                process::tail(&stderr, 600)
            );
        }
        let result = final_json_line(&std::fs::read_to_string(&stdout)?);
        let result = match result {
            Some(result) if result["metric"] == w.metric.name.as_str() && result["value"].is_number() => result,
            // A separate trace command execs the application itself and prints no metric.
            _ if arm == "trace" => Value::Null,
            other => bail!(
                "{arm} run {repeat}: last stdout line {} is not {{\"metric\": {:?}, \"value\": <number>}} ({})",
                other.map_or("missing".into(), |v| v.to_string()),
                w.metric.name,
                stdout.display()
            ),
        };
        say(
            &bar,
            format!(
                "{} {arm} {repeat}: {} {}",
                w.name, result["value"], w.metric.unit
            ),
        );
        record.runs.retain(|r| r.arm != arm || r.repeat != repeat); // an older warm-up
        record.runs.push(Run {
            arm: arm.into(),
            repeat,
            inputs: inputs.into(),
            seconds,
            result,
            load: json!({"start": before, "end": ctx.quiet.borrow().snapshot()}),
            trace: (arm == "trace").then(|| trace.to_string_lossy().into_owned()),
        });
        process::write_json(&path, &record)?;
        bar.inc(1);
    }
    bar.finish_and_clear();
    Ok(())
}

fn final_json_line(stdout: &str) -> Option<Value> {
    stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .and_then(|line| serde_json::from_str::<Value>(line).ok())
        .filter(Value::is_object)
}

/// One value per GPU property from nvidia-smi for the selected device.
fn gpu_query(ctx: &Ctx, field: &str) -> Option<f64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            format!("--query-gpu={field}"),
            "--format=csv,noheader,nounits".into(),
            "-i".into(),
            ctx.gpu.as_ref()?.uuid.clone(),
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

/// Why this machine cannot run a workload: host memory, GPU memory or compute capability, or
/// (with `disk`) free disk under CMT_ENV_ROOT. Setup leaves disk out: environments are shared,
/// so `disk_gib` over-counts an entry whose environment is already installed.
fn missing_capacity(ctx: &Ctx, cwd: &Path, needs: &Needs, disk: bool) -> Option<String> {
    const GIB: f64 = (1u64 << 30) as f64;
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let host = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|rest| {
            rest.trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<f64>()
                .ok()
        })
        .map(|kib| kib * 1024.0 / GIB);
    if let Some(host) = host.filter(|host| *host < needs.host_gib) {
        return Some(format!(
            "needs {} GiB host memory, machine has {host:.1}",
            needs.host_gib
        ));
    }
    if let Some(gpu) = gpu_query(ctx, "memory.total")
        .map(|mib| mib / 1024.0)
        .filter(|g| *g < needs.gpu_gib)
    {
        return Some(format!(
            "needs {} GiB GPU memory, GPU has {gpu:.1}",
            needs.gpu_gib
        ));
    }
    if let (Some(max), Some(compute)) = (needs.max_compute, gpu_query(ctx, "compute_cap")) {
        if compute > max {
            return Some(format!("needs compute <= {max}, GPU has {compute}"));
        }
    }
    if !disk {
        return None;
    }
    // Free space where setup installs: the nearest existing ancestor of $CMT_ENV_ROOT.
    let root = PathBuf::from(&child_env(ctx, cwd, "", &BTreeMap::new())["CMT_ENV_ROOT"]);
    let disk = root
        .ancestors()
        .find_map(|dir| rustix::fs::statvfs(dir).ok())
        .map(|fs| (fs.f_bavail * fs.f_frsize) as f64 / GIB);
    if let Some(disk) = disk.filter(|disk| *disk < needs.disk_gib) {
        return Some(format!(
            "needs {} GiB free disk under {}, has {disk:.1}",
            needs.disk_gib,
            root.display()
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENTRY: &str = r#"{"name": "llama-8b", "group": "llm", "description": "decode",
        "needs": {"gpu_gib": 20, "host_gib": 32, "disk_gib": 20, "max_compute": 9.0},
        "setup": ["python3", "llama.py", "--setup"],
        "run": {"argv": ["python3", "llama.py", "--root", "${CMT_ENV_ROOT}/llama-8b"], "env": {"A": "1"}},
        "trace": {"argv": ["llama-bench"]},
        "repeats": 3, "metric": {"name": "tokens_per_s", "unit": "tok/s", "better": "higher"}}"#;

    #[test]
    fn catalog_file_or_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.json"), format!("[{ENTRY}]")).unwrap();
        std::fs::write(
            dir.path().join("b.json"),
            ENTRY.replace("llama-8b", "other"),
        )
        .unwrap();
        std::fs::write(dir.path().join("notes.txt"), "ignored").unwrap();
        let all = load_catalog(dir.path()).unwrap();
        assert_eq!(
            all.iter().map(|w| w.name.as_str()).collect::<Vec<_>>(),
            ["llama-8b", "other"]
        );
        assert_eq!(
            all[0].setup.commands(),
            [["python3", "llama.py", "--setup"]]
        );
        assert_eq!(all[0].trace.as_ref().unwrap().argv, ["llama-bench"]);
        // A duplicate name across files is refused, not silently merged.
        std::fs::write(dir.path().join("c.json"), ENTRY).unwrap();
        assert!(load_catalog(dir.path()).is_err());
    }

    #[test]
    fn catalog_rejects_unknown_fields_and_bad_names() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("w.json");
        std::fs::write(&file, ENTRY.replace("\"repeats\"", "\"repeat\"")).unwrap();
        assert!(load_catalog(&file).is_err());
        std::fs::write(&file, ENTRY.replace("llama-8b\"", "../x\"")).unwrap();
        assert!(load_catalog(&file).is_err());
        std::fs::write(&file, ENTRY.replace("{\"A\": \"1\"}", "{\"CMT_SCRATCH\": \"/x\"}")).unwrap();
        assert!(load_catalog(&file).is_err(), "an env key the hooked run loses is refused");
    }

    #[test]
    fn shipped_catalog_loads() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("workloads");
        assert!(load_catalog(&dir).unwrap().len() >= 20);
    }

    #[test]
    fn commands_run_from_the_repository_root() {
        let dir = tempfile::tempdir().unwrap();
        let catalog = dir.path().join("evaluation").join("workloads");
        std::fs::create_dir_all(&catalog).unwrap();
        assert_eq!(repository_root(&catalog), dir.path());
        assert_eq!(repository_root(&catalog.join("a.json")), dir.path());
    }

    #[test]
    fn selection_by_name_and_group() {
        let a: Workload = serde_json::from_str(ENTRY).unwrap();
        let mut b = a.clone();
        b.name = "xgb".into();
        b.group = "boosting".into();
        let all = vec![a, b];
        assert_eq!(
            select(all.clone(), &[], Some("boosting")).unwrap()[0].name,
            "xgb"
        );
        assert_eq!(select(all.clone(), &[], None).unwrap().len(), 2);
        assert!(select(all.clone(), &["nope".into()], None).is_err());
        assert!(select(all, &[], Some("nope")).is_err());
    }

    #[test]
    fn plan_warms_up_alternates_the_first_arm_and_traces_once() {
        assert_eq!(
            plan(3, true),
            [
                (WARMUP, 1),
                ("stock", 1),
                ("hook", 1),
                ("hook", 2),
                ("stock", 2),
                ("stock", 3),
                ("hook", 3),
                ("trace", 1)
            ]
        );
        assert_eq!(plan(1, false), [(WARMUP, 1), ("stock", 1), ("hook", 1)]);
    }

    #[test]
    fn warm_up_precedes_missing_timed_runs_only() {
        let run = |arm: &str, repeat| Run {
            arm: arm.into(),
            repeat,
            inputs: String::new(),
            seconds: 1.0,
            result: Value::Null,
            load: Value::Null,
            trace: None,
        };
        let plan = plan(2, true);
        let mut record = Record::default();
        assert_eq!(pending(&plan, &record), plan);
        // A resume warms up again before the timed runs left, though a warm-up is saved.
        record.runs = vec![
            run(WARMUP, 1),
            run("stock", 1),
            run("hook", 1),
            run("hook", 2),
        ];
        assert_eq!(
            pending(&plan, &record),
            [(WARMUP, 1), ("stock", 2), ("trace", 1)]
        );
        // With only the trace left there is nothing to warm up for.
        record.runs.push(run("stock", 2));
        assert_eq!(pending(&plan, &record), [("trace", 1)]);
    }

    #[test]
    fn changed_entry_or_config_retires_saved_runs() {
        let w: Workload = serde_json::from_str(ENTRY).unwrap();
        let old = inputs_hash(&w, b"{\"a\":1}", &Value::Null);
        assert_ne!(
            old,
            inputs_hash(&w, b"{\"a\":2}", &Value::Null),
            "config contents count"
        );
        let mut changed = w.clone();
        changed.run.argv.push("--more".into());
        assert_ne!(
            old,
            inputs_hash(&changed, b"{\"a\":1}", &Value::Null),
            "catalog entry counts"
        );
        let run = |inputs: &str| Run {
            arm: "stock".into(),
            repeat: 1,
            inputs: inputs.into(),
            seconds: 1.0,
            result: Value::Null,
            load: Value::Null,
            trace: None,
        };
        let mut record = Record {
            runs: vec![run(&old), run("other")],
            ..Record::default()
        };
        assert_eq!(retire_stale(&mut record, &old), 1);
        assert_eq!(record.runs.len(), 1);
        assert_eq!(record.stale_runs[0].inputs, "other");
    }

    #[test]
    fn changed_script_module_or_preload_changes_the_hash() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("evaluation/workloads")).unwrap();
        let write = |file: &str, text: &str| std::fs::write(root.join(file), text).unwrap();
        write("evaluation/workloads/drv.py", "import appenv as c\n");
        write("evaluation/workloads/appenv.py", "x = 1\n");
        write("evaluation/workloads/setup.sh", "true\n");
        write("cmt", "cmt");
        write("libcmt_hook.so", "hook");
        write("libcmt_trace.so", "trace");
        let mut w: Workload = serde_json::from_str(ENTRY).unwrap();
        w.setup = Setup::One(vec!["bash".into(), "evaluation/workloads/setup.sh".into()]);
        w.run.argv = vec!["python3".into(), "evaluation/workloads/drv.py".into()];
        let ctx = crate::tests::ctx(&root);
        let cmt = root.join("cmt").to_string_lossy().into_owned();
        let hash = || {
            inputs_hash(
                &w,
                b"{}",
                &ctx.identity(&named_files(&ctx, &root, &w, &cmt)),
            )
        };
        assert_eq!(hash(), hash(), "unchanged inputs keep the hash");
        for (file, text) in [
            ("evaluation/workloads/drv.py", "import appenv as c\n#\n"),
            ("evaluation/workloads/appenv.py", "x = 22\n"),
            ("evaluation/workloads/setup.sh", "true #\n"),
            ("libcmt_hook.so", "hook2"),
            ("libcmt_trace.so", "trace2"),
        ] {
            let before = hash();
            write(file, text);
            assert_ne!(before, hash(), "{file} counts");
        }
    }

    #[test]
    fn expansion_and_final_line() {
        let env = BTreeMap::from([("ROOT".to_string(), "/e".to_string())]);
        assert_eq!(expand("${ROOT}/m-${ROOT}", &env), "/e/m-/e");
        assert_eq!(expand("${MISSING}/x ${", &env), "${MISSING}/x ${");
        let out = "loading\n{\"metric\": \"t\", \"value\": 2.5}\n\n";
        assert_eq!(final_json_line(out).unwrap()["value"], 2.5);
        assert!(final_json_line("{\"value\": 1}\ndone\n").is_none());
    }

    #[test]
    fn child_runs_on_the_selected_gpu_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = crate::tests::ctx(dir.path());
        ctx.gpu = Some(process::Gpu {
            uuid: "GPU-b".into(),
            driver: "580.1".into(),
        });
        let env = child_env(&ctx, dir.path(), "w", &BTreeMap::new());
        assert_eq!(env["CUDA_VISIBLE_DEVICES"], "GPU-b");
    }

    #[test]
    fn hook_variables_are_not_inherited() {
        assert!(process::is_hook_var("STAGING_PIPELINE_WORKERS"));
        assert!(process::is_hook_var("MEMCPY_HOOK_DEBUG"));
        assert!(process::is_hook_var("CMT_TUNE_TABLE"));
        assert!(process::is_hook_var("LD_PRELOAD"));
        assert!(!process::is_hook_var("CUDA_VISIBLE_DEVICES"));
    }
}
