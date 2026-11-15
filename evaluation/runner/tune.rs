//! Tuning tiers: run `cmt tune`, and for `full-with-kernel` swap kernel modules around it.
//!
//! `cmt tune` owns its search, its resume and its tune.json; this file only chooses the flags and
//! the output directory (`results/<name>/tune/<tier>/`). Every run is handed to `cmt tune`, with
//! `--resume` exactly when tune.json exists: cmt then checks that the inputs (including new
//! `--tuning_params`) match and skips finished processes, so this file never decides "complete".
//!
//! full-with-kernel changes the machine. Order: wait for the measurement lock to be free and check
//! that no GPU has compute processes, save `{old_thp, patched}` to `kernel-state.json` in the
//! tier's directory, set THP to `always`, mark `patched` and run `cmt patch`, tune, then restore:
//! `cmt unpatch` (retried with backoff until it succeeds) and the old THP value, then delete the
//! state file. The restore runs from a guard's Drop too (panic), ignores cancellation (Ctrl-C,
//! SIGTERM, SIGHUP), and a runner killed outright leaves the state file, which every later
//! measuring command restores from before it starts. The lock is not held around patch/unpatch:
//! both take the same lock themselves.
//!
//! `bench kernel` uses the same guard with two arms ([`with_kernel_arms`]): a stock arm after THP
//! is set and before the patch, then the patched arm. Its state file is
//! `bench/kernel/kernel-state.json`; [`recover`] checks both.
use crate::{bench, process, Ctx};
use anyhow::{bail, ensure, Context, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
pub(crate) enum Tier {
    Instant,
    Quick,
    Full,
    FullWithKernel,
}

const THP: &str = "/sys/kernel/mm/transparent_hugepage/enabled";
/// Where `cmt patch` keeps its recovery journal (kernel_patches/nvidia-open/README.md).
const JOURNAL: &str = "/var/lib/cmt/kernel/state.json";
pub(crate) const STATE_FILE: &str = "kernel-state.json";

pub(crate) fn kernel_dir(ctx: &Ctx) -> PathBuf {
    ctx.root.join("tune").join("full-with-kernel")
}

/// Where `bench kernel` writes, and its guard's state file.
pub(crate) fn bench_kernel_dir(ctx: &Ctx) -> PathBuf {
    ctx.root.join("bench").join("kernel")
}

/// The state file of every command that changes modules or THP. [`recover`] checks all of them, so
/// a killed `bench kernel` is restored by the next measuring command as a killed tune is.
fn state_files(ctx: &Ctx) -> [PathBuf; 2] {
    [
        kernel_dir(ctx).join(STATE_FILE),
        bench_kernel_dir(ctx).join(STATE_FILE),
    ]
}

/// The two halves of a stock vs patched comparison, in the order they run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Arm {
    /// THP set to `always`, the modules as found.
    Stock,
    /// After a successful `cmt patch`.
    Patched,
}

impl Arm {
    /// Directory and label name; part of the result layout, so not derived from the variant.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Arm::Stock => "stock",
            Arm::Patched => "patched",
        }
    }
}

pub(crate) fn run(
    ctx: &mut Ctx,
    tier: Tier,
    params: Option<String>,
    direction: Option<String>,
) -> Result<()> {
    let params = params.as_deref();
    let direction = direction.as_deref();
    let tune = ctx.root.join("tune");
    match tier {
        Tier::Instant | Tier::Quick => {
            ctx.acquire_lock()?;
            let search = if tier == Tier::Instant {
                "instant"
            } else {
                "quick"
            };
            run_search(ctx, search, &tune.join(search), params, direction, None)
        }
        Tier::Full => {
            ctx.acquire_lock()?;
            run_search(ctx, "full", &tune.join("full"), params, direction, None)?;
            // The FULL evaluation also includes the driver references and inline vs DMA.
            bench::run(ctx, bench::Bench::References, None, None, None)?;
            bench::run(ctx, bench::Bench::Inline, None, None, None)
        }
        Tier::FullWithKernel => {
            let out = kernel_dir(ctx);
            let mut host = system_host(ctx)?;
            with_kernel(&mut host, &out.join(STATE_FILE), || {
                ctx.lock = Some(process::MeasurementLock::wait(&ctx.lock_path)?);
                let result = run_search(
                    ctx,
                    "full_with_kernel",
                    &out,
                    params,
                    direction,
                    Some("thp"),
                );
                ctx.lock = None;
                result
            })
        }
    }
}

/// Restore a machine a killed full-with-kernel or `bench kernel` run left patched. Called before
/// measuring commands.
pub(crate) fn recover(ctx: &Ctx) -> Result<()> {
    for state in state_files(ctx) {
        if !state.is_file() {
            continue;
        }
        note!(
            "{} exists: a module swap did not finish its restore; restoring now",
            state.display()
        );
        restore_from(&mut system_host(ctx)?, &state)?;
    }
    Ok(())
}

/// The real machine: root steps through `sudo -n`, `cmt patch` with the run's measurement lock.
pub(crate) fn system_host(ctx: &Ctx) -> Result<SystemHost> {
    Ok(SystemHost {
        cmt: ctx.exe("cmt")?,
        lock: ctx.lock_path.clone(),
    })
}

pub(crate) fn run_search(
    ctx: &Ctx,
    search: &str,
    out: &Path,
    params: Option<&str>,
    direction: Option<&str>,
    pages: Option<&str>,
) -> Result<()> {
    // cmt tune creates artifacts/ before its first save and refuses --resume without tune.json.
    let resume = out.join("tune.json").is_file();
    let mut command = process::clean_command(&ctx.exe("cmt")?);
    command.args(tune_args(
        search,
        out,
        ctx.device,
        params,
        direction,
        pages,
        resume,
        ctx.progress,
    ));
    // cmt tune passes CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER and LD_LIBRARY_PATH from its own environment
    // (this one) to its children, and nothing else (src/tuner.h, kInheritedVariables).
    ctx.quiet.borrow().wait(&ctx.cancel)?;
    note!(
        "{}: cmt tune --search {search}{}",
        out.display(),
        if resume { " --resume" } else { "" }
    );
    let started = Instant::now();
    let status = process::run_child(&mut command, ctx.lock.as_ref(), &ctx.cancel);
    ctx.quiet.borrow_mut().child_finished(started);
    let status = status?;
    ctx.cancel.check()?;
    ensure!(
        status.success(),
        "cmt tune --search {search} failed: {status}; rerun to resume"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tune_args(
    search: &str,
    out: &Path,
    device: u32,
    params: Option<&str>,
    direction: Option<&str>,
    pages: Option<&str>,
    resume: bool,
    progress: bool,
) -> Vec<String> {
    let mut args = vec![
        "tune".into(),
        "--search".into(),
        search.into(),
        "--out".into(),
    ];
    args.push(out.to_string_lossy().into_owned());
    args.extend(["--device".into(), device.to_string()]);
    args.extend([
        "--progress_bar".into(),
        if progress { "on" } else { "off" }.into(),
    ]);
    for (flag, value) in [
        ("--tuning_params", params),
        ("--direction", direction),
        ("--pages", pages),
    ] {
        if let Some(value) = value {
            args.extend([flag.into(), value.into()]);
        }
    }
    if resume {
        args.push("--resume".into());
    }
    args
}

/// What full-with-kernel does to the machine; a trait so the restore logic can be tested.
pub(crate) trait Host {
    /// Compute processes on any GPU (a module swap affects all of them).
    fn gpu_processes(&mut self) -> Result<Vec<String>>;
    fn read_thp(&mut self) -> Result<String>;
    fn write_thp(&mut self, value: &str) -> Result<()>;
    fn patch(&mut self) -> Result<()>;
    fn unpatch(&mut self) -> Result<()>;
    /// Whether `cmt patch` wrote a recovery journal (without one, nothing was changed).
    fn journal_exists(&mut self) -> Result<bool>;
    /// Block until no other process holds the measurement lock.
    fn wait_lock_free(&mut self) -> Result<()>;
    /// Who holds the lock, for messages.
    fn lock_holders(&mut self) -> String;
    fn sleep(&mut self, duration: Duration);
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct KernelState {
    old_thp: String,
    /// Set before `cmt patch` starts, cleared only after a successful unpatch (or a failed patch,
    /// which restores the original modules itself).
    patched: bool,
}

/// Owns the restore. `restore` is called on every normal path; Drop covers panics.
struct Guard<'a, H: Host> {
    host: &'a mut H,
    path: PathBuf,
    state: KernelState,
    armed: bool,
}

impl<H: Host> Guard<'_, H> {
    fn save(&self) -> Result<()> {
        process::write_json(&self.path, &self.state)
    }

    fn restore(&mut self) -> Result<()> {
        if self.state.patched {
            unpatch_until_done(self.host);
            self.state.patched = false;
            self.save()?;
        }
        if self.host.read_thp().ok().as_deref() != Some(self.state.old_thp.as_str()) {
            self.host.write_thp(&self.state.old_thp).with_context(|| {
                format!(
                    "RESTORE FAILED: write {} to {THP} by hand, then delete {}",
                    self.state.old_thp,
                    self.path.display()
                )
            })?;
        }
        std::fs::remove_file(&self.path)
            .with_context(|| format!("remove {}", self.path.display()))?;
        self.armed = false;
        Ok(())
    }
}

impl<H: Host> Drop for Guard<'_, H> {
    fn drop(&mut self) {
        if self.armed {
            if let Err(error) = self.restore() {
                note!("cmt-evaluate: {error:#}");
            }
        }
    }
}

/// Unpatching is the one step that must not be given up: retry with backoff, saying what blocks
/// it, until `cmt unpatch` succeeds. If the runner is killed meanwhile, the state file remains.
fn unpatch_until_done(host: &mut impl Host) {
    let mut delay = Duration::from_secs(5);
    loop {
        match host.journal_exists() {
            Ok(false) => {
                note!("no cmt patch journal at {JOURNAL}: the modules were never switched");
                return;
            }
            Ok(true) => {
                let result = host.wait_lock_free().and_then(|()| host.unpatch());
                match result {
                    Ok(()) => return,
                    Err(error) => {
                        let gpus = host
                            .gpu_processes()
                            .map(|p| p.join("; "))
                            .unwrap_or_default();
                        note!(
                            "cmt unpatch failed: {error:#}. Lock holders: {}. GPU processes: {}. \
                             Retrying in {} s",
                            host.lock_holders(),
                            if gpus.is_empty() { "none".into() } else { gpus },
                            delay.as_secs()
                        );
                    }
                }
            }
            Err(error) => note!(
                "cannot check {JOURNAL}: {error:#}; retrying in {} s",
                delay.as_secs()
            ),
        }
        host.sleep(delay);
        delay = (delay * 2).min(Duration::from_secs(60));
    }
}

fn restore_from(host: &mut impl Host, path: &Path) -> Result<()> {
    let state: KernelState = process::read_json(path)?;
    Guard {
        host,
        path: path.to_path_buf(),
        state,
        armed: true,
    }
    .restore()
}

/// Setup, `measure` on the patched modules, then the restore, on every path. Returns the first
/// error with later ones attached.
pub(crate) fn with_kernel(
    host: &mut impl Host,
    path: &Path,
    measure: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let mut measure = Some(measure);
    with_kernel_arms(host, path, true, |arm| match arm {
        Arm::Stock => Ok(()),
        Arm::Patched => (measure
            .take()
            .expect("with_kernel_arms runs the patched arm at most once"))(
        ),
    })
}

/// A module swap must find every GPU idle: it affects all of them, not only the selected one.
fn require_idle_gpus(host: &mut impl Host) -> Result<()> {
    let busy = host.gpu_processes()?;
    ensure!(
        busy.is_empty(),
        "GPUs are in use ({}); the module swap needs idle GPUs",
        busy.join("; ")
    );
    Ok(())
}

/// [`with_kernel`] with a stock arm first: THP is set to `always` (the patched arm needs it, and
/// both arms must use the same pages), `measure(Arm::Stock)` runs on the modules as found, then,
/// when `patch` is true and the stock arm succeeded, `cmt patch` and `measure(Arm::Patched)`; the
/// restore follows on every path.
///
/// Trade-offs, on purpose:
/// - A failed or cancelled stock arm ends the run before the patch: the swap is the step that
///   disturbs every GPU on the machine, and a comparison without its baseline is not worth it. A
///   rerun reuses the finished stock items and retries the failed ones.
/// - The lock-free and idle-GPU checks run again before the patch, because the stock arm takes
///   minutes; the caller must have released the measurement lock by then (`cmt patch` takes it).
/// - `patch: false` (nothing left to measure on patched modules) still sets THP for the stock arm.
pub(crate) fn with_kernel_arms(
    host: &mut impl Host,
    path: &Path,
    patch: bool,
    mut measure: impl FnMut(Arm) -> Result<()>,
) -> Result<()> {
    if path.is_file() {
        restore_from(host, path)?;
    }
    host.wait_lock_free()?;
    require_idle_gpus(host)?;
    let old_thp = host.read_thp()?;
    let mut guard = Guard {
        host,
        path: path.to_path_buf(),
        state: KernelState {
            old_thp,
            patched: false,
        },
        armed: true,
    };
    // Saved before anything changes, so a killed runner can always be undone.
    guard.save()?;
    let measured = (|| -> Result<()> {
        if guard.state.old_thp != "always" {
            guard.host.write_thp("always")?;
            let now = guard.host.read_thp()?;
            ensure!(now == "always", "THP is {now} after setting always");
        }
        measure(Arm::Stock)?;
        if !patch {
            return Ok(());
        }
        guard.host.wait_lock_free()?;
        require_idle_gpus(guard.host)?;
        guard.state.patched = true;
        guard.save()?;
        if let Err(error) = guard.host.patch() {
            // A failed `cmt patch` restores the original modules itself.
            guard.state.patched = false;
            guard.save()?;
            return Err(error);
        }
        // Only here, after `cmt patch` succeeded: patched items never run on other modules.
        measure(Arm::Patched)
    })();
    let restored = guard.restore();
    match (measured, restored) {
        (Ok(()), restored) => restored,
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(restore)) => Err(error.context(format!("also: {restore:#}"))),
    }
}

fn parse_thp(text: &str) -> Result<String> {
    let start = text.find('[').context("no [selected] THP value")?;
    let end = text[start..].find(']').context("no [selected] THP value")?;
    Ok(text[start + 1..start + end].to_owned())
}

pub(crate) struct SystemHost {
    cmt: PathBuf,
    lock: PathBuf,
}

/// Root steps are never cancelled: a restore must finish after Ctrl-C.
fn privileged(args: &[&std::ffi::OsStr]) -> Result<()> {
    let mut command = process::privileged(args);
    let status = process::run_child(&mut command, None, &process::Cancellation::never())?;
    ensure!(status.success(), "{:?} failed: {status}", args);
    Ok(())
}

impl Host for SystemHost {
    fn gpu_processes(&mut self) -> Result<Vec<String>> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader",
            ])
            .output()
            .context("run nvidia-smi")?;
        ensure!(
            output.status.success(),
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.contains("No running"))
            .map(str::to_owned)
            .collect())
    }

    fn read_thp(&mut self) -> Result<String> {
        parse_thp(&std::fs::read_to_string(THP).with_context(|| format!("read {THP}"))?)
    }

    fn write_thp(&mut self, value: &str) -> Result<()> {
        if !matches!(value, "always" | "madvise" | "never") {
            bail!("refusing to write THP value {value:?}");
        }
        // The value and the path are arguments of the script, never part of its text.
        privileged(&[
            "sh".as_ref(),
            "-c".as_ref(),
            "printf %s \"$1\" > \"$2\"".as_ref(),
            "sh".as_ref(),
            value.as_ref(),
            THP.as_ref(),
        ])
    }

    fn patch(&mut self) -> Result<()> {
        privileged(&[
            self.cmt.as_os_str(),
            "patch".as_ref(),
            "--measurement-lock".as_ref(),
            self.lock.as_os_str(),
        ])
    }

    fn unpatch(&mut self) -> Result<()> {
        privileged(&[self.cmt.as_os_str(), "unpatch".as_ref()])
    }

    fn journal_exists(&mut self) -> Result<bool> {
        // The journal directory is root-only; ask through the same privilege path.
        let output = process::privileged(&[
            "sh".as_ref(),
            "-c".as_ref(),
            "if [ -e \"$1\" ]; then echo yes; else echo no; fi".as_ref(),
            "sh".as_ref(),
            JOURNAL.as_ref(),
        ])
        .output()?;
        match String::from_utf8_lossy(&output.stdout).trim() {
            "yes" => Ok(true),
            "no" => Ok(false),
            _ => bail!(
                "cannot check {JOURNAL}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        }
    }

    fn wait_lock_free(&mut self) -> Result<()> {
        process::MeasurementLock::wait(&self.lock).map(drop)
    }

    fn lock_holders(&mut self) -> String {
        process::lock_holders(&self.lock)
    }

    fn sleep(&mut self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

/// The fake machine, shared with `bench kernel`'s tests (bench.rs).
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use anyhow::anyhow;

    /// See [`Fake::on_step`].
    pub(crate) type OnStep = Box<dyn FnMut(&str)>;

    #[derive(Default)]
    pub(crate) struct Fake {
        pub(crate) thp: String,
        pub(crate) busy: bool,
        pub(crate) journal: bool,
        pub(crate) fail: &'static str,
        /// `unpatch` fails this many times before it succeeds.
        pub(crate) unpatch_failures: u32,
        pub(crate) state_file: PathBuf,
        pub(crate) log: Vec<String>,
        /// Called with the name of each step that succeeded (`patch`, `unpatch`, `thp=...`), so a
        /// test can change what the children see or cancel the run at that point.
        pub(crate) on_step: Option<OnStep>,
    }

    impl Fake {
        pub(crate) fn new(dir: &Path) -> Self {
            Fake {
                thp: "madvise".into(),
                journal: true,
                state_file: dir.join(STATE_FILE),
                ..Fake::default()
            }
        }

        fn step(&mut self, name: &str) -> Result<()> {
            self.log.push(name.to_owned());
            if self.fail == name {
                return Err(anyhow!("{name} failed"));
            }
            if let Some(on_step) = self.on_step.as_mut() {
                on_step(name);
            }
            Ok(())
        }
    }

    impl Host for Fake {
        fn gpu_processes(&mut self) -> Result<Vec<String>> {
            Ok(if self.busy {
                vec!["GPU-1, 123, python".into()]
            } else {
                vec![]
            })
        }
        fn read_thp(&mut self) -> Result<String> {
            Ok(self.thp.clone())
        }
        fn write_thp(&mut self, value: &str) -> Result<()> {
            // Nothing may change before the state file exists.
            assert!(
                self.state_file.is_file(),
                "THP changed before the state was saved"
            );
            self.step(&format!("thp={value}"))?;
            self.thp = value.into();
            Ok(())
        }
        fn patch(&mut self) -> Result<()> {
            let state: KernelState = process::read_json(&self.state_file)?;
            assert!(state.patched, "patch started before `patched` was saved");
            self.step("patch")
        }
        fn unpatch(&mut self) -> Result<()> {
            if self.unpatch_failures > 0 {
                self.unpatch_failures -= 1;
                self.log.push("unpatch failed".into());
                bail!("module in use");
            }
            self.step("unpatch")
        }
        fn journal_exists(&mut self) -> Result<bool> {
            Ok(self.journal)
        }
        fn wait_lock_free(&mut self) -> Result<()> {
            Ok(())
        }
        fn lock_holders(&mut self) -> String {
            "pid 1".into()
        }
        fn sleep(&mut self, duration: Duration) {
            self.log.push(format!("sleep {}", duration.as_secs()));
        }
    }

    fn run(fake: &mut Fake, measure_fails: bool) -> Result<()> {
        let path = fake.state_file.clone();
        with_kernel(fake, &path, || {
            if measure_fails {
                Err(anyhow!("interrupted by signal 1"))
            } else {
                Ok(())
            }
        })
    }

    #[test]
    fn success_restores_in_reverse_order_and_removes_state() {
        let dir = tempfile::tempdir().unwrap();
        let mut fake = Fake::new(dir.path());
        run(&mut fake, false).unwrap();
        assert_eq!(fake.log, ["thp=always", "patch", "unpatch", "thp=madvise"]);
        assert!(!fake.state_file.exists());
    }

    #[test]
    fn failed_or_cancelled_tune_still_unpatches_and_restores_thp() {
        let dir = tempfile::tempdir().unwrap();
        let mut fake = Fake::new(dir.path());
        let error = run(&mut fake, true).unwrap_err();
        assert!(format!("{error:#}").contains("signal 1"));
        assert_eq!(fake.log, ["thp=always", "patch", "unpatch", "thp=madvise"]);
    }

    #[test]
    fn panic_in_measure_restores_through_drop() {
        let dir = tempfile::tempdir().unwrap();
        let mut fake = Fake::new(dir.path());
        let path = fake.state_file.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_kernel(&mut fake, &path, || panic!("bug"))
        }));
        assert!(result.is_err());
        assert_eq!(fake.log, ["thp=always", "patch", "unpatch", "thp=madvise"]);
        assert!(!path.exists());
    }

    #[test]
    fn unpatch_is_retried_with_backoff_until_it_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let mut fake = Fake::new(dir.path());
        fake.unpatch_failures = 6;
        run(&mut fake, false).unwrap();
        let sleeps: Vec<_> = fake.log.iter().filter(|s| s.starts_with("sleep")).collect();
        assert_eq!(
            sleeps,
            ["sleep 5", "sleep 10", "sleep 20", "sleep 40", "sleep 60", "sleep 60"]
        );
        assert_eq!(fake.log.last().unwrap(), "thp=madvise");
    }

    #[test]
    fn leftover_state_from_a_killed_run_is_restored_first() {
        let dir = tempfile::tempdir().unwrap();
        let mut fake = Fake::new(dir.path());
        // A previous runner died after patching: THP is still `always`.
        fake.thp = "always".into();
        let state = KernelState {
            old_thp: "never".into(),
            patched: true,
        };
        process::write_json(&fake.state_file, &state).unwrap();
        let path = fake.state_file.clone();
        restore_from(&mut fake, &path).unwrap();
        assert_eq!(fake.log, ["unpatch", "thp=never"]);
        assert!(!fake.state_file.exists());
    }

    #[test]
    fn killed_during_patch_without_journal_needs_no_unpatch() {
        let dir = tempfile::tempdir().unwrap();
        let mut fake = Fake::new(dir.path());
        fake.journal = false;
        let state = KernelState {
            old_thp: "madvise".into(),
            patched: true,
        };
        process::write_json(&fake.state_file, &state).unwrap();
        let path = fake.state_file.clone();
        restore_from(&mut fake, &path).unwrap();
        assert!(fake.log.is_empty());
    }

    #[test]
    fn failed_patch_restores_thp_without_unpatch() {
        let dir = tempfile::tempdir().unwrap();
        let mut fake = Fake::new(dir.path());
        fake.fail = "patch";
        assert!(run(&mut fake, false).is_err());
        assert_eq!(fake.log, ["thp=always", "patch", "thp=madvise"]);
        assert!(!fake.state_file.exists());
    }

    #[test]
    fn failed_thp_restore_keeps_the_state_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut fake = Fake::new(dir.path());
        fake.fail = "thp=madvise";
        let error = run(&mut fake, false).unwrap_err();
        assert!(format!("{error:#}").contains("RESTORE FAILED"));
        assert!(
            fake.state_file.exists(),
            "the next run must retry the restore"
        );
    }

    #[test]
    fn busy_gpu_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut fake = Fake::new(dir.path());
        fake.busy = true;
        assert!(run(&mut fake, false).is_err());
        assert!(fake.log.is_empty() && !fake.state_file.exists());
    }

    #[test]
    fn thp_already_always_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let mut fake = Fake::new(dir.path());
        fake.thp = "always".into();
        run(&mut fake, false).unwrap();
        assert_eq!(fake.log, ["patch", "unpatch"]);
    }

    #[test]
    fn thp_parse_and_tune_flags() {
        assert_eq!(parse_thp("always [madvise] never\n").unwrap(), "madvise");
        assert!(parse_thp("garbage").is_err());
        let args = tune_args(
            "full_with_kernel",
            Path::new("/r/t"),
            0,
            Some("threads=1,2"),
            None,
            Some("thp"),
            true,
            false,
        );
        assert_eq!(
            args.join(" "),
            "tune --search full_with_kernel --out /r/t --device 0 --progress_bar off \
             --tuning_params threads=1,2 --pages thp --resume"
        );
    }
}
