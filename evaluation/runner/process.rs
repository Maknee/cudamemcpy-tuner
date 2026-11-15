//! Child processes and the host: cancellation, owned process groups, the shared measurement lock,
//! the quiet-machine gate, root commands and atomic files.
//!
//! Every child runs in its own process group. On SIGINT/SIGTERM/SIGHUP the evaluator forwards the
//! signal to
//! that group, waits up to `CANCELLATION_GRACE`, then kills it. A child that exits normally still has
//! its group killed, so a launcher cannot leave workers measuring behind the next item.
//! Trade-off: a daemon that leaves the group (setsid) escapes; the evaluator cannot see it.
use anyhow::{bail, ensure, Context, Result};
use rustix::process::{kill_process_group, waitid, Pid, Signal, WaitId, WaitIdOptions};
use serde::Serialize;
use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
use std::{
    fs::File,
    io::Write,
    os::{
        fd::AsRawFd,
        unix::{fs::MetadataExt, process::CommandExt},
    },
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(20);
// Give an interrupted workload time to release GPU memory and files before its group is killed.
const CANCELLATION_GRACE: Duration = Duration::from_secs(5);

pub(crate) struct Cancellation {
    pub(crate) signal: Arc<AtomicUsize>,
    handlers: Vec<signal_hook::SigId>,
}

impl Cancellation {
    pub(crate) fn install() -> Result<Self> {
        let mut cancellation = Self::never();
        // SIGHUP too: closing the terminal must not kill the runner before it restores the machine.
        for signal in [SIGINT, SIGTERM, SIGHUP] {
            cancellation
                .handlers
                .push(signal_hook::flag::register_usize(
                    signal,
                    Arc::clone(&cancellation.signal),
                    signal as usize,
                )?);
        }
        Ok(cancellation)
    }

    /// A token that is never cancelled: restore steps (unpatch, THP) must finish after Ctrl-C.
    pub(crate) fn never() -> Self {
        Self {
            signal: Arc::new(AtomicUsize::new(0)),
            handlers: Vec::new(),
        }
    }

    pub(crate) fn signal(&self) -> Option<i32> {
        // Relaxed: the flag carries no data; the handler only stores the signal number.
        let signal = self.signal.load(Ordering::Relaxed);
        (signal != 0).then_some(signal as i32)
    }

    pub(crate) fn check(&self) -> Result<()> {
        if let Some(signal) = self.signal() {
            bail!("interrupted by signal {signal}");
        }
        Ok(())
    }
}

impl Drop for Cancellation {
    fn drop(&mut self) {
        for handler in &self.handlers {
            signal_hook::low_level::unregister(*handler);
        }
    }
}

fn signal_group(group: Pid, signal: Signal) -> Result<()> {
    match kill_process_group(group, signal) {
        Ok(()) | Err(rustix::io::Errno::SRCH) => Ok(()),
        Err(error) => Err(error).context("signal owned child process group"),
    }
}

/// Run `command` in a new process group that the evaluator owns, and reap it.
/// `lock` is made inheritable in the child only (the parent keeps CLOEXEC), so applications started
/// without it cannot keep the measurement lock alive after the evaluator exits.
pub(crate) fn run_child(
    command: &mut Command,
    lock: Option<&MeasurementLock>,
    cancellation: &Cancellation,
) -> Result<ExitStatus> {
    cancellation.check()?;
    let parent = rustix::process::getpid();
    let inherited = lock.map(|lock| lock.file.as_raw_fd());
    if let Some(lock) = lock {
        command
            .env("CMT_MEASUREMENT_LOCK_FD", lock.file.as_raw_fd().to_string())
            .env("CMT_MEASUREMENT_LOCK_PATH", &lock.path);
    }
    // SAFETY: after fork the callback only makes raw syscalls (prctl, getppid, fcntl) and builds an
    // errno value. It performs no allocation, locking or logging, so it is async-signal-safe.
    #[allow(unsafe_code)]
    unsafe {
        command.pre_exec(move || {
            // Linux kills the direct child if the evaluator dies. Set-id execs (sudo) clear it.
            rustix::process::set_parent_process_death_signal(Some(Signal::KILL))?;
            // The parent may have died before the call above; then no signal will come.
            if rustix::process::getppid() != Some(parent) {
                return Err(rustix::io::Errno::CHILD.into());
            }
            if let Some(fd) = inherited {
                // SAFETY: `fd` is the lock file, open in the parent for the whole run_child call.
                let fd = rustix::fd::BorrowedFd::borrow_raw(fd);
                rustix::io::fcntl_setfd(fd, rustix::io::FdFlags::empty())?;
            }
            Ok(())
        });
    }
    let mut child = command
        .process_group(0)
        .spawn()
        .with_context(|| format!("start {}", command.get_program().to_string_lossy()))?;
    let group = Pid::from_child(&child);
    let forward = |signal: i32| {
        signal_group(
            group,
            if signal == SIGINT {
                Signal::INT
            } else {
                Signal::TERM
            },
        )
    };
    let waited = (|| -> Result<()> {
        let mut forwarded = None;
        let mut interrupted_at: Option<Instant> = None;
        loop {
            if let Some(signal) = cancellation.signal() {
                if forwarded != Some(signal) {
                    forward(signal)?;
                    forwarded = Some(signal);
                    interrupted_at.get_or_insert_with(Instant::now);
                }
                if interrupted_at.is_some_and(|at| at.elapsed() >= CANCELLATION_GRACE) {
                    return signal_group(group, Signal::KILL);
                }
            }
            // NOWAIT keeps the exited leader as a zombie, so its PID (= the group id) cannot be
            // reused before the final group kill below.
            match waitid(
                WaitId::Pid(group),
                WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT,
            ) {
                Ok(Some(_)) => return signal_group(group, Signal::KILL),
                Ok(None) | Err(rustix::io::Errno::INTR) => {}
                Err(error) => return Err(error).context("observe child completion"),
            }
            std::thread::sleep(CHILD_POLL_INTERVAL);
        }
    })();
    if waited.is_err() {
        // Never return with a live child: kill, then reap below.
        if let Err(error) = signal_group(group, Signal::KILL) {
            crate::note!("cmt-evaluate: child cleanup: {error:#}");
        }
    }
    let status = child.wait().context("reap child")?;
    waited?;
    Ok(status)
}

/// Variables that change copy behaviour. Children never inherit them from the shell: they start
/// stock unless the item sets the hook itself (`cmt run`). CUDA device selection passes through.
pub(crate) fn is_hook_var(name: &str) -> bool {
    name == "LD_PRELOAD"
        || ["CMT_", "MEMCPY_HOOK_", "STAGING_", "CUDAMEMCPY_TUNER_"]
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

pub(crate) fn clean_command(program: &Path) -> Command {
    let mut command = Command::new(program);
    for (name, _) in std::env::vars_os() {
        let name = name.to_string_lossy();
        if is_hook_var(&name) {
            command.env_remove(name.as_ref());
        }
    }
    command
}

/// `sudo -n ARGS` (fail, never prompt), or ARGS directly when already root (containers often have
/// no sudo).
pub(crate) fn privileged(args: &[&std::ffi::OsStr]) -> Command {
    if rustix::process::geteuid().is_root() {
        as_root(args)
    } else {
        let mut command = Command::new("sudo");
        command.arg("-n").args(args);
        command
    }
}

/// PATH for commands run as root: the usual sudo `secure_path`.
const ROOT_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

/// ARGS as root without sudo, which would otherwise have reset the environment: the caller's PATH
/// and Python search paths must not choose what runs as root (`cmt patch` runs kmod.sh, which runs
/// python3). PATH also resolves ARGS[0] when it is a bare name.
fn as_root(args: &[&std::ffi::OsStr]) -> Command {
    let mut command = Command::new(args[0]);
    command.args(&args[1..]).env("PATH", ROOT_PATH);
    for name in [
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONSTARTUP",
    ] {
        command.env_remove(name);
    }
    command
}

const LOCK_NAME: &str = "cmt-measurements.lock";

/// Where the default measurement lock lives (src/runner.h, MeasurementLockPlaces). The production
/// values are fixed; tests point the bases at scratch directories and the uid at another user.
#[derive(Clone, Debug)]
pub(crate) struct LockPlaces {
    runtime_base: PathBuf,
    fallback_base: PathBuf,
    uid: u32,
}

impl LockPlaces {
    fn production() -> Self {
        Self {
            runtime_base: PathBuf::from("/run/user"),
            fallback_base: PathBuf::from("/tmp"),
            uid: measurement_uid(),
        }
    }

    fn fallback_dir(&self) -> PathBuf {
        self.fallback_base.join(format!("cmt-{}", self.uid))
    }

    /// /run/user/<uid>/cmt-measurements.lock where that runtime directory exists, else
    /// /tmp/cmt-<uid>/cmt-measurements.lock (a private directory, see `private_lock_directory`).
    fn default_lock(&self) -> PathBuf {
        let runtime = self.runtime_base.join(self.uid.to_string());
        match std::fs::symlink_metadata(&runtime) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                self.fallback_dir().join(LOCK_NAME)
            }
            // Present (or not inspectable): opened later without following a link.
            _ => runtime.join(LOCK_NAME),
        }
    }
}

/// The same lock `cmt tune` and `cmt patch --measurement-lock` use, so cooperating sessions never
/// measure at the same time. Held for the life of this value.
///
/// Default path: /run/user/<uid>/cmt-measurements.lock (uid: SUDO_UID under sudo, else the
/// effective uid), in the per-user runtime directory (mode 0700, created by the login session), so
/// no other user can create or replace it. Without that directory (containers, cloud images):
/// /tmp/cmt-<uid>/cmt-measurements.lock, in a directory created 0700 and refused unless it is a
/// real directory owned by <uid> with mode exactly 0700. The lock's directory is opened without
/// following a symbolic link; the file is opened relative to it without following one either, and
/// must be a regular file owned by that user or root.
pub(crate) struct MeasurementLock {
    file: File,
    pub(crate) path: PathBuf,
}

impl MeasurementLock {
    pub(crate) fn default_path() -> PathBuf {
        std::env::var_os("CMT_MEASUREMENT_LOCK_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| LockPlaces::production().default_lock())
    }

    /// Wait for the lock (blocking), saying who holds it.
    pub(crate) fn wait(path: &Path) -> Result<Self> {
        let file = Self::open(path, &LockPlaces::production())?;
        if file.try_lock().is_err() {
            crate::note!(
                "waiting for the measurement lock {} held by {}",
                path.display(),
                lock_holders(path)
            );
            file.lock()
                .with_context(|| format!("lock {}", path.display()))?;
        }
        Ok(Self {
            file,
            path: std::fs::canonicalize(path)?,
        })
    }

    fn open(path: &Path, places: &LockPlaces) -> Result<File> {
        let uid = places.uid;
        let dir = path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let name = path
            .file_name()
            .with_context(|| format!("measurement lock {} names no file", path.display()))?;
        let directory = if normalized(dir) == normalized(&places.fallback_dir()) {
            private_lock_directory(places)?
        } else {
            let flags = rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC;
            match rustix::fs::open(dir, flags, rustix::fs::Mode::empty()) {
                Ok(directory) => directory,
                Err(rustix::io::Errno::NOENT) => bail!(
                    "measurement lock {}: no directory {} (/run/user/<uid> is created by a login \
                     session; without one, pass --lock or set CMT_MEASUREMENT_LOCK_PATH to a file \
                     in a directory only you can write)",
                    path.display(),
                    dir.display()
                ),
                Err(rustix::io::Errno::LOOP | rustix::io::Errno::NOTDIR) => bail!(
                    "measurement lock {}: {} is not a directory, or is a symbolic link (not followed)",
                    path.display(),
                    dir.display()
                ),
                Err(error) => {
                    return Err(anyhow::Error::new(std::io::Error::from(error)).context(format!(
                        "open measurement lock directory {}",
                        dir.display()
                    )))
                }
            }
        };
        let open = |create: bool| {
            let mut flags = rustix::fs::OFlags::RDWR
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC;
            if create {
                flags |= rustix::fs::OFlags::CREATE | rustix::fs::OFlags::EXCL;
            }
            rustix::fs::openat(
                &directory,
                name,
                flags,
                rustix::fs::Mode::from_raw_mode(0o600),
            )
        };
        let mut created = false;
        let file = match open(false) {
            Err(rustix::io::Errno::NOENT) => match open(true) {
                Ok(file) => {
                    created = true;
                    Ok(file)
                }
                Err(rustix::io::Errno::EXIST) => open(false),
                Err(error) => Err(error),
            },
            other => other,
        }
        .map(File::from)
        .map_err(|error| {
            if error == rustix::io::Errno::LOOP {
                anyhow::anyhow!(
                    "measurement lock {} is a symbolic link; refusing to follow it",
                    path.display()
                )
            } else {
                anyhow::Error::new(std::io::Error::from(error))
                    .context(format!("open measurement lock {}", path.display()))
            }
        })?;
        drop(directory);
        let metadata = file
            .metadata()
            .with_context(|| format!("stat measurement lock {}", path.display()))?;
        ensure!(
            metadata.file_type().is_file(),
            "measurement lock {} is not a regular file",
            path.display()
        );
        if created && rustix::process::geteuid().is_root() && uid != 0 {
            // Root under sudo: a root-owned lock would stop the user's next `cmt tune` opening it.
            rustix::fs::fchown(&file, Some(rustix::process::Uid::from_raw(uid)), None)
                .with_context(|| format!("chown measurement lock {}", path.display()))?;
        }
        let owner = file
            .metadata()
            .with_context(|| format!("stat measurement lock {}", path.display()))?
            .uid();
        ensure!(
            owner == uid || owner == 0,
            "measurement lock {} belongs to uid {owner}, not to uid {uid} or root; remove it or \
             choose another path",
            path.display()
        );
        Ok(file)
    }

    /// Take the lock now or fail (tests check who holds it).
    #[cfg(test)]
    pub(crate) fn acquire(path: &Path) -> Result<Self> {
        Self::acquire_in(path, &LockPlaces::production())
    }

    #[cfg(test)]
    fn acquire_in(path: &Path, places: &LockPlaces) -> Result<Self> {
        let file = Self::open(path, places)?;
        file.try_lock().with_context(|| {
            format!(
                "another measurement owns {}; wait for it to finish",
                path.display()
            )
        })?;
        Ok(Self {
            file,
            path: std::fs::canonicalize(path)?,
        })
    }
}

/// `path` without `.` components or repeated separators, for comparing directory names.
fn normalized(path: &Path) -> PathBuf {
    path.components()
        .filter(|part| *part != std::path::Component::CurDir)
        .collect()
}

/// The fallback directory /tmp/cmt-<uid>, opened: created 0700 if missing (given to <uid> when
/// root creates it under sudo), opened without following a symbolic link, and refused unless it is
/// a directory owned by <uid> with mode exactly 0700. Anyone can create a name in /tmp first; these
/// checks make such a squatter fail loudly instead of sharing (or redirecting) the lock.
fn private_lock_directory(places: &LockPlaces) -> Result<std::os::fd::OwnedFd> {
    use rustix::fs::{Mode, OFlags};
    let path = places.fallback_dir();
    let name = format!("cmt-{}", places.uid);
    let base = rustix::fs::open(
        &places.fallback_base,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(std::io::Error::from)
    .with_context(|| format!("open {}", places.fallback_base.display()))?;
    let created = match rustix::fs::mkdirat(&base, name.as_str(), Mode::from_raw_mode(0o700)) {
        Ok(()) => true,
        Err(rustix::io::Errno::EXIST) => false,
        Err(error) => {
            return Err(
                anyhow::Error::new(std::io::Error::from(error)).context(format!(
                    "create measurement lock directory {}",
                    path.display()
                )),
            )
        }
    };
    let directory = rustix::fs::openat(
        &base,
        name.as_str(),
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|error| {
        anyhow::anyhow!(
            "measurement lock directory {} is not a directory, or is a symbolic link (not \
             followed): {}; inspect and remove it, or pass --lock",
            path.display(),
            std::io::Error::from(error)
        )
    })?;
    if created {
        // mkdir's mode is masked by the umask; the owner needs all of 0700.
        rustix::fs::fchmod(&directory, Mode::from_raw_mode(0o700))
            .map_err(std::io::Error::from)
            .with_context(|| format!("chmod {}", path.display()))?;
        if rustix::process::geteuid().is_root() && places.uid != 0 {
            rustix::fs::fchown(
                &directory,
                Some(rustix::process::Uid::from_raw(places.uid)),
                None,
            )
            .map_err(std::io::Error::from)
            .with_context(|| format!("chown {}", path.display()))?;
        }
    }
    let info = rustix::fs::fstat(&directory)
        .map_err(std::io::Error::from)
        .with_context(|| format!("stat {}", path.display()))?;
    let is_dir =
        rustix::fs::FileType::from_raw_mode(info.st_mode) == rustix::fs::FileType::Directory;
    let mode = info.st_mode & 0o7777;
    ensure!(
        is_dir && info.st_uid == places.uid && mode == 0o700,
        "measurement lock directory {} must be a directory owned by uid {} with mode 0700 (it is \
         owned by uid {}, mode {:04o}); inspect and remove it, or pass --lock",
        path.display(),
        places.uid,
        info.st_uid,
        mode
    );
    Ok(directory)
}

/// Whose measurement lock this is: under sudo the invoking user's (SUDO_UID), as kmod.sh and
/// `cmt tune` decide it, else the effective user's.
fn measurement_uid() -> u32 {
    let euid = rustix::process::geteuid();
    if euid.is_root() {
        if let Some(uid) = std::env::var("SUDO_UID")
            .ok()
            .and_then(|uid| uid.parse::<u32>().ok())
        {
            return uid;
        }
    }
    euid.as_raw()
}

/// Processes holding a flock on `path`, from /proc/locks, for "waiting for ..." messages.
pub(crate) fn lock_holders(path: &Path) -> String {
    let Ok(inode) = std::fs::metadata(path).map(|m| m.ino()) else {
        return "nobody (no lock file)".into();
    };
    let locks = std::fs::read_to_string("/proc/locks").unwrap_or_default();
    let holders: Vec<String> = locks
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            // "1: FLOCK ADVISORY WRITE <pid> <maj:min:inode> 0 EOF"
            let id = fields.get(5)?.rsplit(':').next()?.parse::<u64>().ok()?;
            (fields.get(1) == Some(&"FLOCK") && id == inode).then(|| fields[4].to_owned())
        })
        .map(|pid| {
            let command = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
            let command = String::from_utf8_lossy(&command).replace('\0', " ");
            format!(
                "pid {pid} ({})",
                command.trim().chars().take(120).collect::<String>()
            )
        })
        .collect();
    if holders.is_empty() {
        "an unknown process".into()
    } else {
        holders.join(", ")
    }
}

/// The quiet-machine gate. Before each measured item the runner waits until the 1-minute load
/// average, minus this runner's own recent children, is below `max_load`.
///
/// The kernel's 1-minute load is an exponential average (time constant 60 s) of runnable tasks, so
/// it stays high for minutes after the runner's own 16-thread child. `own` models that part: each
/// finished child adds its average CPU use (child CPU seconds / wall seconds, from /proc/self/stat
/// cutime+cstime) with the same exponential. Trade-off: a child that waits on the GPU without
/// spinning is runnable-free, so CPU use is a fair stand-in; one sleeping in D state is not.
pub(crate) struct Quiet {
    max_load: f64,
    own: f64,
    own_at: Instant,
    child_cpu: f64,
}

const LOAD_TIME_CONSTANT: f64 = 60.0;
const QUIET_POLL: Duration = Duration::from_secs(5);

impl Quiet {
    pub(crate) fn new(max_load: f64) -> Self {
        Self {
            max_load,
            own: 0.0,
            own_at: Instant::now(),
            child_cpu: children_cpu_seconds(),
        }
    }

    fn own_now(&self) -> f64 {
        self.own * (-self.own_at.elapsed().as_secs_f64() / LOAD_TIME_CONSTANT).exp()
    }

    /// 1-minute load average, and the part not explained by this runner's children.
    pub(crate) fn load(&self) -> (f64, f64) {
        let load1 = std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|text| text.split_whitespace().next()?.parse().ok())
            .unwrap_or(0.0);
        (load1, (load1 - self.own_now()).max(0.0))
    }

    /// Block until the other load is below the threshold (0 disables the gate).
    pub(crate) fn wait(&self, cancel: &Cancellation) -> Result<()> {
        let mut said = None::<Instant>;
        loop {
            cancel.check()?;
            let (load1, other) = self.load();
            if self.max_load <= 0.0 || other < self.max_load {
                return Ok(());
            }
            if said.is_none_or(|at| at.elapsed() >= Duration::from_secs(60)) {
                crate::note!(
                    "waiting for a quiet machine: 1-min load {load1:.2}, {other:.2} not from this \
                     runner, threshold {} (--max-load)",
                    self.max_load
                );
                said = Some(Instant::now());
            }
            std::thread::sleep(QUIET_POLL);
        }
    }

    /// Account for a child (or children) that ran from `started` until now.
    pub(crate) fn child_finished(&mut self, started: Instant) {
        let cpu = children_cpu_seconds();
        let wall = started.elapsed().as_secs_f64().max(1e-3);
        let used = ((cpu - self.child_cpu) / wall).max(0.0);
        self.child_cpu = cpu;
        let decay = (-wall / LOAD_TIME_CONSTANT).exp();
        self.own = self.own_now() * decay + used * (1.0 - decay);
        self.own_at = Instant::now();
    }

    /// Saved with each item so the report can flag runs measured under load.
    pub(crate) fn snapshot(&self) -> serde_json::Value {
        let (load1, other) = self.load();
        serde_json::json!({"load1": load1, "other": other, "max_load": self.max_load})
    }
}

/// CPU seconds of reaped children (fields 16 and 17 of /proc/self/stat, in clock ticks).
fn children_cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // The command name (field 2) may contain spaces; fields after its closing parenthesis are fixed.
    let fields: Vec<&str> = stat
        .rsplit_once(')')
        .map(|(_, rest)| rest.split_whitespace().collect())
        .unwrap_or_default();
    // After ")" the first field is field 3 (state), so field N is index N - 3.
    let ticks: f64 = [13, 14]
        .iter()
        .filter_map(|&i| fields.get(i)?.parse::<f64>().ok())
        .sum();
    ticks / rustix::param::clock_ticks_per_second() as f64
}

/// Stable 64-bit FNV-1a hash, for "did this input change" checks saved in result files.
pub(crate) fn hash_bytes(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Identity of every existing file named in `argv` (size and modification time): a rebuilt binary
/// or a rewritten tune.json changes it, so a saved result made with the old one is not reused.
/// `cmt run` and `cmt trace` also load a library argv does not name: libcmt_hook.so or
/// libcmt_trace.so beside cmt's real path (ResolvePreloadLibrary, src/runner.cpp).
pub(crate) fn files_identity(argv: &[String]) -> Vec<String> {
    let preloads = argv.windows(2).filter_map(|pair| {
        let library = match pair[1].as_str() {
            "run" => "libcmt_hook.so",
            "trace" => "libcmt_trace.so",
            _ => return None,
        };
        let cmt = Path::new(&pair[0]);
        let real = cmt
            .ends_with("cmt")
            .then(|| std::fs::canonicalize(cmt))
            .and_then(Result::ok)?;
        Some(real.with_file_name(library).to_string_lossy().into_owned())
    });
    argv.iter()
        .cloned()
        .chain(preloads)
        .filter(|arg| arg.starts_with('/'))
        .filter_map(|arg| {
            let meta = std::fs::metadata(&arg).ok().filter(|m| m.is_file())?;
            Some(format!(
                "{arg}:{}:{}.{}",
                meta.len(),
                meta.mtime(),
                meta.mtime_nsec()
            ))
        })
        .collect()
}

/// The GPU `--device N` selects and the driver, as nvidia-smi names them.
#[derive(Serialize)]
pub(crate) struct Gpu {
    pub(crate) uuid: String,
    pub(crate) driver: String,
}

/// None only where nvidia-smi is not installed; any other failure is an error, since a result
/// saved without its GPU would never match again.
pub(crate) fn gpu(device: u32) -> Result<Option<Gpu>> {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,driver_version",
            "--format=csv,noheader",
        ])
        .output()
    {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        output => output.context("nvidia-smi")?,
    };
    ensure!(
        output.status.success(),
        "nvidia-smi failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let order = std::env::var("CUDA_DEVICE_ORDER").ok();
    let listing = String::from_utf8_lossy(&output.stdout);
    select_gpu(&listing, visible.as_deref(), order.as_deref(), device).map(Some)
}

/// The row of `nvidia-smi --query-gpu=index,uuid,driver_version` that CUDA calls device `device`.
/// N counts the entries of CUDA_VISIBLE_DEVICES when it is set, as CUDA does. A numeric entry (or
/// N without the variable) is an nvidia-smi index, which is PCI bus order; CUDA uses that order
/// only under CUDA_DEVICE_ORDER=PCI_BUS_ID (its default lists the fastest GPU first), so with
/// several GPUs it is refused rather than guessed. A UUID entry may be a prefix, as in CUDA.
fn select_gpu(
    listing: &str,
    visible: Option<&str>,
    order: Option<&str>,
    device: u32,
) -> Result<Gpu> {
    let selector = match visible {
        Some(list) => list
            .split(',')
            .nth(device as usize)
            .with_context(|| {
                format!("--device {device}: CUDA_VISIBLE_DEVICES={list} has no such entry")
            })?
            .trim()
            .to_owned(),
        None => device.to_string(),
    };
    ensure!(
        !selector.is_empty(),
        "--device {device}: CUDA_VISIBLE_DEVICES is empty"
    );
    let rows: Vec<Vec<&str>> = listing
        .lines()
        .map(|line| line.split(", ").collect())
        .filter(|row: &Vec<&str>| row.len() == 3)
        .collect();
    let numeric = selector.parse::<u32>().is_ok();
    ensure!(
        !numeric || rows.len() <= 1 || order == Some("PCI_BUS_ID"),
        "GPU {selector}: CUDA numbers several GPUs fastest first; set CUDA_DEVICE_ORDER=PCI_BUS_ID \
         or CUDA_VISIBLE_DEVICES=<UUID>"
    );
    let row = rows
        .iter()
        .find(|row| {
            if numeric {
                row[0] == selector
            } else {
                row[1].starts_with(&selector)
            }
        })
        .with_context(|| format!("nvidia-smi lists no GPU {selector}"))?;
    Ok(Gpu {
        uuid: row[1].into(),
        driver: row[2].into(),
    })
}

/// Write JSON so a reader (or a resumed run) sees the old file or the complete new one, never a
/// prefix: temporary file in the same directory, fsync, rename.
pub(crate) fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let directory = path.parent().context("result path has no directory")?;
    std::fs::create_dir_all(directory)?;
    let mut file = tempfile::NamedTempFile::new_in(directory)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.as_file().sync_all()?;
    file.persist(path)
        .with_context(|| format!("save {}", path.display()))?;
    File::open(directory)?.sync_all()?;
    Ok(())
}

pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&text).with_context(|| format!("parse {}", path.display()))
}

/// Last `limit` bytes of a log, for error messages.
pub(crate) fn tail(path: &Path, limit: usize) -> String {
    let text = std::fs::read(path).unwrap_or_default();
    let start = text.len().saturating_sub(limit);
    String::from_utf8_lossy(&text[start..]).trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn write_json_replaces_whole_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.json");
        write_json(&path, &serde_json::json!({"n": 1})).unwrap();
        write_json(&path, &serde_json::json!({"n": 2})).unwrap();
        let value: serde_json::Value = read_json(&path).unwrap();
        assert_eq!(value["n"], 2);
        // Only the result remains: no temporary files left behind.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn lock_excludes_second_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lock");
        let first = MeasurementLock::acquire(&path).unwrap();
        assert!(MeasurementLock::acquire(&path).is_err());
        drop(first);
        // Other tests fork concurrently; a forked child shares the lock until it execs (CLOEXEC),
        // so the release can take a moment to become visible.
        let deadline = Instant::now() + Duration::from_secs(5);
        while MeasurementLock::acquire(&path).is_err() {
            assert!(Instant::now() < deadline, "lock not released");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn lock_refuses_symlink_and_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target");
        std::fs::write(&target, "").unwrap();
        let link = dir.path().join("lock");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let error = MeasurementLock::acquire(&link).err().unwrap().to_string();
        assert!(error.contains("symbolic link"), "{error}");
        // A dangling link must not create its target either.
        let dangling = dir.path().join("dangling");
        std::os::unix::fs::symlink(dir.path().join("created"), &dangling).unwrap();
        assert!(MeasurementLock::acquire(&dangling).is_err());
        assert!(!dir.path().join("created").exists());

        let missing = dir.path().join("no-such-dir").join("lock");
        let error = MeasurementLock::acquire(&missing)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("no directory"), "{error}");

        let fifo = dir.path().join("fifo");
        assert!(Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success());
        let error = MeasurementLock::acquire(&fifo).err().unwrap().to_string();
        assert!(error.contains("not a regular file"), "{error}");
    }

    /// Scratch stand-ins for /run/user and /tmp: production never changes them, tests cannot.
    fn scratch_places(dir: &Path, uid: u32) -> LockPlaces {
        let places = LockPlaces {
            runtime_base: dir.join("run-user"),
            fallback_base: dir.join("tmp"),
            uid,
        };
        std::fs::create_dir(&places.runtime_base).unwrap();
        std::fs::create_dir(&places.fallback_base).unwrap();
        places
    }

    fn mode(path: &Path) -> u32 {
        std::fs::symlink_metadata(path).unwrap().mode() & 0o7777
    }

    fn euid() -> u32 {
        rustix::process::geteuid().as_raw()
    }

    #[test]
    fn production_lock_places_are_fixed() {
        let places = LockPlaces::production();
        assert_eq!(places.runtime_base, PathBuf::from("/run/user"));
        assert_eq!(places.fallback_base, PathBuf::from("/tmp"));
        assert_eq!(places.uid, measurement_uid());
        if std::env::var_os("CMT_MEASUREMENT_LOCK_PATH").is_none() {
            assert_eq!(MeasurementLock::default_path(), places.default_lock());
        }
    }

    #[test]
    fn default_lock_is_in_the_user_runtime_dir_when_it_exists() {
        let dir = tempfile::tempdir().unwrap();
        let places = scratch_places(dir.path(), euid());
        let runtime = places.runtime_base.join(euid().to_string());
        std::fs::create_dir(&runtime).unwrap();
        let path = places.default_lock();
        assert_eq!(path, runtime.join("cmt-measurements.lock"));
        drop(MeasurementLock::acquire_in(&path, &places).unwrap());
        assert_eq!(mode(&path), 0o600);
        assert!(!places.fallback_dir().exists());
    }

    #[test]
    fn without_a_runtime_dir_the_lock_is_in_a_private_tmp_dir() {
        let dir = tempfile::tempdir().unwrap();
        let places = scratch_places(dir.path(), euid());
        let path = places.default_lock();
        assert_eq!(path, places.fallback_dir().join("cmt-measurements.lock"));
        let first = MeasurementLock::acquire_in(&path, &places).unwrap();
        assert!(MeasurementLock::acquire_in(&path, &places).is_err());
        drop(first);
        assert_eq!(mode(&places.fallback_dir()), 0o700);
        assert_eq!(mode(&path), 0o600);
    }

    #[test]
    fn fallback_dir_of_another_user_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        // The test cannot make another user's directory; it asks for another user's lock instead.
        let places = scratch_places(dir.path(), euid() + 1);
        std::fs::create_dir(places.fallback_dir()).unwrap();
        std::fs::set_permissions(
            places.fallback_dir(),
            std::fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        let path = places.default_lock();
        let error = MeasurementLock::acquire_in(&path, &places)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("owned by uid"), "{error}");
        assert!(!path.exists());
    }

    #[test]
    fn fallback_dir_others_can_read_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let places = scratch_places(dir.path(), euid());
        std::fs::create_dir(places.fallback_dir()).unwrap();
        std::fs::set_permissions(
            places.fallback_dir(),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        let path = places.default_lock();
        let error = MeasurementLock::acquire_in(&path, &places)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("mode 0700"), "{error}");
        assert!(!path.exists());
        assert_eq!(mode(&places.fallback_dir()), 0o755); // reported, never repaired
    }

    #[test]
    fn symlinked_fallback_dir_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let places = scratch_places(dir.path(), euid());
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::os::unix::fs::symlink(&real, places.fallback_dir()).unwrap();
        let error = MeasurementLock::acquire_in(&places.default_lock(), &places)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("symbolic link"), "{error}");
        assert_eq!(std::fs::read_dir(&real).unwrap().count(), 0);
    }

    #[test]
    fn symlinked_lock_in_fallback_dir_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let places = scratch_places(dir.path(), euid());
        std::fs::create_dir(places.fallback_dir()).unwrap();
        std::fs::set_permissions(
            places.fallback_dir(),
            std::fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        let target = dir.path().join("target");
        let path = places.default_lock();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        let error = MeasurementLock::acquire_in(&path, &places)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("symbolic link"), "{error}");
        assert!(!target.exists());
    }

    #[test]
    fn root_commands_get_a_fixed_path_and_no_python_paths() {
        let command = as_root(&["sh".as_ref(), "-c".as_ref(), "true".as_ref()]);
        let envs: std::collections::HashMap<_, _> = command.get_envs().collect();
        assert_eq!(
            envs[std::ffi::OsStr::new("PATH")],
            Some(std::ffi::OsStr::new(ROOT_PATH))
        );
        for name in [
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "PYTHONSTARTUP",
        ] {
            assert_eq!(envs[std::ffi::OsStr::new(name)], None, "{name}");
        }
        assert_eq!(command.get_program(), "sh");
    }

    #[test]
    fn children_cpu_is_counted_after_reaping() {
        let before = children_cpu_seconds();
        let mut command = Command::new("sh");
        command.args(["-c", "i=0; while [ $i -lt 300000 ]; do i=$((i+1)); done"]);
        let started = Instant::now();
        let mut quiet = Quiet::new(1.0);
        run_child(&mut command, None, &Cancellation::never()).unwrap();
        quiet.child_finished(started);
        assert!(
            children_cpu_seconds() > before,
            "busy child used no CPU time"
        );
        // A busy child accounts for most of a CPU while it ran.
        assert!(quiet.own > 0.0 && quiet.own_now() <= quiet.own);
    }

    #[test]
    fn hash_and_holders() {
        assert_eq!(hash_bytes(b""), "cbf29ce484222325");
        assert_ne!(hash_bytes(b"a"), hash_bytes(b"b"));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lock");
        let _lock = MeasurementLock::acquire(&path).unwrap();
        assert!(lock_holders(&path).contains(&format!("pid {}", std::process::id())));
    }

    #[test]
    fn device_is_counted_as_cuda_counts_it() {
        let two = "0, GPU-a, 580.1\n1, GPU-b, 580.1\n";
        let uuid = |visible, order, device| select_gpu(two, visible, order, device).map(|g| g.uuid);
        assert_eq!(uuid(Some("GPU-b,GPU-a"), None, 0).unwrap(), "GPU-b");
        assert_eq!(uuid(Some("GPU-a, GPU-b"), None, 1).unwrap(), "GPU-b");
        assert!(uuid(Some("GPU-a"), None, 1).is_err());
        // An index means PCI bus order only when CUDA is told to use it.
        assert!(uuid(None, None, 1).is_err());
        assert!(uuid(Some("1"), None, 0).is_err());
        assert_eq!(uuid(None, Some("PCI_BUS_ID"), 1).unwrap(), "GPU-b");
        assert_eq!(uuid(Some("1"), Some("PCI_BUS_ID"), 0).unwrap(), "GPU-b");
        assert_eq!(
            select_gpu("0, GPU-a, 580.1\n", None, None, 0)
                .unwrap()
                .driver,
            "580.1"
        );
    }

    #[test]
    fn child_sees_lock_and_parent_keeps_cloexec() {
        let dir = tempfile::tempdir().unwrap();
        let lock = MeasurementLock::acquire(&dir.path().join("lock")).unwrap();
        let mut command = Command::new("sh");
        command.args(["-c", "test -e /proc/self/fd/$CMT_MEASUREMENT_LOCK_FD"]);
        let status = run_child(&mut command, Some(&lock), &Cancellation::never()).unwrap();
        assert!(status.success());
        let flags = rustix::io::fcntl_getfd(&lock.file).unwrap();
        assert!(flags.contains(rustix::io::FdFlags::CLOEXEC));
    }
}
