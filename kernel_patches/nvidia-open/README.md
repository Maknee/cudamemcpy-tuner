# Host-memory registration optimizations

`cmt patch` selects the **loaded NVIDIA driver release**, downloads its exact open-kernel source commit from GitHub, builds the `host-registration` bundle **as root**, saves the original modules and arguments, then loads the new modules. `cmt unpatch` restores that saved state. Neither command changes THP, kills processes, installs user-space drivers, updates initramfs, or changes the modules used at the next boot.

## Read this first

- **Tested scope.** Only NVIDIA 615.71.09 has been loaded and tested, on one RTX 3090 with Linux 6.6.36 (x86_64). The other 58 releases in `releases.json` have only been checked to apply, and to compile on Linux 6.6.36. A's pre-Linux-5.16 path has never been compiled.
- **Requirements.** Linux 5.8 or newer (`minimum_kernel` in `releases.json`, checked before building or loading), Python 3, Git, Make, a C/C++ toolchain, binutils, kmod, the running kernel's headers, and an already working NVIDIA **open** kernel module of a listed release.
- **What it does to the machine.** It downloads NVIDIA source from GitHub (unless `--source` is given), compiles it as root, and loads the result with `insmod`. The display server and anything else using the GPU must be stopped first, and `nvidia_drm`, `nvidia_modeset` and `nvidia_peermem` must be unloaded; `cmt` does not stop or kill them for you.
- **Signed modules.** The modules are not signed. Kernels that enforce module signatures (Secure Boot lockdown, `module.sig_enforce`, `CONFIG_MODULE_SIG_FORCE`) are refused. Elsewhere, loading an unsigned, out-of-tree module **taints the kernel**, and the taint stays set after `cmt unpatch` until the next reboot.
- **Licenses.** The `*.patch` files in the release directories are GPL-2.0-only (see [COPYING](COPYING)); kmod.sh, releases.json and this README are Apache-2.0 like the rest of the repository. NVIDIA licenses the files they patch under MIT or GPL-2.0; with these patches applied, the modules you build are under GPL-2.0.

```sh
./cmt patch --check
# Check actual root privileges, without changing files or modules:
sudo ./cmt patch --check
./cmt patch --build-only --jobs 4
sudo ./cmt patch
sudo ./cmt unpatch
```

`sudo ./cmt` runs the `cmt` in the current directory; `sudo cmt` would search sudo's `secure_path`, which usually does not contain a user's build directory. Run as root, `cmt` runs `kernel_patches/nvidia-open/kmod.sh` from beside itself, and prints a note when that file or a directory above it is not owned by root or is writable by group or others: whoever can change it can change what root runs. A root-owned install (for example `cmake --install build --prefix /opt/cmt --component cmt`, as root) rules that out.

The preflight may refuse the load. Read its reason before proceeding. In particular, a root process still needs `CAP_SYS_MODULE`, an idle GPU, unload support, and a proven restoration path. NVIDIA display, modeset, peer-memory or other module users must stop separately.

The current workflow requires an already working NVIDIA **open** module. It does not infer open-module support from a GPU name.

## What the bundle changes

- **A: faster page pinning.** Pin in batches of 512 native Linux pages, yielding the CPU between batches; a batch is 2 MiB only when the CPU page size is 4 KiB. Where the kernel has `FOLL_NOFAULT` (Linux 5.16 and later, or a kernel that backported it as a macro), `pin_user_pages_fast` with `FOLL_NOFAULT` first pins the pages that are already mapped. It walks the page tables without `mmap_lock` first, and it can fall back to the kernel's locked path, but it never faults pages in. The rest of each batch is pinned by the unpatched driver's `pin_user_pages` call under `mmap_read_lock`. An earlier version of A let `pin_user_pages_fast` fault never-touched shared memory itself (the `/dev/zero` mapping behind `cudaHostAlloc`); it was measured 17% slower than the unpatched driver there on one host, and the cause was not established. Without `FOLL_NOFAULT`, A only splits the unpatched call into batches; that path has not been compiled.
- **B: batched dirty unpinning.** Use `unpin_user_pages_dirty_lock` instead of one dirty/unpin operation per page. In 620.70.06 upstream already releases pages through `NV_UNPIN_USER_PAGES_DIRTY`, which is `unpin_user_pages_dirty_lock` there, so B only adds the guard below.
- **No `FOLL_GET` fallback.** A pins with `pin_user_pages_fast` and B unpins with `unpin_user_pages_dirty_lock`. Both are `FOLL_PIN` operations. If the driver's configure step did not detect `pin_user_pages` (`NV_PIN_USER_PAGES_PRESENT`), the upstream code would fall back to `get_user_pages`/`put_page`; the patched `os-mlock.c` then refuses to compile (`#error`) instead of mixing the two.
- **D: larger eligible GPU mappings (on by default).** Check host and mapping-GPU address alignment and contiguity before permitting 2 MiB mappings. This checks actual addresses, not the global THP setting. Ordinary pages can qualify when the required runs are aligned and contiguous. `patch` requests `RMAllowSysmemLargePages=1`. Other registration settings are retained.
- A successful load does **not** show that a particular buffer received 2 MiB GPU mappings; the journal records mapping evidence as `not_measured`.

### Limits of D

- **All or nothing per registration.** A registration gets 2 MiB mappings only if its page count is a multiple of 512 and every 2 MiB run is aligned and physically contiguous in both the CPU and the owning GPU's DMA address space. One misaligned or non-contiguous run leaves the whole registration on 4 KiB mappings.
- **In practice, only with the IOMMU off or in passthrough mode.** With IOMMU translation, DMA addresses usually do not form aligned 2 MiB runs, so D rarely applies.
- **Mapping size has not been measured.** No test has yet shown that a buffer actually received 2 MiB GPU mappings.
- **Open review concern.** When a registration owned by one GPU is mapped for another GPU, `nv_gpu_ops.c` checks the mapping GPU's DMA addresses before using a large page. Other RM paths that map the memory for another device (for example indirect peer mappings in `virtual_mem.c`, or a duplicate of the memory for another device) have not been shown to re-check contiguity for that device. Until that is reviewed, prefer single-GPU machines, or IOMMU off/passthrough, where every GPU sees the same physical addresses.

### Verification so far

`releases.json` is the release catalog. It records the official source commit, exact patch paths and hashes, and separate source-application, compilation and load evidence. It covers the releases listed in `src/gen/versions_db.inc` that have NVIDIA open-module source (570.86.10 has none and has no entry); a driver-signature row by itself does not establish kernel-patch support. An `apply_checked` entry permits a build attempt, not a claim that the modules compiled, loaded or improved an application.

- All 59 current patch sets apply to their exact source commits without offset or fuzz (checked 2026-11-12).
- With the previous revision of the patch files (`verification.evidence_patches` in `releases.json`), both modules of all 59 releases compiled on x86_64 Linux 6.6.36 without warnings in the patched files. The current files add license headers, comments, the `#error` guards, a `defined(FOLL_NOFAULT)` test, and remove an unused define. The current files were compiled for 570.211.01, 615.71.09 and 620.70.06. For those three, the compiled `os-mlock.o` disassembly is identical to the previous revision's.
- Only 615.71.09 has been loaded, with the previous revision of the patch files, on an RTX 3090 with Linux 6.6.36. It was loaded 11 times with `cmt patch`, interleaved with the unpatched modules. Every load was followed by `cmt unpatch`, which restored the exact original modules and parameters each time. On that host, registration correctness matched the unpatched driver, including a range with a hole, read-only and copy-on-write pages, and untouched shared memory. DMA data checks passed in both directions, and pinned pages balanced with released pages.
- Performance was measured on one RTX 3090 host only; the data is not included in this repository. Registering already-touched memory was faster than unpatched there, and pinning fresh `cudaHostAlloc` or untouched anonymous memory took the same time. Treat these as anecdotes and measure on your own machine.

## Build and file ownership

[NVIDIA's build instructions](https://github.com/NVIDIA/open-gpu-kernel-modules#how-to-build) require matching user-space and GSP firmware for the selected driver release. The kernel interface must use the running kernel's headers and a compatible compiler. CMT additionally checks that its pin/unpin APIs are exported and that modules can be unloaded. No CUDA toolkit version is used to select the driver release.

The running kernel's complete headers and `Module.symvers` must be available at `/lib/modules/$(uname -r)/build`. Existing `/dev/nvidia-uvm` and `/dev/nvidia-uvm-tools` character nodes are required; CMT does not create or repair them. Restoring a compressed original module also needs its decompressor; gzip and xz use Python's standard library, while zstd uses the `zstd` command. `kmod.sh` runs its tools with `PATH=/usr/sbin:/usr/bin:/sbin:/bin`, and Python in isolated mode (`python3 -I`).

`--work-dir` selects the source/build cache. As root it defaults to `/var/cache/cmt/kernel`; otherwise to `~/.cache/cmt/kernel`. As root, the work directory and its existing ancestors must be root-owned and not writable by others, which is checked before anything is created there. `--source PATH` reuses a local checkout's Git objects without copying its working changes or changing that checkout. The commit is fetched from it with `git fetch`, so objects are copied and hash-checked. Git reads the checkout's configuration, which can run commands, so as root `--source` must be a checkout that root owns and nobody else can write. Builds produce independent `stock` and `host-registration` sets and retain their logs. The cache key includes the running kernel build, header ABI/config hashes, compiler identity, source commit and patch hashes. A failed or partial build directory is preserved; use another work directory while investigating it.

The default recovery journal is `/var/lib/cmt/kernel/state.json`. Privileged work, journals and loaded module artifacts must be root-owned and not writable by others. A build made as an ordinary user can be inspected, but loading it requires a separately protected artifact location. `--build-only` does not create the recovery journal or change modules.

`--measurement-lock PATH` defaults to the lock `cmt tune` and `cmt-evaluate` hold while they measure, `/run/user/<uid>/cmt-measurements.lock` (under `sudo`, the invoking user's uid and runtime directory; `CMT_MEASUREMENT_LOCK_PATH` overrides it), so `cmt patch` refuses while a tune of that user runs. If that runtime directory does not exist (the user has no login session), the default is `/tmp/cmt-<uid>/cmt-measurements.lock`, in a directory created with mode 0700 for that user and refused unless it is a directory (not a symlink) owned by that user with exactly that mode; `/run/user/<uid>` itself is never created. The lock is opened without following symlinks and must be owned by that user or by root. This excludes cooperating sessions only. Kernel holder and device-user checks run again before unloading; the lock is not proof that every external program cooperates. Module changes on the machine are also serialized by `/run/cmt/kernel.lock`, in a root-owned `0700` directory.

## Exact restoration

Automatic backup compares the loaded and original object identities, including the GNU build-id where available. It saves the loaded parameter values and reconstructs only replayable arguments. A source-proven nullable parameter can remain unset; the literal display string `(null)` is never passed as its value.

Some NVIDIA parameters, including `NvSwitchRegDwords` and `NvSwitchBlacklist`, are hidden from sysfs. Their default values do not prove the values used to load the current module. If any argument cannot be established, normal `patch` stops before downloading/building/loading and names the missing evidence. `--build-only` can still prepare artifacts without changing modules.

After a successful `unpatch`, a later `patch` can reuse the journal's recorded original set without another `--restore-set`. The journal and saved files must remain protected, and its boot, kernel, driver release, live module/parameter snapshot, user-space libraries, firmware, manifest and original arguments must still match. `--check` reports `restoration_source: "restored_journal"` when these checks pass. Active, incomplete or stale journals are not reused. An explicit `--restore-set` always takes precedence; hidden parameter defaults are never inferred.

`--restore-set DIRECTORY` supplies recorded original load evidence for this case. The directory contains the original `nvidia.ko`, and `nvidia-uvm.ko` if it was loaded, plus:

- `manifest.json`: schema `cmt.kernel-set`, version `1`, `variant: "restore"`, exact `release` and running `kernel`, and `modules` keyed by `nvidia` / `nvidia_uvm`. Each module records `name`, `version`, `srcversion`, `vermagic`, `build_id` (GNU descriptor hex or null), and the file's `sha256`.
- `restore.json`: schema `cmt.kernel-restore`, version `1`, `loaded` equal to a fresh `CMT_INTERNAL_ENABLE=1 cmt kernel inspect` snapshot's `loaded` object, and `arguments` containing the recorded original `NAME=VALUE` arrays for both modules. An originally absent module has an empty argument array.

The argument arrays must come from original load records. Creating a new empty list does not establish that the old load used defaults. The supplied identities and observable arguments are checked again before a switch. Persistent `patch` copies the originals into its protected state directory before unloading anything.

`unpatch` restores from the journal's own records (the saved original modules, their manifest and arguments); it does not need the current `releases.json` to list the patched release.

Failed switches attempt restoration. A journal-write error does not skip that attempt. `SIGINT`, `SIGTERM` and `SIGHUP` (for example a dropped SSH session) during a switch also trigger restoration, and are ignored while restoring. Keep the journal if restoration fails, fix the reported blocker, then run `sudo ./cmt unpatch --state PATH`. Recovery checks the boot ID, kernel, user-space identity, saved objects and arguments. No process can guarantee automatic recovery after a kernel crash, power loss or `SIGKILL`.

**After a reboot.** A reboot discards the module switch, and the journal's recorded state cannot be replayed in a new boot. If the modules and parameters loaded after the reboot equal the journal's recorded originals, `patch` and `unpatch` archive the old journal as `state-stale-<time>.json` next to it and say so. Otherwise they refuse and explain what to do: check the loaded modules with `CMT_INTERNAL_ENABLE=1 cmt kernel inspect`, load the driver you want (the saved originals are in the journal's `restore_set` directory), then move the journal aside as root and retry.

An explicit one-command comparison can still use the same implementation:

```sh
sudo CMT_INTERNAL_ENABLE=1 ./cmt kernel run --measurement-lock /absolute/gpu.lock \
  --set /root/kernel-build/host-registration --restore-set /root/original-set \
  --state /root/run-state.json --large-system-pages -- COMMAND ARGUMENTS
```

That form restores on command completion; `COMMAND` runs as root in the directory `cmt` was started from. `cmt patch` leaves the bundle active until `cmt unpatch`. Tuning reads the default journal and live module identity; it never loads modules itself.
