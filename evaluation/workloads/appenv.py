"""Shared steps for the personal-use workload drivers (personal.json). Standard library only, Python >= 3.8.

Every driver has two commands: `setup` (idempotent: environment, sources, builds, models, data under
$CMT_ENV_ROOT/<app>, default <repo>/tmp/envs) and `run` (stock application, one final JSON line on stdout).

Trade-offs (on purpose):
- Drivers are started by /usr/bin/python3, and Python apps run in a uv venv over a uv-managed CPython. A conda or
  distro interpreter with file capabilities runs in secure mode, and glibc then drops LD_PRELOAD, so neither
  `cmt run` nor `cmt trace` would reach the app (seen with miniforge's python3.12).
  `run` refuses to start in secure mode instead of measuring stock code under a hook label.
- `cmt trace` records only the process it launched, not its children (src/trace.cpp). So `run` ends with exec
  of the CUDA process (an inner Python entry point or the native binary), and that process prints the JSON line.
  llama.cpp is the exception: its numbers come from parsing output, so its driver keeps a child process and the
  catalog gives a separate `trace` argv that execs the binary.
- Models come from the normal Hugging Face cache, so models already on the machine are not downloaded again.
- A stamp file per step holds the step's pinned inputs; a rerun with the same inputs is a no-op.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("CMT_ENV_ROOT") or REPO / "tmp" / "envs")
PYTHON = "3.12"
# uv, when none is on PATH: the release archive, checked against the SHA-256 its release publishes
# (https://github.com/astral-sh/uv/releases/download/<version>/<archive>.sha256).
UV_VERSION = "0.13.6"
UV_SHA256 = {
    "x86_64": "f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645",
    "aarch64": "229e194255c7958d86d56e8790d969a6f74736592133f0dfd0076f754c642c1d",
}


def log(message):
    print(f"[{Path(sys.argv[0]).stem}] {message}", file=sys.stderr, flush=True)


def sh(argv, **kwargs):
    log("$ " + " ".join(str(word) for word in argv))
    subprocess.run([str(word) for word in argv], check=True, **kwargs)


def stamped(directory, step, inputs):
    """True when `step` already finished in `directory` with the same pinned inputs."""
    stamp = Path(directory) / f".cmt-{step}"
    return stamp.exists() and stamp.read_text() == json.dumps(inputs, sort_keys=True)


def stamp(directory, step, inputs):
    (Path(directory) / f".cmt-{step}").write_text(json.dumps(inputs, sort_keys=True))


def download(url, dest, sha256):
    """Download https `url` to `dest` and check its SHA-256; nothing is kept when it differs."""
    dest = Path(dest)
    if dest.exists() and file_sha256(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    log(f"download {url}")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=600) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        partial.unlink()
        sys.exit(f"{url}: SHA-256 {digest.hexdigest()}, expected {sha256}")
    partial.replace(dest)
    return dest


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def uv():
    found = shutil.which("uv")
    if found:
        return found
    binary = ROOT / ".tools" / f"uv-{UV_VERSION}" / "uv"
    if not binary.exists():
        arch = platform.machine()
        if arch not in UV_SHA256:
            sys.exit(f"no pinned uv for {arch}: install uv (https://docs.astral.sh/uv/) and put it on PATH")
        name = f"uv-{arch}-unknown-linux-gnu"
        archive = download(f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{name}.tar.gz",
                           binary.parent / f"{name}.tar.gz", UV_SHA256[arch])
        # The archive holds <name>/uv and <name>/uvx.
        sh(["tar", "-xzf", archive, "-C", binary.parent, "--strip-components=1", f"{name}/uv", f"{name}/uvx"])
        archive.unlink()
    return str(binary)


def venv(env, packages, index=None):
    """Create or update the uv venv `env` with pinned `packages`; return its python."""
    python = Path(env) / "bin" / "python"
    inputs = {"python": PYTHON, "packages": packages, "index": index}
    if not stamped(env, "venv", inputs):
        sh([uv(), "venv", "--quiet", "--allow-existing", "--python", PYTHON, "--python-preference",
            "only-managed", env])
        extra = ["--index-url", index, "--extra-index-url", "https://pypi.org/simple",
                 "--index-strategy", "unsafe-best-match"] if index else []
        sh([uv(), "pip", "install", "--quiet", "--python", python, *extra, *packages])
        stamp(env, "venv", inputs)
    return python


def git_checkout(url, commit, dest, submodules=False):
    dest = Path(dest)
    if not (dest / ".git").exists():
        sh(["git", "clone", "--quiet", "--filter=blob:none", url, dest])
    head = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != commit:
        sh(["git", "-C", dest, "fetch", "--quiet", "origin", commit])
        sh(["git", "-C", dest, "checkout", "--quiet", "--force", commit])
    if submodules:
        sh(["git", "-C", dest, "submodule", "update", "--quiet", "--init", "--recursive"])
    return dest


def hf_download(python, repo, filename=None, allow=None, offline=False):
    """Fetch from the Hugging Face cache (downloads only what is missing); return the local path.
    `offline` resolves from the cache only, so a timed run never waits on the network."""
    if filename:
        code = f"from huggingface_hub import hf_hub_download as d; print(d({repo!r}, {filename!r}))"
    else:
        code = f"from huggingface_hub import snapshot_download as d; print(d({repo!r}, allow_patterns={allow!r}))"
    env = {**os.environ, "HF_HUB_OFFLINE": "1"} if offline else None
    out = subprocess.run([str(python), "-c", code], check=True, capture_output=True, text=True, env=env).stdout
    return Path(out.strip().splitlines()[-1])


def cuda_home():
    """The CUDA toolkit used for builds: $CUDA_HOME, nvcc on PATH, then /usr/local/cuda."""
    for candidate in (os.environ.get("CUDA_HOME"), shutil.which("nvcc") and Path(shutil.which("nvcc")).parents[1],
                      "/usr/local/cuda"):
        if candidate and (Path(candidate) / "bin" / "nvcc").exists():
            return Path(candidate)
    sys.exit("no CUDA toolkit (nvcc) found: install one and set CUDA_HOME")


def nvcc_major(home):
    out = subprocess.run([str(home / "bin" / "nvcc"), "--version"], capture_output=True, text=True).stdout
    return int(out.split("release ")[1].split(".")[0])


def torch_index(home=None):
    """PyTorch wheel index for this machine: CUDA 13 wheels need a CUDA 13 driver, and an app that builds CUDA
    extensions needs the same CUDA major in nvcc and in torch."""
    if home:
        major = nvcc_major(home)
    else:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
        major = int(re.search(r"CUDA (?:UMD )?Version: (\d+)", out).group(1))
    return "https://download.pytorch.org/whl/" + ("cu130" if major >= 13 else "cu126")


def refuse_secure_mode():
    # AT_SECURE (auxv type 23) is 1 when ld.so ignored LD_PRELOAD for this process (setuid or file capabilities).
    words = Path("/proc/self/auxv").read_bytes()
    pairs = [int.from_bytes(words[i:i + 8], sys.byteorder) for i in range(0, len(words), 8)]
    if dict(zip(pairs[::2], pairs[1::2])).get(23):
        sys.exit(f"{sys.executable} runs in secure mode (file capabilities?): LD_PRELOAD would be dropped")


def exec_app(argv, cwd=None, env=None):
    """Replace this process with the CUDA application (keeps the PID, so `cmt trace` still records it)."""
    refuse_secure_mode()
    if cwd:
        os.chdir(cwd)
    argv = [str(word) for word in argv]
    log("exec " + " ".join(argv))
    sys.stdout.flush()
    os.execve(argv[0], argv, {**os.environ, **(env or {})})


def emit(metric, value, unit, **extra):
    # Straight to fd 1: apps replace print/sys.stdout (3DGS appends a timestamp to every line), and
    # the runner reads this line as JSON.
    sys.stdout.flush()
    os.write(1, (json.dumps({"metric": metric, "value": value, "unit": unit, **extra}) + "\n").encode())


def warm_page_cache(path):
    """Read a file once so the timed run starts with it in the page cache; refuse when RAM cannot hold it."""
    size = Path(path).stat().st_size
    available = int(next(line for line in open("/proc/meminfo") if line.startswith("MemAvailable")).split()[1]) << 10
    if size > available:
        sys.exit(f"{path}: {size >> 30} GiB does not fit in available RAM ({available >> 30} GiB); warm cache impossible")
    chunk = bytearray(64 << 20)
    with open(path, "rb", buffering=0) as source:
        while source.readinto(chunk):
            pass
