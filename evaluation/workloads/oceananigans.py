#!/usr/bin/env python3
"""Oceananigans (Julia) GPU ocean simulation with 3D output: ocean.jl, the published wind-mixing example.

usage: oceananigans.py setup
       oceananigans.py run [--n 256] [--minutes 20]     (one final JSON line, printed by ocean.jl: seconds)

Julia is the official binary release (julialang-s3), checked against the SHA-256 julialang.org publishes, and every
Julia file (julia, packages, compiled code) stays in the env directory. `run` execs the julia binary itself, so
`cmt trace` records the process that copies.
Measured on an A100, N = 256, 20 saves: 112 s stock vs 108 s with the hook, 92 s with hook + registration cache.
"""

import argparse
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import appenv as c

ENV = c.ROOT / "oceananigans"
JULIA = "1.13.3"
# https://julialang-s3.julialang.org/bin/checksums/julia-<version>.sha256
JULIA_SHA256 = {
    "x86_64": "5c61cb4b40332cb5a3504e7779e2f5dac093caa4a1116ded19f01413f85c63e4",
    "aarch64": "9ceb053bee6dfb8189bd40afd6925de3a9b7ff8fac8d01a5fcdd84af171ddf03",
}
JULIA_URL_ARCH = {"x86_64": "x64", "aarch64": "aarch64"}
PACKAGES = 'Pkg.add(name="Oceananigans", version="0.117.1"); Pkg.add(["CUDA", "JLD2", "SeawaterPolynomials"])'
JULIA_ENV = {"JULIA_DEPOT_PATH": str(ENV / "depot"), "JULIA_PROJECT": str(ENV / "project")}


def julia_binary():
    return ENV / f"julia-{JULIA}" / "bin" / "julia"


def install_julia():
    arch = platform.machine()
    if arch not in JULIA_SHA256:
        sys.exit(f"no pinned Julia {JULIA} for {arch}")
    minor = ".".join(JULIA.split(".")[:2])
    name = f"julia-{JULIA}-linux-{arch}.tar.gz"
    archive = c.download(f"https://julialang-s3.julialang.org/bin/linux/{JULIA_URL_ARCH[arch]}/{minor}/{name}",
                         ENV / name, JULIA_SHA256[arch])
    # The archive's one top-level directory is julia-<version>/.
    c.sh(["tar", "-xzf", archive, "-C", ENV])
    archive.unlink()


def setup():
    ENV.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **JULIA_ENV}
    if not julia_binary().exists():
        install_julia()
    inputs = {"julia": JULIA, "packages": PACKAGES}
    if not c.stamped(ENV, "packages", inputs):
        # Loading CUDA and Oceananigans once precompiles them and downloads the CUDA runtime artifacts.
        c.sh([julia_binary(), "-e", f"using Pkg; {PACKAGES}; using CUDA, Oceananigans; CUDA.versioninfo()"],
             env=env)
        c.stamp(ENV, "packages", inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["setup", "run"])
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--minutes", type=float, default=20)
    args = parser.parse_args()
    if args.action == "setup":
        setup()
    else:
        out = ENV / "output"
        out.mkdir(exist_ok=True)
        c.exec_app([julia_binary(), Path(__file__).resolve().with_name("ocean.jl")], cwd=out,
                   env={**JULIA_ENV, "OCEAN_N": str(args.n), "OCEAN_OUT": "full", "OCEAN_STOP_MIN": str(args.minutes),
                        "OCEAN_DIR": str(out)})
