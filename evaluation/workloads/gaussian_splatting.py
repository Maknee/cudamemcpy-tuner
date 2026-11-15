#!/usr/bin/env python3
"""3D Gaussian Splatting training (graphdeco-inria train.py, unchanged) on the Tanks&Temples "train" scene.

usage: gaussian_splatting.py setup
       gaussian_splatting.py run gpu | cpu [--iterations N]     (one final JSON line: training seconds)

gpu  the default: all training images are moved to the GPU once at load (2.4 GiB pageable HtoD, then none).
cpu  --data_device cpu, the README's low-VRAM option: images stay in host memory and each iteration copies one
     image (about 8 MB, pageable HtoD).
Measured with `cmt trace` on an A100, 7,000 iterations: 55.7 GiB pageable HtoD with cpu vs 2.4 GiB with gpu.
A100 timing: the hook did not help cpu (164 -> 170 s).
The metric is the whole train.py run (scene load, training, final save), as the user waits for it.
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import appenv as c

ENV = c.ROOT / "gaussian-splatting"
SOURCE = ENV / "gaussian-splatting"
SCENE = ENV / "data" / "tandt" / "train"
COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"  # the commit the numbers above were measured with
DATA = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
PACKAGES = ["torch==2.14.1", "torchvision", "plyfile", "tqdm", "opencv-python-headless", "joblib", "setuptools",
            "wheel", "ninja"]
EXTENSIONS = ["submodules/diff-gaussian-rasterization", "submodules/simple-knn", "submodules/fused-ssim"]


def setup():
    cuda = c.cuda_home()
    python = c.venv(ENV, PACKAGES, c.torch_index(cuda))
    c.git_checkout("https://github.com/graphdeco-inria/gaussian-splatting", COMMIT, SOURCE, submodules=True)
    build = {"commit": COMMIT, "cuda": str(cuda), "packages": PACKAGES}
    if not c.stamped(ENV, "extensions", build):
        # The extensions must be compiled by an nvcc with torch's CUDA major (torch_index picks the wheels).
        # diff-gaussian-rasterization uses uint32_t without <cstdint>, which GCC >= 13 no longer pulls in
        # transitively; the forced include fixes the build without editing the submodule.
        env = {**os.environ, "CUDA_HOME": str(cuda), "PATH": f"{ENV / 'bin'}:{cuda / 'bin'}:{os.environ['PATH']}",
               "NVCC_APPEND_FLAGS": "-include cstdint"}
        c.sh([c.uv(), "pip", "install", "--quiet", "--python", python, "--no-build-isolation", *EXTENSIONS],
             cwd=SOURCE, env=env)
        c.stamp(ENV, "extensions", build)
    if not (SCENE / "sparse").exists():
        c.sh(["curl", "-fsSL", "-o", ENV / "tandt_db.zip", DATA])
        c.sh(["python3", "-m", "zipfile", "-e", ENV / "tandt_db.zip", ENV / "data"])
        (ENV / "tandt_db.zip").unlink()


def inner(mode, iterations):
    import runpy

    output = ENV / "output" / mode
    shutil.rmtree(output, ignore_errors=True)
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    sys.argv = ["train.py", "-s", str(SCENE), "-m", str(output), "--iterations", str(iterations), "--disable_viewer"]
    if mode == "cpu":
        sys.argv += ["--data_device", "cpu"]
    start = time.perf_counter()
    runpy.run_path("train.py", run_name="__main__")
    seconds = time.perf_counter() - start
    c.emit("train_time", round(seconds, 2), "s", iterations=iterations, it_per_s=round(iterations / seconds, 1),
           data_device="cpu" if mode == "cpu" else "cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["setup", "run", "inner"])
    parser.add_argument("mode", nargs="?", choices=["gpu", "cpu"])
    parser.add_argument("--iterations", type=int, default=7000)
    args = parser.parse_args()
    if args.action == "setup":
        setup()
    elif args.action == "run" and args.mode:
        c.exec_app([ENV / "bin" / "python", Path(__file__).resolve(), "inner", args.mode, "--iterations",
                    args.iterations])
    elif args.action == "inner" and args.mode:
        inner(args.mode, args.iterations)
    else:
        parser.error("run needs a mode")
