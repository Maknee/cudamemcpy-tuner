#!/usr/bin/env python3
"""llama.cpp on one GPU, as a home user runs it: an MoE model larger than VRAM, and model startup.

usage: llama_cpp.py setup MODE
       llama_cpp.py run moe-prefill | load-mmap | load-nommap      (one final JSON line)
       llama_cpp.py exec MODE                                       (exec the binary itself, for `cmt trace`)

moe-prefill  Qwen3.6-35B-A3B Q8_0 (36.9 GB, larger than a 24 GB GPU), `-ngl 99 -ot exps=CPU`: attention and
             shared weights on the GPU, experts in the mmap'd file. For batches >= 32 tokens, ggml copies each CPU
             expert tensor to the GPU (pageable HtoD, MiB-sized) and multiplies there. Metric: llama-bench pp2048.
load-*       Qwen3.8-27B UD-Q4_K_XL (17.6 GB, fits) with every layer on the GPU, file in the page cache.
             mmap (default) copies weights from the mapping, pageable HtoD; --no-mmap reads into llama.cpp's own
             pinned staging buffers. Metric: seconds from launch to exit for a 4-token reply.
Evidence (an RTX 3090, 2026-11-02): hook pp2048 344 -> 464 tok/s (1.35x);
warm mmap load 2.71 -> 2.53 s (1.07x); --no-mmap load unchanged.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import appenv as c

ENV = c.ROOT / "llama-cpp"
SOURCE = ENV / "llama.cpp"
BIN = SOURCE / "build" / "bin"
COMMIT = "d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b"  # the build behind the evidence above
MODELS = {"moe": ("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-Q8_0.gguf"),
          "dense": ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_XL.gguf")}
MODES = {"moe-prefill": "moe", "load-mmap": "dense", "load-nommap": "dense"}


def model(mode, offline=True):
    return c.hf_download(ENV / "venv" / "bin" / "python", *MODELS[MODES[mode]], offline=offline)


def setup(mode):
    ENV.mkdir(parents=True, exist_ok=True)
    # CMake and Ninja come from PyPI: a bare cloud container may have a compiler but neither tool.
    python = c.venv(ENV / "venv", ["huggingface_hub==2.4.0", "cmake==4.4.4", "ninja==1.13.2"])
    tools = python.parent
    c.git_checkout("https://github.com/ggml-org/llama.cpp", COMMIT, SOURCE)
    cuda = c.cuda_home()
    build = {"commit": COMMIT, "cuda": str(cuda), "generator": "Ninja"}
    if not c.stamped(ENV, "build", build):
        shutil.rmtree(SOURCE / "build", ignore_errors=True)
        # `native` builds for the GPUs present, which is what a user compiling at home gets. The host compiler is
        # named because CMake may otherwise pair gcc for C with another C++ compiler found on PATH.
        c.sh([tools / "cmake", "-S", SOURCE, "-B", SOURCE / "build", "-G", "Ninja",
              f"-DCMAKE_MAKE_PROGRAM={tools / 'ninja'}", "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
              "-DCMAKE_CUDA_ARCHITECTURES=native", f"-DCMAKE_CUDA_COMPILER={cuda}/bin/nvcc",
              f"-DCMAKE_C_COMPILER={os.environ.get('CC', 'gcc')}", f"-DCMAKE_CXX_COMPILER={os.environ.get('CXX', 'g++')}",
              "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"])
        c.sh([tools / "cmake", "--build", SOURCE / "build", "-j", str(os.cpu_count()), "--target", "llama-bench",
              "llama-completion"])
        c.stamp(ENV, "build", build)
    c.log(f"model: {model(mode, offline=False)}")


def command(mode):
    if mode == "moe-prefill":
        return [BIN / "llama-bench", "-m", model(mode), "-ngl", "99", "-fa", "1", "-ot", "exps=CPU",
                "-p", "2048", "-n", "0", "-r", "3", "-o", "json"]
    no_mmap = ["--no-mmap"] if mode == "load-nommap" else []
    return [BIN / "llama-completion", "-m", model(mode), "-ngl", "99", "-fa", "on", "-c", "4096", "--fit", "off",
            *no_mmap, "-p", "Hello", "-n", "4", "-no-cnv", "--temp", "0", "-s", "1"]


def run(mode):
    argv = [str(word) for word in command(mode)]
    if mode != "moe-prefill":
        c.warm_page_cache(argv[argv.index("-m") + 1])
    start = time.perf_counter()
    done = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    seconds = time.perf_counter() - start
    sys.stderr.write(done.stderr)
    if done.returncode != 0:
        sys.exit(f"{argv[0]} exited with {done.returncode}")
    if mode == "moe-prefill":
        test = json.loads(done.stdout)[0]
        c.emit("prefill", round(test["avg_ts"], 2), "tokens/s", stddev=round(test["stddev_ts"], 2),
               samples_ts=test["samples_ts"], n_prompt=test["n_prompt"], build_commit=test["build_commit"],
               gpu=test["gpu_info"], wall_s=round(seconds, 2))
    else:
        load_ms = re.search(r"load time =\s*([0-9.]+) ms", done.stderr)
        c.emit("startup", round(seconds, 3), "s", llama_load_ms=load_ms and float(load_ms.group(1)),
               mmap=mode == "load-mmap")


if __name__ == "__main__":
    action, mode = (sys.argv[1:] + [None])[:2]
    if mode not in MODES:
        sys.exit(__doc__)
    elif action == "setup":
        setup(mode)
    elif action == "run":
        run(mode)
    elif action == "exec":
        c.exec_app(command(mode))
    else:
        sys.exit(__doc__)
