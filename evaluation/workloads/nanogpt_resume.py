#!/usr/bin/env python3
"""nanoGPT (karpathy, unchanged): resume the README's Shakespeare finetune from a checkpoint, train a few iterations.

usage: nanogpt_resume.py setup       (needs the GPU once: writes the checkpoint by finetuning gpt2-large 5 iterations)
       nanogpt_resume.py run [--iterations N]       (default 10; one final JSON line: seconds for the whole run)

gpt2-large, not the README's gpt2-xl: gpt2-xl with AdamW needs about 25 GB and does not fit a 24 GB GPU.
Copies: `torch.load(ckpt, map_location="cuda")` reads the 9.3 GB checkpoint (model + AdamW moments) into pageable
memory and copies each tensor HtoD; load_state_dict then copies the model back into a CPU model (DtoH) and
`model.to("cuda")` copies it up again. Evaluation and saving are pushed past the run (--eval_interval), so the
checkpoint stays as setup wrote it and every run does the same work. torch.compile (the default) runs from a
warm cache kept in the env directory.
Evidence: A100 run of the same command, whole run 160/144/161 s stock vs 149/151/132 s with the hook
(2026-11): within noise.
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import appenv as c

ENV = c.ROOT / "nanogpt"
SOURCE = ENV / "nanoGPT"
CHECKPOINT = ENV / "checkpoint"
COMMIT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
PACKAGES = ["torch==2.14.1", "numpy", "transformers==5.22.1", "tiktoken==0.14.0", "requests"]
CHECKPOINT_ITER = 5
# TORCHINDUCTOR_CACHE_DIR keeps compiled kernels between runs, as on a user's machine after the first run.
APP_ENV = {"TORCHINDUCTOR_CACHE_DIR": str(ENV / "inductor-cache"), "TRITON_CACHE_DIR": str(ENV / "triton-cache")}


def setup():
    python = c.venv(ENV, PACKAGES, c.torch_index())
    c.git_checkout("https://github.com/karpathy/nanoGPT", COMMIT, SOURCE)
    if not (SOURCE / "data" / "shakespeare" / "val.bin").exists():
        c.sh([python, "data/shakespeare/prepare.py"], cwd=SOURCE)
    if not (CHECKPOINT / "ckpt.pt").exists():
        # Train into a side directory and rename it when done: a save cut short (a full disk) must
        # not leave a ckpt.pt that the existence check above takes as finished.
        partial = CHECKPOINT.with_name(CHECKPOINT.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        c.sh([python, "train.py", "config/finetune_shakespeare.py", "--init_from=gpt2-large",
              f"--out_dir={partial}", f"--max_iters={CHECKPOINT_ITER}", "--eval_iters=5",
              "--always_save_checkpoint=True"], cwd=SOURCE,
             # The legacy `gpt2-large` repo id fails the Xet token request (HTTP 404); plain HTTP works.
             env={**os.environ, **APP_ENV, "HF_HUB_DISABLE_XET": "1"})
        shutil.rmtree(CHECKPOINT, ignore_errors=True)
        partial.rename(CHECKPOINT)


def inner(iterations):
    import runpy

    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    sys.argv = ["train.py", "config/finetune_shakespeare.py", "--init_from=resume", f"--out_dir={CHECKPOINT}",
                f"--max_iters={CHECKPOINT_ITER + iterations}", "--eval_interval=100000"]
    start = time.perf_counter()
    runpy.run_path("train.py", run_name="__main__")
    seconds = time.perf_counter() - start
    c.emit("resume_run", round(seconds, 2), "s", iterations=iterations, model="gpt2-large")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["setup", "run", "inner"])
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.action == "setup":
        setup()
    elif args.action == "run":
        c.exec_app([ENV / "bin" / "python", Path(__file__).resolve(), "inner", "--iterations", args.iterations],
                   env=APP_ENV)
    else:
        inner(args.iterations)
