#!/usr/bin/env python3
"""diffusers text-to-image with CPU offloading, the published low-VRAM recipes.

usage: diffusers_offload.py setup
       diffusers_offload.py run flux2-klein | flux2-klein-group | sdxl [--images N]   (one final JSON line: s/image)

flux2-klein  FLUX.2 [klein] 4B, exactly the model card example: enable_model_cpu_offload(), 1024x1024, 4 steps,
             guidance 1.0. Each image moves the text encoder (8 GB), DiT (7.8 GB) and VAE to the GPU and back to
             CPU when the pipeline frees its hooks: pageable HtoD and DtoH of whole models, tensor by tensor.
flux2-klein-group
             The same model and call with the pipeline's enable_group_offload(block_level, 1 block per group) and
             its default use_stream=False (diffusers "Reduce memory usage" guide, group offloading), VAE kept on
             the GPU: every text-encoder and DiT block is copied pageable HtoD before it runs and back DtoH after.
sdxl         SDXL base 1.0 fp16 with enable_sequential_cpu_offload() (diffusers "Reduce memory usage" guide, the
             lowest-VRAM option), 1024x1024, the pipeline's default 50 steps. Every leaf module's weights are
             copied pageable HtoD on every forward (5 GB of UNet per step); nothing is copied back.
The first image is a warm-up (CUDA/cuDNN init, first offload); the metric is the mean of the next N images.
Evidence (an RTX 3090 on a PCIe x4 link, 2026-11): model offload 8.73 -> 7.25 s/image with the hook,
mostly from faster DtoH into fresh pageable memory; group offload without a stream 5.93 -> 4.77 s/step (the DiT
alone); sequential offload -2% (pageable HtoD).
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import appenv as c

ENV = c.ROOT / "diffusers"
PACKAGES = ["torch==2.14.1", "diffusers==0.42.1", "transformers==5.22.1", "accelerate==1.15.0", "safetensors",
            "huggingface_hub==2.4.0", "sentencepiece"]
RECIPES = {"flux2-klein": "flux2-klein", "flux2-klein-group": "flux2-klein", "sdxl": "sdxl"}  # recipe -> model
MODELS = {  # repo, files to fetch (skips the single-file checkpoints and sample images the pipelines do not read)
    "flux2-klein": ("black-forest-labs/FLUX.2-klein-4B", ["*.json", "*.txt", "text_encoder/*", "tokenizer/*",
                                                         "transformer/*", "vae/*"]),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", ["*.json", "*.txt", "*/*.fp16.safetensors"]),
}


def inner(recipe, images):
    import torch
    from diffusers import DiffusionPipeline

    model = RECIPES[recipe]
    start = time.perf_counter()
    if model == "flux2-klein":
        pipe = DiffusionPipeline.from_pretrained(MODELS[model][0], torch_dtype=torch.bfloat16)
        if recipe == "flux2-klein-group":
            # exclude_modules: block-level groups leave the VAE's decoder convs on the CPU in diffusers 0.42.1
            # ("weight type CPUBFloat16Type" at decode), so the small VAE stays on the GPU, as that option does.
            pipe.enable_group_offload(onload_device=torch.device("cuda"), offload_type="block_level",
                                      num_blocks_per_group=1, exclude_modules=["vae"])
        else:
            pipe.enable_model_cpu_offload()
        call = dict(prompt="A cat holding a sign that says hello world", height=1024, width=1024,
                    guidance_scale=1.0, num_inference_steps=4)
    else:
        pipe = DiffusionPipeline.from_pretrained(MODELS[model][0], torch_dtype=torch.float16, variant="fp16",
                                                 use_safetensors=True)
        pipe.enable_sequential_cpu_offload()
        call = dict(prompt="An astronaut riding a green horse", height=1024, width=1024)
    load_s = time.perf_counter() - start
    seconds = []
    for index in range(images + 1):
        start = time.perf_counter()
        pipe(**call, generator=torch.Generator("cuda").manual_seed(index)).images[0]
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    timed = seconds[1:]
    c.emit("image_time", round(sum(timed) / len(timed), 3), "s/image", images=[round(s, 3) for s in timed],
           warmup_s=round(seconds[0], 3), load_s=round(load_s, 2), model=MODELS[model][0], recipe=recipe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["setup", "run", "inner"])
    parser.add_argument("model", nargs="?", choices=sorted(RECIPES))
    parser.add_argument("--images", type=int, default=3)
    args = parser.parse_args()
    if args.action == "setup":
        python = c.venv(ENV, PACKAGES, c.torch_index())
        for repo, allow in MODELS.values():
            c.log(f"{repo}: {c.hf_download(python, repo, allow=allow)}")
    elif args.action == "run" and args.model:
        c.exec_app([ENV / "bin" / "python", Path(__file__).resolve(), "inner", args.model, "--images", args.images],
                   env={"HF_HUB_OFFLINE": "1"})
    elif args.action == "inner" and args.model:
        inner(args.model, args.images)
    else:
        parser.error("run needs a model")
