#!/usr/bin/env python3
"""Z-Image-Turbo text-to-image in two apps that offload weights by default: SD.Next and DiffSynth-Studio.

usage: zimage.py setup sdnext | diffsynth
       zimage.py run sdnext | diffsynth [--images N]      (one final JSON line: s/image)

Model: Tongyi-MAI/Z-Image-Turbo (6B DiT, Qwen3-4B text encoder, Flux VAE; bf16), downloaded once into the
Hugging Face cache and linked into each app's model folder. Both apps make 1024x1024 images with cfg 1.
sdnext     SD.Next server, txt2img through its HTTP API (what its UI calls), 9 steps (the model card's value).
           No offload flags: SD.Next picks its offload from the detected VRAM ("balanced"; GPU high watermark 0.8
           on a card with >= 22 GB), so the 20 GB of weights go to the GPU and back for every image.
diffsynth  DiffSynth-Studio's examples/z_image/model_inference_low_vram/Z-Image-Turbo.py as published (8 steps,
           vram_limit = total GPU memory - 0.5 GiB). Layers are copied to the GPU while memory allows, and after
           each image the pipeline offloads every model to the CPU (load_models_to_device([])).
Neither default needs a smaller memory limit to copy: on a 24 GB RTX 3090 each image copies 20.4-20.5 GB pageable HtoD
and as much pageable DtoH in both apps, 93% of it as 419 tensors of 16-256 MB (`cmt trace` of 4 images minus one
of 2 images on an RTX 3090, 2026-11-07).
Metric: median s/image over N images after a first one (CUDA init, first onload). load_s: process start to the
model being ready (SD.Next: server launch until a 1-step image comes back, so it includes server startup).
Gives up: image quality is not checked; one fixed prompt.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import appenv as c

STARTED = time.perf_counter()
MODEL = "Tongyi-MAI/Z-Image-Turbo"
# SD.Next lists Hugging Face cache folders (models--ORG--NAME/snapshots/...) found in <models>/Diffusers.
SDNEXT_LINK = Path("Diffusers") / f"models--{MODEL.replace('/', '--')}"
APPS = {  # app: (repository, commit pinned for this workload, run on 2026-11-05 by the A100 prototype)
    "sdnext": ("https://github.com/vladmandic/sdnext", "6657d06112ed8e924e93eed099af44ea8d892417"),
    "diffsynth": ("https://github.com/modelscope/DiffSynth-Studio", "08fdc0da8f41426885d9432e9007e6e460a4c30d"),
}
# SD.Next's first start installs its own pinned torch, diffusers and requirements with `uv pip`, but it installs
# diffusers from git with `python -m pip`, so its venv needs pip. DiffSynth pins nothing: torch and transformers
# are pinned here (the versions of the other workloads), the rest resolves at setup time.
PACKAGES = {"sdnext": ["pip"],
            "diffsynth": ["torch==2.14.1", "torchvision==0.29.1", "transformers==5.22.1", "accelerate==1.15.0"]}


def free_port():
    # Any fixed port can be taken (SD.Next's default 7860 by a server the user runs; 7861 by a cloud image's own nginx),
    # so the kernel picks a free one for each process; the server and its client are the same process tree.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


PORT = free_port()
# The DiffSynth example's prompt, used by both apps.
PROMPT = ("Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
          "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, "
          "bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night "
          "background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.")


def paths(app):
    home = c.ROOT / app
    return home, home / "src", home / "venv" / "bin" / "python", home / "models"


def link(target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.is_symlink() and os.readlink(path) == str(target)):  # Path.readlink needs Python 3.9
        path.unlink(missing_ok=True)
        path.symlink_to(target)


def revision(app):
    """The model snapshot the app read. Not pinned: SD.Next asks the hub for `main` at every load (see run_sdnext),
    so the JSON line names the revision that ran; DiffSynth reads the snapshot linked at setup."""
    models = paths(app)[3]
    if app == "diffsynth":
        return (models / MODEL).resolve().name
    return (models / SDNEXT_LINK / "refs" / "main").read_text().strip()


def setup(app):
    home, src, python, models = paths(app)
    home.mkdir(parents=True, exist_ok=True)
    c.git_checkout(*APPS[app], src)
    if app == "sdnext":
        c.venv(home / "venv", PACKAGES[app])
        install = {"commit": APPS[app][1]}
        if not c.stamped(home, "install", install):
            # --test installs, starts the server once (CUDA init) and exits. It runs without the model linked,
            # because a start with a model present loads it, and offline, because it also calls every pipeline
            # loader with a dummy checkpoint (pip still downloads; only the Hugging Face hub is off).
            (models / SDNEXT_LINK).unlink(missing_ok=True)
            env = {**os.environ, "VIRTUAL_ENV": str(home / "venv"), "HF_HUB_OFFLINE": "1",
                   "PATH": f"{Path(c.uv()).parent}{os.pathsep}{os.environ['PATH']}"}
            c.sh([python, "launch.py", "--test", "--uv", "--port", PORT, "--models-dir", models, "--data-dir",
                  home / "data"],
                 cwd=src, env=env)
            c.stamp(home, "install", install)
        link(c.hf_download(python, MODEL).parents[1], models / SDNEXT_LINK)
    else:
        c.venv(home / "venv", PACKAGES[app] + ["-e", str(src)], c.torch_index())
        # DiffSynth reads $DIFFSYNTH_MODEL_BASE_PATH/<model id>/<file pattern>.
        link(c.hf_download(python, MODEL), models / MODEL)


def api(path, body=None, timeout=1800):
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", headers={"Content-Type": "application/json"},
                                     data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"null")


def txt2img(steps, seed):
    """Seconds for one image, or None when the reply has no image (the model is not loaded)."""
    start = time.perf_counter()
    images = (api("/sdapi/v1/txt2img", {"prompt": PROMPT, "seed": seed, "steps": steps, "cfg_scale": 1.0,
                                        "width": 1024, "height": 1024, "send_images": True,
                                        "save_images": False}) or {}).get("images")
    return time.perf_counter() - start if images else None


def client(server, images):
    """Drive SD.Next's API; runs in a child, so the server keeps the PID that `cmt trace` follows. Ends the server:
    SIGINT after the JSON line (SD.Next exits 0), SIGKILL on any failure so the run fails."""
    ok = False
    try:
        # The runner sets no timeout, so every wait here has one: a server that never answers ends the run.
        while True:
            if os.getppid() != server:
                raise RuntimeError("the SD.Next server exited; its log is on stderr")
            if time.perf_counter() - STARTED > 600:
                raise RuntimeError("the SD.Next API did not answer in 10 minutes")
            try:
                models = api("/sdapi/v1/sd-models", timeout=10)
                break
            except OSError:
                time.sleep(2)
        title = next((m["title"] for m in models if "Z-Image-Turbo" in m["title"]), None)
        if title is None:
            raise RuntimeError(f"{MODEL} is not in SD.Next's model list: run `zimage.py setup sdnext`")
        api("/sdapi/v1/options", {"sd_model_checkpoint": title})
        # The startup load runs in its own thread, and txt2img returns no image until it is done: the A100
        # prototype timed empty replies. A 1-step image marks the model as ready.
        while not txt2img(1, 1):
            if time.perf_counter() - STARTED > 600:  # also the way a failed load ends: replies stay empty
                raise RuntimeError("the model did not load in 10 minutes")
            time.sleep(5)
        load_s = time.perf_counter() - STARTED
        seconds = [txt2img(9, 1000 + index) or 0.0 for index in range(images + 1)]
        if not all(seconds):
            raise RuntimeError(f"an image came back empty: {seconds}")
        options = api("/sdapi/v1/options")
        offload = {key: options.get(key) for key in ("diffusers_offload_mode", "diffusers_offload_min_gpu_memory",
                                                     "diffusers_offload_max_gpu_memory")}
        c.emit("image_time", round(statistics.median(seconds[1:]), 3), "s/image",
               images=[round(s, 3) for s in seconds[1:]], first_s=round(seconds[0], 3), load_s=round(load_s, 2),
               app="sdnext", commit=APPS["sdnext"][1], model_revision=revision("sdnext"), **offload)
        ok = True
    except Exception as error:  # any failure must end the server, or the run never ends
        c.log(f"client failed: {error!r}")
    finally:
        if os.getppid() == server:
            os.kill(server, signal.SIGINT if ok else signal.SIGKILL)
        os._exit(0 if ok else 1)


def run_sdnext(images):
    # A server already on the port would answer every request, without the hook or the tracer (A100 prototype).
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            sys.exit(f"port {PORT} is in use: refusing to time another server")
    home, src, python, models = paths("sdnext")
    server = os.getpid()
    if os.fork() == 0:
        client(server, images)
    # The server's log goes to stderr, so the client's JSON line stays the last line on stdout.
    os.dup2(2, 1)
    # SD.Next loads the transformer and text encoder by repository id from its own cache folder (default
    # models/huggingface: a second 31 GB download), so SD_HFCACHEDIR names the shared cache. Not offline: under
    # HF_HUB_OFFLINE, diffusers' sharded-checkpoint lookup raises instead of reading the cache, so the load asks
    # the hub for the model's revision once, before any image.
    hub = (models / SDNEXT_LINK).resolve().parent
    c.exec_app([python, "launch.py", "--port", PORT, "--models-dir", models, "--data-dir", home / "data",
                "--skip-all"], cwd=src, env={"SD_HFCACHEDIR": str(hub)})


def diffsynth(images):
    import torch
    from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig

    # The example, unchanged except for the timing around it.
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    vram_limit = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id=MODEL, origin_file_pattern="transformer/*.safetensors", **vram_config),
            ModelConfig(model_id=MODEL, origin_file_pattern="text_encoder/*.safetensors", **vram_config),
            ModelConfig(model_id=MODEL, origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id=MODEL, origin_file_pattern="tokenizer/"),
        vram_limit=vram_limit,
    )
    load_s = time.perf_counter() - STARTED
    seconds = []
    for index in range(images + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        pipe(prompt=PROMPT, seed=42 + index, rand_device="cuda")
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    c.emit("image_time", round(statistics.median(seconds[1:]), 3), "s/image", images=[round(s, 3) for s in seconds[1:]],
           first_s=round(seconds[0], 3), load_s=round(load_s, 2), app="diffsynth", commit=APPS["diffsynth"][1],
           model_revision=revision("diffsynth"),
           vram_limit_gib=round(vram_limit, 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["setup", "run", "inner"])
    parser.add_argument("app", choices=sorted(APPS))
    parser.add_argument("--images", type=int, default=3, help="timed images after the first (default 3)")
    args = parser.parse_args()
    if args.action == "setup":
        setup(args.app)
    elif args.action == "run" and args.app == "sdnext":
        run_sdnext(args.images)
    elif args.action == "run":
        home, src, python, models = paths("diffsynth")
        c.exec_app([python, Path(__file__).resolve(), "inner", "diffsynth", "--images", args.images], cwd=home,
                   env={"DIFFSYNTH_MODEL_BASE_PATH": str(models), "DIFFSYNTH_SKIP_DOWNLOAD": "True",
                        "HF_HUB_OFFLINE": "1"})
    elif args.app == "diffsynth":
        diffsynth(args.images)
    else:
        parser.error("inner is for diffsynth only")
