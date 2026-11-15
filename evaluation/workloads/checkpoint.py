"""Save or restore a training checkpoint the way common PyTorch code does, and time it.

Operations (--op), all with the library defaults:
  save   torch.save({"model": ..., "optimizer": ...}) as nanoGPT and many trainers do. Pickling a CUDA tensor
         calls .cpu(): one pageable GPU->CPU copy per tensor into fresh host memory.
  load   torch.load(..., map_location="cuda") + load_state_dict, nanoGPT's resume path: one pageable
         CPU->GPU copy per tensor.
  async  torch.distributed.checkpoint.async_save with its default writer; the metric is the time until it
         returns (training is blocked while it stages the state to CPU). Torch 2.5.1 stages with pageable
         .to("cpu") copies (BlockingAsyncStager without caching); newer Torch may stage differently, which is
         why this runs in both the torch251 and the torch environment.

State: GPT-2 shapes as in nanoGPT (fp32 weights + AdamW exp_avg/exp_avg_sq after one step). gpt2-large is
774M parameters, 9.3 GB of state; the largest tensor is the 257 MB token embedding. The weights are random:
the copy pattern depends only on tensor shapes, and random weights avoid a download.
Gives up: the end-to-end training job, and the disk. save/load use an in-memory file (io.BytesIO): with a real
file on this host, the second 9.3 GB save in a process took 62 s against 8 s for the first, because page-cache
writeback throttling made it disk-bound, which would hide any copy change. async_save writes real files under
--dir, but only its blocking staging time is measured.
"""
import argparse
import io
import json
import os
import shutil
import statistics
import tempfile
import time

import torch

PRESETS = {"gpt2-medium": (24, 1024), "gpt2-large": (36, 1280), "gpt2-xl": (48, 1600)}


def gpt2_parameters(layers, width, vocab=50304, context=1024):
    # Parameter names cannot contain dots, so nanoGPT's "h.0.attn.c_attn.weight" becomes "h0_attn_c_attn_weight".
    shapes = {"wte": (vocab, width), "wpe": (context, width), "ln_f_weight": (width,), "ln_f_bias": (width,)}
    for i in range(layers):
        for name, shape in {"ln_1_weight": (width,), "ln_1_bias": (width,), "attn_c_attn_weight": (3 * width, width),
                            "attn_c_attn_bias": (3 * width,), "attn_c_proj_weight": (width, width),
                            "attn_c_proj_bias": (width,), "ln_2_weight": (width,), "ln_2_bias": (width,),
                            "mlp_c_fc_weight": (4 * width, width), "mlp_c_fc_bias": (4 * width,),
                            "mlp_c_proj_weight": (width, 4 * width), "mlp_c_proj_bias": (width,)}.items():
            shapes[f"h{i}_{name}"] = shape
    module = torch.nn.Module()
    for name, shape in shapes.items():
        module.register_parameter(name, torch.nn.Parameter(torch.randn(shape, device="cuda") * 0.02))
    return module


def timed(operation):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return time.perf_counter() - start, value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--op", choices=["save", "load", "async"], required=True)
    parser.add_argument("--preset", choices=PRESETS, default="gpt2-large")
    parser.add_argument("--repeat", type=int, default=2, help="timed operations in this process (default 2)")
    parser.add_argument("--dir", default=tempfile.gettempdir(),
                        help="where checkpoint files go (default: the temp dir; the catalog passes $CMT_ENV_ROOT)")
    args = parser.parse_args()
    # Torch 2.5.1 (cu124) stops at sm_90, so on a Blackwell GPU it fails mid-run with "no kernel image". Run one
    # kernel to find out instead of matching the arch list: sm_100 cubins also run on sm_103 (B300), and PTX JITs.
    try:
        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    except RuntimeError as error:
        raise SystemExit(f"checkpoint.py: torch {torch.__version__} cannot run kernels on "
                         f"{torch.cuda.get_device_name()} (built for {torch.cuda.get_arch_list()}): {error}")

    model = gpt2_parameters(*PRESETS[args.preset])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()  # creates exp_avg / exp_avg_sq, as after the first training step
    model.zero_grad(set_to_none=True)
    state_bytes = sum(t.numel() * t.element_size() for t in model.parameters()) * 3

    work = tempfile.mkdtemp(prefix="cmt-ckpt-", dir=args.dir)
    seconds = []
    try:
        def save():
            buffer = io.BytesIO()
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
            return buffer
        saved = save() if args.op == "load" else None
        for i in range(args.repeat):
            if args.op == "save":
                saved = None  # free the previous buffer outside the timer: peak host memory stays ~1x the state
                s, saved = timed(save)
            elif args.op == "load":
                def restore():
                    saved.seek(0)
                    checkpoint = torch.load(saved, map_location="cuda", weights_only=False)
                    model.load_state_dict(checkpoint["model"])
                    optimizer.load_state_dict(checkpoint["optimizer"])
                s, _ = timed(restore)
            else:
                import torch.distributed.checkpoint as dcp
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
                s, future = timed(lambda: dcp.async_save(state, checkpoint_id=os.path.join(work, f"dcp{i}")))
                future.result()
            seconds.append(s)
    finally:
        shutil.rmtree(work)
    print(json.dumps({"metric": f"checkpoint_{args.op}_seconds", "value": statistics.median(seconds), "unit": "s",
                      "seconds": seconds, "state_bytes": state_bytes, "preset": args.preset,
                      "torch": torch.__version__, "gb_per_s": state_bytes / statistics.median(seconds) / 1e9}))


if __name__ == "__main__":
    main()
