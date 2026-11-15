"""Load a Hugging Face causal LM onto the GPU, or run it with Accelerate big-model offload, and time it.

Operations (--op), library defaults otherwise:
  load     AutoModelForCausalLM.from_pretrained(model, dtype=bfloat16, device_map="cuda"). Safetensors are
           memory-mapped (pageable) and each tensor is copied CPU->GPU, one copy per tensor (Qwen3-8B: 16.4 GB).
           Not Qwen3.5-9B: transformers 5.22.1 gives its Qwen3_5ForCausalLM no device map, so device_map="auto"
           left it whole on the CPU, computing there and copying nothing (3090 smoke run, 2026-11-07). Metric: seconds. The first process after boot also reads the disk.
  offload  The Transformers "big model inference" recipe: device_map="auto" with max_memory={0: --gpu-budget}.
           Accelerate keeps the layers that do not fit in pageable CPU memory and copies each one CPU->GPU with
           a blocking .to() every time it runs, so every generated token streams all offloaded weights
           (measured 2026-11-02: 35 GB per token, 83% copy). Metric: s/token.
Gives up: answer quality (greedy, fixed prompt, a few tokens).
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--op", choices=["load", "offload"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--gpu-budget", default="3GiB", help="offload: max_memory for GPU 0 (default 3GiB)")
    parser.add_argument("--tokens", type=int, default=8, help="offload: generated tokens (default 8)")
    args = parser.parse_args()

    start = time.perf_counter()
    if args.op == "load":
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        weights = sum(p.numel() * p.element_size() for p in model.parameters())
        print(json.dumps({"metric": "load_seconds", "value": seconds, "unit": "s", "model": args.model,
                          "weight_bytes": weights, "gb_per_s": weights / seconds / 1e9}))
        return

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto",
                                                 max_memory={0: args.gpu_budget, "cpu": "512GiB"})
    load_seconds = time.perf_counter() - start
    # Accelerate keeps an offloaded weight on the "meta" device (its data sits in the CPU weights map), so count
    # everything not on the GPU; counting device "cpu" gave 0 for every run before 2026-11-07.
    offloaded = sum(p.numel() * p.element_size() for p in model.parameters() if p.device.type != "cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer("Explain what a PCIe bus does in one paragraph.", return_tensors="pt").to("cuda")

    def generate(tokens):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        out = model.generate(**prompt, max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False)
        torch.cuda.synchronize()
        return time.perf_counter() - begin, out

    generate(1)  # warm-up: kernels, allocator, Accelerate hooks
    seconds, out = generate(args.tokens)
    print(json.dumps({"metric": "seconds_per_token", "value": seconds / args.tokens, "unit": "s/token",
                      "model": args.model, "gpu_budget": args.gpu_budget, "offloaded_bytes": offloaded,
                      "tokens": args.tokens, "load_seconds": load_seconds,
                      "text": tokenizer.decode(out[0, prompt["input_ids"].shape[1]:])}))


if __name__ == "__main__":
    main()
