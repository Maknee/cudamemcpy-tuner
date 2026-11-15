"""Train a torchvision image classifier on real JPEGs and report images per second.

The loop follows torchvision's classification reference (ImageFolder, RandomResizedCrop(224), flip, normalize,
DataLoader workers, SGD, bf16 autocast, channels_last, `image.to("cuda", non_blocking=True)`).

Copies this shows:
- DataLoader input: one batch per step CPU->GPU. With DataLoader's default pin_memory=False the batch is pageable
  (batch 256 x 3x224x224 float32 = 154 MB per copy). --pin-memory 1 is the common fix and the pinned reference.
- --offload-activations cpu wraps the forward pass in torch.autograd.graph.save_on_cpu(), PyTorch's documented
  activation offload. Its default pin_memory=False makes every saved activation a pageable GPU->CPU copy in the
  forward pass and a CPU->GPU copy in the backward pass (tens of MB to ~100 MB each for ResNet at batch 128).
Gives up: accuracy (a few dozen steps, no evaluation). Data: Imagenette2-320 (setup.sh imagenette).
"""
import argparse
import json
import os
import time
from contextlib import nullcontext

import torch
import torchvision
from torchvision.transforms import v2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(os.environ.get("CMT_ENV_ROOT", "tmp/envs"), "data/imagenette2-320"))
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--pin-memory", type=int, choices=[0, 1], default=0, help="DataLoader pin_memory (default 0, as in DataLoader)")
    parser.add_argument("--offload-activations", choices=["none", "cpu"], default="none")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=25, help="timed steps after --warmup")
    args = parser.parse_args()

    transform = v2.Compose([v2.RandomResizedCrop(224), v2.RandomHorizontalFlip(), v2.PILToTensor(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = torchvision.datasets.ImageFolder(os.path.join(args.data, "train"), transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         pin_memory=bool(args.pin_memory), drop_last=True, persistent_workers=True)
    model = torchvision.models.get_model(args.model, num_classes=len(dataset.classes))
    model = model.cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    offload = torch.autograd.graph.save_on_cpu if args.offload_activations == "cpu" else nullcontext

    step, start, batches = 0, None, iter(loader)
    while step < args.warmup + args.steps:
        try:
            image, target = next(batches)
        except StopIteration:
            batches = iter(loader)
            continue
        if step == args.warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()
        image = image.to("cuda", non_blocking=True).contiguous(memory_format=torch.channels_last)
        target = target.to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16), offload():
            loss = criterion(model(image), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    print(json.dumps({"metric": "images_per_second", "value": args.steps * args.batch / seconds, "unit": "images/s",
                      "seconds": seconds, "steps": args.steps, "batch": args.batch, "model": args.model,
                      "pin_memory": bool(args.pin_memory), "offload_activations": args.offload_activations,
                      "batch_bytes": args.batch * 3 * 224 * 224 * 4, "final_loss": float(loss),
                      "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30, "torch": torch.__version__}))


if __name__ == "__main__":
    main()
