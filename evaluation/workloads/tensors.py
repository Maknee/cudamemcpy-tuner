"""Move ordinary (pageable) NumPy/PyTorch arrays to the GPU and back, as application code does, and time it.

  torch  x.to("cuda") and y.cpu()            (CPU->GPU from pageable memory; GPU->CPU into a fresh pageable tensor)
  jax    jax.device_put(x) and np.asarray(y)  (the same two directions through XLA's transfer manager)
Each size is copied --repeat times per direction after one untimed copy. Destinations are new allocations,
as in real code. Metric: total seconds over all timed copies (both directions); per-size GB/s are extras.
"""
import argparse
import json
import time

import numpy as np

SUFFIX = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


def torch_copies(nbytes):
    """Returns {direction: (prepare, copy)}; only copy(prepare()) is timed."""
    import torch
    host = torch.ones(nbytes, dtype=torch.uint8)
    device = host.to("cuda")
    sync = torch.cuda.synchronize
    return {"htod": (sync, lambda _: (host.to("cuda"), sync())),
            "dtoh": (sync, lambda _: device.cpu())}


def jax_copies(nbytes):
    import jax
    import jax.numpy as jnp
    host = np.ones(nbytes, dtype=np.uint8)
    device = jax.device_put(host).block_until_ready()
    # A jax.Array caches its host value after the first np.asarray, so each GPU->CPU copy reads a fresh array.
    return {"htod": (lambda: None, lambda _: jax.device_put(host).block_until_ready()),
            "dtoh": (lambda: jnp.copy(device).block_until_ready(), np.asarray)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework", choices=["torch", "jax"], required=True)
    parser.add_argument("--sizes", default="16M,256M,1G", help="bytes per copy, K/M/G suffixes (default 16M,256M,1G)")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    rows, total = [], 0.0
    for text in args.sizes.split(","):
        nbytes = int(text[:-1]) * SUFFIX[text[-1]] if text[-1] in SUFFIX else int(text)
        copies = (torch_copies if args.framework == "torch" else jax_copies)(nbytes)
        for direction, (prepare, copy) in copies.items():
            copy(prepare())
            seconds = 0.0
            for _ in range(args.repeat):
                source = prepare()
                start = time.perf_counter()
                copy(source)
                seconds += time.perf_counter() - start
            total += seconds
            rows.append({"bytes": nbytes, "direction": direction, "gb_per_s": nbytes * args.repeat / seconds / 1e9})
        del copies
    print(json.dumps({"metric": "copy_seconds", "value": total, "unit": "s", "framework": args.framework,
                      "repeat": args.repeat, "rows": rows}))


if __name__ == "__main__":
    main()
