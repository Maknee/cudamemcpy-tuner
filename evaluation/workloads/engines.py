"""Start an LLM inference engine (vLLM or SGLang) or cycle vLLM sleep mode, and time it.

Operations (--op), engine defaults otherwise:
  startup  Construct the offline engine (vllm.LLM / sglang.Engine) and generate one token. Metrics: seconds from
           process start to ready and to the first token. Weight loading copies the safetensors (pageable,
           memory-mapped) CPU->GPU, one copy per tensor: Qwen3.5-9B is 19 GB. Compile and CUDA-graph capture are
           the other big terms. --caches picks their caches (vLLM, SGLang, Inductor, Triton, FlashInfer):
           warm = $CMT_ENV_ROOT/engine-cache/<engine>, filled by the runner's untimed warm-up run, as on a
           restarted server; cold = a new empty directory per run, deleted at exit, as in a fresh container
           (FlashInfer may then download prebuilt kernels for newer GPUs; on the 3090 it compiles them).
           The weight files stay in the page cache in both, so neither measures the disk. Also reported:
           the engine's own weight-load seconds (vLLM: DefaultModelLoader.load_weights, timed here; SGLang:
           startup_time["load_weight"]), the part of startup the copies are in.
  sleep    vLLM sleep mode, as RL trainers (verl, OpenRLHF) use it between rollouts: generate, sleep, wake up.
           Level 1 backs the weights up to CPU and restores them: vLLM allocates the backup with
           pin_memory=True, so the first sleep pins ~1.3x the weight bytes through torch's host allocator
           (cudaHostAlloc; measured 11 GB in 4.9 s for Qwen3-4B's 8 GB) and later cycles reuse it. Level 2 discards
           the weights and reload_weights() reads them again from the safetensors files: all weight bytes as
           pageable CPU->GPU copies per cycle (measured with `cmt trace` on an RTX 3090). Metric: sum of sleep + wake (+ reload) seconds.
           --backup pageable allocates the level-1 backup pageable instead, as vLLM does by default under WSL2
           (CudaPlatform.is_pin_memory_available() is False there unless VLLM_WSL2_ENABLE_PIN_MEMORY=1): every
           sleep then copies the weights into fresh pageable memory and every wake copies them back. Only the sleep backup
           changes (vllm.device_allocator.cumem.PIN_MEMORY); vLLM's other, small pinned buffers stay pinned.
  rl-disk-update
           SGLang only. The RL weight sync from disk as slime does it: the trainer saves a checkpoint, then every
           engine reloads ALL its weights from that folder in place (SGLang's update_weights_from_disk, the call
           behind slime's POST /update_weights_from_disk {"model_path", "flush_cache": true}). Untimed: start the
           engine, then a trainer stand-in writes two checkpoint folders A and B under --dir (see
           write_checkpoint). Timed: one discarded warm-up update, then --updates updates alternating A/B, each
           with its folder entirely in the page cache (the state right after a trainer's save). Every update
           must succeed and must change greedy generation as predicted, or the run fails (see head_swap_tokens).
           Metric: median seconds per update. SGLang copies each tensor from the memory-mapped safetensors
           (pageable) CPU->GPU; Qwen3.5-9B: 18.8 GB per update (an RTX 3090 run, 2026-11-09).
           One more, untimed update under SGLang's torch profiler (--h2d-bytes) splits an update's time into
           CPU->GPU copies and the rest (SGLang's per-tensor Python), and counts the CPU->GPU bytes, which
           differ from the bytes the files hold: SGLang skips some tensors (MTP layers; glm5_next.py load_weights
           skips layers >= num_hidden_layers) and keeps some on the host (qwen4_exp's PLE tables in pinned memory,
           a CPU->CPU copy).
vLLM runs with VLLM_ENABLE_V1_MULTIPROCESSING=0 (set in the catalog) so the engine and its copies live in the
launched process, which is the only process `cmt trace` records. SGLang has no such switch: its scheduler is a
child process, so a trace of SGLang shows no copies, while `cmt run` still reaches it through LD_PRELOAD.
"""
import argparse
import ctypes
import gzip
import json
import mmap
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time
from typing import NamedTuple

import appenv as c

MODEL = "Qwen/Qwen3.5-9B"
# Limits that fit MODEL on a 24 GB GPU next to its KV cache; engine defaults (262k context) do not
# (measured on the 3090: vLLM ready in 128 s, SGLang in 46 s). Text only: the vision encoder gets no
# inputs. The same limits on every GPU, so the work per run does not depend on its memory.
VLLM_LIMITS = {"max_model_len": 8192, "max_num_seqs": 16, "limit_mm_per_prompt": {"image": 0, "video": 0}}
SGLANG_LIMITS = {"context_length": 8192, "max_running_requests": 4, "mem_fraction_static": 0.9,
                 "cuda_graph_config": {"decode": {"max_bs": 4}, "prefill": {"max_bs": 4}}}
START = time.perf_counter()  # before importing the engine: import time is part of startup


def use_caches(engine, caches):
    """Point every compile cache of `engine` at one directory (see --caches); return it and its size now."""
    base = c.ROOT / "engine-cache"
    if caches == "warm":
        directory = base / engine
        directory.mkdir(parents=True, exist_ok=True)
    else:
        base.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f"cold-{engine}-", dir=base))
    for name, sub in [("VLLM_CACHE_ROOT", "vllm"), ("SGLANG_CACHE_DIR", "sglang"), ("TRITON_CACHE_DIR", "triton"),
                      ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("FLASHINFER_WORKSPACE_BASE", "flashinfer")]:
        os.environ[name] = str(directory / sub)  # before the engine is imported: some read them at import
    size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return directory, size


def startup(args):
    directory, cache_bytes = use_caches(args.engine, args.caches)
    try:
        if args.engine == "vllm":
            from vllm import LLM, SamplingParams
            from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
            # Time the loader's own weight load, the span its "Loading weights took" line reports. (A logging
            # handler for that line saw nothing in the 3090 smoke runs, on the "vllm" logger or the module's.)
            weight_seconds = []
            load_weights = DefaultModelLoader.load_weights

            def timed_load_weights(self, *args, **kwargs):
                begin = time.perf_counter()
                try:
                    return load_weights(self, *args, **kwargs)
                finally:
                    weight_seconds.append(time.perf_counter() - begin)

            DefaultModelLoader.load_weights = timed_load_weights
            engine = LLM(model=args.model, seed=0, **VLLM_LIMITS)
            ready = time.perf_counter() - START
            text = engine.generate(["The PCIe bus"], SamplingParams(max_tokens=1, temperature=0),
                                   use_tqdm=False)[0].outputs[0].text
            weights = sum(weight_seconds) if weight_seconds else None
        else:
            import sglang
            engine = sglang.Engine(model_path=args.model, random_seed=0, **SGLANG_LIMITS)
            ready = time.perf_counter() - START
            text = engine.generate("The PCIe bus", {"max_new_tokens": 1, "temperature": 0})["text"]
            weights = (getattr(engine.tokenizer_manager, "startup_time", None) or {}).get("load_weight")
        first = time.perf_counter() - START
        print(json.dumps({"metric": "first_token_seconds", "value": first, "unit": "s", "ready_seconds": ready,
                          "weight_load_seconds": weights, "caches": args.caches, "cache_bytes_at_start": cache_bytes,
                          "engine": args.engine, "model": args.model, "text": text}))
        if args.engine == "sglang":
            engine.shutdown()
    finally:
        if args.caches == "cold":
            shutil.rmtree(directory, ignore_errors=True)  # best effort: a leftover only costs disk


def sleep(args):
    use_caches("vllm", "warm")
    from vllm import LLM, SamplingParams
    if args.backup == "pageable":
        import vllm.device_allocator.cumem as cumem
        cumem.PIN_MEMORY = False  # read at each sleep (CuMemAllocator.sleep), so setting it here is enough
    engine = LLM(model=args.model, enable_sleep_mode=True, seed=0, **VLLM_LIMITS)
    params = SamplingParams(max_tokens=32, temperature=0)
    prompts = [f"Question {i}: why is the sky blue?" for i in range(16)]
    phases = []
    for cycle in range(args.cycles):
        engine.generate(prompts, params, use_tqdm=False)
        steps = [("sleep", lambda: engine.sleep(level=args.level))]
        if args.level == 1:
            steps.append(("wake", engine.wake_up))
        else:
            steps += [("wake_weights", lambda: engine.wake_up(tags=["weights"])),
                      ("reload_weights", lambda: engine.collective_rpc("reload_weights")),
                      ("wake_kv", lambda: engine.wake_up(tags=["kv_cache"]))]
        for name, step in steps:
            begin = time.perf_counter()
            step()
            phases.append({"cycle": cycle, "phase": name, "seconds": time.perf_counter() - begin})
    text = engine.generate(prompts[:1], params, use_tqdm=False)[0].outputs[0].text
    print(json.dumps({"metric": f"sleep{args.level}_cycle_seconds", "value": sum(p["seconds"] for p in phases),
                      "unit": "s", "cycles": args.cycles, "level": args.level, "backup": args.backup, "model": args.model,
                      "phases": phases, "text": text}))


# --- rl-disk-update ---------------------------------------------------------------------------------------------
#
# Verification (why a row swap in the output projection). SGLang 0.5.23 has no get_weights_by_name for Qwen3.5,
# Qwen3.8 (qwen4_exp) or GLM-5.3 (model_executor/model_runner_components/weight_exporter.py:169 calls the model's own
# method, which qwen3_5.py, qwen4_exp.py and glm5_next.py do not define), and a small change to some weight did not
# change greedy generation in an RTX 3090 run (2026-11-09). So A and B make a change whose effect on
# generation is known in advance. With h the final hidden state for PROMPT, the logits are h . W[t] for every token t
# (W = lm_head.weight). Exchanging rows t1 and t2 of W exchanges exactly the logits of t1 and t2 and nothing else: h
# does not depend on W (untied head), and the head has no bias or scale for the rows to disagree with (find_head
# refuses otherwise). With t1 the base model's greedy token and t2 any other token, the swapped model's greedy token
# is t2. A swaps (t1, t2) and B swaps (t1, t3), so base, A and B each give a different first token, and every update
# must switch it to its folder's token. The only way to be wrong is a near tie: the swapped row's logit is computed in
# another column of the GEMM and may round differently, so the base gap between the first and second candidates must
# be at least MIN_MARGIN. The change is 2 rows (16 KiB for the 9B) in one tensor; SGLang still reloads every tensor.
PROMPT = "The capital of France is"
# A guess with headroom, not a measurement: a bf16 logit near 20 has a rounding step of 0.125.
MIN_MARGIN = 1.0
HEAD_DTYPES = {"BF16": 2, "F16": 2, "F32": 4}  # safetensors dtype -> bytes; never FP8 (its scales would not swap)


class Head(NamedTuple):
    """Where the output projection's rows are: row r is `row_bytes` bytes at `start + r * row_bytes` of `shard`."""
    shard: Path
    name: str
    start: int
    rows: int
    row_bytes: int


def safetensors_header(path):
    """(file offset of the data section, header dict). The format: an 8-byte little-endian header length, that many
    bytes of JSON ({name: {dtype, shape, data_offsets: [begin, end)}}, offsets relative to the data section), data."""
    with open(path, "rb") as f:
        length = int.from_bytes(f.read(8), "little")
        size = os.fstat(f.fileno()).st_size
        if not 0 < length <= size - 8:
            raise SystemExit(f"{path}: not a safetensors file (header length {length}, file size {size})")
        return 8 + length, json.loads(f.read(length))


def find_head(snapshot):
    """The tensor the trainer stand-in changes. Refuses every case where swapping two of its rows would not swap
    exactly two logits (see the verification comment above)."""
    config = json.loads((snapshot / "config.json").read_text())
    for scope in (config, config.get("text_config") or {}):
        if scope.get("tie_word_embeddings"):
            raise SystemExit(f"{snapshot}: tied embeddings; swapping head rows would change the input too")
    heads, names = [], set()
    for shard in sorted(snapshot.glob("*.safetensors")):
        offset, header = safetensors_header(shard)
        names.update(header)
        heads += [(shard, name, header[name], offset) for name in header
                  if name.endswith("lm_head.weight") and "mtp" not in name]  # SGLang skips MTP weights
    if len(heads) != 1:
        raise SystemExit(f"{snapshot}: expected one lm_head.weight, found {[name for _, name, _, _ in heads]}")
    shard, name, entry, offset = heads[0]
    siblings = sorted(n for n in names if n.startswith(name[:-len("weight")]) and n != name)
    if siblings:
        raise SystemExit(f"{shard}: {name} comes with {siblings} (quantized or biased): "
                         f"a row swap would not swap logits")
    if entry["dtype"] not in HEAD_DTYPES or len(entry["shape"]) != 2:
        raise SystemExit(f"{shard}: {name} is {entry['dtype']} {entry['shape']}; "
                         f"need a 2-D {sorted(HEAD_DTYPES)} tensor")
    rows, width = entry["shape"]
    begin, end = entry["data_offsets"]
    row_bytes = width * HEAD_DTYPES[entry["dtype"]]
    if end - begin != rows * row_bytes or offset + end > shard.stat().st_size:
        raise SystemExit(f"{shard}: {name} spans bytes [{begin}, {end}), inconsistent with {entry['shape']}")
    return Head(shard, name, offset + begin, rows, row_bytes)


def write_checkpoint(snapshot, head, dest, swap):
    """Trainer stand-in: write checkpoint folder `dest`, which is `snapshot` with rows swap[0] and swap[1] of the head
    exchanged. The shard that holds the head is a new file: a copy, the two rows rewritten in place, fsync'ed so its
    writeback cannot compete with the timed updates. Every other weight shard is a symlink to the snapshot's file
    (HF snapshots are symlinks already), so A and B share them in the page cache: about 1x the model plus two
    shards instead of 2x the model (a 186 GB checkpoint twice would not fit a 251 GB container). The small files
    (config, index, tokenizer) are copies. Holds two rows in memory, never a shard. Returns the bytes written."""
    first, second = swap
    if first == second or not (0 <= first < head.rows and 0 <= second < head.rows):
        raise ValueError(f"rows {swap} of {head.name} ({head.rows} rows)")
    dest.mkdir()
    for entry in sorted(snapshot.iterdir()):
        if entry.name != head.shard.name and (entry.suffix == ".safetensors" or entry.is_dir()):
            (dest / entry.name).symlink_to(entry.resolve())
        else:
            shutil.copyfile(entry, dest / entry.name)  # follows the snapshot's symlink: a real file
    written = dest / head.shard.name
    source = os.open(head.shard, os.O_RDONLY)
    target = os.open(written, os.O_RDWR)
    try:
        rows = {row: os.pread(source, head.row_bytes, head.start + row * head.row_bytes) for row in swap}
        for row, other in ((first, second), (second, first)):
            if os.pwrite(target, rows[other], head.start + row * head.row_bytes) != head.row_bytes:
                raise SystemExit(f"{written}: short write of row {row}")
        os.fsync(target)
        for row, other in ((first, second), (second, first)):
            if os.pread(target, head.row_bytes, head.start + row * head.row_bytes) != rows[other]:
                raise SystemExit(f"{written}: row {row} reads back wrong")
    finally:
        os.close(source)
        os.close(target)
    return written.stat().st_size


_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
MAP_FAILED = ctypes.c_void_p(-1).value
_LOW_BIT = bytes(b & 1 for b in range(256))  # mincore sets bit 0 per page; the other bits are undefined


def resident_bytes(path):
    """(size, bytes of it in the page cache) by mincore(2) on a read-only mapping (maps, never reads the file)."""
    page = os.sysconf("SC_PAGE_SIZE")
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            return 0, 0
        address = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if address in (None, MAP_FAILED):
            raise OSError(ctypes.get_errno(), f"mmap {path}")
        try:
            pages = (ctypes.c_ubyte * ((size + page - 1) // page))()
            if _libc.mincore(address, size, pages) != 0:
                raise OSError(ctypes.get_errno(), f"mincore {path}")
            # translate/count run in C: 45M pages for a 186 GB checkpoint, checked twice per update.
            return size, min(size, bytes(pages).translate(_LOW_BIT).count(1) * page)
        finally:
            _libc.munmap(address, size)
    finally:
        os.close(fd)


def not_resident(paths):
    """Bytes of `paths` that are not in the page cache now."""
    return sum(size - resident for size, resident in map(resident_bytes, paths))


def major_faults(pids):
    """Major page faults (faults that waited for I/O) of these processes so far: field 12 of /proc/PID/stat."""
    total = 0
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            continue  # a helper process that has exited
        total += int(stat.rsplit(")", 1)[1].split()[9])
    return total


def memory_limits():
    """This process's cgroup (v2) memory limit, use and page cache (file) in bytes, as the kernel enforces them: in a
    container, /proc/meminfo shows the host, while page cache counts against the container's limit."""
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    path = next((line[3:] for line in lines if line.startswith("0::")), None)
    if path is None:
        return {"cgroup": None, "reason": "no cgroup v2 entry in /proc/self/cgroup"}
    root = Path("/sys/fs/cgroup")
    directory = root / path.lstrip("/")
    try:
        current = int((directory / "memory.current").read_text())
        stat = dict(line.split() for line in (directory / "memory.stat").read_text().splitlines())
    except FileNotFoundError as error:
        return {"cgroup": path, "reason": f"memory controller not visible: {error.filename}"}
    # The effective limit is the smallest memory.max of this group and its ancestors.
    limits = []
    for group in [directory, *directory.parents]:
        if (group / "memory.max").exists() and (value := (group / "memory.max").read_text().strip()) != "max":
            limits.append(int(value))
        if group == root:
            break
    return {"cgroup": path, "limit_bytes": min(limits, default=None), "current_bytes": current,
            "page_cache_bytes": int(stat["file"])}


def make_resident(paths):
    """Read every file that is not entirely in the page cache; return the bytes that were missing. Fails when a file
    still is not resident afterwards: the run would then time the disk, not the copies."""
    missing = 0
    for path in paths:
        size, resident = resident_bytes(path)
        if resident < size:
            missing += size - resident
            c.warm_page_cache(path)
            size, resident = resident_bytes(path)
            if resident < size:
                raise SystemExit(f"{path}: {(size - resident) >> 20} MiB not in the page cache right after reading "
                                 f"it; host memory is too small for a warm update")
    return missing


def first_token(engine, prompt):
    """Greedy first token for `prompt` and the top 3 candidates as (logprob, token id), from SGLang's generate."""
    out = engine.generate(prompt=prompt, sampling_params={"max_new_tokens": 1, "temperature": 0},
                          return_logprob=True, top_logprobs_num=3)
    top = [(float(entry[0]), int(entry[1])) for entry in out["meta_info"]["output_top_logprobs"][0]]
    return int(out["output_ids"][0]), top


def head_swap_tokens(engine, prompt):
    """(t1, t2, t3): the base model's greedy token and the next two candidates; refuses a near tie (MIN_MARGIN)."""
    token, top = first_token(engine, prompt)
    if len(top) < 3 or top[0][1] != token:
        raise SystemExit(f"greedy token {token} is not the first of the top logprobs {top}")
    if top[0][0] - top[1][0] < MIN_MARGIN:
        raise SystemExit(f"{prompt!r}: top-2 logprob gap {top[0][0] - top[1][0]:.3f} < {MIN_MARGIN}; a swapped row "
                         f"could lose to the runner-up by rounding. Pass another --prompt for this model.")
    return token, top[1][1], top[2][1]


def profiled_update(engine, update, directory):
    """Run `update` (one full reload) under SGLang's own profiler (Engine.start_profile: torch.profiler with CUDA
    activity, CUPTI in the scheduler process) and read the trace: the CPU->GPU bytes by kind ("Memcpy HtoD (Pageable
    -> Device)", ...), whatever Python path made the copies, and the split of the update's wall time into copies and
    the rest. copy_seconds is the union of [API call start, end of its transfer] over all CPU->GPU copies; the rest is
    SGLang's per-tensor Python, safetensors header parsing, post-processing and the cache flush. The hook copies from
    its own threads (union, not sum) and some of its calls may have no runtime-API record; its staged
    copies show as Pinned -> Device. CUPTI cost 0.5% of a 9B update on the 3090 (2.173 s profiled, 2.185 s not); still
    run after the timed updates, because CUPTI may not fully detach."""
    directory.mkdir()
    engine.start_profile(output_dir=str(directory), activities=["GPU"])
    try:
        result = update()
    finally:
        engine.stop_profile()
    traces = sorted(directory.rglob("*.trace.json.gz"))
    if len(traces) != 1:
        raise SystemExit(f"{directory}: expected one profiler trace, found {[str(t) for t in traces]}")
    with gzip.open(traces[0], "rt") as f:
        copies = h2d_copies(json.load(f)["traceEvents"])
    if not copies["h2d_bytes"]:
        raise SystemExit(f"{traces[0]}: no CPU->GPU copies recorded during a full weight update")
    return {"seconds": result["seconds"], "copy_seconds": copies["copy_seconds"],
            "other_seconds": result["seconds"] - copies["copy_seconds"], **copies}


def h2d_copies(events):
    """CPU->GPU copies in a torch.profiler (Kineto) trace: bytes by kind, count, copies whose API call is not in the
    trace (`copies_without_api_record`; their span starts at the transfer), and copy_seconds, the union of their spans
    [API call start, end of the transfer] (trace times are microseconds)."""
    calls = {e["args"]["correlation"]: e for e in events
             if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    kinds, spans, unmatched = {}, [], 0
    for event in events:
        if event.get("cat") != "gpu_memcpy" or "HtoD" not in event.get("name", ""):
            continue
        kinds[event["name"]] = kinds.get(event["name"], 0) + int(event["args"]["bytes"])
        call = calls.get(event["args"].get("correlation"))
        unmatched += call is None
        begin, end = event["ts"], event["ts"] + event["dur"]
        if call is not None:
            begin, end = min(begin, call["ts"]), max(end, call["ts"] + call["dur"])
        spans.append((begin, end))
    covered, reach = 0.0, None
    for begin, end in sorted(spans):
        if reach is None or begin > reach:
            covered, reach = covered + end - begin, end
        elif end > reach:
            covered, reach = covered + end - reach, end
    return {"h2d_bytes": sum(kinds.values()), "h2d_by_kind": kinds, "h2d_copies": len(spans),
            "copies_without_api_record": unmatched, "copy_seconds": covered / 1e6}


# Checkpoint folders live in WORK_PREFIX<pid>-XXXX under --dir. `finally` deletes them, but an interrupted run never
# gets there: the runner signals the whole process group, the SGLang scheduler dies, and SGLang's own handlers in
# this process (asyncio SIGTERM/SIGQUIT handlers installed at the first request) kill the process tree without
# unwinding (seen on the 3090 after SIGTERM to cmt-evaluate). So each run first deletes the folders of runs whose
# process no longer exists.
WORK_PREFIX = "cmt-rl-disk-update-"
# Major page faults of the scheduler processes allowed during a timed update. Warm 9B updates took 0 and cold ones
# 72k (one per 261 KB read) in an RTX 3090 run (2026-11-09), so 1000 (~260 MB from disk) leaves room
# for stray faults (a library page) and still fails an update that read its checkpoint. Residency before and after
# is not enough: a file system that drops its cache at open (FUSE without keep_cache) refills it during the update.
MAX_MAJOR_FAULTS = 1000


def reclaim_work_dirs(directory):
    """Delete WORK_PREFIX<pid>-* folders in `directory` whose process is gone; return what was deleted. A folder
    whose pid is alive (another run, or a reused pid) is left alone: a leftover costs disk, a wrong delete a run."""
    reclaimed = []
    for work in sorted(directory.glob(f"{WORK_PREFIX}*-*")):
        pid = work.name[len(WORK_PREFIX):].split("-", 1)[0]
        if not pid.isdigit() or not work.is_dir():
            continue
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            shutil.rmtree(work)
            reclaimed.append(str(work))
        except PermissionError:
            pass  # alive, another user's
    return reclaimed


def local_snapshot(model):
    """The model's directory: `model` itself, or its snapshot in the Hugging Face cache (never downloads)."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, local_files_only=True))


def rl_disk_update(args):
    use_caches("sglang", "warm")
    import sglang
    from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput
    # Engine.update_weights_from_disk (sglang/srt/entrypoints/engine.py:1702) takes model_path and load_format only;
    # flush_cache comes from the request's default (managers/io_struct.py:1990), which slime sets explicitly
    # (env/rl-slime/src/slime/backends/sglang_utils/sglang_engine.py:416). Refuse if a new SGLang changes it.
    if UpdateWeightFromDiskReqInput(model_path="").flush_cache is not True:
        raise SystemExit(f"SGLang {sglang.__version__}: update_weights_from_disk no longer flushes the cache "
                         f"by default")
    snapshot = local_snapshot(args.model)
    head = find_head(snapshot)
    tensors = sum(len([n for n in safetensors_header(f)[1] if n != "__metadata__"])
                  for f in snapshot.glob("*.safetensors"))
    memory_at_start = memory_limits()
    reclaimed = reclaim_work_dirs(Path(args.dir))
    engine = sglang.Engine(model_path=str(snapshot), random_seed=0, **{**SGLANG_LIMITS, **args.engine_args})
    work = None
    try:
        base, *others = head_swap_tokens(engine, args.prompt)
        expected = dict(zip("AB", others))
        work = Path(tempfile.mkdtemp(prefix=f"{WORK_PREFIX}{os.getpid()}-", dir=args.dir))
        folders = {tag: work / tag for tag in "AB"}
        begin = time.perf_counter()
        written = {tag: write_checkpoint(snapshot, head, folders[tag], (base, expected[tag])) for tag in "AB"}
        write_seconds = time.perf_counter() - begin
        # Real files behind each folder: the shared shards of the snapshot plus the folder's own head shard.
        shards = {tag: [p.resolve() for p in sorted(folders[tag].glob("*.safetensors"))] for tag in "AB"}
        checkpoint_bytes = sum(p.stat().st_size for p in shards["A"])
        make_resident(sorted(set(shards["A"] + shards["B"])))
        scheduler = engine.get_all_child_pids()

        def update(tag, timed):
            """One full reload from folder `tag`, timed and then verified. Warm means: every byte of the folder is
            in the page cache before the update (else it is read first, untimed), still is after it, and the
            scheduler took at most MAX_MAJOR_FAULTS major faults (checked for timed updates only: the warm-up may
            page in code). Otherwise the run fails instead of reporting a disk time as warm."""
            missing = make_resident(shards[tag])
            faults = major_faults(scheduler)
            begin = time.perf_counter()
            success, message, _ = engine.update_weights_from_disk(model_path=str(folders[tag]))
            seconds = time.perf_counter() - begin
            faults = major_faults(scheduler) - faults
            if not success:
                raise SystemExit(f"update from {folders[tag]} failed: {message}")
            if timed and faults > MAX_MAJOR_FAULTS:
                raise SystemExit(f"the update from {folders[tag]} took {faults} major page faults: it read from disk, "
                                 f"so it was not warm (is the file system dropping its page cache at open?)")
            evicted = not_resident(shards[tag])
            if evicted:
                raise SystemExit(f"{evicted >> 20} MiB of {folders[tag]} left the page cache during the update: the "
                                 f"update was not warm (cgroup memory: {memory_limits()})")
            token, top = first_token(engine, args.prompt)
            if token != expected[tag]:
                raise SystemExit(f"after the update from {folders[tag]} the greedy token is {token}, expected "
                                 f"{expected[tag]} (base {base}): the update did not take effect")
            return {"folder": tag, "seconds": seconds, "missing_bytes_read_before": missing,
                    "scheduler_major_faults": faults, "top": top}

        tags = ["A" if i % 2 == 0 else "B" for i in range(args.updates + 2)]
        warmup = update(tags[0], timed=False)
        timed = [update(tag, timed=True) for tag in tags[1:-1]]
        memory_at_end = memory_limits()
        profiled = (profiled_update(engine, lambda: update(tags[-1], timed=False), work / "profile")
                    if args.h2d_bytes == "profile" else None)
    finally:
        if work is not None:
            try:
                shutil.rmtree(work)
            except OSError as error:  # the run's own result stands; say loudly what is left on disk
                print(f"engines.py: could not delete {work}: {error}; delete it by hand", file=sys.stderr)
        engine.shutdown()
    seconds = [u["seconds"] for u in timed]
    median = statistics.median(seconds)
    print(json.dumps({"metric": "rl_disk_update_seconds", "value": median, "unit": "s", "seconds": seconds,
                      "warmup_seconds": warmup["seconds"], "updates": [warmup] + timed,
                      "checkpoint_bytes": checkpoint_bytes, "checkpoint_gb_per_s": checkpoint_bytes / median / 1e9,
                      "checkpoint_tensors": tensors,
                      "h2d_bytes": None if profiled is None else profiled["h2d_bytes"], "profiled_update": profiled,
                      "stand_in": {"written_bytes": written, "write_seconds": write_seconds, "tensor": head.name,
                                   "shard": head.shard.name},
                      "verification": {"prompt": args.prompt, "base_token": base, "expected_tokens": expected},
                      "memory": {"start": memory_at_start, "end_of_timed_updates": memory_at_end},
                      "model": args.model, "snapshot": str(snapshot), "engine_args": args.engine_args,
                      "reclaimed_work_dirs": reclaimed,
                      "sglang": sglang.__version__}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--op", choices=["startup", "sleep", "rl-disk-update"], required=True)
    parser.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--model", default=MODEL, help=f"default {MODEL}; the limits above assume it")
    parser.add_argument("--level", type=int, choices=[1, 2], default=1, help="sleep level (default 1)")
    parser.add_argument("--cycles", type=int, default=3, help="sleep/wake cycles (default 3)")
    parser.add_argument("--caches", choices=["warm", "cold"], default="warm",
                        help="startup: compile caches kept between runs (warm, default) or empty per run (cold)")
    parser.add_argument("--backup", choices=["pinned", "pageable"], default="pinned",
                        help="level-1 weight backup memory (default pinned, vLLM's default outside WSL2)")
    # 5 timed updates per process: warm updates varied by +-4% in an RTX 3090 run (2026-11-09, 20 per
    # arm), and the runner repeats the process (3 x 5 per arm); the warm-up absorbs first-update costs.
    parser.add_argument("--updates", type=int, default=5,
                        help="rl-disk-update: timed updates after the discarded warm-up (default 5)")
    parser.add_argument("--dir", default=tempfile.gettempdir(),
                        help="rl-disk-update: where checkpoint folders A and B go, deleted at exit (the catalog passes "
                             "$CMT_ENV_ROOT; needs 2 x the shard that holds lm_head)")
    parser.add_argument("--h2d-bytes", choices=["profile", "off"], default="profile",
                        help="rl-disk-update: profile one extra, untimed update with SGLang's profiler for its "
                             "CPU->GPU bytes and copy vs other time (default), or skip that (off)")
    parser.add_argument("--prompt", default=PROMPT,
                        help=f"rl-disk-update: prompt whose greedy first token verifies each update "
                             f"(default {PROMPT!r})")
    parser.add_argument("--engine-args", type=json.loads, default={}, metavar="JSON",
                        help="rl-disk-update: sglang.Engine keyword arguments over the limits above, e.g. a "
                             "quantized model's flags")
    args = parser.parse_args()
    if args.op == "sleep" and args.engine != "vllm":
        parser.error("--op sleep is vLLM only")
    if args.op == "rl-disk-update" and args.engine != "sglang":
        parser.error("--op rl-disk-update is SGLang only (pass --engine sglang)")
    if args.op != "rl-disk-update" and args.engine_args:
        parser.error("--engine-args applies to --op rl-disk-update only")
    if args.updates < 1:
        parser.error("--updates must be at least 1")
    if args.caches == "cold" and args.op != "startup":
        parser.error("--caches cold applies to --op startup only")
    if args.backup == "pageable" and (args.op, args.level) != ("sleep", 1):
        parser.error("--backup applies to --op sleep --level 1 only")
    # FlashInfer's and SGLang's JIT run `ninja` by name; setup.sh installs it next to this python.
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    {"startup": startup, "sleep": sleep, "rl-disk-update": rl_disk_update}[args.op](args)


if __name__ == "__main__":
    main()
