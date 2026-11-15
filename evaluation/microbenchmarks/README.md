# Component microbenchmarks

Use **Google Benchmark** for timing. Keep byte checks and resource checks outside
the timed loop. The existing CTest tests check correctness and failure handling;
they do not fail because one implementation is slower on a particular machine.

| Executable | Question | What is timed? |
| --- | --- | --- |
| `handoff` | Does splitting a CPU copy help? Does spinning hurt? | One-thread `memcpy`, or the actual CPU pool's copy, handoff and completion. |
| `chunk_pipeline_bench` | Do smaller chunks or more workers help at the same memory cost? | The actual hook `ChunkPipeline`, including CPU copying and completed DMA. |
| `patch_decode` | How much does patch preparation cost? | Version lookup or instruction decoding and trampoline planning. No patch is installed. |

The main CMake file defines these targets. There is no additional build system
in this directory. Application campaigns, large CPU/DMA sweeps, metrics and
plotting remain in `evaluation/`.

## Build and check

`CMT_BUILD_BENCHMARKS` is on by default; the commands below set it explicitly. These are example commands;
use the project's installed CUDA toolkit and dependency overrides if needed.

```bash
cmake -S . -B build-components -DCMAKE_BUILD_TYPE=Release -DCMT_BUILD_BENCHMARKS=ON
cmake --build build-components --parallel 2 \
  --target handoff chunk_pipeline_bench patch_decode
build-components/handoff --benchmark_list_tests=true
build-components/chunk_pipeline_bench --benchmark_list_tests=true
```

Run them on an otherwise idle machine, with `LD_PRELOAD` unset.

```bash
mkdir -p results/components
env -u LD_PRELOAD build-components/handoff \
  --benchmark_filter='^(serial|condvar|mpmc)/bytes:16777216/.*busy:0/spin_us:0' \
  --benchmark_repetitions=5 --benchmark_min_time=0.2s \
  --benchmark_out=results/components/handoff.json --benchmark_out_format=json

env -u LD_PRELOAD build-components/chunk_pipeline_bench \
  --benchmark_filter='^(pipeline/reused|pinned_dma/reused|pageable_driver)/.*bytes:16777216/' \
  --benchmark_repetitions=5 --benchmark_min_time=0.2s \
  --benchmark_out=results/components/pipeline-16MiB.json --benchmark_out_format=json
```

Use `bytes_per_second / 1e9` for decimal GB/s. Size arguments are bytes; `MiB`
means 1,048,576 bytes. Repeat whole processes as well as iterations before
choosing defaults. These commands alone do not randomize run order.

## Pipeline comparison

Each layout allocates **32 MiB of CPU-cached pinned staging memory**:

| Workers | Buffers per worker | Bytes per buffer | Purpose |
| ---: | ---: | ---: | --- |
| 1 | 2 | 16 MiB | Large-chunk control |
| 1 | 4 | 8 MiB | Smaller chunks, more buffers, same CPU worker |
| 1 | 8 | 4 MiB | More opportunities to overlap CPU work and DMA |
| 2 | 2 | 8 MiB | Two CPU workers, same total buffer budget |
| 4 | 2 | 4 MiB | Four CPU workers, same total buffer budget |

- A 16 MiB transfer with 16 MiB chunks has one chunk. Extra workers cannot help it.
- Two 8 MiB buffers do not imply two simultaneous DMA transfers or twice the bandwidth.
- The table selects a row by the full CUDA transfer size. CPU loops operate on
  chunks, which can be smaller. Select a loop from a measured complete pipeline
  configuration, not just the standalone CPU winner at the full transfer size.
- `actual_workers` and `max_slots_used_per_worker` record what the verified warm-up used.
- `pipeline/reused` allocates/pins memory and starts workers before timing. It
  reuses them across copies. Each timed copy waits for the caller stream.
- `pipeline/lifecycle` includes pool allocation/pinning, thread startup, a
  completed copy, worker shutdown and pool release in **each** timed iteration.
  Context creation and application host/device allocation remain outside timing.
- `pinned_dma/reused` times a completed CUDA copy from/to preallocated pinned
  memory. Its allocation and release are excluded.
- `pageable_driver` times a completed CUDA copy with ordinary CPU memory and no
  hook. It is a component control; the application baseline remains the **bare
  application without the hook**.
- Both directions run. No registration-cache policy, WC staging, pacing,
  internal driver patch, or automatic CPU placement is enabled here.
- The focused sizes are 4 KiB, 1 MiB, 16 MiB, 64 MiB, 256 MiB and 1 GiB.
  These do not replace the wider size sweep in `evaluation/`.

The benchmark uses two changed source patterns before timing, checks every byte
and guard region, and checks the final result after timing. It checks driver
errors during timing and resource-release errors afterward. It does not inspect
every timed iteration's bytes. A failed check sets an error and a nonzero exit;
never plot that row as a successful result.

## CPU pool comparison

- `workers` counts helper threads. The caller also copies, so `copy_threads = workers + 1`.
- All participants use `memcpy`. This isolates scheduling from ISA selection.
- Pool construction, buffer allocation and final release are excluded.
- Both protocols park through Abseil. `abseil_mutex` is always 1.
  Nonzero `spin_us` is unsupported.
- `mpmc` is an existing protocol name. It uses a lock around shared job storage;
  it is **not a lock-free algorithm**.
- `busy:1` adds a CPU competitor. It does not bind threads to the same CPU.
  Use a permitted CPU with `taskset -c CPU` around the whole process to test
  a caller and workers sharing one CPU. Record that affinity with the result.
- Repeated copies reuse touched buffers. None of these benchmarks drops CPU
  caches, drops file caches, or claims to measure first-touch page faults.

## Changes that need evidence

- Keep the ring lifetime lock: it prevents teardown while a copy uses the pool.
- Keep job serialization: a pipeline has one shared job and buffer pool.
- An opaque pointer names a buffer; it does not grant exclusive ownership or
  detect a stale reference. A future slot handle needs a generation and a clear
  borrow/release rule before the locks can change.
- The CPU-only split pool is optional. Do not nest it inside `ChunkPipeline`
  without a separate measurement that shows a benefit.
- Direction overrides apply to both the caller and helpers in the CPU pool.
  Its API accepts one loop per job. The next job may select a different loop
  without rebuilding the workers.
