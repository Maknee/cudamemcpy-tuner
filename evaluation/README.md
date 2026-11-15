# Evaluation

`cmt-evaluate` runs native benchmarks and application comparisons, saves resumable
results and produces an HTML report. Linux x86_64 with an NVIDIA GPU is tested;
aarch64 has not been validated on hardware.

## Build and commands

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMT_BUILD_EVALUATION=ON
cmake --build build --parallel
build/cmt-evaluate tune quick
build/cmt-evaluate bench references
build/cmt-evaluate workload list
build/cmt-evaluate workload run --group personal
build/cmt-evaluate report
```

Building requires Rust 1.89+ and Python 3. Reports also need matplotlib; `report --python PATH`
selects that interpreter. `--bin-dir` or `CMT_BIN_DIR` selects the directory holding
`cmt`, `gpu_copy` and `cpu_loops`, defaulting to the runner's own directory.
`cmake --install build --prefix DIR` installs all companions, `evaluation/` resources
and license files together. Keep the bundle intact when copying it.

Use `build/cmt-evaluate --help` or a command’s `--help` for options.
