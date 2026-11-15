---
name: Tune request
about: Ask for a shared tune for hardware you use but cannot tune yourself
labels: tune request
---

Shared tunes are the ones others contributed under `tunes/`. If you can run `cmt tune --search quick` on the
machine, please do, and contribute the result instead (see tunes/README.md).

**Machine** (what `cmt run --remote-config` sends)

- GPU name, as `nvidia-smi --query-gpu=name --format=csv,noheader` prints it:
- CPU model, as in `/proc/cpuinfo` "model name":
- NVIDIA driver version:
- CPU architecture (x86_64; aarch64 builds from source but is untested on hardware):

**What `cmt run --remote-config -- true` printed** (the tune it chose, or none):

```
```

**What you copy** (sizes and directions, if you know them; `cmt trace` records them):
