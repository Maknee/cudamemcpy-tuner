---
name: Bug report
about: cmt fails, is slower than the stock driver, or gives a wrong result
labels: bug
---

**What happened, and what you expected**

**The command and its full output** (use `cmt tune --debug -- APP` for hook diagnostics):

```
```

**`cmt bench --info`**

```
```

**Versions**

- `cmt`: what `cmt --version` prints
- NVIDIA driver (`cat /proc/driver/nvidia/version`; delete the `(user@host)` part of the first line, it names
  whoever built the module):
- CUDA toolkit `cmt` was built with (`nvcc --version`):
- OS and glibc (`cat /etc/os-release`, `ldd --version | head -1`):
- Application and framework (PyTorch, vLLM, ...), with versions:

If a tune is involved, attach the file from `cmt export` (it holds nothing identifying), not the raw `tune.json`.
