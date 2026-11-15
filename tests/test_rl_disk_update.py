"""The untimed parts of `engines.py --op rl-disk-update` (sglang-rl-disk-update), without a GPU or SGLang.

The workload's numbers are only as good as what it does before and around the timer. The tests check:
- the trainer stand-in's checkpoint folder: only the head's shard is a new file, every other weight shard is a
  symlink to the snapshot's own file, the small files are copies, the snapshot is untouched, and the new shard
  differs from the original in exactly the two swapped rows;
- the property the verification relies on: with the written head, the argmax of the logits h . W[t] moves from the
  base token t1 to t2, for a head read back from the written file;
- every refusal of find_head (tied, quantized, biased, missing or ambiguous head) and of the token choice (near tie);
- the copy/other split of a profiled update: CPU->GPU bytes by kind, and copy time as the union of call-to-transfer
  spans (overlapping copies from several threads count once);
- the major-fault count the warm check uses (field 12 of /proc/PID/stat);
- that a run deletes only the checkpoint folders of runs whose process is gone;
- page-cache residency as mincore(2) reports it: a file read is resident, the same file after fsync +
  POSIX_FADV_DONTNEED is not (skipped on tmpfs and ramfs, which keep the pages).
"""
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ROOT / "evaluation" / "workloads"
sys.path.insert(0, str(WORKLOADS))  # engines.py imports appenv from its own directory
spec = importlib.util.spec_from_file_location("engines", WORKLOADS / "engines.py")
engines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engines)


def safetensors(path, tensors):
    """Write {name: (dtype, shape, bytes)} in the safetensors format (no third-party code)."""
    header, data = {}, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    text = json.dumps(header).encode()
    path.write_bytes(len(text).to_bytes(8, "little") + text + data)


def f32(values):
    return struct.pack(f"<{len(values)}f", *values)


ROWS, WIDTH = 6, 4
# Row t of the head is the logit weight of token t; with H below, the base logits are 0, 1, 2, ..., 5 + 10 for
# token 4, so the greedy token is 4 with a gap of 10 over token 5.
HEAD = [[float(t)] + [0.0] * (WIDTH - 1) for t in range(ROWS)]
HEAD[4][0] = 15.0
H = [1.0, 0.5, -2.0, 3.0]


def logits(rows):
    return [sum(w * h for w, h in zip(row, H)) for row in rows]


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


class Snapshot:
    """A Hugging Face style snapshot: files are symlinks into blobs/, as the HF cache stores them."""

    def __init__(self, root, head=None, config=None, extra=None):
        self.blobs = root / "blobs"
        self.dir = root / "snapshot"
        self.blobs.mkdir()
        self.dir.mkdir()
        head = head if head is not None else {"lm_head.weight": ("F32", [ROWS, WIDTH], f32(sum(HEAD, [])))}
        self.add("model-00001-of-00002.safetensors", None,
                 {"model.embed.weight": ("BF16", [2, 2], bytes(range(8))), **head})
        self.add("model-00002-of-00002.safetensors", None,
                 {"model.norm.weight": ("BF16", [4], bytes(range(8, 16))), **(extra or {})})
        self.add("config.json", json.dumps(config or {"architectures": ["Test"]}).encode())
        self.add("tokenizer.json", b'{"tokens": 1}')

    def add(self, name, raw, tensors=None):
        blob = self.blobs / name
        if tensors is not None:
            safetensors(blob, tensors)
        else:
            blob.write_bytes(raw)
        (self.dir / name).symlink_to(os.path.relpath(blob, self.dir))


def read_head(path):
    offset, header = engines.safetensors_header(path)
    begin, end = header["lm_head.weight"]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(offset + begin)
        values = struct.unpack(f"<{ROWS * WIDTH}f", f.read(end - begin))
    return [list(values[r * WIDTH:(r + 1) * WIDTH]) for r in range(ROWS)]


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.snapshot = Snapshot(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_folder_shares_unchanged_shards_and_rewrites_only_the_head_shard(self):
        head = engines.find_head(self.snapshot.dir)
        self.assertEqual(head.shard.name, "model-00001-of-00002.safetensors")
        before = {p.name: p.read_bytes() for p in self.snapshot.blobs.iterdir()}
        dest = self.root / "A"
        written = engines.write_checkpoint(self.snapshot.dir, head, dest, (4, 1))

        self.assertEqual(sorted(p.name for p in dest.iterdir()), sorted(p.name for p in self.snapshot.dir.iterdir()))
        shared = dest / "model-00002-of-00002.safetensors"
        self.assertTrue(shared.is_symlink())
        self.assertEqual(shared.resolve(), (self.snapshot.blobs / shared.name).resolve())
        for small in ("config.json", "tokenizer.json"):
            self.assertFalse((dest / small).is_symlink(), small)
            self.assertEqual((dest / small).read_bytes(), before[small])
        new = dest / head.shard.name
        self.assertFalse(new.is_symlink())
        self.assertEqual(written, new.stat().st_size)
        self.assertEqual({p.name: p.read_bytes() for p in self.snapshot.blobs.iterdir()}, before, "snapshot changed")

        # Byte for byte: the original with rows 4 and 1 exchanged, nothing else.
        original = bytearray(before[head.shard.name])
        expected = bytearray(original)
        row = lambda r: slice(head.start + r * head.row_bytes, head.start + (r + 1) * head.row_bytes)  # noqa: E731
        expected[row(4)], expected[row(1)] = original[row(1)], original[row(4)]
        self.assertEqual(new.read_bytes(), bytes(expected))

    def test_written_head_moves_the_greedy_token_to_the_swapped_row(self):
        head = engines.find_head(self.snapshot.dir)
        base = argmax(logits(HEAD))
        self.assertEqual(base, 4)
        for tag, other in (("A", 5), ("B", 2)):
            engines.write_checkpoint(self.snapshot.dir, head, self.root / tag, (base, other))
            swapped = logits(read_head(self.root / tag / head.shard.name))
            self.assertEqual(argmax(swapped), other, tag)
            # Exactly two logits exchanged, all others equal: a permutation, not some other change.
            unchanged = [t for t in range(ROWS) if t not in (base, other)]
            self.assertEqual([swapped[t] for t in unchanged], [logits(HEAD)[t] for t in unchanged])

    def test_refuses_rows_that_are_not_a_swap(self):
        head = engines.find_head(self.snapshot.dir)
        for rows in ((1, 1), (0, ROWS), (-1, 2)):
            with self.assertRaises(ValueError, msg=rows):
                engines.write_checkpoint(self.snapshot.dir, head, self.root / f"bad{rows}", rows)


class FindHeadTest(unittest.TestCase):
    def refused(self, **snapshot):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                engines.find_head(Snapshot(Path(tmp), **snapshot).dir)
        return str(caught.exception)

    def test_refusals(self):
        raw = f32(sum(HEAD, []))
        self.assertIn("tied", self.refused(config={"tie_word_embeddings": True}))
        self.assertIn("tied", self.refused(config={"text_config": {"tie_word_embeddings": True}}))
        self.assertIn("F8_E4M3", self.refused(head={"lm_head.weight": ("F8_E4M3", [ROWS, WIDTH * 4], raw)}))
        self.assertIn("weight_scale", self.refused(head={"lm_head.weight": ("F32", [ROWS, WIDTH], raw),
                                                         "lm_head.weight_scale": ("F32", [1], f32([1.0]))}))
        self.assertIn("lm_head.bias", self.refused(head={"lm_head.weight": ("F32", [ROWS, WIDTH], raw),
                                                         "lm_head.bias": ("F32", [ROWS], f32([0.0] * ROWS))}))
        self.assertIn("expected one", self.refused(head={}))
        self.assertIn("expected one", self.refused(
            extra={"model.language_model.lm_head.weight": ("F32", [ROWS, WIDTH], raw)}))
        self.assertIn("inconsistent", self.refused(head={"lm_head.weight": ("F32", [ROWS, WIDTH + 1], raw)}))

    def test_mtp_head_is_not_the_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = f32(sum(HEAD, []))
            snapshot = Snapshot(Path(tmp), extra={"mtp.lm_head.weight": ("F32", [ROWS, WIDTH], raw)})
            self.assertEqual(engines.find_head(snapshot.dir).name, "lm_head.weight")


class FakeEngine:
    def __init__(self, top):
        self.top = top

    def generate(self, prompt, sampling_params, return_logprob, top_logprobs_num):
        assert sampling_params == {"max_new_tokens": 1, "temperature": 0} and return_logprob
        return {"output_ids": [self.top[0][1]],
                "meta_info": {"output_top_logprobs": [[(lp, t, None) for lp, t in self.top[:top_logprobs_num]]]}}


class TokenChoiceTest(unittest.TestCase):
    def test_takes_the_next_two_candidates(self):
        engine = FakeEngine([(-0.1, 7), (-3.0, 9), (-4.0, 2)])
        self.assertEqual(engines.head_swap_tokens(engine, "p"), (7, 9, 2))

    def test_refuses_a_near_tie(self):
        gap = engines.MIN_MARGIN / 2
        with self.assertRaises(SystemExit):
            engines.head_swap_tokens(FakeEngine([(-1.0, 7), (-1.0 - gap, 9), (-5.0, 2)]), "p")


def memcpy(ts, dur, nbytes, correlation, kind="Memcpy HtoD (Pageable -> Device)"):
    return {"cat": "gpu_memcpy", "name": kind, "ts": ts, "dur": dur,
            "args": {"bytes": nbytes, "correlation": correlation}}


def call(ts, dur, correlation, name="cudaMemcpyAsync"):
    return {"cat": "cuda_runtime", "name": name, "ts": ts, "dur": dur, "args": {"correlation": correlation}}


class CopySplitTest(unittest.TestCase):
    def test_union_of_call_to_transfer_end_spans(self):
        events = [
            call(0, 10, 1), memcpy(2, 20, 100, 1),                 # span [0, 22]
            call(15, 2, 2), memcpy(18, 30, 50, 2),                 # [15, 48], overlaps the first: union [0, 48]
            call(100, 5, 3), memcpy(101, 9, 7, 3, "Memcpy HtoD (Pinned -> Device)"),  # [100, 110]
            memcpy(200, 4, 1, 99),                                 # no API record: the transfer only, [200, 204]
            call(300, 50, 4), memcpy(305, 10, 10**6, 4, "Memcpy DtoH (Device -> Pageable)"),  # not CPU->GPU
            call(400, 1, 5, "cudaLaunchKernel"),
        ]
        split = engines.h2d_copies(events)
        self.assertEqual(split["h2d_by_kind"], {"Memcpy HtoD (Pageable -> Device)": 151,
                                                "Memcpy HtoD (Pinned -> Device)": 7})
        self.assertEqual((split["h2d_bytes"], split["h2d_copies"], split["copies_without_api_record"]), (158, 4, 1))
        self.assertAlmostEqual(split["copy_seconds"], (48 + 10 + 4) / 1e6)

    def test_api_call_longer_than_transfer(self):
        # A pageable copy's call returns after its transfer ends (staging): the span is the call.
        split = engines.h2d_copies([call(0, 100, 1), memcpy(10, 20, 8, 1)])
        self.assertAlmostEqual(split["copy_seconds"], 100 / 1e6)


class ReclaimTest(unittest.TestCase):
    def test_deletes_only_folders_of_dead_runs(self):
        import subprocess
        dead = subprocess.Popen(["true"])
        dead.wait()  # reaped: its pid names no process (unless reused within this test, which leaves it alone)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mine = root / f"{engines.WORK_PREFIX}{os.getpid()}-abc"
            gone = root / f"{engines.WORK_PREFIX}{dead.pid}-def"
            other = root / "cmt-rl-disk-update-notapid-x"
            for folder in (mine, gone, other):
                (folder / "A").mkdir(parents=True)
                (folder / "A" / "shard.safetensors").write_bytes(b"x")
            self.assertEqual(engines.reclaim_work_dirs(root), [str(gone)])
            self.assertEqual(sorted(p.name for p in root.iterdir()), sorted([mine.name, other.name]))


class MajorFaultTest(unittest.TestCase):
    def test_reads_the_kernels_major_fault_count(self):
        import mmap
        import resource
        with tempfile.TemporaryDirectory() as tmp:
            if fs_type(tmp) in ("tmpfs", "ramfs"):
                self.skipTest(f"{fs_type(tmp)} at {tmp}: no major faults to provoke")
            # Provoke major faults (so a zero from a wrong field cannot pass): touch an evicted file through mmap.
            path = Path(tmp) / "file"
            path.write_bytes(os.urandom(8 << 20))
            with open(path, "rb") as f:
                os.fsync(f.fileno())
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as mapped:
                    sum(mapped[i] for i in range(0, len(mapped), 4096))
        self.assertGreater(resource.getrusage(resource.RUSAGE_SELF).ru_majflt, 0)
        # Field 12 of /proc/PID/stat is what getrusage reports as ru_majflt for the whole process.
        for _ in range(3):  # a fault between the two reads would differ; retry rather than flake
            if engines.major_faults([os.getpid()]) == resource.getrusage(resource.RUSAGE_SELF).ru_majflt:
                break
        else:
            self.fail("major_faults disagrees with getrusage")
        self.assertEqual(engines.major_faults([os.getpid(), 2 ** 22 + 1]), engines.major_faults([os.getpid()]),
                         "a pid that does not exist counts zero")


def fs_type(path):
    """File-system type of the mount that holds `path` (longest matching mount point in /proc/mounts)."""
    best, kind = "", None
    for line in Path("/proc/mounts").read_text().splitlines():
        _, mount, fs, *_ = line.split()
        if (str(path) + "/").startswith(mount.rstrip("/") + "/") and len(mount) > len(best):
            best, kind = mount, fs
    return kind


class ResidencyTest(unittest.TestCase):
    def test_mincore_sees_reads_and_evictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Decided without the code under test, so a broken mincore cannot turn this into a skip.
            if fs_type(tmp) in ("tmpfs", "ramfs"):
                self.skipTest(f"{fs_type(tmp)} at {tmp} keeps pages after POSIX_FADV_DONTNEED")
            path = Path(tmp) / "file"
            path.write_bytes(os.urandom(8 << 20))
            engines.c.warm_page_cache(path)
            self.assertEqual(engines.resident_bytes(path), (8 << 20, 8 << 20))
            self.assertEqual(engines.make_resident([path]), 0)
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            missing = engines.not_resident([path])
            self.assertGreater(missing, 0)
            self.assertEqual(engines.make_resident([path]), missing)
            self.assertEqual(engines.not_resident([path]), 0)


if __name__ == "__main__":
    unittest.main()
