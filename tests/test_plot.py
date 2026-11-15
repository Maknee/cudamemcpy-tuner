"""evaluation/plot.py on a tiny fixture run, without a GPU.

The fixture (written by `fixture_run`) holds what `cmt-evaluate bench kernel` and `workload run`
save: bench/kernel/{stock,patched} items (the stock progressive child refused, as on stock modules)
and two workload records. The tests check:
- one drawn point and one table cell against values recomputed here from the fixture's own numbers;
- the workload verdict rule on crafted cases, both `better` directions, and that the table uses it;
- the single-run page against tests/fixtures/plot_report.html, so any change to it is deliberate
  (CMT_UPDATE_GOLDEN=1 rewrites the file; review the diff);
- the several-runs page: one figure per run in each section, each the same image as drawn alone;
- the tuning rows rebuilt from tune.json (`selected_rows`) with and without a registration limit
  (an older request's 1 GiB default is none), above a limit, and when a recorded line 1 is used, on
  the cases tests/test_selection.cpp checks in BuildCopyLookup;
- the copy-size table under the trace figure: one row per trace (identical distributions too),
  only the copies the figure counts, its order, cells and drawn flags, each trace read once.

Exit 77 (skipped, never passed) without matplotlib.
"""
import html
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import matplotlib  # noqa: F401
except ImportError:
    print("matplotlib is not installed; skipped")
    sys.exit(77)

ROOT = Path(__file__).resolve().parents[1]
PLOT = ROOT / "evaluation" / "plot.py"
GOLDEN = ROOT / "tests" / "fixtures" / "plot_report.html"
MiB = 1 << 20
SIZES = (4 * MiB, 64 * MiB)

spec = importlib.util.spec_from_file_location("plot", PLOT)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)

# GB/s per process (process 1, 2, 3) of the point the tests recompute: patched progressive pinning,
# CPU -> GPU, 64 MiB. Each process's other sample is 1.1x faster (see `row`), so the point, the
# median of all 6 samples, is 12.6: not the median of the rows' own values (12, the slower samples),
# not their mean (14) and not the middle process (20).
PROGRESSIVE_64M = (10.0, 20.0, 12.0)
# Pin rate of the 4K register items at 64 MiB (the table's largest size), per process.
PIN_4K = {"stock": (5.0, 7.0, 6.0), "patched": (30.0, 8.0, 9.0)}


def row(size, rate, direction=None, phases=None):
    """One complete native row whose median sample ran at `rate` GB/s; the other sample is 1.1x
    faster, as the runner reports the slower of 2 samples as the row's median."""
    seconds = size / rate / 1e9
    found = {"status": "complete", "bytes": size, "gbps": rate, "seconds": seconds,
             "samples": [{"seconds": seconds}, {"seconds": seconds / 1.1}]}
    if direction:
        found["direction"] = direction
    if phases:
        found["phases_seconds"] = phases
    return found


def item(arm, kind, process, runs, status="complete", **labels):
    labels.update(kind=kind, arm=arm, process=process)
    return {"id": f"{kind}-{process}", "labels": labels, "status": status, "runs": runs,
            "argv": ["/bin/gpu_copy", "--out", "{out}", "--", "--min-bytes", "1"]}


def fixture_run(directory, name, kernel=True):
    root = directory / name
    for arm in ("stock", "patched") if kernel else ():
        found = []
        for process in (1, 2, 3):
            for direction in plot.DIRECTIONS:
                base = 10.0 if arm == "stock" else 14.0
                for kind, rate in (("driver", 9.0), ("pin_pipeline", base)):
                    found.append(item(arm, kind, process, [row(s, rate + process, direction) for s in SIZES],
                                      direction=direction, pages="thp"))
                if arm == "stock":
                    refused = [{"status": "failed", "reason": "path/count mismatch", "bytes": s} for s in SIZES]
                    found.append(item(arm, "progressive", process, refused, "refused", direction=direction,
                                      pages="thp"))
                else:
                    rate = PROGRESSIVE_64M[process - 1] if direction == "htod" else 15.0
                    found.append(item(arm, "progressive", process, [row(s, rate, direction) for s in SIZES],
                                      direction=direction, pages="thp"))
                for pages in ("4k", "thp"):
                    found.append(item(arm, "pin_copy_unpin", process,
                                      [row(s, 7.0 + process, direction) for s in SIZES],
                                      direction=direction, pages=pages))
            for pages in ("4k", "thp"):
                pin = PIN_4K[arm][process - 1] if pages == "4k" else 50.0
                runs = [row(s, 4.0, phases={"setup": s / pin / 1e9, "copy": 0.0, "teardown": s / 40e9})
                        for s in SIZES]
                found.append(item(arm, "register", process, runs, pages=pages))
        directory_out = root / "bench" / "kernel" / arm
        directory_out.mkdir(parents=True)
        for index, saved in enumerate(found):
            saved["id"] = f"{saved['id']}-{index}"
            (directory_out / f"{saved['id']}.json").write_text(json.dumps(saved))
        (directory_out / "machine.json").write_text(json.dumps(
            {"gpus": [{"name": "Fixture GPU"}], "host": {"cpu": {"model": "Fixture CPU"}},
             "driver": {"kernel_module": {"version": "1.0"}}}))
    workloads = root / "workload"
    workloads.mkdir(parents=True)
    for name_, better, stock, hook in (("w-time", "lower", (10, 11, 12), (8, 8.5, 9)),
                                       ("w-rate", "higher", (100, 105, 110), (104, 111, 112))):
        runs = [{"arm": arm, "repeat": index + 1, "result": {"value": value}}
                for arm, values in (("stock", stock), ("hook", hook)) for index, value in enumerate(values)]
        record = {"workload": {"name": name_, "repeats": 3,
                               "metric": {"name": "m", "unit": "s" if better == "lower" else "/s",
                                          "better": better}},
                  "config": "tune/full/tune.json", "runs": runs}
        (workloads / f"{name_}.json").write_text(json.dumps(record))
    return root


def run_plot(*args):
    return subprocess.run([sys.executable, str(PLOT), *map(str, args)], capture_output=True, text=True)


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="plot-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.run_a = fixture_run(self.directory, "fixture-a")

    def drawn_lines(self):
        """Draw the fixture in this process; {(png, panel title, line label): {x: y}}."""
        lines, save = {}, plot.save

        def capture(fig, out, name, rect=None):
            for ax in fig.axes:
                for drawn in ax.get_lines():
                    points = dict(zip(drawn.get_xdata(), drawn.get_ydata()))
                    lines[(name, ax.get_title(loc="left"), drawn.get_label())] = points
            return save(fig, out, name, rect)

        plot.save = capture
        try:
            page = plot.single_run(self.run_a, self.directory / "in-process")
        finally:
            plot.save = save
        return lines, page

    def test_a_point_and_a_cell_match_values_recomputed_from_the_fixture(self):
        lines, page = self.drawn_lines()
        expected = statistics.median(rate * faster for rate in PROGRESSIVE_64M for faster in (1.0, 1.1))
        drawn = lines[("regcache.png", "Registration cache: CPU → GPU", "progressive pinning, patched")]
        self.assertAlmostEqual(drawn[64 * MiB], expected, places=9)
        self.assertAlmostEqual(expected, 12.6, places=9)
        # The stock child refused in every process, so there is no stock progressive line.
        self.assertNotIn(("regcache.png", "Registration cache: CPU → GPU", "progressive pinning, stock modules"),
                         lines)
        self.assertIn("refused in 6 of 6 processes", page)
        stock, patched = (statistics.median(PIN_4K[arm]) for arm in ("stock", "patched"))
        self.assertIn(f"<tr><td>pin, 4K pages</td><td>{stock:.1f}</td><td>{patched:.1f}</td>"
                      f"<td>{patched / stock:.2f}×</td></tr>", page)
        self.assertEqual((stock, patched), (6.0, 9.0))

    def test_verdict_rule(self):
        verdict = plot.verdict
        # Lower is better (seconds): every hooked run faster than every stock run, 1.29x.
        self.assertEqual(verdict([10, 11, 12], [8, 8.5, 9], "lower")[1], "faster")
        self.assertAlmostEqual(verdict([10, 11, 12], [8, 8.5, 9], "lower")[0], 11 / 8.5)
        # The same numbers as a rate (higher is better) are a slowdown.
        self.assertEqual(verdict([10, 11, 12], [8, 8.5, 9], "higher")[1], "slower")
        # Higher is better, separated and 1.19x: faster.
        self.assertEqual(verdict([100, 105, 110], [120, 125, 130], "higher")[1], "faster")
        # A large median gain whose runs overlap is not called.
        self.assertEqual(verdict([10, 11, 12], [8, 9, 10.5], "lower")[1], "no clear change")
        # Separated but below the gain on either side (1.02x, 0.98x): not called.
        self.assertEqual(verdict([100, 100.5, 101], [98, 98.5, 99], "lower")[1], "no clear change")
        self.assertEqual(verdict([100, 101, 102], [98, 99, 99.5], "higher")[1], "no clear change")
        # Exactly at the gain counts; just under it does not.
        self.assertEqual(verdict([105, 105, 105], [100, 100, 100], "lower")[1], "faster")
        self.assertEqual(verdict([104.9, 104.9, 104.9], [100, 100, 100], "lower")[1], "no clear change")
        # Slower needs every hooked run worse as well as the median drop.
        self.assertEqual(verdict([10, 10, 10], [12, 12, 12], "lower")[1], "slower")
        self.assertEqual(verdict([10, 10, 13], [12, 12, 12], "lower")[1], "no clear change")
        # An empty arm has no verdict; an unknown direction is an error, not a guess.
        self.assertEqual(verdict([], [1, 2], "lower"), (None, ""))
        with self.assertRaises(ValueError):
            verdict([1], [2], "bigger")

    def test_workload_table_uses_the_verdict(self):
        _, page = self.drawn_lines()
        self.assertIn("<td>w-time</td><td>m</td><td>lower</td><td>11</td><td>8.5</td>"
                      "<td>1.294×</td><td>faster</td>", page)
        # 111 / 105 = 1.057 clears the gain, but a hooked run (104) lost to a stock run (105, 110).
        self.assertIn("<td>w-rate</td><td>m</td><td>higher</td><td>105</td><td>111</td>"
                      "<td>1.057×</td><td>no clear change</td>", page)

    def test_single_run_page_is_unchanged(self):
        result = run_plot(self.run_a)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.run_a / "report"
        self.assertEqual(result.stdout.strip(), str(out / "index.html"))
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["index.html", "regcache.png", "workloads.png"])
        page = (out / "index.html").read_text()
        if os.environ.get("CMT_UPDATE_GOLDEN") == "1":
            GOLDEN.write_text(page)
        self.assertEqual(page, GOLDEN.read_text(),
                         f"the single-run page changed; if on purpose, rerun with CMT_UPDATE_GOLDEN=1 "
                         f"and review the diff of {GOLDEN}")
        # --out for one run draws the same page elsewhere.
        result = run_plot(self.run_a, "--out", self.directory / "elsewhere")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.directory / "elsewhere" / "index.html").read_text(), page)

    def test_several_runs_draw_one_figure_per_run_in_each_section(self):
        run_b = fixture_run(self.directory, "fixture-b", kernel=False)
        out = self.directory / "compare"
        result = run_plot(self.run_a, run_b, "--out", out)
        self.assertEqual(result.returncode, 0, result.stderr)
        page = (out / "index.html").read_text()
        self.assertIn("<h1>CMT evaluation: fixture-a, fixture-b</h1>", page)
        workloads = page[page.index("<h2>Workloads: stock vs hook</h2>"):]
        self.assertLess(workloads.index("<h3>fixture-a</h3><p>"), workloads.index("<h3>fixture-b</h3><p>"))
        self.assertIn("<img src='fixture-a/workloads.png'", workloads)
        self.assertIn("<img src='fixture-b/workloads.png'", workloads)
        kernel = page[page.index("<h2>Registration cache"):page.index("<h2>Workloads")]
        self.assertIn("<img src='fixture-a/regcache.png'", kernel)
        self.assertIn("<h3>fixture-b</h3><p class='muted'>Not measured in this run.</p>", kernel)
        # Sections no run measured are listed once, not drawn.
        self.assertIn("Not measured in any run: Driver references", page)
        self.assertNotIn("<h2>Driver references</h2>", page)
        # Each run's figure is the image it gets drawn alone.
        alone = run_plot(self.run_a)
        self.assertEqual(alone.returncode, 0, alone.stderr)
        for name in ("regcache.png", "workloads.png"):
            self.assertEqual((out / "fixture-a" / name).read_bytes(),
                             (self.run_a / "report" / name).read_bytes(), name)
        # Several runs need --out; run names label figures, so they must differ.
        self.assertEqual(run_plot(self.run_a, run_b).returncode, 2)
        self.assertEqual(run_plot(self.run_a, self.run_a, "--out", out).returncode, 2)


GiB = 1 << 30
PINNING = {"method": "progressive_pinning", "first_chunk_bytes": 4 * MiB, "max_chunk_bytes": 64 * MiB,
           "max_registered_bytes": GiB, "read_only": False}
DRIVER = {"method": "driver"}


def staged(workers, slots, chunk):
    return {"method": "staged", "cpu_loop": "memcpy", "pipeline_workers": workers,
            "slots_per_worker": slots, "chunk_bytes": chunk, "staging_memory": "cached_pinned"}


def tune_run(size, rate, settings, coverage="measured"):
    """One complete GPU -> CPU tune.json run of 2 equal samples at `rate` GB/s."""
    seconds = size / rate / 1e9
    execution = ("bare_driver" if settings["method"] == "driver"
                 else {"measured": "hooked_copy", "instant": "direct_engine"}[coverage])
    return {"direction": "dtoh", "bytes": size, "process_round": 1, "status": "complete",
            "coverage": coverage, "timing_boundary": "completed_pageable_copy",
            "measurement_class": "primary", "execution": execution, "settings": settings,
            "samples": [{"seconds": seconds}, {"seconds": seconds}]}


# At 1 GiB (the largest size and, in an explicit request, the limit), the shape of the 3090 FULL
# with-kernel table: pinning 25.7, staged 20 and 19, one staged setting under the driver's 18.4
# (fails both driver gates).
A, B, SLOW = staged(2, 1, 8 * MiB), staged(1, 1, 4 * MiB), staged(1, 2, MiB)
AT_BUDGET = [tune_run(GiB, 18.4, DRIVER), tune_run(GiB, 25.7, PINNING), tune_run(GiB, 20, A),
             tune_run(GiB, 19, B), tune_run(GiB, 18.3, SLOW)]
INSTANT = [tune_run(4 * MiB, 10, DRIVER, "instant"), tune_run(4 * MiB, 20, SLOW, "instant")]


# A request from `cmt tune --max-registered-mib 1024`: the recorded 1 GiB is a limit. Without the
# statement (an older request) the same 1 GiB is the old default and no limit.
EXPLICIT = {"process_rounds": 1, "registration_limit": "explicit_only"}
OLD = {"process_rounds": 1}


def dtoh_rows(runs, selection=None, request=EXPLICIT):
    tune = {"request": request, "runs": runs}
    if selection:
        tune["selection"] = selection
    return [(low, settings) for direction, low, _, settings
            in plot.selected_rows(tune, plot.summaries(tune)) if direction == "dtoh"]


class SelectedRowsTests(unittest.TestCase):
    """The rebuild mirrors BuildCopyLookup (tests/test_selection.cpp has the same cases)."""

    def test_above_a_pinning_budget_the_best_unpinned_setting_runs(self):
        self.assertEqual(dtoh_rows(AT_BUDGET), [(0, DRIVER), (GiB, PINNING), (GiB + 1, A)])

    def test_above_a_pinning_budget_with_nothing_else_compared(self):
        pinning_only = AT_BUDGET[:2]
        self.assertEqual(dtoh_rows(pinning_only)[-1], (GiB + 1, DRIVER))
        # Nothing measured but pinning: the INSTANT row, as before the rule. Something else measured
        # that did not qualify: a measured driver row, which replaces INSTANT's.
        self.assertEqual(dtoh_rows(pinning_only + INSTANT)[-1], (GiB + 1, SLOW))
        self.assertEqual(dtoh_rows(AT_BUDGET[:2] + AT_BUDGET[4:] + INSTANT)[-1], (GiB + 1, DRIVER))

    def test_between_a_pinning_budget_and_the_next_size(self):
        c = staged(4, 1, 4 * MiB)
        runs = [tune_run(512 * MiB, 18.4, DRIVER), tune_run(512 * MiB, 25.7, PINNING),
                tune_run(512 * MiB, 20, A), tune_run(2 * GiB, 18, DRIVER), tune_run(2 * GiB, 21, c)]
        self.assertEqual(dtoh_rows(runs), [(0, DRIVER), (512 * MiB, PINNING), (GiB + 1, A), (2 * GiB, c)])

    def test_a_line_with_a_pinning_row_is_used_only_if_it_states_the_rule(self):
        def line(above):
            rows = [{"direction": "dtoh", "min_bytes": low, "max_bytes": high, "settings": settings}
                    for low, high, settings in ((0, GiB, DRIVER), (GiB, GiB + 1, PINNING),
                                                (GiB + 1, plot.NO_LIMIT, above))]
            return {"rows": rows}
        rules = {"above_pinning_budget": "unpinned", "registration_limit": "explicit_only"}
        # Stated: used as recorded (B, where the rebuild selects A).
        self.assertEqual(dtoh_rows(AT_BUDGET, {**line(B), **rules})[-1], (GiB + 1, B))
        # As an older cmt wrote it (the driver above the budget, no statement), or without either
        # statement: rebuilt.
        self.assertEqual(dtoh_rows(AT_BUDGET, line(DRIVER))[-1], (GiB + 1, A))
        for missing in rules:
            stated = {**line(B), **{k: v for k, v in rules.items() if k != missing}}
            self.assertEqual(dtoh_rows(AT_BUDGET, stated)[-1], (GiB + 1, A), missing)
        # A line without a pinning row needs no statement: used as recorded.
        recorded = {"rows": [{"direction": "dtoh", "min_bytes": 0, "max_bytes": plot.NO_LIMIT,
                              "settings": B}]}
        self.assertEqual(dtoh_rows(AT_BUDGET, recorded), [(0, B)])

    def test_an_old_default_limit_does_not_end_pinning_rows(self):
        # An older request (RTX 3090 shape): its 1 GiB is the default, so pinning extends.
        self.assertEqual(dtoh_rows(AT_BUDGET, request=OLD), [(0, DRIVER), (GiB, PINNING)])
        unlimited = {**PINNING, "max_registered_bytes": None}
        runs = [tune_run(GiB, 18.4, DRIVER), tune_run(GiB, 25.7, unlimited), tune_run(GiB, 20, A)]
        self.assertEqual(dtoh_rows(runs), [(0, DRIVER), (GiB, unlimited)])
        # Any other value in an older request is explicit, and stays a limit.
        limited = {**PINNING, "max_registered_bytes": 512 * MiB}
        runs = [tune_run(512 * MiB, 18.4, DRIVER), tune_run(512 * MiB, 25.7, limited),
                tune_run(512 * MiB, 20, A)]
        self.assertEqual(dtoh_rows(runs, request=OLD),
                         [(0, DRIVER), (512 * MiB, limited), (512 * MiB + 1, A)])

    def test_a_line_row_without_a_limit_finds_the_rate_of_runs_that_recorded_one(self):
        # `cmt check --write` on an older file writes its pinning rows with null; the runs keep 1 GiB.
        tune = {"request": OLD, "runs": AT_BUDGET}
        summary = plot.summaries(tune)
        unlimited = {**PINNING, "max_registered_bytes": None}
        self.assertEqual(plot.row_rate(summary, "dtoh", 4 * GiB, unlimited), (25.7, 25.7, 25.7))
        self.assertEqual(plot.row_rate(summary, "dtoh", 4 * GiB, PINNING), (25.7, 25.7, 25.7))


def copy(size, src="pageable", dst="device", result=0, src_last=None):
    """One `cmt trace` copy record (the fields plot_traces reads, as the tracer writes them)."""
    return json.dumps({"type": "copy", "api": "cudaMemcpyAsync", "bytes_requested": size, "result": result,
                       "src_first": src, "src_last": src_last or src, "dst_first": dst, "dst_last": dst})


# Lines in every trace that the figure, and so the table, must not count. Each is large enough that
# counting it would change a cell: a failed copy, pinned and device-device copies, a host range that
# is pageable only at its first byte, a zero-byte copy, pageable to pageable, and lines that are not
# copies or not JSON. (No 2D record: the tracer does not record 2D/3D copies at all.)
NOISE = [json.dumps({"type": "trace_begin", "schema": 1}),
         copy(999 * MiB, result=1), copy(999 * MiB, "device", "pageable", result=700),
         copy(500 * MiB, "pinned", "device"), copy(500 * MiB, "device", "pinned"),
         copy(500 * MiB, "device", "device"), copy(500 * MiB, src_last="pinned"),
         copy(0), copy(500 * MiB, "pageable", "pageable"),
         json.dumps({"type": "host_allocation", "api": "cudaHostAlloc", "bytes": 500 * MiB, "result": 0}),
         "{not json", copy(500 * MiB)[:40]]
# name -> (CPU -> GPU sizes, GPU -> CPU sizes) of the pageable copies that count.
TRACES = {
    "twin-b": ([60 * MiB, MiB, 2 * MiB, MiB], [3000, 1000]),  # the same as twin-a, in another order
    "twin-a": ([MiB, MiB, 2 * MiB, 60 * MiB], [1000, 3000]),
    "dtoh-only": ([], [10 * MiB, 3 * GiB, 10 * MiB]),
    "no-pageable": ([], []),
    "small": ([512], []),
    "fill-3": ([3 * MiB + 5, 3 * MiB + 5], []),
    # Half the bytes exactly at the second copy: the byte-weighted median is 2.5M, not the next 5M.
    "fill-5": ([5 * MiB // 2, 5 * MiB // 2, 5 * MiB], []),
    **{f"fill-{k}": ([k * MiB, k * MiB], []) for k in (2, 4, 6, 7)},
}


class TraceTableTests(unittest.TestCase):
    """The table under the copy-size figure (`plot_traces`), on 11 traces: 9 distinct pageable
    distributions (so the smallest is not drawn), two traces with the same one, one with GPU -> CPU
    copies only and one with no pageable copy, each with NOISE mixed in."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="plot-traces-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "run"
        traces = self.root / "workload" / "traces"
        traces.mkdir(parents=True)
        for name, (htod, dtoh) in TRACES.items():
            lines = [copy(s) for s in htod] + [copy(s, "device", "pageable") for s in dtoh]
            (traces / f"{name}.jsonl").write_text("\n".join(NOISE[:6] + lines + NOISE[6:]) + "\n")

    def draw(self):
        """(caption, [table row as cell texts], legend labels, {trace path: times read})."""
        reads, legend, read_text, save = {}, [], Path.read_text, plot.save

        def counting(path, *args, **kwargs):
            reads[path.name] = reads.get(path.name, 0) + 1
            return read_text(path, *args, **kwargs)

        def capture(fig, out, name, rect=None):
            legend.extend(text.get_text() for drawn in fig.legends for text in drawn.get_texts())
            return save(fig, out, name, rect)

        (self.root / "report").mkdir()
        Path.read_text, plot.save = counting, capture
        try:
            _, (caption, table) = plot.plot_traces(self.root, self.root / "report")
        finally:
            Path.read_text, plot.save = read_text, save
        rows = [[html.unescape(c) for c in re.findall(r"<t[dh]>(.*?)</t[dh]>", row)]
                for row in re.findall(r"<tr>(.*?)</tr>", table)]
        return caption, rows, legend, reads

    def test_every_trace_has_a_row_of_only_the_copies_the_figure_counts(self):
        caption, rows, legend, reads = self.draw()
        self.assertEqual(rows[0], ["workload"] + [f"{d} {c}" for d in ("CPU → GPU", "GPU → CPU") for c in (
            "copies", "GB", "median", "byte-weighted median", "largest")] + ["total pageable GB", "drawn"])
        rows = {row[0]: row for row in rows[1:]}
        # Largest total first, a tie by name, no pageable copies last.
        self.assertEqual(list(rows), ["dtoh-only", "twin-a", "twin-b", "fill-7", "fill-6", "fill-5", "fill-4",
                                      "fill-3", "fill-2", "small", "no-pageable"])
        dash = ["–"] * 5
        # 64 MiB + 4000 bytes. Median of 4 is the lower middle (1M); half the bytes are reached at 60M;
        # 1000 + 3000 reaches half of 4000 only at 3000 (2.93 KiB).
        twin = ["4", "0.07", "1M", "60M", "60M", "2", "<0.01", "1000", "2.93K", "2.93K", "0.07", "yes"]
        self.assertEqual(rows["twin-a"][1:], twin)
        self.assertEqual(rows["twin-b"][1:], twin)  # same distribution, still its own row
        self.assertEqual(rows["dtoh-only"][1:], dash + ["3", "3.24", "10M", "3G", "3G", "3.24", "yes"])
        self.assertEqual(rows["fill-3"][1:], ["2", "0.01", "3.00M", "3.00M", "3.00M"] + dash + ["0.01", "yes"])
        self.assertEqual(rows["fill-2"][1:], ["2", "<0.01", "2M", "2M", "2M"] + dash + ["<0.01", "yes"])
        self.assertEqual(rows["fill-5"][1:], ["3", "0.01", "2.50M", "2.50M", "5M"] + dash + ["0.01", "yes"])
        self.assertEqual(rows["fill-7"][1:], ["2", "0.01", "7M", "7M", "7M"] + dash + ["0.01", "yes"])
        self.assertEqual(rows["small"][1:], ["1", "<0.01", "512", "512", "512"] + dash + ["<0.01", "no"])
        self.assertEqual(rows["no-pageable"][1:], dash + dash + ["0.00", "no"])
        # "drawn" is the figure: the legend names exactly the rows marked yes (twins on one line).
        drawn = {name for label in legend for name in label.rsplit(" (", 1)[0].split(" = ")}
        self.assertEqual(drawn, {name for name, row in rows.items() if row[-1] == "yes"})
        self.assertEqual(len(legend), 8)
        # The caption is the figure's, with one sentence added for the table.
        self.assertIn("Fewer pageable bytes, not drawn: small. No pageable copies: no-pageable. The table "
                      "below lists every workload, one row per trace;", caption)
        # One pass: the figure and the table come from the same single read of each trace.
        self.assertEqual(reads, {f"{name}.jsonl": 1 for name in TRACES})

    def test_byte_label(self):
        label = plot.byte_label
        # In the largest unit not above the size: a whole multiple of it and small counts as the axis
        # writes them; anything else rounded, with a decimal point so it never reads as a whole multiple
        # (size_label writes 2560K and 1187M).
        self.assertEqual([label(n) for n in (0, 512, 1023, 4 * MiB, 3 * GiB)], ["0", "512", "1023", "4M", "3G"])
        self.assertEqual([label(n) for n in (1025, 37632, 3 * MiB + 5, 102359, MiB - 1, 5 * MiB // 2, 1187 * MiB)],
                         ["1.00K", "36.8K", "3.00M", "100.0K", "1024.0K", "2.50M", "1.16G"])


if __name__ == "__main__":
    unittest.main()
