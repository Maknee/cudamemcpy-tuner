#!/usr/bin/env python3
"""Graphs and one HTML page from results/<name>/, as written by cmt-evaluate.

Usage: python3 plot.py results/<name>   ->  results/<name>/report/index.html + PNGs
       python3 plot.py RUN RUN... --out DIR  ->  DIR/index.html, DIR/<run name>/*.png

With several runs (machines), the page has one section per graph and one figure per run inside it,
each labelled by the run's directory name; one run's page is the same as when drawn alone, apart
from the run tables at the top. A section no run measured is listed, not drawn.

Python is used for plotting only. A bench point is the median of every sample behind it, over all
processes that measured the same setting: with one process of 2 samples, their mean. (The native
row's own median, `gbps`, is the slower of 2 samples, as gpu_copy and cpu_loops pick the upper
middle by time; plotting it would put every point at its slower sample.) The shaded band spans
every sample behind a point, and a cross marks a point with an item measured under load.
Tuning graphs draw the rows `cmt run` uses from tune.json (see `selected_rows`).
A graph whose inputs are missing is left out and listed as missing, never drawn empty; a bench
that ran but completed no row shows its items' reasons instead of a graph.
"""
import argparse
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated default palette, light surface).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# More slots for figures with more lines than COLORS (the CPU graph draws up to 10 loops per panel), and a
# light grey for a second stock/driver line. Every line in one figure gets its own colour; line style is
# only a second cue.
EXTRA = ["#8a5a2b", "#0f8f8f", "#a33ea1", "#6b7a1e"]
# Most distinct first: the two extra greens (dark green, teal) come last, after brown, magenta and olive.
PALETTE = COLORS[:5] + COLORS[6:] + [EXTRA[0], EXTRA[2], EXTRA[3], COLORS[5], EXTRA[1]]
LIGHT = "#a3a199"
DIRECTIONS = {"htod": "CPU → GPU", "dtoh": "GPU → CPU"}
# One colour per series name in every figure, so a series keeps its colour across panels and
# graphs. The stock driver is always grey.
SERIES = {
    "pageable": COLORS[0], "registered": COLORS[1], "hostalloc": COLORS[2], "cached": COLORS[2],
    "hostalloc_wc": COLORS[3], "wc": COLORS[3], "dma": COLORS[4], "inline": COLORS[5],
    "hook": COLORS[6], "instant": COLORS[0], "quick": COLORS[1], "full": COLORS[2],
    "tune-check": COLORS[5], "full-with-kernel": COLORS[7], "stock": MUTED,
}
PAGE_STYLE = {"4k": "-", "thp": "--", "driver": "-"}
LOADED = "measured under load"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 2, "lines.markersize": 5, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def size_label(n):
    for unit, shift in (("G", 30), ("M", 20), ("K", 10)):
        if n >= 1 << shift and n % (1 << shift) == 0:
            return f"{n >> shift}{unit}"
    return str(n)


def byte_label(n):
    """`size_label` for any byte count, always in the largest binary unit not above n: a whole
    multiple of it as size_label writes it ("3M"), bytes below 1 KiB as they are, otherwise rounded
    to 3 significant digits but never fewer than one decimal ("36.8K", "3.00M", "741.9M", not
    size_label's "759680K"), so a size that is not a whole multiple always shows a decimal point."""
    if n < 1 << 10:
        return str(n)
    unit, shift = next((u, s) for u, s in (("G", 30), ("M", 20), ("K", 10)) if n >= 1 << s)
    if n % (1 << shift) == 0:
        return size_label(n)
    value = n / (1 << shift)
    # Not .3g above 99.95: it drops the decimal ("100") and from 999.5 writes "1e+03".
    return f"{value:#.3g}{unit}" if value < 99.95 else f"{value:.1f}{unit}"


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def items(directory):
    """Saved item results of one command directory (not logs, not machine.json)."""
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.glob("*.json")):
        if path.name != "machine.json" and not path.name.endswith(".partial.json"):
            item = load(path)
            if item is not None:
                found.append(item)
    return found


def complete_rows(item):
    return [r for r in item.get("runs", []) if r.get("status") == "complete" and r.get("gbps")]


class NotMeasured(Exception):
    """A bench ran but completed no row; the message lists its items' reasons."""


GPU_COPY_DEFAULT = "64M (gpu_copy's default)"


def min_bytes(item):
    """The --min-bytes a gpu_copy item ran with, as text; None for other children (cpu_loops).
    The runner passes 1 (one copy per sample, as cmt tune times); items saved before it did ran
    gpu_copy's default, which repeats a small copy of one warm buffer inside each sample."""
    argv = item.get("argv", [])
    start = next((i for i, a in enumerate(argv) if Path(a).name == "gpu_copy"), None)
    if start is None:
        return None
    flags = argv[start:]
    return next((v for f, v in zip(flags, flags[1:]) if f == "--min-bytes"), GPU_COPY_DEFAULT)


class Mixed(Exception):
    """The items behind one graph were timed with different --min-bytes, so they do not compare."""


def same_sampling(**benches):
    """Raise Mixed when the gpu_copy items of these benches ({name: items}) did not all run with
    one --min-bytes: a run directory holds both until every item has rerun with the new runner."""
    counts = {name: Counter(v for v in map(min_bytes, found) if v is not None)
              for name, found in benches.items()}
    if len({v for c in counts.values() for v in c}) > 1:
        raise Mixed("; ".join(f"bench/{name}: {n} items with --min-bytes {v}"
                              for name, c in counts.items() for v, n in sorted(c.items())))


def bench_items(root, name):
    found = items(root / "bench" / name)
    same_sampling(**{name: found})
    if found and not any(complete_rows(i) for i in found):
        reasons = Counter(f"{i.get('status', 'unknown')} ({i.get('reason') or 'no complete row'})"
                          for i in found)
        raise NotMeasured("; ".join(f"{n} items {reason}" for reason, n in sorted(reasons.items())))
    return found


def gbps(row, seconds):
    return row["gbps"] * row["seconds"] / seconds  # the row's rate is its median sample's


def curve(found, x=lambda item, row: row["bytes"], value=gbps):
    """{x: (median, min, max, under load)} over the complete rows of `found`: the median and the
    range of every sample behind a point (a row without samples counts its own value), and whether
    an item behind it was measured under load. Rows whose `x` is None are left out."""
    rates, loaded = defaultdict(list), defaultdict(bool)
    for item in found:
        for row in complete_rows(item):
            key = x(item, row)
            if key is not None:
                rates[key] += ([value(row, s["seconds"]) for s in row.get("samples", []) if s.get("seconds")]
                               or [value(row, row["seconds"])])
                loaded[key] |= under_load(item)
    return {k: (statistics.median(v), min(v), max(v), loaded[k]) for k, v in sorted(rates.items())}


def size_axis(ax, ylabel="GB/s (decimal)"):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: size_label(int(x))))
    ax.set_xlabel("copy size")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def line(ax, points, label, color, style="-"):
    """A `curve`: the median line, the sample range shaded, a cross on each point under load."""
    if points:
        xs = list(points)
        ax.plot(xs, [points[x][0] for x in xs], style, color=color, label=label, marker="o")
        band(ax, xs, [points[x][1] for x in xs], [points[x][2] for x in xs], color)
        loaded = [x for x in xs if points[x][3]]
        if loaded:
            seen = LOADED in [c.get_label() for c in ax.collections]
            ax.scatter(loaded, [points[x][0] for x in loaded], marker="x", s=16, color=INK, zorder=4,
                       label="_" if seen else LOADED)


def band(ax, xs, low, high, color):
    ax.fill_between(xs, low, high, color=color, alpha=0.15, linewidth=0)


def legend(ax, **where):
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == LOADED)  # stable: the cross last
    ax.legend([handles[i] for i in order], [labels[i] for i in order], **where)


def save(fig, out, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    return name


def direction_panels(title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        ax.set_title(f"{title}: {DIRECTIONS[direction]}", loc="left")
    return fig, dict(zip(DIRECTIONS, axes))


def finish_panels(panels):
    for ax in panels.values():
        size_axis(ax)
        if ax.lines:
            legend(ax, loc="best")


# --- microbenchmarks --------------------------------------------------------------------------

def plot_references(root, out):
    found = bench_items(root, "references")
    if not found:
        return None
    fig, panels = direction_panels("Plain driver")
    # Name each memory by how it is made, and say which ones are pinned: the gap between the pinned lines and
    # the pageable one is what the hook can recover.
    names = {"pageable": "pageable (malloc)", "registered": "pinned: cudaHostRegister",
             "hostalloc": "pinned: cudaHostAlloc", "hostalloc_wc": "pinned: cudaHostAlloc, write-combined"}
    series = sorted({(i["labels"]["memory"], i["labels"]["pages"]) for i in found})
    for memory, pages in series:
        for direction, ax in panels.items():
            chosen = [i for i in found if i["labels"]["direction"] == direction
                      and (i["labels"]["memory"], i["labels"]["pages"]) == (memory, pages)]
            color = {("pageable", "thp"): EXTRA[1], ("registered", "thp"): EXTRA[0]}.get(
                (memory, pages), SERIES.get(memory, INK))  # THP gets its own colour, not only a dash
            line(ax, curve(chosen), f"{names.get(memory, memory)}, {pages} pages", color,
                 PAGE_STYLE.get(pages, "-"))
    for ax in panels.values():
        size_axis(ax)
    # One legend under both panels: six long labels inside a panel covered the small-copy curves.
    handles, labels = panels["htod"].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == LOADED)  # as legend(): the cross last
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3, fontsize=8)
    return save(fig, out, "references.png", rect=(0, 0.75 / fig.get_figheight(), 1, 1)), (
        "Completed copies through the stock driver for each host memory type: ordinary pageable memory, "
        "and three kinds of pinned (page-locked) memory, which the GPU reads directly. cudaHostAlloc memory "
        "uses pages the driver chooses, so it has one curve, not 4K and THP.")


def plot_dma(root, out):
    found = bench_items(root, "dma")
    if not found:
        return None
    references = items(root / "bench" / "references")
    same_sampling(dma=found, references=references)
    fig, panels = direction_panels("DMA policy")
    for direction, ax in panels.items():
        line(ax, curve([i for i in found if i["labels"]["direction"] == direction]),
             "DMA requested (cudaHostAlloc)", SERIES["dma"])
        line(ax, curve([i for i in references if i["labels"]["direction"] == direction
                        and i["labels"]["memory"] == "hostalloc"]),
             "stock driver (cudaHostAlloc)", SERIES["stock"], "--")
    finish_panels(panels)
    return save(fig, out, "dma.png"), (
        "Pinned copies requesting CMT_INLINE_POLICY=dma, against the stock driver on the same memory. "
        "The executed route is not independently audited.")


def plot_inline(root, out):
    found = bench_items(root, "inline")
    if not found:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, path in zip(axes, ("pageable", "pinned")):
        ax.set_title(f"CPU → GPU, {path} memory", loc="left")
        for policy in ("stock", "inline", "dma"):
            line(ax, curve([i for i in found if (i["labels"]["path"], i["labels"]["policy"]) == (path, policy)]),
                 policy if policy == "stock" else f"{policy} requested", SERIES[policy])
    for ax in axes:  # after drawing: sharey freezes the limits of the first ax set
        size_axis(ax)  # GB/s like every other graph, so the small-copy rates compare with the references
        if ax.lines:
            legend(ax, loc="upper left")
    return save(fig, out, "inline.png"), (
        "Rate per copy (bytes / time), median of every sample (3 processes); at these sizes the time per copy is "
        "mostly fixed overhead (4 KiB at 1 GB/s is 4 µs). "
        "Inline and DMA labels are requested policies; the executed route is not independently audited.")


def plot_wc(root, out):
    found = bench_items(root, "wc")
    if not found:
        return None
    fig, panels = direction_panels("Staging memory")
    series = sorted({(i["labels"]["memory"], i["labels"]["cpu_loop"]) for i in found})
    for memory, cpu_loop in series:
        for direction, ax in panels.items():
            chosen = [i for i in found if i["labels"]["direction"] == direction
                      and (i["labels"]["memory"], i["labels"]["cpu_loop"]) == (memory, cpu_loop)]
            plain = cpu_loop == "memcpy"
            style = "-" if plain else "--"
            isa = "libc" if plain else "" if "@" in cpu_loop else "widest ISA"
            # Its own colour per memory x loop (dash for streaming loops as a second cue): with colour by
            # memory only, memcpy and the streaming loop on the same memory looked alike.
            color = {("cached", True): COLORS[2], ("cached", False): COLORS[0],
                     ("wc", True): COLORS[3], ("wc", False): COLORS[7]}.get((memory, plain), INK)
            line(ax, curve(chosen), f"{memory} pinned, {cpu_loop}" + (f" ({isa})" if isa else ""),
                 color, style)
    finish_panels(panels)
    return save(fig, out, "wc.png"), (
        "Staged pageable copies through cached vs write-combined pinned slots (4 threads, 2 slots "
        "of 1 MiB each). Streaming loads/stores are the loops WC memory is meant for. A loop "
        "named without @ISA (older results) ran its widest variant, avx512 where the CPU has it, "
        "which reads WC memory far slower than avx2 (see the drain-wc panel of the CPU graph).")


def plot_pages(root, out):
    found = bench_items(root, "pages")
    if not found:
        return None
    references = items(root / "bench" / "references")
    same_sampling(pages=found, references=references)
    fig, panels = direction_panels("Application buffer pages")
    for direction, ax in panels.items():
        for pages in ("4k", "thp"):
            line(ax, curve([i for i in found if i["labels"]["direction"] == direction
                            and i["labels"]["pages"] == pages]), f"hook, {pages}",
                 SERIES["hook"] if pages == "4k" else COLORS[1], PAGE_STYLE[pages])  # purple vs orange
            line(ax, curve([i for i in references if i["labels"]["direction"] == direction
                            and i["labels"]["memory"] == "pageable" and i["labels"]["pages"] == pages]),
                 f"stock, {pages}", SERIES["stock"] if pages == "4k" else LIGHT, PAGE_STYLE[pages])
    finish_panels(panels)
    return save(fig, out, "pages.png"), (
        "Pageable copies from 4 KiB-page (solid) and THP (dashed) buffers, with the hook (purple 4K, orange THP) "
        "and with the stock driver (dark and light grey, from the references bench).")


def plot_cpu(root, out):
    found = bench_items(root, "cpu")
    if not found:
        return None
    # One colour per loop, the same loop in the same colour in every panel; line style = ISA is a second cue.
    styles = {"": "-", "sse2": "-", "avx2": "--", "avx512": ":"}
    every_loop = sorted({i["labels"]["cpu_loop"] for i in found})
    controls = sorted({i["labels"]["control"] for i in found})
    sizes = sorted({r["bytes"] for i in found for r in complete_rows(i)})
    if not sizes:
        return None
    size = sizes[-1]
    fig, axes = plt.subplots(1, len(controls), figsize=(3.6 * len(controls), 3.6), sharey=True,
                             squeeze=False)
    for ax, control in zip(axes[0], controls):
        ax.set_title(f"{control}, {size_label(size)}", loc="left")
        loops = sorted({i["labels"]["cpu_loop"] for i in found if i["labels"]["control"] == control})
        for cpu_loop in loops:
            isa = cpu_loop.partition("@")[2]
            points = curve([i for i in found if (i["labels"]["control"], i["labels"]["cpu_loop"])
                            == (control, cpu_loop)],
                           x=lambda item, row: item["labels"]["workers"] if row["bytes"] == size else None)
            line(ax, points, cpu_loop, PALETTE[every_loop.index(cpu_loop) % len(PALETTE)], styles.get(isa, "-"))
        ax.set_xlabel("copying threads")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0][0].set_ylim(bottom=0)  # after drawing: sharey freezes the limits of the first ax set
    axes[0][0].set_ylabel("GB/s (decimal)")
    legend(axes[0][-1], loc="center left", bbox_to_anchor=(1, 0.5), handlelength=4)  # long enough to show the ISA dash style
    return save(fig, out, "cpu.png"), (
        f"CPU copy bandwidth vs copying threads at {size_label(size)}, cold caches. Panels: "
        "pageable copies pageable memory to pageable memory; fill copies pageable memory into "
        "pinned staging memory (the CPU half of a staged CPU → GPU copy); drain copies pinned "
        "staging memory out to pageable memory (the CPU half of GPU → CPU). The pinned memory is "
        "cached (ordinary write-back) or wc (write-combined: fast to write with streaming stores, "
        "slow to read without streaming loads). Each loop has its own colour, the same in every "
        "panel; line style repeats the ISA (solid sse2/none, dashed avx2, dotted avx512). Threads "
        "own whole 1 MiB chunks, so small "
        "copies use fewer threads than requested.")


KERNEL_ARMS = ("stock", "patched")  # bench/kernel/<arm>/, as cmt-evaluate bench kernel writes them


def phase_rate(row, phase):
    """GB/s of one phase (`setup` = pin, `teardown` = unpin) of a register row: the phases of the
    sample gpu_copy chose as the row's median by total time, as every bench point uses the row's
    median sample. The median of each phase over the samples (session.sh's summarize.py) can differ:
    by up to 8% in an RTX 3090 run (2026-11-08; THP unpin, patched)."""
    seconds = (row.get("phases_seconds") or {}).get(phase)
    return row["bytes"] / seconds / 1e9 if seconds else None


def median_at(found, size, values):
    """Median of every value values(row) gives over the complete rows of `found` at `size`
    (None entries left out); None without one."""
    pooled = [v for i in found for r in complete_rows(i) if r["bytes"] == size
              for v in values(r) if v is not None]
    return statistics.median(pooled) if pooled else None


def plot_kernel(root, out):
    """bench kernel: the registration-cache figure (driver, progressive pinning and pin_pipeline, stock vs
    patched modules, THP buffers), and a table of pin, unpin and pin + copy + unpin at the largest size."""
    arms = {arm: bench_items(root, f"kernel/{arm}") for arm in KERNEL_ARMS}
    if not any(arms.values()):
        return None
    same_sampling(**{f"kernel/{arm}": found for arm, found in arms.items()})

    def pick(arm, kind, direction=None):
        return [i for i in arms[arm] if i["labels"]["kind"] == kind
                and i["labels"].get("direction") == direction]

    fig, panels = direction_panels("Registration cache")
    for direction, ax in panels.items():
        # Colours and styles as the registration-cache figure: driver grey, progressive purple,
        # pin_pipeline orange; dotted = stock modules, solid = patched.
        line(ax, curve(pick("stock", "driver", direction)), "driver, stock modules", SERIES["stock"], "--")
        line(ax, curve(pick("patched", "driver", direction)), "driver, patched modules", LIGHT, ":")
        line(ax, curve(pick("stock", "progressive", direction)), "progressive pinning, stock modules",
             SERIES["hook"], ":")
        line(ax, curve(pick("patched", "progressive", direction)), "progressive pinning, patched",
             SERIES["hook"])
        line(ax, curve(pick("stock", "pin_pipeline", direction)), "pin_pipeline, stock modules",
             COLORS[1], ":")
        line(ax, curve(pick("patched", "pin_pipeline", direction)), "pin_pipeline, patched", COLORS[1])
    finish_panels(panels)
    name = save(fig, out, "regcache.png")
    stock_progressive = [i for i in arms["stock"] if i["labels"]["kind"] == "progressive"]
    refused = sum(i.get("status") == "refused" for i in stock_progressive)
    sizes = [r["bytes"] for found in arms.values() for i in found for r in complete_rows(i)
             if i["labels"]["kind"] in ("register", "pin_copy_unpin")]
    table = ""
    if sizes:
        size = max(sizes)
        pages = (("4k", "4K pages"), ("thp", "THP"))
        # (row label, kind, pages, direction, GB/s values of one native row: a phase exists only for
        # the row's median sample, a whole copy for every sample)
        specs = [(f"{verb}, {label}", "register", kind_pages, None,
                  lambda row, phase=phase: [phase_rate(row, phase)])
                 for kind_pages, label in pages for phase, verb in (("setup", "pin"), ("teardown", "unpin"))]
        specs += [(f"pin + copy + unpin, {label}, {DIRECTIONS[direction]}", "pin_copy_unpin", kind_pages,
                   direction, sample_rates) for kind_pages, label in pages for direction in DIRECTIONS]
        cells = []
        for label, kind, kind_pages, direction, value in specs:
            stock, patched = (median_at([i for i in pick(arm, kind, direction)
                                         if i["labels"]["pages"] == kind_pages], size, value)
                              for arm in KERNEL_ARMS)
            shown = ["–" if v is None else f"{v:.1f}" for v in (stock, patched)]
            cells.append((label, *shown, "–" if None in (stock, patched) else f"{patched / stock:.2f}×"))
        table = html_table((size_label(size), "stock GB/s", "patched GB/s", "patched / stock"), cells)
    return name, (
        "Pageable copies from THP buffers at the sizes progressive pinning runs (4 MiB, its first chunk, to "
        "1 GiB, the default registration cap), median of every sample (3 processes of 2 each). Progressive pinning "
        "is the hook with the default request, audited as cmt tune audits it; on the stock modules the hook "
        f"cannot pin, and the audited child refused in {refused} of {len(stock_progressive)} processes "
        "(path/count mismatch), so there is no stock line unless one completed. pin_pipeline (register 32 MiB "
        "ahead of the copy, unpin at the end) runs on both modules: its gain from stock to patched is the "
        "patch's, not the algorithm's. The table: pin and unpin rates of the whole buffer, and pin + copy + "
        "unpin, at the largest size.", table)


# --- tuning ------------------------------------------------------------------------------------

# The selected rows are tune.json's line-1 `selection.rows` when present (what `cmt run` loads).
# Files without that line get them rebuilt as BuildCopyLookup in src/config.cpp does (selection
# policy measured_steps_v4): per direction, size and coverage, pool every sample rate (bytes /
# seconds per copy) of a setting across its processes; a setting needs exactly rounds
# 1..process_rounds (and >= 2 samples when measured). A candidate wins only if its slowest sample
# beats the driver's fastest (measured coverage) and its median beats the driver's by more than 2%;
# the fastest such median is selected, else the driver. A row runs from a size to the next size,
# taking the measured choice over the instant one. A progressive-pinning choice extends like any
# other unless it has a registration limit (`registration_limit`); then it ends there, and above it
# the same size's choice among the other candidates runs (the driver if none qualifies; nothing from
# that coverage if pinning was the only one measured). A line 1 with a pinning row but without
# `above_pinning_budget` and `registration_limit` is one `cmt run` does not use, so it is rebuilt too. Not rebuilt: the tie band (request.tie_percent) and the CPU-ISA and
# pinned-budget filters, which only differ when the plot runs on another machine than the tuning.
MIN_GAIN = 0.02
NO_LIMIT = (1 << 64) - 1
# A request without "registration_limit": "explicit_only" records 1 GiB as max_registered_mib unless
# --max-registered-mib was given; that value is a default, not a limit (config.h,
# kLegacyDefaultRegistrationMiB).
LEGACY_DEFAULT_REGISTRATION = 1024 << 20


def registration_limit(tune, settings):
    """Bytes a pinning setting may register, or None for no limit, as RegistrationLimitOf
    (src/config.cpp) reads it: recorded as is under "registration_limit": "explicit_only"; in an
    older request the old default (1 GiB) is not a limit, any other value is."""
    recorded = settings.get("max_registered_bytes")
    if tune.get("request", {}).get("registration_limit") == "explicit_only":
        return recorded
    return None if recorded in (None, LEGACY_DEFAULT_REGISTRATION) else recorded


def sample_rates(row):
    return [row["bytes"] / s["seconds"] / 1e9 for s in row.get("samples", []) if s.get("seconds")]


def summaries(tune):
    """{(direction, coverage, bytes): {settings key: (median, min, max) GB/s}} of every setting
    that qualifies for selection."""
    rounds = tune.get("request", {}).get("process_rounds", 1)
    execution = {"driver": "bare_driver", "measured": "hooked_copy", "instant": "direct_engine"}
    points = defaultdict(lambda: defaultdict(list))
    for row in tune.get("runs", []):
        coverage = row.get("coverage")
        if (row.get("status") != "complete" or coverage not in ("measured", "instant")
                or row.get("timing_boundary") != "completed_pageable_copy"
                or row.get("measurement_class") != "primary"):
            continue
        settings = row.get("settings") or {}
        driver = settings.get("method") == "driver"
        if row.get("execution") != execution["driver" if driver else coverage]:
            continue
        points[(row["direction"], coverage, row["bytes"])][json.dumps(settings, sort_keys=True)].append(row)
    found = {}
    for (direction, coverage, size), settings in points.items():
        found[(direction, coverage, size)] = summary = {}
        for key, rows in settings.items():
            required = 1 if coverage == "instant" else rounds
            seen = sorted({r.get("process_round", 1) for r in rows})
            rates = [rate for r in rows for rate in sample_rates(r)]
            if (seen != list(range(1, required + 1)) or len(seen) != len(rows)
                    or (coverage == "measured" and len(rates) < 2) or not rates):
                continue
            summary[key] = (statistics.median(rates), min(rates), max(rates))
    return found


def selected_rows(tune, summary):
    """[(direction, min_bytes, max_bytes, settings)]: line 1's rows, else rebuilt (see above)."""
    selection = tune.get("selection") or {}
    recorded = selection.get("rows")
    stale = (selection.get("above_pinning_budget") != "unpinned"
             or selection.get("registration_limit") != "explicit_only") and any(
        r["settings"].get("method") == "progressive_pinning"
        for r in [*(recorded or []), *(selection.get("tie_reference_rows") or [])])
    if recorded and not stale:
        return [(r["direction"], r["min_bytes"], r["max_bytes"], r["settings"]) for r in recorded]

    def best_of(candidates, coverage, d_median, d_max):
        best = None
        for key, (median, low, _) in candidates.items():
            if coverage == "measured" and low <= d_max:
                continue
            if median <= d_median * (1 + MIN_GAIN):
                continue
            if best is None or median > candidates[best][0]:
                best = key
        return best

    winners, unpinned = defaultdict(dict), defaultdict(dict)
    for (direction, coverage, size), rates in summary.items():
        driver_key = next((k for k in rates if json.loads(k).get("method") == "driver"), None)
        others = {k: v for k, v in rates.items() if k != driver_key}
        if driver_key is None or (coverage == "measured" and not others):
            continue
        d_median, _, d_max = rates[driver_key]
        best = best_of(others, coverage, d_median, d_max)
        winners[(direction, coverage)][size] = json.loads(best or driver_key)
        rest = {k: v for k, v in others.items() if json.loads(k).get("method") != "progressive_pinning"}
        if best and json.loads(best).get("method") == "progressive_pinning" and (coverage == "instant" or rest):
            unpinned[(direction, coverage)][size] = json.loads(
                best_of(rest, coverage, d_median, d_max) or driver_key)
    rows = []
    for direction in DIRECTIONS:
        asked = {r["bytes"] for r in tune.get("runs", []) if r.get("direction") == direction
                 and r.get("coverage") == "measured" and r.get("measurement_class") == "primary"
                 and r.get("timing_boundary") == "completed_pageable_copy"}
        instant, measured = winners[(direction, "instant")], winners[(direction, "measured")]
        bounds = {0, NO_LIMIT, *asked, *instant, *measured}
        bounds |= {registration_limit(tune, s) + 1 for s in [*instant.values(), *measured.values()]
                   if s.get("method") == "progressive_pinning" and registration_limit(tune, s) is not None}
        bounds = sorted(bounds)
        for low, high in zip(bounds, bounds[1:]):
            settings = {"method": "driver"}
            for coverage, points in (("instant", instant), ("measured", measured)):
                below = [size for size in points if size <= low]
                if not below or (points is measured and max(below) != max(s for s in asked if s <= low)):
                    continue
                chosen = points[max(below)]
                limit = registration_limit(tune, chosen)
                if chosen.get("method") == "progressive_pinning" and not (
                        chosen["first_chunk_bytes"] <= low and (limit is None or low <= limit)):
                    chosen = unpinned[(direction, coverage)].get(max(below))
                if chosen is not None:
                    settings = chosen
            rows.append((direction, low, high, settings))
    return rows


def rate_key(settings):
    """`settings` as a summary key, a pinning setting's limit left out: a file's pinning runs share
    one limit, and a line 1 records null where an older request's runs record the old default."""
    if settings.get("method") == "progressive_pinning":
        settings = {k: v for k, v in settings.items() if k != "max_registered_bytes"}
    return json.dumps(settings, sort_keys=True)


def row_rate(summary, direction, low, settings):
    """(median, min, max) of `settings` at the largest size <= low that measured it, measured
    coverage first; None below the smallest size."""
    key = rate_key(settings)
    for coverage in ("measured", "instant"):
        found = [(size, rates[k]) for (d, c, size), rates in summary.items()
                 if d == direction and c == coverage and size <= low
                 for k in rates if rate_key(json.loads(k)) == key]
        if found:
            return max(found, key=lambda point: point[0])[1]
    return None


def steps(ax, segments, label, color, style="-"):
    """[(low, high, (median, min, max))] as one stepped line over the shaded sample range; a
    segment without a rate is a gap."""
    xs, rates = [], []
    for low, high, rate in segments:
        xs += [low, high]
        rates += [rate or (float("nan"),) * 3] * 2
    ax.plot(xs, [r[0] for r in rates], style, color=color, label=label)
    band(ax, xs, [r[1] for r in rates], [r[2] for r in rates], color)


def plot_tune(root, out, groups, name, caption, modules=None):
    """One row of direction panels per group of tiers. Each tier's selected rows are steps at the
    rate measured for that setting at the row's first size; the driver's own steps are dotted, and
    the pages bench's hook curve on the tier it used shows the rate between sizes. `modules`
    ({tier: text}) makes a comparison: drawn only when every tier has runs, and each label names
    the tier's buffer pages (from its request) and its modules."""
    hooked = items(root / "bench" / "pages")
    drawn = []
    for group in groups:
        found = {}
        for tier in group:
            # bench/full-thorough is the earlier name of bench/tune-check; draw either.
            source = "bench/full-thorough" if tier == "bench/tune-check" and not (root / tier).exists() else tier
            tune = load(root / source / "tune.json")
            if tune and tune.get("runs"):
                summary = summaries(tune)
                found[tier] = (tune, summary, selected_rows(tune, summary), source)
        if found and (not modules or len(found) == len(group)):
            drawn.append(found)
    if not drawn:
        return None
    fig, axes = plt.subplots(len(drawn), 2, figsize=(11, 4.6 * len(drawn)), sharex=True,
                             sharey=True, squeeze=False)
    zoom = {ax: zoom_inset(ax) for ax in axes.flat}
    left_out = False
    for row_axes, found in zip(axes, drawn):
        for ax, direction in zip(row_axes, DIRECTIONS):
            names = " vs ".join(Path(tier).name for tier in found)
            ax.set_title(f"{names}: {DIRECTIONS[direction]}", loc="left")
            for tier, (tune, summary, rows, source) in found.items():
                label, pages = Path(tier).name, tune.get("request", {}).get("pages")
                if modules:
                    label += f": {'4K' if pages == 'small' else str(pages).upper()} pages, {modules[label]}"
                label += "" if tune.get("state") == "complete" else f" ({tune.get('state')})"
                color = SERIES.get(Path(tier).name, INK)
                with_table = [i for i in hooked if i["labels"]["direction"] == direction
                              and Path(i["labels"].get("config", "")).parent.parts[-2:] == Path(source).parts
                              and i["labels"]["pages"] == ("4k" if pages == "small" else pages)]
                # Only items timed as the table was ranked: a curve with more copies per sample
                # runs above the selected rows for that reason alone.
                timed = str(tune.get("request", {}).get("min_bytes"))
                hook = curve([i for i in with_table if min_bytes(i) == timed])
                left_out |= any(min_bytes(i) != timed for i in with_table)
                end = max([2 * size for (_, _, size) in summary] + list(hook))
                rows = [r for r in rows if r[0] == direction and r[1] > 0]
                for target in (ax, zoom[ax]):  # the inset redraws the same lines
                    steps(target, [(low, min(high, end), row_rate(summary, direction, low, settings))
                                   for _, low, high, settings in rows], f"selected, {label}", color)
                    steps(target, [(low, min(high, end), row_rate(summary, direction, low, {"method": "driver"}))
                                   for _, low, high, _ in rows], f"driver, {label}",
                      MUTED if tier == next(iter(found)) else LIGHT, ":")  # driver grey: not the tier's colour
                    line(target, hook, "hook with this table (bench pages)", SERIES["hook"])
            if not modules:  # upper left: left of the inset, above the curves
                legend(ax, loc="upper left")
    for ax in axes.flat:  # after drawing: sharey freezes the limits of the first ax set
        size_axis(ax)
    if any(len(zoomed(zoom[ax])) > 1 for ax in axes.flat):  # room above the curves only when an inset is drawn
        axes[0][0].set_ylim(top=axes[0][0].get_ylim()[1] / ZOOM_BELOW)
    for ax in axes.flat:
        frame_zoom(ax, zoom[ax])
    if left_out:
        caption += (" Hook items timed with another --min-bytes than their table was ranked with "
                    "(saved before the runner passed it) are left out; rerun `bench pages`.")
    if modules:  # a comparison's labels are too long to share a panel with the inset
        fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=2)
        return save(fig, out, name, rect=(0, 0.75 / fig.get_figheight(), 1, 1)), caption
    return save(fig, out, name), caption


ZOOM_TO = 64 << 10  # small copies run at a few GB/s and crowd the bottom of a 0-27 GB/s axis
ZOOM_BELOW = 0.62  # every curve stays below this fraction of the axis height; the inset sits above


def zoom_inset(ax):
    """An inset in the top right of `ax`, above every curve (see ZOOM_BELOW) and right of the
    upper-left legend, so it never hides a line."""
    return ax.inset_axes([0.64, ZOOM_BELOW + 0.06, 0.34, 0.98 - ZOOM_BELOW - 0.06])


def zoomed(inset):
    """The sizes up to ZOOM_TO that have a point in `inset`."""
    return {x for line in inset.get_lines() for x, y in zip(*line.get_data()) if x <= ZOOM_TO and y == y}


def frame_zoom(ax, inset):
    """Limit `inset` to copies up to ZOOM_TO, tall enough for every line and sample band there, and
    mark that region on `ax` with a dotted rectangle; drop an inset with fewer than two sizes in it (a tier
    that measured no small size, or only one: nothing to compare)."""
    points = [p for l in inset.get_lines() for p in zip(*l.get_data())]
    points += [p for c in inset.collections if isinstance(c, matplotlib.collections.PolyCollection)
               for path in c.get_paths() for p in path.vertices]  # the bands; not the × markers
    seen = [(x, y) for x, y in points if x <= ZOOM_TO and y == y]  # y == y drops a gap's NaN
    if len({x for x, _ in seen}) < 2:
        inset.remove()
        return
    left, right, top = min(x for x, _ in seen) / 1.2, ZOOM_TO * 1.2, 1.15 * max(y for _, y in seen)
    inset.set_xscale("log", base=2)
    inset.set_xlim(left, right)
    inset.set_ylim(0, top)
    inset.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: size_label(int(x))))
    inset.tick_params(labelsize=7)
    inset.set_facecolor(SURFACE)
    # Drawn by hand: indicate_inset_zoom's return type changed in matplotlib 3.10, and its connector
    # lines cross the main curves.
    ax.add_patch(matplotlib.patches.Rectangle((left, 0), right - left, top, fill=False,
                                              edgecolor=MUTED, linestyle=":", linewidth=1))


# --- workloads ---------------------------------------------------------------------------------

# The verdict rule of the campaign tables, with "slower" made the mirror of "faster": a workload is
# "faster" when the median speedup is at least VERDICT_GAIN AND every hooked run beat every stock run,
# "slower" when it is at most 1 / VERDICT_GAIN AND every hooked run lost to every stock run, and "no
# clear change" otherwise. Both halves are needed: with 3 runs per arm and no real effect, all three
# hooked runs beat all three stock runs by chance with probability 1 / C(6, 3) = 5%, so separation
# alone calls 1 workload in 20 faster; and a 2% gain with separated runs is within what a repeat on
# another day moves. 1.05 is the campaign's convention, not a measured noise floor.
VERDICT_GAIN = 1.05


def verdict(stock, hook, better):
    """(speedup, result) of one workload from its runs' values. speedup is the stock / hook median
    when lower is better, hook / stock when higher is better (above 1: the hook helped), and result
    is "faster", "slower" or "no clear change" by the rule above; (None, "") without runs in both arms."""
    if better not in ("lower", "higher"):
        raise ValueError(f"metric.better is {better!r}, not lower or higher")
    if not stock or not hook:
        return None, ""
    lower = better == "lower"
    speedup = (statistics.median(stock) / statistics.median(hook) if lower
               else statistics.median(hook) / statistics.median(stock))
    hook_better = max(hook) < min(stock) if lower else min(hook) > max(stock)
    hook_worse = min(hook) > max(stock) if lower else max(hook) < min(stock)
    if speedup >= VERDICT_GAIN and hook_better:
        return speedup, "faster"
    if speedup <= 1 / VERDICT_GAIN and hook_worse:
        return speedup, "slower"
    return speedup, "no clear change"


def plot_workloads(root, out):
    records = items(root / "workload")
    records = [r for r in records if r.get("runs")]
    if not records:
        return None
    columns = min(6, len(records))
    rows_count = -(-len(records) // columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(2.2 * columns, 2.7 * rows_count),
                             squeeze=False)
    rows = []
    for ax, record in zip(axes.flat, records):
        workload = record["workload"]
        metric = workload["metric"]
        values = {arm: [r["result"]["value"] for r in record["runs"]
                        if r["arm"] == arm and isinstance(r.get("result"), dict)]
                  for arm in ("stock", "hook", "warmup")}
        # The catalog's repeats, or more when the saved repeat numbers show a --repeats override.
        requested = max([workload["repeats"]] + [r["repeat"] for r in record["runs"]
                                                 if r["arm"] in ("stock", "hook")])
        counts = (len(values["stock"]), len(values["hook"]))
        short = f"runs {counts[0]}/{counts[1]} of {requested}" if min(counts) < requested or (
            counts[0] != counts[1]) else ""
        medians = {arm: statistics.median(values[arm]) for arm in ("stock", "hook") if values[arm]}
        speedup, result = verdict(values["stock"], values["hook"], metric["better"])
        for index, arm in enumerate(("stock", "hook")):
            if values[arm]:
                ax.scatter([index] * len(values[arm]), values[arm], color=SERIES[arm], zorder=3)
                ax.hlines(medians[arm], index - 0.25, index + 0.25, color=INK)
        ax.set_xticks([0, 1], ["stock", "hook"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 1.1 * max(values["stock"] + values["hook"] + [0]) or 1)  # top dots unclipped
        ax.set_title(f"{workload['name']}\n" + (f"hook {speedup:.3f}× " if speedup else "")
                     + (f"! {short}" if short else ""), loc="left", fontsize=8)
        ax.set_ylabel(f"{metric['unit']}, {metric['better']} is better", fontsize=8)
        rows.append((workload["name"], metric["name"], metric["better"],
                     f"{medians['stock']:.4g}" if "stock" in medians else "",
                     f"{medians['hook']:.4g}" if "hook" in medians else "",
                     f"{speedup:.3f}×" if speedup else "", result,
                     f"{counts[0]}/{counts[1]} of {requested}",
                     ", ".join(f"{v:.4g}" for v in values["warmup"]), short))
    for ax in axes.flat[len(records):]:
        ax.set_visible(False)
    name = save(fig, out, "workloads.png")
    table = html_table(("workload", "metric", "better", "stock median", "hook median",
                        "hook speedup", "result", "runs stock/hook", "warm-up (excluded)", "flag"), rows)
    return name, ("Each dot is one run; the bar is the median; y starts at zero. The number under "
                  "a name is the hook speedup: stock / hook median when lower is better, hook / "
                  "stock when higher is better, so above 1 means the hook helped. \"!\" flags fewer "
                  "runs than requested or unequal arms; such records are still drawn. A warm-up "
                  "run of the stock command, when the runner recorded one, is listed but never in "
                  f"a median. The hook arm used {html.escape(str(records[0].get('config')))}. "
                  f"Result: faster when the speedup is at least {VERDICT_GAIN:.2f}× and every hooked run "
                  f"beat every stock run; slower when it is at most 1/{VERDICT_GAIN:.2f}× and every "
                  "hooked run lost to every stock run; otherwise no clear change.",
                  table)


def plot_traces(root, out):
    traces = sorted((root / "workload" / "traces").glob("*.jsonl"))
    if not traces:
        return None
    # The copies the hook changes: whole range pageable on the host side, device on the other.
    kinds = {("pageable", "device"): "htod", ("device", "pageable"): "dtoh"}
    groups = defaultdict(list)  # identical {direction: sorted sizes} -> trace names
    per_trace = []  # (trace name, its key in groups): the table's rows, one per trace file
    for trace in traces:
        sizes = {direction: [] for direction in DIRECTIONS}
        for text in trace.read_text().splitlines():
            try:
                record = json.loads(text)
            except ValueError:
                continue
            direction = kinds.get((record.get("src_first"), record.get("dst_first")))
            if (record.get("type") == "copy" and record.get("result") == 0 and direction
                    and record.get("bytes_requested") and record.get("src_last") == record["src_first"]
                    and record.get("dst_last") == record["dst_first"]):
                sizes[direction].append(record["bytes_requested"])
        key = tuple(tuple(sorted(sizes[d])) for d in DIRECTIONS)
        groups[key].append(trace.stem)
        per_trace.append((trace.stem, key))
    total = {key: sum(map(sum, key)) for key in groups}
    chosen = sorted((key for key in groups if total[key]), key=lambda k: -total[k])[:8]
    if not chosen:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    handles = {}  # label -> line, largest first
    for index, key in enumerate(chosen):
        label = f"{' = '.join(groups[key])} ({total[key] / 1e9:.1f} GB)"
        for (by_calls, by_bytes), sizes in zip(axes, key):
            if sizes:
                running = [0]
                for size in sizes:
                    running.append(running[-1] + size)
                handles.setdefault(label, by_calls.step(
                    sizes, [(i + 1) / len(sizes) for i in range(len(sizes))], where="post",
                    color=COLORS[index])[0])
                by_bytes.step(sizes, [r / running[-1] for r in running[1:]], where="post",
                              color=COLORS[index])
    for (by_calls, by_bytes), direction in zip(axes, DIRECTIONS):
        for ax, title in ((by_calls, "share of calls"), (by_bytes, "share of bytes")):
            size_axis(ax, "cumulative share")
            ax.set_title(f"Pageable {DIRECTIONS[direction]}: {title}", loc="left")
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=2)
    bottom = 0.03 + 0.03 * ((len(handles) + 1) // 2)  # room for the figure legend
    fewer = [" = ".join(groups[key]) for key in groups if total[key] and key not in chosen]
    none = sorted(name for key in groups if not total[key] for name in groups[key])
    # Every trace, largest total first (no pageable copies: total 0, so last), then by name.
    rows = [(name, *trace_cells(key), gigabytes(total[key]), "yes" if key in chosen else "no")
            for name, key in sorted(per_trace, key=lambda t: (-total[t[1]], t[0]))]
    header = ["workload"] + [f"{DIRECTIONS[d]} {column}" for d in DIRECTIONS for column in (
        "copies", "GB", "median", "byte-weighted median", "largest")] + ["total pageable GB", "drawn"]
    return save(fig, out, "traces.png", (0, bottom, 1, 1)), (
        "Copy-size distributions recorded by cmt trace: successful 1D copies between pageable host "
        "memory and the device, the copies the hook changes, split by direction. The 8 workloads "
        "with the most pageable bytes are drawn, largest first; workloads with identical "
        "distributions share a line (names joined by =). Fewer pageable bytes, not drawn: "
        f"{html.escape(', '.join(fewer)) or 'none'}. No pageable copies: "
        f"{html.escape(', '.join(none)) or 'none'}. The table below lists every workload, one row "
        "per trace; its medians are the sizes where the curves first reach one half.",
        html_table(header, rows))


def gigabytes(n):
    """Decimal GB with 2 decimals, as the legend's GB but one more digit; "<0.01" for a nonzero
    count that would print as 0.00."""
    return f"{n / 1e9:.2f}" if n == 0 or n >= 5e6 else "<0.01"


def trace_cells(key):
    """A trace's table cells per direction from its sorted sizes (a `plot_traces` key): copies,
    GB, median, byte-weighted median and largest copy, or "–" in each without copies. The medians
    are read off the figure's step curves: the first size whose share of calls (median, so the
    lower middle of an even count) or share of bytes (byte-weighted) reaches one half."""
    cells = []
    for sizes in key:
        if not sizes:
            cells += ["–"] * 5
            continue
        whole, running = sum(sizes), 0
        for by_bytes in sizes:
            running += by_bytes
            if 2 * running >= whole:  # integers: no rounding at exactly one half
                break
        cells += [f"{len(sizes):,}", gigabytes(whole), byte_label(sizes[(len(sizes) - 1) // 2]),
                  byte_label(by_bytes), byte_label(sizes[-1])]
    return cells


# --- page --------------------------------------------------------------------------------------

def html_table(header, rows):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
                   for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def under_load(item):
    """True when load not from the runner reached the threshold at the start or end of the item."""
    for edge in ("start", "end"):
        snap = (item.get("load") or {}).get(edge) or {}
        if snap.get("max_load", 0) > 0 and snap.get("other", 0) >= snap["max_load"]:
            return True
    return False


def bench_dirs(root):
    """Every directory of saved bench items: bench/<bench>/, or for bench kernel its arms (its own
    directory holds only the restore's state file while it runs)."""
    for directory in sorted(p for p in (root / "bench").glob("*") if p.is_dir()):
        arms = [directory / arm for arm in KERNEL_ARMS if (directory / arm).is_dir()]
        yield from arms or [directory]


def status_rows(root):
    rows = []
    for directory in bench_dirs(root):
        counts = defaultdict(int)
        found = items(directory)
        for item in found:
            counts[item.get("status", "complete")] += 1
        loaded = [item["id"] for item in found if under_load(item)]
        if counts:
            text = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
            sampling = sorted({min_bytes(item) for item in found} - {None})
            if sampling:
                text += f"; --min-bytes {' and '.join(sampling)}"
            if loaded:
                text += f"; measured under load: {', '.join(loaded[:10])}"
                text += " ..." if len(loaded) > 10 else ""
            rows.append((f"bench/{directory.relative_to(root / 'bench').as_posix()}", text))
    for tune in sorted((root / "tune").glob("*/tune.json")):
        rows.append((f"tune/{tune.parent.name}", (load(tune) or {}).get("state", "unreadable")))
    return rows


def machine(root):
    for path in [*root.glob("bench/*/machine.json"), *root.glob("tune/*/tune.json"),
                 *root.glob("bench/kernel/*/machine.json")]:
        data = load(path)
        if data and "source_machine" in data:
            data = data["source_machine"]
        if data and data.get("gpus"):
            gpu, cpu = data["gpus"][0], data.get("host", {}).get("cpu", {})
            driver = data.get("driver", {}).get("kernel_module", {}).get("version", "?")
            return f"{gpu.get('name')} · {cpu.get('model', '?')} · driver {driver}"
    return "machine unknown (no native result yet)"


SECTIONS = [
    ("Driver references", plot_references),
    ("Copy engine only", plot_dma),
    ("Inline vs DMA", plot_inline),
    ("CPU copy loops", plot_cpu),
    ("Tuned settings vs the driver", lambda r, o: plot_tune(
        r, o, [(tier,) for tier in ("tune/instant", "tune/quick", "tune/full",
                                    "bench/tune-check", "tune/full-with-kernel")], "tune.png",
        "Solid: the setting the tuner selected for each size range (threads × buffers × chunk size "
        "× copy loop), drawn from the range's first size to the next row at the rate it measured at "
        "that first size. Dotted: the plain driver (cudaMemcpy without the hook) measured in the "
        "same tuning run at the same size; both are medians of all samples. Where no setting beat "
        "the driver clearly, the tuner selects the driver for that range and the two lines meet. "
        "The purple curve in the pages bench is the same table timed at many more sizes, so it "
        "shows the rate between the table's sizes.")),
    ("4K vs THP pages", plot_pages),
    ("Cached vs write-combined staging", plot_wc),
    ("Kernel tier", lambda r, o: plot_tune(
        r, o, [("tune/full", "tune/full-with-kernel")], "kernel.png",
        "FULL on the stock modules vs FULL_WITH_KERNEL on the host-registration (A+B+D) "
        "modules: selected rows (solid) and each tier's own driver (dotted), drawn as in the "
        "graph above. The tiers also differ in buffer pages (legend), so a gap mixes both changes.",
        {"full": "stock modules", "full-with-kernel": "patched modules"})),
    ("Registration cache: stock vs patched modules", plot_kernel),
    ("Workloads: stock vs hook", plot_workloads),
    ("Workload copy sizes", plot_traces),
]


def draw(root, out):
    """Every section of one run, its PNGs written to `out`: [(title, content)], where content is
    (image, caption, table) for a graph, an HTML paragraph for a bench that ran but is not drawn,
    or None when nothing was measured."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()  # a graph no longer drawn must not stay behind looking current
    drawn = []
    for title, plot in SECTIONS:
        try:
            result = plot(root, out)
        except NotMeasured as reasons:
            drawn.append((title, f"<p>Ran, but no row completed, so no graph: "
                                 f"{html.escape(str(reasons))}.</p>"))
            continue
        except Mixed as counts:
            drawn.append((title, f"<p>Not drawn: its items were timed with different --min-bytes, "
                                 f"so their rates do not compare ({html.escape(str(counts))}). Rerun "
                                 f"those benches with the current cmt-evaluate; it reruns the items "
                                 f"saved with other inputs.</p>"))
            continue
        if result is None:
            drawn.append((title, None))
            continue
        image, caption = result
        caption, table = (caption if isinstance(caption, tuple) else (caption, ""))
        drawn.append((title, (image, caption, table)))
    return drawn


def body(title, content, prefix=""):
    """One section's content as HTML; `prefix` is the directory of its PNGs relative to the page."""
    if isinstance(content, str):
        return content
    image, caption, table = content
    return f"<p>{caption}</p><img src='{prefix}{image}' alt='{html.escape(title)}'>{table}"


def page(heading, header, parts):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CMT evaluation</title>
<style>
:root {{ --surface: #fcfcfb; --ink: #0b0b0b; --muted: #52514e; --rule: #e4e3df; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --surface: #1a1a19; --ink: #f0efec; --muted: #c3c2b7; --rule: #383835; }}
}}
:root[data-theme="dark"] {{ --surface: #1a1a19; --ink: #f0efec; --muted: #c3c2b7; --rule: #383835; }}
body {{ background: var(--surface); color: var(--ink); font: 15px/1.5 system-ui, sans-serif;
       max-width: 1180px; margin: 0 auto; padding: 16px; }}
img {{ max-width: 100%; height: auto; border-radius: 6px; background: #fcfcfb; }}
table {{ border-collapse: collapse; margin: 8px 0 0; font-size: 13px; display: block; overflow-x: auto; }}
th, td {{ border-bottom: 1px solid var(--rule); padding: 3px 10px; text-align: left; }}
.muted, p {{ color: var(--muted); }}
section {{ margin: 28px 0; }}
</style></head><body>
<h1>CMT evaluation: {html.escape(heading)}</h1>
{header}
<p>In the graphs, a shaded band spans every sample behind a line (minimum to maximum over the
processes), and a black × marks a point from an item measured under load (listed above).
tune.json records no load per run, so the tuning graphs have no ×.</p>
{''.join(parts)}
</body></html>
"""


def missing_note(missing, text="Not measured yet"):
    return f"<p class='muted'>{text}: {html.escape(', '.join(missing))}.</p>" if missing else ""


def single_run(root, out):
    """The page of one run, its PNGs beside it."""
    drawn = draw(root, out)
    parts = [f"<section><h2>{html.escape(title)}</h2>{body(title, content)}</section>"
             for title, content in drawn if content is not None]
    missing = [title for title, content in drawn if content is None]
    status = html_table(("command", "items"), status_rows(root))
    header = f"<p>{html.escape(machine(root))}</p>\n{status}{missing_note(missing)}"
    return page(root.name, header, parts)


def several_runs(roots, out):
    """One section per graph and, inside it, one figure per run, labelled by run name. Each run's
    PNGs go to `out/<run name>/`, so runs with the same graphs do not overwrite each other."""
    drawn = {root.name: draw(root, out / root.name) for root in roots}
    parts, missing = [], []
    for index, (title, _) in enumerate(SECTIONS):
        if all(drawn[root.name][index][1] is None for root in roots):
            missing.append(title)
            continue
        figures = []
        for root in roots:
            content = drawn[root.name][index][1]
            figures.append(f"<h3>{html.escape(root.name)}</h3>" + (
                body(title, content, f"{quote(root.name)}/") if content is not None
                else "<p class='muted'>Not measured in this run.</p>"))
        parts.append(f"<section><h2>{html.escape(title)}</h2>{''.join(figures)}</section>")
    machines = html_table(("run", "machine"), [(root.name, machine(root)) for root in roots])
    status = html_table(("run", "command", "items"),
                        [(root.name, *row) for root in roots for row in status_rows(root)])
    header = f"{machines}\n{status}{missing_note(missing, 'Not measured in any run')}"
    return page(", ".join(root.name for root in roots), header, parts)


def main():
    parser = argparse.ArgumentParser(
        description="Graphs and one HTML page from cmt-evaluate result directories (results/<name>). "
                    "One run: the page goes to RUN/report/. Several: one section per graph, one "
                    "figure per run, in --out.")
    parser.add_argument("runs", nargs="+", type=Path, metavar="RUN", help="a results/<name> directory")
    parser.add_argument("--out", type=Path, help="report directory (default for one run: RUN/report)")
    args = parser.parse_args()
    for root in args.runs:
        if not root.is_dir():
            parser.error(f"{root} is not a directory")
    names = Counter(root.name for root in args.runs)
    if any(n > 1 for n in names.values()):
        parser.error(f"run names label the figures, so they must differ: {', '.join(sorted(names))}")
    if len(args.runs) > 1 and args.out is None:
        parser.error("several runs need --out DIR")
    out = args.out or args.runs[0] / "report"
    out.mkdir(parents=True, exist_ok=True)
    text = single_run(args.runs[0], out) if len(args.runs) == 1 else several_runs(args.runs, out)
    (out / "index.html").write_text(text)
    print(out / "index.html")


if __name__ == "__main__":
    main()


