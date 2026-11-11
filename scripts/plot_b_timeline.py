#!/usr/bin/env python3
"""Timeline figures for blog post #3 (B), drawn from the Nsight Systems reports' data.

These are NOT screenshots of the Nsight GUI: the NVTX ranges and CUDA runtime API calls are
read from the reports' SQLite export (nsys export --type sqlite) and drawn with matplotlib.
The GPU-side trace is missing in these WSL reports (LEDGER R21), so every bar is CPU-side
time: `reduce` is only the kernel launch, and the copy time shows up in the stream syncs.

Usage (repo root): python3 scripts/plot_b_timeline.py
"""
import re
import sqlite3
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = "docs/results"
OUT = "docs/results/figs"
WARMUP = 5  # hxccl_perf: 5 warm-up calls, then the measured ones

# light theme from the dataviz reference palette (validated: 4 categorical slots)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#b9b8b2"
COLOR = {
    "hxccl:allreduce": NEUTRAL,
    "ipc:copy": "#2a78d6",          # blue
    "reduce": "#eb6834",            # orange
    "cudaMallocAsync": "#1baf7a",   # aqua
    "cudaStreamSynchronize": "#eda100",  # yellow
}
LANES = ["hxccl:allreduce", "ipc:copy", "reduce", "cudaMallocAsync", "cudaStreamSynchronize",
         "cudaMemcpyAsync", "cudaLaunchKernel", "cudaFreeAsync"]
LABEL = {"reduce": "reduce (kernel launch)"}


def load(path):
    c = sqlite3.connect(path)
    t0 = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    nvtx = [(s, e, t) for s, e, t in c.execute(
        "select start, end, text from NVTX_EVENTS where end is not null and text in "
        "('hxccl:allreduce', 'ipc:copy', 'reduce')")]
    api = []
    for s, e, name in c.execute(
            "select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r "
            "join StringIds s on r.nameId = s.id"):
        api.append((s, e, re.sub(r"_v\d+$", "", name)))
    return t0, nvtx, api


def calls(nvtx):
    return sorted((s, e) for s, e, t in nvtx if t == "hxccl:allreduce")


def median_call(nvtx):
    """Index of the measured call whose length is the median (skips the warm-up calls)."""
    cs = calls(nvtx)[WARMUP:]
    med = statistics.median(e - s for s, e in cs)
    i = min(range(len(cs)), key=lambda k: abs((cs[k][1] - cs[k][0]) - med))
    return WARMUP + i


def spans(nvtx, api, lo, hi):
    out = {}
    for s, e, t in nvtx + api:
        if s >= lo and e <= hi and t in LANES:
            out.setdefault(t, []).append((s, e))
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw(ax, rows, base, label_min_ms):
    """rows: list of (lane label, [(start, end), ...], color) from top to bottom."""
    names = []
    for y, (name, ss, color) in enumerate(rows):
        names.append(name)
        for s, e in ss:
            x, w = (s - base) / 1e6, (e - s) / 1e6
            ax.barh(y, max(w, 0.002), left=x, height=0.56, color=color,
                    edgecolor=SURFACE, linewidth=1.0)
            if w >= label_min_ms and name.split(" ")[-1] in ("ipc:copy", "cudaMallocAsync",
                                                          "hxccl:allreduce"):
                ax.text(x + w / 2, y - 0.42, f"{w * 1e3:.0f} µs", ha="center", va="bottom",
                        fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), names)
    ax.invert_yaxis()
    ax.set_xlabel("time since the start of the call (ms)", color=INK2, fontsize=9)


def lane_rows(sp, prefix=""):
    rows = []
    for lane in LANES:
        if lane in sp:
            rows.append((prefix + LABEL.get(lane, lane), sp[lane], COLOR.get(lane, NEUTRAL)))
    return rows


def single(size, fig_name):
    t0, nvtx, api = load(f"{RES}/2026-11-11_b_nsys_{size}_r0.sqlite")
    k = median_call(nvtx)
    lo, hi = calls(nvtx)[k]
    sp = spans(nvtx, api, lo, hi)
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURFACE)
    style(ax)
    draw(ax, lane_rows(sp), lo, label_min_ms=(hi - lo) / 1e6 * 0.04)
    n_copy = len(sp.get("ipc:copy", []))
    ax.set_title(f"One device AllReduce, {size}, np 2, rank 0 (call {k + 1} of 25, the median "
                 f"length): {(hi - lo) / 1e6:.2f} ms, {n_copy} × ipc:copy",
                 loc="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.01, "Drawn from the Nsight Systems report data (NVTX + CUDA API, "
             "CPU side), not a GUI screenshot. No GPU trace in WSL.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{OUT}/{fig_name}", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return k, (hi - lo) / 1e3, n_copy


def both_ranks(size, fig_name):
    data = [load(f"{RES}/2026-11-11_b_nsys_{size}_r{r}.sqlite") for r in (0, 1)]
    k = median_call(data[0][1])
    rows, ranges = [], []
    for r, (t0, nvtx, api) in enumerate(data):
        lo, hi = calls(nvtx)[k]
        ranges.append((t0 + lo, t0 + hi))  # on the common UTC clock
        sp = spans(nvtx, api, lo, hi)
        shifted = {lane: [(t0 + s, t0 + e) for s, e in ss] for lane, ss in sp.items()
                   if lane in ("hxccl:allreduce", "ipc:copy", "cudaStreamSynchronize")}
        rows += lane_rows(shifted, prefix=f"rank {r}  ")
    base = min(lo for lo, _ in ranges)
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURFACE)
    style(ax)
    draw(ax, rows, base, label_min_ms=10.0)  # durations only on the whole call
    ax.set_title(f"Device AllReduce {size}, np 2: the same call on both ranks "
                 f"(call {k + 1} of 25)", loc="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.01, "Drawn from the Nsight Systems report data. Clock alignment uses "
             "TARGET_INFO_SESSION_START_TIME of each report (two nsys processes, one machine).",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{OUT}/{fig_name}", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return k, [(b - a) / 1e3 for a, b in ranges], (ranges[1][0] - ranges[0][0]) / 1e3


if __name__ == "__main__":
    import os

    os.makedirs(OUT, exist_ok=True)
    print("64K:", single("64K", "fig_b_timeline_64k.png"))
    print("64M:", single("64M", "fig_b_timeline_64m.png"))
    print("4M r0/r1:", both_ranks("4M", "fig_b_timeline_4m_r0r1.png"))
