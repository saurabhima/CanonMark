"""Plot watermark trade-off curves from a sweep: Pass@1 (x) against AUROC (y).

One panel per condition (default: clean code, and after all three edits),
shared y-axis. Each method is a line through its delta settings. Error bars
show the 95% problem-level bootstrap CI of the key-averaged AUROC
(or ± sd over keys if the sweep has no CIs) vertically, and ± sd of Pass@1
over keys horizontally.

Example:
  python scripts/plot_sweep.py --sweep outputs/sweep_he
  # -> outputs/sweep_he/tradeoff.png and tradeoff.pdf (for LaTeX \\includegraphics)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots 1-3 (dataviz reference palette, light mode; all pairs pass CVD checks).
# Two of them are below 3:1 contrast on white, so every series also gets a marker shape and a direct label.
STYLE = {
    "canonmark": {"color": "#257de0", "marker": "o", "label": "CanonMark"},
    "hybrid":    {"color": "#f47230", "marker": "s", "label": "Hybrid"},
    "kgw":       {"color": "#00b894", "marker": "^", "label": "KGW"},
}
TEXT, TEXT2, GRID, REF = "#191919", "#666666", "#e6e6e6", "#9c9c9c"
TITLES = {"clean": "Clean code", "black": "After black", "strip_comments": "After comment stripping",
          "rename": "After renaming", "all_three": "After all three edits"}


def fnum(x):
    return None if x in (None, "", "None") else float(x)


def place_labels(fig, ax, labels, points, segs, base):
    """Greedy collision-free placement: each label tries a few offsets and keeps the first
    whose box overlaps no placed label, no marker and stays inside the axes."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = 3.5                                        # half marker size in display px, roughly
    to_disp = ax.transData.transform
    marks = [to_disp(p) for p in points]
    import numpy as np
    line_pts = []                                    # dense samples along lines / error bars / reference lines
    for a0, a1 in segs:
        d0, d1 = to_disp(a0), to_disp(a1)
        k = int(max(2, min(400, np.hypot(*(d1 - d0)) / 3)))
        line_pts += [d0 + (d1 - d0) * t for t in np.linspace(0, 1, k)]
    frame = ax.get_window_extent(r)
    placed = []
    if base is not None:                             # reference-line label, top of the line
        t = ax.annotate("no watermark", (base, ax.get_ylim()[1]), xytext=(-3, -2), textcoords="offset points",
                        rotation=90, fontsize=6.5, color=TEXT2, ha="right", va="top")
        placed.append(t.get_window_extent(r))
    t = ax.annotate("chance", (ax.get_xlim()[0], 0.5), xytext=(3, 2), textcoords="offset points",
                    fontsize=6.5, color=TEXT2, ha="left", va="bottom")
    placed.append(t.get_window_extent(r))
    dpi_scale = fig.dpi / 72.0
    for text, x, y, size, color, is_series in labels:
        offs = ([(-9, -4, "right"), (9, -4, "left"), (-6, 7, "right"), (6, 7, "left"), (-6, -15, "right"),
                 (6, -15, "left"), (0, 12, "center"), (0, -20, "center")]
                if is_series else
                [(6, -11, "left"), (6, 4, "left"), (-6, -11, "right"), (-6, 4, "right"), (0, 7, "center"),
                 (0, -18, "center"), (10, -4, "left"), (-10, -4, "right")])
        best = None
        for dx, dy, ha in offs:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=size,
                            color=color, ha=ha, va="bottom")
            bb = t.get_window_extent(r).expanded(1.05, 1.15)
            hit = any(bb.overlaps(o) for o in placed) or any(
                bb.x0 - pad * dpi_scale < mx < bb.x1 + pad * dpi_scale and
                bb.y0 - pad * dpi_scale < my < bb.y1 + pad * dpi_scale for mx, my in marks) or any(
                bb.x0 < lx < bb.x1 and bb.y0 < ly < bb.y1 for lx, ly in line_pts)
            inside = frame.x0 <= bb.x0 and bb.x1 <= frame.x1 and frame.y0 <= bb.y0 and bb.y1 <= frame.y1
            if not hit and inside:
                best = t
                break
            t.remove()
        if best is None:                             # nothing free: fall back to the first choice
            dx, dy, ha = offs[0]
            best = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=size,
                               color=color, ha=ha, va="bottom")
        placed.append(best.get_window_extent(r))


def load(sweep: Path, sfx: str = ""):
    rows = list(csv.DictReader(open(sweep / f"sweep_summary{sfx}.csv")))
    base = None
    res = sweep / f"sweep_results{sfx}.json"
    if res.is_file():
        base = json.loads(res.read_text()).get("unwatermarked", {}).get("pass@1")
    return rows, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="sweep output directory (has sweep_summary.csv)")
    ap.add_argument("--panels", nargs="+", default=["clean", "all_three"])
    ap.add_argument("--methods", nargs="+", default=["canonmark", "hybrid", "kgw"])
    ap.add_argument("--dataset_label", default="HumanEval")
    ap.add_argument("--suffix", default="", help="read sweep files with this suffix, e.g. _weighted")
    ap.add_argument("--out", help="output path without extension (default: <sweep>/tradeoff<suffix>)")
    a = ap.parse_args()
    sweep = Path(a.sweep)
    rows, base = load(sweep, a.suffix)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "font.family": "sans-serif",
                         "axes.edgecolor": REF, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2})
    n = len(a.panels)
    fig, axes = plt.subplots(1, n, figsize=(3.25 * n, 2.7), sharey=True, squeeze=False)
    axes = axes[0]

    all_y = []
    pending = []
    for ax, panel in zip(axes, a.panels):
        labels, points, segs = [], [], []   # (text, x, y, size, color, is_series), markers, line segments
        for method in a.methods:
            pts = sorted((r for r in rows if r["method"] == method), key=lambda r: float(r["delta"]))
            if not pts:
                continue
            st = STYLE.get(method, {"color": TEXT2, "marker": "D", "label": method})
            x = [fnum(r["pass@1_mean"]) for r in pts]
            y = [fnum(r[f"auroc_{panel}_mean"]) for r in pts]
            xe = [fnum(r["pass@1_sd"]) or 0.0 for r in pts]
            lo = [fnum(r.get(f"auroc_{panel}_ci_lo")) for r in pts]
            hi = [fnum(r.get(f"auroc_{panel}_ci_hi")) for r in pts]
            if all(v is not None for v in lo + hi):
                ye = [[yi - l for yi, l in zip(y, lo)], [h - yi for yi, h in zip(y, hi)]]
                all_y += lo + hi
            else:
                sd = [fnum(r[f"auroc_{panel}_sd"]) or 0.0 for r in pts]
                ye = [sd, sd]
                all_y += [yi - s for yi, s in zip(y, sd)] + [yi + s for yi, s in zip(y, sd)]
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=st["color"], elinewidth=0.8,
                        capsize=2, alpha=0.55, zorder=2)
            ax.plot(x, y, color=st["color"], linewidth=1.5, zorder=3)
            ax.plot(x, y, linestyle="none", marker=st["marker"], markersize=6, color=st["color"],
                    markeredgecolor="white", markeredgewidth=1.0, zorder=4, label=st["label"])
            for (x0, y0), (x1, y1) in zip(zip(x, y), zip(x[1:], y[1:])):     # line segments as obstacles
                segs.append(((x0, y0), (x1, y1)))
            for xi, yi, xei, (ylo, yhi) in zip(x, y, xe, zip(*[[yi - d for yi, d in zip(y, ye[0])],
                                                               [yi + d for yi, d in zip(y, ye[1])]])):
                segs += [((xi - xei, yi), (xi + xei, yi)), ((xi, ylo), (xi, yhi))]   # error bars
            labels.append((st["label"], x[-1], y[-1], 7.5, TEXT, True))   # series name at strongest setting
            labels += [(f"δ={float(r['delta']):g}", xi, yi, 6.5, TEXT2, False) for xi, yi, r in zip(x, y, pts)]
            points += list(zip(x, y))
        if base is not None:
            ax.axvline(base, color=REF, linestyle="--", linewidth=0.8, zorder=1)
        ax.axhline(0.5, color=REF, linestyle=":", linewidth=0.8, zorder=1)
        if base is not None:
            segs.append(((base, 0.0), (base, 1.0)))
        segs.append(((0.0, 0.5), (1.0, 0.5)))
        pending.append((ax, labels, points, segs))
        ax.set_title(TITLES.get(panel, panel), color=TEXT, loc="left")
        ax.set_xlabel(f"Pass@1 ({a.dataset_label})")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("AUROC vs. unwatermarked output")
    if all_y:
        axes[0].set_ylim(min(0.48, min(all_y) - 0.02), min(1.0, max(all_y) + 0.03))
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.5, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ax, labels, points, segs in pending:
        place_labels(fig, ax, labels, points, segs, base)
    out = Path(a.out) if a.out else sweep / f"tradeoff{a.suffix}"
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
