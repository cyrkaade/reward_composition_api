"""Readable summary of the Atari half of the weighted-sum grid.

analyze_wsum's default figure does not work for Atari: with 10 eval episodes on
a high-variance game the per-eval curve swings by a factor of three between
adjacent points, and seven overlapping IQR bands turn the panel into mush.  So
here the curves are smoothed and the bands dropped, and a third column shows the
statistic the conclusions actually rest on -- tail5 per seed -- as raw points.

Usage:
    python jobs/plot_atari_summary.py [--logs logs] [--out logs/atari_summary.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_wsum import load_runs, split_arm  # noqa: E402

CELLS = [(c, t) for c, t in [("qbert", "Qbert"), ("mspacman", "MsPacman"),
                             ("breakout", "Breakout"), ("pong", "Pong")]]
ARMS = ["true", "vanilla", "naive", "ws020", "ws040", "ws060", "ws080"]
LABEL = {
    "true": "true reward", "vanilla": "vanilla RLHF", "naive": "naive (partial+model)",
    "ws020": "weighted a=0.2", "ws040": "weighted a=0.4",
    "ws060": "weighted a=0.6", "ws080": "weighted a=0.8",
}
COLOR = {
    "true": "black", "vanilla": "#2CA02C", "naive": "#FF7F0E",
    "ws020": "#4C72B0", "ws040": "#7B68AE", "ws060": "#C44E52", "ws080": "#D95F02",
}
STYLE = {"true": (2.4, "-"), "vanilla": (2.0, "-"), "naive": (1.8, "--")}
BUDGETS = [2800, 5600]
WINDOW = 9  # eval points; 40 per run, so this is a ~22% span


def smooth(y, window=WINDOW):
    """Centred rolling mean that keeps the endpoints instead of trimming them."""
    out = np.empty_like(y, dtype=float)
    half = window // 2
    for i in range(len(y)):
        out[i] = y[max(0, i - half): i + half + 1].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="logs/atari_summary.png")
    args = ap.parse_args()

    runs = load_runs(Path(args.logs), "wsuma")
    grouped = {}
    for r in runs:
        base, budget = split_arm(r["arm"])
        grouped.setdefault((r["cell"], budget, base), []).append(r)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9),
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})

    for row, (cell, title) in enumerate(CELLS):
        for col, budget in enumerate(BUDGETS):
            ax = axes[row][col]
            for arm in ARMS:
                # the true arm has no budget, so it is drawn in both columns
                key = (cell, None if arm == "true" else budget, arm)
                rs = grouped.get(key)
                if not rs:
                    continue
                usable = [(r["steps"], r["curve"]) for r in rs
                          if r["steps"] is not None and r["curve"] is not None]
                if not usable:
                    continue
                n = min(len(c) for _, c in usable)
                steps = usable[0][0][:n]
                med = np.median(np.vstack([c[:n] for _, c in usable]), axis=0)
                lw, ls = STYLE.get(arm, (1.7, "-"))
                ax.plot(steps, smooth(med), color=COLOR[arm], linewidth=lw,
                        linestyle=ls, label=LABEL[arm])
            ax.set_title(f"{title} - {budget} queries", fontsize=12)
            ax.set_xlabel("policy timesteps")
            if col == 0:
                ax.set_ylabel("true episode return")
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=8.5, loc="upper left")

        # third column: the statistic the conclusions rest on, per seed
        ax = axes[row][2]
        offset = {2800: -0.16, 5600: 0.16}
        marker = {2800: "o", 5600: "s"}
        for y, arm in enumerate(ARMS):
            # true has no query budget, so it gets one centred row rather than
            # the same five seeds drawn twice under two different markers
            budgets = [None] if arm == "true" else BUDGETS
            for budget in budgets:
                rs = grouped.get((cell, budget, arm))
                if not rs:
                    continue
                vals = [r["tail5"] for r in rs if r["tail5"] is not None]
                if not vals:
                    continue
                pos = y + offset.get(budget, 0.0)
                ax.scatter(vals, [pos] * len(vals), s=34,
                           marker=marker.get(budget, "D"),
                           color=COLOR[arm], alpha=0.55,
                           edgecolors="none", zorder=3)
                ax.plot([np.median(vals)] * 2, [pos - 0.13, pos + 0.13],
                        color=COLOR[arm], linewidth=2.6, zorder=4)
        ax.set_yticks(range(len(ARMS)), [LABEL[a] for a in ARMS], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("final return (mean of last 5 evals)")
        ax.set_title(f"{title} - every seed's final   o = 2800,  s = 5600", fontsize=12)
        ax.grid(alpha=0.25, axis="x")

    fig.suptitle(
        "Atari weighted sum:  alpha*norm(partial) + (1-alpha)*norm(model)\n"
        f"curves are the across-seed median, smoothed over {WINDOW} eval points; "
        "right column is every seed's final, bar = median",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = Path(args.out)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
