"""Read back the weighted-sum alpha sweep and draw the two-column figure.

Layout: one row per environment, one column per query budget.  Inside a panel
every arm is a line over policy timesteps -- the median across seeds of the
evaluation curve, with the inter-quartile band shaded.  true is black, vanilla
is green, naive is orange, and the four alphas run along a blue-to-orange ramp.

Usage:
    python jobs/analyze_wsum.py [--logs logs] [--out logs/wsum]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CELLS = ["ll", "ant", "hopper", "reacher", "bipedal", "pusher", "swimmer", "walker",
         "mspacman", "qbert", "breakout", "pong"]
CELL_TITLES = {
    "ll": "LunarLander-v3",
    "ant": "Ant-v5",
    "hopper": "Hopper-v5",
    "reacher": "Reacher-v5",
    "bipedal": "BipedalWalker-v3",
    "pusher": "Pusher-v5",
    "swimmer": "Swimmer-v5",
    "walker": "Walker2d-v5",
    "mspacman": "MsPacman",
    "qbert": "Qbert",
    "breakout": "Breakout",
    "pong": "Pong",
}
ALPHAS = ["020", "040", "050", "060", "080"]
ALPHA_LABEL = {
    "020": "alpha=0.2",
    "040": "alpha=0.4",
    "050": "alpha=0.5",
    "060": "alpha=0.6",
    "080": "alpha=0.8",
}
# blue -> orange ramp, so the eye reads increasing weight on the prior
ALPHA_COLOR = {
    "020": "#4C72B0",
    "040": "#7B68AE",
    "050": "#9B59B6",
    "060": "#C44E52",
    "080": "#D95F02",
}
BASE_STYLE = {
    "true": ("true reward", "black", "-", 2.2),
    "vanilla": ("vanilla RLHF", "#2CA02C", "-", 1.8),
    "naive": ("naive (partial+model)", "#FF7F0E", "--", 1.6),
}


def load_runs(root: Path, prefix: str = "wsum"):
    """run dirs are logs/<prefix>_<cell>_<arm>/<prefix>_<cell>_<arm>_seed<n>/

    The Atari half of the grid was submitted under the `wsuma` prefix, so the
    prefix is a parameter rather than a literal.  Globbing "wsum_*" would not
    pick those up: the fifth character is 'a', not '_'.
    """
    runs = []
    for meta_path in sorted(root.glob(f"{prefix}_*/*/metadata.json")):
        cell_arm = meta_path.parent.parent.name[len(prefix) + 1:]
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        cell = next((c for c in CELLS if cell_arm == c or cell_arm.startswith(c + "_")), None)
        if cell is None:
            continue
        arm = cell_arm[len(cell) + 1:] or "true"
        curve, steps = None, None
        npz = meta_path.parent / "eval" / "evaluations.npz"
        if npz.exists():
            try:
                data = np.load(npz)
                steps = np.asarray(data["timesteps"], dtype=float)
                curve = np.asarray(data["results"], dtype=float).mean(axis=1)
            except Exception:
                curve, steps = None, None
        runs.append({
            "cell": cell,
            "arm": arm,
            "seed": meta.get("seed"),
            "final": meta.get("selected_policy_true_reward_mean"),
            "peak": meta.get("best_logged_true_reward"),
            "queries": meta.get("synthetic_queries"),
            "budget": meta.get("query_budget") or 0,
            "alpha": meta.get("partial_alpha"),
            "mode": meta.get("mode"),
            "steps": steps,
            "curve": curve,
            "tail5": float(np.mean(curve[-5:])) if curve is not None and len(curve) else None,
        })
    return runs


def split_arm(arm: str):
    """'ws060_q400' -> ('ws060', 400); 'true' -> ('true', None)"""
    if arm == "true":
        return "true", None
    base, _, budget = arm.rpartition("_q")
    return base, (int(budget) if budget.isdigit() else None)


def band(curves_steps):
    """median curve + IQR across seeds on the shortest common grid"""
    usable = [(s, c) for s, c in curves_steps if s is not None and c is not None and len(c)]
    if not usable:
        return None
    n = min(len(c) for _, c in usable)
    steps = usable[0][0][:n]
    stack = np.vstack([c[:n] for _, c in usable])
    return (
        steps,
        np.median(stack, axis=0),
        np.percentile(stack, 25, axis=0),
        np.percentile(stack, 75, axis=0),
        stack.shape[0],
    )


def paired_delta(left, right):
    """median of (left - right) over the seeds both arms share, and the win count"""
    lmap = {r["seed"]: r["tail5"] for r in left if r["tail5"] is not None}
    rmap = {r["seed"]: r["tail5"] for r in right if r["tail5"] is not None}
    shared = sorted(set(lmap) & set(rmap))
    if not shared:
        return None
    diffs = [lmap[s] - rmap[s] for s in shared]
    return float(np.median(diffs)), sum(d > 0 for d in diffs), len(shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="logs/wsum")
    ap.add_argument(
        "--prefix",
        default="wsum",
        help="log-dir prefix; use wsuma for the Atari half of the grid",
    )
    ap.add_argument(
        "--complete-seeds-only",
        action="store_true",
        help="keep only seeds for which EVERY arm of a cell has finished, so all "
             "arms in a panel are compared on the same seeds rather than at different n",
    )
    args = ap.parse_args()

    root = Path(args.logs)
    runs = load_runs(root, args.prefix)
    if not runs:
        raise SystemExit(f"no wsum runs found under {root}")

    if args.complete_seeds_only:
        # A seed counts as complete for a cell when every arm of that cell has a
        # finished run for it. Mid-flight the grid is ragged, and comparing a
        # 9-seed median against a 4-seed one is not a comparison.
        by_cell_arm = defaultdict(set)
        for r in runs:
            by_cell_arm[r["cell"]].add(r["arm"])
        seeds_by_cell_arm = defaultdict(set)
        for r in runs:
            seeds_by_cell_arm[(r["cell"], r["arm"])].add(r["seed"])
        keep = {}
        for cell, arms_present in by_cell_arm.items():
            common = None
            for arm in arms_present:
                seeds = seeds_by_cell_arm[(cell, arm)]
                common = seeds if common is None else (common & seeds)
            keep[cell] = common or set()
            print(f"{cell}: {len(arms_present)} arms, complete seeds {sorted(keep[cell])}")
        before = len(runs)
        runs = [r for r in runs if r["seed"] in keep.get(r["cell"], set())]
        print(f"complete-seeds filter: {before} -> {len(runs)} runs")
        if not runs:
            raise SystemExit("no seed is complete across every arm yet")

    grouped = defaultdict(list)
    for r in runs:
        base, budget = split_arm(r["arm"])
        grouped[(r["cell"], base, budget)].append(r)

    cells = [c for c in CELLS if any(k[0] == c for k in grouped)]
    budgets = sorted({k[2] for k in grouped if k[2] is not None})
    print(f"loaded {len(runs)} runs | cells={cells} | budgets={budgets}")
    if not budgets:
        # only the label-free ceiling has landed so far; still draw it so the
        # partial grid is inspectable mid-flight
        budgets = [0]
        print("no budgeted arm has finished yet; plotting the true arm alone")

    order = ["true", "vanilla", "naive"] + [f"ws{a}" for a in ALPHAS]

    print(
        f"\n{'cell':9s} {'budget':>6s} {'arm':9s} {'n':>3s} {'tail5':>9s} "
        f"{'final':>9s} {'deliv':>6s} {'vs_van':>9s} {'wins':>6s}"
    )
    report = {}
    for cell in cells:
        for budget in budgets:
            van = grouped.get((cell, "vanilla", budget), [])
            for base in order:
                key = (cell, base, None if base == "true" else budget)
                rs = grouped.get(key, [])
                if not rs:
                    continue
                tails = [r["tail5"] for r in rs if r["tail5"] is not None]
                finals = [r["final"] for r in rs if r["final"] is not None]
                deliv = [r["queries"] / r["budget"] for r in rs if r.get("budget")]
                cmp = paired_delta(rs, van) if (van and base != "vanilla") else None
                cmp_txt = f"{cmp[0]:9.1f}" if cmp else f"{'-':>9s}"
                win_txt = f"{cmp[1]}/{cmp[2]}" if cmp else "-"
                print(
                    f"{cell:9s} {budget:6d} {base:9s} {len(rs):3d} "
                    f"{np.median(tails) if tails else float('nan'):9.1f} "
                    f"{np.median(finals) if finals else float('nan'):9.1f} "
                    f"{(np.mean(deliv) if deliv else 1.0):6.2f} {cmp_txt} {win_txt:>6s}"
                )
                report[f"{cell}|{budget}|{base}"] = {
                    "n": len(rs),
                    "tail5_median": float(np.median(tails)) if tails else None,
                    "final_median": float(np.median(finals)) if finals else None,
                    "delivery": float(np.mean(deliv)) if deliv else None,
                    "vs_vanilla_median": cmp[0] if cmp else None,
                    "vs_vanilla_wins": f"{cmp[1]}/{cmp[2]}" if cmp else None,
                }

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out_base) + "_report.json").write_text(json.dumps(report, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = len(cells), max(len(budgets), 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 3.6 * nrows), squeeze=False)

    for r, cell in enumerate(cells):
        for c, budget in enumerate(budgets):
            ax = axes[r][c]
            for base, (label, color, ls, lw) in BASE_STYLE.items():
                key = (cell, base, None if base == "true" else budget)
                got = band([(x["steps"], x["curve"]) for x in grouped.get(key, [])])
                if not got:
                    continue
                steps, med, lo, hi, n = got
                ax.plot(steps, med, color=color, linestyle=ls, linewidth=lw, label=f"{label} (n={n})")
                ax.fill_between(steps, lo, hi, color=color, alpha=0.12, linewidth=0)
            for a in ALPHAS:
                got = band([(x["steps"], x["curve"]) for x in grouped.get((cell, f"ws{a}", budget), [])])
                if not got:
                    continue
                steps, med, lo, hi, n = got
                ax.plot(steps, med, color=ALPHA_COLOR[a], linewidth=1.7, label=f"weighted {ALPHA_LABEL[a]} (n={n})")
                ax.fill_between(steps, lo, hi, color=ALPHA_COLOR[a], alpha=0.10, linewidth=0)
            ax.set_title(f"{CELL_TITLES.get(cell, cell)} - {budget} queries", fontsize=11)
            ax.grid(alpha=0.25)
            ax.set_xlabel("policy timesteps")
            if c == 0:
                ax.set_ylabel("true episode return")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Normalized weighted sum: alpha*norm(partial) + (1-alpha)*norm(model)\n"
        "median over seeds, shaded IQR",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_png = str(out_base) + "_curves.png"
    fig.savefig(out_png, dpi=140)
    print(f"\nwrote {out_png}")
    print(f"wrote {out_base}_report.json")


if __name__ == "__main__":
    main()
