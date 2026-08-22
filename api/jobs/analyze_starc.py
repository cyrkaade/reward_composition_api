"""Read back the STARC alignment x alpha grid on LunarLander.

Two figures from one pass over logs/starc_*:

  *_curves.png   one panel per STARC rung, five alpha curves inside it,
                 with the shared true (black) and vanilla (green) references
                 drawn in every panel so each rung has a scale.
  *_heatmap.png  the 5x5 itself: rung on the y axis, alpha on the x axis,
                 colour and annotation = median final return across seeds.

The reported statistic is `tail5`, the mean of the last five evaluation points,
which is steadier than a single final evaluation without smuggling in the
best-of-N selection on ground truth that `--final-policy best` would.  The raw
`selected_policy_true_reward_mean` is printed next to it.

Usage:
    python jobs/analyze_starc.py [--logs logs] [--out logs/starc_grid]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Ladder order is fixed here, not inferred, so a missing rung leaves a visible
# hole in the figure instead of silently renumbering the axis.
RUNGS = ["s20", "s40", "s60", "s80", "s100"]
# Held-out measured STARC alignment (seeds 5-9), not the design target.
RUNG_ALIGN = {"s20": 0.207, "s40": 0.414, "s60": 0.614, "s80": 0.807, "s100": 1.000}
ALPHAS = ["020", "040", "050", "060", "080"]
ALPHA_VALUE = {a: int(a) / 100.0 for a in ALPHAS}
ALPHA_COLOR = {
    "020": "#4C72B0",
    "040": "#7B68AE",
    "050": "#9B59B6",
    "060": "#C44E52",
    "080": "#D95F02",
}
REF_STYLE = {
    "true": ("true reward", "black", "-", 2.2),
    "vanilla": ("vanilla RLHF (no prior)", "#2CA02C", "-", 1.8),
}


def load_runs(root: Path):
    """run dirs are logs/starc_<cell>_<arm>/starc_<cell>_<arm>_seed<n>/"""
    runs = []
    for meta_path in sorted(root.glob("starc_*/*/metadata.json")):
        cell_arm = meta_path.parent.parent.name[len("starc_"):]
        cell, _, arm = cell_arm.partition("_")
        if not arm:
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        steps = curve = None
        npz = meta_path.parent / "eval" / "evaluations.npz"
        if npz.exists():
            try:
                data = np.load(npz)
                steps = np.asarray(data["timesteps"], dtype=float)
                curve = np.asarray(data["results"], dtype=float).mean(axis=1)
            except Exception:
                steps = curve = None
        runs.append({
            "cell": cell,
            "arm": arm,
            "base": arm.rpartition("_q")[0] or arm,
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


def band(curves_steps):
    """median curve + IQR across seeds on the shortest common grid"""
    usable = [(s, c) for s, c in curves_steps if s is not None and c is not None and len(c)]
    if not usable:
        return None
    n = min(len(c) for _, c in usable)
    stack = np.vstack([c[:n] for _, c in usable])
    return (usable[0][0][:n], np.median(stack, axis=0),
            np.percentile(stack, 25, axis=0), np.percentile(stack, 75, axis=0),
            stack.shape[0])


def paired_delta(left, right):
    """median of (left - right) over the seeds both arms share, plus the win count"""
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
    ap.add_argument("--out", default="logs/starc_grid")
    ap.add_argument(
        "--complete-seeds-only",
        action="store_true",
        help="keep only seeds finished by EVERY arm in the grid, so every cell "
             "of the 5x5 is a median over the same seeds",
    )
    args = ap.parse_args()

    root = Path(args.logs)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no starc runs found under {root}")

    if args.complete_seeds_only:
        seeds_by_arm = defaultdict(set)
        for r in runs:
            seeds_by_arm[(r["cell"], r["arm"])].add(r["seed"])
        common = None
        for seeds in seeds_by_arm.values():
            common = seeds if common is None else (common & seeds)
        common = common or set()
        before = len(runs)
        runs = [r for r in runs if r["seed"] in common]
        print(f"complete-seeds filter: {before} -> {len(runs)} runs, seeds {sorted(common)}")
        if not runs:
            raise SystemExit("no seed is complete across every arm yet")

    grouped = defaultdict(list)
    for r in runs:
        grouped[(r["cell"], r["base"])].append(r)

    refs = {name: grouped.get(("ref", name), []) for name in ("true", "vanilla")}
    van = refs["vanilla"]

    print(f"loaded {len(runs)} runs")
    print(f"\n{'rung':6s} {'align':>6s} {'arm':7s} {'n':>3s} {'tail5':>9s} {'final':>9s} "
          f"{'peak':>9s} {'deliv':>6s} {'vs_van':>9s} {'wins':>7s}")

    report, table = {}, np.full((len(RUNGS), len(ALPHAS)), np.nan)
    for name, rs in refs.items():
        if not rs:
            continue
        tails = [r["tail5"] for r in rs if r["tail5"] is not None]
        finals = [r["final"] for r in rs if r["final"] is not None]
        peaks = [r["peak"] for r in rs if r["peak"] is not None]
        deliv = [r["queries"] / r["budget"] for r in rs if r.get("budget")]
        print(f"{'ref':6s} {'-':>6s} {name:7s} {len(rs):3d} "
              f"{np.median(tails) if tails else np.nan:9.1f} "
              f"{np.median(finals) if finals else np.nan:9.1f} "
              f"{np.median(peaks) if peaks else np.nan:9.1f} "
              f"{(np.mean(deliv) if deliv else 1.0):6.2f} {'-':>9s} {'-':>7s}")
        report[f"ref|{name}"] = {
            "n": len(rs),
            "tail5_median": float(np.median(tails)) if tails else None,
            "final_median": float(np.median(finals)) if finals else None,
        }

    for i, rung in enumerate(RUNGS):
        for j, a in enumerate(ALPHAS):
            rs = grouped.get((rung, f"ws{a}"), [])
            if not rs:
                continue
            tails = [r["tail5"] for r in rs if r["tail5"] is not None]
            finals = [r["final"] for r in rs if r["final"] is not None]
            peaks = [r["peak"] for r in rs if r["peak"] is not None]
            deliv = [r["queries"] / r["budget"] for r in rs if r.get("budget")]
            cmp = paired_delta(rs, van) if van else None
            if tails:
                table[i, j] = float(np.median(tails))
            print(f"{rung:6s} {RUNG_ALIGN[rung]:6.3f} {'ws' + a:7s} {len(rs):3d} "
                  f"{np.median(tails) if tails else np.nan:9.1f} "
                  f"{np.median(finals) if finals else np.nan:9.1f} "
                  f"{np.median(peaks) if peaks else np.nan:9.1f} "
                  f"{(np.mean(deliv) if deliv else 1.0):6.2f} "
                  f"{cmp[0] if cmp else np.nan:9.1f} "
                  f"{(f'{cmp[1]}/{cmp[2]}' if cmp else '-'):>7s}")
            report[f"{rung}|ws{a}"] = {
                "n": len(rs),
                "alignment": RUNG_ALIGN[rung],
                "alpha": ALPHA_VALUE[a],
                "tail5_median": float(np.median(tails)) if tails else None,
                "final_median": float(np.median(finals)) if finals else None,
                "peak_median": float(np.median(peaks)) if peaks else None,
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

    # ---- curves: one panel per rung -------------------------------------
    present = [r for r in RUNGS if any(k[0] == r for k in grouped)]
    if present:
        ncols = min(3, len(present))
        nrows = int(np.ceil(len(present) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.8 * nrows), squeeze=False)
        for idx, rung in enumerate(present):
            ax = axes[idx // ncols][idx % ncols]
            for name, (label, color, ls, lw) in REF_STYLE.items():
                got = band([(x["steps"], x["curve"]) for x in refs.get(name, [])])
                if not got:
                    continue
                steps, med, lo, hi, n = got
                ax.plot(steps, med, color=color, linestyle=ls, linewidth=lw, label=f"{label} (n={n})")
                ax.fill_between(steps, lo, hi, color=color, alpha=0.12, linewidth=0)
            for a in ALPHAS:
                got = band([(x["steps"], x["curve"]) for x in grouped.get((rung, f"ws{a}"), [])])
                if not got:
                    continue
                steps, med, lo, hi, n = got
                ax.plot(steps, med, color=ALPHA_COLOR[a], linewidth=1.7,
                        label=f"alpha={ALPHA_VALUE[a]:.1f} (n={n})")
                ax.fill_between(steps, lo, hi, color=ALPHA_COLOR[a], alpha=0.10, linewidth=0)
            ax.set_title(f"prior {rung}  (alignment {RUNG_ALIGN[rung]:.3f})", fontsize=11)
            ax.grid(alpha=0.25)
            ax.set_xlabel("policy timesteps")
            if idx % ncols == 0:
                ax.set_ylabel("true episode return")
            if idx == 0:
                ax.legend(fontsize=7, loc="best")
        for idx in range(len(present), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        fig.suptitle(
            "LunarLander-v3, q400, 1M steps: prior alignment x weighted-sum alpha\n"
            "median over seeds, shaded IQR",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(str(out_base) + "_curves.png", dpi=140)
        print(f"\nwrote {out_base}_curves.png")

    # ---- the 5x5 itself --------------------------------------------------
    if np.isfinite(table).any():
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        im = ax.imshow(table, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(ALPHAS)), [f"{ALPHA_VALUE[a]:.1f}" for a in ALPHAS])
        ax.set_yticks(range(len(RUNGS)), [f"{r}\n({RUNG_ALIGN[r]:.2f})" for r in RUNGS])
        ax.set_xlabel("alpha  (weight on the prior)")
        ax.set_ylabel("prior alignment rung")
        for i in range(len(RUNGS)):
            for j in range(len(ALPHAS)):
                if np.isfinite(table[i, j]):
                    lo, hi = np.nanmin(table), np.nanmax(table)
                    mid = lo + 0.55 * (hi - lo)
                    ax.text(j, i, f"{table[i, j]:.0f}", ha="center", va="center",
                            fontsize=10, color="white" if table[i, j] < mid else "black")
        fig.colorbar(im, ax=ax, label="median true return (tail5)")
        sub = []
        for name, rs in refs.items():
            tails = [r["tail5"] for r in rs if r["tail5"] is not None]
            if tails:
                sub.append(f"{name} {np.median(tails):.0f}")
        ax.set_title("LunarLander-v3, q400, 1M steps"
                     + (f"\nreference: {', '.join(sub)}" if sub else ""), fontsize=12)
        fig.tight_layout()
        fig.savefig(str(out_base) + "_heatmap.png", dpi=140)
        print(f"wrote {out_base}_heatmap.png")

    print(f"wrote {out_base}_report.json")


if __name__ == "__main__":
    main()
