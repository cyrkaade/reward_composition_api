#!/usr/bin/env python
"""Rank the Hopper-v5 hyperparameter sweep (jobs/run_hpo_hopper.sh).

Ranks by median FINAL, not median peak. Every experiment in this project runs
with --final-policy last, and Hopper's failure mode is collapsing after it
peaks, so a config that reaches 3500 and ends at 800 is worse than useless here.
Peak and drawdown are printed alongside so the gap stays visible.

`base` is the config currently in rcomp/ppo_presets.py. A variant only earns a
change if it beats base on median final by more than the seed spread - with 3
seeds nothing here is significant on its own, so treat this as a screen that
tells you which knobs to grid properly in stage 2, not as a result.

Usage (from api/):
    python jobs/analyze_hpo_hopper.py
    python jobs/analyze_hpo_hopper.py --root logs --per-seed
"""

from __future__ import annotations

import argparse
import glob
import statistics as st
from pathlib import Path

import numpy as np

# What each variant changes relative to `base`, for the printed table.
CHANGES = {
    "base": "(the current preset)",
    "lr1e5": "learning_rate 3e-5 -> 1e-5",
    "lr1e4": "learning_rate 3e-5 -> 1e-4",
    "lr3e4": "learning_rate 3e-5 -> 3e-4",
    "ep5": "n_epochs 20 -> 5",
    "ep10": "n_epochs 20 -> 10",
    "clip02": "clip_range 0.4 -> 0.2",
    "novfclip": "clip_range_vf 0.5 -> off",
    "gam999": "gamma 0.99 -> 0.999",
    "gae95": "gae_lambda 0.9 -> 0.95",
    "logstd_m1": "log_std_init 0 -> -1",
    "ent001": "ent_coef 0 -> 0.001",
    "steps1024": "n_steps 512 -> 1024",
    "batch64": "batch_size 32 -> 64",
    "nenv8": "n_envs 1 -> 8",
    "relu": "activation Tanh -> ReLU",
}


def collect(cell: Path) -> list[tuple[str, float, float]]:
    """(run_name, final, peak) per seed."""
    rows = []
    for curve in sorted(cell.glob("*/eval/evaluations.npz")):
        data = np.load(curve)
        rewards = data["results"].mean(axis=1)
        rows.append((curve.parent.parent.name, float(rewards[-1]), float(rewards.max())))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs", help="Directory holding the hpo_hopper_* cells")
    parser.add_argument("--per-seed", action="store_true", help="Also print every seed")
    args = parser.parse_args()

    cells = sorted(Path(p) for p in glob.glob(f"{args.root}/hpo_hopper_*"))
    if not cells:
        print(f"no hpo_hopper_* cells under {args.root}/ - has the job run and been fetched?")
        return 1

    results = []
    for cell in cells:
        rows = collect(cell)
        if not rows:
            print(f"{cell.name}: no completed runs")
            continue
        variant = cell.name.replace("hpo_hopper_", "")
        finals = [r[1] for r in rows]
        peaks = [r[2] for r in rows]
        drops = [1 - f / p for f, p in zip(finals, peaks) if p > 0]
        results.append({
            "variant": variant, "n": len(rows), "rows": rows,
            "final": st.median(finals), "peak": st.median(peaks),
            "worst": min(finals), "drop": st.median(drops) if drops else float("nan"),
        })

    baseline = next((r for r in results if r["variant"] == "base"), None)
    results.sort(key=lambda r: r["final"], reverse=True)

    print(f"{'variant':11s} {'n':>2s} {'final':>8s} {'peak':>8s} {'drop':>6s} {'worst':>8s} "
          f"{'vs base':>9s}  what it changes")
    print("-" * 100)
    for r in results:
        delta = f"{r['final'] - baseline['final']:+9.0f}" if baseline else "        -"
        mark = " <-- current" if r["variant"] == "base" else ""
        print(f"{r['variant']:11s} {r['n']:2d} {r['final']:8.0f} {r['peak']:8.0f} "
              f"{100 * r['drop']:5.0f}% {r['worst']:8.0f} {delta}  "
              f"{CHANGES.get(r['variant'], '?')}{mark}")
        if args.per_seed:
            for name, final, peak in r["rows"]:
                print(f"            {name:26s} peak {peak:8.0f} -> final {final:8.0f}")

    if baseline:
        better = [r for r in results if r["final"] > baseline["final"] and r["variant"] != "base"]
        print()
        if not better:
            print("Nothing beats the current preset on median final. Keep it.")
        else:
            names = ", ".join(f"{r['variant']} ({r['final'] - baseline['final']:+.0f})" for r in better[:4])
            print(f"Beats base on median final: {names}")
            print("With 3 seeds none of this is significant. Grid the winning knobs at "
                  "5+ seeds (stage 2) before changing rcomp/ppo_presets.py.")
        print("\nReminder: Hopper collapses under every config tried so far (base drops "
              f"{100 * baseline['drop']:.0f}%). If nothing here gets the drawdown near "
              "LunarLander's 3%, Hopper stays out of the overoptimization claim regardless "
              "of which config wins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
