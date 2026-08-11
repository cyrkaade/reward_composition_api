"""Read jobs/run_hopper5m.sh: our Hopper preset vs SB3 out-of-the-box.

Answers, per arm, from eval/evaluations.npz (one point every 50K steps):

    value at 1M      what the earlier 3-seed / 1M validation could see
    peak + WHEN      the timestep matters: a peak at 4.9M is still climbing,
                     a peak at 1.5M followed by 3.5M of decline is a collapse
    final            what every experiment in this project actually scores
    drawdown         (peak - final) / peak

Medians, never means, and the per-seed table underneath - Hopper is bimodal
(some seeds park on the survive-only optimum at ~1000 and never leave), so a
mean is a blend of two populations that no single seed represents.

Usage:
    python jobs/analyze_hopper5m.py
    python jobs/analyze_hopper5m.py --logs logs --at 1000000 2000000 5000000
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

# label -> log directory. The archived rows are references, not the comparison:
# they ran at n_envs=8 and are marked so nobody reads the gap as hyperparameters.
ARMS = [
    ("ours (preset)", "hop5m_ours", "n_envs 1, 5M"),
    ("stock SB3", "hop5m_stock", "n_envs 1, 5M"),
    ("ours + partial", "hop5m_ours_partial", "n_envs 1, 5M"),
    ("stock + partial", "hop5m_stock_partial", "n_envs 1, 5M"),
    ("[archive] stock", "base_hopper_true", "n_envs 8, 5M - reference only"),
    ("[archive] stock partial", "base_hopper_partial", "n_envs 8, 5M - reference only"),
    ("[archive] rl-zoo", "e0t_hopper_true", "n_envs 8, 2M - never learns"),
]


def load_curve(run_dir: Path):
    """(timesteps, mean true reward per eval) for one run, or None."""
    npz = run_dir / "eval" / "evaluations.npz"
    if not npz.exists():
        return None
    with np.load(npz) as data:
        steps = np.asarray(data["timesteps"], dtype=np.int64)
        results = np.asarray(data["results"], dtype=np.float64)
    if steps.size == 0:
        return None
    return steps, results.mean(axis=1)


def value_at(steps, curve, target: int):
    """The evaluation at or immediately before `target`; None if training
    stopped earlier. Not interpolated - these are measurements, not a model."""
    usable = steps <= target
    if not usable.any():
        return None
    return float(curve[np.flatnonzero(usable)[-1]])


def summarize(run_dir: Path, checkpoints: list[int]):
    loaded = load_curve(run_dir)
    if loaded is None:
        return None
    steps, curve = loaded
    peak_idx = int(np.argmax(curve))
    final = float(curve[-1])
    peak = float(curve[peak_idx])
    return {
        "seed": run_dir.name,
        "at": [value_at(steps, curve, c) for c in checkpoints],
        "peak": peak,
        "peak_step": int(steps[peak_idx]),
        "final": final,
        "last_step": int(steps[-1]),
        "drawdown": (peak - final) / peak * 100.0 if peak > 0 else float("nan"),
    }


def median(values):
    clean = [v for v in values if v is not None]
    return st.median(clean) if clean else None


def fmt(value, width=8, digits=0):
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width},.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs", help="root of the log directories")
    parser.add_argument(
        "--at",
        type=int,
        nargs="+",
        default=[1_000_000, 2_000_000, 5_000_000],
        help="checkpoints to read the curve at",
    )
    args = parser.parse_args()
    root = Path(args.logs)
    checkpoints = args.at

    collected: dict[str, list[dict]] = {}

    for label, dirname, note in ARMS:
        arm_dir = root / dirname
        if not arm_dir.is_dir():
            continue
        runs = sorted(p for p in arm_dir.iterdir() if p.is_dir())
        rows = [r for r in (summarize(p, checkpoints) for p in runs) if r]
        if not rows:
            continue
        collected[label] = rows

        print(f"\n=== {label}  ({note})  n={len(rows)}")
        head = "".join(f"{f'@{c / 1e6:g}M':>10}" for c in checkpoints)
        print(f"   {'run':>26}{head}{'peak':>9}{'at step':>11}{'final':>9}{'drawdown':>10}")
        for row in rows:
            cells = "".join(fmt(v, 10) for v in row["at"])
            print(
                f"   {row['seed'][:26]:>26}{cells}"
                f"{fmt(row['peak'], 9)}{row['peak_step']:>11,}"
                f"{fmt(row['final'], 9)}{fmt(row['drawdown'], 9, 1)}%"
            )
        med_cells = "".join(fmt(median([r["at"][i] for r in rows]), 10) for i in range(len(checkpoints)))
        print(
            f"   {'MEDIAN':>26}{med_cells}"
            f"{fmt(median([r['peak'] for r in rows]), 9)}"
            f"{median([r['peak_step'] for r in rows]):>11,.0f}"
            f"{fmt(median([r['final'] for r in rows]), 9)}"
            f"{fmt(median([r['drawdown'] for r in rows]), 9, 1)}%"
        )
        stuck = sum(1 for r in rows if r["peak"] < 1200)
        if stuck:
            print(f"   NOTE: {stuck}/{len(rows)} seeds never left the survive-only optimum (peak < 1200)")

    if "ours (preset)" in collected and "stock SB3" in collected:
        ours, stock = collected["ours (preset)"], collected["stock SB3"]
        print("\n=== HEAD TO HEAD (medians, equal n_envs and budget)")
        for field, digits in (("final", 0), ("peak", 0), ("drawdown", 1)):
            a, b = median([r[field] for r in ours]), median([r[field] for r in stock])
            gap = a - b
            print(f"   {field:>10}   ours {a:>9,.{digits}f}   stock {b:>9,.{digits}f}   gap {gap:>+9,.{digits}f}")
        for index, checkpoint in enumerate(checkpoints):
            a = median([r["at"][index] for r in ours])
            b = median([r["at"][index] for r in stock])
            if a is None or b is None:
                continue
            print(f"   @{checkpoint / 1e6:>8g}M   ours {a:>9,.0f}   stock {b:>9,.0f}   gap {a - b:>+9,.0f}")
        print(
            "\n   Exact one-sided Wilcoxon on 10 paired seeds bottoms out at p=0.00098;"
            "\n   with unpaired arms use Mann-Whitney. A gap inside the seed spread is not a result."
        )

    if not collected:
        print(f"no runs found under {root}/ - expected e.g. {root}/hop5m_ours/")


if __name__ == "__main__":
    main()
