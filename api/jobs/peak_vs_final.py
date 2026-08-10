#!/usr/bin/env python
"""Report median PEAK next to median FINAL for each run directory.

The endpoint alone is what made the Hopper/Walker "partial beats true" result
look real: PPO on the true Hopper reward reaches a better policy than the
partial and then loses it, and `--final-policy last` measures it at its worst
moment. Reading peak and final side by side makes that visible in one line.

Usage (from api/):
    python jobs/peak_vs_final.py logs/e0t_hopper_true logs/e0t_hopper_partial
    python jobs/peak_vs_final.py 'logs/e0t_*'
    python jobs/peak_vs_final.py --compare logs/e0t_hopper_true logs/e0t_hopper_partial

Peak comes from each run's eval curve (eval/evaluations.npz) when present, else
from metadata's best_logged_true_reward; final always comes from the last eval
point, which matches selected_policy_true_reward_mean under --final-policy last.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import numpy as np


def _runs(cell: Path) -> list[tuple[str, float, float, float]]:
    """(run_name, final, peak, peak_timestep) per run under `cell`."""
    rows = []
    for meta_path in sorted(cell.glob("*/metadata.json")):
        run_dir = meta_path.parent
        meta = json.loads(meta_path.read_text())
        curve = run_dir / "eval" / "evaluations.npz"
        if curve.exists():
            data = np.load(curve)
            rewards = data["results"].mean(axis=1)
            timesteps = data["timesteps"]
            final = float(rewards[-1])
            best = int(np.argmax(rewards))
            peak, peak_at = float(rewards[best]), float(timesteps[best])
        else:
            final = meta.get("selected_policy_true_reward_mean")
            peak = meta.get("best_logged_true_reward")
            peak_at = meta.get("best_logged_timestep")
            if final is None or peak is None:
                continue
        rows.append((run_dir.name, final, float(peak), float(peak_at)))
    return rows


def _summarize(cell: Path, per_seed: bool) -> dict | None:
    rows = _runs(cell)
    if not rows:
        print(f"{cell}: no completed runs")
        return None

    finals = [r[1] for r in rows]
    peaks = [r[2] for r in rows]
    # Drawdown is only meaningful when the scale is positive; Reacher/Pusher
    # rewards are negative and "closer to 0 is better", so a ratio is nonsense.
    drops = [1 - f / p for f, p in zip(finals, peaks) if p > 0]
    over_20 = sum(1 for d in drops if d > 0.2)

    dd = f"{100 * statistics.median(drops):5.1f}%" if drops else "  n/a"
    frac = f"{over_20}/{len(drops)}" if drops else "n/a"
    print(
        f"{cell.name:30s} n={len(rows):3d}  final={statistics.median(finals):9.1f}  "
        f"peak={statistics.median(peaks):9.1f}  drawdown={dd}  >20%:{frac:>7s}  "
        f"peak@={statistics.median([r[3] for r in rows]) / 1e6:.2f}M"
    )
    if per_seed:
        for name, final, peak, peak_at in rows:
            drop = f"{100 * (1 - final / peak):5.1f}%" if peak > 0 else "  n/a"
            print(f"    {name:34s} peak {peak:9.1f} @ {peak_at / 1e6:5.2f}M -> final {final:9.1f}  drop {drop}")
    return {"finals": finals, "peaks": peaks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells", nargs="+", help="Run directories or globs, e.g. logs/e0t_hopper_true")
    parser.add_argument("--per-seed", action="store_true", help="Also print every seed")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Mann-Whitney the first two cells on final and on peak (needs scipy)",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.cells:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        paths.extend(matches or [Path(pattern)])

    summaries = [(path, _summarize(path, args.per_seed)) for path in paths]

    if args.compare:
        usable = [(path, data) for path, data in summaries if data]
        if len(usable) < 2:
            print("\n--compare needs two cells with completed runs")
            return 1
        from scipy import stats

        (name_a, a), (name_b, b) = usable[0], usable[1]
        print(f"\n{name_a.name} vs {name_b.name}")
        for label, key in (("FINAL", "finals"), ("PEAK", "peaks")):
            first, second = a[key], b[key]
            pvalue = stats.mannwhitneyu(first, second, alternative="two-sided").pvalue
            diff = statistics.median(first) - statistics.median(second)
            print(f"  {label:6s} diff={diff:+9.1f}   Mann-Whitney p={pvalue:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
