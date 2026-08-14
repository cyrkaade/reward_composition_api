#!/usr/bin/env python
"""Read the stability pilot (jobs/run_stab.sh).

    python jobs/analyze_stab.py
    python jobs/analyze_stab.py --only walker

Uses scipy's EXACT two-sided Wilcoxon. jobs/analyze_pre4.py used a normal
approximation with no continuity or tie correction and its output was quoted in
notes as "exact"; at n=10 what it prints as 0.005 is exactly 0.002. Do not mix
the two families of numbers in one table.

WHAT TO READ, AND IN WHICH ORDER
--------------------------------
1. PEAK TIMESTEP. If an arm's peak sits at the first eval point it never
   learned, and its "peak - final drawdown" is decay from an untrained policy,
   not the loss of a policy that once worked. Ant's true arm did exactly this in
   the 1M selection grid (peak 273-461 at 20k steps, final -14 to -55). Read
   this column before reading any contrast on that arm.

2. THE PREMISE, per treatment: does `true` beat its own `partial` arm? A fix
   that raises both equally has not repaired anything this project needs. In the
   1M selection grid the premise failed in all three cells - HalfCheetah true
   1326 against shc_forward 1586 and vanilla 2058, Walker2d true 1398 against
   swk_cap15 2542 in 5/5 seeds, Ant negative in both arms.

3. THE CONTRAST, treatment - ctrl, paired by seed, on FINAL and on PEAK
   together. They can disagree in sign, and on Walker2d the whole question is
   whether a fix converts peak into final rather than raising peak.

4. DRAWDOWN and the seed spread. A fix that raises the median while leaving
   three seeds at half their peak has not made the ceiling trustworthy.

5. For `noterm` only, MODIFIED-ENV SCORE next to the standard one. Every arm's
   headline number is the standard environment, the same yardstick as every
   other arm in the project; the modified-env column shows how much the policy
   gained under the objective it actually optimized, so the size of the
   train/eval mismatch is stated rather than left unmeasured.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover - analysis-only dependency
    wilcoxon = None

CELLS = ["cheetah", "walker", "ant"]
ENV_NAME = {"cheetah": "HalfCheetah-v5", "walker": "Walker2d-v5", "ant": "Ant-v5"}
TREATMENTS = ["ctrl", "sde", "noterm"]
TREATMENT_LABEL = {
    "ctrl": "control (no fix)",
    "sde": "experiment 1: use_sde=True",
    "noterm": "experiment 2: no termination + persistent +-1",
}
# No published benchmark score is printed here on purpose. Every tuned PPO block
# and every benchmark number in circulation is keyed to -v2/-v3/-v4, and the -v5
# envs are not the same task (Ant-v5 changed both its observation space and its
# reward relative to -v4). The only valid reference for a -v5 arm is another -v5
# arm measured in this project, which is what the ctrl rows are for.


def load(root: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(root.glob("stab_*/*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # stab_<cell>_<treatment>_<arm>
        _, cell, treatment, arm = meta_path.parent.parent.name.split("_", 3)
        curve_path = meta_path.parent / "eval" / "evaluations.npz"
        if not curve_path.exists():
            continue
        curve = np.load(curve_path)
        rewards = curve["results"].mean(axis=1)
        timesteps = curve["timesteps"]
        peak_index = int(np.argmax(rewards))
        rows.append(
            {
                "cell": cell,
                "treatment": treatment,
                "arm": "true" if arm == "true" else "partial",
                "partial_name": None if arm == "true" else arm,
                "seed": meta.get("seed"),
                "final": float(meta["selected_policy_true_reward_mean"]),
                "curve_final": float(rewards[-1]),
                "peak": float(rewards[peak_index]),
                "peak_step": int(timesteps[peak_index]),
                "first_step": int(timesteps[0]),
                "first": float(rewards[0]),
                "modified": meta.get("selected_policy_modified_env_reward_mean"),
                "unhealthy_penalty": meta.get("unhealthy_penalty"),
                "plk": meta.get("policy_learning_kwargs") or {},
                "timesteps": meta.get("actual_timesteps"),
            }
        )
    return rows


def wilcoxon_p(pairs: list[tuple[float, float]]) -> float:
    """Exact two-sided Wilcoxon on a - b. nan when scipy is absent or all ties."""
    diffs = [a - b for a, b in pairs]
    if not diffs or wilcoxon is None or all(d == 0 for d in diffs):
        return float("nan")
    return float(wilcoxon(diffs, alternative="two-sided", zero_method="wilcox", method="exact").pvalue)


def paired(rows_a: list[dict], rows_b: list[dict], key: str) -> tuple[list[tuple[float, float]], list[int]]:
    """Pair by seed. Unmatched seeds are dropped and reported by the caller."""
    by_seed_b = {row["seed"]: row for row in rows_b}
    pairs, seeds = [], []
    for row in sorted(rows_a, key=lambda r: r["seed"]):
        other = by_seed_b.get(row["seed"])
        if other is not None:
            pairs.append((row[key], other[key]))
            seeds.append(row["seed"])
    return pairs, seeds


def contrast(label: str, rows_a: list[dict], rows_b: list[dict]) -> None:
    if not rows_a or not rows_b:
        print(f"    {label:34s} -- missing arm")
        return
    line = f"    {label:34s}"
    for key in ("final", "peak"):
        pairs, _ = paired(rows_a, rows_b, key)
        if not pairs:
            line += f"  {key}: no shared seeds"
            continue
        diffs = [a - b for a, b in pairs]
        wins = sum(1 for d in diffs if d > 0)
        line += f"  {key} {st.median(diffs):+9.1f} ({wins}/{len(diffs)}, p={wilcoxon_p(pairs):.3f})"
    print(line)


def arm_table(rows: list[dict]) -> None:
    finals = sorted(row["final"] for row in rows)
    peaks = sorted(row["peak"] for row in rows)
    # Fraction of peak given back by the end. Reported as a fraction of PEAK,
    # not of the env range, so it stays comparable across these three cells;
    # it exceeds 100% whenever the final return falls below zero.
    losses = [(row["peak"] - row["final"]) / abs(row["peak"]) if row["peak"] else float("nan") for row in rows]
    peak_steps = [row["peak_step"] for row in rows]
    # Against the run's OWN first eval timestep, not the earliest peak in the
    # arm -- otherwise an arm where nothing peaks early still reports a count.
    at_first = sum(1 for row in rows if row["peak_step"] <= row["first_step"])
    print(
        f"      n={len(rows):2d}  final med {st.median(finals):9.1f} "
        f"[{finals[0]:8.1f}, {finals[-1]:8.1f}]   peak med {st.median(peaks):9.1f} "
        f"[{peaks[0]:8.1f}, {peaks[-1]:8.1f}]"
    )
    print(
        f"            drawdown med {st.median(losses) * 100:5.1f}% of peak, "
        f"{sum(1 for l in losses if l > 0.5)}/{len(losses)} seeds losing >50%   "
        f"peak@ med {st.median(peak_steps) / 1000:.0f}k, {at_first}/{len(rows)} at the first eval point"
    )
    modified = [row["modified"] for row in rows if row["modified"] is not None]
    if modified:
        print(
            f"            modified-env final med {st.median(sorted(modified)):9.1f} "
            f"(standard-env median above is the headline number)"
        )


def config_audit(rows: list[dict]) -> None:
    """Every key that differs between arms, so a contrast is never read blind."""
    print("CONFIG AUDIT -- every arm must differ only in the treatment")
    seen: dict[tuple, set] = defaultdict(set)
    for row in rows:
        seen[(row["cell"], row["treatment"])].add(
            (json.dumps(row["plk"], sort_keys=True), row["unhealthy_penalty"], row["timesteps"])
        )
    for (cell, treatment), configs in sorted(seen.items()):
        for plk, penalty, steps in sorted(configs):
            print(f"    {cell:8s} {treatment:7s} plk={plk}  unhealthy_penalty={penalty}  steps={steps}")
    if any(len(configs) > 1 for configs in seen.values()):
        print("    WARNING: a cell/treatment contains more than one config; the contrast is confounded")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--only", choices=CELLS, help="restrict to one cell")
    args = parser.parse_args()

    rows = load(args.logs)
    if not rows:
        print(f"no stab_* runs with metadata.json under {args.logs}")
        return 1
    if wilcoxon is None:
        print("WARNING: scipy is not installed; every p-value below will be nan.\n")

    expected = 140
    print(f"{len(rows)}/{expected} runs loaded\n")
    config_audit(rows)

    index: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        index[(row["cell"], row["treatment"], row["arm"])].append(row)

    cells = [args.only] if args.only else CELLS
    for cell in cells:
        present = [t for t in TREATMENTS if any(index.get((cell, t, arm)) for arm in ("true", "partial"))]
        if not present:
            continue
        print(f"=== {ENV_NAME[cell]} ===")
        for treatment in present:
            print(f"  {TREATMENT_LABEL[treatment]}")
            for arm in ("true", "partial"):
                arm_rows = index.get((cell, treatment, arm))
                if not arm_rows:
                    continue
                name = arm if arm == "true" else f"partial ({arm_rows[0]['partial_name']})"
                print(f"    {name}")
                arm_table(arm_rows)
            # 2. The premise, inside this treatment.
            contrast(
                "PREMISE true - partial",
                index.get((cell, treatment, "true"), []),
                index.get((cell, treatment, "partial"), []),
            )
            print()

        # 3. The contrast that the pilot exists to measure.
        for treatment in present:
            if treatment == "ctrl":
                continue
            print(f"  {treatment} - ctrl, paired by seed:")
            for arm in ("true", "partial"):
                contrast(
                    f"{arm} arm",
                    index.get((cell, treatment, arm), []),
                    index.get((cell, "ctrl", arm), []),
                )
            print()
        print()

    print("REMINDERS")
    print("  * A 5-seed arm estimates an effect size; it cannot test one. The exact")
    print("    two-sided Wilcoxon floor at n=10 is 0.002.")
    print("  * Read peak@ before any drawdown number: a peak at the first eval point")
    print("    means the arm never learned and there is nothing to stabilize.")
    print("  * A fix that raises `true` and `partial` equally has not repaired the")
    print("    premise, which is the only reason this pilot exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
