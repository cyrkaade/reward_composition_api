#!/usr/bin/env python
"""Validate and summarize the 100-run pre3 standard-reward-model gate.

Run from ``api/`` after fetching the ``pre3g_*`` directories::

    python jobs/analyze_pre3_gate.py --cell ll       # after lines 1-60
    python jobs/analyze_pre3_gate.py                 # after all 100

This is a pilot gate, not an inferential multi-environment analysis.  It prints
no pooled p-values.  Do not launch later pretraining/active-learning stages if
any cell is short of 10 seeds or any preference run has other than 350/350
synthetic queries.  LunarLander qualifies only when true beats partial on both
median peak and median final.  Walker qualification comes from run_e0tuned.sh.

Use the bounded model downstream.  As a go/no-go pilot heuristic, advance the
collapse contrast only if bounded feedback has at least 3/10 >20% drawdowns,
at least two more collapsed seeds than bounded naive, and a larger median
drawdown.  This is not an inferential conclusion.  If both bounded modes have
median final below half their respective unbounded medians, stop and diagnose
reward scaling instead of launching the main grid.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPECTED: tuple[tuple[str, str], ...] = (
    ("ll", "true"),
    ("ll", "partial"),
    ("ll", "feedback_unbounded"),
    ("ll", "feedback_bounded"),
    ("ll", "naive_unbounded"),
    ("ll", "naive_bounded"),
    ("walker", "feedback_unbounded"),
    ("walker", "feedback_bounded"),
    ("walker", "naive_unbounded"),
    ("walker", "naive_bounded"),
)
PREFERENCE_VARIANTS = {
    "feedback_unbounded",
    "feedback_bounded",
    "naive_unbounded",
    "naive_bounded",
}
SEED_RE = re.compile(r"_seed(\d+)$")


@dataclass(frozen=True)
class Run:
    seed: int
    final: float
    peak: float
    peak_timestep: float
    synthetic_queries: int
    path: Path

    @property
    def drawdown(self) -> float | None:
        if self.peak <= 0:
            return None
        return 1.0 - self.final / self.peak


def load_run(meta_path: Path) -> Run:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    seed = meta.get("seed")
    if seed is None:
        match = SEED_RE.search(meta_path.parent.name)
        if match is None:
            raise ValueError(f"cannot recover seed from {meta_path}")
        seed = int(match.group(1))

    curve_path = meta_path.parent / "eval" / "evaluations.npz"
    if curve_path.exists():
        with np.load(curve_path) as data:
            rewards = np.asarray(data["results"], dtype=float).mean(axis=1)
            timesteps = np.asarray(data["timesteps"], dtype=float)
        if len(rewards) == 0:
            raise ValueError(f"empty evaluation curve: {curve_path}")
        best = int(np.argmax(rewards))
        final = float(rewards[-1])
        peak = float(rewards[best])
        peak_timestep = float(timesteps[best])
    else:
        final = meta.get("selected_policy_true_reward_mean")
        peak = meta.get("best_logged_true_reward")
        peak_timestep = meta.get("best_logged_timestep")
        if final is None or peak is None or peak_timestep is None:
            raise ValueError(f"missing curve and peak/final metadata: {meta_path}")
        final = float(final)
        peak = float(peak)
        peak_timestep = float(peak_timestep)

    return Run(
        seed=int(seed),
        final=final,
        peak=peak,
        peak_timestep=peak_timestep,
        synthetic_queries=int(meta.get("synthetic_queries", 0)),
        path=meta_path.parent,
    )


def load_cell(root: Path, cell: str, variant: str) -> tuple[list[Run], list[str]]:
    directory = root / f"pre3g_{cell}_{variant}"
    errors: list[str] = []
    by_seed: dict[int, Run] = {}
    for meta_path in sorted(directory.glob("*/metadata.json")):
        try:
            run = load_run(meta_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
            continue
        if run.seed in by_seed:
            errors.append(
                f"duplicate seed {run.seed} in {directory}: "
                f"{by_seed[run.seed].path.name}, {run.path.name}"
            )
            continue
        by_seed[run.seed] = run
    return [by_seed[seed] for seed in sorted(by_seed)], errors


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def summarize(runs: list[Run]) -> dict[str, float | int | None]:
    drops = [run.drawdown for run in runs if run.drawdown is not None]
    return {
        "n": len(runs),
        "final": med([run.final for run in runs]) if runs else None,
        "peak": med([run.peak for run in runs]) if runs else None,
        "peak_at": med([run.peak_timestep for run in runs]) if runs else None,
        "drawdown": med([float(drop) for drop in drops]) if drops else None,
        "collapsed": sum(float(drop) > 0.2 for drop in drops),
        "drop_n": len(drops),
    }


def fmt(value: float | int | None, width: int = 9) -> str:
    return "n/a".rjust(width) if value is None else f"{float(value):{width}.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("logs"),
        help="Log root containing pre3g_* directories (default: logs)",
    )
    parser.add_argument(
        "--cell",
        choices=("all", "ll", "walker"),
        default="all",
        help="Validate all cells (default), LunarLander only, or Walker2d only",
    )
    parser.add_argument("--per-seed", action="store_true", help="Print every run")
    args = parser.parse_args()

    selected_cells = ("ll", "walker") if args.cell == "all" else (args.cell,)
    expected = tuple(item for item in EXPECTED if item[0] in selected_cells)

    cells: dict[tuple[str, str], list[Run]] = {}
    problems: list[str] = []

    print("cell                              n     queries     final      peak   drawdown    >20%  peak@")
    print("-" * 100)
    for cell, variant in expected:
        runs, errors = load_cell(args.root, cell, variant)
        cells[(cell, variant)] = runs
        problems.extend(errors)
        summary = summarize(runs)

        expected_seeds = set(range(10))
        actual_seeds = {run.seed for run in runs}
        missing = sorted(expected_seeds - actual_seeds)
        unexpected = sorted(actual_seeds - expected_seeds)
        if missing:
            problems.append(f"pre3g_{cell}_{variant}: missing seeds {missing}")
        if unexpected:
            problems.append(f"pre3g_{cell}_{variant}: unexpected seeds {unexpected}")

        if variant in PREFERENCE_VARIANTS:
            delivered = sum(run.synthetic_queries == 350 for run in runs)
            query_text = f"{delivered:2d}/{len(runs):<2d}@350"
            for run in runs:
                if run.synthetic_queries != 350:
                    problems.append(
                        f"{run.path}: delivered {run.synthetic_queries}/350 queries"
                    )
        else:
            query_text = "reference"

        dd = summary["drawdown"]
        dd_text = "   n/a" if dd is None else f"{100 * float(dd):6.1f}%"
        peak_at = summary["peak_at"]
        peak_at_text = " n/a" if peak_at is None else f"{float(peak_at) / 1e6:4.2f}M"
        collapse_text = f"{summary['collapsed']}/{summary['drop_n']}"
        label = f"{cell}/{variant}"
        print(
            f"{label:33s} {summary['n']:2d}  {query_text:>11s} "
            f"{fmt(summary['final'])} {fmt(summary['peak'])} {dd_text:>10s} "
            f"{collapse_text:>7s} {peak_at_text:>6s}"
        )

        if args.per_seed:
            for run in runs:
                drop = run.drawdown
                drop_text = "n/a" if drop is None else f"{100 * drop:.1f}%"
                print(
                    f"    seed {run.seed:2d}: q={run.synthetic_queries:3d} "
                    f"peak={run.peak:9.1f} @ {run.peak_timestep / 1e6:.2f}M "
                    f"-> final={run.final:9.1f}, drop={drop_text}"
                )

    print("\nGate contrasts (median differences only; no pooled p-values)")
    if "ll" in selected_cells:
        ll_true = summarize(cells[("ll", "true")])
        ll_partial = summarize(cells[("ll", "partial")])
        if ll_true["n"] and ll_partial["n"]:
            peak_diff = float(ll_true["peak"]) - float(ll_partial["peak"])
            final_diff = float(ll_true["final"]) - float(ll_partial["final"])
            qualifier = "PASS" if peak_diff > 0 and final_diff > 0 else "FAIL"
            print(
                f"  LunarLander premise: true-partial peak={peak_diff:+.1f}, "
                f"final={final_diff:+.1f} -> {qualifier}"
            )
        else:
            print("  LunarLander premise: incomplete")

    bounded_collapse_passes: dict[str, bool] = {}
    scaling_failures: dict[str, bool] = {}
    for cell in selected_cells:
        summaries = {
            variant: summarize(cells[(cell, variant)])
            for variant in PREFERENCE_VARIANTS
        }
        if all(summary["n"] for summary in summaries.values()):
            for mode in ("feedback", "naive"):
                bounded = summaries[f"{mode}_bounded"]
                unbounded = summaries[f"{mode}_unbounded"]
                print(
                    f"  {cell}/{mode}: bounded-unbounded final="
                    f"{float(bounded['final']) - float(unbounded['final']):+.1f}; "
                    f"collapse {bounded['collapsed']}/{bounded['drop_n']} vs "
                    f"{unbounded['collapsed']}/{unbounded['drop_n']}"
                )

            feedback_bounded = summaries["feedback_bounded"]
            naive_bounded = summaries["naive_bounded"]
            feedback_bounded_dd = feedback_bounded["drawdown"]
            naive_bounded_dd = naive_bounded["drawdown"]
            bounded_collapse_passes[cell] = (
                int(feedback_bounded["collapsed"]) >= 3
                and int(feedback_bounded["collapsed"])
                >= int(naive_bounded["collapsed"]) + 2
                and feedback_bounded_dd is not None
                and naive_bounded_dd is not None
                and float(feedback_bounded_dd) > float(naive_bounded_dd)
            )
            scaling_failures[cell] = all(
                float(summaries[f"{mode}_bounded"]["final"])
                < 0.5 * float(summaries[f"{mode}_unbounded"]["final"])
                for mode in ("feedback", "naive")
            )
            print(
                f"  {cell}: bounded collapse pilot heuristic="
                f"{'PASS' if bounded_collapse_passes[cell] else 'FAIL'}; "
                f"bounded non-learning criterion="
                f"{'TRIGGERED' if scaling_failures[cell] else 'not triggered'}"
            )
        else:
            print(f"  {cell}: reward-model gate incomplete")

    if "walker" in selected_cells:
        print("  Walker premise: read run_e0tuned.sh results; it is not duplicated here.")

    if problems:
        print("\nVALIDATION FAILED -- do not launch a later stage")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    expected_runs = 10 * len(expected)
    print(
        f"\nVALIDATION PASSED: all {expected_runs} selected runs present; "
        "every preference run delivered 350/350 queries."
    )
    print("Apply the documented premise, collapse, and scaling gates before advancing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
