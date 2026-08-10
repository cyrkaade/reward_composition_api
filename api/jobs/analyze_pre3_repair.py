#!/usr/bin/env python
"""Validate and summarize the staged LunarLander pre3 repair.

After array lines 1--20::

    python jobs/analyze_pre3_repair.py --stage primary

Only if that command prints a primary FAIL and explicitly recommends the
fallback, submit lines 21--40 and run with ``--stage fallback``.  This is a
pilot decision tool, not a paper-level significance analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SOLVED = 200.0
MAX_EPOCHS = 100
STOP_THRESHOLD = 0.97
# "Material" means the new rule changed at least a quarter of member fits and
# did so in at least half the seeds.  This is deliberately a pilot heuristic.
MIN_EARLY_STOP_FRACTION = 0.25
MIN_RUNS_WITH_EARLY_STOP = 5
SEED_RE = re.compile(r"_seed(\d+)$")

STAGE_VARIANTS = {
    "primary": ("adaptive_feedback", "adaptive_naive"),
    "fallback": ("legacy_feedback", "legacy_naive"),
}
CONTROL_VARIANTS = ("feedback_bounded", "naive_bounded")


@dataclass(frozen=True)
class Run:
    variant: str
    seed: int
    final: float
    peak: float
    peak_timestep: float
    synthetic_queries: int
    path: Path
    metadata: dict[str, Any]

    @property
    def learned(self) -> bool:
        return self.peak >= SOLVED

    @property
    def final_solved(self) -> bool:
        return self.final >= SOLVED

    @property
    def lost_solved(self) -> bool:
        return self.learned and not self.final_solved

    @property
    def dropped_after_learning(self) -> bool:
        return self.learned and 1.0 - self.final / self.peak > 0.2


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def load_run(meta_path: Path, variant: str) -> Run:
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
        if not len(rewards):
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
        variant=variant,
        seed=int(seed),
        final=final,
        peak=peak,
        peak_timestep=peak_timestep,
        synthetic_queries=int(meta.get("synthetic_queries", 0)),
        path=meta_path.parent,
        metadata=meta,
    )


def load_variant(root: Path, variant: str) -> tuple[list[Run], list[str]]:
    directory = root / f"pre3r_ll_{variant}"
    by_seed: dict[int, Run] = {}
    errors: list[str] = []
    for meta_path in sorted(directory.glob("*/metadata.json")):
        try:
            run = load_run(meta_path, variant)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
            continue
        if run.seed in by_seed:
            errors.append(
                f"duplicate seed {run.seed} in {directory}: "
                f"{by_seed[run.seed].path.name}, {run.path.name}"
            )
        else:
            by_seed[run.seed] = run
    return [by_seed[seed] for seed in sorted(by_seed)], errors


def load_gate_control(root: Path, variant: str) -> list[Run]:
    directory = root / f"pre3g_ll_{variant}"
    runs: list[Run] = []
    for meta_path in sorted(directory.glob("*/metadata.json")):
        try:
            runs.append(load_run(meta_path, variant))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    return sorted(runs, key=lambda run: run.seed)


def summary(runs: list[Run]) -> dict[str, float | int | None]:
    learned = [run for run in runs if run.learned]
    return {
        "n": len(runs),
        "final": med([run.final for run in runs]) if runs else None,
        "peak": med([run.peak for run in runs]) if runs else None,
        "peak_at": med([run.peak_timestep for run in runs]) if runs else None,
        "ever": len(learned),
        "final_solved": sum(run.final_solved for run in runs),
        "lost": sum(run.lost_solved for run in runs),
        "drop20": sum(run.dropped_after_learning for run in learned),
    }


def equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-8, abs_tol=1e-10)
        except (TypeError, ValueError):
            return False
    return actual == expected


def expect(
    run: Run,
    key: str,
    expected: Any,
    problems: list[str],
) -> None:
    actual = run.metadata.get(key, "<missing>")
    if not equal(actual, expected):
        problems.append(f"{run.path}: {key}={actual!r}, expected {expected!r}")


def training_rounds(run: Run) -> list[dict[str, Any]]:
    value = run.metadata.get("reward_model_training")
    return value if isinstance(value, list) else []


def validate_training_telemetry(run: Run, primary: bool, problems: list[str]) -> None:
    rounds = training_rounds(run)
    if len(rounds) != 5:
        problems.append(
            f"{run.path}: reward_model_training has {len(rounds)} rounds, expected 5"
        )
        return

    cumulative: list[int] = []
    expected_members = 3 if primary else 1
    for round_report in rounds:
        try:
            cumulative.append(int(round_report["cumulative_queries"]))
        except (KeyError, TypeError, ValueError):
            problems.append(f"{run.path}: malformed cumulative_queries in training telemetry")
            continue
        members = round_report.get("members")
        for key in ("model_reward_output_mean", "model_reward_output_std"):
            value = round_report.get(key)
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                problems.append(f"{run.path}: invalid per-round {key}={value!r}")
        if not isinstance(members, list) or len(members) != expected_members:
            n_members = len(members) if isinstance(members, list) else "missing"
            problems.append(
                f"{run.path}: training round has {n_members} members, "
                f"expected {expected_members}"
            )
            continue
        for member in members:
            epochs = member.get("epochs_completed")
            try:
                epochs_int = int(epochs)
            except (TypeError, ValueError):
                problems.append(f"{run.path}: invalid epochs_completed={epochs!r}")
                continue
            if not 1 <= epochs_int <= MAX_EPOCHS:
                problems.append(
                    f"{run.path}: epochs_completed={epochs_int}, expected 1..{MAX_EPOCHS}"
                )
            last_loss = member.get("last_epoch_train_loss")
            try:
                finite_loss = math.isfinite(float(last_loss))
            except (TypeError, ValueError):
                finite_loss = False
            if not finite_loss:
                problems.append(
                    f"{run.path}: invalid last_epoch_train_loss={last_loss!r}"
                )
            stopped = member.get("stopped_by_train_accuracy")
            threshold = member.get("train_accuracy_stop")
            n_train = member.get("n_train_pairs")
            n_val = member.get("n_val_pairs")
            if not isinstance(n_train, int) or n_train <= 0:
                problems.append(f"{run.path}: invalid n_train_pairs={n_train!r}")
            if not isinstance(n_val, int) or n_val < 0:
                problems.append(f"{run.path}: invalid n_val_pairs={n_val!r}")

            if primary:
                if member.get("training_mode") != "full":
                    problems.append(
                        f"{run.path}: training_mode={member.get('training_mode')!r}, "
                        "expected 'full'"
                    )
                if not equal(threshold, STOP_THRESHOLD):
                    problems.append(
                        f"{run.path}: member train_accuracy_stop={threshold!r}, "
                        f"expected {STOP_THRESHOLD}"
                    )
                if n_val != 0:
                    problems.append(f"{run.path}: full-data member has n_val_pairs={n_val}")
                if stopped is True and epochs_int >= MAX_EPOCHS:
                    problems.append(
                        f"{run.path}: marked accuracy-stopped only at max epoch"
                    )
                if stopped not in (True, False):
                    problems.append(
                        f"{run.path}: invalid stopped_by_train_accuracy={stopped!r}"
                    )
            else:
                if member.get("training_mode") not in ("single", "kfold"):
                    problems.append(
                        f"{run.path}: legacy training_mode={member.get('training_mode')!r}, "
                        "expected 'single' or 'kfold'"
                    )
                if threshold is not None:
                    problems.append(
                        f"{run.path}: legacy member unexpectedly has "
                        f"train_accuracy_stop={threshold!r}"
                    )
                if stopped is not False:
                    problems.append(
                        f"{run.path}: legacy stopped_by_train_accuracy={stopped!r}, "
                        "expected false"
                    )
                if isinstance(n_val, int) and n_val <= 0:
                    problems.append(
                        f"{run.path}: legacy validation early-stop has n_val_pairs={n_val}"
                    )

    if cumulative != [70, 140, 210, 280, 350]:
        problems.append(
            f"{run.path}: cumulative query telemetry={cumulative!r}, "
            "expected [70, 140, 210, 280, 350]"
        )


def validate_run(run: Run, problems: list[str]) -> None:
    primary = run.variant.startswith("adaptive_")
    mode = "feedback" if run.variant.endswith("feedback") else "naive"

    common = {
        "env_id": "LunarLander-v3",
        "mode": mode,
        "requested_timesteps": 2_000_000,
        "final_policy": "last",
        "tuned_hyperparams": True,
        "query_budget": 350,
        "fragment_length": 25,
        "collection_timesteps": 30_000,
        "active_learning": False,
        "round0_data_protocol": "separate",
        "dedicated_query_rng": True,
        "reward_model_epochs": 100,
        "reward_model_patience": 10,
        "include_partial_feature": False,
    }
    recipe = (
        {
            "reward_hidden_sizes": [256, 256, 256],
            "reward_model_ensemble_size": 3,
            "reward_model_lr": 0.0003,
            "reward_model_batch_size": 128,
            "reward_model_loss_reduction": "mean",
            "reward_model_l1": 0.0,
            "reward_output_l1": 0.0,
            "ensemble_training": "full",
            "tanh_model_reward": True,
            "reward_model_train_accuracy_stop": STOP_THRESHOLD,
        }
        if primary
        else {
            "reward_hidden_sizes": [200],
            "reward_model_ensemble_size": 1,
            "reward_model_lr": 0.01,
            "reward_model_batch_size": 32,
            "reward_model_loss_reduction": "sum",
            "reward_model_l1": 0.01,
            "reward_output_l1": 0.001,
            "ensemble_training": "kfold",
            "tanh_model_reward": False,
            "reward_model_train_accuracy_stop": None,
        }
    )
    for key, expected in {**common, **recipe}.items():
        expect(run, key, expected, problems)
    if mode == "naive":
        expect(run, "partial_alpha", 1.0, problems)
    if run.synthetic_queries != 350:
        problems.append(f"{run.path}: delivered {run.synthetic_queries}/350 queries")
    if run.metadata.get("rm_diagnostics_before_training") is None:
        problems.append(f"{run.path}: missing before-training reward-model diagnostics")
    if run.metadata.get("model_reward_output_std") is None:
        problems.append(f"{run.path}: missing reward-model output diagnostics")
    if not (run.path / "reward_model.pt").is_file():
        problems.append(f"{run.path}: missing saved reward_model.pt")
    validate_training_telemetry(run, primary, problems)


def stop_summary(runs: list[Run]) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    runs_with_stop = 0
    by_round: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        run_stopped = False
        for round_index, report in enumerate(training_rounds(run)):
            round_members = report.get("members", [])
            if not isinstance(round_members, list):
                continue
            for member in round_members:
                if not isinstance(member, dict):
                    continue
                members.append(member)
                by_round.setdefault(round_index, []).append(member)
                if (
                    member.get("stopped_by_train_accuracy") is True
                    and int(member.get("epochs_completed", MAX_EPOCHS)) < MAX_EPOCHS
                ):
                    run_stopped = True
        runs_with_stop += int(run_stopped)

    early = [
        member
        for member in members
        if member.get("stopped_by_train_accuracy") is True
        and int(member.get("epochs_completed", MAX_EPOCHS)) < MAX_EPOCHS
    ]
    fraction = len(early) / len(members) if members else 0.0
    material = (
        fraction >= MIN_EARLY_STOP_FRACTION
        and runs_with_stop >= MIN_RUNS_WITH_EARLY_STOP
    )
    return {
        "members": members,
        "early": early,
        "fraction": fraction,
        "runs_with_stop": runs_with_stop,
        "material": material,
        "by_round": by_round,
    }


def fmt(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def print_table(cells: dict[str, list[Run]]) -> None:
    print(
        "variant                    n    final     peak  peak@  "
        "ever>=200 final>=200 lost-solved >20%-after-learn"
    )
    print("-" * 104)
    for variant, runs in cells.items():
        stats = summary(runs)
        peak_at = stats["peak_at"]
        peak_at_text = "n/a" if peak_at is None else f"{float(peak_at) / 1e6:.2f}M"
        print(
            f"{variant:26s} {int(stats['n']):2d} {fmt(stats['final']):>8s} "
            f"{fmt(stats['peak']):>8s} {peak_at_text:>6s} "
            f"{int(stats['ever']):7d}/10 {int(stats['final_solved']):8d}/10 "
            f"{int(stats['lost']):11d}/{int(stats['ever']):<2d} "
            f"{int(stats['drop20']):15d}/{int(stats['ever']):<2d}"
        )


def print_gate_context(root: Path) -> None:
    controls = {
        f"pre3g/{variant}": load_gate_control(root, variant)
        for variant in CONTROL_VARIANTS
    }
    if not any(controls.values()):
        return
    print("\nPrior fixed-100-epoch bounded pre3 gate (context only; not revalidated)")
    print_table(controls)


def print_stopping(runs: list[Run]) -> bool:
    stats = stop_summary(runs)
    members = stats["members"]
    early = stats["early"]
    epochs = [float(member["epochs_completed"]) for member in members]
    print("\nAdaptive-stop telemetry")
    print(
        f"  early-stopped member fits: {len(early)}/{len(members)} "
        f"({100 * stats['fraction']:.1f}%); runs with any stop: "
        f"{stats['runs_with_stop']}/{len(runs)}; median epochs="
        f"{med(epochs):.1f}" if epochs else "  no member telemetry"
    )
    for round_index in sorted(stats["by_round"]):
        round_members = stats["by_round"][round_index]
        round_early = [
            member
            for member in round_members
            if member.get("stopped_by_train_accuracy") is True
            and int(member.get("epochs_completed", MAX_EPOCHS)) < MAX_EPOCHS
        ]
        round_epochs = [float(member["epochs_completed"]) for member in round_members]
        print(
            f"  round {round_index}: stopped {len(round_early)}/{len(round_members)}, "
            f"median epochs={med(round_epochs):.1f}"
        )
    print(
        "  material-stop heuristic: "
        f">={100 * MIN_EARLY_STOP_FRACTION:.0f}% member fits and >="
        f"{MIN_RUNS_WITH_EARLY_STOP}/10 seeds affected -> "
        f"{'PASS' if stats['material'] else 'FAIL'}"
    )
    return bool(stats["material"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("logs"),
        help="Log root containing pre3r_* and optional pre3g_* dirs",
    )
    parser.add_argument(
        "--stage",
        choices=("primary", "fallback", "all"),
        default="primary",
        help="Validate primary 1-20 (default), fallback 21-40, or all",
    )
    parser.add_argument("--per-seed", action="store_true")
    args = parser.parse_args()

    stages = ("primary", "fallback") if args.stage == "all" else (args.stage,)
    variants = tuple(
        variant for stage in stages for variant in STAGE_VARIANTS[stage]
    )
    cells: dict[str, list[Run]] = {}
    problems: list[str] = []

    for variant in variants:
        runs, load_errors = load_variant(args.root, variant)
        cells[variant] = runs
        problems.extend(load_errors)
        expected_seeds = set(range(10))
        actual_seeds = {run.seed for run in runs}
        missing = sorted(expected_seeds - actual_seeds)
        unexpected = sorted(actual_seeds - expected_seeds)
        if missing:
            problems.append(f"pre3r_ll_{variant}: missing seeds {missing}")
        if unexpected:
            problems.append(f"pre3r_ll_{variant}: unexpected seeds {unexpected}")
        for run in runs:
            validate_run(run, problems)

    print_table(cells)
    if args.per_seed:
        print("\nPer-seed outcomes")
        for variant, runs in cells.items():
            for run in runs:
                print(
                    f"  {variant:20s} seed={run.seed:2d} q={run.synthetic_queries:3d} "
                    f"peak={run.peak:8.1f} @ {run.peak_timestep / 1e6:.2f}M "
                    f"final={run.final:8.1f} learned={run.learned} "
                    f"lost={run.lost_solved} drop20={run.dropped_after_learning}"
                )

    print_gate_context(args.root)

    if problems:
        print("\nVALIDATION FAILED -- do not interpret or launch another stage")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"\nVALIDATION PASSED: all {10 * len(variants)} selected runs are complete and configured correctly.")

    if "primary" in stages:
        feedback = summary(cells["adaptive_feedback"])
        naive = summary(cells["adaptive_naive"])
        stop_material = print_stopping(
            cells["adaptive_feedback"] + cells["adaptive_naive"]
        )
        policy_go = (
            int(naive["ever"]) >= 6
            and int(naive["final_solved"]) >= 5
            and int(naive["lost"]) < int(feedback["lost"])
            and int(feedback["ever"]) >= 2
        )
        primary_go = policy_go and stop_material
        print("\nPrimary decision")
        print(
            "  policy heuristic: naive ever>=6, naive final>=5, "
            "naive loses fewer solved policies than feedback, feedback ever>=2 -> "
            f"{'PASS' if policy_go else 'FAIL'}"
        )
        print(f"  overall primary repair -> {'GO' if primary_go else 'FAIL'}")
        if primary_go:
            print("  Do not submit fallback lines 21-40; this recipe can advance to the small pretraining pilot.")
        else:
            print("  Submit the legacy fallback only:")
            print("    sbatch --array=21-40%20 jobs/run_pre3_repair.sh")

    if "fallback" in stages:
        feedback = summary(cells["legacy_feedback"])
        naive = summary(cells["legacy_naive"])
        viable = (
            int(naive["final_solved"]) >= 5
            and int(naive["lost"]) < int(feedback["lost"])
        )
        print("\nFallback decision")
        print(
            "  legacy naive final>=5/10 and loses fewer solved policies "
            f"than feedback -> {'VIABLE' if viable else 'FAIL'}"
        )
        if viable:
            print("  The tuned legacy recipe is a viable base for a small pretraining pilot.")
        else:
            print("  Stop: neither repair provides a viable LunarLander base; do not launch pretraining/AL.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
