"""Ablation sweeps (plan / dry-run / execute via the rcomp CLI) and run
summaries aggregating metadata.json files into CSVs."""

from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import (
    ATARI_SUITE,
    BOX2D_SUITE,
    GYM_SUITE,
    MUJOCO_SUITE,
    PREFERENCE_MODES,
    SummaryConfig,
    SweepConfig,
    normalize_summary_config,
    normalize_sweep_config,
)
from .suites import get_suite


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    active_learning: bool | None = None
    pretrain: bool = False
    pretrain_target: str = "partial"
    include_partial_feature: bool | None = None


@dataclass(frozen=True)
class PlannedRun:
    env_id: str
    seed: int
    variant: str
    mode: str
    run_dir: Path
    completed: bool
    command: list[str]

    def as_manifest_row(self) -> dict:
        return {
            "env_id": self.env_id,
            "seed": self.seed,
            "variant": self.variant,
            "mode": self.mode,
            "run_dir": str(self.run_dir),
            "completed": self.completed,
            "command": self.command,
        }


@dataclass(frozen=True)
class SweepResult:
    manifest_path: Path
    planned_runs: list[PlannedRun]
    pending_runs: list[PlannedRun]
    executed: bool


@dataclass(frozen=True)
class SummaryResult:
    summary_csv: Path
    aggregate_csv: Path
    rows: list[dict]
    aggregate_rows: list[dict]


def ablation_variants(suite: str) -> list[Variant]:
    if suite == MUJOCO_SUITE:
        variants = [
            Variant(name="true_reference", mode="true"),
            Variant(name="partial_only", mode="partial"),
            Variant(name="feedback_only_al", mode="feedback", active_learning=True, include_partial_feature=False),
        ]
        for mode in ["naive", "delta"]:
            for pretrain in [False, True]:
                for active_learning in [False, True]:
                    name_parts = [mode]
                    name_parts.append("pretrain" if pretrain else "scratch")
                    name_parts.append("al" if active_learning else "random")
                    variants.append(
                        Variant(
                            name="_".join(name_parts),
                            mode=mode,
                            active_learning=active_learning,
                            pretrain=pretrain,
                            pretrain_target="partial",
                            include_partial_feature=True,
                        )
                    )
        return variants

    if suite == ATARI_SUITE:
        return [
            Variant(name="true_reference", mode="true"),
            Variant(name="partial_only", mode="partial"),
            Variant(name="feedback_only_random", mode="feedback", active_learning=False, include_partial_feature=False),
            Variant(name="naive_scratch_random", mode="naive", active_learning=False, include_partial_feature=True),
            Variant(name="delta_scratch_random", mode="delta", active_learning=False, include_partial_feature=True),
        ]

    if suite in (BOX2D_SUITE, GYM_SUITE):
        return [
            Variant(name="true_reference", mode="true"),
            Variant(name="partial_only", mode="partial"),
            Variant(name="naive_scratch_random", mode="naive", active_learning=False, include_partial_feature=True),
            Variant(name="delta_scratch_random", mode="delta", active_learning=False, include_partial_feature=True),
        ]

    raise ValueError(f"Unsupported sweep suite: {suite}")


def plan_sweep(config: SweepConfig) -> list[PlannedRun]:
    config = normalize_sweep_config(config)
    planned = []
    for env_id in config.env_ids or ():
        for seed in config.seeds:
            for variant in ablation_variants(config.suite):
                command, run_dir = build_command(config, env_id, seed, variant)
                planned.append(
                    PlannedRun(
                        env_id=env_id,
                        seed=seed,
                        variant=variant.name,
                        mode=variant.mode,
                        run_dir=run_dir,
                        completed=(run_dir / "metadata.json").exists(),
                        command=command,
                    )
                )
    return planned


def run_sweep(config: SweepConfig) -> SweepResult:
    config = normalize_sweep_config(config)
    planned_runs = plan_sweep(config)
    pending_runs = [run for run in planned_runs if not (run.completed and config.skip_completed)]
    write_manifest(Path(config.manifest), [run.as_manifest_row() for run in planned_runs])
    print(f"wrote manifest with {len(planned_runs)} planned runs to {config.manifest}")
    print(f"{len(pending_runs)} runs pending")

    if not config.execute:
        for run in pending_runs:
            print(shell_join(run.command))
        print("\nDry run only. Re-run with --execute to launch the study.")
        return SweepResult(Path(config.manifest), planned_runs, pending_runs, executed=False)

    for index, run in enumerate(pending_runs, start=1):
        print(f"\n[{index}/{len(pending_runs)}] running {run.run_dir}")
        subprocess.run(run.command, check=True)
    return SweepResult(Path(config.manifest), planned_runs, pending_runs, executed=True)


def build_command(config: SweepConfig, env_id: str, seed: int, variant: Variant) -> tuple[list[str], Path]:
    slug = get_suite(config.suite).slug(env_id)
    run_name = f"{slug}_{variant.name}_{config.timesteps // 1000}k_seed{seed}"
    run_dir = Path(config.log_dir) / run_name
    command = [
        sys.executable,
        "-m",
        "rcomp",
        "train",
        "--suite",
        config.suite,
        "--env-id",
        env_id,
        "--mode",
        variant.mode,
        "--variant-name",
        variant.name,
        "--run-name",
        run_name,
        "--log-dir",
        str(config.log_dir),
        "--timesteps",
        str(config.timesteps),
        "--seed",
        str(seed),
        "--n-envs",
        str(config.n_envs),
        "--eval-freq",
        str(config.eval_freq),
        "--n-eval-episodes",
        str(config.n_eval_episodes),
        "--final-eval-episodes",
        str(config.final_eval_episodes),
        "--query-budget",
        str(config.query_budget),
        "--rlhf-rounds",
        str(config.rlhf_rounds),
        "--collection-timesteps",
        str(config.collection_timesteps),
        "--fragment-length",
        str(config.fragment_length),
        "--reward-model-epochs",
        str(config.reward_model_epochs),
        "--reward-model-patience",
        str(config.reward_model_patience),
        "--reward-model-batch-size",
        str(config.reward_model_batch_size),
        "--reward-model-ensemble-size",
        str(config.reward_model_ensemble_size),
        "--active-query-strategy",
        config.active_query_strategy,
        "--active-learning-batches",
        str(config.active_learning_batches),
        "--device",
        config.device,
    ]

    if config.suite == MUJOCO_SUITE:
        command.extend(["--preset", str(config.preset)])
    if config.partial:
        command.extend(["--partial", config.partial])
    if config.progress_bar:
        command.append("--progress-bar")
    if config.normalize_model_reward:
        command.append("--normalize-model-reward")
    if config.model_reward_min is not None:
        command.extend(["--model-reward-min", str(config.model_reward_min)])
    if config.model_reward_max is not None:
        command.extend(["--model-reward-max", str(config.model_reward_max)])
    if config.model_reward_target_mean is not None:
        command.extend(["--model-reward-target-mean", str(config.model_reward_target_mean)])
    if config.model_reward_target_std is not None:
        command.extend(["--model-reward-target-std", str(config.model_reward_target_std)])

    if variant.mode in PREFERENCE_MODES:
        command.append("--active-learning" if variant.active_learning else "--no-active-learning")
        if variant.pretrain:
            command.extend(
                [
                    "--pretrain-reward-model",
                    "--pretrain-target",
                    variant.pretrain_target,
                    "--pretrain-epochs",
                    str(config.pretrain_epochs),
                    "--pretrain-batch-size",
                    str(config.pretrain_batch_size),
                    "--pretrain-lr",
                    str(config.pretrain_lr),
                ]
            )
        if variant.include_partial_feature is True:
            command.append("--include-partial-feature")
        elif variant.include_partial_feature is False:
            command.append("--no-include-partial-feature")

    return command, run_dir


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


SUMMARY_FIELDS = [
    "env_id",
    "variant",
    "mode",
    "seed",
    "run_name",
    "requested_timesteps",
    "actual_timesteps",
    "synthetic_queries",
    "active_learning",
    "active_query_strategy",
    "reward_model_ensemble_size",
    "pretrain_reward_model",
    "pretrain_target",
    "best_logged_true_reward",
    "best_logged_timestep",
    "selected_policy_true_reward_mean",
    "selected_policy_true_reward_std",
    "selected_mean_partial",
    "selected_mean_residual",
    "run_dir",
]

ATARI_SUMMARY_FIELDS = [
    *SUMMARY_FIELDS[:-1],
    "selected_mean_lost_lives",
    "selected_mean_lives",
    "run_dir",
]

AGGREGATE_FIELDS = [
    "env_id",
    "variant",
    "mode",
    "n",
    "mean_selected_true_reward",
    "std_selected_true_reward",
    "mean_best_logged_true_reward",
    "mean_synthetic_queries",
]

ATARI_AGGREGATE_FIELDS = [*AGGREGATE_FIELDS, "mean_selected_lost_lives"]


def summarize_runs(config: SummaryConfig) -> SummaryResult:
    config = normalize_summary_config(config)
    rows = load_metadata_rows(Path(config.root), config.suite)
    aggregate_rows = aggregate(rows, include_lost_lives=config.suite == ATARI_SUITE)
    summary_fields = ATARI_SUMMARY_FIELDS if config.suite == ATARI_SUITE else SUMMARY_FIELDS
    aggregate_fields = ATARI_AGGREGATE_FIELDS if config.suite == ATARI_SUITE else AGGREGATE_FIELDS
    write_csv(Path(config.summary_csv), rows, summary_fields)
    write_csv(Path(config.aggregate_csv), aggregate_rows, aggregate_fields)
    print(f"wrote {len(rows)} run rows to {config.summary_csv}")
    print(f"wrote {len(aggregate_rows)} aggregate rows to {config.aggregate_csv}")
    return SummaryResult(Path(config.summary_csv), Path(config.aggregate_csv), rows, aggregate_rows)


def load_metadata_rows(root: Path, suite: str) -> list[dict]:
    rows = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        components = metadata.get("selected_policy_components") or {}
        row = {
            "env_id": metadata.get("env_id"),
            "variant": metadata.get("variant", metadata.get("mode")),
            "mode": metadata.get("mode"),
            "seed": metadata.get("seed"),
            "run_name": metadata.get("run_name"),
            "requested_timesteps": metadata.get("requested_timesteps"),
            "actual_timesteps": metadata.get("actual_timesteps"),
            "synthetic_queries": metadata.get("synthetic_queries"),
            "active_learning": metadata.get("active_learning"),
            "active_query_strategy": metadata.get("active_query_strategy"),
            "reward_model_ensemble_size": metadata.get("reward_model_ensemble_size"),
            "pretrain_reward_model": metadata.get("pretrain_reward_model"),
            "pretrain_target": metadata.get("pretrain_target"),
            "best_logged_true_reward": metadata.get("best_logged_true_reward"),
            "best_logged_timestep": metadata.get("best_logged_timestep"),
            "selected_policy_true_reward_mean": metadata.get("selected_policy_true_reward_mean"),
            "selected_policy_true_reward_std": metadata.get("selected_policy_true_reward_std"),
            "selected_mean_partial": components.get("mean_partial"),
            "selected_mean_residual": components.get("mean_residual"),
            "run_dir": str(metadata_path.parent),
        }
        if suite == ATARI_SUITE:
            row.update(
                {
                    "selected_mean_lost_lives": components.get("mean_lost_lives"),
                    "selected_mean_lives": components.get("mean_lives"),
                }
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def std(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return 0.0 if clean else None
    m = mean(clean)
    return (sum((value - m) ** 2 for value in clean) / len(clean)) ** 0.5


def aggregate(rows: list[dict], include_lost_lives: bool = False) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["env_id"], row["variant"], row["mode"])].append(row)

    aggregate_rows = []
    for (env_id, variant, mode), group_rows in sorted(groups.items()):
        selected = [row["selected_policy_true_reward_mean"] for row in group_rows]
        best = [row["best_logged_true_reward"] for row in group_rows]
        queries = [row["synthetic_queries"] for row in group_rows]
        aggregate_row = {
            "env_id": env_id,
            "variant": variant,
            "mode": mode,
            "n": len(group_rows),
            "mean_selected_true_reward": mean(selected),
            "std_selected_true_reward": std(selected),
            "mean_best_logged_true_reward": mean(best),
            "mean_synthetic_queries": mean(queries),
        }
        if include_lost_lives:
            aggregate_row["mean_selected_lost_lives"] = mean([row["selected_mean_lost_lives"] for row in group_rows])
        aggregate_rows.append(aggregate_row)
    return aggregate_rows
