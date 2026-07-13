from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .config import ATARI_SUITE, SummaryConfig, normalize_summary_config
from .results import SummaryResult


MUJOCO_SUMMARY_FIELDS = [
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
    summary_fields = ATARI_SUMMARY_FIELDS if config.suite == ATARI_SUITE else MUJOCO_SUMMARY_FIELDS
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
