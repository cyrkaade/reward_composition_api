"""Partiality tools: estimate how much a manual partial reward explains the
true reward over random-policy fragments, and plot final reward by
partiality x query budget across completed runs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ConfigError, suite_supported_envs
from .data import Trajectory
from .partials import PartialRegistry, PartialSpec, load_partial_reference
from .rewards.preferences import fragment_trajectories


@dataclass(frozen=True)
class PartialityConfig:
    suite: str
    env_id: str
    partial: str
    timesteps: int = 100_000
    fragment_length: int = 25
    seed: int = 0


def estimate_partiality_from_returns(true_returns: list[float], partial_returns: list[float]) -> dict[str, Any]:
    if len(true_returns) != len(partial_returns):
        raise ConfigError("true and partial return arrays must have the same length")
    if len(true_returns) < 2:
        raise ConfigError("partiality requires at least two trajectories or fragments")

    true_values = [float(value) for value in true_returns]
    partial_values = [float(value) for value in partial_returns]
    true_mean = sum(true_values) / len(true_values)
    partial_mean = sum(partial_values) / len(partial_values)
    true_centered = [value - true_mean for value in true_values]
    partial_centered = [value - partial_mean for value in partial_values]
    denominator = len(true_values) - 1
    var_true = sum(value * value for value in true_centered) / denominator
    var_partial = sum(value * value for value in partial_centered) / denominator
    if var_true <= 0.0:
        raise ConfigError("true returns have zero variance; collect more diverse samples")

    cov = sum(true_delta * partial_delta for true_delta, partial_delta in zip(true_centered, partial_centered)) / denominator
    rho = None
    if var_partial > 0.0:
        rho = cov / math.sqrt(var_true * var_partial)

    partiality = cov / var_true
    return {
        "partiality": float(partiality),
        "partiality_clipped_0_1": float(min(max(partiality, 0.0), 1.0)),
        "cov": cov,
        "var_true": var_true,
        "var_partial": var_partial,
        "rho": rho,
        "assumption_a_holds": bool(var_partial <= 2.0 * cov),
        "n_samples": int(len(true_values)),
        "mean_true_return": float(true_mean),
        "mean_partial_return": float(partial_mean),
    }


def estimate_partiality(config: PartialityConfig) -> dict[str, Any]:
    _validate_config(config)
    partial_spec = load_partial_reference(config.partial, config.suite, PartialRegistry())
    trajectories = collect_random_trajectories(
        env_id=config.env_id,
        partial_spec=partial_spec,
        total_timesteps=config.timesteps,
        seed=config.seed,
    )
    samples = fragment_trajectories(trajectories, config.fragment_length)
    if not samples:
        samples = trajectories
    true_returns = [sum(float(state["rew"]) for state in sample.states) for sample in samples]
    partial_returns = [sum(float(state["partial_rew"]) for state in sample.states) for sample in samples]
    metrics = estimate_partiality_from_returns(true_returns, partial_returns)
    return {
        "suite": config.suite,
        "env_id": config.env_id,
        "partial": partial_spec.name,
        "timesteps": config.timesteps,
        "fragment_length": config.fragment_length,
        "seed": config.seed,
        "n_trajectories": len(trajectories),
        **metrics,
    }


def collect_random_trajectories(
    env_id: str,
    partial_spec: PartialSpec,
    total_timesteps: int,
    seed: int,
) -> list[Trajectory]:
    env = gym.make(env_id)
    partial = partial_spec.create(env_id)
    trajectories: list[Trajectory] = []
    trajectory = Trajectory()
    steps = 0

    try:
        env.action_space.seed(seed)
        obs, info = env.reset(seed=seed)
        partial.reset(info)
        while steps < total_timesteps:
            action = env.action_space.sample()
            next_obs, true_reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            partial_step = partial.step(obs, action, next_obs, float(true_reward), terminated, truncated, info)
            trajectory.push_state(next_obs, action, done, dict(info), float(true_reward), partial_step.partial)
            steps += 1

            if done:
                trajectories.append(trajectory)
                trajectory = Trajectory()
                obs, info = env.reset()
                partial.reset(info)
            else:
                obs = next_obs
    finally:
        env.close()

    if trajectory.states:
        trajectories.append(trajectory)
    return trajectories


def partiality_json(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True)


def default_partiality_output_path(metrics: dict[str, Any]) -> Path:
    env_slug = _slugify(str(metrics["env_id"]))
    partial_slug = _slugify(str(metrics["partial"]))
    return (
        Path("logs")
        / "partiality"
        / env_slug
        / f"{partial_slug}_seed{metrics['seed']}_steps{metrics['timesteps']}_frag{metrics['fragment_length']}.json"
    )


def save_partiality_result(metrics: dict[str, Any], output: str | Path | None = None) -> Path:
    path = Path(output) if output is not None else default_partiality_output_path(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(partiality_json(metrics) + "\n", encoding="utf-8")
    return path


def _validate_config(config: PartialityConfig) -> None:
    if config.env_id not in suite_supported_envs(config.suite):
        raise ConfigError(f"Unsupported {config.suite} env '{config.env_id}'")
    if config.timesteps <= 0:
        raise ConfigError("timesteps must be greater than zero")
    if config.fragment_length <= 0:
        raise ConfigError("fragment_length must be greater than zero")


def _slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


@dataclass(frozen=True)
class PartialityGridConfig:
    runs_root: str | Path = "logs"
    partiality_root: str | Path = Path("logs") / "partiality"
    output: str | Path = Path("logs") / "partiality" / "partiality_grid.png"
    env_id: str | None = None
    title: str = "Partiality vs RLHF queries"


def plot_partiality_grid(config: PartialityGridConfig) -> Path:
    partiality_by_key = load_partiality_index(Path(config.partiality_root))
    rows = load_grid_rows(Path(config.runs_root), partiality_by_key, config.env_id)
    if not rows:
        raise ConfigError("No matching run metadata could be joined with partiality results")

    table = build_grid(rows)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_grid(table, output, config.title)
    return output


def load_partiality_index(root: Path) -> dict[tuple[str, str], float]:
    index = {}
    if not root.exists():
        return index
    for path in sorted(root.rglob("*.json")):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        env_id = metrics.get("env_id")
        partial = metrics.get("partial")
        partiality = metrics.get("partiality_clipped_0_1", metrics.get("partiality"))
        if env_id and partial and partiality is not None:
            index[(str(env_id), _partial_name(str(partial)))] = float(partiality)
    return index


def load_grid_rows(runs_root: Path, partiality_by_key: dict[tuple[str, str], float], env_id: str | None = None) -> list[dict[str, Any]]:
    rows = []
    if not runs_root.exists():
        return rows
    for metadata_path in sorted(runs_root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        row_env = metadata.get("env_id")
        if env_id is not None and row_env != env_id:
            continue

        reward = metadata.get("selected_policy_true_reward_mean")
        if reward is None:
            continue

        partiality = _metadata_partiality(metadata, partiality_by_key)
        if partiality is None:
            continue

        rows.append(
            {
                "env_id": row_env,
                "partiality": float(partiality),
                "queries": int(metadata.get("query_budget") or 0),
                "reward": float(reward),
                "variant": metadata.get("variant", metadata.get("mode")),
                "run_dir": str(metadata_path.parent),
            }
        )
    return rows


def build_grid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        partiality = round(float(row["partiality"]), 2)
        queries = int(row["queries"])
        grouped[(queries, partiality)].append(float(row["reward"]))

    query_values = sorted({query for query, _partiality in grouped}, reverse=True)
    partiality_values = sorted({_partiality for _query, _partiality in grouped})
    values = np.full((len(query_values), len(partiality_values)), np.nan, dtype=np.float64)
    counts = np.zeros_like(values, dtype=np.int64)

    for row_idx, query in enumerate(query_values):
        for col_idx, partiality in enumerate(partiality_values):
            rewards = grouped.get((query, partiality), [])
            if rewards:
                values[row_idx, col_idx] = float(sum(rewards) / len(rewards))
                counts[row_idx, col_idx] = len(rewards)

    return {
        "queries": query_values,
        "partialities": partiality_values,
        "values": values,
        "counts": counts,
    }


def render_grid(table: dict[str, Any], output: Path, title: str) -> None:
    values = table["values"]
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("rainbow_r").copy()
    cmap.set_bad("#eeeeee")

    size = max(4.5, 0.7 * max(values.shape))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(masked, cmap=cmap, aspect="equal")

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Partiality")
    ax.set_ylabel("No. RLHF queries")
    ax.set_xticks(range(len(table["partialities"])))
    ax.set_xticklabels([_format_axis_value(value) for value in table["partialities"]])
    ax.set_yticks(range(len(table["queries"])))
    ax.set_yticklabels([_format_query_value(value) for value in table["queries"]])

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            if np.isfinite(values[row_idx, col_idx]):
                ax.text(col_idx, row_idx, f"{values[row_idx, col_idx]:.0f}", ha="center", va="center", color="white", fontsize=9)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Final reward")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _metadata_partiality(metadata: dict[str, Any], partiality_by_key: dict[tuple[str, str], float]) -> float | None:
    if metadata.get("mode") == "true":
        return 1.0
    partial_ref = metadata.get("partial_reference")
    if not partial_ref:
        return None
    return partiality_by_key.get((str(metadata.get("env_id")), _partial_name(str(partial_ref))))


def _partial_name(reference: str) -> str:
    if ":" in reference:
        return reference.rsplit(":", 1)[1]
    if "/" in reference:
        return reference.rsplit("/", 1)[1]
    return Path(reference).stem


def _format_axis_value(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_query_value(value: int) -> str:
    if value >= 1000 and value % 1000 == 0:
        return f"{value // 1000}k"
    return str(value)
