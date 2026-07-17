from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym

from reward_composition_api.config import suite_supported_envs
from reward_composition_api.data_structures import Trajectory
from reward_composition_api.errors import ConfigError
from reward_composition_api.partial_reward import build_builtin_registry
from reward_composition_api.registry import PartialSpec, load_partial_reference


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
    registry = build_builtin_registry()
    partial_spec = load_partial_reference(config.partial, config.suite, registry)
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


def fragment_trajectories(trajectories: list[Trajectory], fragment_length: int) -> list[Trajectory]:
    fragments = []
    for trajectory in trajectories:
        states = trajectory.get_states()
        for start_idx in range(0, len(states), fragment_length):
            if start_idx + fragment_length <= len(states):
                fragments.append(Trajectory(states[start_idx : start_idx + fragment_length]))
    return fragments


def _validate_config(config: PartialityConfig) -> None:
    if config.env_id not in suite_supported_envs(config.suite):
        raise ConfigError(f"Unsupported {config.suite} env '{config.env_id}'")
    if config.timesteps <= 0:
        raise ConfigError("timesteps must be greater than zero")
    if config.fragment_length <= 0:
        raise ConfigError("fragment_length must be greater than zero")


def _slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
