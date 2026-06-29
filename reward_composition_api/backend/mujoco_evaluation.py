from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from local_gym.classes.mujoco_reward_specs import MuJoCoRewardSpec
from reward_composition_api.registry import PartialSpec

from .common import normalize_obs, summarize_component_rows
from .reporting import ComponentEvalCallback, write_component_summary_csv


def evaluate_mujoco_components(
    model,
    env_id: str,
    spec: MuJoCoRewardSpec,
    custom_partial: PartialSpec | None = None,
    stats_source=None,
    n_eval_episodes: int = 10,
    seed: int = 0,
    deterministic: bool = True,
):
    rows = []
    env = gym.make(env_id)
    partial = custom_partial.create(env_id) if custom_partial else None
    try:
        for episode_index in range(n_eval_episodes):
            obs, info = env.reset(seed=seed + episode_index)
            if partial is not None:
                partial.reset(info)
            done = False
            total = 0.0
            length = 0
            components = _empty_accumulators(spec, custom_partial)

            while not done:
                model_obs = normalize_obs(stats_source, obs)
                action, _ = model.predict(model_obs, deterministic=deterministic)
                new_obs, reward, terminated, truncated, info = env.step(action[0])
                done = terminated or truncated
                total += float(reward)
                length += 1

                if partial is None:
                    step_components = spec.components(info)
                else:
                    partial_step = partial.step(obs, action[0], new_obs, reward, terminated, truncated, info)
                    step_components = {"partial": partial_step.partial, **partial_step.components}
                for key, value in step_components.items():
                    if key in components:
                        components[key] += value
                obs = new_obs

            residual = total - components["partial"]
            row = {"total": total, "residual": residual, "length": float(length), **components}
            rows.append(row)
    finally:
        env.close()

    keys = ["total", "partial", "residual", *_component_keys(spec, custom_partial), "length"]
    return _summarize_rows(rows, keys)


def _empty_accumulators(spec: MuJoCoRewardSpec, custom_partial: PartialSpec | None) -> dict[str, float]:
    values = {key: 0.0 for key in _component_keys(spec, custom_partial)}
    values["partial"] = 0.0
    return values


def _component_keys(spec: MuJoCoRewardSpec, custom_partial: PartialSpec | None) -> tuple[str, ...]:
    if custom_partial is not None:
        return custom_partial.component_keys
    return spec.component_keys


def _summarize_rows(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    return summarize_component_rows(rows, keys)


def component_fieldnames(spec: MuJoCoRewardSpec, custom_partial: PartialSpec | None = None) -> list[str]:
    keys = ["total", "partial", "residual", *_component_keys(spec, custom_partial), "length"]
    fields = ["timesteps"]
    for key in keys:
        fields.extend([f"mean_{key}", f"std_{key}"])
    return fields


def write_mujoco_component_summary(
    path: Path,
    timestep: int,
    spec: MuJoCoRewardSpec,
    stats: dict,
    custom_partial: PartialSpec | None = None,
):
    write_component_summary_csv(path, timestep, stats, component_fieldnames(spec, custom_partial))


class MuJoCoComponentEvalCallback(ComponentEvalCallback):
    def __init__(
        self,
        log_path: Path,
        env_id: str,
        spec: MuJoCoRewardSpec,
        custom_partial: PartialSpec | None,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(log_path, eval_freq, n_eval_episodes, seed=seed, verbose=verbose)
        self.env_id = env_id
        self.spec = spec
        self.custom_partial = custom_partial

    def component_fieldnames(self) -> list[str]:
        return component_fieldnames(self.spec, self.custom_partial)

    def evaluate_components(self) -> dict:
        return evaluate_mujoco_components(
            self.model,
            self.env_id,
            self.spec,
            custom_partial=self.custom_partial,
            stats_source=self.training_env,
            n_eval_episodes=self.n_eval_episodes,
            seed=self.seed + self.num_timesteps,
            deterministic=True,
        )

    def write_summary(self, stats: dict) -> None:
        write_mujoco_component_summary(self.log_path, self.num_timesteps, self.spec, stats, self.custom_partial)

    def log_message(self, stats: dict) -> str:
        return (
            "Component eval "
            f"env={self.env_id} t={self.num_timesteps}: "
            f"total={stats['mean_total']:.3f}, "
            f"partial={stats['mean_partial']:.3f}, "
            f"residual={stats['mean_residual']:.3f}"
        )
