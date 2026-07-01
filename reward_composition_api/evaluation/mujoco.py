from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from local_gym.classes.mujoco_reward_specs import MuJoCoRewardSpec
from reward_composition_api.environments.vectorized import normalize_obs
from reward_composition_api.evaluation.component_evaluator import evaluate_policy_components
from reward_composition_api.registry import PartialSpec

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
    return evaluate_policy_components(
        model=model,
        env_id=env_id,
        make_env=gym.make,
        custom_partial=custom_partial,
        stats_source=stats_source,
        n_eval_episodes=n_eval_episodes,
        seed=seed,
        deterministic=deterministic,
        component_keys=_component_keys(spec, custom_partial),
        model_observation=normalize_obs,
        action_converter=lambda _env, action: action[0],
        default_partial_step=lambda _obs, _action, _new_obs, _reward, _terminated, _truncated, info: (
            spec.partial_reward(info),
            spec.components(info),
        ),
    )


def _component_keys(spec: MuJoCoRewardSpec, custom_partial: PartialSpec | None) -> tuple[str, ...]:
    if custom_partial is not None:
        return custom_partial.component_keys
    return spec.component_keys


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
