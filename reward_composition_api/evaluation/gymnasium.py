from __future__ import annotations

from pathlib import Path

from reward_composition_api.environments.spaces import action_for_space, policy_observation
from reward_composition_api.evaluation.component_evaluator import evaluate_policy_components
from reward_composition_api.registry import PartialSpec

from .reporting import ComponentEvalCallback, write_component_summary_csv


def evaluate_gym_components(
    model,
    env_id: str,
    make_env,
    custom_partial: PartialSpec | None = None,
    stats_source=None,
    n_eval_episodes: int = 10,
    seed: int = 0,
    deterministic: bool = True,
):
    return evaluate_policy_components(
        model=model,
        env_id=env_id,
        make_env=make_env,
        custom_partial=custom_partial,
        stats_source=stats_source,
        n_eval_episodes=n_eval_episodes,
        seed=seed,
        deterministic=deterministic,
        component_keys=component_keys(custom_partial),
        model_observation=policy_observation,
        action_converter=lambda env, action: action_for_space(env.action_space, action),
    )


def component_keys(custom_partial: PartialSpec | None = None) -> tuple[str, ...]:
    return custom_partial.component_keys if custom_partial is not None else ()


def component_fieldnames(custom_partial: PartialSpec | None = None) -> list[str]:
    fields = ["timesteps"]
    for key in ["total", "partial", "residual", *component_keys(custom_partial), "length"]:
        fields.extend([f"mean_{key}", f"std_{key}"])
    return fields


def write_gym_component_summary(path: Path, timestep: int, stats: dict, custom_partial: PartialSpec | None = None):
    write_component_summary_csv(path, timestep, stats, component_fieldnames(custom_partial))


class GymComponentEvalCallback(ComponentEvalCallback):
    def __init__(
        self,
        log_path: Path,
        env_id: str,
        make_env,
        custom_partial: PartialSpec | None,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(log_path, eval_freq, n_eval_episodes, seed=seed, verbose=verbose)
        self.env_id = env_id
        self.make_env = make_env
        self.custom_partial = custom_partial

    def component_fieldnames(self) -> list[str]:
        return component_fieldnames(self.custom_partial)

    def evaluate_components(self) -> dict:
        return evaluate_gym_components(
            self.model,
            self.env_id,
            make_env=self.make_env,
            custom_partial=self.custom_partial,
            stats_source=self.training_env,
            n_eval_episodes=self.n_eval_episodes,
            seed=self.seed + self.num_timesteps,
            deterministic=True,
        )

    def write_summary(self, stats: dict) -> None:
        write_gym_component_summary(self.log_path, self.num_timesteps, stats, self.custom_partial)

    def log_message(self, stats: dict) -> str:
        return (
            "Gym component eval "
            f"env={self.env_id} t={self.num_timesteps}: "
            f"total={stats['mean_total']:.3f}, "
            f"partial={stats['mean_partial']:.3f}, "
            f"residual={stats['mean_residual']:.3f}"
        )
