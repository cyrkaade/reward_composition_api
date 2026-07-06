from __future__ import annotations

from pathlib import Path

import numpy as np

from local_gym.classes.atari_reward_specs import AtariRewardSpec
from reward_composition_api.environments.vectorized import normalize_obs
from reward_composition_api.evaluation.component_evaluator import evaluate_policy_components
from reward_composition_api.registry import PartialSpec

from .reporting import ComponentEvalCallback, write_component_summary_csv


def evaluate_atari_components(
    model,
    env_id: str,
    spec: AtariRewardSpec,
    make_env,
    partial_source: str = "life_loss",
    custom_partial: PartialSpec | None = None,
    stats_source=None,
    n_eval_episodes: int = 10,
    seed: int = 0,
    deterministic: bool = True,
):
    tracker = None

    def reset_tracker(info: dict) -> None:
        nonlocal tracker
        tracker = spec.new_tracker()
        tracker.reset(info)

    def default_partial_step(_obs, _action, _new_obs, reward, _terminated, _truncated, info):
        step = tracker.step(info, true_reward=reward, partial_source=partial_source)
        return step.partial, {
            "life_loss_penalty": step.life_loss_penalty,
            "score_partial": step.score_partial,
            "lost_lives": step.lost_lives,
            "lives": step.lives,
        }

    return evaluate_policy_components(
        model=model,
        env_id=env_id,
        make_env=make_env,
        custom_partial=custom_partial,
        stats_source=stats_source,
        n_eval_episodes=n_eval_episodes,
        seed=seed,
        deterministic=deterministic,
        component_keys=_component_keys(custom_partial),
        model_observation=normalize_obs,
        action_converter=lambda _env, action: int(np.asarray(action).reshape(-1)[0]),
        default_partial_step=default_partial_step,
        reset_reward_state=reset_tracker,
    )


def _component_keys(custom_partial: PartialSpec | None) -> tuple[str, ...]:
    if custom_partial is not None:
        return custom_partial.component_keys
    return ("life_loss_penalty", "score_partial", "lost_lives", "lives")


def component_keys(custom_partial: PartialSpec | None = None) -> list[str]:
    return ["total", "partial", "residual", *_component_keys(custom_partial), "length"]


def component_fieldnames(custom_partial: PartialSpec | None = None) -> list[str]:
    fields = ["timesteps"]
    for key in component_keys(custom_partial):
        fields.extend([f"mean_{key}", f"std_{key}"])
    return fields


def write_atari_component_summary(path: Path, timestep: int, stats: dict, custom_partial: PartialSpec | None = None):
    write_component_summary_csv(path, timestep, stats, component_fieldnames(custom_partial))


class AtariComponentEvalCallback(ComponentEvalCallback):
    def __init__(
        self,
        log_path: Path,
        env_id: str,
        spec: AtariRewardSpec,
        make_env,
        partial_source: str,
        custom_partial: PartialSpec | None,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(log_path, eval_freq, n_eval_episodes, seed=seed, verbose=verbose)
        self.env_id = env_id
        self.spec = spec
        self.make_env = make_env
        self.partial_source = partial_source
        self.custom_partial = custom_partial

    def component_fieldnames(self) -> list[str]:
        return component_fieldnames(self.custom_partial)

    def evaluate_components(self) -> dict:
        return evaluate_atari_components(
            self.model,
            self.env_id,
            self.spec,
            make_env=self.make_env,
            partial_source=self.partial_source,
            custom_partial=self.custom_partial,
            stats_source=self.training_env,
            n_eval_episodes=self.n_eval_episodes,
            seed=self.seed + self.num_timesteps,
            deterministic=True,
        )

    def write_summary(self, stats: dict) -> None:
        write_atari_component_summary(self.log_path, self.num_timesteps, stats, self.custom_partial)

    def log_message(self, stats: dict) -> str:
        return (
            "Atari component eval "
            f"env={self.env_id} t={self.num_timesteps}: "
            f"total={stats['mean_total']:.3f}, "
            f"partial={stats['mean_partial']:.3f}, "
            f"residual={stats['mean_residual']:.3f}, "
            f"lost_lives={stats.get('mean_lost_lives', 0.0):.3f}"
        )
