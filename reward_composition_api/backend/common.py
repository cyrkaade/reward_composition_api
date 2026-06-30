from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from reward_composition_api.partial_reward import build_builtin_registry
from reward_composition_api.registry import PartialSpec, load_partial_reference
from reward_composition_api.reward_models import (
    choose_query_pairs,
    dropout_active_learning_pairs,
    ensemble_active_learning_pairs,
    fragment_trajectories,
    partial_reward_tensor,
    pretrain_reward_model,
    random_query_pairs,
    rate_pairs_from_true_reward,
    rated_pairs_to_tensors,
    reward_model_io_stats,
    split_preference_k_folds,
    train_preference_reward_ensemble,
    train_preference_reward_model,
    validate_preference_reward_model,
)

__all__ = [
    "SaveVecNormalizeOnBest",
    "choose_query_pairs",
    "dropout_active_learning_pairs",
    "ensemble_active_learning_pairs",
    "fragment_trajectories",
    "include_partial_feature",
    "learn_policy",
    "load_eval_curve",
    "load_vecnormalize_eval_env",
    "make_raw_eval_env",
    "normalize_obs",
    "partial_reward_tensor",
    "plot_true_reward_curve",
    "policy_training_schedule",
    "pretrain_reward_model",
    "query_schedule",
    "random_query_pairs",
    "rate_pairs_from_true_reward",
    "rated_pairs_to_tensors",
    "resolve_custom_partial",
    "reward_model_io_stats",
    "smooth_curve",
    "split_preference_k_folds",
    "summarize_component_rows",
    "train_preference_reward_ensemble",
    "train_preference_reward_model",
    "validate_preference_reward_model",
]


class SaveVecNormalizeOnBest(BaseCallback):
    def __init__(self, env: VecNormalize, save_path: Path):
        super().__init__()
        self.env = env
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.env.save(self.save_path)
        return True


def normalize_obs(stats_source, observation):
    observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    if isinstance(stats_source, VecNormalize):
        return stats_source.normalize_obs(observation)
    return observation


def resolve_custom_partial(config) -> PartialSpec | None:
    if not config.partial:
        return None
    registry = build_builtin_registry()
    return load_partial_reference(config.partial, config.suite, registry)


def include_partial_feature(config) -> bool:
    if config.include_partial_feature is not None:
        return bool(config.include_partial_feature)
    return config.mode in {"naive", "delta"}


def make_raw_eval_env(make_raw_env, env_id: str):
    return DummyVecEnv([lambda: Monitor(make_raw_env(env_id))])


def load_vecnormalize_eval_env(env_id: str, stats_path: Path, make_raw_eval_env_fn) -> VecNormalize:
    env = VecNormalize.load(stats_path, make_raw_eval_env_fn(env_id))
    env.training = False
    env.norm_reward = False
    return env


def summarize_component_rows(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    stats = {}
    for key in keys:
        values = np.asarray([row.get(key, 0.0) for row in rows], dtype=np.float64)
        stats[f"mean_{key}"] = float(values.mean())
        stats[f"std_{key}"] = float(values.std())
    return stats


def query_schedule(query_budget: int, rounds: int) -> list[int]:
    unit = query_budget // rounds
    schedule = [unit] * rounds
    for i in range(query_budget - sum(schedule)):
        schedule[i % len(schedule)] += 1
    return schedule


def policy_training_schedule(total_timesteps: int, rounds: int, timesteps_per_round: int | None = None) -> list[int]:
    if timesteps_per_round is not None:
        return [timesteps_per_round] * rounds

    policy_steps_per_round = total_timesteps // rounds
    leftover_policy_steps = total_timesteps - policy_steps_per_round * rounds
    return [
        policy_steps_per_round + (leftover_policy_steps if round_index == rounds - 1 else 0)
        for round_index in range(rounds)
    ]


def learn_policy(
    model,
    total_timesteps: int,
    callback,
    progress_bar: bool,
    reset_num_timesteps: bool = True,
    log_interval: int | None = None,
) -> None:
    if total_timesteps <= 0:
        return

    learn_kwargs = {
        "total_timesteps": int(total_timesteps),
        "callback": callback,
        "progress_bar": progress_bar,
        "reset_num_timesteps": reset_num_timesteps,
    }
    if log_interval is not None:
        learn_kwargs["log_interval"] = log_interval
    model.learn(**learn_kwargs)


def load_eval_curve(evaluations_path: Path):
    if not evaluations_path.exists():
        raise FileNotFoundError(f"No evaluation log found at {evaluations_path}")
    eval_data = np.load(evaluations_path)
    return eval_data["timesteps"], eval_data["results"].mean(axis=1)


def smooth_curve(values, window):
    if window <= 1 or len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_true_reward_curve(
    evaluations_path: Path,
    output_path: Path,
    total_timesteps: int,
    plot_mode: str,
    smooth_window: int,
    x_scale: float,
    x_label: str,
    y_floor: float | None = None,
) -> None:
    raw_timesteps, raw_rewards = load_eval_curve(evaluations_path)
    timesteps = raw_timesteps / x_scale
    rewards = np.maximum.accumulate(raw_rewards) if plot_mode == "best" else raw_rewards
    rewards = smooth_curve(rewards, smooth_window)

    fig, ax = plt.subplots(figsize=(9, 5))
    label = "PPO best true reward" if plot_mode == "best" else "PPO true reward"
    ax.plot(timesteps, rewards, color="#2f6f9f", linewidth=2.2, label=label)
    ax.set_xlim(0, max(total_timesteps / x_scale, float(timesteps.max()) if len(timesteps) else 1.0, 1.0))
    ax.set_xlabel(x_label)

    if y_floor is not None:
        y_min = min(y_floor, int(np.floor(rewards.min() / 2.0) * 2))
        y_max = 0
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(0, y_min - 0.001, -2))
    else:
        y_min = float(np.min(rewards))
        y_max = float(np.max(rewards))
        if y_min == y_max:
            pad = max(abs(y_min) * 0.1, 1.0)
            y_min -= pad
            y_max += pad
        else:
            pad = max((y_max - y_min) * 0.08, 1.0)
            y_min -= pad
            y_max += pad
        ax.set_ylim(y_min, y_max)

    ax.set_ylabel("True reward")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
