"""Run outputs and evaluation: run-directory paths, component evaluation
(periodic callback + final), reward-curve loading/smoothing/plotting, and
final-policy selection."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .envs import action_for_space
from .partials import PartialSpec
from .suites import Suite


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @property
    def final_model(self) -> Path:
        return self.run_dir / "final_model"

    @property
    def vecnormalize(self) -> Path:
        return self.run_dir / "vecnormalize.pkl"

    @property
    def eval_log(self) -> Path:
        return self.run_dir / "eval" / "evaluations.npz"

    @property
    def true_reward_curve(self) -> Path:
        return self.run_dir / "true_reward_curve.png"

    @property
    def final_component_evaluation(self) -> Path:
        return self.run_dir / "eval" / "final_component_evaluation.csv"

    @property
    def metadata(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def best_model(self) -> Path:
        return self.run_dir / "best_model" / "best_model.zip"

    @property
    def best_vecnormalize(self) -> Path:
        return self.run_dir / "best_model" / "best_vecnormalize.pkl"


def summarize_component_rows(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    stats = {}
    for key in keys:
        values = np.asarray([row.get(key, 0.0) for row in rows], dtype=np.float64)
        stats[f"mean_{key}"] = float(values.mean())
        stats[f"std_{key}"] = float(values.std())
    return stats


def component_keys(custom_partial: PartialSpec | None) -> tuple[str, ...]:
    return custom_partial.component_keys if custom_partial is not None else ()


def component_fieldnames(custom_partial: PartialSpec | None) -> list[str]:
    fields = ["timesteps"]
    for key in ["total", "partial", "residual", *component_keys(custom_partial), "length"]:
        fields.extend([f"mean_{key}", f"std_{key}"])
    return fields


def evaluate_components(
    model,
    suite: Suite,
    env_id: str,
    custom_partial: PartialSpec | None,
    stats_source,
    n_eval_episodes: int,
    seed: int,
    deterministic: bool = True,
) -> dict[str, float]:
    rows = []
    env = suite.make_raw_env(env_id)
    partial = custom_partial.create(env_id) if custom_partial else None
    keys = component_keys(custom_partial)
    summary_keys = ["total", "partial", "residual", *keys, "length"]

    try:
        for episode_index in range(n_eval_episodes):
            obs, info = env.reset(seed=seed + episode_index)
            if partial is not None:
                partial.reset(info)
            done = False
            total = 0.0
            partial_total = 0.0
            length = 0
            components = {key: 0.0 for key in keys}

            while not done:
                model_obs = suite.eval_model_observation(stats_source, obs)
                action, _ = model.predict(model_obs, deterministic=deterministic)
                env_action = action_for_space(env.action_space, action)
                new_obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
                total += float(reward)
                length += 1

                if partial is not None:
                    partial_step = partial.step(obs, env_action, new_obs, reward, terminated, truncated, info)
                    partial_total += float(partial_step.partial)
                    for key, value in partial_step.components.items():
                        if key in components:
                            components[key] += float(value)
                obs = new_obs

            rows.append(
                {
                    "total": total,
                    "partial": partial_total,
                    "residual": total - partial_total,
                    "length": float(length),
                    **components,
                }
            )
    finally:
        env.close()

    return summarize_component_rows(rows, summary_keys)


def write_component_summary_csv(path: Path, timestep: int, stats: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        row = {"timesteps": timestep}
        row.update({field: stats.get(field, "") for field in fieldnames if field != "timesteps"})
        writer.writerow(row)


class ComponentEvalCallback(BaseCallback):
    def __init__(
        self,
        log_path: Path,
        suite: Suite,
        env_id: str,
        custom_partial: PartialSpec | None,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.log_path = Path(log_path)
        self.suite = suite
        self.env_id = env_id
        self.custom_partial = custom_partial
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed

    def _init_callback(self) -> None:
        self.log_path.parent.mkdir(exist_ok=True, parents=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=component_fieldnames(self.custom_partial))
                writer.writeheader()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        stats = evaluate_components(
            self.model,
            self.suite,
            self.env_id,
            custom_partial=self.custom_partial,
            stats_source=self.training_env,
            n_eval_episodes=self.n_eval_episodes,
            seed=self.seed + self.num_timesteps,
        )
        write_component_summary_csv(self.log_path, self.num_timesteps, stats, component_fieldnames(self.custom_partial))
        if self.verbose:
            values = ", ".join(f"{key}={stats.get(f'mean_{key}', 0.0):.3f}" for key in self.suite.summary_component_keys)
            print(f"{self.suite.name} component eval env={self.env_id} t={self.num_timesteps}: {values}")
        return True


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


def report_eval_curve(
    eval_log_path: Path,
    plot_path: Path,
    total_timesteps: int,
    plot_mode: str,
    smooth_window: int,
    x_scale: float,
    x_label: str,
    y_floor: float | None = None,
) -> tuple[float | None, int | None]:
    if not eval_log_path.exists():
        return None, None

    plot_true_reward_curve(
        eval_log_path,
        plot_path,
        total_timesteps,
        plot_mode,
        smooth_window,
        x_scale=x_scale,
        x_label=x_label,
        y_floor=y_floor,
    )
    eval_timesteps, eval_rewards = load_eval_curve(eval_log_path)
    best_idx = int(np.argmax(eval_rewards))
    best_logged_reward = float(eval_rewards[best_idx])
    best_logged_timestep = int(eval_timesteps[best_idx])
    print(f"Best logged true reward: {best_logged_reward:.3f} at {best_logged_timestep} timesteps")
    return best_logged_reward, best_logged_timestep


def select_final_policy(
    config,
    model,
    eval_env,
    run_dir: Path,
    load_eval_env_fn,
    load_policy_fn,
    load_best_stats: bool,
):
    paths = RunPaths(run_dir)
    final_policy = model
    final_eval_env = eval_env
    if config.final_policy == "best" and paths.best_model.exists():
        if load_best_stats and paths.best_vecnormalize.exists():
            final_eval_env.close()
            final_eval_env = load_eval_env_fn(config.env_id, paths.best_vecnormalize)
        final_policy = load_policy_fn(paths.best_model, env=final_eval_env, device=config.device)
    return final_policy, final_eval_env
