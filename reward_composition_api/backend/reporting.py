from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .common import load_eval_curve, plot_true_reward_curve


@dataclass(frozen=True)
class BackendRunPaths:
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


class ComponentEvalCallback(BaseCallback):
    def __init__(
        self,
        log_path: Path,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.log_path = Path(log_path)
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed

    def component_fieldnames(self) -> list[str]:
        raise NotImplementedError

    def evaluate_components(self) -> dict:
        raise NotImplementedError

    def write_summary(self, stats: dict) -> None:
        raise NotImplementedError

    def log_message(self, stats: dict) -> str:
        return ""

    def _init_callback(self) -> None:
        self.log_path.parent.mkdir(exist_ok=True, parents=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.component_fieldnames())
                writer.writeheader()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        stats = self.evaluate_components()
        self.write_summary(stats)
        if self.verbose:
            message = self.log_message(stats)
            if message:
                print(message)
        return True


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
    paths = BackendRunPaths(run_dir)
    final_policy = model
    final_eval_env = eval_env
    if config.final_policy == "best" and paths.best_model.exists():
        if load_best_stats and paths.best_vecnormalize.exists():
            final_eval_env.close()
            final_eval_env = load_eval_env_fn(config.env_id, paths.best_vecnormalize)
        final_policy = load_policy_fn(paths.best_model, env=final_eval_env, device=config.device)
    return final_policy, final_eval_env
