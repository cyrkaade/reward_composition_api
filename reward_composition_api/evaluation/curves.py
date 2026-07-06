from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
