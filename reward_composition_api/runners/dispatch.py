from __future__ import annotations

from reward_composition_api.config import (
    ATARI_SUITE,
    BOX2D_SUITE,
    GYM_SUITE,
    MUJOCO_SUITE,
    ExperimentConfig,
    normalize_experiment_config,
)
from reward_composition_api.errors import ConfigError
from reward_composition_api.results import RunResult


def run_experiment(config: ExperimentConfig) -> RunResult:
    normalized = normalize_experiment_config(config)
    if normalized.suite == MUJOCO_SUITE:
        from .mujoco import run_mujoco_experiment

        return run_mujoco_experiment(normalized)
    if normalized.suite == ATARI_SUITE:
        from .atari import run_atari_experiment

        return run_atari_experiment(normalized)
    if normalized.suite in (BOX2D_SUITE, GYM_SUITE):
        from .gymnasium import run_gym_experiment

        return run_gym_experiment(normalized)
    raise ConfigError(f"Unsupported train suite '{normalized.suite}'")
