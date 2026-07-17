"""rcomp — compare RLHF reward-composition strategies with PPO.

Heavy dependencies (torch, stable-baselines3, matplotlib) are imported
lazily by the functions that need them.
"""

from __future__ import annotations

from .config import ConfigError, ExperimentConfig, RewardCompositionError, SummaryConfig, SweepConfig
from .partials import PartialRegistryError

__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "PartialRegistryError",
    "RewardCompositionError",
    "RunResult",
    "SummaryConfig",
    "SweepConfig",
    "run_experiment",
    "run_sweep",
    "summarize_runs",
]


def run_experiment(config: ExperimentConfig):
    from .trainer import run_experiment as _run_experiment

    return _run_experiment(config)


def run_sweep(config: SweepConfig):
    from .sweeps import run_sweep as _run_sweep

    return _run_sweep(config)


def summarize_runs(config: SummaryConfig):
    from .sweeps import summarize_runs as _summarize_runs

    return _summarize_runs(config)


def __getattr__(name: str):
    if name == "RunResult":
        from .trainer import RunResult

        return RunResult
    raise AttributeError(f"module 'rcomp' has no attribute '{name}'")
