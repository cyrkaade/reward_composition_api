from __future__ import annotations

from .config import ExperimentConfig, SummaryConfig, SweepConfig
from .results import PlannedRun, RunResult, SummaryResult, SweepResult

__all__ = [
    "ExperimentConfig",
    "SweepConfig",
    "SummaryConfig",
    "RunResult",
    "PlannedRun",
    "SweepResult",
    "SummaryResult",
    "run_experiment",
    "plan_sweep",
    "run_sweep",
    "summarize_runs",
]


def run_experiment(config: ExperimentConfig) -> RunResult:
    from .runners.dispatch import run_experiment as _run_experiment

    return _run_experiment(config)


def plan_sweep(config: SweepConfig) -> list[PlannedRun]:
    from .sweeps import plan_sweep as _plan_sweep

    return _plan_sweep(config)


def run_sweep(config: SweepConfig) -> SweepResult:
    from .sweeps import run_sweep as _run_sweep

    return _run_sweep(config)


def summarize_runs(config: SummaryConfig) -> SummaryResult:
    from .summaries import summarize_runs as _summarize_runs

    return _summarize_runs(config)
