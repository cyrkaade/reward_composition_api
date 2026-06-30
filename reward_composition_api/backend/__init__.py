from __future__ import annotations

from reward_composition_api.config import ExperimentConfig
from reward_composition_api.results import RunResult
from reward_composition_api.runners.dispatch import run_experiment as _run_experiment


def run_experiment(config: ExperimentConfig) -> RunResult:
    return _run_experiment(config)
