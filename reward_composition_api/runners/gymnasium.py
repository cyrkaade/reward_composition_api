from __future__ import annotations

from reward_composition_api.config import ExperimentConfig
from reward_composition_api.results import RunResult

from .experiment import GymExperimentRunner


def run_gym_experiment(config: ExperimentConfig) -> RunResult:
    return GymExperimentRunner(config).run()
