from __future__ import annotations

from reward_composition_api.config import ExperimentConfig
from reward_composition_api.results import RunResult

from .experiment import MuJoCoExperimentRunner


def run_mujoco_experiment(config: ExperimentConfig) -> RunResult:
    return MuJoCoExperimentRunner(config).run()
