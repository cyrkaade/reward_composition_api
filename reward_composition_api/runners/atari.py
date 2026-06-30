from __future__ import annotations

from reward_composition_api.config import ExperimentConfig
from reward_composition_api.results import RunResult

from .experiment import AtariExperimentRunner


def run_atari_experiment(config: ExperimentConfig) -> RunResult:
    return AtariExperimentRunner(config).run()
