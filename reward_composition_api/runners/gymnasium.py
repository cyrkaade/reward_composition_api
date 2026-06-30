from __future__ import annotations

from reward_composition_api.config import ExperimentConfig
from reward_composition_api.partial_reward import resolve_custom_partial
from reward_composition_api.registry import PartialSpec
from reward_composition_api.results import RunResult

from .experiment import GymExperimentRunner


def run_gym_experiment(config: ExperimentConfig) -> RunResult:
    return GymExperimentRunner(config).run()


def build_callbacks(config: ExperimentConfig, run_dir, train_env, eval_env, custom_partial: PartialSpec | None):
    return GymExperimentRunner(config, custom_partial).build_callbacks(run_dir, train_env, eval_env)


def train_true_or_partial(config: ExperimentConfig, custom_partial: PartialSpec | None) -> RunResult:
    return GymExperimentRunner(config, custom_partial).train_true_or_partial()


def train_preference_mode(config: ExperimentConfig, custom_partial: PartialSpec | None) -> RunResult:
    return GymExperimentRunner(config, custom_partial).train_preference_mode()


def save_and_report(
    config: ExperimentConfig,
    model,
    train_env,
    eval_env,
    run_dir,
    synthetic_queries: int,
    runtime=None,
    custom_partial: PartialSpec | None = None,
) -> RunResult:
    return GymExperimentRunner(config, custom_partial).save_and_report(
        model,
        train_env,
        eval_env,
        run_dir,
        synthetic_queries,
        runtime=runtime,
    )


def default_run_name(config: ExperimentConfig) -> str:
    return GymExperimentRunner.default_run_name(config)


def slugify(env_id: str) -> str:
    return GymExperimentRunner.slugify(env_id)


def _resolve_custom_partial(config: ExperimentConfig) -> PartialSpec | None:
    return resolve_custom_partial(config)
