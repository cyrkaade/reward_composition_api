from __future__ import annotations

from dataclasses import replace

from reward_composition_api.config import ExperimentConfig
from reward_composition_api.registry import PartialSpec
from reward_composition_api.results import RunResult

from .common import resolve_custom_partial
from .runners import MuJoCoExperimentRunner


def run_mujoco_experiment(config: ExperimentConfig) -> RunResult:
    return MuJoCoExperimentRunner(config).run()


def build_callbacks(config, run_dir, train_env, eval_env, spec, custom_partial):
    return MuJoCoExperimentRunner(config, spec, custom_partial).build_callbacks(run_dir, train_env, eval_env)


def train_true_or_partial(config: ExperimentConfig, spec, custom_partial: PartialSpec | None) -> RunResult:
    return MuJoCoExperimentRunner(config, spec, custom_partial).train_true_or_partial()


def train_preference_mode(config: ExperimentConfig, spec, custom_partial: PartialSpec | None) -> RunResult:
    return MuJoCoExperimentRunner(config, spec, custom_partial).train_preference_mode()


def save_and_report(
    config: ExperimentConfig,
    model,
    train_env,
    eval_env,
    run_dir,
    spec,
    synthetic_queries: int,
    runtime=None,
    custom_partial: PartialSpec | None = None,
) -> RunResult:
    return MuJoCoExperimentRunner(config, spec, custom_partial).save_and_report(
        model,
        train_env,
        eval_env,
        run_dir,
        synthetic_queries,
        runtime=runtime,
    )


def default_run_name(config: ExperimentConfig, spec) -> str:
    return MuJoCoExperimentRunner.default_run_name(config, spec)


def _resolve_custom_partial(config: ExperimentConfig) -> PartialSpec | None:
    return resolve_custom_partial(config)


def _with_run_identity(config: ExperimentConfig, run_name: str, variant_name: str) -> ExperimentConfig:
    return replace(config, run_name=run_name, variant_name=variant_name)
