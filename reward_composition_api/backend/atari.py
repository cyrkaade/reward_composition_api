from __future__ import annotations

from reward_composition_api.config import ExperimentConfig
from reward_composition_api.evaluation.atari import _component_keys as atari_component_keys
from reward_composition_api.partial_reward import resolve_custom_partial
from reward_composition_api.registry import PartialSpec
from reward_composition_api.results import RunResult

from .runners import AtariExperimentRunner


def run_atari_experiment(config: ExperimentConfig) -> RunResult:
    return AtariExperimentRunner(config).run()


def build_callbacks(config, run_dir, train_env, eval_env, spec, custom_partial):
    return AtariExperimentRunner(config, spec, custom_partial).build_callbacks(run_dir, train_env, eval_env)


def train_true_or_partial(config: ExperimentConfig, spec, custom_partial: PartialSpec | None) -> RunResult:
    return AtariExperimentRunner(config, spec, custom_partial).train_true_or_partial()


def train_preference_mode(config: ExperimentConfig, spec, custom_partial: PartialSpec | None) -> RunResult:
    return AtariExperimentRunner(config, spec, custom_partial).train_preference_mode()


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
    return AtariExperimentRunner(config, spec, custom_partial).save_and_report(
        model,
        train_env,
        eval_env,
        run_dir,
        synthetic_queries,
        runtime=runtime,
    )


def default_run_name(config: ExperimentConfig, spec) -> str:
    return AtariExperimentRunner.default_run_name(config, spec)


def _partial_keys(config: ExperimentConfig, custom_partial: PartialSpec | None):
    return AtariExperimentRunner(config, custom_partial=custom_partial).partial_keys()


def _component_keys(custom_partial: PartialSpec | None):
    return atari_component_keys(custom_partial)


def _resolve_custom_partial(config: ExperimentConfig) -> PartialSpec | None:
    return resolve_custom_partial(config)
