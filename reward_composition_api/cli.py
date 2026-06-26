from __future__ import annotations

import argparse
import sys

import numpy as np

from . import run_experiment, run_sweep, summarize_runs
from .config import (
    ATARI_PARTIAL_SOURCES,
    ATARI_SUITE,
    BOX2D_SUITE,
    DEVICES,
    FINAL_POLICIES,
    GYM_SUITE,
    MUJOCO_PARTIAL_PROFILES,
    MUJOCO_PRESETS,
    MUJOCO_SUITE,
    PLOT_MODES,
    PRETRAIN_TARGETS,
    SUITES,
    TRAIN_SUITES,
    ExperimentConfig,
    SummaryConfig,
    SweepConfig,
    normalize_experiment_config,
    suite_supported_envs,
)
from .errors import RewardCompositionError
from .parsing import parse_int_tuple, parse_key_value_mapping
from .partials import build_builtin_registry, partials_for_display
from .registry import load_partial_reference


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return _handle_train(args)
        if args.command == "sweep":
            return _handle_sweep(args)
        if args.command == "summarize":
            return _handle_summarize(args)
        if args.command == "list-envs":
            return _handle_list_envs(args)
        if args.command == "list-partials":
            return _handle_list_partials(args)
        if args.command == "validate-partial":
            return _handle_validate_partial(args)
        parser.print_help()
        return 1
    except RewardCompositionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m reward_composition_api")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run one experiment")
    add_train_args(train_parser)

    sweep_parser = subparsers.add_parser("sweep", help="Plan or execute ablation-style sweeps")
    add_sweep_args(sweep_parser)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize run metadata into CSV files")
    summarize_parser.add_argument("--suite", choices=TRAIN_SUITES, default=MUJOCO_SUITE)
    summarize_parser.add_argument("--root", default=None)
    summarize_parser.add_argument("--summary-csv", default=None)
    summarize_parser.add_argument("--aggregate-csv", default=None)

    list_envs_parser = subparsers.add_parser("list-envs", help="List supported environments")
    list_envs_parser.add_argument("--suite", choices=SUITES, default=None)

    list_partials_parser = subparsers.add_parser("list-partials", help="List built-in partial rewards")
    list_partials_parser.add_argument("--suite", choices=TRAIN_SUITES, default=None)

    validate_parser = subparsers.add_parser("validate-partial", help="Import and smoke-check a partial reward")
    validate_parser.add_argument("--suite", choices=TRAIN_SUITES, default=MUJOCO_SUITE)
    validate_parser.add_argument("--env-id", default=None)
    validate_parser.add_argument("--partial", required=True)

    return parser


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", choices=TRAIN_SUITES, default=MUJOCO_SUITE)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--mode", choices=("true", "partial", "feedback", "naive", "delta"), default="delta")
    parser.add_argument("--variant-name", default=None)
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", choices=DEVICES, default="auto")
    parser.add_argument("--eval-freq", "--eval-interval", dest="eval_freq", type=int, default=100_000)
    parser.add_argument("--n-eval-episodes", "--eval-episodes", dest="n_eval_episodes", type=int, default=None)
    parser.add_argument("--final-eval-episodes", type=int, default=None)
    parser.add_argument("--stop-reward", type=float, default=None)
    parser.add_argument("--final-policy", choices=FINAL_POLICIES, default="best")
    parser.add_argument("--plot-mode", choices=PLOT_MODES, default="best")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--preset", choices=MUJOCO_PRESETS, default=None)
    parser.add_argument("--partial-profile", choices=MUJOCO_PARTIAL_PROFILES, default="default")
    parser.add_argument("--partial-source", choices=ATARI_PARTIAL_SOURCES, default="life_loss")
    parser.add_argument("--partial", "--closed-form-file", dest="partial", default=None)

    parser.add_argument("--rlhf-rounds", type=int, default=5)
    parser.add_argument("--query-budget", "--n-pairs", dest="query_budget", type=int, default=1400)
    parser.add_argument("--initial-timesteps", type=int, default=0)
    parser.add_argument("--policy-timesteps-per-round", type=int, default=None)
    parser.add_argument("--final-policy-timesteps", type=int, default=0)
    parser.add_argument("--policy-log-interval", type=int, default=None)
    parser.add_argument("--policy-learning-kwargs", type=parse_key_value_mapping, default=None)
    parser.add_argument("--collection-timesteps", type=int, default=None)
    parser.add_argument("--fragment-length", type=int, default=None)
    parser.add_argument("--active-learning", action="store_true", default=None)
    parser.add_argument("--no-active-learning", action="store_false", dest="active_learning")
    parser.add_argument("--dropout-samples", type=int, default=8)
    parser.add_argument("--dropout-p", type=float, default=0.25)
    parser.add_argument("--active-learning-batches", type=int, default=512)

    parser.add_argument("--reward-hidden-sizes", type=parse_int_tuple, default=(200,))
    parser.add_argument("--reward-model-lr", type=float, default=0.01)
    parser.add_argument("--reward-model-epochs", "--reward-epochs", dest="reward_model_epochs", type=int, default=100)
    parser.add_argument("--reward-model-patience", type=int, default=10)
    parser.add_argument("--reward-model-batch-size", "--reward-batch-size", dest="reward_model_batch_size", type=int, default=32)
    parser.add_argument("--model-reward-scale", type=float, default=1.0)
    parser.add_argument("--model-reward-min", type=parse_optional_float, default=None)
    parser.add_argument("--model-reward-max", type=parse_optional_float, default=None)
    parser.add_argument("--normalize-model-reward", action="store_true", default=False)
    parser.add_argument("--no-normalize-model-reward", action="store_false", dest="normalize_model_reward")
    parser.add_argument("--model-reward-target-mean", type=float, default=0.0)
    parser.add_argument("--model-reward-target-std", type=float, default=1.0)
    parser.add_argument("--include-partial-feature", action="store_true", dest="include_partial_feature", default=None)
    parser.add_argument("--no-partial-feature", action="store_false", dest="include_partial_feature")

    parser.add_argument("--pretrain-reward-model", action="store_true")
    parser.add_argument("--pretrain-target", choices=PRETRAIN_TARGETS, default="partial")
    parser.add_argument("--pretrain-epochs", type=int, default=25)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)


def add_sweep_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", choices=TRAIN_SUITES, default=MUJOCO_SUITE)
    parser.add_argument("--env-ids", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2])
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--no-skip-completed", action="store_false", dest="skip_completed")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--n-eval-episodes", type=int, default=None)
    parser.add_argument("--final-eval-episodes", type=int, default=None)
    parser.add_argument("--query-budget", type=int, default=1400)
    parser.add_argument("--rlhf-rounds", type=int, default=5)
    parser.add_argument("--collection-timesteps", type=int, default=None)
    parser.add_argument("--fragment-length", type=int, default=None)
    parser.add_argument("--reward-model-epochs", type=int, default=100)
    parser.add_argument("--reward-model-patience", type=int, default=10)
    parser.add_argument("--reward-model-batch-size", type=int, default=32)
    parser.add_argument("--active-learning-batches", type=int, default=512)
    parser.add_argument("--pretrain-epochs", type=int, default=25)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=DEVICES, default="auto")
    parser.add_argument("--preset", choices=MUJOCO_PRESETS, default=None)
    parser.add_argument("--partial-source", choices=ATARI_PARTIAL_SOURCES, default="life_loss")
    parser.add_argument("--partial", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--normalize-model-reward", action="store_true")
    parser.add_argument("--model-reward-min", type=float, default=None)
    parser.add_argument("--model-reward-max", type=float, default=None)
    parser.add_argument("--model-reward-target-mean", type=float, default=None)
    parser.add_argument("--model-reward-target-std", type=float, default=None)


def _handle_train(args) -> int:
    config = ExperimentConfig(
        suite=args.suite,
        env_id=args.env_id,
        mode=args.mode,
        variant_name=args.variant_name,
        timesteps=args.timesteps,
        run_name=args.run_name,
        log_dir=args.log_dir,
        seed=args.seed,
        n_envs=args.n_envs,
        device=args.device,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        final_eval_episodes=args.final_eval_episodes,
        stop_reward=args.stop_reward,
        final_policy=args.final_policy,
        plot_mode=args.plot_mode,
        smooth_window=args.smooth_window,
        progress_bar=args.progress_bar,
        preset=args.preset,
        partial_profile=args.partial_profile,
        partial_source=args.partial_source,
        partial=args.partial,
        rlhf_rounds=args.rlhf_rounds,
        query_budget=args.query_budget,
        initial_timesteps=args.initial_timesteps,
        policy_timesteps_per_round=args.policy_timesteps_per_round,
        final_policy_timesteps=args.final_policy_timesteps,
        policy_log_interval=args.policy_log_interval,
        policy_learning_kwargs=args.policy_learning_kwargs,
        collection_timesteps=args.collection_timesteps,
        fragment_length=args.fragment_length,
        active_learning=args.active_learning,
        dropout_samples=args.dropout_samples,
        dropout_p=args.dropout_p,
        active_learning_batches=args.active_learning_batches,
        reward_hidden_sizes=args.reward_hidden_sizes,
        reward_model_lr=args.reward_model_lr,
        reward_model_epochs=args.reward_model_epochs,
        reward_model_patience=args.reward_model_patience,
        reward_model_batch_size=args.reward_model_batch_size,
        model_reward_scale=args.model_reward_scale,
        model_reward_min=args.model_reward_min,
        model_reward_max=args.model_reward_max,
        normalize_model_reward=args.normalize_model_reward,
        model_reward_target_mean=args.model_reward_target_mean,
        model_reward_target_std=args.model_reward_target_std,
        include_partial_feature=args.include_partial_feature,
        pretrain_reward_model=args.pretrain_reward_model,
        pretrain_target=args.pretrain_target,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        pretrain_lr=args.pretrain_lr,
    )
    run_experiment(config)
    return 0


def _handle_sweep(args) -> int:
    run_sweep(
        SweepConfig(
            suite=args.suite,
            env_ids=tuple(args.env_ids) if args.env_ids else None,
            seeds=tuple(args.seeds),
            timesteps=args.timesteps,
            log_dir=args.log_dir,
            manifest=args.manifest,
            execute=args.execute,
            skip_completed=args.skip_completed,
            n_envs=args.n_envs,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            final_eval_episodes=args.final_eval_episodes,
            query_budget=args.query_budget,
            rlhf_rounds=args.rlhf_rounds,
            collection_timesteps=args.collection_timesteps,
            fragment_length=args.fragment_length,
            reward_model_epochs=args.reward_model_epochs,
            reward_model_patience=args.reward_model_patience,
            reward_model_batch_size=args.reward_model_batch_size,
            active_learning_batches=args.active_learning_batches,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_batch_size=args.pretrain_batch_size,
            pretrain_lr=args.pretrain_lr,
            device=args.device,
            preset=args.preset,
            partial_source=args.partial_source,
            progress_bar=args.progress_bar,
            normalize_model_reward=args.normalize_model_reward,
            model_reward_min=args.model_reward_min,
            model_reward_max=args.model_reward_max,
            model_reward_target_mean=args.model_reward_target_mean,
            model_reward_target_std=args.model_reward_target_std,
            partial=args.partial,
        )
    )
    return 0


def _handle_summarize(args) -> int:
    summarize_runs(SummaryConfig(suite=args.suite, root=args.root, summary_csv=args.summary_csv, aggregate_csv=args.aggregate_csv))
    return 0


def _handle_list_envs(args) -> int:
    suites = [args.suite] if args.suite else list(TRAIN_SUITES)
    for suite in suites:
        print(f"{suite}:")
        for env_id in suite_supported_envs(suite):
            print(f"  {env_id}")
    return 0


def _handle_list_partials(args) -> int:
    rows = partials_for_display(args.suite)
    for row in rows:
        print(f"{row['suite']}/{row['name']}: {row['description']}")
    return 0


def _handle_validate_partial(args) -> int:
    env_id = args.env_id or normalize_experiment_config(ExperimentConfig(suite=args.suite)).env_id
    registry = build_builtin_registry()
    partial_spec = load_partial_reference(args.partial, args.suite, registry)
    partial = partial_spec.create(env_id)
    info = {"reward_dist": 1.0, "reward_ctrl": -0.5, "lives": 3}
    partial.reset(info)
    step = partial.step(
        obs=np.zeros(4, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        next_obs=np.ones(4, dtype=np.float32),
        true_reward=1.0,
        terminated=False,
        truncated=False,
        info=info,
    )
    print(f"partial '{partial_spec.suite}/{partial_spec.name}' OK for {env_id}: partial={step.partial}")
    return 0


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", "off"}:
        return None
    return float(value)
