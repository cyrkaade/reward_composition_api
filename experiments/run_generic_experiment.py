from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_composition_api import ExperimentConfig, run_experiment
from reward_composition_api.config import (
    ATARI_PARTIAL_SOURCES,
    ATARI_SUITE,
    BOX2D_SUITE,
    FINAL_POLICIES,
    GYM_SUITE,
    MUJOCO_PARTIAL_PROFILES,
    MUJOCO_PRESETS,
    MUJOCO_SUITE,
    PLOT_MODES,
    TRAIN_SUITES,
    suite_supported_envs,
)
from reward_composition_api.errors import RewardCompositionError
from reward_composition_api.parsing import parse_int_tuple, parse_key_value_mapping


METHOD_ALIASES = {
    "true": "true",
    "true_reward": "true",
    "true-reward": "true",
    "partial": "partial",
    "partial_only": "partial",
    "partial-only": "partial",
    "feedback": "feedback",
    "vanilla": "feedback",
    "vanilla_rlhf": "feedback",
    "vanilla-rlhf": "feedback",
    "full": "feedback",
    "rlhf": "feedback",
    "naive": "naive",
    "delta": "delta",
}

DEFAULT_VARIANTS = {
    "true": "true_reference",
    "partial": "partial_only",
    "feedback": "vanilla",
    "naive": "naive",
    "delta": "delta",
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        suite = infer_suite(args.env_id) if args.suite == "auto" else args.suite
        methods = parse_methods(args.methods, args.method)
        seeds = parse_seeds(args.seed, args.seeds)
        validate_partial_request(suite, methods, args.closed_form_file)
        total_timesteps = requested_timesteps(args)
        log_dir = Path(args.log_dir) if args.log_dir else Path("logs") / args.experiment_name

        print(f"experiment={args.experiment_name} suite={suite} env={args.env_id}")
        print(f"methods={','.join(methods)} seeds={','.join(str(seed) for seed in seeds)}")

        for seed in seeds:
            for method in methods:
                run_name = make_run_name(args.env_id, method, total_timesteps, seed, args.run_name, len(methods) * len(seeds))
                config = ExperimentConfig(
                    suite=suite,
                    env_id=args.env_id,
                    mode=method,
                    variant_name=DEFAULT_VARIANTS[method],
                    timesteps=total_timesteps,
                    run_name=run_name,
                    log_dir=log_dir,
                    seed=seed,
                    n_envs=args.n_envs,
                    device=args.device,
                    eval_freq=args.eval_interval,
                    n_eval_episodes=args.eval_episodes,
                    final_eval_episodes=args.final_eval_episodes or args.eval_episodes,
                    stop_reward=args.stop_reward,
                    final_policy=args.final_policy,
                    plot_mode=args.plot_mode,
                    progress_bar=args.progress_bar,
                    preset=args.preset,
                    partial_profile=args.partial_profile,
                    partial_source=args.partial_source,
                    partial=args.closed_form_file,
                    rlhf_rounds=args.rlhf_rounds,
                    query_budget=args.n_pairs,
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
                    reward_model_epochs=args.reward_epochs,
                    reward_model_patience=args.reward_patience,
                    reward_model_batch_size=args.reward_batch_size,
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

                if args.dry_run:
                    print(f"dry-run {method} seed={seed}: {log_dir / run_name}")
                    continue
                run_experiment(config)
        return 0
    except (RewardCompositionError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one generic reward-composition experiment or a small method/seed grid.")
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--suite", choices=("auto", *TRAIN_SUITES), default="auto")
    parser.add_argument("--closed-form-file", default=None, help="Partial/closed-form reward module file or built-in partial name.")
    parser.add_argument("--experiment-name", default="generic_experiment")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--method", action="append", default=None, help="Method alias. Can be repeated.")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods, e.g. delta naive vanilla partial true.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--initial-timesteps", type=int, default=0)
    parser.add_argument("--policy-timesteps-per-round", type=int, default=None)
    parser.add_argument("--final-policy-timesteps", type=int, default=0)
    parser.add_argument("--rlhf-rounds", type=int, default=5)
    parser.add_argument("--n-pairs", "--query-budget", dest="n_pairs", type=int, default=1400)
    parser.add_argument("--collection-timesteps", type=int, default=None)
    parser.add_argument("--fragment-length", type=int, default=None)
    parser.add_argument("--active-learning", action="store_true", default=None)
    parser.add_argument("--no-active-learning", action="store_false", dest="active_learning")
    parser.add_argument("--dropout-samples", type=int, default=8)
    parser.add_argument("--dropout-p", type=float, default=0.25)
    parser.add_argument("--active-learning-batches", type=int, default=512)
    parser.add_argument("--reward-hidden-sizes", type=parse_int_tuple, default=(200,))
    parser.add_argument("--reward-epochs", type=int, default=100)
    parser.add_argument("--reward-patience", type=int, default=10)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    parser.add_argument("--reward-model-lr", type=float, default=0.01)
    parser.add_argument("--normalize-model-reward", action="store_true", default=False)
    parser.add_argument("--model-reward-scale", type=float, default=1.0)
    parser.add_argument("--model-reward-min", type=float, default=None)
    parser.add_argument("--model-reward-max", type=float, default=None)
    parser.add_argument("--model-reward-target-mean", type=float, default=0.0)
    parser.add_argument("--model-reward-target-std", type=float, default=1.0)
    parser.add_argument("--include-partial-feature", action="store_true", dest="include_partial_feature", default=None)
    parser.add_argument("--no-partial-feature", action="store_false", dest="include_partial_feature")
    parser.add_argument("--pretrain-reward-model", action="store_true")
    parser.add_argument("--pretrain-target", choices=("partial", "residual", "true"), default="partial")
    parser.add_argument("--pretrain-epochs", type=int, default=25)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--final-eval-episodes", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--stop-reward", type=float, default=None)
    parser.add_argument("--final-policy", choices=FINAL_POLICIES, default="best")
    parser.add_argument("--plot-mode", choices=PLOT_MODES, default="best")
    parser.add_argument("--policy-learning-kwargs", type=parse_key_value_mapping, default=None)
    parser.add_argument("--policy-log-interval", type=int, default=None)
    parser.add_argument("--preset", choices=MUJOCO_PRESETS, default=None)
    parser.add_argument("--partial-profile", choices=MUJOCO_PARTIAL_PROFILES, default="default")
    parser.add_argument("--partial-source", choices=ATARI_PARTIAL_SOURCES, default="life_loss")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_methods(methods: list[str] | None, repeated_methods: list[str] | None) -> list[str]:
    raw_methods = []
    for source in (methods, repeated_methods):
        for item in source or []:
            raw_methods.extend(part.strip() for part in item.split(",") if part.strip())
    if not raw_methods:
        raw_methods = ["delta"]

    normalized = []
    for method in raw_methods:
        key = method.lower()
        if key not in METHOD_ALIASES:
            choices = ", ".join(sorted(METHOD_ALIASES))
            raise ValueError(f"unknown method '{method}'. Choices: {choices}")
        normalized.append(METHOD_ALIASES[key])
    return list(dict.fromkeys(normalized))


def parse_seeds(seed: int, seeds: list[str] | None) -> list[int]:
    if not seeds:
        return [seed]
    values = []
    for item in seeds:
        values.extend(int(part.strip()) for part in item.split(",") if part.strip())
    return values


def infer_suite(env_id: str) -> str:
    if env_id.startswith("ALE/"):
        return ATARI_SUITE
    if env_id in suite_supported_envs(MUJOCO_SUITE):
        return MUJOCO_SUITE
    if env_id in suite_supported_envs(BOX2D_SUITE):
        return BOX2D_SUITE
    return GYM_SUITE


def validate_partial_request(suite: str, methods: list[str], closed_form_file: str | None) -> None:
    needs_partial = any(method in {"partial", "naive", "delta"} for method in methods)
    if suite in {BOX2D_SUITE, GYM_SUITE} and needs_partial and not closed_form_file:
        raise ValueError(f"{suite} partial/naive/delta runs need --closed-form-file")


def requested_timesteps(args) -> int:
    phase_total = (
        args.initial_timesteps
        + (args.policy_timesteps_per_round or 0) * args.rlhf_rounds
        + args.final_policy_timesteps
    )
    if args.timesteps is None:
        return phase_total if phase_total else 5_000_000
    return max(args.timesteps, phase_total)


def make_run_name(env_id: str, method: str, timesteps: int, seed: int, explicit_name: str | None, run_count: int) -> str:
    if explicit_name and run_count == 1:
        return explicit_name
    prefix = explicit_name or slugify(env_id)
    return f"{prefix}_{DEFAULT_VARIANTS[method]}_{format_steps(timesteps)}_seed{seed}"


def format_steps(timesteps: int) -> str:
    if timesteps >= 1_000_000 and timesteps % 1_000_000 == 0:
        return f"{timesteps // 1_000_000}m"
    if timesteps % 1000 == 0:
        return f"{timesteps // 1000}k"
    return str(timesteps)


def slugify(env_id: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in env_id.rsplit("-", 1)[0]).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
