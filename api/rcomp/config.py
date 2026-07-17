from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .suites import SUITE_NAMES, get_suite


class RewardCompositionError(Exception):
    pass


class ConfigError(RewardCompositionError):
    pass


MUJOCO_SUITE = "mujoco"
ATARI_SUITE = "atari"
BOX2D_SUITE = "box2d"
GYM_SUITE = "gym"
TRAIN_SUITES = SUITE_NAMES

TRAIN_MODES = ("true", "partial", "feedback", "naive", "delta")
PREFERENCE_MODES = ("feedback", "naive", "delta")
PARTIAL_REQUIRED_MODES = ("partial", "naive", "delta")

FINAL_POLICIES = ("best", "last")
PLOT_MODES = ("best", "raw")
DEVICES = ("auto", "cpu", "cuda")
PRETRAIN_TARGETS = ("partial", "residual", "true")
ACTIVE_QUERY_STRATEGIES = ("auto", "dropout", "ensemble")


def _f(default, help: str, **meta):
    """Dataclass field with CLI metadata (help/choices/nargs/parse)."""
    return field(default=default, metadata={"help": help, **meta})


@dataclass(frozen=True)
class ExperimentConfig:
    suite: str = _f(MUJOCO_SUITE, "Environment suite", choices=TRAIN_SUITES)
    env_id: str | None = _f(None, "Gymnasium env id (suite default when omitted)")
    mode: str = _f("delta", "Reward composition mode", choices=TRAIN_MODES)
    variant_name: str | None = _f(None, "Variant label stored in metadata (defaults to mode)")
    timesteps: int = _f(5_000_000, "Total PPO timesteps")
    run_name: str | None = _f(None, "Run directory name (auto-generated when omitted)")
    log_dir: str | Path | None = _f(None, "Root log directory (suite default when omitted)")
    seed: int = _f(0, "Random seed")
    n_envs: int = _f(8, "Number of vectorized training envs")
    device: str = _f("auto", "Torch device", choices=DEVICES)
    eval_freq: int = _f(100_000, "Timesteps between evaluations")
    n_eval_episodes: int | None = _f(None, "Episodes per periodic evaluation (suite default when omitted)")
    final_eval_episodes: int | None = _f(None, "Episodes for the final evaluation (suite default when omitted)")
    stop_reward: float | None = _f(None, "Stop training once eval reward reaches this threshold")
    final_policy: str = _f("best", "Policy evaluated at the end", choices=FINAL_POLICIES)
    plot_mode: str = _f("best", "Reward-curve plotting mode", choices=PLOT_MODES)
    smooth_window: int = _f(5, "Smoothing window for the reward curve")
    progress_bar: bool = _f(False, "Show the stable-baselines3 progress bar")

    preset: str | None = _f(None, "MuJoCo PPO preset", choices=("auto", "generic", "reacher"))
    partial: str | None = _f(None, "Manual partial reward reference: <module> or <module>:<name>")

    rlhf_rounds: int = _f(5, "Number of RLHF rounds")
    query_budget: int = _f(1400, "Total synthetic preference queries")
    initial_timesteps: int = _f(0, "PPO timesteps before the first RLHF round")
    policy_timesteps_per_round: int | None = _f(None, "Override PPO timesteps per RLHF round")
    final_policy_timesteps: int = _f(0, "Extra PPO timesteps after the last RLHF round")
    policy_log_interval: int | None = _f(None, "stable-baselines3 log interval")
    policy_learning_kwargs: dict[str, Any] | None = _f(None, "PPO hyperparameter overrides, e.g. '{n_steps:256,batch_size:64}'")
    collection_timesteps: int | None = _f(None, "Trajectory-collection timesteps per round (suite default when omitted)")
    fragment_length: int | None = _f(None, "Preference fragment length (suite default when omitted)")
    active_learning: bool | None = _f(None, "Use active query selection (suite default when omitted)")
    active_query_strategy: str = _f("auto", "Active learning strategy", choices=ACTIVE_QUERY_STRATEGIES)
    dropout_samples: int = _f(8, "MC-dropout samples for active learning")
    dropout_p: float = _f(0.25, "MC-dropout probability")
    active_learning_batches: int = _f(512, "Candidate batches scored during active learning")

    reward_hidden_sizes: tuple[int, ...] = _f((200,), "Reward model hidden sizes, e.g. '200' or '64,64'", parse="int_tuple")
    reward_model_lr: float = _f(0.01, "Reward model learning rate")
    reward_model_epochs: int = _f(100, "Reward model training epochs")
    reward_model_patience: int = _f(10, "Early-stopping patience (epochs)")
    reward_model_batch_size: int = _f(32, "Reward model batch size")
    reward_model_ensemble_size: int = _f(1, "Reward model ensemble size (1 = single model)")
    model_reward_scale: float = _f(1.0, "Scale applied to model rewards")
    model_reward_min: float | None = _f(None, "Clip model rewards below this value ('none' disables)")
    model_reward_max: float | None = _f(None, "Clip model rewards above this value ('none' disables)")
    normalize_model_reward: bool = _f(False, "Standardize model rewards to the target mean/std")
    model_reward_target_mean: float = _f(0.0, "Target mean for normalized model rewards")
    model_reward_target_std: float = _f(1.0, "Target std for normalized model rewards")
    include_partial_feature: bool | None = _f(None, "Feed the partial reward to the reward model (defaults to naive/delta modes)")

    pretrain_reward_model: bool = _f(False, "Pretrain the reward model before preference training")
    pretrain_target: str = _f("partial", "Pretraining regression target", choices=PRETRAIN_TARGETS)
    pretrain_epochs: int = _f(25, "Pretraining epochs")
    pretrain_batch_size: int = _f(256, "Pretraining batch size")
    pretrain_lr: float = _f(1e-3, "Pretraining learning rate")


@dataclass(frozen=True)
class SweepConfig:
    suite: str = _f(MUJOCO_SUITE, "Environment suite", choices=TRAIN_SUITES)
    env_ids: tuple[str, ...] | None = _f(None, "Env ids to sweep (suite defaults when omitted)", nargs="+")
    seeds: tuple[int, ...] = _f((2,), "Seeds to sweep", nargs="+")
    timesteps: int = _f(5_000_000, "Total PPO timesteps per run")
    log_dir: str | Path | None = _f(None, "Root log directory (suite default when omitted)")
    manifest: str | Path | None = _f(None, "Manifest JSONL path (defaults to <log_dir>/manifest.jsonl)")
    execute: bool = _f(False, "Launch the runs instead of dry-running")
    skip_completed: bool = _f(True, "Skip runs whose metadata.json already exists")

    n_envs: int = _f(8, "Number of vectorized training envs")
    eval_freq: int = _f(100_000, "Timesteps between evaluations")
    n_eval_episodes: int | None = _f(None, "Episodes per periodic evaluation (suite default when omitted)")
    final_eval_episodes: int | None = _f(None, "Episodes for the final evaluation (suite default when omitted)")
    query_budget: int = _f(1400, "Total synthetic preference queries")
    rlhf_rounds: int = _f(5, "Number of RLHF rounds")
    collection_timesteps: int | None = _f(None, "Trajectory-collection timesteps per round (suite default when omitted)")
    fragment_length: int | None = _f(None, "Preference fragment length (suite default when omitted)")
    reward_model_epochs: int = _f(100, "Reward model training epochs")
    reward_model_patience: int = _f(10, "Early-stopping patience (epochs)")
    reward_model_batch_size: int = _f(32, "Reward model batch size")
    reward_model_ensemble_size: int = _f(1, "Reward model ensemble size (1 = single model)")
    active_query_strategy: str = _f("auto", "Active learning strategy", choices=ACTIVE_QUERY_STRATEGIES)
    active_learning_batches: int = _f(512, "Candidate batches scored during active learning")
    pretrain_epochs: int = _f(25, "Pretraining epochs")
    pretrain_batch_size: int = _f(256, "Pretraining batch size")
    pretrain_lr: float = _f(1e-3, "Pretraining learning rate")
    device: str = _f("auto", "Torch device", choices=DEVICES)
    preset: str | None = _f(None, "MuJoCo PPO preset", choices=("auto", "generic", "reacher"))
    progress_bar: bool = _f(False, "Show the stable-baselines3 progress bar")
    normalize_model_reward: bool = _f(False, "Standardize model rewards to the target mean/std")
    model_reward_min: float | None = _f(None, "Clip model rewards below this value ('none' disables)")
    model_reward_max: float | None = _f(None, "Clip model rewards above this value ('none' disables)")
    model_reward_target_mean: float | None = _f(None, "Target mean for normalized model rewards")
    model_reward_target_std: float | None = _f(None, "Target std for normalized model rewards")
    partial: str | None = _f(None, "Manual partial reward reference: <module> or <module>:<name>")


@dataclass(frozen=True)
class SummaryConfig:
    suite: str = _f(MUJOCO_SUITE, "Environment suite", choices=TRAIN_SUITES)
    root: str | Path | None = _f(None, "Directory scanned for metadata.json files (suite log dir when omitted)")
    summary_csv: str | Path | None = _f(None, "Per-run summary CSV path (defaults to <root>/summary.csv)")
    aggregate_csv: str | Path | None = _f(None, "Aggregated CSV path (defaults to <root>/aggregate.csv)")


def suite_default_envs(suite: str) -> tuple[str, ...]:
    return get_suite(_validate_suite(suite)).default_envs()


def suite_supported_envs(suite: str) -> tuple[str, ...]:
    return get_suite(_validate_suite(suite)).supported_envs()


def normalize_experiment_config(config: ExperimentConfig) -> ExperimentConfig:
    suite = _validate_suite(config.suite)
    spec = get_suite(suite)
    env_id = config.env_id or spec.default_env_id()
    default_collection, default_fragment = spec.collection_defaults((env_id,))

    normalized = replace(
        config,
        suite=suite,
        env_id=env_id,
        log_dir=Path(config.log_dir or spec.default_log_dir),
        preset=config.preset or spec.default_preset,
        n_eval_episodes=spec.default_n_eval_episodes if config.n_eval_episodes is None else config.n_eval_episodes,
        final_eval_episodes=spec.default_final_eval_episodes if config.final_eval_episodes is None else config.final_eval_episodes,
        collection_timesteps=default_collection if config.collection_timesteps is None else config.collection_timesteps,
        fragment_length=default_fragment if config.fragment_length is None else config.fragment_length,
        active_learning=spec.default_active_learning if config.active_learning is None else config.active_learning,
    )
    _validate_experiment(normalized)
    return normalized


def normalize_sweep_config(config: SweepConfig) -> SweepConfig:
    suite = _validate_suite(config.suite)
    spec = get_suite(suite)
    env_ids = tuple(config.env_ids or spec.default_envs())
    log_dir = Path(config.log_dir or spec.default_log_dir)
    default_collection, default_fragment = spec.collection_defaults(env_ids)

    normalized = replace(
        config,
        suite=suite,
        env_ids=env_ids,
        log_dir=log_dir,
        manifest=Path(config.manifest or (log_dir / "manifest.jsonl")),
        n_eval_episodes=spec.default_n_eval_episodes if config.n_eval_episodes is None else config.n_eval_episodes,
        final_eval_episodes=spec.default_final_eval_episodes if config.final_eval_episodes is None else config.final_eval_episodes,
        collection_timesteps=default_collection if config.collection_timesteps is None else config.collection_timesteps,
        fragment_length=default_fragment if config.fragment_length is None else config.fragment_length,
        preset=config.preset or spec.default_preset,
    )
    _validate_sweep(normalized)
    return normalized


def normalize_summary_config(config: SummaryConfig) -> SummaryConfig:
    suite = _validate_suite(config.suite)
    root = Path(config.root or get_suite(suite).default_log_dir)
    return replace(
        config,
        suite=suite,
        root=root,
        summary_csv=Path(config.summary_csv or (root / "summary.csv")),
        aggregate_csv=Path(config.aggregate_csv or (root / "aggregate.csv")),
    )


def _validate_suite(suite: str) -> str:
    if suite not in SUITE_NAMES:
        raise ConfigError(f"Unsupported suite '{suite}'. Supported suites: {', '.join(SUITE_NAMES)}")
    return suite


def _validate_experiment(config: ExperimentConfig) -> None:
    spec = get_suite(config.suite)
    if config.env_id not in spec.supported_envs():
        raise ConfigError(f"Unsupported {config.suite} env '{config.env_id}'. Try `list-envs --suite {config.suite}` for available envs.")
    if config.mode not in TRAIN_MODES:
        raise ConfigError(f"Unsupported mode '{config.mode}'. Supported modes: {', '.join(TRAIN_MODES)}")
    if config.mode in PARTIAL_REQUIRED_MODES and not config.partial:
        raise ConfigError(f"Mode '{config.mode}' requires --partial with a manually written partial reward.")
    _validate_common_numeric(config.timesteps, config.rlhf_rounds, config.query_budget, config.fragment_length or 0)
    if config.initial_timesteps < 0:
        raise ConfigError("initial_timesteps must be non-negative")
    if config.policy_timesteps_per_round is not None and config.policy_timesteps_per_round < 0:
        raise ConfigError("policy_timesteps_per_round must be non-negative")
    if config.final_policy_timesteps < 0:
        raise ConfigError("final_policy_timesteps must be non-negative")
    if config.policy_log_interval is not None and config.policy_log_interval <= 0:
        raise ConfigError("policy_log_interval must be greater than zero")
    if any(size <= 0 for size in config.reward_hidden_sizes):
        raise ConfigError("reward_hidden_sizes must contain positive integers")
    if config.reward_model_lr <= 0:
        raise ConfigError("reward_model_lr must be greater than zero")
    if config.reward_model_ensemble_size <= 0:
        raise ConfigError("reward_model_ensemble_size must be greater than zero")
    if config.active_query_strategy not in ACTIVE_QUERY_STRATEGIES:
        raise ConfigError(f"Unsupported active_query_strategy '{config.active_query_strategy}'")
    if config.device not in DEVICES:
        raise ConfigError(f"Unsupported device '{config.device}'. Supported devices: {', '.join(DEVICES)}")
    if config.final_policy not in FINAL_POLICIES:
        raise ConfigError(f"Unsupported final_policy '{config.final_policy}'")
    if config.plot_mode not in PLOT_MODES:
        raise ConfigError(f"Unsupported plot_mode '{config.plot_mode}'")
    if config.pretrain_target not in PRETRAIN_TARGETS:
        raise ConfigError(f"Unsupported pretrain target '{config.pretrain_target}'")
    if spec.presets is not None and config.preset not in spec.presets:
        raise ConfigError(f"Unsupported {config.suite} preset '{config.preset}'")


def _validate_sweep(config: SweepConfig) -> None:
    spec = get_suite(config.suite)
    supported = spec.supported_envs()
    bad_envs = [env_id for env_id in config.env_ids or () if env_id not in supported]
    if bad_envs:
        raise ConfigError(f"Unsupported envs for suite '{config.suite}': {', '.join(bad_envs)}")
    if not config.seeds:
        raise ConfigError("At least one seed is required")
    if not config.partial:
        raise ConfigError("Sweeps include partial-reward variants and require --partial.")
    _validate_common_numeric(config.timesteps, config.rlhf_rounds, config.query_budget, config.fragment_length or 0)
    if config.device not in DEVICES:
        raise ConfigError(f"Unsupported device '{config.device}'")
    if config.reward_model_ensemble_size <= 0:
        raise ConfigError("reward_model_ensemble_size must be greater than zero")
    if config.active_query_strategy not in ACTIVE_QUERY_STRATEGIES:
        raise ConfigError(f"Unsupported active_query_strategy '{config.active_query_strategy}'")
    if spec.presets is not None and config.preset not in spec.presets:
        raise ConfigError(f"Unsupported {config.suite} preset '{config.preset}'")


def _validate_common_numeric(timesteps: int, rlhf_rounds: int, query_budget: int, fragment_length: int) -> None:
    if timesteps < 0:
        raise ConfigError("timesteps must be non-negative")
    if rlhf_rounds <= 0:
        raise ConfigError("rlhf_rounds must be greater than zero")
    if query_budget < 0:
        raise ConfigError("query_budget must be non-negative")
    if fragment_length <= 0:
        raise ConfigError("fragment_length must be greater than zero")
