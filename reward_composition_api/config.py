from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import gymnasium as gym

from local_gym.classes.atari_reward_specs import supported_atari_envs
from local_gym.classes.mujoco_reward_specs import supported_mujoco_envs

from .errors import ConfigError


MUJOCO_SUITE = "mujoco"
ATARI_SUITE = "atari"
BOX2D_SUITE = "box2d"
GYM_SUITE = "gym"
LEGACY_SUITE = "legacy"
TRAIN_SUITES = (MUJOCO_SUITE, ATARI_SUITE, BOX2D_SUITE, GYM_SUITE)
SUITES = (*TRAIN_SUITES, LEGACY_SUITE)

TRAIN_MODES = ("true", "partial", "feedback", "naive", "delta")
ATARI_TRAIN_MODES = ("true", "partial", "feedback", "naive", "delta")
PREFERENCE_MODES = ("feedback", "naive", "delta")

FINAL_POLICIES = ("best", "last")
PLOT_MODES = ("best", "raw")
DEVICES = ("auto", "cpu", "cuda")
PRETRAIN_TARGETS = ("partial", "residual", "true")
ACTIVE_QUERY_STRATEGIES = ("auto", "dropout", "ensemble")
MUJOCO_PRESETS = ("auto", "generic", "reacher")
MUJOCO_PARTIAL_PROFILES = ("default", "ctrl_half", "true_like")
ATARI_PARTIAL_SOURCES = ("life_loss", "clipped_score_life_loss", "score", "score_life_loss")


@dataclass(frozen=True)
class ExperimentConfig:
    suite: str = MUJOCO_SUITE
    env_id: str | None = None
    mode: str = "delta"
    variant_name: str | None = None
    timesteps: int = 5_000_000
    run_name: str | None = None
    log_dir: str | Path | None = None
    seed: int = 0
    n_envs: int = 8
    device: str = "auto"
    eval_freq: int = 100_000
    n_eval_episodes: int | None = None
    final_eval_episodes: int | None = None
    stop_reward: float | None = None
    final_policy: str = "best"
    plot_mode: str = "best"
    smooth_window: int = 5
    progress_bar: bool = False

    preset: str | None = None
    partial_profile: str = "default"
    partial_source: str = "life_loss"
    partial: str | None = None

    rlhf_rounds: int = 5
    query_budget: int = 1400
    initial_timesteps: int = 0
    policy_timesteps_per_round: int | None = None
    final_policy_timesteps: int = 0
    policy_log_interval: int | None = None
    policy_learning_kwargs: dict[str, Any] | None = None
    collection_timesteps: int | None = None
    fragment_length: int | None = None
    active_learning: bool | None = None
    active_query_strategy: str = "auto"
    dropout_samples: int = 8
    dropout_p: float = 0.25
    active_learning_batches: int = 512

    reward_hidden_sizes: tuple[int, ...] = (200,)
    reward_model_lr: float = 0.01
    reward_model_epochs: int = 100
    reward_model_patience: int = 10
    reward_model_batch_size: int = 32
    reward_model_ensemble_size: int = 1
    model_reward_scale: float = 1.0
    model_reward_min: float | None = None
    model_reward_max: float | None = None
    normalize_model_reward: bool = False
    model_reward_target_mean: float = 0.0
    model_reward_target_std: float = 1.0
    include_partial_feature: bool | None = None

    pretrain_reward_model: bool = False
    pretrain_target: str = "partial"
    pretrain_epochs: int = 25
    pretrain_batch_size: int = 256
    pretrain_lr: float = 1e-3


@dataclass(frozen=True)
class SweepConfig:
    suite: str = MUJOCO_SUITE
    env_ids: tuple[str, ...] | None = None
    seeds: tuple[int, ...] = (2,)
    timesteps: int = 5_000_000
    log_dir: str | Path | None = None
    manifest: str | Path | None = None
    execute: bool = False
    skip_completed: bool = True

    n_envs: int = 8
    eval_freq: int = 100_000
    n_eval_episodes: int | None = None
    final_eval_episodes: int | None = None
    query_budget: int = 1400
    rlhf_rounds: int = 5
    collection_timesteps: int | None = None
    fragment_length: int | None = None
    reward_model_epochs: int = 100
    reward_model_patience: int = 10
    reward_model_batch_size: int = 32
    reward_model_ensemble_size: int = 1
    active_query_strategy: str = "auto"
    active_learning_batches: int = 512
    pretrain_epochs: int = 25
    pretrain_batch_size: int = 256
    pretrain_lr: float = 1e-3
    device: str = "auto"
    preset: str | None = None
    partial_source: str = "life_loss"
    progress_bar: bool = False
    normalize_model_reward: bool = False
    model_reward_min: float | None = None
    model_reward_max: float | None = None
    model_reward_target_mean: float | None = None
    model_reward_target_std: float | None = None
    partial: str | None = None


@dataclass(frozen=True)
class SummaryConfig:
    suite: str = MUJOCO_SUITE
    root: str | Path | None = None
    summary_csv: str | Path | None = None
    aggregate_csv: str | Path | None = None


def suite_default_envs(suite: str) -> tuple[str, ...]:
    suite = _validate_suite(suite)
    if suite == MUJOCO_SUITE:
        return ("Reacher-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
    if suite == ATARI_SUITE:
        return ("ALE/Breakout-v5", "ALE/Seaquest-v5", "ALE/Qbert-v5", "ALE/SpaceInvaders-v5")
    if suite == BOX2D_SUITE:
        return tuple(env for env in ("LunarLander-v3", "BipedalWalker-v3", "CarRacing-v3") if env in suite_supported_envs(BOX2D_SUITE))
    if suite == GYM_SUITE:
        return ("CartPole-v1",)
    return ("LunarLander-v3", "Reacher-v5")


def suite_supported_envs(suite: str) -> tuple[str, ...]:
    suite = _validate_suite(suite)
    if suite == MUJOCO_SUITE:
        return tuple(sorted(supported_mujoco_envs()))
    if suite == ATARI_SUITE:
        return tuple(sorted(supported_atari_envs()))
    if suite == BOX2D_SUITE:
        return tuple(env_id for env_id in _registered_gym_envs() if _is_box2d_env(env_id))
    if suite == GYM_SUITE:
        return _registered_gym_envs()
    return ("LunarLander-v3", "Reacher-v5")


def normalize_experiment_config(config: ExperimentConfig) -> ExperimentConfig:
    suite = _validate_suite(config.suite)
    env_id = config.env_id or _default_env_id(suite)
    log_dir = Path(config.log_dir or _default_log_dir(suite))
    preset = config.preset or ("auto" if suite == MUJOCO_SUITE else None)

    n_eval_episodes = config.n_eval_episodes
    final_eval_episodes = config.final_eval_episodes
    collection_timesteps = config.collection_timesteps
    fragment_length = config.fragment_length
    active_learning = config.active_learning
    if suite == ATARI_SUITE:
        n_eval_episodes = 5 if n_eval_episodes is None else n_eval_episodes
        final_eval_episodes = 10 if final_eval_episodes is None else final_eval_episodes
        collection_timesteps = 50_000 if collection_timesteps is None else collection_timesteps
        fragment_length = 64 if fragment_length is None else fragment_length
        active_learning = False if active_learning is None else active_learning
    elif suite in (BOX2D_SUITE, GYM_SUITE):
        n_eval_episodes = 5 if n_eval_episodes is None else n_eval_episodes
        final_eval_episodes = 10 if final_eval_episodes is None else final_eval_episodes
        if suite == BOX2D_SUITE and _is_lunar_lander_env(env_id):
            collection_timesteps = 10_000 if collection_timesteps is None else collection_timesteps
            fragment_length = 25 if fragment_length is None else fragment_length
        else:
            collection_timesteps = 2000 if collection_timesteps is None else collection_timesteps
            fragment_length = 1 if fragment_length is None else fragment_length
        active_learning = True if active_learning is None else active_learning
    else:
        n_eval_episodes = 10 if n_eval_episodes is None else n_eval_episodes
        final_eval_episodes = 50 if final_eval_episodes is None else final_eval_episodes
        collection_timesteps = 1500 if collection_timesteps is None else collection_timesteps
        fragment_length = 1 if fragment_length is None else fragment_length
        active_learning = True if active_learning is None else active_learning

    normalized = replace(
        config,
        suite=suite,
        env_id=env_id,
        log_dir=log_dir,
        preset=preset,
        n_eval_episodes=n_eval_episodes,
        final_eval_episodes=final_eval_episodes,
        collection_timesteps=collection_timesteps,
        fragment_length=fragment_length,
        active_learning=active_learning,
    )
    _validate_experiment(normalized)
    return normalized


def normalize_sweep_config(config: SweepConfig) -> SweepConfig:
    suite = _validate_suite(config.suite)
    env_ids = config.env_ids or suite_default_envs(suite)
    log_dir = Path(config.log_dir or _default_log_dir(suite))
    manifest = Path(config.manifest or (log_dir / "manifest.jsonl"))

    n_eval_episodes = config.n_eval_episodes
    final_eval_episodes = config.final_eval_episodes
    collection_timesteps = config.collection_timesteps
    fragment_length = config.fragment_length
    preset = config.preset
    if suite == ATARI_SUITE:
        n_eval_episodes = 5 if n_eval_episodes is None else n_eval_episodes
        final_eval_episodes = 10 if final_eval_episodes is None else final_eval_episodes
        collection_timesteps = 50_000 if collection_timesteps is None else collection_timesteps
        fragment_length = 64 if fragment_length is None else fragment_length
    elif suite in (BOX2D_SUITE, GYM_SUITE):
        n_eval_episodes = 5 if n_eval_episodes is None else n_eval_episodes
        final_eval_episodes = 10 if final_eval_episodes is None else final_eval_episodes
        if suite == BOX2D_SUITE and env_ids and all(_is_lunar_lander_env(env_id) for env_id in env_ids):
            collection_timesteps = 10_000 if collection_timesteps is None else collection_timesteps
            fragment_length = 25 if fragment_length is None else fragment_length
        else:
            collection_timesteps = 2000 if collection_timesteps is None else collection_timesteps
            fragment_length = 1 if fragment_length is None else fragment_length
    else:
        n_eval_episodes = 10 if n_eval_episodes is None else n_eval_episodes
        final_eval_episodes = 50 if final_eval_episodes is None else final_eval_episodes
        collection_timesteps = 1500 if collection_timesteps is None else collection_timesteps
        fragment_length = 1 if fragment_length is None else fragment_length
        preset = preset or "auto"

    normalized = replace(
        config,
        suite=suite,
        env_ids=tuple(env_ids),
        log_dir=log_dir,
        manifest=manifest,
        n_eval_episodes=n_eval_episodes,
        final_eval_episodes=final_eval_episodes,
        collection_timesteps=collection_timesteps,
        fragment_length=fragment_length,
        preset=preset,
    )
    _validate_sweep(normalized)
    return normalized


def normalize_summary_config(config: SummaryConfig) -> SummaryConfig:
    suite = _validate_suite(config.suite)
    root = Path(config.root or _default_log_dir(suite))
    normalized = replace(
        config,
        suite=suite,
        root=root,
        summary_csv=Path(config.summary_csv or (root / "summary.csv")),
        aggregate_csv=Path(config.aggregate_csv or (root / "aggregate.csv")),
    )
    if suite not in TRAIN_SUITES:
        raise ConfigError("summaries are currently supported for trainable suites")
    return normalized


def _validate_suite(suite: str) -> str:
    if suite not in SUITES:
        raise ConfigError(f"Unsupported suite '{suite}'. Supported suites: {', '.join(SUITES)}")
    return suite


def _validate_experiment(config: ExperimentConfig) -> None:
    supported = suite_supported_envs(config.suite)
    if config.env_id not in supported:
        raise ConfigError(f"Unsupported {config.suite} env '{config.env_id}'. Try `list-envs --suite {config.suite}` for available envs.")
    modes = ATARI_TRAIN_MODES if config.suite == ATARI_SUITE else TRAIN_MODES
    if config.mode not in modes:
        raise ConfigError(f"Unsupported mode '{config.mode}' for suite '{config.suite}'. Supported modes: {', '.join(modes)}")
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
    if config.suite == MUJOCO_SUITE:
        if config.preset not in MUJOCO_PRESETS:
            raise ConfigError(f"Unsupported MuJoCo preset '{config.preset}'")
        if config.partial_profile not in MUJOCO_PARTIAL_PROFILES:
            raise ConfigError(f"Unsupported MuJoCo partial profile '{config.partial_profile}'")
    if config.suite == ATARI_SUITE and config.partial_source not in ATARI_PARTIAL_SOURCES:
        raise ConfigError(f"Unsupported Atari partial source '{config.partial_source}'")


def _validate_sweep(config: SweepConfig) -> None:
    if config.suite not in TRAIN_SUITES:
        raise ConfigError("sweeps are currently supported for trainable suites")
    supported = suite_supported_envs(config.suite)
    bad_envs = [env_id for env_id in config.env_ids or () if env_id not in supported]
    if bad_envs:
        raise ConfigError(f"Unsupported envs for suite '{config.suite}': {', '.join(bad_envs)}")
    if not config.seeds:
        raise ConfigError("At least one seed is required")
    _validate_common_numeric(config.timesteps, config.rlhf_rounds, config.query_budget, config.fragment_length or 0)
    if config.device not in DEVICES:
        raise ConfigError(f"Unsupported device '{config.device}'")
    if config.reward_model_ensemble_size <= 0:
        raise ConfigError("reward_model_ensemble_size must be greater than zero")
    if config.active_query_strategy not in ACTIVE_QUERY_STRATEGIES:
        raise ConfigError(f"Unsupported active_query_strategy '{config.active_query_strategy}'")
    if config.suite == MUJOCO_SUITE and config.preset not in MUJOCO_PRESETS:
        raise ConfigError(f"Unsupported MuJoCo preset '{config.preset}'")
    if config.suite == ATARI_SUITE and config.partial_source not in ATARI_PARTIAL_SOURCES:
        raise ConfigError(f"Unsupported Atari partial source '{config.partial_source}'")


def _default_env_id(suite: str) -> str:
    if suite == ATARI_SUITE:
        return "ALE/Breakout-v5"
    if suite == BOX2D_SUITE:
        defaults = suite_default_envs(BOX2D_SUITE)
        return defaults[0] if defaults else "LunarLander-v3"
    if suite == GYM_SUITE:
        return "CartPole-v1"
    return "Reacher-v5"


def _default_log_dir(suite: str) -> str:
    if suite == ATARI_SUITE:
        return "logs/atari_ablations"
    if suite == BOX2D_SUITE:
        return "logs/box2d_ablations"
    if suite == GYM_SUITE:
        return "logs/gym_ablations"
    return "logs/mujoco_ablations"


def _registered_gym_envs() -> tuple[str, ...]:
    _try_register_atari_envs()
    return tuple(sorted(gym.envs.registry.keys()))


def _try_register_atari_envs() -> None:
    try:
        import ale_py
    except ImportError:
        return
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)


def _is_box2d_env(env_id: str) -> bool:
    return env_id.startswith(("LunarLander", "BipedalWalker", "CarRacing"))


def _is_lunar_lander_env(env_id: str) -> bool:
    return env_id.startswith("LunarLander")


def _validate_common_numeric(timesteps: int, rlhf_rounds: int, query_budget: int, fragment_length: int) -> None:
    if timesteps < 0:
        raise ConfigError("timesteps must be non-negative")
    if rlhf_rounds <= 0:
        raise ConfigError("rlhf_rounds must be greater than zero")
    if query_budget < 0:
        raise ConfigError("query_budget must be non-negative")
    if fragment_length <= 0:
        raise ConfigError("fragment_length must be greater than zero")

