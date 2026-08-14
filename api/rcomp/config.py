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
PRETRAIN_LOSSES = ("mse", "bt")
ACTIVE_QUERY_STRATEGIES = ("auto", "dropout", "ensemble")
ACTIVE_CANDIDATE_PROTOCOLS = ("matching", "pool")
ROUND0_DATA_PROTOCOLS = ("legacy", "separate", "overlap")
REWARD_MODEL_LOSS_REDUCTIONS = ("sum", "mean")
ENSEMBLE_TRAINING_MODES = ("kfold", "full")
ENV_NORMALIZE_MODES = ("auto", "on", "off")


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
    tuned_hyperparams: bool = _f(False, "Use the literature-tuned per-env PPO hyperparameters in rcomp/ppo_presets.py (off = stock defaults, behaviour unchanged)")
    partial: str | None = _f(None, "Manual partial reward reference: <module> or <module>:<name>")

    rlhf_rounds: int = _f(5, "Number of RLHF rounds")
    query_budget: int = _f(1400, "Total synthetic preference queries")
    initial_timesteps: int = _f(0, "PPO timesteps before the first RLHF round")
    policy_timesteps_per_round: int | None = _f(None, "Override PPO timesteps per RLHF round")
    final_policy_timesteps: int = _f(0, "Extra PPO timesteps after the last RLHF round")
    policy_log_interval: int | None = _f(None, "stable-baselines3 log interval")
    policy_learning_kwargs: dict[str, Any] | None = _f(None, "PPO hyperparameter overrides, e.g. '{n_steps:256,batch_size:64}'")
    collection_timesteps: int | None = _f(None, "Trajectory-collection timesteps per round (suite default when omitted)")
    round0_collection_timesteps: int | None = _f(
        None,
        "Trajectory-collection timesteps for round 0 only (collection_timesteps when omitted). Round 0's "
        "untrained policy is the most diverse sampler the run will ever have, so it can justify a larger "
        "sample than later rounds; with --round0-data-protocol separate/overlap EACH of the two round-0 "
        "streams collects this many steps, exactly as collection_timesteps did before",
    )
    fragment_length: int | None = _f(None, "Preference fragment length (suite default when omitted)")
    active_learning: bool | None = _f(None, "Use active query selection (suite default when omitted)")
    active_query_strategy: str = _f("auto", "Active learning strategy", choices=ACTIVE_QUERY_STRATEGIES)
    dropout_samples: int = _f(8, "MC-dropout samples for active learning")
    dropout_p: float = _f(0.25, "MC-dropout probability")
    active_learning_batches: int = _f(512, "Candidate batches scored during active learning (matching protocol only)")
    active_candidate_protocol: str = _f("matching", "How active learning builds its candidate set: 'matching' scores whole perfect matchings and keeps the best (legacy); 'pool' draws an independent pool of pool-multiplier x query_count pairs and keeps the per-pair top-k, as B-Pref does", choices=ACTIVE_CANDIDATE_PROTOCOLS)
    active_pool_multiplier: int = _f(10, "Candidate pool size as a multiple of the round's query count (pool protocol only); B-Pref uses 10")
    dedicated_query_rng: bool = _f(
        False,
        "Use a deterministic per-round Python RNG for query pairing/candidate matchings so reward-training and pretraining shuffles cannot change which pairs are considered",
    )

    unhealthy_penalty: float | None = _f(
        None,
        "TRAINING-ENV ONLY: replace the healthy/unhealthy termination cliff with a persistent "
        "reward of +P while healthy and -P while unhealthy, and stop terminating on unhealthy "
        "(TimeLimit truncation is unchanged). Off when omitted, which leaves the environment "
        "exactly as it is today. Requires an env with an 'is_healthy' property (Walker2d-v5, "
        "Ant-v5, Hopper-v5, Humanoid-v5). Every evaluation path keeps scoring the STANDARD "
        "environment; when this is set the final policy is additionally scored on the modified "
        "environment and recorded as 'selected_policy_modified_env_reward_mean'",
    )

    env_normalize: str = _f(
        "auto",
        "Whether to wrap the training env in VecNormalize (obs + reward): 'auto' asks the suite "
        "(MuJoCo always on, Box2D on for 1-D Box spaces), 'on'/'off' force it. Evaluation ALWAYS "
        "scores raw reward, so this changes what the policy trains on, not how it is measured. "
        "Off makes an SB3-defaults arm literally out-of-the-box; note the rl-zoo MuJoCo blocks "
        "assume normalize: True, so forcing it off changes what a tuned preset was tuned for",
        choices=ENV_NORMALIZE_MODES,
    )

    reward_hidden_sizes: tuple[int, ...] = _f((200,), "Reward model hidden sizes, e.g. '200' or '64,64'", parse="int_tuple")
    reward_model_lr: float = _f(0.01, "Reward model learning rate")
    reward_model_loss_reduction: str = _f(
        "sum",
        "Reduce the per-pair reward-model loss by 'sum' (historical behavior) or 'mean' (PEBBLE/B-Pref-style minibatch scaling)",
        choices=REWARD_MODEL_LOSS_REDUCTIONS,
    )
    reward_model_l1: float = _f(0.01, "L1 penalty on reward-model weights (0 disables; historical default 0.01)")
    reward_output_l1: float = _f(0.001, "L1 penalty on per-state reward outputs (0 disables; historical default 0.001)")
    reward_model_epochs: int = _f(100, "Reward model training epochs")
    reward_model_patience: int = _f(10, "Early-stopping patience (epochs)")
    reward_model_train_accuracy_stop: float | None = _f(
        None,
        "Stop each reward-model member after a full epoch when its mean training ranking accuracy is strictly above this threshold (0.97 gives a member-local B-Pref-inspired stop; 'none' disables)",
    )
    reward_model_batch_size: int = _f(32, "Reward model batch size")
    reward_model_ensemble_size: int = _f(1, "Reward model ensemble size (1 = single model)")
    ensemble_training: str = _f(
        "kfold",
        "How reward ensembles use preference data: 'kfold' preserves the historical held-out-fold scheme; 'full' trains every member on all pairs, as PEBBLE/B-Pref do",
        choices=ENSEMBLE_TRAINING_MODES,
    )
    model_reward_scale: float = _f(1.0, "Scale applied to model rewards")
    model_reward_min: float | None = _f(None, "Clip model rewards below this value ('none' disables)")
    model_reward_max: float | None = _f(None, "Clip model rewards above this value ('none' disables)")
    normalize_model_reward: bool = _f(False, "Standardize model rewards to the target mean/std")
    model_reward_target_mean: float = _f(0.0, "Target mean for normalized model rewards")
    model_reward_target_std: float = _f(1.0, "Target std for normalized model rewards")
    include_partial_feature: bool | None = _f(None, "Feed the partial reward to the reward model (defaults to naive/delta modes)")
    normalize_partial_reward: bool = _f(False, "Normalize the partial reward with running stats in the model input, delta loss, and composed reward")
    partial_alpha: float = _f(1.0, "Expert-confidence coefficient on the partial reward in the delta loss and composed reward")
    learn_partial_alpha: bool = _f(False, "Learn alpha as a reward-model parameter, anchored to partial_alpha by an MSE term")
    partial_alpha_penalty: float = _f(1.0, "Weight of the mse(alpha, partial_alpha) anchor when alpha is learned")
    partial_prediction_coef: float = _f(0.0, "Weight of the auxiliary MSE loss for predicting the (normalized) partial reward; 0 disables the extra head")
    batchnorm_model_reward: bool = _f(False, "Mini-batch normalize the model output (delta) inside the loss and at inference (batch stats in training, running stats at eval)")
    tanh_model_reward: bool = _f(False, "Bound the reward model's per-state output to [-1,1] with tanh INSIDE the model, as PEBBLE/B-Pref do (model_reward_min/max only clip afterwards in the wrapper and do not shape the Bradley-Terry loss)")
    tanh_scale: float = _f(1.0, "Divide the pre-activation value by this before tanh, i.e. tanh(x/scale): the hard [-1,1] bound is unchanged but the near-linear region widens and saturation gradients vanish more slowly. 1.0 reproduces tanh(x) exactly; values != 1 require --tanh-model-reward")
    gate_partial: bool = _f(False, "Learn a per-state gate g(s,a) in [0,1] so the composed reward is g*partial + delta (the model decides how much to trust the partial per state)")
    gate_holdout: bool = _f(False, "Train the naive frozen-trunk gate on held-out preferences with early stopping, instead of the pairs the reward model was already fit on")
    gate_lr: float | None = _f(None, "Learning rate for the gate head (reward_model_lr when omitted); lower values avoid saturating the sigmoid")
    gate_epochs: int | None = _f(None, "Epochs for the gate phase (reward_model_epochs when omitted)")
    gate_patience: int = _f(10, "Early-stopping patience for the gate phase (requires --gate-holdout)")
    gate_init: float = _f(0.5, "Initial value of the per-state gate, applied as a bias init (1.0 = start by fully trusting the partial, i.e. the naive baseline)")
    gate_prior_penalty: float = _f(0.0, "Weight of a penalty pulling g toward 1, so shrinking the partial requires evidence from the preferences")
    gate_diagnostic: bool = _f(False, "Record the rank correlation between the gate g(s,a) and the partial's error |partial - true reward| (negative = the gate distrusts the partial where it is wrong)")
    reward_model_diagnostics: bool = _f(False, "Record Bradley-Terry loss and ranking accuracy on held-out preferences, before and after preference training, and with the partial input feature ablated (measures how much the model relies on that feature)")
    holdout_pairs: int = _f(
        0,
        "Reserve this many preference pairs per round as a TRUE held-out diagnostic set: they come from "
        "trajectories no query is drawn from, are never trained on by any ensemble member, and are scored "
        "before and after each round's training. This is an oracle diagnostic for the experimenter and is "
        "NOT charged to --query-budget (0 disables). Fixes the fact that ensemble_training=full leaves "
        "n_val_pairs=0 and kfold's 'held-out' fold was still trained on by the other members.",
    )
    ensemble_bootstrap: bool = _f(
        False,
        "Give each reward-ensemble member its own bootstrap resample (draw N pairs with replacement from the "
        "N collected pairs) instead of the identical buffer. B-Pref gives every member the same data in a "
        "different order, which lets members converge to the same function once training accuracy saturates; "
        "disagreement-based active learning then has nothing to measure. Requires --ensemble-training full.",
    )
    save_reward_model: bool = _f(False, "Save the trained reward model weights to reward_model.pt for offline analysis")
    query_fisher_diagnostic: bool = _f(False, "Record the Bradley-Terry Fisher information of the queries active learning selected each round (how much the answers are expected to pin down the reward parameters)")

    pretrain_reward_model: bool = _f(False, "Pretrain the reward model before preference training")
    pretrain_target: str = _f("partial", "Pretraining regression target", choices=PRETRAIN_TARGETS)
    pretrain_loss: str = _f("mse", "Pretraining objective: 'mse' regresses onto the target's raw per-state values; 'bt' trains on preference labels generated by the target, i.e. the same Bradley-Terry objective preference learning uses (labels are free and are NOT charged to the query budget)", choices=PRETRAIN_LOSSES)
    pretrain_pairs: int = _f(0, "Maximum synthetic preference pairs for 'bt' pretraining (0 = every pair the collected fragments allow)")
    pretrain_holdout: bool = _f(False, "Pretrain on one half of the round-0 rollout and draw the round-0 preference queries from the other half, so the pretrained model has not already seen the states it is queried about")
    round0_data_protocol: str = _f(
        "legacy",
        "Round-0 rollout protocol: 'legacy' preserves the historical doubled rollout and optional ordered holdout split; 'separate' collects equal-size independently seeded pretrain/query rollouts; 'overlap' collects the same two rollouts but pretrains and queries on the first one for an equal-size leak control",
        choices=ROUND0_DATA_PROTOCOLS,
    )
    pretrain_bt_temperature: float | None = _f(
        None,
        "Temperature for soft BT pseudo-labels: P(first preferred)=sigmoid((target_return_1-target_return_2)/temperature); 'none' preserves historical hard labels",
    )
    pretrain_bt_tie_margin: float | None = _f(
        None,
        "Assign BT pseudo-label 0.5 when the absolute target-return difference is at most this margin; 'none' preserves historical tie handling",
    )
    pretrain_bt_match_reward_training: bool = _f(
        False,
        "Use the downstream reward-model loss reduction and L1 settings during BT pretraining; off preserves historical summed-loss/no-L1 BT pretraining",
    )
    pretrain_epochs: int = _f(25, "Pretraining epochs")
    pretrain_batch_size: int = _f(256, "Pretraining batch size")
    pretrain_lr: float = _f(1e-3, "Pretraining learning rate")
    pretrain_patience: int = _f(10, "Early-stopping patience for 'bt' pretraining (epochs)")


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
    reward_model_train_accuracy_stop: float | None = _f(
        None,
        "Stop each reward-model member after a full epoch when its mean training ranking accuracy is strictly above this threshold (0.97 gives a member-local B-Pref-inspired stop; 'none' disables)",
    )
    reward_model_batch_size: int = _f(32, "Reward model batch size")
    reward_model_ensemble_size: int = _f(1, "Reward model ensemble size (1 = single model)")
    active_query_strategy: str = _f("auto", "Active learning strategy", choices=ACTIVE_QUERY_STRATEGIES)
    active_learning_batches: int = _f(512, "Candidate batches scored during active learning")
    dedicated_query_rng: bool = _f(
        False,
        "Use a deterministic per-round Python RNG for query pairing/candidate matchings so reward-training and pretraining shuffles cannot change which pairs are considered",
    )
    pretrain_epochs: int = _f(25, "Pretraining epochs")
    pretrain_batch_size: int = _f(256, "Pretraining batch size")
    pretrain_lr: float = _f(1e-3, "Pretraining learning rate")
    device: str = _f("auto", "Torch device", choices=DEVICES)
    preset: str | None = _f(None, "MuJoCo PPO preset", choices=("auto", "generic", "reacher"))
    tuned_hyperparams: bool = _f(False, "Use the literature-tuned per-env PPO hyperparameters in rcomp/ppo_presets.py (off = stock defaults, behaviour unchanged)")
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
    # Checked here, not only by argparse: a typo would otherwise fall through
    # the `mode == "on"` test in probe_spaces and SILENTLY disable normalization.
    if config.unhealthy_penalty is not None and config.unhealthy_penalty <= 0:
        raise ConfigError("unhealthy_penalty must be positive; omit the flag to leave the environment unmodified")
    if config.env_normalize not in ENV_NORMALIZE_MODES:
        raise ConfigError(
            f"Unsupported env_normalize '{config.env_normalize}'. Supported: {', '.join(ENV_NORMALIZE_MODES)}"
        )
    if config.mode in PARTIAL_REQUIRED_MODES and not config.partial:
        raise ConfigError(f"Mode '{config.mode}' requires --partial with a manually written partial reward.")
    if config.normalize_partial_reward and config.mode not in ("naive", "delta"):
        raise ConfigError("normalize_partial_reward requires mode 'naive' or 'delta'")
    if config.partial_alpha != 1.0 and config.mode not in ("naive", "delta"):
        raise ConfigError("partial_alpha requires mode 'naive' or 'delta'")
    if config.learn_partial_alpha and config.mode != "delta":
        raise ConfigError("learn_partial_alpha requires mode 'delta'")
    if config.partial_alpha_penalty < 0:
        raise ConfigError("partial_alpha_penalty must be non-negative")
    if config.partial_prediction_coef < 0:
        raise ConfigError("partial_prediction_coef must be non-negative")
    if config.partial_prediction_coef > 0 and config.mode != "delta":
        raise ConfigError("partial_prediction_coef requires mode 'delta'")
    if config.batchnorm_model_reward and config.mode not in PREFERENCE_MODES:
        raise ConfigError("batchnorm_model_reward requires a preference mode (feedback/naive/delta)")
    if config.gate_partial and config.mode not in ("delta", "naive"):
        raise ConfigError("gate_partial requires mode 'delta' (joint) or 'naive' (frozen-trunk gate)")
    if config.gate_partial and config.learn_partial_alpha:
        raise ConfigError("gate_partial and learn_partial_alpha are mutually exclusive")
    if not 0.0 < config.gate_init <= 1.0:
        raise ConfigError("gate_init must be in (0, 1]")
    if config.gate_lr is not None and config.gate_lr <= 0:
        raise ConfigError("gate_lr must be greater than zero")
    if config.gate_epochs is not None and config.gate_epochs <= 0:
        raise ConfigError("gate_epochs must be greater than zero")
    if config.gate_patience <= 0:
        raise ConfigError("gate_patience must be greater than zero")
    if config.gate_prior_penalty < 0:
        raise ConfigError("gate_prior_penalty must be non-negative")
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
    if config.reward_model_loss_reduction not in REWARD_MODEL_LOSS_REDUCTIONS:
        raise ConfigError(
            f"Unsupported reward_model_loss_reduction '{config.reward_model_loss_reduction}'. "
            f"Supported: {', '.join(REWARD_MODEL_LOSS_REDUCTIONS)}"
        )
    if config.reward_model_l1 < 0:
        raise ConfigError("reward_model_l1 must be non-negative")
    if config.reward_output_l1 < 0:
        raise ConfigError("reward_output_l1 must be non-negative")
    if config.reward_model_train_accuracy_stop is not None and not 0.0 <= config.reward_model_train_accuracy_stop <= 1.0:
        raise ConfigError("reward_model_train_accuracy_stop must be between 0 and 1")
    if config.ensemble_training not in ENSEMBLE_TRAINING_MODES:
        raise ConfigError(
            f"Unsupported ensemble_training '{config.ensemble_training}'. "
            f"Supported: {', '.join(ENSEMBLE_TRAINING_MODES)}"
        )
    if config.ensemble_training == "full" and config.gate_holdout:
        raise ConfigError("gate_holdout requires held-out folds and cannot be combined with ensemble_training='full'")
    if config.reward_model_ensemble_size <= 0:
        raise ConfigError("reward_model_ensemble_size must be greater than zero")
    if config.active_query_strategy not in ACTIVE_QUERY_STRATEGIES:
        raise ConfigError(f"Unsupported active_query_strategy '{config.active_query_strategy}'")
    if config.active_candidate_protocol not in ACTIVE_CANDIDATE_PROTOCOLS:
        raise ConfigError(f"Unsupported active_candidate_protocol '{config.active_candidate_protocol}'. Supported: {', '.join(ACTIVE_CANDIDATE_PROTOCOLS)}")
    if config.active_pool_multiplier <= 0:
        raise ConfigError("active_pool_multiplier must be greater than zero")
    if config.device not in DEVICES:
        raise ConfigError(f"Unsupported device '{config.device}'. Supported devices: {', '.join(DEVICES)}")
    if config.final_policy not in FINAL_POLICIES:
        raise ConfigError(f"Unsupported final_policy '{config.final_policy}'")
    if config.plot_mode not in PLOT_MODES:
        raise ConfigError(f"Unsupported plot_mode '{config.plot_mode}'")
    if config.pretrain_target not in PRETRAIN_TARGETS:
        raise ConfigError(f"Unsupported pretrain target '{config.pretrain_target}'")
    if config.pretrain_loss not in PRETRAIN_LOSSES:
        raise ConfigError(f"Unsupported pretrain loss '{config.pretrain_loss}'. Supported: {', '.join(PRETRAIN_LOSSES)}")
    if config.round0_data_protocol not in ROUND0_DATA_PROTOCOLS:
        raise ConfigError(
            f"Unsupported round0_data_protocol '{config.round0_data_protocol}'. "
            f"Supported: {', '.join(ROUND0_DATA_PROTOCOLS)}"
        )
    if config.round0_data_protocol != "legacy" and config.pretrain_holdout:
        raise ConfigError("pretrain_holdout is the legacy ordered split; do not combine it with round0_data_protocol")
    if config.pretrain_pairs < 0:
        raise ConfigError("pretrain_pairs must be non-negative")
    if config.pretrain_bt_temperature is not None and config.pretrain_bt_temperature <= 0:
        raise ConfigError("pretrain_bt_temperature must be greater than zero")
    if config.pretrain_bt_tie_margin is not None and config.pretrain_bt_tie_margin < 0:
        raise ConfigError("pretrain_bt_tie_margin must be non-negative")
    if config.pretrain_patience <= 0:
        raise ConfigError("pretrain_patience must be greater than zero")
    if config.holdout_pairs < 0:
        raise ConfigError("holdout_pairs must be non-negative")
    if config.holdout_pairs and config.mode not in PREFERENCE_MODES:
        raise ConfigError("holdout_pairs requires a preference mode (feedback/naive/delta)")
    if config.ensemble_bootstrap and config.reward_model_ensemble_size <= 1:
        raise ConfigError("ensemble_bootstrap requires reward_model_ensemble_size > 1")
    if config.ensemble_bootstrap and config.ensemble_training != "full":
        raise ConfigError(
            "ensemble_bootstrap resamples the full buffer per member, so it requires "
            "--ensemble-training full (kfold already gives members different data)"
        )
    if config.tanh_model_reward and config.batchnorm_model_reward:
        raise ConfigError("tanh_model_reward and batchnorm_model_reward are mutually exclusive")
    if config.tanh_model_reward and config.mode not in PREFERENCE_MODES:
        raise ConfigError("tanh_model_reward requires a preference mode (feedback/naive/delta)")
    if config.tanh_scale <= 0:
        raise ConfigError("tanh_scale must be greater than zero")
    if config.tanh_scale != 1.0 and not config.tanh_model_reward:
        raise ConfigError("tanh_scale rescales the tanh output bound and requires --tanh-model-reward")
    if config.round0_collection_timesteps is not None and config.round0_collection_timesteps <= 0:
        raise ConfigError("round0_collection_timesteps must be greater than zero")
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
    if config.reward_model_train_accuracy_stop is not None and not 0.0 <= config.reward_model_train_accuracy_stop <= 1.0:
        raise ConfigError("reward_model_train_accuracy_stop must be between 0 and 1")
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
