from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from gymnasium import spaces
from gymnasium.spaces.utils import flatdim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from reward_model.reward_model import RewardModel
from reward_composition_api.config import ExperimentConfig
from reward_composition_api.registry import PartialSpec
from reward_composition_api.results import RunResult

from .common import (
    SaveVecNormalizeOnBest,
    include_partial_feature,
    learn_policy,
    resolve_custom_partial,
)
from .gym_env import (
    GymLearnedRewardRuntime,
    GymPreferenceRewardWrapper,
    collect_policy_trajectories,
    load_eval_env,
    make_eval_env,
    make_raw_env,
    make_train_env,
    make_trajectory_converter,
    ppo_hyperparams,
)
from .gym_evaluation import (
    GymComponentEvalCallback,
    component_keys,
    evaluate_gym_components,
    write_gym_component_summary,
)
from .gym_spaces import (
    should_normalize_observation,
)
from .rlhf import RlhfTrainer
from .reporting import (
    BackendRunPaths,
    report_eval_curve,
    select_final_policy,
)


def run_gym_experiment(config: ExperimentConfig) -> RunResult:
    custom_partial = _resolve_custom_partial(config)
    run_name = config.run_name or default_run_name(config)
    variant_name = config.variant_name or config.mode
    config = replace(config, run_name=run_name, variant_name=variant_name)
    if config.mode in {"true", "partial"}:
        return train_true_or_partial(config, custom_partial)
    return train_preference_mode(config, custom_partial)


def build_callbacks(config: ExperimentConfig, run_dir: Path, train_env, eval_env, custom_partial: PartialSpec | None):
    eval_freq = max(config.eval_freq // config.n_envs, 1)
    best_callbacks: list[BaseCallback] = []
    best_stats_path = run_dir / "best_model" / "best_vecnormalize.pkl"
    if isinstance(train_env, VecNormalize):
        best_callbacks.append(SaveVecNormalizeOnBest(train_env, best_stats_path))
    if config.stop_reward is not None:
        best_callbacks.append(StopTrainingOnRewardThreshold(reward_threshold=config.stop_reward, verbose=1))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=CallbackList(best_callbacks) if best_callbacks else None,
    )
    component_callback = GymComponentEvalCallback(
        run_dir / "eval" / "component_evaluations.csv",
        config.env_id,
        make_env=make_raw_env,
        custom_partial=custom_partial,
        eval_freq=eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        verbose=1,
    )
    return CallbackList([eval_callback, component_callback])


def train_true_or_partial(config: ExperimentConfig, custom_partial: PartialSpec | None) -> RunResult:
    run_dir = Path(config.log_dir) / config.run_name
    run_dir.mkdir(exist_ok=True, parents=True)
    probe_env = make_raw_env(config.env_id)
    normalize = should_normalize_observation(probe_env.observation_space)
    hyperparams = ppo_hyperparams(probe_env, config)
    observation_space = probe_env.observation_space
    action_space = probe_env.action_space
    probe_env.close()

    if config.mode == "true":
        env_fn = lambda: make_raw_env(config.env_id)
    elif config.mode == "partial":
        runtime = GymLearnedRewardRuntime(
            env_id=config.env_id,
            composition="partial",
            observation_space=observation_space,
            action_space=action_space,
            custom_partial=custom_partial,
        )
        env_fn = lambda: GymPreferenceRewardWrapper(make_raw_env(config.env_id), runtime)
    else:
        raise ValueError(f"Unsupported mode for this path: {config.mode}")

    train_env = make_train_env(env_fn, config.n_envs, run_dir / "monitor", normalize)
    eval_env = make_eval_env(config.env_id, train_env)
    callbacks = build_callbacks(config, run_dir, train_env, eval_env, custom_partial)
    model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **hyperparams)
    learn_policy(
        model,
        config.timesteps,
        callbacks,
        progress_bar=config.progress_bar,
        log_interval=config.policy_log_interval,
    )

    return save_and_report(config, model, train_env, eval_env, run_dir, synthetic_queries=0, custom_partial=custom_partial)


def train_preference_mode(config: ExperimentConfig, custom_partial: PartialSpec | None) -> RunResult:
    run_dir = Path(config.log_dir) / config.run_name
    run_dir.mkdir(exist_ok=True, parents=True)
    probe_env = make_raw_env(config.env_id)
    normalize = should_normalize_observation(probe_env.observation_space)
    hyperparams = ppo_hyperparams(probe_env, config)
    observation_space = probe_env.observation_space
    action_space = probe_env.action_space
    reward_model_input_size = flatdim(observation_space) + flatdim(action_space) + 1
    probe_env.close()

    runtime = GymLearnedRewardRuntime(
        env_id=config.env_id,
        composition=config.mode,
        observation_space=observation_space,
        action_space=action_space,
        custom_partial=custom_partial,
        target_mean=config.model_reward_target_mean,
        target_std=config.model_reward_target_std,
        reward_min=config.model_reward_min,
        reward_max=config.model_reward_max,
        reward_scale=config.model_reward_scale,
        normalize=config.normalize_model_reward,
        include_partial_feature=include_partial_feature(config),
    )
    train_env = make_train_env(
        lambda: GymPreferenceRewardWrapper(make_raw_env(config.env_id), runtime),
        config.n_envs,
        run_dir / "monitor",
        normalize,
    )
    eval_env = make_eval_env(config.env_id, train_env)
    callbacks = build_callbacks(config, run_dir, train_env, eval_env, custom_partial)
    model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **hyperparams)

    reward_model = RewardModel(input_size=reward_model_input_size, hidden_sizes=config.reward_hidden_sizes)
    convert_traj = make_trajectory_converter(observation_space, action_space, runtime.include_partial_feature)
    total_queries = RlhfTrainer(
        config,
        model,
        runtime,
        callbacks,
        reward_model,
        convert_traj,
        lambda round_index, collection_steps: collect_policy_trajectories(
            model,
            train_env,
            env_id=config.env_id,
            custom_partial=custom_partial,
            total_timesteps=collection_steps,
            seed=config.seed * 1000 + round_index * 100,
        ),
        continuous=not isinstance(action_space, spaces.Discrete),
        collection_label="Gym steps",
    ).run()

    return save_and_report(
        config,
        model,
        train_env,
        eval_env,
        run_dir,
        synthetic_queries=total_queries,
        runtime=runtime,
        custom_partial=custom_partial,
    )


def save_and_report(
    config: ExperimentConfig,
    model: PPO,
    train_env,
    eval_env,
    run_dir: Path,
    synthetic_queries: int,
    runtime: GymLearnedRewardRuntime | None = None,
    custom_partial: PartialSpec | None = None,
) -> RunResult:
    paths = BackendRunPaths(run_dir)
    model.save(paths.final_model)
    vecnormalize_path = None
    if isinstance(train_env, VecNormalize):
        train_env.save(paths.vecnormalize)
        vecnormalize_path = paths.vecnormalize

    actual_timesteps = int(model.num_timesteps)
    best_logged_reward, best_logged_timestep = report_eval_curve(
        paths.eval_log,
        paths.true_reward_curve,
        max(config.timesteps, actual_timesteps),
        config.plot_mode,
        config.smooth_window,
        x_scale=1e6,
        x_label="Timesteps (millions)",
    )

    final_stats = evaluate_gym_components(
        model,
        config.env_id,
        make_env=make_raw_env,
        custom_partial=custom_partial,
        stats_source=train_env,
        n_eval_episodes=config.final_eval_episodes,
        seed=config.seed + 50_000,
    )
    write_gym_component_summary(paths.final_component_evaluation, actual_timesteps, final_stats, custom_partial)

    final_policy, final_eval_env = select_final_policy(
        config,
        model,
        eval_env,
        run_dir,
        load_eval_env,
        PPO.load,
        load_best_stats=isinstance(train_env, VecNormalize),
    )

    mean_reward, std_reward = evaluate_policy(
        final_policy,
        final_eval_env,
        n_eval_episodes=config.final_eval_episodes,
        deterministic=True,
        return_episode_rewards=False,
    )
    selected_stats = evaluate_gym_components(
        final_policy,
        config.env_id,
        make_env=make_raw_env,
        custom_partial=custom_partial,
        stats_source=final_eval_env,
        n_eval_episodes=config.final_eval_episodes,
        seed=config.seed + 60_000,
    )

    metadata = {
        "env_id": config.env_id,
        "env_slug": slugify(config.env_id),
        "mode": config.mode,
        "run_name": config.run_name,
        "variant": config.variant_name,
        "requested_timesteps": config.timesteps,
        "actual_timesteps": actual_timesteps,
        "seed": config.seed,
        "n_envs": config.n_envs,
        "initial_timesteps": config.initial_timesteps,
        "policy_timesteps_per_round": config.policy_timesteps_per_round,
        "final_policy_timesteps": config.final_policy_timesteps,
        "policy_learning_kwargs": config.policy_learning_kwargs or {},
        "synthetic_queries": synthetic_queries,
        "query_budget": config.query_budget if config.mode in {"feedback", "naive", "delta"} else 0,
        "fragment_length": config.fragment_length if config.mode in {"feedback", "naive", "delta"} else None,
        "active_learning": config.active_learning if config.mode in {"feedback", "naive", "delta"} else None,
        "reward_hidden_sizes": list(config.reward_hidden_sizes),
        "reward_model_lr": config.reward_model_lr if config.mode in {"feedback", "naive", "delta"} else None,
        "pretrain_reward_model": config.pretrain_reward_model if config.mode in {"feedback", "naive", "delta"} else None,
        "pretrain_target": config.pretrain_target if config.pretrain_reward_model else None,
        "include_partial_feature": include_partial_feature(config) if config.mode in {"feedback", "naive", "delta"} else None,
        "partial_reference": config.partial,
        "partial_keys": [custom_partial.name] if custom_partial else [],
        "component_keys": list(component_keys(custom_partial)),
        "best_logged_true_reward": best_logged_reward,
        "best_logged_timestep": best_logged_timestep,
        "selected_policy_true_reward_mean": float(mean_reward),
        "selected_policy_true_reward_std": float(std_reward),
        "selected_policy_components": selected_stats,
    }
    if runtime is not None:
        metadata.update(
            {
                "model_reward_min": runtime.reward_min,
                "model_reward_max": runtime.reward_max,
                "model_reward_scale": runtime.reward_scale,
                "normalize_model_reward": runtime.normalize,
                "model_reward_output_mean": runtime.output_mean,
                "model_reward_output_std": runtime.output_std,
                "model_reward_target_mean": runtime.target_mean,
                "model_reward_target_std": runtime.target_std,
                "reward_composition": runtime.composition,
            }
        )

    metadata_path = paths.metadata
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"{config.final_policy.title()} deterministic true reward: {mean_reward:.3f} +/- {std_reward:.3f}")
    print(
        "Component means: "
        f"total={selected_stats['mean_total']:.3f}, "
        f"partial={selected_stats['mean_partial']:.3f}, "
        f"residual={selected_stats['mean_residual']:.3f}"
    )
    print(f"Synthetic queries consumed: {synthetic_queries}")
    print(f"Saved model and logs to {run_dir}")

    train_env.close()
    final_eval_env.close()
    return RunResult(
        run_dir=run_dir,
        metadata_path=metadata_path,
        model_path=paths.final_model.with_suffix(".zip"),
        vecnormalize_path=vecnormalize_path,
        synthetic_queries=synthetic_queries,
        metadata=metadata,
    )


def default_run_name(config: ExperimentConfig) -> str:
    variant = config.variant_name or config.mode
    steps = f"{config.timesteps // 1_000_000}m" if config.timesteps >= 1_000_000 else f"{config.timesteps}"
    return f"{slugify(config.env_id)}_{variant}_{steps}_seed{config.seed}"


def slugify(env_id: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in env_id.rsplit("-", 1)[0]).strip("_")


def _resolve_custom_partial(config: ExperimentConfig) -> PartialSpec | None:
    return resolve_custom_partial(config)
