from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.spaces.utils import flatdim, flatten
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from local_gym.wrappers.lunar_lander_rewards_wrapper import LunarLanderSaveInfo
from local_gym.wrappers.buffering_wrapper import Trajectory
from reward_model.reward_model import RewardModel
from reward_composition_api.config import ExperimentConfig
from reward_composition_api.registry import PartialSpec
from reward_composition_api.results import RunResult

from .common import (
    BackendRunPaths,
    ComponentEvalCallback,
    SaveVecNormalizeOnBest,
    choose_query_pairs,
    include_partial_feature,
    learn_policy,
    load_vecnormalize_eval_env,
    make_raw_eval_env as make_common_raw_eval_env,
    policy_training_schedule,
    pretrain_reward_model,
    query_schedule,
    rate_pairs_from_true_reward,
    report_eval_curve,
    resolve_custom_partial,
    reward_model_io_stats,
    select_final_policy,
    summarize_component_rows,
    train_preference_reward_model,
    write_component_summary_csv,
)


@dataclass
class GymLearnedRewardRuntime:
    env_id: str
    composition: str
    observation_space: spaces.Space
    action_space: spaces.Space
    custom_partial: PartialSpec | None = None
    reward_model: RewardModel | None = None
    output_mean: float | None = None
    output_std: float | None = None
    target_mean: float = 0.0
    target_std: float = 1.0
    reward_min: float | None = None
    reward_max: float | None = None
    reward_scale: float = 1.0
    normalize: bool = False
    include_partial_feature: bool = True


class GymPreferenceRewardWrapper(gym.Wrapper):
    def __init__(self, env, runtime: GymLearnedRewardRuntime):
        super().__init__(env)
        self.runtime = runtime
        self.partial = runtime.custom_partial.create(runtime.env_id) if runtime.custom_partial else None
        self._last_obs = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_obs = observation
        if self.partial is not None:
            self.partial.reset(info)
        return observation, info

    def _partial_reward(self, previous_obs, action, observation, true_reward, terminated, truncated, info):
        if self.partial is None:
            return 0.0, {}
        step = self.partial.step(previous_obs, action, observation, true_reward, terminated, truncated, info)
        return step.partial, step.components

    def _model_reward(self, observation, action, partial_reward):
        if self.runtime.reward_model is None:
            return 0.0

        partial_feature = partial_reward if self.runtime.include_partial_feature else 0.0
        model_input = np.concatenate(
            [
                observation_features(self.runtime.observation_space, observation),
                action_features(self.runtime.action_space, action),
                np.asarray([partial_feature], dtype=np.float32),
            ]
        )
        with th.no_grad():
            output = self.runtime.reward_model(th.as_tensor(model_input, dtype=th.float32).view(1, -1)).reshape(-1)[0]

        value = float(output.item())
        if self.runtime.normalize and self.runtime.output_mean is not None and self.runtime.output_std is not None:
            value = (
                (value - self.runtime.output_mean)
                / max(self.runtime.output_std, 1e-8)
                * self.runtime.target_std
                + self.runtime.target_mean
            )
        value *= self.runtime.reward_scale
        if self.runtime.reward_min is not None or self.runtime.reward_max is not None:
            value = float(np.clip(value, self.runtime.reward_min, self.runtime.reward_max))
        return value

    def step(self, action):
        previous_obs = self._last_obs
        observation, true_reward, terminated, truncated, info = self.env.step(action)
        partial_reward, partial_components = self._partial_reward(
            previous_obs,
            action,
            observation,
            true_reward,
            terminated,
            truncated,
            info,
        )
        model_reward = self._model_reward(observation, action, partial_reward)

        if self.runtime.composition == "partial":
            training_reward = partial_reward
        elif self.runtime.composition == "feedback":
            training_reward = model_reward
        elif self.runtime.composition in {"naive", "delta"}:
            training_reward = partial_reward + model_reward
        else:
            raise ValueError(f"Unsupported reward composition: {self.runtime.composition}")

        info["true_reward"] = float(true_reward)
        info["partial_reward"] = partial_reward
        info["partial_components"] = partial_components
        info["model_reward"] = model_reward
        info["learned_reward"] = training_reward
        self._last_obs = observation
        return observation, training_reward, terminated, truncated, info


def run_gym_experiment(config: ExperimentConfig) -> RunResult:
    custom_partial = _resolve_custom_partial(config)
    run_name = config.run_name or default_run_name(config)
    variant_name = config.variant_name or config.mode
    config = replace(config, run_name=run_name, variant_name=variant_name)
    if config.mode in {"true", "partial"}:
        return train_true_or_partial(config, custom_partial)
    return train_preference_mode(config, custom_partial)


def make_raw_env(env_id: str):
    env = gym.make(env_id)
    if env_id.startswith("LunarLander"):
        return LunarLanderSaveInfo(env)
    return env


def make_raw_eval_env(env_id: str):
    return make_common_raw_eval_env(make_raw_env, env_id)


def should_normalize_observation(space: spaces.Space) -> bool:
    return isinstance(space, spaces.Box) and len(space.shape or ()) == 1


def make_train_env(env_fn, n_envs: int, monitor_dir: Path, normalize: bool):
    env = make_vec_env(env_fn, n_envs=n_envs, vec_env_cls=DummyVecEnv, monitor_dir=str(monitor_dir))
    if normalize:
        return VecNormalize(env, norm_obs=True, norm_reward=True)
    return env


def make_eval_env(env_id: str, stats_source=None):
    env = make_raw_eval_env(env_id)
    if isinstance(stats_source, VecNormalize):
        eval_env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = stats_source.obs_rms
        eval_env.ret_rms = stats_source.ret_rms
        return eval_env
    return env


def load_eval_env(env_id: str, stats_path: Path):
    return load_vecnormalize_eval_env(env_id, stats_path, make_raw_eval_env)


def policy_observation(stats_source, observation):
    if isinstance(stats_source, VecNormalize):
        return stats_source.normalize_obs(np.asarray(observation)[None])
    return observation


def observation_features(space: spaces.Space, observation) -> np.ndarray:
    return np.asarray(flatten(space, observation), dtype=np.float32).reshape(-1)


def action_features(space: spaces.Space, action) -> np.ndarray:
    return np.asarray(flatten(space, action_for_space(space, action)), dtype=np.float32).reshape(-1)


def action_for_space(space: spaces.Space, action):
    if isinstance(space, spaces.Discrete):
        return int(np.asarray(action).reshape(-1)[0])
    if isinstance(space, spaces.MultiDiscrete):
        return np.asarray(action, dtype=space.dtype).reshape(space.shape)
    if isinstance(space, spaces.MultiBinary):
        return np.asarray(action, dtype=space.dtype).reshape(space.shape)
    return np.asarray(action, dtype=getattr(space, "dtype", np.float32)).reshape(space.shape)


def make_trajectory_converter(observation_space: spaces.Space, action_space: spaces.Space, include_partial_feature: bool):
    def convert(trajectory: Trajectory):
        rows = []
        for state in trajectory.states:
            partial_feature = state["partial_rew"] if include_partial_feature else 0.0
            rows.append(
                [
                    *observation_features(observation_space, state["obs"]).tolist(),
                    *action_features(action_space, state["act"]).tolist(),
                    float(partial_feature),
                ]
            )
        return rows

    return convert


def collect_policy_trajectories(
    model: PPO,
    stats_source,
    env_id: str,
    custom_partial: PartialSpec | None,
    total_timesteps: int,
    seed: int,
) -> list[Trajectory]:
    env = make_raw_env(env_id)
    partial = custom_partial.create(env_id) if custom_partial else None
    trajectories = []
    trajectory = Trajectory()
    obs, info = env.reset(seed=seed)
    if partial is not None:
        partial.reset(info)
    steps = 0

    try:
        while steps < total_timesteps:
            model_obs = policy_observation(stats_source, obs)
            action, _ = model.predict(model_obs, deterministic=False)
            env_action = action_for_space(env.action_space, action)
            new_obs, true_reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            if partial is None:
                partial_reward = 0.0
            else:
                partial_reward = partial.step(obs, env_action, new_obs, true_reward, terminated, truncated, info).partial
            trajectory.push_state(new_obs, env_action, done, info, float(true_reward), partial_reward)
            steps += 1

            if done:
                trajectories.append(trajectory)
                trajectory = Trajectory()
                obs, info = env.reset()
                if partial is not None:
                    partial.reset(info)
            else:
                obs = new_obs
    finally:
        env.close()

    if trajectory.states:
        trajectories.append(trajectory)
    return trajectories


def ppo_hyperparams(env: gym.Env, config: ExperimentConfig):
    hyperparams = {
        "policy": "CnnPolicy" if _is_image_space(env.observation_space) else "MlpPolicy",
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "ent_coef": 0.0,
        "clip_range": 0.2,
        "n_epochs": 10,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
    }
    hyperparams.update(deepcopy(config.policy_learning_kwargs or {}))
    return hyperparams


def _is_image_space(space: spaces.Space) -> bool:
    return isinstance(space, spaces.Box) and len(space.shape or ()) == 3


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
    rated_train = []
    rated_val = []
    total_queries = 0
    pretraining_done = False
    schedule = query_schedule(config.query_budget, config.rlhf_rounds)
    policy_steps_by_round = policy_training_schedule(
        config.timesteps,
        config.rlhf_rounds,
        config.policy_timesteps_per_round,
    )
    add_partial_to_predictions = config.mode in {"naive", "delta"}

    if config.initial_timesteps:
        print(f"initial PPO training on {config.mode} reward for {config.initial_timesteps} timesteps")
        learn_policy(
            model,
            config.initial_timesteps,
            callbacks,
            progress_bar=config.progress_bar,
            reset_num_timesteps=False,
            log_interval=config.policy_log_interval,
        )

    for round_index, round_query_budget in enumerate(schedule):
        collection_steps = config.collection_timesteps * (2 if round_index == 0 else 1)
        print(f"\nPreference round {round_index}: collecting {collection_steps} Gym steps for {round_query_budget} queries")
        trajectories = collect_policy_trajectories(
            model,
            train_env,
            env_id=config.env_id,
            custom_partial=custom_partial,
            total_timesteps=collection_steps,
            seed=config.seed * 1000 + round_index * 100,
        )

        if config.pretrain_reward_model and not pretraining_done:
            print(f"pretraining reward model on {config.pretrain_target} target")
            pretrain_reward_model(
                reward_model,
                trajectories,
                convert_traj,
                target=config.pretrain_target,
                epochs=config.pretrain_epochs,
                batch_size=config.pretrain_batch_size,
                learning_rate=config.pretrain_lr,
            )
            pretraining_done = True

        query_model = reward_model if (total_queries > 0 or pretraining_done) else None
        pairs = choose_query_pairs(
            trajectories,
            query_model,
            query_count=min(round_query_budget, config.query_budget - total_queries),
            fragment_length=config.fragment_length,
            active_learning=config.active_learning,
            convert_traj=convert_traj,
            add_partial_to_predictions=add_partial_to_predictions,
            dropout_samples=config.dropout_samples,
            dropout_p=config.dropout_p,
            active_learning_batches=config.active_learning_batches,
            continuous=not isinstance(action_space, spaces.Discrete),
        )
        rated_pairs = rate_pairs_from_true_reward(pairs)
        split = int(len(rated_pairs) * 0.8)
        rated_train.extend(rated_pairs[:split])
        rated_val.extend(rated_pairs[split:])
        total_queries += len(rated_pairs)
        print(f"rated {len(rated_pairs)} synthetic preference pairs; cumulative={total_queries}")

        if rated_train:
            train_preference_reward_model(
                reward_model,
                rated_train,
                rated_val,
                convert_traj=convert_traj,
                use_delta_loss=config.mode == "delta",
                batch_size=config.reward_model_batch_size,
                epochs=config.reward_model_epochs,
                patience=config.reward_model_patience,
                learning_rate=config.reward_model_lr,
            )
            runtime.reward_model = reward_model
            stat_trajectories = [pair.t1 for pair in rated_train + rated_val] + [pair.t2 for pair in rated_train + rated_val]
            runtime.output_mean, runtime.output_std = reward_model_io_stats(reward_model, stat_trajectories, convert_traj)
            print(f"reward model output stats: mean={runtime.output_mean}, std={runtime.output_std}")

        policy_steps = policy_steps_by_round[round_index]
        print(f"training PPO on {config.mode} reward for {policy_steps} timesteps")
        learn_policy(
            model,
            policy_steps,
            callbacks,
            progress_bar=config.progress_bar,
            reset_num_timesteps=False,
            log_interval=config.policy_log_interval,
        )

        if total_queries >= config.query_budget:
            print("synthetic query budget exhausted")

    if config.final_policy_timesteps:
        print(f"final PPO training on {config.mode} reward for {config.final_policy_timesteps} timesteps")
        learn_policy(
            model,
            config.final_policy_timesteps,
            callbacks,
            progress_bar=config.progress_bar,
            reset_num_timesteps=False,
            log_interval=config.policy_log_interval,
        )

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


def evaluate_gym_components(
    model,
    env_id: str,
    custom_partial: PartialSpec | None = None,
    stats_source=None,
    n_eval_episodes: int = 10,
    seed: int = 0,
    deterministic: bool = True,
):
    rows = []
    env = make_raw_env(env_id)
    partial = custom_partial.create(env_id) if custom_partial else None
    try:
        for episode_index in range(n_eval_episodes):
            obs, info = env.reset(seed=seed + episode_index)
            if partial is not None:
                partial.reset(info)
            done = False
            total = 0.0
            length = 0
            components = {key: 0.0 for key in component_keys(custom_partial)}
            partial_total = 0.0

            while not done:
                model_obs = policy_observation(stats_source, obs)
                action, _ = model.predict(model_obs, deterministic=deterministic)
                env_action = action_for_space(env.action_space, action)
                new_obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
                total += float(reward)
                length += 1

                if partial is None:
                    step_partial = 0.0
                    step_components = {}
                else:
                    partial_step = partial.step(obs, env_action, new_obs, reward, terminated, truncated, info)
                    step_partial = partial_step.partial
                    step_components = partial_step.components
                partial_total += step_partial
                for key, value in step_components.items():
                    if key in components:
                        components[key] += value
                obs = new_obs

            residual = total - partial_total
            rows.append({"total": total, "partial": partial_total, "residual": residual, "length": float(length), **components})
    finally:
        env.close()

    return _summarize_rows(rows, ["total", "partial", "residual", *component_keys(custom_partial), "length"])


def _summarize_rows(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    return summarize_component_rows(rows, keys)


def component_keys(custom_partial: PartialSpec | None = None) -> tuple[str, ...]:
    return custom_partial.component_keys if custom_partial is not None else ()


def component_fieldnames(custom_partial: PartialSpec | None = None) -> list[str]:
    fields = ["timesteps"]
    for key in ["total", "partial", "residual", *component_keys(custom_partial), "length"]:
        fields.extend([f"mean_{key}", f"std_{key}"])
    return fields


def write_gym_component_summary(path: Path, timestep: int, stats: dict, custom_partial: PartialSpec | None = None):
    write_component_summary_csv(path, timestep, stats, component_fieldnames(custom_partial))


class GymComponentEvalCallback(ComponentEvalCallback):
    def __init__(
        self,
        log_path: Path,
        env_id: str,
        custom_partial: PartialSpec | None,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(log_path, eval_freq, n_eval_episodes, seed=seed, verbose=verbose)
        self.env_id = env_id
        self.custom_partial = custom_partial

    def component_fieldnames(self) -> list[str]:
        return component_fieldnames(self.custom_partial)

    def evaluate_components(self) -> dict:
        return evaluate_gym_components(
            self.model,
            self.env_id,
            custom_partial=self.custom_partial,
            stats_source=self.training_env,
            n_eval_episodes=self.n_eval_episodes,
            seed=self.seed + self.num_timesteps,
            deterministic=True,
        )

    def write_summary(self, stats: dict) -> None:
        write_gym_component_summary(self.log_path, self.num_timesteps, stats, self.custom_partial)

    def log_message(self, stats: dict) -> str:
        return (
            "Gym component eval "
            f"env={self.env_id} t={self.num_timesteps}: "
            f"total={stats['mean_total']:.3f}, "
            f"partial={stats['mean_partial']:.3f}, "
            f"residual={stats['mean_residual']:.3f}"
        )


def default_run_name(config: ExperimentConfig) -> str:
    variant = config.variant_name or config.mode
    steps = f"{config.timesteps // 1_000_000}m" if config.timesteps >= 1_000_000 else f"{config.timesteps}"
    return f"{slugify(config.env_id)}_{variant}_{steps}_seed{config.seed}"


def slugify(env_id: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in env_id.rsplit("-", 1)[0]).strip("_")


def _resolve_custom_partial(config: ExperimentConfig) -> PartialSpec | None:
    return resolve_custom_partial(config)
