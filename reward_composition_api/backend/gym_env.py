from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from local_gym.wrappers.buffering_wrapper import Trajectory
from local_gym.wrappers.lunar_lander_rewards_wrapper import LunarLanderSaveInfo
from reward_composition_api.config import ExperimentConfig
from reward_composition_api.registry import PartialSpec
from reward_model.reward_model import RewardModel

from .common import load_vecnormalize_eval_env, make_raw_eval_env as make_common_raw_eval_env
from .gym_spaces import (
    action_features,
    action_for_space,
    is_image_space,
    observation_features,
    policy_observation,
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


def make_raw_env(env_id: str):
    env = gym.make(env_id)
    if env_id.startswith("LunarLander"):
        return LunarLanderSaveInfo(env)
    return env


def make_raw_eval_env(env_id: str):
    return make_common_raw_eval_env(make_raw_env, env_id)


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
        "policy": "CnnPolicy" if is_image_space(env.observation_space) else "MlpPolicy",
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
