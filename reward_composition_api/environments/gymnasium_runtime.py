from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from reward_composition_api.config import ExperimentConfig
from reward_composition_api.data_structures import Trajectory
from reward_composition_api.environments.spaces import (
    action_features,
    action_for_space,
    is_image_space,
    observation_features,
    policy_observation,
)
from reward_composition_api.environments.trajectory_collector import PolicyTrajectoryCollector
from reward_composition_api.environments.vectorized import load_vecnormalize_eval_env, make_raw_eval_env as make_common_raw_eval_env
from reward_composition_api.registry import PartialSpec
from reward_composition_api.wrappers.lunar_lander import LunarLanderSaveInfo
from reward_composition_api.wrappers.preference_reward import BaseLearnedRewardRuntime, BasePreferenceRewardWrapper
from reward_composition_api.reward_models.reward_model import RewardModel

@dataclass
class GymLearnedRewardRuntime(BaseLearnedRewardRuntime):
    env_id: str
    composition: str
    observation_space: spaces.Space
    action_space: spaces.Space
    custom_partial: PartialSpec | None = None
    reward_model: RewardModel | None = None
    reward_models: list[RewardModel] | None = None
    output_mean: float | None = None
    output_std: float | None = None
    target_mean: float = 0.0
    target_std: float = 1.0
    reward_min: float | None = None
    reward_max: float | None = None
    reward_scale: float = 1.0
    normalize: bool = False
    include_partial_feature: bool = True


class GymPreferenceRewardWrapper(BasePreferenceRewardWrapper):
    def __init__(self, env, runtime: GymLearnedRewardRuntime):
        super().__init__(env, runtime)
        self.partial = runtime.custom_partial.create(runtime.env_id) if runtime.custom_partial else None

    def reset_reward_state(self, info: dict) -> dict:
        if self.partial is not None:
            self.partial.reset(info)
        return {}

    def partial_reward(self, previous_obs, action, observation, true_reward, terminated, truncated, info):
        if self.partial is None:
            return 0.0, {}
        step = self.partial.step(previous_obs, action, observation, true_reward, terminated, truncated, info)
        return step.partial, step.components

    def model_features(self, observation, action, partial_reward: float) -> np.ndarray:
        partial_feature = partial_reward if self.runtime.include_partial_feature else 0.0
        return np.concatenate(
            [
                observation_features(self.runtime.observation_space, observation),
                action_features(self.runtime.action_space, action),
                np.asarray([partial_feature], dtype=np.float32),
            ]
        )


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
    collector = PolicyTrajectoryCollector(
        model=model,
        stats_source=stats_source,
        make_env=make_raw_env,
        env_id=env_id,
        custom_partial=custom_partial,
        model_observation=policy_observation,
        action_converter=lambda env, action: action_for_space(env.action_space, action),
    )
    return collector.rollout_trajectories(total_timesteps=total_timesteps, seed=seed)


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
