from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from local_gym.classes.mujoco_reward_specs import MuJoCoRewardSpec
from reward_composition_api.config import ExperimentConfig
from reward_composition_api.data_structures import Trajectory
from reward_composition_api.environments.vectorized import (
    load_vecnormalize_eval_env,
    make_raw_eval_env as make_common_raw_eval_env,
    normalize_obs,
)
from reward_composition_api.registry import PartialSpec
from reward_composition_api.wrappers.preference_reward import BaseLearnedRewardRuntime, BasePreferenceRewardWrapper
from reward_model.reward_model import RewardModel

REACHER_V5_PPO_PRESETS = {
    "mujoco_reacher": {
        "recommended_n_envs": 1,
        "hyperparams": {
            "policy": "MlpPolicy",
            "n_steps": 512,
            "batch_size": 32,
            "gamma": 0.9,
            "learning_rate": 0.000104019,
            "ent_coef": 7.52585e-08,
            "clip_range": 0.3,
            "n_epochs": 5,
            "gae_lambda": 1.0,
            "max_grad_norm": 0.9,
            "vf_coef": 0.950368,
            "policy_kwargs": {
                "log_std_init": -2,
                "ortho_init": False,
                "activation_fn": nn.ReLU,
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
        },
    }
}


GENERIC_MUJOCO_PPO_PRESET = {
    "policy": "MlpPolicy",
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
    "policy_kwargs": {
        "activation_fn": nn.Tanh,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    },
}


@dataclass
class MuJoCoLearnedRewardRuntime(BaseLearnedRewardRuntime):
    spec: MuJoCoRewardSpec
    composition: str
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


class MuJoCoPreferenceRewardWrapper(BasePreferenceRewardWrapper):
    def __init__(self, env, runtime: MuJoCoLearnedRewardRuntime):
        super().__init__(env, runtime)
        self.partial = runtime.custom_partial.create(runtime.spec.env_id) if runtime.custom_partial else None

    def reset_reward_state(self, info: dict) -> dict:
        if self.partial is not None:
            self.partial.reset(info)
        return {}

    def partial_reward(self, previous_obs, action, observation, true_reward, terminated, truncated, info):
        if self.partial is None:
            return self.runtime.spec.partial_reward(info), {}
        step = self.partial.step(previous_obs, action, observation, true_reward, terminated, truncated, info)
        return step.partial, step.components

    def model_features(self, observation, action, partial_reward: float) -> np.ndarray:
        partial_feature = partial_reward if self.runtime.include_partial_feature else 0.0
        return np.concatenate(
            [
                np.asarray(observation, dtype=np.float32).reshape(-1),
                np.asarray(action, dtype=np.float32).reshape(-1),
                np.asarray([partial_feature], dtype=np.float32),
            ]
        )

    def unsupported_composition_message(self) -> str:
        return f"Unsupported learned-reward composition: {self.runtime.composition}"

    def true_reward_info_value(self, true_reward):
        return true_reward


def make_raw_env(env_id: str):
    return gym.make(env_id)


def make_raw_eval_env(env_id: str):
    return make_common_raw_eval_env(make_raw_env, env_id)


def make_vecnormalize_env(env_fn, n_envs: int, monitor_dir: Path) -> VecNormalize:
    env = make_vec_env(
        env_fn,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,
        monitor_dir=str(monitor_dir),
    )
    return VecNormalize(env, norm_obs=True, norm_reward=True)


def make_eval_env(env_id: str, stats_source: VecNormalize | None = None) -> VecNormalize:
    env = VecNormalize(make_raw_eval_env(env_id), norm_obs=True, norm_reward=False, training=False)
    if stats_source is not None:
        env.obs_rms = stats_source.obs_rms
        env.ret_rms = stats_source.ret_rms
    return env


def load_eval_env(env_id: str, stats_path: Path) -> VecNormalize:
    return load_vecnormalize_eval_env(env_id, stats_path, make_raw_eval_env)


def make_trajectory_converter(include_partial_feature: bool):
    def convert(trajectory: Trajectory):
        rows = []
        for state in trajectory.states:
            partial_feature = state["partial_rew"] if include_partial_feature else 0.0
            rows.append(
                [
                    *np.asarray(state["obs"], dtype=np.float32).reshape(-1).tolist(),
                    *np.asarray(state["act"], dtype=np.float32).reshape(-1).tolist(),
                    float(partial_feature),
                ]
            )
        return rows

    return convert


def collect_policy_trajectories(
    model: PPO,
    stats_source,
    env_id: str,
    spec: MuJoCoRewardSpec,
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
            model_obs = normalize_obs(stats_source, obs)
            action, _ = model.predict(model_obs, deterministic=False)
            action = action[0]
            new_obs, true_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if partial is None:
                partial_reward = spec.partial_reward(info)
            else:
                partial_reward = partial.step(obs, action, new_obs, true_reward, terminated, truncated, info).partial
            trajectory.push_state(new_obs, action, done, info, float(true_reward), partial_reward)
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


def ppo_hyperparams(config: ExperimentConfig):
    if config.preset == "reacher" or (config.preset == "auto" and config.env_id == "Reacher-v5"):
        hyperparams = deepcopy(REACHER_V5_PPO_PRESETS["mujoco_reacher"]["hyperparams"])
    else:
        hyperparams = deepcopy(GENERIC_MUJOCO_PPO_PRESET)
    hyperparams.update(config.policy_learning_kwargs or {})
    return hyperparams
