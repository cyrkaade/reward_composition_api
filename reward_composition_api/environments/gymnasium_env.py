from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatdim
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from reward_composition_api.config import ExperimentConfig
from reward_composition_api.data_structures import Trajectory
from reward_composition_api.environments.trajectory_collector import PolicyTrajectoryCollector
from reward_composition_api.registry import PartialSpec

from .gymnasium_runtime import GymLearnedRewardRuntime, GymPreferenceRewardWrapper, reward_model_features
from .spaces import (
    action_for_space,
    is_image_space,
    policy_observation,
    should_normalize_observation,
)
from .vectorized import load_vecnormalize_eval_env, make_raw_eval_env as make_common_raw_eval_env


class GymnasiumEnvironmentProfile:
    def make_raw_env(self, env_id: str):
        return gym.make(env_id)

    def make_raw_eval_env(self, env_id: str):
        return make_common_raw_eval_env(self.make_raw_env, env_id)

    def make_train_env(self, env_fn, n_envs: int, monitor_dir: Path, normalize: bool):
        env = make_vec_env(env_fn, n_envs=n_envs, vec_env_cls=DummyVecEnv, monitor_dir=str(monitor_dir))
        if normalize:
            return VecNormalize(env, norm_obs=True, norm_reward=True)
        return env

    def make_eval_env(self, env_id: str, stats_source=None):
        env = self.make_raw_eval_env(env_id)
        if isinstance(stats_source, VecNormalize):
            eval_env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
            eval_env.obs_rms = stats_source.obs_rms
            eval_env.ret_rms = stats_source.ret_rms
            return eval_env
        return env

    def load_eval_env(self, env_id: str, stats_path: Path):
        return load_vecnormalize_eval_env(env_id, stats_path, self.make_raw_eval_env)

    def ppo_hyperparams(self, env: gym.Env, config: ExperimentConfig):
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

    def should_normalize_observation(self, observation_space: spaces.Space) -> bool:
        return should_normalize_observation(observation_space)

    def learned_runtime(
        self,
        env_id: str,
        composition: str,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        custom_partial: PartialSpec | None,
        **kwargs,
    ):
        return GymLearnedRewardRuntime(
            env_id=env_id,
            composition=composition,
            observation_space=observation_space,
            action_space=action_space,
            custom_partial=custom_partial,
            **kwargs,
        )

    def preference_wrapper(self, env, runtime):
        return GymPreferenceRewardWrapper(env, runtime)

    def reward_model_input_size(self, observation_space: spaces.Space, action_space: spaces.Space) -> int:
        return flatdim(observation_space) + flatdim(action_space) + 1

    def trajectory_converter(self, observation_space: spaces.Space, action_space: spaces.Space, include_partial_feature: bool):
        def convert(trajectory: Trajectory):
            return [
                reward_model_features(
                    observation_space,
                    action_space,
                    state["obs"],
                    state["act"],
                    state["partial_rew"],
                    include_partial_feature,
                ).tolist()
                for state in trajectory.states
            ]

        return convert

    def collect_policy_trajectories(
        self,
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
            make_env=self.make_raw_env,
            env_id=env_id,
            custom_partial=custom_partial,
            model_observation=policy_observation,
            action_converter=lambda env, action: action_for_space(env.action_space, action),
        )
        return collector.rollout_trajectories(total_timesteps=total_timesteps, seed=seed)
