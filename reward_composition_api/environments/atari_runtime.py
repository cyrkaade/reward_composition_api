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

from local_gym.classes.atari_reward_specs import AtariRewardSpec
from reward_composition_api.config import ExperimentConfig
from reward_composition_api.data_structures import Trajectory
from reward_composition_api.environments.vectorized import (
    load_vecnormalize_eval_env,
    make_raw_eval_env as make_common_raw_eval_env,
    normalize_obs,
)
from reward_composition_api.environments.trajectory_collector import PolicyTrajectoryCollector
from reward_composition_api.registry import PartialSpec
from reward_composition_api.wrappers.preference_reward import BaseLearnedRewardRuntime, BasePreferenceRewardWrapper
from reward_composition_api.reward_models.reward_model import RewardModel

from reward_composition_api.wrappers.atari import AtariFireResetEnv


GENERIC_ATARI_RAM_PPO_PRESET = {
    "policy": "MlpPolicy",
    "n_steps": 128,
    "batch_size": 256,
    "gamma": 0.99,
    "learning_rate": 2.5e-4,
    "ent_coef": 0.01,
    "clip_range": 0.1,
    "n_epochs": 4,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    "policy_kwargs": {
        "activation_fn": nn.ReLU,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    },
}


@dataclass
class AtariLearnedRewardRuntime(BaseLearnedRewardRuntime):
    spec: AtariRewardSpec
    composition: str
    action_n: int
    partial_source: str = "life_loss"
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


class AtariPreferenceRewardWrapper(BasePreferenceRewardWrapper):
    def __init__(self, env, runtime: AtariLearnedRewardRuntime):
        super().__init__(env, runtime)
        self.tracker = runtime.spec.new_tracker()
        self.partial = runtime.custom_partial.create(runtime.spec.env_id) if runtime.custom_partial else None

    def reset_reward_state(self, info: dict) -> dict:
        step = self.tracker.reset(info)
        if self.partial is not None:
            self.partial.reset(info)
        reset_info = step.as_info()
        reset_info["model_reward"] = 0.0
        reset_info["learned_reward"] = 0.0
        return reset_info

    def partial_reward(self, previous_obs, action, observation, true_reward, terminated, truncated, info):
        if self.partial is not None:
            step = self.partial.step(previous_obs, action, observation, true_reward, terminated, truncated, info)
            return step.partial, step.components
        step = self.tracker.step(info, true_reward=float(true_reward), partial_source=self.runtime.partial_source)
        info.update(step.as_info())
        return step.partial, {
            "life_loss_penalty": step.life_loss_penalty,
            "score_partial": step.score_partial,
            "lost_lives": step.lost_lives,
            "lives": step.lives,
        }

    def model_features(self, observation, action, partial_reward: float) -> np.ndarray:
        return reward_model_features(
            observation,
            action,
            partial_reward,
            self.runtime.action_n,
            self.runtime.include_partial_feature,
        )

    def unsupported_composition_message(self) -> str:
        return f"Unsupported Atari reward composition: {self.runtime.composition}"


def register_atari_envs() -> None:
    try:
        import ale_py
    except ImportError as exc:
        raise RuntimeError("Atari experiments require ale-py. Install it with `pip install ale-py`.") from exc

    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)

    if "ALE/Breakout-v5" not in gym.envs.registry:
        from ale_py.registration import register_v5_envs

        register_v5_envs()


def atari_observation_features(observation) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32).reshape(-1) / 255.0


def one_hot_action(action, action_n: int) -> np.ndarray:
    action_index = int(np.asarray(action).reshape(-1)[0])
    if action_index < 0 or action_index >= action_n:
        raise ValueError(f"Action index {action_index} is outside action space size {action_n}")
    features = np.zeros(action_n, dtype=np.float32)
    features[action_index] = 1.0
    return features


def reward_model_features(observation, action, partial_reward: float, action_n: int, include_partial_feature: bool) -> np.ndarray:
    partial_feature = partial_reward if include_partial_feature else 0.0
    return np.concatenate(
        [
            atari_observation_features(observation),
            one_hot_action(action, action_n),
            np.asarray([partial_feature], dtype=np.float32),
        ]
    )


def make_raw_env(env_id: str):
    register_atari_envs()
    env = gym.make(env_id, obs_type="ram", frameskip=4, repeat_action_probability=0.25)
    return AtariFireResetEnv(env)


def ppo_hyperparams(config: ExperimentConfig):
    hyperparams = deepcopy(GENERIC_ATARI_RAM_PPO_PRESET)
    hyperparams.update(config.policy_learning_kwargs or {})
    return hyperparams


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


def make_trajectory_converter(action_n: int, include_partial_feature: bool):
    def convert(trajectory: Trajectory):
        return [
            reward_model_features(
                state["obs"],
                state["act"],
                state["partial_rew"],
                action_n,
                include_partial_feature,
            ).tolist()
            for state in trajectory.states
        ]

    return convert


def collect_policy_trajectories(
    model: PPO,
    stats_source,
    env_id: str,
    spec: AtariRewardSpec,
    partial_source: str,
    custom_partial: PartialSpec | None,
    total_timesteps: int,
    seed: int,
) -> list[Trajectory]:
    tracker = spec.new_tracker()
    collector = PolicyTrajectoryCollector(
        model=model,
        stats_source=stats_source,
        make_env=make_raw_env,
        env_id=env_id,
        custom_partial=custom_partial,
        model_observation=normalize_obs,
        action_converter=lambda _env, action: int(np.asarray(action).reshape(-1)[0]),
        default_partial_reward=lambda _obs, _action, _new_obs, true_reward, _terminated, _truncated, info: tracker.step(
            info,
            true_reward=true_reward,
            partial_source=partial_source,
        ).partial,
        reset_reward_state=tracker.reset,
    )
    return collector.rollout_trajectories(total_timesteps=total_timesteps, seed=seed)
