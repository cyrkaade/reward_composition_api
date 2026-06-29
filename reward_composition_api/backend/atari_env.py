from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from local_gym.classes.atari_reward_specs import AtariRewardSpec
from local_gym.wrappers.buffering_wrapper import Trajectory
from reward_composition_api.config import ExperimentConfig
from reward_composition_api.registry import PartialSpec
from reward_model.reward_model import RewardModel

from .common import load_vecnormalize_eval_env, make_raw_eval_env as make_common_raw_eval_env, normalize_obs


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
class AtariLearnedRewardRuntime:
    spec: AtariRewardSpec
    composition: str
    action_n: int
    partial_source: str = "life_loss"
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


class AtariPreferenceRewardWrapper(gym.Wrapper):
    def __init__(self, env, runtime: AtariLearnedRewardRuntime):
        super().__init__(env)
        self.runtime = runtime
        self.tracker = runtime.spec.new_tracker()
        self.partial = runtime.custom_partial.create(runtime.spec.env_id) if runtime.custom_partial else None
        self._last_obs = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_obs = observation
        step = self.tracker.reset(info)
        if self.partial is not None:
            self.partial.reset(info)
        info.update(step.as_info())
        info["model_reward"] = 0.0
        info["learned_reward"] = 0.0
        return observation, info

    def _partial_reward(self, previous_obs, action, observation, true_reward, terminated, truncated, info):
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

    def _model_reward(self, observation, action, partial_reward):
        if self.runtime.reward_model is None:
            return 0.0

        partial_feature = partial_reward if self.runtime.include_partial_feature else 0.0
        model_input = np.concatenate(
            [
                atari_observation_features(observation),
                one_hot_action(action, self.runtime.action_n),
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
            raise ValueError(f"Unsupported Atari reward composition: {self.runtime.composition}")

        info["true_reward"] = float(true_reward)
        info["partial_reward"] = partial_reward
        info["partial_components"] = partial_components
        info["model_reward"] = model_reward
        info["learned_reward"] = training_reward
        self._last_obs = observation
        return observation, training_reward, terminated, truncated, info


class AtariFireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        meanings = list(env.unwrapped.get_action_meanings())
        self.fire_action = meanings.index("FIRE") if "FIRE" in meanings else None
        self.second_action = 2 if self.fire_action is not None and len(meanings) > 2 else None
        self.prev_lives = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.fire_action is None:
            self.prev_lives = int(info.get("lives", 0))
            return observation, info

        observation, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            observation, info = self.env.reset(**kwargs)
            self.prev_lives = int(info.get("lives", 0))
            return observation, info

        if self.second_action is not None:
            observation, _, terminated, truncated, info = self.env.step(self.second_action)
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)
                self.prev_lives = int(info.get("lives", 0))
                return observation, info

        self.prev_lives = int(info.get("lives", 0))
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        lives = int(info.get("lives", 0))
        lost_life = self.fire_action is not None and not (terminated or truncated) and 0 < lives < self.prev_lives

        if lost_life:
            observation, extra_reward, terminated, truncated, info = self.env.step(self.fire_action)
            reward += extra_reward
            if self.second_action is not None and not (terminated or truncated):
                observation, extra_reward, terminated, truncated, info = self.env.step(self.second_action)
                reward += extra_reward

        self.prev_lives = int(info.get("lives", 0))
        return observation, reward, terminated, truncated, info


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
        rows = []
        for state in trajectory.states:
            partial_feature = state["partial_rew"] if include_partial_feature else 0.0
            rows.append(
                [
                    *atari_observation_features(state["obs"]).tolist(),
                    *one_hot_action(state["act"], action_n).tolist(),
                    float(partial_feature),
                ]
            )
        return rows

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
    env = make_raw_env(env_id)
    partial = custom_partial.create(env_id) if custom_partial else None
    trajectories = []
    trajectory = Trajectory()
    obs, info = env.reset(seed=seed)
    tracker = spec.new_tracker()
    tracker.reset(info)
    if partial is not None:
        partial.reset(info)
    steps = 0

    try:
        while steps < total_timesteps:
            model_obs = normalize_obs(stats_source, obs)
            action, _ = model.predict(model_obs, deterministic=False)
            action = int(np.asarray(action).reshape(-1)[0])
            new_obs, true_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if partial is None:
                partial_reward = tracker.step(info, true_reward=float(true_reward), partial_source=partial_source).partial
            else:
                partial_reward = partial.step(obs, action, new_obs, true_reward, terminated, truncated, info).partial
            trajectory.push_state(new_obs, action, done, info, float(true_reward), partial_reward)
            steps += 1

            if done:
                trajectories.append(trajectory)
                trajectory = Trajectory()
                obs, info = env.reset()
                tracker.reset(info)
                if partial is not None:
                    partial.reset(info)
            else:
                obs = new_obs
    finally:
        env.close()

    if trajectory.states:
        trajectories.append(trajectory)
    return trajectories
