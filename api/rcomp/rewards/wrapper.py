"""The single learned-reward runtime and env wrapper shared by all suites
and all five reward compositions. Suite differences are injected through the
runtime: the observation-feature function, the info dict emitted at reset,
and whether the true reward is cast to float in ``info``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from ..envs import action_features
from ..partials import PartialSpec
from .model import RewardModel


def reward_model_features(
    observation_features: Callable[[spaces.Space, Any], np.ndarray],
    observation_space: spaces.Space,
    action_space: spaces.Space,
    observation,
    action,
    partial_reward: float,
    include_partial_feature: bool,
) -> np.ndarray:
    partial_feature = partial_reward if include_partial_feature else 0.0
    return np.concatenate(
        [
            observation_features(observation_space, observation),
            action_features(action_space, action),
            np.asarray([partial_feature], dtype=np.float32),
        ]
    )


@dataclass
class LearnedRewardRuntime:
    env_id: str
    composition: str
    observation_space: spaces.Space
    action_space: spaces.Space
    observation_features: Callable[[spaces.Space, Any], np.ndarray]
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
    normalize_partial: bool = False
    partial_mean: float = 0.0
    partial_std: float = 1.0
    partial_alpha: float = 1.0
    include_partial_feature: bool = True
    reset_info: dict[str, float] = field(default_factory=dict)
    cast_true_reward: bool = True

    def transform_model_output(self, value: float) -> float:
        if self.normalize and self.output_mean is not None and self.output_std is not None:
            value = (value - self.output_mean) / max(self.output_std, 1e-8) * self.target_std + self.target_mean
        value *= self.reward_scale
        if self.reward_min is not None or self.reward_max is not None:
            value = float(np.clip(value, self.reward_min, self.reward_max))
        return value

    def transform_partial_reward(self, value: float) -> float:
        if self.normalize_partial:
            value = (value - self.partial_mean) / max(self.partial_std, 1e-8)
        return value

    def composed_partial_reward(self, value: float) -> float:
        return self.partial_alpha * self.transform_partial_reward(value)

    def model_features(self, observation, action, partial_reward: float) -> np.ndarray:
        return reward_model_features(
            self.observation_features,
            self.observation_space,
            self.action_space,
            observation,
            action,
            self.transform_partial_reward(partial_reward),
            self.include_partial_feature,
        )


class PreferenceRewardWrapper(gym.Wrapper):
    def __init__(self, env, runtime: LearnedRewardRuntime):
        super().__init__(env)
        self.runtime = runtime
        self.partial = runtime.custom_partial.create(runtime.env_id) if runtime.custom_partial else None
        self._last_obs = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_obs = observation
        if self.partial is not None:
            self.partial.reset(info)
        info.update(dict(self.runtime.reset_info))
        return observation, info

    def partial_reward(self, previous_obs, action, observation, true_reward, terminated, truncated, info) -> tuple[float, dict]:
        if self.partial is None:
            return 0.0, {}
        step = self.partial.step(previous_obs, action, observation, true_reward, terminated, truncated, info)
        return step.partial, step.components

    def model_reward(self, observation, action, partial_reward: float) -> float:
        reward_models = self.runtime.reward_models or ([self.runtime.reward_model] if self.runtime.reward_model is not None else [])
        if not reward_models:
            return 0.0

        model_input = self.runtime.model_features(observation, action, partial_reward)
        with th.no_grad():
            model_tensor = th.as_tensor(model_input, dtype=th.float32).view(1, -1)
            outputs = th.stack([model(model_tensor).reshape(-1)[0] for model in reward_models])
            output = th.mean(outputs)
        return self.runtime.transform_model_output(float(output.item()))

    def compose_reward(self, partial_reward: float, model_reward: float) -> float:
        if self.runtime.composition == "partial":
            return partial_reward
        if self.runtime.composition == "feedback":
            return model_reward
        if self.runtime.composition in {"naive", "delta"}:
            return self.runtime.composed_partial_reward(partial_reward) + model_reward
        raise ValueError(f"Unsupported reward composition: {self.runtime.composition}")

    def step(self, action):
        previous_obs = self._last_obs
        observation, true_reward, terminated, truncated, info = self.env.step(action)
        partial_reward, partial_components = self.partial_reward(
            previous_obs,
            action,
            observation,
            true_reward,
            terminated,
            truncated,
            info,
        )
        model_reward = self.model_reward(observation, action, partial_reward)
        training_reward = self.compose_reward(partial_reward, model_reward)

        info["true_reward"] = float(true_reward) if self.runtime.cast_true_reward else true_reward
        info["partial_reward"] = partial_reward
        info["partial_components"] = partial_components
        info["model_reward"] = model_reward
        info["learned_reward"] = training_reward
        self._last_obs = observation
        return observation, training_reward, terminated, truncated, info
