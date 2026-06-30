from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch as th

from reward_model.reward_model import RewardModel


class BaseLearnedRewardRuntime:
    composition: str
    custom_partial: Any | None
    reward_model: RewardModel | None
    reward_models: list[RewardModel] | None
    output_mean: float | None
    output_std: float | None
    target_mean: float
    target_std: float
    reward_min: float | None
    reward_max: float | None
    reward_scale: float
    normalize: bool
    include_partial_feature: bool

    def transform_model_output(self, value: float) -> float:
        if self.normalize and self.output_mean is not None and self.output_std is not None:
            value = (value - self.output_mean) / max(self.output_std, 1e-8) * self.target_std + self.target_mean
        value *= self.reward_scale
        if self.reward_min is not None or self.reward_max is not None:
            value = float(np.clip(value, self.reward_min, self.reward_max))
        return value


class BasePreferenceRewardWrapper(gym.Wrapper):
    def __init__(self, env, runtime: BaseLearnedRewardRuntime):
        super().__init__(env)
        self.runtime = runtime
        self._last_obs = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_obs = observation
        info.update(self.reset_reward_state(info))
        return observation, info

    def reset_reward_state(self, info: dict) -> dict:
        return {}

    def partial_reward(self, previous_obs, action, observation, true_reward, terminated, truncated, info) -> tuple[float, dict]:
        raise NotImplementedError

    def model_features(self, observation, action, partial_reward: float) -> np.ndarray:
        raise NotImplementedError

    def unsupported_composition_message(self) -> str:
        return f"Unsupported reward composition: {self.runtime.composition}"

    def true_reward_info_value(self, true_reward):
        return float(true_reward)

    def model_reward(self, observation, action, partial_reward: float) -> float:
        reward_models = self.runtime.reward_models or ([self.runtime.reward_model] if self.runtime.reward_model is not None else [])
        if not reward_models:
            return 0.0

        model_input = self.model_features(observation, action, partial_reward)
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
            return partial_reward + model_reward
        raise ValueError(self.unsupported_composition_message())

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

        info["true_reward"] = self.true_reward_info_value(true_reward)
        info["partial_reward"] = partial_reward
        info["partial_components"] = partial_components
        info["model_reward"] = model_reward
        info["learned_reward"] = training_reward
        self._last_obs = observation
        return observation, training_reward, terminated, truncated, info
