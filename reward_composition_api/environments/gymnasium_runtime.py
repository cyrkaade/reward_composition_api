from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from reward_composition_api.environments.spaces import (
    action_features,
    observation_features,
)
from reward_composition_api.registry import PartialSpec
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
        return reward_model_features(
            self.runtime.observation_space,
            self.runtime.action_space,
            observation,
            action,
            partial_reward,
            self.runtime.include_partial_feature,
        )


def reward_model_features(
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
