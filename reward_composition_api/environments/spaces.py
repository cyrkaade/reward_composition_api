from __future__ import annotations

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import flatten
from stable_baselines3.common.vec_env import VecNormalize


def should_normalize_observation(space: spaces.Space) -> bool:
    return isinstance(space, spaces.Box) and len(space.shape or ()) == 1


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


def is_image_space(space: spaces.Space) -> bool:
    return isinstance(space, spaces.Box) and len(space.shape or ()) == 3
