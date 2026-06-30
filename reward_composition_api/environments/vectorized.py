from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def normalize_obs(stats_source, observation):
    observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    if isinstance(stats_source, VecNormalize):
        return stats_source.normalize_obs(observation)
    return observation


def make_raw_eval_env(make_raw_env, env_id: str):
    return DummyVecEnv([lambda: Monitor(make_raw_env(env_id))])


def load_vecnormalize_eval_env(env_id: str, stats_path: Path, make_raw_eval_env_fn) -> VecNormalize:
    env = VecNormalize.load(stats_path, make_raw_eval_env_fn(env_id))
    env.training = False
    env.norm_reward = False
    return env
