from __future__ import annotations

import gymnasium as gym

from reward_composition_api.wrappers.lunar_lander import LunarLanderSaveInfo

from .gymnasium_env import GymnasiumEnvironmentProfile


class Box2DEnvironmentProfile(GymnasiumEnvironmentProfile):
    def make_raw_env(self, env_id: str):
        env = gym.make(env_id)
        if env_id.startswith("LunarLander"):
            return LunarLanderSaveInfo(env)
        return env
