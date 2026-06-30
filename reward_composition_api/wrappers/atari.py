from __future__ import annotations

import gymnasium as gym


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
