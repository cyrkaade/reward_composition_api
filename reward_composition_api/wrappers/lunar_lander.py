from __future__ import annotations

from gymnasium.core import Wrapper


class LunarLanderSaveInfo(Wrapper):
    """Expose LunarLander terminal flags in ``info`` like the original code."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        unwrapped = self.env.unwrapped
        info["game_over"] = bool(getattr(unwrapped, "game_over", False))
        lander = getattr(unwrapped, "lander", None)
        info["awake"] = bool(getattr(lander, "awake", False)) if lander is not None else False
        return observation, reward, terminated, truncated, info
