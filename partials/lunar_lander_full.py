"""Full-form LunarLander reward partial.

This mirrors Gymnasium's LunarLander shaping reward with all component weights
set to 1.0, so using it as a partial is effectively using the environment's
ideal closed-form reward.
"""

from __future__ import annotations

import numpy as np

from reward_composition_api.registry import PartialRewardStep


class LunarLanderFullPartial:
    def __init__(
        self,
        *,
        continuous: bool = False,
        distance: float = 1.0,
        speed: float = 1.0,
        tilt: float = 1.0,
        leg: float = 1.0,
        side_engine: float = 1.0,
        main_engine: float = 1.0,
        game_over: float = 1.0,
        landed: float = 1.0,
    ):
        self.continuous = bool(continuous)
        self.weights = {
            "distance": float(distance),
            "speed": float(speed),
            "tilt": float(tilt),
            "leg": float(leg),
            "side_engine": float(side_engine),
            "main_engine": float(main_engine),
            "game_over": float(game_over),
            "landed": float(landed),
        }
        self.prev_shaping: float | None = None

    def reset(self, info: dict | None = None) -> None:
        self.prev_shaping = None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        if self.prev_shaping is None and obs is not None:
            self.prev_shaping = self._shaping(obs)

        shaping = self._shaping(next_obs)
        shaping_delta = 0.0 if self.prev_shaping is None else shaping - self.prev_shaping
        self.prev_shaping = shaping

        main_engine_penalty = -self._main_power(action) * 0.3 * self.weights["main_engine"]
        side_engine_penalty = -self._side_power(action) * 0.03 * self.weights["side_engine"]
        reward = shaping_delta + main_engine_penalty + side_engine_penalty
        terminal_reward = 0.0

        info = info or {}
        has_lunar_flags = "game_over" in info or "awake" in info
        game_over = bool(info.get("game_over", False))
        awake = bool(info.get("awake", True))
        if game_over or abs(np.asarray(next_obs, dtype=np.float64)[0]) >= 1.0:
            terminal_reward = -100.0 * self.weights["game_over"]
            reward = terminal_reward
        if not awake:
            terminal_reward = 100.0 * self.weights["landed"]
            reward = terminal_reward
        if terminated and not has_lunar_flags:
            terminal_reward = (
                100.0 * self.weights["landed"]
                if float(true_reward) >= 0.0
                else -100.0 * self.weights["game_over"]
            )
            reward = terminal_reward

        return PartialRewardStep(
            partial=float(reward),
            components={
                "shaping_delta": float(shaping_delta),
                "main_engine_penalty": float(main_engine_penalty),
                "side_engine_penalty": float(side_engine_penalty),
                "terminal_reward": float(terminal_reward),
            },
        )

    def _shaping(self, state) -> float:
        state = np.asarray(state, dtype=np.float64)
        return float(
            -100.0 * self.weights["distance"] * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100.0 * self.weights["speed"] * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100.0 * self.weights["tilt"] * abs(state[4])
            + 10.0 * self.weights["leg"] * state[6]
            + 10.0 * self.weights["leg"] * state[7]
        )

    def _main_power(self, action) -> float:
        if self.continuous:
            action = np.asarray(action, dtype=np.float64)
            if action[0] > 0.0:
                return float((np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5)
            return 0.0

        discrete_action = int(np.asarray(action).reshape(-1)[0])
        return 1.0 if discrete_action == 2 else 0.0

    def _side_power(self, action) -> float:
        if self.continuous:
            action = np.asarray(action, dtype=np.float64)
            if np.abs(action[1]) > 0.5:
                return float(np.clip(np.abs(action[1]), 0.5, 1.0))
            return 0.0

        discrete_action = int(np.asarray(action).reshape(-1)[0])
        return 1.0 if discrete_action in (1, 3) else 0.0


def make_reward_function(name, continuous, **weights):
    return LunarLanderFullPartial(continuous=continuous, **weights)


def register(registry) -> None:
    registry.register(
        name="lunar_lander_full",
        suite="box2d",
        factory=lambda env_id: LunarLanderFullPartial(continuous="Continuous" in env_id),
        description="Full Gymnasium LunarLander reward with all shaping weights set to 1.0.",
        env_ids=("LunarLander-v3", "LunarLanderContinuous-v3"),
        component_keys=("shaping_delta", "main_engine_penalty", "side_engine_penalty", "terminal_reward"),
    )
