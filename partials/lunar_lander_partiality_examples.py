"""Small LunarLander partiality examples.

These are intentionally hand-written reward formulas, not scaled copies of
the environment reward.
"""

from __future__ import annotations

import numpy as np

from reward_composition_api.registry import PartialRewardStep
from partials.lunar_lander_full import LunarLanderFullPartial


class LunarLanderDistanceTiltPartial:
    component_keys = ("distance_delta", "tilt_delta", "leg_delta")

    def __init__(self):
        self.prev_shaping: float | None = None

    def reset(self, info: dict | None = None) -> None:
        self.prev_shaping = None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        if self.prev_shaping is None and obs is not None:
            self.prev_shaping = self._shaping(obs)
        shaping = self._shaping(next_obs)
        shaping_delta = 0.0 if self.prev_shaping is None else shaping - self.prev_shaping
        self.prev_shaping = shaping
        return PartialRewardStep(
            partial=float(shaping_delta),
            components={
                "distance_delta": float(shaping_delta),
                "tilt_delta": 0.0,
                "leg_delta": 0.0,
            },
        )

    def _shaping(self, state) -> float:
        state = np.asarray(state, dtype=np.float64)
        distance = -100.0 * np.sqrt(state[0] * state[0] + state[1] * state[1])
        tilt = -100.0 * abs(state[4])
        legs = 10.0 * state[6] + 10.0 * state[7]
        return float(distance + tilt + legs)


class LunarLanderLowFuelSignalPartial:
    component_keys = ("idle_bonus", "engine_penalty")

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        discrete_action = int(np.asarray(action).reshape(-1)[0])
        if discrete_action == 0:
            reward = 0.2
            idle_bonus = 0.2
            engine_penalty = 0.0
        elif discrete_action == 2:
            reward = -0.3
            idle_bonus = 0.0
            engine_penalty = -0.3
        else:
            reward = -0.03
            idle_bonus = 0.0
            engine_penalty = -0.03
        return PartialRewardStep(
            partial=float(reward),
            components={
                "idle_bonus": float(idle_bonus),
                "engine_penalty": float(engine_penalty),
            },
        )


def register(registry) -> None:
    registry.register(
        name="lunar_lander_example_full",
        suite="box2d",
        factory=lambda env_id: LunarLanderFullPartial(continuous=False),
        description="Full LunarLander reward formula.",
        env_ids=("LunarLander-v3",),
        component_keys=("shaping_delta", "main_engine_penalty", "side_engine_penalty", "terminal_reward"),
    )
    registry.register(
        name="lunar_lander_example_medium",
        suite="box2d",
        factory=lambda env_id: LunarLanderDistanceTiltPartial(),
        description="Medium LunarLander partial: distance, tilt, and leg contact shaping only.",
        env_ids=("LunarLander-v3",),
        component_keys=LunarLanderDistanceTiltPartial.component_keys,
    )
    registry.register(
        name="lunar_lander_example_low",
        suite="box2d",
        factory=lambda env_id: LunarLanderLowFuelSignalPartial(),
        description="Low-information LunarLander partial: only weak fuel/action signal.",
        env_ids=("LunarLander-v3",),
        component_keys=LunarLanderLowFuelSignalPartial.component_keys,
    )
