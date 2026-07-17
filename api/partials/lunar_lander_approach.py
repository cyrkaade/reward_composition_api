"""Incomplete LunarLander reward partial for composition experiments.

Dense shaping for approaching the pad, slowing down, staying upright, and
leg contact. Intentionally omits the engine penalties and the terminal
landed/crashed bonuses from the full reward.
"""

from __future__ import annotations

import numpy as np


class LunarLanderApproachPartial:
    def __init__(self, *, distance: float = 1.0, speed: float = 0.8, tilt: float = 0.5, leg: float = 0.5):
        self.weights = {"distance": float(distance), "speed": float(speed), "tilt": float(tilt), "leg": float(leg)}
        self.prev_shaping: float | None = None

    def reset(self, info: dict | None = None) -> None:
        self.prev_shaping = None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        if self.prev_shaping is None and obs is not None:
            self.prev_shaping = self._shaping(obs)

        shaping = self._shaping(next_obs)
        shaping_delta = 0.0 if self.prev_shaping is None else shaping - self.prev_shaping
        self.prev_shaping = shaping

        return {
            "partial": float(shaping_delta),
            "components": {"approach_shaping_delta": float(shaping_delta)},
        }

    def _shaping(self, state) -> float:
        state = np.asarray(state, dtype=np.float64)
        return float(
            -100.0 * self.weights["distance"] * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100.0 * self.weights["speed"] * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100.0 * self.weights["tilt"] * abs(state[4])
            + 10.0 * self.weights["leg"] * state[6]
            + 10.0 * self.weights["leg"] * state[7]
        )


def register(registry) -> None:
    registry.register(
        name="lunar_lander_approach",
        suite="box2d",
        factory=lambda env_id: LunarLanderApproachPartial(),
        description=(
            "Incomplete LunarLander shaping partial: approach, speed, tilt, "
            "and leg contact only; no engine or terminal terms."
        ),
        env_ids=("LunarLander-v3", "LunarLanderContinuous-v3"),
        component_keys=("approach_shaping_delta",),
    )
