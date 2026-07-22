"""Hand-written LunarLander partials at target partiality ~0.25 / 0.50 / 0.75.

Each is a genuine reward function, NOT a scaling of the true reward:
potential-based approach shaping (distance / speed / tilt / leg), plus a
hand-written landing-vs-crash judgement applied at termination for the
higher levels. Measured partiality on random rollouts (fragment 25):
  lunarlander_p25 ~ 0.24, lunarlander_p50 ~ 0.53, lunarlander_p75 ~ 0.75.
"""

from __future__ import annotations

import numpy as np


class LunarLanderLevelPartial:
    def __init__(self, w_dist, w_speed, w_tilt, w_leg, term_bonus):
        self.w_dist = float(w_dist)
        self.w_speed = float(w_speed)
        self.w_tilt = float(w_tilt)
        self.w_leg = float(w_leg)
        self.term_bonus = float(term_bonus)
        self.prev = None

    def reset(self, info=None):
        self.prev = None

    def _potential(self, state):
        s = np.asarray(state, dtype=np.float64)
        x, y, vx, vy, ang = s[0], s[1], s[2], s[3], s[4]
        leg1, leg2 = s[6], s[7]
        return float(
            -100.0 * self.w_dist * np.sqrt(x * x + y * y)
            - 100.0 * self.w_speed * np.sqrt(vx * vx + vy * vy)
            - 100.0 * self.w_tilt * abs(ang)
            + 10.0 * self.w_leg * (leg1 + leg2)
        )

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        if self.prev is None and obs is not None:
            self.prev = self._potential(obs)
        current = self._potential(next_obs)
        shaping_delta = 0.0 if self.prev is None else current - self.prev
        self.prev = current

        partial = shaping_delta
        terminal = 0.0
        if terminated:
            n = np.asarray(next_obs, dtype=np.float64)
            landed = (
                n[6] > 0.5 and n[7] > 0.5
                and abs(n[2]) < 0.5 and abs(n[3]) < 0.5 and abs(n[4]) < 0.3
            )
            terminal = self.term_bonus if landed else -self.term_bonus
            partial += terminal

        return {
            "partial": float(partial),
            "components": {"shaping_delta": float(shaping_delta), "terminal_judgement": float(terminal)},
        }


_LEVELS = {
    "lunarlander_p25": dict(w_dist=1.0, w_speed=0.8, w_tilt=0.0, w_leg=0.0, term_bonus=0.0),
    "lunarlander_p50": dict(w_dist=1.0, w_speed=0.8, w_tilt=0.5, w_leg=0.5, term_bonus=30.0),
    "lunarlander_p75": dict(w_dist=1.0, w_speed=0.8, w_tilt=0.5, w_leg=0.5, term_bonus=200.0),
}


def _factory(kwargs):
    return lambda env_id: LunarLanderLevelPartial(**kwargs)


def register(registry):
    for name, kwargs in _LEVELS.items():
        registry.register(
            name=name,
            suite="box2d",
            factory=_factory(kwargs),
            description=f"Hand-written LunarLander partial, target partiality ~0.{name[-2:]}",
            env_ids=("LunarLander-v3",),
            component_keys=("shaping_delta", "terminal_judgement"),
        )
