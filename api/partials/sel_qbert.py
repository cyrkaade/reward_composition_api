"""Archived hand-written priors for the old ALE/Qbert-v5 sel3 screen.

Same historical construction as sel_mspacman.py. These are retained only for
sel3 reproducibility and are excluded from the new screen, whose candidates in
``atari_ram_screen.py`` read synchronized RAM while PPO and the learned reward
model read pixels.

The parameters differ from MsPacman's because the games do:

    Qbert      a cube is 25 points, 4 lives, random policy scores a median 200
               over a median 312 steps
    MsPacman   a dot is 10, a ghost 200+, 3 lives, random median 190 over 449

MEASURED on installed ale-py 0.10.1, frameskip 4, sticky actions 0.25.

Qbert's death dynamic is the interesting one: falling off the pyramid ends a
life immediately and the score says nothing about it, so the gap between "score
only" and "score plus death" is wider here than on MsPacman. The life penalties
below are scaled to Qbert's 4 lives rather than MsPacman's 3.

The five kinds of wrong are the same taxonomy the other sel_* families use:
omission, saturation, over-priced, axis/proxy, over-specified.
"""

from __future__ import annotations

import numpy as np


class QbertPartial:
    component_keys = ("score_event", "life_penalty", "time_bonus", "lost_lives")

    def __init__(
        self,
        score_weight: float = 1.0,
        score_cap: float = 1.0,
        life_penalty: float = 0.0,
        time_bonus: float = 0.0,
    ):
        self.score_weight = float(score_weight)
        self.score_cap = float(score_cap)
        self.life_penalty = float(life_penalty)
        self.time_bonus = float(time_bonus)
        self.previous_lives: int | None = None

    def reset(self, info: dict | None = None) -> None:
        self.previous_lives = self._lives(info)

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        lives = self._lives(info)
        lost = 0
        if lives is not None and self.previous_lives is not None:
            lost = max(self.previous_lives - lives, 0)
        self.previous_lives = lives

        # Clipping throws away the difference between a single cube (25) and a
        # completed level bonus, which is the omission every arm here shares.
        score_event = self.score_weight * float(np.clip(float(true_reward), 0.0, self.score_cap))
        life_penalty = -self.life_penalty * float(lost)
        time_bonus = self.time_bonus

        return {
            "partial": score_event + life_penalty + time_bonus,
            "components": {
                "score_event": score_event,
                "life_penalty": life_penalty,
                "time_bonus": time_bonus,
                "lost_lives": float(lost),
            },
        }

    def _lives(self, info: dict | None) -> int | None:
        if not info or "lives" not in info:
            return None
        return int(info["lives"])


_PARTIALS = {
    # 1. OMISSION: clipped score only. Falling off the pyramid is free.
    "sqb_score": dict(score_weight=1.0, score_cap=1.0),
    # 2. SATURATION: at most 0.2 a step, so clearing a cube is worth barely
    #    more than brushing one.
    "sqb_score_cap": dict(score_weight=1.0, score_cap=0.2),
    # 3. OVER-PRICED DEATH: 25 per life against 4 lives, so the prior would
    #    rather sit on a safe cube than finish the level.
    "sqb_life_heavy": dict(score_weight=1.0, score_cap=1.0, life_penalty=25.0),
    # 4. AXIS / GAMEABLE: survival only, score never mentioned -- sitting still
    #    on the top cube is a perfect policy under this prior.
    "sqb_survive": dict(score_weight=0.0, life_penalty=5.0, time_bonus=0.05),
    # 5. OVER-SPECIFIED: score, a sane death penalty, and a per-step alive bonus
    #    that quietly pays the agent to stall rather than clear cubes.
    "sqb_score_life_time": dict(score_weight=1.0, score_cap=1.0, life_penalty=3.0, time_bonus=0.02),
}


def _factory(kwargs):
    return lambda env_id: QbertPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="atari",
            factory=_factory(kwargs),
            description=f"ALE/Qbert-v5 sel3 prior '{name}'",
            env_ids=("ALE/Qbert-v5",),
            component_keys=QbertPartial.component_keys,
        )
