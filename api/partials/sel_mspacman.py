"""Archived hand-written priors for the old ALE/MsPacman-v5 sel3 screen.

These oracle-score-based priors are retained only to reproduce sel3. They are
not eligible for the new non-timid screen, which uses ``atari_ram_screen.py``.
The current Atari pipeline gives policies/reward models stacked pixels and
passes synchronized RAM snapshots to partial functions.

WHAT A PRIOR CAN SEE HERE
-------------------------
At the time of sel3, the Atari suite exposed 128 RAM bytes as the policy
observation and these priors avoided a memory map. Every archived prior below
is therefore built from the three signals that were verified present then:

    true_reward                  the raw score delta for this step
    info["lives"]                3 at the start of MsPacman
    info["episode_frame_number"] frames elapsed this episode

MEASURED on the installed build, frameskip 4, sticky actions 0.25: a random
policy runs a median 449 steps and scores a median 190.

WHY THESE FIVE
--------------
The true reward is the raw score. The interesting failures on an Atari game are
about MAGNITUDE (a dot is 10, a ghost is 200) and about DYING, which the score
never mentions -- losing a life costs you the rest of the episode but shows up
in the reward as nothing at all. So the five span:

    omission        clipped score, no death term at all
    saturation      score credit stops accruing per step
    over-priced     death charged far above what it is worth
    axis / proxy    survival only, score never mentioned -- gameable by hiding
    over-specified  score, death, AND a per-step survival bonus that pays the
                    agent to stall

Every one of them clips the score event, which is itself an omission: it throws
away the difference between eating a dot and eating a ghost.
"""

from __future__ import annotations

import numpy as np


class MsPacmanPartial:
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

        # Clipping is itself an omission: a dot (10) and a ghost (200) become
        # the same event.
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
    # 1. OMISSION: clipped score only. Dying is never mentioned, so the prior
    #    has no opinion about walking into a ghost.
    "smp_score": dict(score_weight=1.0, score_cap=1.0),
    # 2. SATURATION: the same, but a step may earn at most 0.2, so a big score
    #    event is worth barely more than a small one.
    "smp_score_cap": dict(score_weight=1.0, score_cap=0.2),
    # 3. OVER-PRICED DEATH: score plus a 20-per-life penalty, roughly two
    #    hundred dots' worth, so the prior would rather stand still than risk
    #    a ghost.
    "smp_life_heavy": dict(score_weight=1.0, score_cap=1.0, life_penalty=20.0),
    # 4. AXIS / GAMEABLE: survival only. Score is never mentioned at all, so
    #    hiding in a corner for the whole episode is a perfect policy.
    "smp_survive": dict(score_weight=0.0, life_penalty=5.0, time_bonus=0.05),
    # 5. OVER-SPECIFIED: score, a sane death penalty, AND a per-step bonus for
    #    still being alive -- which quietly pays the agent to stall.
    "smp_score_life_time": dict(score_weight=1.0, score_cap=1.0, life_penalty=2.0, time_bonus=0.02),
}


def _factory(kwargs):
    return lambda env_id: MsPacmanPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="atari",
            factory=_factory(kwargs),
            description=f"ALE/MsPacman-v5 sel3 prior '{name}'",
            env_ids=("ALE/MsPacman-v5",),
            component_keys=MsPacmanPartial.component_keys,
        )
