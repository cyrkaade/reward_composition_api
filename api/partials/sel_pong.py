"""Hand-written priors for ALE/Pong-v5.

WHY PONG NEEDS A PRIOR MORE THAN ANY OTHER CELL HERE
----------------------------------------------------
Measured on the installed build (ale-py 0.10.1, obs_type="ram", frameskip 4,
sticky actions 0.25), a random policy: scores -21, wins 0 points, concedes 21,
over a median 910 steps. So only 21 of 910 steps carry any reward at all --
a reward density of 2.3% -- and the sign is negative on every single one.

That is the worst case for preference learning. A reward model trained on
fragment comparisons has to find the 2% of steps that mattered, and at the start
of training NO fragment contains a won point, so most comparisons are between
two fragments that are equally bad. This is the cell where a hand-written prior
should help the most, which is the point of including it.

Pong reports lives = 0 (verified), so the life-based construction used for
MsPacman and Qbert does not apply. The dense signal available instead is RALLY
LENGTH: steps since the last point was scored by either side. A random policy
concedes every ~43 steps; a competent one keeps the ball in play far longer, so
rally length is a dense, monotone proxy for "am I tracking the ball" that needs
no RAM addresses (which are undocumented and would break silently).

The rally bonus also penalizes conceding INDIRECTLY and correctly: the episode
ends when either side reaches 21, so conceding shortens the episode and costs
the agent all the rally bonus it would have collected. The prior never has to
state "conceding is bad" -- which is exactly the half of the true reward it
omits, and the half the reward model has to supply.
"""

from __future__ import annotations

import numpy as np


class PongPartial:
    component_keys = ("score_event", "concede_penalty", "rally_bonus", "rally_length")

    def __init__(
        self,
        score_weight: float = 1.0,
        concede_weight: float = 0.0,
        rally_bonus: float = 0.02,
    ):
        self.score_weight = float(score_weight)
        self.concede_weight = float(concede_weight)
        self.rally_bonus = float(rally_bonus)
        self.rally = 0

    def reset(self, info: dict | None = None) -> None:
        self.rally = 0

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        reward = float(true_reward)
        scored = 1.0 if reward > 0 else 0.0
        conceded = 1.0 if reward < 0 else 0.0

        # A point by either side ends the rally.
        if scored or conceded:
            rally_length = float(self.rally)
            self.rally = 0
        else:
            self.rally += 1
            rally_length = float(self.rally)

        score_event = self.score_weight * scored
        # concede_weight 0.0 in the arm this grid uses: the prior simply never
        # mentions that losing a point is bad. That omission is the whole point.
        concede_penalty = -self.concede_weight * conceded
        bonus = self.rally_bonus * (0.0 if (scored or conceded) else 1.0)

        return {
            "partial": score_event + concede_penalty + bonus,
            "components": {
                "score_event": score_event,
                "concede_penalty": concede_penalty,
                "rally_bonus": bonus,
                "rally_length": rally_length,
            },
        }


_PARTIALS = {
    # THE ONE THE COMPOSITION GRID USES.
    # Dense rally bonus plus credit for winning a point, and no penalty at all
    # for conceding. Turns a 2.3%-density reward into a signal on every step,
    # while omitting exactly half of what the true reward says.
    "spg_rally": dict(score_weight=1.0, concede_weight=0.0, rally_bonus=0.02),
    # OMISSION, sparse version: winning a point counts, conceding is free, and
    # there is no dense term. Kept as the control that isolates how much of
    # spg_rally's effect is the density rather than the omission.
    "spg_score": dict(score_weight=1.0, concede_weight=0.0, rally_bonus=0.0),
    # AXIS / GAMEABLE: rally length only. Winning is never mentioned, so a
    # policy that returns the ball forever without ever scoring is perfect.
    "spg_rally_only": dict(score_weight=0.0, concede_weight=0.0, rally_bonus=0.02),
}


def _factory(kwargs):
    return lambda env_id: PongPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="atari",
            factory=_factory(kwargs),
            description=f"ALE/Pong-v5 prior '{name}'",
            env_ids=("ALE/Pong-v5",),
            component_keys=PongPartial.component_keys,
        )
