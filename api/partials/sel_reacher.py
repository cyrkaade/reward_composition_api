"""Candidate Reacher-v5 priors for the environment/partial selection grid.

Ground truth (gymnasium 1.2.3 reacher_v5.py):

    reward_dist = -1.0 * ||fingertip - target||     (3-D norm)
    reward_ctrl = -1.0 * sum(action**2)
    reward      = reward_dist + reward_ctrl

Both weights are 1.0, and over 20 random episodes the control term is the LARGER
of the two per step (mean -0.644 against -0.194 for distance). Reacher is
therefore the one environment here where "the designer forgot the effort cost" is
a big omission rather than a rounding error.

Observation layout (10): [cos t1, cos t2, sin t1, sin t2, target_x, target_y,
qvel1, qvel2, (fingertip - target)_x, (fingertip - target)_y].
"""

from __future__ import annotations

import numpy as np


class ReacherPartial:
    component_keys = ("reach", "effort", "omitted_ctrl")

    def __init__(self, shape: str = "linear", dist_floor: float | None = None, ctrl_weight: float = 0.0):
        self.shape = str(shape)
        self.dist_floor = dist_floor
        self.ctrl_weight = float(ctrl_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        info = info or {}
        reward_dist = float(info.get("reward_dist", 0.0))
        omitted_ctrl = float(info.get("reward_ctrl", 0.0))
        distance = -reward_dist

        if self.shape == "linear":
            reach = reward_dist
        elif self.shape == "quadratic":
            # Same argmin as the true reward but a vanishing gradient near it: the
            # arm gets into the neighbourhood and stops refining.
            reach = -distance * distance
        elif self.shape == "x_only":
            # Blind to the y axis of the error.
            reach = -abs(float(np.asarray(next_obs, dtype=np.float64)[8]))
        elif self.shape == "hit":
            # A hand-written success signal instead of continuous shaping.
            reach = 1.0 if distance < 0.05 else 0.0
        else:
            raise ValueError(f"unknown shape: {self.shape}")

        if self.dist_floor is not None:
            # Saturating: once the fingertip is this close, getting closer buys
            # nothing. Nonlinear, so NOT equivalent to scaling the prior.
            reach = max(reach, -self.dist_floor)

        effort = self.ctrl_weight * float(np.square(np.asarray(action, dtype=np.float64)).sum())
        return {
            "partial": float(reach - effort),
            "components": {
                "reach": float(reach),
                "effort": float(-effort),
                "omitted_ctrl": omitted_ctrl,
            },
        }


_PARTIALS = {
    # 1. OMISSION: distance only. Drops a cost term that is bigger per step than
    #    the objective the designer kept.
    "srch_dist": dict(shape="linear"),
    # 2. SATURATION: distance, but credit stops accruing inside 5 cm.
    "srch_dist_cap": dict(shape="linear", dist_floor=0.05),
    # 3. AXIS: only the x component of the reach error, effort cost kept intact.
    "srch_x": dict(shape="x_only", ctrl_weight=1.0),
    # 4. SHAPE: squared distance. Correct optimum, no gradient near it.
    "srch_quad": dict(shape="quadratic"),
    # 5. SPARSE + OVER-PENALIZED: a binary "close enough" bonus with half the true
    #    effort cost. The designer wrote a success criterion, not a gradient.
    "srch_hit": dict(shape="hit", ctrl_weight=0.5),
}


def _factory(kwargs):
    return lambda env_id: ReacherPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="mujoco",
            factory=_factory(kwargs),
            description=f"Reacher-v5 selection-grid prior '{name}'",
            env_ids=("Reacher-v5",),
            component_keys=ReacherPartial.component_keys,
        )
