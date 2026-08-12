"""Candidate Ant-v5 priors for the environment/partial selection grid.

Ground truth (gymnasium 1.2.3 ant_v5.py):

    reward = 1.0 * x_velocity + 1.0 * healthy - 0.5 * sum(action**2) - contact_cost

Terminates when the torso leaves the healthy z range. Measured per step under a
random policy: x_velocity -0.008, reward_survive 0.995, reward_ctrl -1.340,
reward_contact -0.001; median random episode 76 steps.

Unlike Walker2d, Ant's control weight is 0.5 and its random-policy control bill
is the single largest per-step term, so "dropped the effort cost" is a genuine
omission here rather than a rounding error. The forward-only direction is the
other lever: the true reward pays for +x displacement and nothing else, so a
prior that credits motion in any direction is wrong in an interesting way.
"""

from __future__ import annotations

import numpy as np


class AntPartial:
    component_keys = ("progress", "survive", "effort", "omitted_ctrl")

    def __init__(
        self,
        mode: str = "forward",
        speed_cap: float | None = None,
        ctrl_weight: float = 0.0,
    ):
        self.mode = str(mode)
        self.speed_cap = speed_cap
        self.ctrl_weight = float(ctrl_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        info = info or {}
        vx = float(info.get("x_velocity", 0.0))
        vy = float(info.get("y_velocity", 0.0))
        x = float(info.get("x_position", 0.0))
        y = float(info.get("y_position", 0.0))
        survive = float(info.get("reward_survive", 0.0))
        omitted_ctrl = float(info.get("reward_ctrl", 0.0))

        if self.mode == "none":
            progress = 0.0
        elif self.mode == "forward":
            progress = vx
        elif self.mode == "radial":
            # Direction-blind: rate of increase of the distance from the origin,
            # so walking sideways or backwards pays exactly as well as walking
            # the way the true reward measures. Same units as x_velocity.
            radius = float(np.sqrt(x * x + y * y))
            progress = (x * vx + y * vy) / radius if radius > 1e-6 else 0.0
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        if self.speed_cap is not None:
            progress = min(progress, self.speed_cap)

        effort = self.ctrl_weight * float(np.square(np.asarray(action, dtype=np.float64)).sum())
        return {
            "partial": float(progress + survive - effort),
            "components": {
                "progress": float(progress),
                "survive": float(survive),
                "effort": float(-effort),
                "omitted_ctrl": omitted_ctrl,
            },
        }


_PARTIALS = {
    # 1. SATURATION, low: stay healthy, and walk, but no credit above 0.3 m/s.
    "sant_cap03": dict(mode="forward", speed_cap=0.3),
    # 2. SATURATION, mid: the same knob at 1.0 m/s.
    "sant_cap10": dict(mode="forward", speed_cap=1.0),
    # 3. DIRECTION-BLIND: healthy plus radial speed. Any direction of travel
    #    counts; the true reward pays only for +x.
    "sant_radial": dict(mode="radial"),
    # 4. OMISSION of the objective: stay healthy, full stop. No forward term at
    #    all, so the prior is satisfied by an ant that stands there.
    "sant_stand": dict(mode="none"),
    # 5. OVER-PENALIZED EFFORT: effort at 1.5 against the true 0.5, so the prior
    #    prefers a careful shuffle to a stride. Only 3x, because Ant's true weight
    #    is already the largest per-step term: measured under a random policy the
    #    control bill is 1.34 against a survival bonus of 1.0. At the 5.0 first
    #    tried the effort term was 13x survival and the prior's optimum was an ant
    #    that sits still, which sant_stand already covers.
    "sant_timid": dict(mode="forward", ctrl_weight=1.5),
}


def _factory(kwargs):
    return lambda env_id: AntPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="mujoco",
            factory=_factory(kwargs),
            description=f"Ant-v5 selection-grid prior '{name}'",
            env_ids=("Ant-v5",),
            component_keys=AntPartial.component_keys,
        )
