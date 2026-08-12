"""Candidate Swimmer-v5 priors for the environment/partial selection grid.

Ground truth (gymnasium 1.2.3 swimmer_v5.py):

    reward = 1.0 * x_velocity - 1e-4 * sum(action**2)

The control weight is 1e-4, so "forgot the effort cost" is NOT an incomplete
prior here - it is the true reward to four decimal places. Swimmer's prior has to
be wrong about the objective itself, so these are built around the direction of
travel, a saturating speed target, and a mis-set effort weight.

Cap levels are set against the published PPO reference for this task (rl-zoo
Swimmer-v3 = 281.6 over a fixed 1000-step episode, i.e. about 0.28 m/s), so 0.10
and 0.18 should both bind well below the ceiling and 0.18 sits roughly two thirds
of the way up. info exposes x_velocity and y_velocity directly.
"""

from __future__ import annotations

import numpy as np


class SwimmerPartial:
    component_keys = ("progress", "effort", "omitted_ctrl")

    def __init__(
        self,
        mode: str = "forward",
        speed_cap: float | None = None,
        lateral_penalty: float = 0.0,
        ctrl_weight: float = 0.0,
    ):
        self.mode = str(mode)
        self.speed_cap = speed_cap
        self.lateral_penalty = float(lateral_penalty)
        self.ctrl_weight = float(ctrl_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        info = info or {}
        vx = float(info.get("x_velocity", 0.0))
        vy = float(info.get("y_velocity", 0.0))
        omitted_ctrl = float(info.get("reward_ctrl", 0.0))

        if self.mode == "forward":
            progress = vx
        elif self.mode == "speed":
            # Direction-blind: any motion counts, including straight backwards.
            progress = float(np.sqrt(vx * vx + vy * vy))
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        if self.speed_cap is not None:
            # Saturating: swimming faster than the cap earns nothing. Negative
            # velocities still pass through, so backing up is still discouraged.
            progress = min(progress, self.speed_cap)

        progress -= self.lateral_penalty * abs(vy)
        effort = self.ctrl_weight * float(np.square(np.asarray(action, dtype=np.float64)).sum())
        return {
            "partial": float(progress - effort),
            "components": {
                "progress": float(progress),
                "effort": float(-effort),
                "omitted_ctrl": omitted_ctrl,
            },
        }


_PARTIALS = {
    # 1. SATURATION, low: no credit for swimming faster than 0.10 m/s.
    "sswm_cap10": dict(mode="forward", speed_cap=0.10),
    # 2. SATURATION, mid: the same knob at 0.18 m/s.
    "sswm_cap18": dict(mode="forward", speed_cap=0.18),
    # 3. DIRECTION-BLIND: rewards speed, not progress. A policy that thrashes
    #    sideways or swims backwards scores exactly as well as one that swims.
    "sswm_speed": dict(mode="speed"),
    # 4. OVER-CONSTRAINED: forward progress minus a penalty on lateral velocity.
    #    A designer who wants a tidy straight line fights the side-to-side
    #    undulation the body actually needs to generate thrust.
    "sswm_straight": dict(mode="forward", lateral_penalty=0.5),
    # 5. OVER-PENALIZED EFFORT: effort charged at 0.05 against the true 1e-4,
    #    i.e. 500x. The designer priced a torque budget the task does not have.
    #    Weight picked by measurement, not taste: a swimming policy runs near
    #    bang-bang, so sum(a**2) is order 1-2 and the bill lands at 0.05-0.10 per
    #    step against a ceiling pace of 0.28, i.e. a real handicap. At the 0.02
    #    first tried it was under 5% and the prior was the true reward again.
    "sswm_timid": dict(mode="forward", ctrl_weight=0.05),
}


def _factory(kwargs):
    return lambda env_id: SwimmerPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="mujoco",
            factory=_factory(kwargs),
            description=f"Swimmer-v5 selection-grid prior '{name}'",
            env_ids=("Swimmer-v5",),
            component_keys=SwimmerPartial.component_keys,
        )
