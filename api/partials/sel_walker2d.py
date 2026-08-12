"""Candidate Walker2d-v5 priors for the environment/partial selection grid.

Ground truth (gymnasium 1.2.3 walker2d_v5.py):

    reward = 1.0 * x_velocity + 1.0 * healthy - 1e-3 * sum(action**2)

Terminates when the torso leaves the healthy z/angle box; measured random episode
length is 18 steps, so a random policy scores about 1.3.

THE TRAP THIS FILE AVOIDS. The archive's `walker2d_survive_forward` prior
(survive + 0.4*forward, control cost dropped) produced BETTER true-reward policies
than training on the true reward itself, because the control weight here is 1e-3 -
dropping it changes almost nothing, while removing the speed/stability tension
makes the task easier. A prior of the form "survive plus unbounded forward" is
therefore not incomplete on Walker2d, it is an improvement. Every candidate below
either saturates the forward term, replaces it with a target, or over-prices
something, so none of them is the true reward with a rounding error deleted.

Cap levels are set against the published PPO reference (rl-zoo Walker2d-v3 = 3478
over up to 1000 steps, so roughly 2.5 m/s once the survival bonus is subtracted).

Observation layout (17): qpos[1:] = [z, torso_angle, thigh, leg, foot, thigh_l,
leg_l, foot_l] then qvel[0:9]. obs[1] is the torso angle.
"""

from __future__ import annotations

import numpy as np


class Walker2dPartial:
    component_keys = ("progress", "survive", "posture", "effort", "omitted_ctrl")

    def __init__(
        self,
        forward: str = "capped",
        speed_cap: float | None = None,
        target_speed: float | None = None,
        angle_penalty: float = 0.0,
        ctrl_weight: float = 0.0,
    ):
        self.forward = str(forward)
        self.speed_cap = speed_cap
        self.target_speed = target_speed
        self.angle_penalty = float(angle_penalty)
        self.ctrl_weight = float(ctrl_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        info = info or {}
        vx = float(info.get("x_velocity", 0.0))
        survive = float(info.get("reward_survive", 0.0))
        omitted_ctrl = float(info.get("reward_ctrl", 0.0))

        if self.forward == "none":
            progress = 0.0
        elif self.forward == "capped":
            # Saturating: walking faster than the cap earns nothing.
            progress = vx if self.speed_cap is None else min(vx, self.speed_cap)
        elif self.forward == "target":
            # Over-specified: the designer picked a speed and penalizes both
            # sides of it, so the prior actively opposes going faster.
            progress = 1.0 - abs(vx - float(self.target_speed))
        else:
            raise ValueError(f"unknown forward mode: {self.forward}")

        posture = 0.0
        if self.angle_penalty:
            angle = float(np.asarray(next_obs, dtype=np.float64)[1])
            posture = -self.angle_penalty * abs(angle)

        effort = self.ctrl_weight * float(np.square(np.asarray(action, dtype=np.float64)).sum())
        return {
            "partial": float(progress + survive + posture - effort),
            "components": {
                "progress": float(progress),
                "survive": float(survive),
                "posture": float(posture),
                "effort": float(-effort),
                "omitted_ctrl": omitted_ctrl,
            },
        }


_PARTIALS = {
    # 1. SATURATION, low: stay up, and walk, but no credit above 0.5 m/s.
    "swk_cap05": dict(forward="capped", speed_cap=0.5),
    # 2. SATURATION, mid: the same knob at 1.5 m/s.
    "swk_cap15": dict(forward="capped", speed_cap=1.5),
    # 3. OMISSION of the objective: stay upright and level, full stop. Forward
    #    progress is never mentioned, so the prior is happy standing still.
    "swk_stand": dict(forward="none", angle_penalty=1.0),
    # 4. OVER-SPECIFIED: a hand-picked 1.0 m/s target speed, penalized on both
    #    sides. This is the one candidate whose optimum is bounded by design
    #    rather than by saturation.
    "swk_target": dict(forward="target", target_speed=1.0),
    # 5. OVER-PENALIZED EFFORT: effort at 0.2 against the true 1e-3, i.e. 200x,
    #    which buys a cautious shuffle instead of a stride.
    "swk_timid": dict(forward="capped", ctrl_weight=0.2),
}


def _factory(kwargs):
    return lambda env_id: Walker2dPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="mujoco",
            factory=_factory(kwargs),
            description=f"Walker2d-v5 selection-grid prior '{name}'",
            env_ids=("Walker2d-v5",),
            component_keys=Walker2dPartial.component_keys,
        )
