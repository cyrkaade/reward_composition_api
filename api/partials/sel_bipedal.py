"""Five hand-written priors for BipedalWalker-v3, for the sel3 screen.

BipedalWalker's step() returns an EMPTY info dict -- verified on installed
gymnasium 1.2.3 -- so unlike every MuJoCo cell there are no reward-component
keys to read. Every prior here is computed from the OBSERVATION instead.

Observation layout, read from the installed source (`bipedal_walker.py`, the
`state = [...]` block):

    obs[0]   hull angle
    obs[1]   hull angular velocity (2.0 * angularVelocity / FPS)
    obs[2]   forward velocity      (0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS)
    obs[3]   vertical velocity
    obs[4:8] joint 0/1 angle and speed
    obs[8]   leg 1 ground contact (0/1)
    obs[9:13] joint 2/3 angle and speed
    obs[13]  leg 2 ground contact (0/1)
    obs[14:] 10 lidar readings

The TRUE reward, also read from source:

    shaping = 130 * pos.x / SCALE - 5.0 * abs(hull.angle)
    reward  = shaping - prev_shaping
              - 0.00035 * MOTORS_TORQUE * sum(clip(|a_i|, 0, 1))
    reward  = -100                            if the hull touches the ground

so there are exactly three things a prior can be wrong about: forward progress,
the upright/angle term, and the torque cost -- plus the -100 fall penalty, which
is the term most worth omitting because it is the one that makes the task hard.

The five are five different KINDS of wrong, matching the taxonomy the other
sel_* families use, not five strengths of one thing:

    omission        a whole true-reward term deleted
    saturation      credit stops accruing past a cap
    axis / proxy    a surrogate objective, or one axis of the real one
    over-priced     a real cost term charged far above its true weight
    over-specified  a target the designer picked, penalized on both sides

NOTE ON SCALE: the training env is wrapped in VecNormalize(norm_reward=True), and
in mode=partial a uniform rescale of the prior is normalized straight back out,
so the absolute magnitudes below do not matter. Only the SHAPE does.
"""

from __future__ import annotations

import numpy as np


class BipedalPartial:
    component_keys = ("progress", "upright", "effort", "contact")

    def __init__(
        self,
        forward: str = "linear",
        speed_cap: float | None = None,
        target_speed: float = 0.30,
        angle_weight: float = 0.0,
        effort_weight: float = 0.0,
        speed_scale: float = 10.0,
    ):
        self.forward = forward
        self.speed_cap = speed_cap
        self.target_speed = float(target_speed)
        self.angle_weight = float(angle_weight)
        self.effort_weight = float(effort_weight)
        self.speed_scale = float(speed_scale)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        state = np.asarray(next_obs, dtype=np.float64).reshape(-1)
        angle = float(state[0])
        # obs[2] is already velocity-scaled by the env; speed_scale only lifts it
        # into the same rough per-step range as the shaping delta it stands in for.
        velocity = float(state[2]) * self.speed_scale
        contact = float(state[8]) + float(state[13])

        if self.forward == "none":
            progress = 0.0
        elif self.forward == "capped":
            # Saturating: walking faster than the cap earns nothing extra.
            progress = velocity if self.speed_cap is None else min(velocity, self.speed_cap)
        elif self.forward == "target":
            # Over-specified: the designer picked a speed and penalizes BOTH
            # sides of it, so the prior actively opposes going faster.
            progress = 1.0 - abs(velocity - self.target_speed)
        else:
            progress = velocity

        upright = -self.angle_weight * abs(angle)
        effort = -self.effort_weight * float(np.sum(np.clip(np.abs(np.asarray(action, dtype=np.float64)), 0.0, 1.0)))

        return {
            # No prior here reproduces the -100 fall penalty. That omission is
            # deliberate and shared by all five: it is the term that makes the
            # task hard, and leaving it out is the most realistic way a
            # hand-written reward goes wrong on this environment.
            "partial": progress + upright + effort,
            "components": {
                "progress": progress,
                "upright": upright,
                "effort": effort,
                "contact": contact,
            },
        }


_PARTIALS = {
    # 1. OMISSION: forward velocity, nothing else. No upright term, no torque
    #    cost, no fall penalty. The straightforward "I described the task and
    #    forgot everything it costs" prior.
    "sbw_speed": dict(forward="linear"),
    # 2. SATURATION: the same thing, but no credit for going faster than 0.3.
    "sbw_speed_cap": dict(forward="capped", speed_cap=0.30),
    # 3. AXIS: stay upright, full stop. Forward progress is never mentioned, so
    #    the prior is perfectly happy standing still forever.
    "sbw_upright": dict(forward="none", angle_weight=1.0),
    # 4. OVER-PRICED EFFORT: forward velocity minus a torque cost charged at
    #    0.05 per motor against a true weight of 0.00035 * MOTORS_TORQUE, i.e.
    #    far above its real price, so the prior prefers a slow shuffle.
    "sbw_speed_effort": dict(forward="linear", effort_weight=0.05),
    # 5. OVER-SPECIFIED: a hand-picked 0.3 target speed penalized on both sides,
    #    plus an upright term. Bounded by design rather than by saturation.
    "sbw_target": dict(forward="target", target_speed=0.30, angle_weight=0.5),
}


def _factory(kwargs):
    return lambda env_id: BipedalPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="box2d",
            factory=_factory(kwargs),
            description=f"BipedalWalker-v3 sel3 prior '{name}'",
            env_ids=("BipedalWalker-v3",),
            component_keys=BipedalPartial.component_keys,
        )
