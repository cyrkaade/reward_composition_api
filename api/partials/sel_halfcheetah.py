"""Candidate HalfCheetah-v5 priors for the environment/partial selection grid.

Ground truth (gymnasium 1.2.3 half_cheetah_v5.py):

    reward = 1.0 * x_velocity - 0.1 * sum(action**2)

Never terminates, fixed 1000-step episode. Measured per step under a random
policy: x_velocity -0.099, reward_ctrl -0.200.

Cap levels are set against the published PPO reference (rl-zoo HalfCheetah-v3 =
5819 over 1000 steps, about 5.8 m/s; stock SB3 at 1M steps runs well below that),
so 1.0 and 2.0 m/s should both bind. This environment is on the list with a known
risk attached: the archive records ~45% of HalfCheetah seeds learning to run and
the rest not, which makes its medians mode-counting. The 5-seed mode=true arm in
this grid is the measurement of whether that still holds at these settings.

Observation layout (17): qpos[1:] = [z, root_pitch, bthigh, bshin, bfoot, fthigh,
fshin, ffoot] then qvel[0:9]. So obs[0] is torso height and obs[1] is torso pitch.
"""

from __future__ import annotations

import numpy as np


class HalfCheetahPartial:
    component_keys = ("progress", "posture", "effort", "omitted_ctrl")

    def __init__(
        self,
        speed_cap: float | None = None,
        pitch_penalty: float = 0.0,
        ctrl_weight: float = 0.0,
    ):
        self.speed_cap = speed_cap
        self.pitch_penalty = float(pitch_penalty)
        self.ctrl_weight = float(ctrl_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        info = info or {}
        progress = float(info.get("x_velocity", 0.0))
        omitted_ctrl = float(info.get("reward_ctrl", 0.0))

        if self.speed_cap is not None:
            # Saturating: running faster than the cap earns nothing. Nonlinear, so
            # this is not the same lever as --partial-alpha.
            progress = min(progress, self.speed_cap)

        posture = 0.0
        if self.pitch_penalty:
            pitch = float(np.asarray(next_obs, dtype=np.float64)[1])
            posture = -self.pitch_penalty * abs(pitch)

        effort = self.ctrl_weight * float(np.square(np.asarray(action, dtype=np.float64)).sum())
        return {
            "partial": float(progress + posture - effort),
            "components": {
                "progress": float(progress),
                "posture": float(posture),
                "effort": float(-effort),
                "omitted_ctrl": omitted_ctrl,
            },
        }


_PARTIALS = {
    # 1. SATURATION, low: no credit above 1.0 m/s.
    "shc_cap10": dict(speed_cap=1.0),
    # 2. SATURATION, mid: the same knob at 2.0 m/s.
    "shc_cap20": dict(speed_cap=2.0),
    # 3. OMISSION: forward velocity, no effort cost at all. The mildest of the
    #    five - the omitted term is worth about 0.2/step against a target of
    #    several m/s - and it is here as the calibration point for the others.
    "shc_forward": dict(),
    # 4. OVER-PENALIZED EFFORT: effort at weight 1.0 against the true 0.1, so the
    #    prior prefers a slow shuffle to a gallop.
    "shc_timid": dict(ctrl_weight=1.0),
    # 5. OVER-CONSTRAINED POSTURE: forward velocity minus a torso-pitch penalty.
    #    A designer who thinks a level back means a healthy gait; the fastest
    #    HalfCheetah gaits pitch heavily.
    "shc_level": dict(pitch_penalty=2.0),
}


def _factory(kwargs):
    return lambda env_id: HalfCheetahPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="mujoco",
            factory=_factory(kwargs),
            description=f"HalfCheetah-v5 selection-grid prior '{name}'",
            env_ids=("HalfCheetah-v5",),
            component_keys=HalfCheetahPartial.component_keys,
        )
