"""Candidate HumanoidStandup-v5 priors for the environment/partial selection grid.

Ground truth (gymnasium 1.2.3 humanoidstandup_v5.py):

    uph        = torso_z / model.opt.timestep          (timestep 0.003, so z*333)
    quad_ctrl  = 0.1  * sum(ctrl**2)
    quad_impact= clip(5e-7 * sum(cfrc_ext**2), 0, 10)
    reward     = uph - quad_ctrl - quad_impact + 1

Never terminates, fixed 1000-step episode - the lowest-variance setting in this
grid. Measured per step under a random policy: reward_linup 32.1 (torso at about
0.096 m, i.e. lying down), reward_quadctrl -0.091, reward_impact -0.146.

That measurement rules out the usual "designer forgot the cost term" prior here:
both costs are under 0.5 against an objective worth 32 and rising, so deleting
them is a no-op. Everything below therefore attacks the objective itself - where
to stop, what height to aim for, or level versus rate. All of them work in
reward_linup units (multiply by 0.003 for metres) and drop the constant +1, which
is a pure offset on a fixed-horizon episode.

For scale: linup 60 is a torso at 0.18 m, 120 is 0.36 m, 150 is 0.45 m; a fully
standing humanoid is around 1.3 m, i.e. linup ~430.
"""

from __future__ import annotations

import numpy as np


class HumanoidStandupPartial:
    component_keys = ("height", "effort", "omitted_ctrl", "omitted_impact")

    def __init__(
        self,
        mode: str = "level",
        cap: float | None = None,
        target: float | None = None,
        ctrl_weight: float = 0.0,
    ):
        self.mode = str(mode)
        self.cap = cap
        self.target = target
        self.ctrl_weight = float(ctrl_weight)
        self.prev = None

    def reset(self, info: dict | None = None) -> None:
        self.prev = None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        info = info or {}
        linup = float(info.get("reward_linup", 0.0))
        omitted_ctrl = float(info.get("reward_quadctrl", 0.0))
        omitted_impact = float(info.get("reward_impact", 0.0))

        if self.mode == "level":
            height = linup
            if self.cap is not None:
                # Saturating: no credit for getting the torso higher than this.
                height = min(height, self.cap)
        elif self.mode == "target":
            # Over-specified: the designer picked a height and penalizes both
            # sides of it, so the prior actively opposes standing up fully.
            height = -abs(linup - float(self.target))
        elif self.mode == "rise":
            # Rate instead of level: potential-based on torso height, so the prior
            # pays for GETTING up rather than for BEING up. On a fixed 1000-step
            # episode that means lying still for 900 steps then standing scores
            # the same as standing throughout, which the true reward does not.
            height = 0.0 if self.prev is None else linup - self.prev
            self.prev = linup
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        if self.mode != "rise":
            self.prev = linup

        effort = self.ctrl_weight * float(np.square(np.asarray(action, dtype=np.float64)).sum())
        return {
            "partial": float(height - effort),
            "components": {
                "height": float(height),
                "effort": float(-effort),
                "omitted_ctrl": omitted_ctrl,
                "omitted_impact": omitted_impact,
            },
        }


_PARTIALS = {
    # 1. SATURATION, low: torso height credited only up to 0.18 m.
    "shs_cap60": dict(mode="level", cap=60.0),
    # 2. SATURATION, mid: the same knob at 0.36 m.
    "shs_cap120": dict(mode="level", cap=120.0),
    # 3. RATE NOT LEVEL: potential-based on torso height. Rewards the act of
    #    rising, not the state of being up.
    "shs_rise": dict(mode="rise"),
    # 4. OVER-SPECIFIED: aim for a 0.27 m crouch and penalize overshooting it.
    #    Deliberately below any plausible ceiling: a target the true arm cannot
    #    reach would make this prior an easier version of the task rather than a
    #    weaker one, which is how Hopper and Walker2d priors ended up beating the
    #    reward they were meant to approximate.
    "shs_crouch": dict(mode="target", target=90.0),
    # 5. OVER-PENALIZED EFFORT: height minus effort at weight 50.0 against the
    #    true 0.1, i.e. 500x. Measured, sum(ctrl**2) here is about 0.9, so the
    #    true cost is 0.09 per step against an objective worth 32 and rising -
    #    anything under about 10x is numerically invisible on this environment.
    "shs_timid": dict(mode="level", ctrl_weight=50.0),
}


def _factory(kwargs):
    return lambda env_id: HumanoidStandupPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="mujoco",
            factory=_factory(kwargs),
            description=f"HumanoidStandup-v5 selection-grid prior '{name}'",
            env_ids=("HumanoidStandup-v5",),
            component_keys=HumanoidStandupPartial.component_keys,
        )
