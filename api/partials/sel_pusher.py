"""Candidate Pusher-v5 priors for the environment/partial selection grid.

Ground truth (gymnasium 1.2.3 pusher_v5.py):

    reward_dist = -1.0 * ||object - goal||
    reward_near = -0.5 * ||object - fingertip||
    reward_ctrl = -0.1 * sum(action**2)
    reward      = reward_dist + reward_near + reward_ctrl

Measured per step over 20 random episodes: dist -0.253, near -0.307, ctrl -0.925.
The control term is again the largest single per-step contribution under a random
policy, and Pusher has 7 actuators, so dropping or mis-weighting it is a real
change rather than a rounding error.
"""

from __future__ import annotations

import numpy as np


class PusherPartial:
    component_keys = ("goal", "reach", "effort", "omitted_ctrl")

    def __init__(
        self,
        use_dist: bool = True,
        use_near: bool = True,
        dist_floor: float | None = None,
        ctrl_weight: float = 0.0,
    ):
        self.use_dist = bool(use_dist)
        self.use_near = bool(use_near)
        self.dist_floor = dist_floor
        self.ctrl_weight = float(ctrl_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        info = info or {}
        omitted_ctrl = float(info.get("reward_ctrl", 0.0))

        goal = float(info.get("reward_dist", 0.0)) if self.use_dist else 0.0
        if self.dist_floor is not None:
            # Saturating: once the puck is this close to the goal the prior stops
            # asking for more. Nonlinear, so not a rescaling of the true reward.
            goal = max(goal, -self.dist_floor)
        reach = float(info.get("reward_near", 0.0)) if self.use_near else 0.0

        effort = self.ctrl_weight * float(np.square(np.asarray(action, dtype=np.float64)).sum())
        return {
            "partial": float(goal + reach - effort),
            "components": {
                "goal": float(goal),
                "reach": float(reach),
                "effort": float(-effort),
                "omitted_ctrl": omitted_ctrl,
            },
        }


_PARTIALS = {
    # 1. OMISSION: the two task terms, no effort cost. The straightforward
    #    "I described the task and forgot the actuators" prior.
    "spsh_goal_reach": dict(use_dist=True, use_near=True),
    # 2. GAMEABLE: reach the object and nothing else. Touching the puck is
    #    rewarded; moving it toward the goal is not mentioned at all.
    "spsh_reach": dict(use_dist=False, use_near=True),
    # 3. SATURATION: the reach shaping plus a goal term that stops paying once the
    #    puck is within 15 cm.
    "spsh_goal_cap": dict(use_dist=True, use_near=True, dist_floor=0.15),
    # 4. OMISSION of the crutch: the goal term alone, with neither the reach
    #    shaping that guides the arm to the puck nor the effort cost.
    "spsh_goal": dict(use_dist=True, use_near=False),
    # 5. OVER-PENALIZED EFFORT: both task terms, but effort charged at weight 0.3
    #    instead of the true 0.1. The designer over-corrected for jitter.
    #    Only 3x, because Pusher's action space is [-2,2]^7: sum(a**2) is about
    #    9.3 under a random policy, so the true 0.1 already outweighs both task
    #    terms combined there. At the 1.0 first tried, the effort bill was 20x the
    #    task and the prior's optimum was an arm that does not move.
    "spsh_timid": dict(use_dist=True, use_near=True, ctrl_weight=0.3),
}


def _factory(kwargs):
    return lambda env_id: PusherPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="mujoco",
            factory=_factory(kwargs),
            description=f"Pusher-v5 selection-grid prior '{name}'",
            env_ids=("Pusher-v5",),
            component_keys=PusherPartial.component_keys,
        )
