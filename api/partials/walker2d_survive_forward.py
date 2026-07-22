"""Hand-written Walker2d partial at ~0.5 partiality.

Built from the real reward components the env exposes (reward_survive,
reward_forward), NOT a scaling of the true reward. Keeps the full "stay
upright" signal plus a fraction of the forward-progress signal, and omits
the control cost. Measured partiality on random rollouts (fragment 1) ~ 0.49.
"""

from __future__ import annotations


class Walker2dSurviveForwardPartial:
    component_keys = ("reward_survive", "reward_forward")

    def __init__(self, w_survive: float = 1.0, w_forward: float = 0.4):
        self.w_survive = float(w_survive)
        self.w_forward = float(w_forward)

    def reset(self, info=None):
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        info = info or {}
        survive = float(info.get("reward_survive", 0.0))
        forward = float(info.get("reward_forward", 0.0))
        return {
            "partial": self.w_survive * survive + self.w_forward * forward,
            "components": {"reward_survive": survive, "reward_forward": forward},
        }


def register(registry):
    registry.register(
        name="walker2d_survive_forward",
        suite="mujoco",
        factory=lambda env_id: Walker2dSurviveForwardPartial(),
        description="Walker2d-v5 partial: reward_survive + 0.4*reward_forward (target partiality ~0.5).",
        env_ids=("Walker2d-v5",),
        component_keys=Walker2dSurviveForwardPartial.component_keys,
    )
