"""Aligned HalfCheetah partial: reward forward running, omit the control cost.

HalfCheetah's true reward is forward progress minus a control cost. This
partial keeps the forward-run term (reward_forward) and drops the control
cost, so it is well-aligned with the true reward but incomplete. HalfCheetah
has a fixed horizon (no early termination), which makes it a low-variance,
reproducible environment for reward-composition experiments.
"""

from __future__ import annotations


class HalfCheetahForwardPartial:
    component_keys = ("reward_forward", "reward_ctrl")

    def reset(self, info=None):
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        info = info or {}
        reward_forward = float(info.get("reward_forward", 0.0))
        reward_ctrl = float(info.get("reward_ctrl", 0.0))
        return {
            "partial": reward_forward,
            "components": {"reward_forward": reward_forward, "reward_ctrl": reward_ctrl},
        }


def register(registry):
    registry.register(
        name="halfcheetah_forward",
        suite="mujoco",
        factory=lambda env_id: HalfCheetahForwardPartial(),
        description="Aligned HalfCheetah partial: reward_forward only (omits control cost).",
        env_ids=("HalfCheetah-v5",),
        component_keys=HalfCheetahForwardPartial.component_keys,
    )
