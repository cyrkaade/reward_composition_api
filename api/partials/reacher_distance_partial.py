"""Reacher-v5 distance-only partial reward.

Uses ``reward_dist`` (how close the fingertip is to the target) as the
partial reward and omits the control-cost penalty from the full reward.
"""

from __future__ import annotations


class ReacherDistancePartial:
    def reset(self, info=None):
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        reward_dist = float(info.get("reward_dist", 0.0))
        reward_ctrl = float(info.get("reward_ctrl", 0.0))
        return {
            "partial": reward_dist,
            "components": {"reward_dist": reward_dist, "reward_ctrl": reward_ctrl},
        }


def register(registry):
    registry.register(
        name="reacher_distance_partial",
        suite="mujoco",
        factory=lambda env_id: ReacherDistancePartial(),
        description="Reacher-v5 partial reward using reward_dist only (no control penalty).",
        env_ids=("Reacher-v5",),
        component_keys=("reward_dist", "reward_ctrl"),
    )
