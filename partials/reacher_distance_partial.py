"""Reacher-v5 distance-only partial reward.

This matches the old main-branch MuJoCo default profile for Reacher:
use ``reward_dist`` as the partial reward and omit the control penalty.
"""

from reward_composition_api.registry import PartialRewardStep


class ReacherDistancePartial:
    def reset(self, info=None):
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        reward_dist = float(info.get("reward_dist", 0.0))
        reward_ctrl = float(info.get("reward_ctrl", 0.0))
        return PartialRewardStep(
            partial=reward_dist,
            components={
                "reward_dist": reward_dist,
                "reward_ctrl": reward_ctrl,
            },
        )


def register(registry):
    registry.register(
        name="reacher_distance_partial",
        suite="mujoco",
        factory=lambda env_id: ReacherDistancePartial(),
        description="Reacher-v5 partial reward using reward_dist only.",
        env_ids=("Reacher-v5",),
        component_keys=("reward_dist", "reward_ctrl"),
    )
