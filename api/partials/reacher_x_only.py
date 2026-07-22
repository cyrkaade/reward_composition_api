"""Reacher-v5 partial that rewards closing only the x-component of the
fingertip-to-target distance, ignoring y entirely (keeps the control cost).

True Reacher reward = -sqrt(dx^2 + dy^2) + reward_ctrl. This partial replaces
the distance term with -|dx| (x-axis only), so the y-axis error is the single
component the partial is blind to. In delta mode the reward model can learn
that missing y term; in naive mode the model never sees the partial, so it
effectively re-learns and double-weights the x information.

dx = obs[8], dy = obs[9] in the Reacher-v5 observation.
"""

from __future__ import annotations


class ReacherXOnlyPartial:
    component_keys = ("reward_dist_x", "reward_ctrl", "y_error_omitted")

    def reset(self, info=None):
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        if next_obs is None or len(next_obs) < 10:
            dx = dy = 0.0
        else:
            dx = float(next_obs[8])
            dy = float(next_obs[9])
        reward_ctrl = float((info or {}).get("reward_ctrl", 0.0))
        reward_dist_x = -abs(dx)
        return {
            "partial": reward_dist_x + reward_ctrl,
            "components": {
                "reward_dist_x": reward_dist_x,
                "reward_ctrl": reward_ctrl,
                "y_error_omitted": -abs(dy),
            },
        }


def register(registry):
    registry.register(
        name="reacher_x_only",
        suite="mujoco",
        factory=lambda env_id: ReacherXOnlyPartial(),
        description="Reacher-v5 partial: -|x-distance| + control cost; blind to the y-axis error.",
        env_ids=("Reacher-v5",),
        component_keys=ReacherXOnlyPartial.component_keys,
    )
