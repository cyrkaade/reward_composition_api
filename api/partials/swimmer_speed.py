"""Swimmer-v5 partial: reward SPEED, not progress in the rewarded direction.

Swimmer's true reward is ``forward_reward_weight * x_velocity - ctrl_cost``, and
its control cost carries weight 1e-4. Deleting that cost - the recipe that makes
``pusher_honest`` a genuinely incomplete prior - leaves something numerically
indistinguishable from the true reward, so it is useless as a partial here.

This partial makes the mistake a human designer actually makes instead: it
rewards the magnitude of the front tip's velocity and is blind to its direction.
Measured over 400 random steps, ``sqrt(vx^2 + vy^2)`` correlates 0.23 with
``reward_forward`` while ``vx`` alone correlates 0.97 - so a policy can score
well on this partial by swimming fast the wrong way, or sideways, and collect
approximately nothing on the true reward.

Reads the observation rather than ``info`` on purpose: Swimmer's info exposes the
answer (``reward_forward``), and a partial that reads the answer is not a prior.
With the default ``exclude_current_positions_from_observation=True`` the
observation is ``[3 joint angles, vx, vy, 3 angular velocities]``, so the tip
velocity is at indices 3 and 4.
"""

from __future__ import annotations

import numpy as np

_VX, _VY = 3, 4


class SwimmerSpeedPartial:
    component_keys = ("speed", "x_velocity")

    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def reset(self, info=None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        state = np.asarray(next_obs, dtype=np.float64)
        vx, vy = float(state[_VX]), float(state[_VY])
        speed = float(np.hypot(vx, vy))
        return {
            "partial": self.weight * speed,
            # x_velocity is recorded, never rewarded: it is what the partial is
            # blind to, so logging it makes the gap directly measurable.
            "components": {"speed": speed, "x_velocity": vx},
        }


def register(registry) -> None:
    registry.register(
        name="swimmer_speed",
        suite="mujoco",
        factory=lambda env_id: SwimmerSpeedPartial(),
        description=(
            "Incomplete Swimmer partial: rewards |velocity| of the front tip, blind to "
            "direction (correlates 0.23 with the true forward reward)."
        ),
        env_ids=("Swimmer-v5",),
        component_keys=SwimmerSpeedPartial.component_keys,
    )
