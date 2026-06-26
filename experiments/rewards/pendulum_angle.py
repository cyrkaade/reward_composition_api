from __future__ import annotations

import math


def partial_reward(obs, action, next_obs, true_reward, terminated, truncated, info):
    cos_theta = float(next_obs[0])
    sin_theta = float(next_obs[1])
    theta = math.atan2(sin_theta, cos_theta)
    angle_reward = -(theta**2)
    return {
        "partial": angle_reward,
        "components": {"angle_reward": angle_reward},
    }
