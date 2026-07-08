"""Generic true-reward scalar partials for cross-suite baselines.

These partials intentionally use the environment reward passed to the partial
API, which makes them useful as controlled half-reward and ideal-reward
baselines across MuJoCo, Atari, Box2D, and Gymnasium environments.
"""

from __future__ import annotations

from reward_composition_api.registry import PartialRewardStep


class ScaledTrueRewardPartial:
    def __init__(self, scale: float):
        self.scale = float(scale)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        reward = float(true_reward)
        partial = self.scale * reward
        return PartialRewardStep(
            partial=partial,
            components={
                "true_reward": reward,
                "scaled_true_reward": partial,
            },
        )


def register(registry) -> None:
    component_keys = ("true_reward", "scaled_true_reward")
    for suite in ("mujoco", "atari", "box2d", "gym"):
        registry.register(
            name="half_true_reward",
            suite=suite,
            factory=lambda env_id: ScaledTrueRewardPartial(0.5),
            description="Half of the environment reward, for controlled partial-reward baselines.",
            component_keys=component_keys,
        )
        registry.register(
            name="full_true_reward",
            suite=suite,
            factory=lambda env_id: ScaledTrueRewardPartial(1.0),
            description="The environment reward exactly, for ideal full-form partial baselines.",
            component_keys=component_keys,
        )
