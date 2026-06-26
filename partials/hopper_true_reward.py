"""Full-form Hopper reward partial.

This mirrors Gymnasium Hopper-v5's reward decomposition:

    true reward = reward_forward + reward_survive + reward_ctrl
"""

from __future__ import annotations

from reward_composition_api.registry import PartialRewardStep


class HopperTrueRewardPartial:
    component_keys = ("reward_forward", "reward_survive", "reward_ctrl")

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        info = info or {}
        components = {key: float(info.get(key, 0.0)) for key in self.component_keys}
        return PartialRewardStep(partial=float(sum(components.values())), components=components)


def register(registry) -> None:
    registry.register(
        name="hopper_true_reward",
        suite="mujoco",
        factory=lambda env_id: HopperTrueRewardPartial(),
        description="Full Hopper-v5 true reward: reward_forward + reward_survive + reward_ctrl.",
        env_ids=("Hopper-v5",),
        component_keys=HopperTrueRewardPartial.component_keys,
    )
