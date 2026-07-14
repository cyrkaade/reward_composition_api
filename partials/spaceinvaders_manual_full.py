"""Full-form Space Invaders reward partial.

This is an ideal manual partial for composition experiments: it returns the
environment's true reward exactly.
"""

from __future__ import annotations

from reward_composition_api.registry import PartialRewardStep


class SpaceInvadersManualFull:
    component_keys = ("true_env_reward",)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        reward = float(true_reward)
        return PartialRewardStep(
            partial=reward,
            components={"true_env_reward": reward},
        )


def register(registry) -> None:
    registry.register(
        name="spaceinvaders_manual_full",
        suite="atari",
        factory=lambda env_id: SpaceInvadersManualFull(),
        description="SpaceInvaders full-form partial: exact environment true reward.",
        env_ids=("ALE/SpaceInvaders-v5", "ALE/SpaceInvaders-ram-v5"),
        component_keys=SpaceInvadersManualFull.component_keys,
    )
