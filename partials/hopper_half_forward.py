"""Weak Hopper partial reward for composition experiments.

This intentionally keeps only a down-weighted forward-progress term and
omits the survive bonus and control cost. It is a weaker signal than the
default Hopper partial, which is already very close to the true reward.
"""

from __future__ import annotations

from reward_composition_api.registry import PartialRewardStep


class HopperHalfForwardPartial:
    component_keys = ("half_reward_forward", "survive_omitted", "ctrl_omitted")

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        info = info or {}
        half_forward = 0.5 * float(info.get("reward_forward", 0.0))
        survive_omitted = float(info.get("reward_survive", 0.0))
        ctrl_omitted = float(info.get("reward_ctrl", 0.0))
        return PartialRewardStep(
            partial=half_forward,
            components={
                "half_reward_forward": half_forward,
                "survive_omitted": survive_omitted,
                "ctrl_omitted": ctrl_omitted,
            },
        )


def register(registry) -> None:
    registry.register(
        name="hopper_half_forward",
        suite="mujoco",
        factory=lambda env_id: HopperHalfForwardPartial(),
        description="Weak Hopper-v5 partial: 0.5 * reward_forward only; omits survive bonus and control cost.",
        env_ids=("Hopper-v5",),
        component_keys=HopperHalfForwardPartial.component_keys,
    )
