"""Imperfect Hopper partial reward with capped forward progress.

Keeps the survival signal and rewards forward progress only up to a
per-step cap. Intentionally omits the control cost and does not
distinguish between moderate and very fast hopping once the cap is hit.
"""

from __future__ import annotations


class HopperCappedForwardSurvivePartial:
    component_keys = ("capped_reward_forward", "reward_survive", "ctrl_omitted")

    def __init__(self, forward_cap: float = 1.0):
        self.forward_cap = float(forward_cap)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        info = info or {}
        capped_forward = min(float(info.get("reward_forward", 0.0)), self.forward_cap)
        survive = float(info.get("reward_survive", 0.0))
        ctrl_omitted = float(info.get("reward_ctrl", 0.0))
        return {
            "partial": capped_forward + survive,
            "components": {
                "capped_reward_forward": capped_forward,
                "reward_survive": survive,
                "ctrl_omitted": ctrl_omitted,
            },
        }


def register(registry) -> None:
    registry.register(
        name="hopper_capped_forward_survive",
        suite="mujoco",
        factory=lambda env_id: HopperCappedForwardSurvivePartial(),
        description=(
            "Imperfect Hopper-v5 partial: reward_survive + min(reward_forward, 1.0); "
            "omits control cost and high-speed forward differences."
        ),
        env_ids=("Hopper-v5",),
        component_keys=HopperCappedForwardSurvivePartial.component_keys,
    )
