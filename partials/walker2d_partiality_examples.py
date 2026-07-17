"""Walker2d partiality examples.

These are hand-written reward formulas for quick partiality checks. The
medium and low variants are not scaled copies of the true reward.
"""

from __future__ import annotations

import numpy as np

from reward_composition_api.registry import PartialRewardStep


class Walker2dFullPartial:
    component_keys = ("reward_forward", "reward_survive", "reward_ctrl")

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        info = info or {}
        components = {key: float(info.get(key, 0.0)) for key in self.component_keys}
        return PartialRewardStep(
            partial=float(sum(components.values())),
            components=components,
        )


class Walker2dCappedForwardSurvivePartial:
    component_keys = ("capped_forward", "reward_survive", "ctrl_omitted")

    def __init__(self, forward_cap: float = 0.75):
        self.forward_cap = float(forward_cap)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        info = info or {}
        forward = float(info.get("reward_forward", 0.0))
        capped_forward = max(min(forward, self.forward_cap), -self.forward_cap)
        survive = float(info.get("reward_survive", 0.0))
        ctrl_omitted = float(info.get("reward_ctrl", 0.0))
        return PartialRewardStep(
            partial=capped_forward + survive,
            components={
                "capped_forward": capped_forward,
                "reward_survive": survive,
                "ctrl_omitted": ctrl_omitted,
            },
        )


class Walker2dWeakCappedForwardSurvivePartial(Walker2dCappedForwardSurvivePartial):
    def __init__(self):
        super().__init__(forward_cap=0.25)


class Walker2dLowPosturePartial:
    component_keys = ("upright_bonus", "action_penalty")

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        state = np.asarray(next_obs, dtype=np.float64)
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        height = float(state[0]) if state.size > 0 else 0.0
        angle = float(state[1]) if state.size > 1 else 0.0
        upright_bonus = 0.1 if height > 0.8 and abs(angle) < 0.5 else 0.0
        action_penalty = -0.001 * float(np.sum(np.square(act)))
        return PartialRewardStep(
            partial=upright_bonus + action_penalty,
            components={
                "upright_bonus": upright_bonus,
                "action_penalty": action_penalty,
            },
        )


def register(registry) -> None:
    registry.register(
        name="walker2d_example_full",
        suite="mujoco",
        factory=lambda env_id: Walker2dFullPartial(),
        description="Full Walker2d reward: forward + survive + control.",
        env_ids=("Walker2d-v5",),
        component_keys=Walker2dFullPartial.component_keys,
    )
    registry.register(
        name="walker2d_example_medium",
        suite="mujoco",
        factory=lambda env_id: Walker2dCappedForwardSurvivePartial(forward_cap=0.75),
        description="Medium Walker2d partial: survive plus capped forward progress; omits control cost.",
        env_ids=("Walker2d-v5",),
        component_keys=Walker2dCappedForwardSurvivePartial.component_keys,
    )
    registry.register(
        name="walker2d_example_weak",
        suite="mujoco",
        factory=lambda env_id: Walker2dWeakCappedForwardSurvivePartial(),
        description="Weak Walker2d partial: survive plus tightly capped forward progress; omits control cost.",
        env_ids=("Walker2d-v5",),
        component_keys=Walker2dWeakCappedForwardSurvivePartial.component_keys,
    )
    registry.register(
        name="walker2d_example_low",
        suite="mujoco",
        factory=lambda env_id: Walker2dLowPosturePartial(),
        description="Low-information Walker2d partial: small upright/action signal only.",
        env_ids=("Walker2d-v5",),
        component_keys=Walker2dLowPosturePartial.component_keys,
    )
