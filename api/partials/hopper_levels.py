"""Hand-written Hopper partials at target partiality ~0.25 / 0.50 / 0.75.

Built from real reward components the env exposes (reward_survive,
reward_forward), NOT a scaling of the true reward. Different hand-chosen
weightings of "stay alive" vs "move forward" give different alignment with
the true return. Measured partiality on random rollouts (fragment 1):
  hopper_p25 ~ 0.22, hopper_p50 ~ 0.53, hopper_p75 ~ 0.78.
"""

from __future__ import annotations


class HopperLevelPartial:
    def __init__(self, w_survive, w_forward):
        self.w_survive = float(w_survive)
        self.w_forward = float(w_forward)

    def reset(self, info=None):
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        info = info or {}
        survive = float(info.get("reward_survive", 0.0))
        forward = float(info.get("reward_forward", 0.0))
        return {
            "partial": self.w_survive * survive + self.w_forward * forward,
            "components": {"reward_survive": survive, "reward_forward": forward},
        }


_LEVELS = {
    "hopper_p25": dict(w_survive=1.0, w_forward=0.0),
    "hopper_p50": dict(w_survive=1.0, w_forward=0.4),
    "hopper_p75": dict(w_survive=0.0, w_forward=1.0),
}


def _factory(kwargs):
    return lambda env_id: HopperLevelPartial(**kwargs)


def register(registry):
    for name, kwargs in _LEVELS.items():
        registry.register(
            name=name,
            suite="mujoco",
            factory=_factory(kwargs),
            description=f"Hand-written Hopper partial, target partiality ~0.{name[-2:]}",
            env_ids=("Hopper-v5",),
            component_keys=("reward_survive", "reward_forward"),
        )
