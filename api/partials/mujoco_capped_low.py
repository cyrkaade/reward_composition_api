"""Genuinely incomplete Hopper / Walker2d partials (low forward cap).

The existing `hopper_capped_forward_survive` and `walker2d_survive_forward`
turned out to produce BETTER policies than training on the ground-truth reward
(Hopper 2436 vs 1677, Walker2d 5854 vs 5723, both scored on true reward). That
inverts the premise of the composition experiments: if the partial alone already
beats the target, there is no gap for preference learning to fill.

The cause is that both keep an effectively unbounded forward-progress term. In
these environments an episode ends the moment the robot falls, and the survival
bonus is +1 per step for up to 1000 steps. A high forward-speed term tempts the
policy into unstable running that ends episodes early, which is what makes the
ground-truth reward hard to optimise. Down-weighting or capping that term high
still leaves enough incentive to chase speed.

These variants cap forward credit LOW. The partial then describes a designer who
knows the robot should stay upright and make some forward progress, but has not
specified how much forward progress is worth - the credit saturates almost
immediately. The result is a slow, stable gait: fine by the partial, clearly
short of what the true reward can reach.

Expected true-reward outcome (survival ~1000 steps at ~cap m/s):
  hopper_capped_low    ~1000 + 200 = ~1200   vs 1677 for the true reward
  walker2d_capped_low  ~1000 + 300 = ~1300   vs 5723 for the true reward
"""

from __future__ import annotations


class CappedForwardSurvivePartial:
    component_keys = ("capped_reward_forward", "reward_survive", "ctrl_omitted")

    def __init__(self, forward_cap: float):
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


_LEVELS = {
    "hopper_capped_low": ("Hopper-v5", 0.2),
    "walker2d_capped_low": ("Walker2d-v5", 0.3),
}


def _factory(cap):
    return lambda env_id: CappedForwardSurvivePartial(forward_cap=cap)


def register(registry) -> None:
    for name, (env_id, cap) in _LEVELS.items():
        registry.register(
            name=name,
            suite="mujoco",
            factory=_factory(cap),
            description=(
                f"{env_id} partial: reward_survive + min(reward_forward, {cap}); "
                "omits the control cost and saturates forward credit early, so it "
                "cannot reach the policy the true reward can."
            ),
            env_ids=(env_id,),
            component_keys=CappedForwardSurvivePartial.component_keys,
        )
