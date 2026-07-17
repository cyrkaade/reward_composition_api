from __future__ import annotations

from dataclasses import dataclass


class Trajectory:
    def __init__(self, states=None):
        self.states = list(states or [])

    def push_state(self, obs, act, done, info, true_rew, partial_rew):
        assert len(self.states) == 0 or not self.states[-1]["done"], "trying to push a state to a trajectory that is already done"

        self.states.append(
            {
                "obs": obs,
                "act": act,
                "done": done,
                "info": info,
                "rew": true_rew,
                "partial_rew": partial_rew,
            }
        )

    def get_states(self):
        return self.states

    def get_summed_reward(self):
        return sum(state["rew"] for state in self.states)


@dataclass
class Preference:
    t1: Trajectory
    t2: Trajectory
    rating: float
