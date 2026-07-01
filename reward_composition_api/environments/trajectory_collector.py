from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from stable_baselines3 import PPO

from reward_composition_api.data_structures import Trajectory
from reward_composition_api.registry import PartialSpec


ModelObservationFn = Callable[[Any, Any], Any]
ActionConverterFn = Callable[[Any, Any], Any]
ResetRewardStateFn = Callable[[dict], None]
DefaultPartialRewardFn = Callable[[Any, Any, Any, float, bool, bool, dict], float]


def zero_partial_reward(previous_obs, action, observation, true_reward, terminated, truncated, info) -> float:
    return 0.0


@dataclass
class PolicyTrajectoryCollector:
    model: PPO
    stats_source: Any
    make_env: Callable[[str], Any]
    env_id: str
    custom_partial: PartialSpec | None
    model_observation: ModelObservationFn
    action_converter: ActionConverterFn
    default_partial_reward: DefaultPartialRewardFn = zero_partial_reward
    reset_reward_state: ResetRewardStateFn | None = None

    def rollout_trajectories(self, total_timesteps: int, seed: int) -> list[Trajectory]:
        env = self.make_env(self.env_id)
        partial = self.custom_partial.create(self.env_id) if self.custom_partial else None
        trajectories = []
        trajectory = Trajectory()
        obs, info = env.reset(seed=seed)
        self._reset_reward_state(info, partial)
        steps = 0

        try:
            while steps < total_timesteps:
                model_obs = self.model_observation(self.stats_source, obs)
                action, _ = self.model.predict(model_obs, deterministic=False)
                env_action = self.action_converter(env, action)
                new_obs, true_reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
                if partial is None:
                    partial_reward = self.default_partial_reward(
                        obs,
                        env_action,
                        new_obs,
                        float(true_reward),
                        terminated,
                        truncated,
                        info,
                    )
                else:
                    partial_reward = partial.step(
                        obs,
                        env_action,
                        new_obs,
                        true_reward,
                        terminated,
                        truncated,
                        info,
                    ).partial
                trajectory.push_state(new_obs, env_action, done, info, float(true_reward), partial_reward)
                steps += 1

                if done:
                    trajectories.append(trajectory)
                    trajectory = Trajectory()
                    obs, info = env.reset()
                    self._reset_reward_state(info, partial)
                else:
                    obs = new_obs
        finally:
            env.close()

        if trajectory.states:
            trajectories.append(trajectory)
        return trajectories

    def _reset_reward_state(self, info: dict, partial) -> None:
        if self.reset_reward_state is not None:
            self.reset_reward_state(info)
        if partial is not None:
            partial.reset(info)
