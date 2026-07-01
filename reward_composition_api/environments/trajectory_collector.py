from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

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


@dataclass
class VectorizedPolicyTrajectoryCollector:
    model: PPO
    vec_env: VecEnv
    true_reward_key: str = "true_reward"
    partial_reward_key: str = "partial_reward"

    def rollout_trajectories(self, total_timesteps: int, seed: int | None = None) -> list[Trajectory]:
        previous_training = self.vec_env.training if isinstance(self.vec_env, VecNormalize) else None
        if isinstance(self.vec_env, VecNormalize):
            self.vec_env.training = False

        trajectories = [Trajectory() for _ in range(self.vec_env.num_envs)]
        completed: list[Trajectory] = []
        steps = 0

        try:
            if seed is not None:
                self.vec_env.seed(seed)
            model_obs = self.vec_env.reset()

            while steps < total_timesteps:
                action, _ = self.model.predict(model_obs, deterministic=False)
                next_model_obs, rewards, dones, infos = self.vec_env.step(action)
                raw_next_obs = self._raw_current_observation(next_model_obs)

                for env_index, info in enumerate(infos):
                    done = bool(dones[env_index])
                    state_obs = self._state_observation(info, raw_next_obs, env_index, done)
                    env_action = _batch_item(action, env_index)
                    true_reward = float(info.get(self.true_reward_key, rewards[env_index]))
                    partial_reward = float(info.get(self.partial_reward_key, 0.0))

                    trajectories[env_index].push_state(
                        state_obs,
                        env_action,
                        done,
                        dict(info),
                        true_reward,
                        partial_reward,
                    )

                    if done:
                        completed.append(trajectories[env_index])
                        trajectories[env_index] = Trajectory()

                steps += self.vec_env.num_envs
                model_obs = next_model_obs
        finally:
            if isinstance(self.vec_env, VecNormalize):
                self.vec_env.training = bool(previous_training)
            _clear_model_rollout_state(self.model)

        completed.extend(trajectory for trajectory in trajectories if trajectory.states)
        return completed

    def _raw_current_observation(self, observation):
        if isinstance(self.vec_env, VecNormalize):
            return self.vec_env.get_original_obs()
        return observation

    def _state_observation(self, info: dict, raw_next_obs, env_index: int, done: bool):
        if done and "terminal_observation" in info:
            terminal_observation = info["terminal_observation"]
            if isinstance(self.vec_env, VecNormalize):
                terminal_observation = self.vec_env.unnormalize_obs(terminal_observation)
            return terminal_observation
        return _batch_item(raw_next_obs, env_index)


def _batch_item(value, index: int):
    if isinstance(value, dict):
        return {key: _batch_item(item, index) for key, item in value.items()}
    return np.asarray(value)[index]


def _clear_model_rollout_state(model) -> None:
    if hasattr(model, "_last_obs"):
        model._last_obs = None
