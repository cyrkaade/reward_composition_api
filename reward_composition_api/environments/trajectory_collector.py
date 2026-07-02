from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, VecNormalize

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


class BufferingWrapper(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        true_reward_key: str = "true_reward",
        partial_reward_key: str = "partial_reward",
    ):
        super().__init__(venv)
        self.true_reward_key = true_reward_key
        self.partial_reward_key = partial_reward_key
        self.temp_trajectories = [Trajectory() for _ in range(self.num_envs)]
        self.finished_trajectories: list[Trajectory] = []
        self._saved_acts = None

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        self.temp_trajectories = [Trajectory() for _ in range(self.num_envs)]
        self.finished_trajectories = []
        return obs

    def step_async(self, actions):
        self._saved_acts = actions
        self.venv.step_async(actions)

    def step_wait(self):
        new_obs, rewards, dones, infos = self.venv.step_wait()
        for env_idx in range(self.num_envs):
            info = infos[env_idx]
            trajectory_obs = info.get("terminal_observation", _batch_item(new_obs, env_idx))
            true_reward = float(info.get(self.true_reward_key, rewards[env_idx]))
            partial_reward = float(info.get(self.partial_reward_key, 0.0))
            self.temp_trajectories[env_idx].push_state(
                trajectory_obs,
                _batch_item(self._saved_acts, env_idx),
                bool(dones[env_idx]),
                dict(info),
                true_reward,
                partial_reward,
            )
            if dones[env_idx]:
                self.finished_trajectories.append(self.temp_trajectories[env_idx])
                self.temp_trajectories[env_idx] = Trajectory()
        return new_obs, rewards, dones, infos

    def pop_trajectories(self) -> list[Trajectory]:
        trajectories = self.finished_trajectories
        self.finished_trajectories = []
        return trajectories


class TrajectoryCollector:
    def __init__(self, vec_env: VecEnv, agent: PPO, verbose: bool = False):
        self.vec_env, self.buffering_wrapper = _buffered_vec_env(vec_env)
        self.agent = agent
        self.verbose = verbose

    def _run_algo(self, total_timesteps: int) -> None:
        trained_steps = 0
        obs = self.vec_env.reset()
        while trained_steps < total_timesteps:
            actions, _ = self.agent.predict(obs, deterministic=False)
            obs, _, _, _ = self.vec_env.step(actions)
            trained_steps += self.vec_env.num_envs

    def rollout_trajectories(self, total_timesteps: int, seed: int | None = None) -> list[Trajectory]:
        previous_training = self.vec_env.training if isinstance(self.vec_env, VecNormalize) else None
        if isinstance(self.vec_env, VecNormalize):
            self.vec_env.training = False
        try:
            if seed is not None:
                self.vec_env.seed(seed)
            self._run_algo(total_timesteps)
            return self.buffering_wrapper.pop_trajectories()
        finally:
            if isinstance(self.vec_env, VecNormalize):
                self.vec_env.training = bool(previous_training)
            _clear_model_rollout_state(self.agent)


def _buffered_vec_env(vec_env: VecEnv) -> tuple[VecEnv, BufferingWrapper]:
    if isinstance(vec_env, BufferingWrapper):
        return vec_env, vec_env
    if isinstance(vec_env, VecNormalize):
        if not isinstance(vec_env.venv, BufferingWrapper):
            vec_env.venv = BufferingWrapper(vec_env.venv)
        return vec_env, vec_env.venv
    wrapper = BufferingWrapper(vec_env)
    return wrapper, wrapper


def _batch_item(value, index: int):
    if isinstance(value, dict):
        return {key: _batch_item(item, index) for key, item in value.items()}
    return np.asarray(value)[index]


def _clear_model_rollout_state(model) -> None:
    if hasattr(model, "_last_obs"):
        model._last_obs = None
