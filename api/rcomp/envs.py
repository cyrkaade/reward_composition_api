"""Shared environment mechanics: vectorized env construction, VecNormalize
handling, observation/action feature extraction, and trajectory collection
over the live training VecEnv."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import flatten
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper, VecNormalize

from .data import Trajectory


def policy_observation(stats_source, observation):
    if isinstance(stats_source, VecNormalize):
        return stats_source.normalize_obs(np.asarray(observation)[None])
    return observation


def normalize_obs(stats_source, observation):
    observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    if isinstance(stats_source, VecNormalize):
        return stats_source.normalize_obs(observation)
    return observation


def observation_features(space: spaces.Space, observation) -> np.ndarray:
    return np.asarray(flatten(space, observation), dtype=np.float32).reshape(-1)


def action_features(space: spaces.Space, action) -> np.ndarray:
    return np.asarray(flatten(space, action_for_space(space, action)), dtype=np.float32).reshape(-1)


def action_for_space(space: spaces.Space, action):
    if isinstance(space, spaces.Discrete):
        return int(np.asarray(action).reshape(-1)[0])
    if isinstance(space, spaces.MultiDiscrete):
        return np.asarray(action, dtype=space.dtype).reshape(space.shape)
    if isinstance(space, spaces.MultiBinary):
        return np.asarray(action, dtype=space.dtype).reshape(space.shape)
    return np.asarray(action, dtype=getattr(space, "dtype", np.float32)).reshape(space.shape)


class UnhealthyPenaltyWrapper(gym.Wrapper):
    """Trade the healthy/unhealthy TERMINATION cliff for a persistent penalty.

    Opt-in, applied to the TRAINING env only (see ``RlhfTrainer.train_env_fn``);
    every evaluation path keeps using ``suite.make_raw_env`` unchanged, so the
    yardstick a run is scored on is the same standard environment every other
    arm in the project is scored on.

    Two changes, and only these two:

    1. ``terminated`` is forced ``False``. In gymnasium 1.2.3 the sole source of
       ``terminated`` in ``walker2d_v5.py`` (L293) and ``ant_v5.py`` (L359) is
       ``(not self.is_healthy) and self._terminate_when_unhealthy``, so forcing
       it is exactly ``terminate_when_unhealthy=False`` -- verified step for step
       against the constructor route (identical observations and identical
       rewards over 400 steps on both envs, only the flag differs). Doing it in a
       wrapper rather than through ``gym.make`` kwargs is what keeps the change
       off the evaluation envs. ``gym.make`` applies ``TimeLimit`` INSIDE this
       wrapper, so truncation at 1000 steps is untouched.

    2. The env's healthy bonus is a boolean gate -- ``healthy_reward`` is
       ``is_healthy * self._healthy_reward``, i.e. +1 while healthy and 0 while
       unhealthy -- so there is no constructor argument that produces a genuine
       penalty (a negative ``healthy_reward`` weight gives -1 while healthy and 0
       while unhealthy, the opposite of what is wanted). This wrapper subtracts
       that term and adds ``+penalty`` while healthy / ``-penalty`` while not.

    The replacement is written back into ``info["reward_survive"]`` as well as
    into the reward, because the hand-written priors read their survive term
    from that key (``partials/sel_walker2d.py``, ``partials/sel_ant.py``).
    Without it the ``partial`` arm would receive the termination removal but not
    the penalty, and would no longer be the same treatment as the ``true`` arm.

    NOTE ON MAGNITUDE: falling costs ``2 * penalty`` per step here (the healthy
    bonus lost plus the penalty gained), whereas in the standard environment it
    forfeits the entire remaining episode -- roughly 3.5/step for a Walker2d
    already running at 2.5 m/s. At ``penalty=1.0`` the modified objective is
    therefore more tolerant of falling than the true one, by about the forward
    reward rate. That gap is what the second, modified-env evaluation measures.
    """

    def __init__(self, env, penalty: float = 1.0):
        super().__init__(env)
        if not hasattr(env.unwrapped, "is_healthy"):
            raise ValueError(
                f"{type(env.unwrapped).__name__} has no 'is_healthy' property; the unhealthy "
                "penalty only applies to environments with a healthy/unhealthy termination rule "
                "(Walker2d-v5, Ant-v5, Hopper-v5, Humanoid-v5)"
            )
        self.penalty = float(penalty)

    def step(self, action):
        observation, reward, _terminated, truncated, info = self.env.step(action)
        replacement = self.penalty if bool(self.env.unwrapped.is_healthy) else -self.penalty
        info = dict(info)
        reward = float(reward) - float(info.get("reward_survive", 0.0)) + replacement
        info["reward_survive"] = replacement
        info["unhealthy_penalty"] = self.penalty
        return observation, reward, False, truncated, info


def apply_unhealthy_penalty(env, penalty: float | None):
    """Wrap ``env`` when ``penalty`` is set; return it untouched when it is not."""
    if penalty is None:
        return env
    return UnhealthyPenaltyWrapper(env, penalty=penalty)


def make_train_env(env_fn, n_envs: int, monitor_dir: Path, normalize: bool):
    env = make_vec_env(env_fn, n_envs=n_envs, vec_env_cls=DummyVecEnv, monitor_dir=str(monitor_dir))
    if normalize:
        return VecNormalize(env, norm_obs=True, norm_reward=True)
    return env


def make_raw_eval_env(make_raw_env: Callable[[str], object], env_id: str):
    return DummyVecEnv([lambda: Monitor(make_raw_env(env_id))])


def make_eval_env(make_raw_env: Callable[[str], object], env_id: str, stats_source=None):
    env = make_raw_eval_env(make_raw_env, env_id)
    if isinstance(stats_source, VecNormalize):
        eval_env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = stats_source.obs_rms
        eval_env.ret_rms = stats_source.ret_rms
        return eval_env
    return env


def load_eval_env(make_raw_env: Callable[[str], object], env_id: str, stats_path: Path) -> VecNormalize:
    env = VecNormalize.load(stats_path, make_raw_eval_env(make_raw_env, env_id))
    env.training = False
    env.norm_reward = False
    return env


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
        trajectories = [
            *self.finished_trajectories,
            *(trajectory for trajectory in self.temp_trajectories if trajectory.states),
        ]
        self.finished_trajectories = []
        self.temp_trajectories = [Trajectory() for _ in range(self.num_envs)]
        return trajectories


class TrajectoryCollector:
    """Rolls out the current policy on the live training VecEnv, buffering
    per-env trajectories, and restores the env exactly as it was."""

    def __init__(self, vec_env: VecEnv, agent: PPO, verbose: bool = False):
        self.source_vec_env = vec_env
        self.vec_env = vec_env
        self.buffering_wrapper: BufferingWrapper | None = None
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
        self.vec_env, self.buffering_wrapper, restore_env = _buffered_vec_env(self.source_vec_env)
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
            restore_env()
            self.vec_env = self.source_vec_env
            _clear_model_rollout_state(self.agent)


def _buffered_vec_env(vec_env: VecEnv) -> tuple[VecEnv, BufferingWrapper, Callable[[], None]]:
    if isinstance(vec_env, BufferingWrapper):
        return vec_env, vec_env, lambda: None
    if isinstance(vec_env, VecNormalize):
        original_venv = vec_env.venv
        if isinstance(original_venv, BufferingWrapper):
            return vec_env, original_venv, lambda: None
        wrapper = BufferingWrapper(original_venv)
        vec_env.venv = wrapper

        def restore() -> None:
            if vec_env.venv is wrapper:
                vec_env.venv = original_venv

        return vec_env, wrapper, restore
    wrapper = BufferingWrapper(vec_env)
    return wrapper, wrapper, lambda: None


def _batch_item(value, index: int):
    if isinstance(value, dict):
        return {key: _batch_item(item, index) for key, item in value.items()}
    return np.asarray(value)[index]


def _clear_model_rollout_state(model) -> None:
    if hasattr(model, "_last_obs"):
        model._last_obs = None
