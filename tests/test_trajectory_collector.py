from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from reward_composition_api.environments.trajectory_collector import BufferingWrapper, PolicyTrajectoryCollector, TrajectoryCollector


class DummyModel:
    def __init__(self):
        self.observations = []

    def predict(self, observation, deterministic=False):
        self.observations.append((observation, deterministic))
        return np.asarray([[1.0]], dtype=np.float32), None


class VectorDummyModel:
    def __init__(self):
        self.observations = []
        self._last_obs = "stale"

    def predict(self, observation, deterministic=False):
        self.observations.append((np.asarray(observation).copy(), deterministic))
        return np.ones((np.asarray(observation).shape[0], 1), dtype=np.float32), None


class CountingEnv:
    observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self):
        self.reset_count = 0
        self.actions = []
        self.closed = False

    def reset(self, seed=None):
        self.reset_count += 1
        return np.asarray([float(self.reset_count)], dtype=np.float32), {"reset": self.reset_count, "seed": seed}

    def step(self, action):
        self.actions.append(action)
        done = len(self.actions) == 2
        return (
            np.asarray([float(len(self.actions) + 1)], dtype=np.float32),
            10.0 + len(self.actions),
            done,
            False,
            {"step": len(self.actions)},
        )

    def close(self):
        self.closed = True


class VectorCountingEnv(gym.Env):
    metadata = {}

    def __init__(self, offset: float = 0.0):
        self.offset = float(offset)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.step_count = 0
        self.seed_value = None

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.seed_value = seed
        return np.asarray([self.offset], dtype=np.float32), {"seed": seed}

    def step(self, action):
        self.step_count += 1
        terminated = self.step_count == 2
        observation = np.asarray([self.offset + self.step_count], dtype=np.float32)
        return (
            observation,
            99.0,
            terminated,
            False,
            {
                "true_reward": self.offset + self.step_count,
                "partial_reward": self.step_count / 10.0,
                "action": np.asarray(action, dtype=np.float32).copy(),
            },
        )


class NonTerminatingVectorEnv(gym.Env):
    metadata = {}

    def __init__(self, offset: float = 0.0):
        self.offset = float(offset)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return np.asarray([self.offset], dtype=np.float32), {"seed": seed}

    def step(self, action):
        self.step_count += 1
        observation = np.asarray([self.offset + self.step_count], dtype=np.float32)
        return (
            observation,
            99.0,
            False,
            False,
            {
                "true_reward": self.offset + self.step_count,
                "partial_reward": self.step_count / 10.0,
            },
        )


class CustomPartial:
    def __init__(self):
        self.reset_infos = []
        self.step_actions = []

    def reset(self, info):
        self.reset_infos.append(dict(info))

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        self.step_actions.append(action)
        return type("PartialStep", (), {"partial": 7.5})()


class CustomPartialSpec:
    def __init__(self):
        self.partial = CustomPartial()
        self.created_env_ids = []

    def create(self, env_id):
        self.created_env_ids.append(env_id)
        return self.partial


class PolicyTrajectoryCollectorTest(unittest.TestCase):
    def test_default_partial_and_reset_hooks_across_episode_boundary(self):
        env = CountingEnv()
        reset_infos = []
        default_calls = []

        collector = PolicyTrajectoryCollector(
            model=DummyModel(),
            stats_source="stats",
            make_env=lambda env_id: env,
            env_id="Dummy-v0",
            custom_partial=None,
            model_observation=lambda stats_source, obs: (stats_source, obs.copy()),
            action_converter=lambda action_env, action: action[0],
            default_partial_reward=lambda obs, action, next_obs, reward, terminated, truncated, info: default_calls.append(
                (obs.copy(), action.copy(), next_obs.copy(), reward, terminated, truncated, dict(info))
            )
            or reward / 10.0,
            reset_reward_state=lambda info: reset_infos.append(dict(info)),
        )

        trajectories = collector.rollout_trajectories(total_timesteps=3, seed=123)

        self.assertTrue(env.closed)
        self.assertEqual([len(trajectory.states) for trajectory in trajectories], [2, 1])
        self.assertEqual([state["partial_rew"] for state in trajectories[0].states], [1.1, 1.2])
        self.assertEqual(trajectories[1].states[0]["partial_rew"], 1.3)
        self.assertEqual(reset_infos, [{"reset": 1, "seed": 123}, {"reset": 2, "seed": None}])
        self.assertEqual(len(default_calls), 3)

    def test_custom_partial_overrides_default_partial_reward(self):
        env = CountingEnv()
        partial_spec = CustomPartialSpec()

        collector = PolicyTrajectoryCollector(
            model=DummyModel(),
            stats_source=None,
            make_env=lambda env_id: env,
            env_id="Dummy-v0",
            custom_partial=partial_spec,
            model_observation=lambda _stats_source, obs: obs,
            action_converter=lambda _env, action: action[0],
            default_partial_reward=lambda *_args: self.fail("default partial should not be used"),
        )

        trajectories = collector.rollout_trajectories(total_timesteps=1, seed=5)

        self.assertEqual(partial_spec.created_env_ids, ["Dummy-v0"])
        self.assertEqual(partial_spec.partial.reset_infos, [{"reset": 1, "seed": 5}])
        self.assertEqual(trajectories[0].states[0]["partial_rew"], 7.5)

    def test_buffering_wrapper_collector_uses_vec_env_infos_and_episode_boundaries(self):
        model = VectorDummyModel()
        vec_env = DummyVecEnv([lambda: VectorCountingEnv(0.0), lambda: VectorCountingEnv(10.0)])
        collector = TrajectoryCollector(agent=model, vec_env=vec_env)

        trajectories = collector.rollout_trajectories(total_timesteps=4, seed=7)

        self.assertIsInstance(collector.buffering_wrapper, BufferingWrapper)
        self.assertEqual([len(trajectory.states) for trajectory in trajectories], [2, 2])
        self.assertEqual([state["rew"] for state in trajectories[0].states], [1.0, 2.0])
        self.assertEqual([state["partial_rew"] for state in trajectories[1].states], [0.1, 0.2])
        self.assertTrue(trajectories[0].states[-1]["done"])
        self.assertTrue(np.allclose(trajectories[1].states[-1]["obs"], np.asarray([12.0], dtype=np.float32)))
        self.assertIsNone(model._last_obs)

    def test_buffering_wrapper_collector_wraps_under_vecnormalize(self):
        model = VectorDummyModel()
        vec_env = VecNormalize(
            DummyVecEnv([lambda: VectorCountingEnv(0.0)]),
            norm_obs=True,
            norm_reward=True,
            training=True,
        )
        original_venv = vec_env.venv
        collector = TrajectoryCollector(agent=model, vec_env=vec_env)

        trajectories = collector.rollout_trajectories(total_timesteps=2, seed=11)

        self.assertIs(vec_env.venv, original_venv)
        self.assertTrue(vec_env.training)
        self.assertIsInstance(collector.buffering_wrapper, BufferingWrapper)
        self.assertEqual(len(trajectories[0].states), 2)
        self.assertTrue(np.allclose(trajectories[0].states[-1]["obs"], np.asarray([2.0], dtype=np.float32)))

    def test_buffering_wrapper_collector_returns_unfinished_rollouts(self):
        model = VectorDummyModel()
        vec_env = DummyVecEnv([lambda: NonTerminatingVectorEnv(0.0), lambda: NonTerminatingVectorEnv(10.0)])
        collector = TrajectoryCollector(agent=model, vec_env=vec_env)

        trajectories = collector.rollout_trajectories(total_timesteps=4, seed=7)

        self.assertEqual([len(trajectory.states) for trajectory in trajectories], [2, 2])
        self.assertFalse(any(trajectory.states[-1]["done"] for trajectory in trajectories))
        self.assertEqual([state["rew"] for state in trajectories[0].states], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
