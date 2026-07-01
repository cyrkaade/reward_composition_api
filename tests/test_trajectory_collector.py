from __future__ import annotations

import unittest

import numpy as np
from gymnasium import spaces

from reward_composition_api.environments.trajectory_collector import PolicyTrajectoryCollector


class DummyModel:
    def __init__(self):
        self.observations = []

    def predict(self, observation, deterministic=False):
        self.observations.append((observation, deterministic))
        return np.asarray([[1.0]], dtype=np.float32), None


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


if __name__ == "__main__":
    unittest.main()
