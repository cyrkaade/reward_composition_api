from __future__ import annotations

import unittest

import numpy as np
from gymnasium import spaces

from reward_composition_api.evaluation.component_evaluator import evaluate_policy_components


class DummyModel:
    def predict(self, observation, deterministic=True):
        return np.asarray([[1.0]], dtype=np.float32), None


class TwoStepEnv:
    observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self):
        self.step_count = 0
        self.closed = False

    def reset(self, seed=None):
        self.step_count = 0
        return np.asarray([0.0], dtype=np.float32), {"seed": seed}

    def step(self, action):
        self.step_count += 1
        done = self.step_count == 2
        return (
            np.asarray([float(self.step_count)], dtype=np.float32),
            float(self.step_count * 10),
            done,
            False,
            {"partial": float(self.step_count), "bonus": float(self.step_count + 1)},
        )

    def close(self):
        self.closed = True


class CustomPartial:
    component_keys = ("custom",)

    def __init__(self):
        self.reset_infos = []

    def reset(self, info):
        self.reset_infos.append(dict(info))

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        return type("PartialStep", (), {"partial": 3.0, "components": {"custom": 2.0}})()


class CustomPartialSpec:
    component_keys = ("custom",)

    def __init__(self):
        self.partial = CustomPartial()

    def create(self, env_id):
        return self.partial


class ComponentEvaluatorTest(unittest.TestCase):
    def test_evaluates_default_partial_components(self):
        env = TwoStepEnv()

        stats = evaluate_policy_components(
            model=DummyModel(),
            env_id="Dummy-v0",
            make_env=lambda env_id: env,
            custom_partial=None,
            stats_source=None,
            n_eval_episodes=1,
            seed=7,
            deterministic=True,
            component_keys=("bonus",),
            model_observation=lambda _stats_source, obs: obs,
            action_converter=lambda _env, action: action[0],
            default_partial_step=lambda _obs, _action, _new_obs, _reward, _terminated, _truncated, info: (
                info["partial"],
                {"bonus": info["bonus"]},
            ),
        )

        self.assertTrue(env.closed)
        self.assertEqual(stats["mean_total"], 30.0)
        self.assertEqual(stats["mean_partial"], 3.0)
        self.assertEqual(stats["mean_residual"], 27.0)
        self.assertEqual(stats["mean_bonus"], 5.0)
        self.assertEqual(stats["mean_length"], 2.0)

    def test_custom_partial_overrides_default_partial_step(self):
        partial_spec = CustomPartialSpec()

        stats = evaluate_policy_components(
            model=DummyModel(),
            env_id="Dummy-v0",
            make_env=lambda env_id: TwoStepEnv(),
            custom_partial=partial_spec,
            stats_source=None,
            n_eval_episodes=1,
            seed=11,
            deterministic=True,
            component_keys=partial_spec.component_keys,
            model_observation=lambda _stats_source, obs: obs,
            action_converter=lambda _env, action: action[0],
            default_partial_step=lambda *_args: self.fail("default partial should not be used"),
        )

        self.assertEqual(partial_spec.partial.reset_infos, [{"seed": 11}])
        self.assertEqual(stats["mean_partial"], 6.0)
        self.assertEqual(stats["mean_custom"], 4.0)


if __name__ == "__main__":
    unittest.main()
