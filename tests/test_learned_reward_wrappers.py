from __future__ import annotations

import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from local_gym.wrappers.lunar_lander_rewards_wrapper import LunarLanderSaveInfo
from local_gym.classes.mujoco_reward_specs import MuJoCoRewardSpec
from reward_composition_api.backend.atari import run_atari_experiment
from reward_composition_api.backend.atari_env import AtariLearnedRewardRuntime, AtariPreferenceRewardWrapper
from reward_composition_api.backend.gym_env import make_raw_env as make_gym_raw_env
from reward_composition_api.backend.gym_env import GymLearnedRewardRuntime, GymPreferenceRewardWrapper
from reward_composition_api.backend.gymnasium import run_gym_experiment
from reward_composition_api.backend.mujoco import run_mujoco_experiment
from reward_composition_api.backend.mujoco_env import MuJoCoLearnedRewardRuntime, MuJoCoPreferenceRewardWrapper


class ConstantRewardModel(th.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return th.full((x.shape[0], 1), self.value, dtype=th.float32)


class OneStepEnv(gym.Env):
    metadata = {}

    def __init__(self, initial_info=None, step_info=None, reward: float = 5.0):
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.initial_info = initial_info or {}
        self.step_info = step_info or {}
        self.reward = reward

    def reset(self, **kwargs):
        return np.asarray([0.0, 0.5], dtype=np.float32), dict(self.initial_info)

    def step(self, action):
        return np.asarray([1.0, 2.0], dtype=np.float32), self.reward, True, False, dict(self.step_info)


class DiscreteOneStepEnv(OneStepEnv):
    def __init__(self, initial_info=None, step_info=None, reward: float = 5.0):
        super().__init__(initial_info, step_info, reward)
        self.action_space = spaces.Discrete(3)


class DummyLander:
    awake = True


class DummyLunarEnv(DiscreteOneStepEnv):
    @property
    def unwrapped(self):
        return self

    @property
    def lander(self):
        return DummyLander()

    @property
    def game_over(self):
        return True


class LearnedRewardWrapperTest(unittest.TestCase):
    def test_mujoco_wrapper_reward_composition_modes(self):
        spec = MuJoCoRewardSpec(
            env_id="DummyMuJoCo-v0",
            slug="dummy",
            partial_keys=("reward_dist",),
            component_keys=("reward_dist", "reward_ctrl"),
        )

        for mode, expected_reward in {
            "partial": 2.5,
            "feedback": 0.0,
            "naive": 2.5,
            "delta": 2.5,
        }.items():
            with self.subTest(mode=mode):
                runtime = MuJoCoLearnedRewardRuntime(spec=spec, composition=mode)
                wrapper = MuJoCoPreferenceRewardWrapper(
                    OneStepEnv(step_info={"reward_dist": 2.5, "reward_ctrl": -0.25}),
                    runtime,
                )

                wrapper.reset()
                _, reward, terminated, truncated, info = wrapper.step(np.asarray([0.0], dtype=np.float32))

                self.assertEqual(reward, expected_reward)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertEqual(info["true_reward"], 5.0)
                self.assertEqual(info["partial_reward"], 2.5)
                self.assertEqual(info["model_reward"], 0.0)
                self.assertEqual(info["learned_reward"], expected_reward)

    def test_mujoco_model_reward_normalizes_scales_and_clips(self):
        spec = MuJoCoRewardSpec(
            env_id="DummyMuJoCo-v0",
            slug="dummy",
            partial_keys=("reward_dist",),
            component_keys=("reward_dist",),
        )
        runtime = MuJoCoLearnedRewardRuntime(
            spec=spec,
            composition="feedback",
            reward_model=ConstantRewardModel(3.0),
            output_mean=1.0,
            output_std=2.0,
            target_mean=10.0,
            target_std=4.0,
            reward_scale=0.5,
            reward_min=-10.0,
            reward_max=6.0,
            normalize=True,
        )
        wrapper = MuJoCoPreferenceRewardWrapper(OneStepEnv(step_info={"reward_dist": 2.5}), runtime)

        wrapper.reset()
        _, reward, _, _, info = wrapper.step(np.asarray([0.0], dtype=np.float32))

        self.assertEqual(reward, 6.0)
        self.assertEqual(info["model_reward"], 6.0)

    def test_mujoco_model_reward_averages_ensemble_outputs(self):
        spec = MuJoCoRewardSpec(
            env_id="DummyMuJoCo-v0",
            slug="dummy",
            partial_keys=("reward_dist",),
            component_keys=("reward_dist",),
        )
        runtime = MuJoCoLearnedRewardRuntime(
            spec=spec,
            composition="feedback",
            reward_models=[ConstantRewardModel(1.0), ConstantRewardModel(3.0)],
        )
        wrapper = MuJoCoPreferenceRewardWrapper(OneStepEnv(step_info={"reward_dist": 2.5}), runtime)

        wrapper.reset()
        _, reward, _, _, info = wrapper.step(np.asarray([0.0], dtype=np.float32))

        self.assertEqual(reward, 2.0)
        self.assertEqual(info["model_reward"], 2.0)

    def test_gym_and_atari_wrapper_names_remain_importable(self):
        self.assertTrue(callable(run_mujoco_experiment))
        self.assertTrue(callable(run_atari_experiment))
        self.assertTrue(callable(run_gym_experiment))
        self.assertTrue(callable(MuJoCoPreferenceRewardWrapper))
        self.assertTrue(callable(GymPreferenceRewardWrapper))
        self.assertTrue(callable(AtariPreferenceRewardWrapper))

    def test_gym_feedback_without_model_returns_zero(self):
        runtime = GymLearnedRewardRuntime(
            env_id="DummyGym-v0",
            composition="feedback",
            observation_space=spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
            action_space=spaces.Discrete(3),
        )
        wrapper = GymPreferenceRewardWrapper(DiscreteOneStepEnv(), runtime)

        wrapper.reset()
        _, reward, _, _, info = wrapper.step(1)

        self.assertEqual(reward, 0.0)
        self.assertEqual(info["model_reward"], 0.0)
        self.assertEqual(info["partial_reward"], 0.0)

    def test_atari_partial_mode_updates_reset_and_step_info(self):
        spec = type(
            "Spec",
            (),
            {
                "env_id": "DummyAtari-v0",
                "new_tracker": lambda self: Tracker(),
            },
        )()
        runtime = AtariLearnedRewardRuntime(spec=spec, composition="partial", action_n=3)
        wrapper = AtariPreferenceRewardWrapper(DiscreteOneStepEnv(initial_info={"lives": 2}, step_info={"lives": 1}), runtime)

        _, reset_info = wrapper.reset()
        _, reward, _, _, step_info = wrapper.step(1)

        self.assertEqual(reset_info["partial_reward"], 0.0)
        self.assertEqual(reward, -1.0)
        self.assertEqual(step_info["partial_reward"], -1.0)
        self.assertEqual(step_info["learned_reward"], -1.0)

    def test_lunar_lander_save_info_exposes_terminal_flags(self):
        wrapper = LunarLanderSaveInfo(DummyLunarEnv())

        _, _, _, _, info = wrapper.step(1)

        self.assertTrue(info["game_over"])
        self.assertTrue(info["awake"])

    def test_gym_raw_env_only_wraps_lunar_lander_with_save_info(self):
        with patch("reward_composition_api.backend.gym_env.gym.make", return_value=DummyLunarEnv()):
            lunar_env = make_gym_raw_env("LunarLander-v3")

        with patch("reward_composition_api.backend.gym_env.gym.make", return_value=DiscreteOneStepEnv()):
            cartpole_env = make_gym_raw_env("CartPole-v1")

        self.assertIsInstance(lunar_env, LunarLanderSaveInfo)
        self.assertNotIsInstance(cartpole_env, LunarLanderSaveInfo)


class Tracker:
    def reset(self, info):
        return TrackerStep(partial=0.0, lost_lives=0.0, lives=float(info.get("lives", 0.0)))

    def step(self, info, true_reward=0.0, partial_source="life_loss"):
        return TrackerStep(partial=-1.0, lost_lives=1.0, lives=float(info.get("lives", 0.0)))


class TrackerStep:
    def __init__(self, partial: float, lost_lives: float, lives: float):
        self.partial = partial
        self.life_loss_penalty = partial
        self.score_partial = 0.0
        self.lost_lives = lost_lives
        self.lives = lives

    def as_info(self):
        return {
            "partial_reward": self.partial,
            "life_loss_penalty": self.life_loss_penalty,
            "score_partial": self.score_partial,
            "lost_lives": self.lost_lives,
            "lives": self.lives,
        }


if __name__ == "__main__":
    unittest.main()
