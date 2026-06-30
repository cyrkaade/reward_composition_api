from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from local_gym.classes.atari_reward_specs import AtariRewardSpec, get_atari_reward_spec
from reward_composition_api.config import ExperimentConfig
from reward_composition_api.registry import PartialSpec
from reward_composition_api.wrappers.trajectory_buffering import Trajectory

from reward_composition_api.backend import atari_env as plumbing


class AtariEnvironmentProfile:
    collection_label = "Atari steps"
    continuous_actions = False

    def setup(self, config: ExperimentConfig) -> None:
        plumbing.register_atari_envs()
        random.seed(config.seed)
        np.random.seed(config.seed)
        th.manual_seed(config.seed)

    def reward_spec(self, config: ExperimentConfig) -> AtariRewardSpec:
        return get_atari_reward_spec(config.env_id)

    def make_raw_env(self, env_id: str):
        return plumbing.make_raw_env(env_id)

    def make_vecnormalize_env(self, env_fn, n_envs: int, monitor_dir: Path) -> VecNormalize:
        return plumbing.make_vecnormalize_env(env_fn, n_envs, monitor_dir)

    def make_eval_env(self, env_id: str, stats_source: VecNormalize | None = None) -> VecNormalize:
        return plumbing.make_eval_env(env_id, stats_source)

    def load_eval_env(self, env_id: str, stats_path: Path) -> VecNormalize:
        return plumbing.load_eval_env(env_id, stats_path)

    def ppo_hyperparams(self, config: ExperimentConfig):
        return plumbing.ppo_hyperparams(config)

    def learned_runtime(self, spec: AtariRewardSpec, composition: str, action_n: int, custom_partial: PartialSpec | None, **kwargs):
        return plumbing.AtariLearnedRewardRuntime(
            spec=spec,
            composition=composition,
            action_n=action_n,
            partial_source=kwargs.pop("partial_source", "life_loss"),
            custom_partial=custom_partial,
            **kwargs,
        )

    def preference_wrapper(self, env, runtime):
        return plumbing.AtariPreferenceRewardWrapper(env, runtime)

    def probe_spaces(self, env_id: str) -> tuple[int, int]:
        probe_env = self.make_raw_env(env_id)
        obs_size = int(np.prod(probe_env.observation_space.shape))
        action_n = int(probe_env.action_space.n)
        probe_env.close()
        return obs_size, action_n

    def trajectory_converter(self, action_n: int, include_partial_feature: bool):
        return plumbing.make_trajectory_converter(action_n, include_partial_feature)

    def collect_policy_trajectories(
        self,
        model: PPO,
        stats_source,
        env_id: str,
        spec: AtariRewardSpec,
        partial_source: str,
        custom_partial: PartialSpec | None,
        total_timesteps: int,
        seed: int,
    ) -> list[Trajectory]:
        return plumbing.collect_policy_trajectories(
            model,
            stats_source,
            env_id=env_id,
            spec=spec,
            partial_source=partial_source,
            custom_partial=custom_partial,
            total_timesteps=total_timesteps,
            seed=seed,
        )
