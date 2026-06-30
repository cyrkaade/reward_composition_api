from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from local_gym.classes.mujoco_reward_specs import MuJoCoRewardSpec, get_mujoco_reward_spec
from reward_composition_api.config import ExperimentConfig
from reward_composition_api.registry import PartialSpec
from reward_composition_api.wrappers.trajectory_buffering import Trajectory

from reward_composition_api.backend import mujoco_env as plumbing


class MuJoCoEnvironmentProfile:
    collection_label = "steps"
    continuous_actions = True

    def reward_spec(self, config: ExperimentConfig) -> MuJoCoRewardSpec:
        return get_mujoco_reward_spec(config.env_id).with_partial_profile(config.partial_profile)

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

    def learned_runtime(self, spec: MuJoCoRewardSpec, composition: str, custom_partial: PartialSpec | None, **kwargs):
        return plumbing.MuJoCoLearnedRewardRuntime(
            spec=spec,
            composition=composition,
            custom_partial=custom_partial,
            **kwargs,
        )

    def preference_wrapper(self, env, runtime):
        return plumbing.MuJoCoPreferenceRewardWrapper(env, runtime)

    def reward_model_input_size(self, env_id: str) -> int:
        probe_env = self.make_raw_env(env_id)
        action_shape = probe_env.action_space.shape
        input_size = probe_env.observation_space.shape[0] + action_shape[0] + 1
        probe_env.close()
        return input_size

    def trajectory_converter(self, include_partial_feature: bool):
        return plumbing.make_trajectory_converter(include_partial_feature)

    def collect_policy_trajectories(
        self,
        model: PPO,
        stats_source,
        env_id: str,
        spec: MuJoCoRewardSpec,
        custom_partial: PartialSpec | None,
        total_timesteps: int,
        seed: int,
    ) -> list[Trajectory]:
        return plumbing.collect_policy_trajectories(
            model,
            stats_source,
            env_id=env_id,
            spec=spec,
            custom_partial=custom_partial,
            total_timesteps=total_timesteps,
            seed=seed,
        )
