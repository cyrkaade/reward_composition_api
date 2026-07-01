from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from reward_composition_api.config import BOX2D_SUITE, ExperimentConfig
from reward_composition_api.environments.box2d_env import Box2DEnvironmentProfile
from reward_composition_api.environments.gymnasium_env import GymnasiumEnvironmentProfile
from reward_composition_api.evaluation.gymnasium import (
    GymComponentEvalCallback,
    component_keys as gym_component_keys,
    evaluate_gym_components,
    write_gym_component_summary,
)
from reward_composition_api.evaluation.reporting import RunPaths, report_eval_curve, select_final_policy
from reward_composition_api.partial_reward import include_partial_feature
from reward_composition_api.registry import PartialSpec
from reward_composition_api.results import RunResult
from reward_composition_api.training import learn_policy
from reward_composition_api.training.rlhf import RlhfTrainer

from .base import BaseExperimentRunner, make_reward_models


class GymExperimentRunner(BaseExperimentRunner):
    def __init__(
        self,
        config: ExperimentConfig,
        custom_partial: PartialSpec | None = None,
        profile: GymnasiumEnvironmentProfile | None = None,
    ):
        self.profile = profile or self.default_profile(config)
        run_name = config.run_name or self.default_run_name(config)
        variant_name = config.variant_name or config.mode
        super().__init__(replace(config, run_name=run_name, variant_name=variant_name), custom_partial)

    @staticmethod
    def default_profile(config: ExperimentConfig):
        if config.suite == BOX2D_SUITE or str(config.env_id).startswith(("LunarLander", "BipedalWalker", "CarRacing")):
            return Box2DEnvironmentProfile()
        return GymnasiumEnvironmentProfile()

    @staticmethod
    def slugify(env_id: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in env_id.rsplit("-", 1)[0]).strip("_")

    @classmethod
    def default_run_name(cls, config: ExperimentConfig) -> str:
        variant = config.variant_name or config.mode
        steps = f"{config.timesteps // 1_000_000}m" if config.timesteps >= 1_000_000 else f"{config.timesteps}"
        return f"{cls.slugify(config.env_id)}_{variant}_{steps}_seed{config.seed}"

    def build_callbacks(self, run_dir: Path, train_env, eval_env):
        component_callback = GymComponentEvalCallback(
            run_dir / "eval" / "component_evaluations.csv",
            self.config.env_id,
            make_env=self.profile.make_raw_env,
            custom_partial=self.custom_partial,
            eval_freq=self.eval_freq(),
            n_eval_episodes=self.config.n_eval_episodes,
            verbose=1,
        )
        return CallbackList([self.eval_callback(run_dir, train_env, eval_env), component_callback])

    def train_true_or_partial(self) -> RunResult:
        config = self.config
        run_dir = self.ensure_run_dir()
        probe_env = self.profile.make_raw_env(config.env_id)
        normalize = self.profile.should_normalize_observation(probe_env.observation_space)
        hyperparams = self.profile.ppo_hyperparams(probe_env, config)
        observation_space = probe_env.observation_space
        action_space = probe_env.action_space
        probe_env.close()

        if config.mode == "true":
            env_fn = lambda: self.profile.make_raw_env(config.env_id)
        elif config.mode == "partial":
            runtime = self.profile.learned_runtime(
                config.env_id,
                "partial",
                observation_space,
                action_space,
                self.custom_partial,
            )
            env_fn = lambda: self.profile.preference_wrapper(self.profile.make_raw_env(config.env_id), runtime)
        else:
            raise ValueError(f"Unsupported mode for this path: {config.mode}")

        train_env = self.profile.make_train_env(env_fn, config.n_envs, run_dir / "monitor", normalize)
        eval_env = self.profile.make_eval_env(config.env_id, train_env)
        callbacks = self.build_callbacks(run_dir, train_env, eval_env)
        model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **hyperparams)
        learn_policy(
            model,
            config.timesteps,
            callbacks,
            progress_bar=config.progress_bar,
            log_interval=config.policy_log_interval,
        )

        return self.save_and_report(model, train_env, eval_env, run_dir, synthetic_queries=0)

    def train_preference_mode(self) -> RunResult:
        config = self.config
        run_dir = self.ensure_run_dir()
        probe_env = self.profile.make_raw_env(config.env_id)
        normalize = self.profile.should_normalize_observation(probe_env.observation_space)
        hyperparams = self.profile.ppo_hyperparams(probe_env, config)
        observation_space = probe_env.observation_space
        action_space = probe_env.action_space
        reward_model_input_size = self.profile.reward_model_input_size(observation_space, action_space)
        probe_env.close()

        runtime = self.profile.learned_runtime(
            config.env_id,
            config.mode,
            observation_space,
            action_space,
            self.custom_partial,
            target_mean=config.model_reward_target_mean,
            target_std=config.model_reward_target_std,
            reward_min=config.model_reward_min,
            reward_max=config.model_reward_max,
            reward_scale=config.model_reward_scale,
            normalize=config.normalize_model_reward,
            include_partial_feature=include_partial_feature(config),
        )
        train_env = self.profile.make_train_env(
            lambda: self.profile.preference_wrapper(self.profile.make_raw_env(config.env_id), runtime),
            config.n_envs,
            run_dir / "monitor",
            normalize,
        )
        eval_env = self.profile.make_eval_env(config.env_id, train_env)
        callbacks = self.build_callbacks(run_dir, train_env, eval_env)
        model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **hyperparams)

        reward_model = make_reward_models(reward_model_input_size, config)
        convert_traj = self.profile.trajectory_converter(observation_space, action_space, runtime.include_partial_feature)
        total_queries = RlhfTrainer(
            config,
            model,
            runtime,
            callbacks,
            reward_model,
            convert_traj,
            lambda round_index, collection_steps: self.profile.collect_policy_trajectories(
                model,
                train_env,
                env_id=config.env_id,
                custom_partial=self.custom_partial,
                total_timesteps=collection_steps,
                seed=config.seed * 1000 + round_index * 100,
            ),
            collection_label="Gym steps",
        ).run()

        return self.save_and_report(model, train_env, eval_env, run_dir, synthetic_queries=total_queries, runtime=runtime)

    def save_and_report(
        self,
        model: PPO,
        train_env,
        eval_env,
        run_dir: Path,
        synthetic_queries: int,
        runtime=None,
    ) -> RunResult:
        config = self.config
        paths = RunPaths(run_dir)
        model.save(paths.final_model)
        vecnormalize_path = None
        if isinstance(train_env, VecNormalize):
            train_env.save(paths.vecnormalize)
            vecnormalize_path = paths.vecnormalize

        actual_timesteps = int(model.num_timesteps)
        best_logged_reward, best_logged_timestep = report_eval_curve(
            paths.eval_log,
            paths.true_reward_curve,
            max(config.timesteps, actual_timesteps),
            config.plot_mode,
            config.smooth_window,
            x_scale=1e6,
            x_label="Timesteps (millions)",
        )

        final_stats = evaluate_gym_components(
            model,
            config.env_id,
            make_env=self.profile.make_raw_env,
            custom_partial=self.custom_partial,
            stats_source=train_env,
            n_eval_episodes=config.final_eval_episodes,
            seed=config.seed + 50_000,
        )
        write_gym_component_summary(paths.final_component_evaluation, actual_timesteps, final_stats, self.custom_partial)

        final_policy, final_eval_env = select_final_policy(
            config,
            model,
            eval_env,
            run_dir,
            self.profile.load_eval_env,
            PPO.load,
            load_best_stats=isinstance(train_env, VecNormalize),
        )

        mean_reward, std_reward = evaluate_policy(
            final_policy,
            final_eval_env,
            n_eval_episodes=config.final_eval_episodes,
            deterministic=True,
            return_episode_rewards=False,
        )
        selected_stats = evaluate_gym_components(
            final_policy,
            config.env_id,
            make_env=self.profile.make_raw_env,
            custom_partial=self.custom_partial,
            stats_source=final_eval_env,
            n_eval_episodes=config.final_eval_episodes,
            seed=config.seed + 60_000,
        )

        metadata = {
            **self.common_metadata(actual_timesteps, synthetic_queries, best_logged_reward, best_logged_timestep),
            "env_slug": self.slugify(config.env_id),
            "partial_keys": [self.custom_partial.name] if self.custom_partial else [],
            "component_keys": list(gym_component_keys(self.custom_partial)),
            "selected_policy_true_reward_mean": float(mean_reward),
            "selected_policy_true_reward_std": float(std_reward),
            "selected_policy_components": selected_stats,
            **self.runtime_metadata(runtime),
        }

        metadata_path = self.write_metadata(paths, metadata)
        self.print_summary(float(mean_reward), float(std_reward), selected_stats, synthetic_queries)

        train_env.close()
        final_eval_env.close()
        return RunResult(
            run_dir=run_dir,
            metadata_path=metadata_path,
            model_path=paths.final_model.with_suffix(".zip"),
            vecnormalize_path=vecnormalize_path,
            synthetic_queries=synthetic_queries,
            metadata=metadata,
        )




def run_gym_experiment(config: ExperimentConfig) -> RunResult:
    return GymExperimentRunner(config).run()
