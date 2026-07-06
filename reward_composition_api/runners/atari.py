from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from local_gym.classes.atari_reward_specs import AtariRewardSpec
from reward_composition_api.config import ExperimentConfig
from reward_composition_api.environments.atari import AtariEnvironmentProfile
from reward_composition_api.evaluation.atari import (
    AtariComponentEvalCallback,
    _component_keys as atari_component_keys,
    evaluate_atari_components,
    write_atari_component_summary,
)
from reward_composition_api.evaluation.reporting import RunPaths, report_eval_curve, select_final_policy
from reward_composition_api.partial_reward import include_partial_feature
from reward_composition_api.registry import PartialSpec
from reward_composition_api.results import RunResult
from reward_composition_api.training import learn_policy
from reward_composition_api.training.rlhf import RlhfTrainer

from .base import BaseExperimentRunner, make_reward_models


class AtariExperimentRunner(BaseExperimentRunner):
    def __init__(
        self,
        config: ExperimentConfig,
        spec: AtariRewardSpec | None = None,
        custom_partial: PartialSpec | None = None,
        profile: AtariEnvironmentProfile | None = None,
    ):
        self.profile = profile or AtariEnvironmentProfile()
        self.spec = spec or self.profile.reward_spec(config)
        run_name = config.run_name or self.default_run_name(config, self.spec)
        variant_name = config.variant_name or config.mode
        super().__init__(replace(config, run_name=run_name, variant_name=variant_name), custom_partial)

    @staticmethod
    def default_run_name(config: ExperimentConfig, spec: AtariRewardSpec) -> str:
        variant = config.variant_name or config.mode
        steps = f"{config.timesteps // 1_000_000}m" if config.timesteps >= 1_000_000 else f"{config.timesteps}"
        return f"{spec.slug}_{variant}_{steps}_seed{config.seed}"

    def setup(self) -> None:
        self.profile.setup(self.config)

    def build_callbacks(self, run_dir: Path, train_env: VecNormalize, eval_env: VecNormalize):
        component_callback = AtariComponentEvalCallback(
            run_dir / "eval" / "component_evaluations.csv",
            self.config.env_id,
            self.spec,
            make_env=self.profile.make_raw_env,
            partial_source=self.config.partial_source,
            custom_partial=self.custom_partial,
            eval_freq=self.eval_freq(),
            n_eval_episodes=self.config.n_eval_episodes,
            verbose=1,
        )
        return CallbackList([self.eval_callback(run_dir, train_env, eval_env), component_callback])

    def train_true_or_partial(self) -> RunResult:
        config = self.config
        run_dir = self.ensure_run_dir()

        if config.mode == "true":
            env_fn = lambda: self.profile.make_raw_env(config.env_id)
        elif config.mode == "partial":
            _, action_n = self.profile.probe_spaces(config.env_id)
            runtime = self.profile.learned_runtime(
                self.spec,
                "partial",
                action_n,
                self.custom_partial,
                partial_source=config.partial_source,
            )
            env_fn = lambda: self.profile.preference_wrapper(self.profile.make_raw_env(config.env_id), runtime)
        else:
            raise ValueError(f"Unsupported mode for this path: {config.mode}")

        train_env = self.profile.make_vecnormalize_env(env_fn, config.n_envs, run_dir / "monitor")
        eval_env = self.profile.make_eval_env(config.env_id, train_env)
        callbacks = self.build_callbacks(run_dir, train_env, eval_env)

        model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **self.profile.ppo_hyperparams(config))
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

        obs_size, action_n = self.profile.probe_spaces(config.env_id)

        runtime = self.profile.learned_runtime(
            self.spec,
            config.mode,
            action_n,
            self.custom_partial,
            partial_source=config.partial_source,
            target_mean=config.model_reward_target_mean,
            target_std=config.model_reward_target_std,
            reward_min=config.model_reward_min,
            reward_max=config.model_reward_max,
            reward_scale=config.model_reward_scale,
            normalize=config.normalize_model_reward,
            include_partial_feature=include_partial_feature(config),
        )
        train_env = self.profile.make_vecnormalize_env(
            lambda: self.profile.preference_wrapper(self.profile.make_raw_env(config.env_id), runtime),
            config.n_envs,
            run_dir / "monitor",
        )
        eval_env = self.profile.make_eval_env(config.env_id, train_env)
        callbacks = self.build_callbacks(run_dir, train_env, eval_env)
        model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **self.profile.ppo_hyperparams(config))

        reward_model = make_reward_models(obs_size + action_n + 1, config)
        convert_traj = self.profile.trajectory_converter(action_n, runtime.include_partial_feature)
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
                spec=self.spec,
                partial_source=config.partial_source,
                custom_partial=self.custom_partial,
                total_timesteps=collection_steps,
                seed=config.seed * 1000 + round_index * 100,
            ),
            collection_label="Atari steps",
        ).run()

        return self.save_and_report(model, train_env, eval_env, run_dir, synthetic_queries=total_queries, runtime=runtime)

    def partial_keys(self) -> list[str]:
        if self.custom_partial is not None:
            return [self.custom_partial.name]
        if self.config.partial_source == "life_loss":
            return ["life_loss_penalty"]
        return ["life_loss_penalty", "score_partial"]

    def save_and_report(
        self,
        model: PPO,
        train_env: VecNormalize,
        eval_env: VecNormalize,
        run_dir: Path,
        synthetic_queries: int,
        runtime=None,
    ) -> RunResult:
        config = self.config
        paths = RunPaths(run_dir)
        model.save(paths.final_model)
        train_env.save(paths.vecnormalize)

        actual_timesteps = int(model.num_timesteps)
        best_logged_reward, best_logged_timestep = report_eval_curve(
            paths.eval_log,
            paths.true_reward_curve,
            max(config.timesteps, actual_timesteps),
            config.plot_mode,
            config.smooth_window,
            x_scale=1e6,
            x_label="Timesteps (millions)",
            y_floor=None,
        )

        final_stats = evaluate_atari_components(
            model,
            config.env_id,
            self.spec,
            make_env=self.profile.make_raw_env,
            partial_source=config.partial_source,
            custom_partial=self.custom_partial,
            stats_source=train_env,
            n_eval_episodes=config.final_eval_episodes,
            seed=config.seed + 50_000,
        )
        write_atari_component_summary(paths.final_component_evaluation, actual_timesteps, final_stats, self.custom_partial)

        final_policy, final_eval_env = select_final_policy(
            config,
            model,
            eval_env,
            run_dir,
            self.profile.load_eval_env,
            PPO.load,
            load_best_stats=True,
        )

        mean_reward, std_reward = evaluate_policy(
            final_policy,
            final_eval_env,
            n_eval_episodes=config.final_eval_episodes,
            deterministic=True,
            return_episode_rewards=False,
        )
        selected_stats = evaluate_atari_components(
            final_policy,
            config.env_id,
            self.spec,
            make_env=self.profile.make_raw_env,
            partial_source=config.partial_source,
            custom_partial=self.custom_partial,
            stats_source=final_eval_env,
            n_eval_episodes=config.final_eval_episodes,
            seed=config.seed + 60_000,
        )

        metadata = {
            **self.common_metadata(actual_timesteps, synthetic_queries, best_logged_reward, best_logged_timestep),
            "env_slug": self.spec.slug,
            "obs_type": "ram",
            "frameskip": 4,
            "repeat_action_probability": 0.25,
            "fire_reset": True,
            "auto_fire_after_life_loss": True,
            "action_encoding": "one_hot",
            "partial_source": config.partial_source,
            "partial_keys": self.partial_keys(),
            "component_keys": list(atari_component_keys(self.custom_partial)),
            "life_loss_penalty_weight": self.spec.life_loss_penalty,
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
            vecnormalize_path=paths.vecnormalize,
            synthetic_queries=synthetic_queries,
            metadata=metadata,
        )

    def print_summary(self, mean_reward: float, std_reward: float, selected_stats: dict, synthetic_queries: int) -> None:
        print(f"{self.config.final_policy.title()} deterministic true reward: {mean_reward:.3f} +/- {std_reward:.3f}")
        print(
            "Component means: "
            f"total={selected_stats['mean_total']:.3f}, "
            f"partial={selected_stats['mean_partial']:.3f}, "
            f"residual={selected_stats['mean_residual']:.3f}, "
            f"lost_lives={selected_stats.get('mean_lost_lives', 0.0):.3f}"
        )
        print(f"Synthetic queries consumed: {synthetic_queries}")
        print(f"Saved model and logs to {self.run_dir}")


def run_atari_experiment(config: ExperimentConfig) -> RunResult:
    return AtariExperimentRunner(config).run()
