from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from local_gym.classes.atari_reward_specs import AtariRewardSpec
from local_gym.classes.mujoco_reward_specs import MuJoCoRewardSpec
from reward_composition_api.config import BOX2D_SUITE, ExperimentConfig
from reward_composition_api.environments.atari import AtariEnvironmentProfile
from reward_composition_api.environments.box2d_env import Box2DEnvironmentProfile
from reward_composition_api.environments.gymnasium_env import GymnasiumEnvironmentProfile
from reward_composition_api.environments.mujoco import MuJoCoEnvironmentProfile
from reward_composition_api.evaluation.atari import (
    AtariComponentEvalCallback,
    _component_keys as atari_component_keys,
    evaluate_atari_components,
    write_atari_component_summary,
)
from reward_composition_api.evaluation.gymnasium import (
    GymComponentEvalCallback,
    component_keys as gym_component_keys,
    evaluate_gym_components,
    write_gym_component_summary,
)
from reward_composition_api.evaluation.mujoco import (
    MuJoCoComponentEvalCallback,
    _component_keys as mujoco_component_keys,
    evaluate_mujoco_components,
    write_mujoco_component_summary,
)
from reward_composition_api.evaluation.reporting import (
    BackendRunPaths,
    report_eval_curve,
    select_final_policy,
)
from reward_composition_api.partial_reward import include_partial_feature, resolve_custom_partial
from reward_composition_api.registry import PartialSpec
from reward_composition_api.results import RunResult
from reward_composition_api.training import SaveVecNormalizeOnBest, learn_policy
from reward_model.reward_model import RewardModel

from .rlhf import RlhfTrainer


def make_reward_models(input_size: int, config: ExperimentConfig) -> RewardModel | list[RewardModel]:
    models = [
        RewardModel(input_size=input_size, hidden_sizes=config.reward_hidden_sizes)
        for _ in range(config.reward_model_ensemble_size)
    ]
    return models[0] if len(models) == 1 else models


class BaseExperimentRunner:
    preference_modes = {"feedback", "naive", "delta"}
    direct_modes = {"true", "partial"}

    def __init__(self, config: ExperimentConfig, custom_partial: PartialSpec | None = None):
        self.config = config
        self.custom_partial = custom_partial if custom_partial is not None else resolve_custom_partial(config)

    @property
    def run_dir(self) -> Path:
        return Path(self.config.log_dir) / self.config.run_name

    def ensure_run_dir(self) -> Path:
        run_dir = self.run_dir
        run_dir.mkdir(exist_ok=True, parents=True)
        return run_dir

    def eval_freq(self) -> int:
        return max(self.config.eval_freq // self.config.n_envs, 1)

    def best_callbacks(self, train_env, best_stats_path: Path) -> list[BaseCallback]:
        callbacks: list[BaseCallback] = []
        if isinstance(train_env, VecNormalize):
            callbacks.append(SaveVecNormalizeOnBest(train_env, best_stats_path))
        if self.config.stop_reward is not None:
            callbacks.append(StopTrainingOnRewardThreshold(reward_threshold=self.config.stop_reward, verbose=1))
        return callbacks

    def eval_callback(self, run_dir: Path, train_env, eval_env):
        best_callbacks = self.best_callbacks(train_env, run_dir / "best_model" / "best_vecnormalize.pkl")
        return EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(run_dir / "eval"),
            eval_freq=self.eval_freq(),
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False,
            callback_on_new_best=CallbackList(best_callbacks) if best_callbacks else None,
        )

    def common_metadata(
        self,
        actual_timesteps: int,
        synthetic_queries: int,
        best_logged_reward: float | None,
        best_logged_timestep: int | None,
    ) -> dict:
        config = self.config
        return {
            "env_id": config.env_id,
            "mode": config.mode,
            "run_name": config.run_name,
            "variant": config.variant_name,
            "requested_timesteps": config.timesteps,
            "actual_timesteps": actual_timesteps,
            "seed": config.seed,
            "n_envs": config.n_envs,
            "initial_timesteps": config.initial_timesteps,
            "policy_timesteps_per_round": config.policy_timesteps_per_round,
            "final_policy_timesteps": config.final_policy_timesteps,
            "policy_learning_kwargs": config.policy_learning_kwargs or {},
            "synthetic_queries": synthetic_queries,
            "query_budget": config.query_budget if config.mode in self.preference_modes else 0,
            "fragment_length": config.fragment_length if config.mode in self.preference_modes else None,
            "active_learning": config.active_learning if config.mode in self.preference_modes else None,
            "active_query_strategy": config.active_query_strategy if config.mode in self.preference_modes else None,
            "reward_hidden_sizes": list(config.reward_hidden_sizes),
            "reward_model_lr": config.reward_model_lr if config.mode in self.preference_modes else None,
            "reward_model_ensemble_size": config.reward_model_ensemble_size if config.mode in self.preference_modes else None,
            "pretrain_reward_model": config.pretrain_reward_model if config.mode in self.preference_modes else None,
            "pretrain_target": config.pretrain_target if config.pretrain_reward_model else None,
            "include_partial_feature": include_partial_feature(config) if config.mode in self.preference_modes else None,
            "partial_reference": config.partial,
            "best_logged_true_reward": best_logged_reward,
            "best_logged_timestep": best_logged_timestep,
        }

    def runtime_metadata(self, runtime) -> dict:
        if runtime is None:
            return {}
        return {
            "model_reward_min": runtime.reward_min,
            "model_reward_max": runtime.reward_max,
            "model_reward_scale": runtime.reward_scale,
            "normalize_model_reward": runtime.normalize,
            "model_reward_output_mean": runtime.output_mean,
            "model_reward_output_std": runtime.output_std,
            "model_reward_target_mean": runtime.target_mean,
            "model_reward_target_std": runtime.target_std,
            "reward_composition": runtime.composition,
        }

    def write_metadata(self, paths: BackendRunPaths, metadata: dict) -> Path:
        metadata_path = paths.metadata
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata_path

    def print_summary(self, mean_reward: float, std_reward: float, selected_stats: dict, synthetic_queries: int) -> None:
        print(f"{self.config.final_policy.title()} deterministic true reward: {mean_reward:.3f} +/- {std_reward:.3f}")
        print(
            "Component means: "
            f"total={selected_stats['mean_total']:.3f}, "
            f"partial={selected_stats['mean_partial']:.3f}, "
            f"residual={selected_stats['mean_residual']:.3f}"
        )
        print(f"Synthetic queries consumed: {synthetic_queries}")
        print(f"Saved model and logs to {self.run_dir}")

    def setup(self) -> None:
        return None

    def run(self) -> RunResult:
        self.setup()
        if self.config.mode in self.direct_modes:
            return self.train_true_or_partial()
        return self.train_preference_mode()

    def train_true_or_partial(self) -> RunResult:
        raise NotImplementedError

    def train_preference_mode(self) -> RunResult:
        raise NotImplementedError


class MuJoCoExperimentRunner(BaseExperimentRunner):
    def __init__(
        self,
        config: ExperimentConfig,
        spec: MuJoCoRewardSpec | None = None,
        custom_partial: PartialSpec | None = None,
        profile: MuJoCoEnvironmentProfile | None = None,
    ):
        self.profile = profile or MuJoCoEnvironmentProfile()
        self.spec = spec or self.profile.reward_spec(config)
        run_name = config.run_name or self.default_run_name(config, self.spec)
        variant_name = config.variant_name or config.mode
        super().__init__(replace(config, run_name=run_name, variant_name=variant_name), custom_partial)

    @staticmethod
    def default_run_name(config: ExperimentConfig, spec: MuJoCoRewardSpec) -> str:
        variant = config.variant_name or config.mode
        steps = f"{config.timesteps // 1_000_000}m" if config.timesteps >= 1_000_000 else f"{config.timesteps}"
        return f"{spec.slug}_{variant}_{steps}_seed{config.seed}"

    def build_callbacks(self, run_dir: Path, train_env: VecNormalize, eval_env: VecNormalize):
        component_callback = MuJoCoComponentEvalCallback(
            run_dir / "eval" / "component_evaluations.csv",
            self.config.env_id,
            self.spec,
            custom_partial=self.custom_partial,
            eval_freq=self.eval_freq(),
            n_eval_episodes=self.config.n_eval_episodes,
            verbose=1,
        )
        return CallbackList([self.eval_callback(run_dir, train_env, eval_env), component_callback])

    def train_true_or_partial(self) -> RunResult:
        config = self.config
        run_dir = self.ensure_run_dir()
        hyperparams = self.profile.ppo_hyperparams(config)

        if config.mode == "true":
            env_fn = lambda: self.profile.make_raw_env(config.env_id)
        elif config.mode == "partial":
            runtime = self.profile.learned_runtime(self.spec, "partial", self.custom_partial)
            env_fn = lambda: self.profile.preference_wrapper(self.profile.make_raw_env(config.env_id), runtime)
        else:
            raise ValueError(f"Unsupported mode for this path: {config.mode}")

        train_env = self.profile.make_vecnormalize_env(env_fn, config.n_envs, run_dir / "monitor")
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
        hyperparams = self.profile.ppo_hyperparams(config)

        runtime = self.profile.learned_runtime(
            self.spec,
            config.mode,
            self.custom_partial,
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
        model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **hyperparams)

        input_size = self.profile.reward_model_input_size(config.env_id)
        reward_model = make_reward_models(input_size, config)
        convert_traj = self.profile.trajectory_converter(runtime.include_partial_feature)
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
                custom_partial=self.custom_partial,
                total_timesteps=collection_steps,
                seed=config.seed * 1000 + round_index * 100,
            ),
            continuous=True,
            collection_label="steps",
        ).run()

        return self.save_and_report(model, train_env, eval_env, run_dir, synthetic_queries=total_queries, runtime=runtime)

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
        paths = BackendRunPaths(run_dir)
        model.save(paths.final_model)
        train_env.save(paths.vecnormalize)

        actual_timesteps = int(model.num_timesteps)
        best_logged_reward, best_logged_timestep = report_eval_curve(
            paths.eval_log,
            paths.true_reward_curve,
            max(config.timesteps, actual_timesteps),
            config.plot_mode,
            config.smooth_window,
            x_scale=1e7,
            x_label="Timesteps (1e7)",
            y_floor=-4,
        )

        final_stats = evaluate_mujoco_components(
            model,
            config.env_id,
            self.spec,
            custom_partial=self.custom_partial,
            stats_source=train_env,
            n_eval_episodes=config.final_eval_episodes,
            seed=config.seed + 50_000,
        )
        write_mujoco_component_summary(
            paths.final_component_evaluation,
            actual_timesteps,
            self.spec,
            final_stats,
            custom_partial=self.custom_partial,
        )

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
        selected_stats = evaluate_mujoco_components(
            final_policy,
            config.env_id,
            self.spec,
            custom_partial=self.custom_partial,
            stats_source=final_eval_env,
            n_eval_episodes=config.final_eval_episodes,
            seed=config.seed + 60_000,
        )

        metadata = {
            **self.common_metadata(actual_timesteps, synthetic_queries, best_logged_reward, best_logged_timestep),
            "env_slug": self.spec.slug,
            "preset": config.preset,
            "partial_profile": config.partial_profile,
            "partial_keys": list(self.spec.partial_keys) if self.custom_partial is None else [self.custom_partial.name],
            "partial_weights": (
                list(self.spec.partial_weights or tuple(1.0 for _ in self.spec.partial_keys))
                if self.custom_partial is None
                else None
            ),
            "component_keys": list(mujoco_component_keys(self.spec, self.custom_partial)),
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
            continuous=not isinstance(action_space, spaces.Discrete),
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
        paths = BackendRunPaths(run_dir)
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
            continuous=False,
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
        paths = BackendRunPaths(run_dir)
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
