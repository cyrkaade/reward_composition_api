from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from local_gym.classes.mujoco_reward_specs import MuJoCoRewardSpec, get_mujoco_reward_spec
from reward_composition_api.config import ExperimentConfig
from reward_composition_api.registry import PartialSpec
from reward_composition_api.results import RunResult
from reward_model.reward_model import RewardModel

from .common import (
    SaveVecNormalizeOnBest,
    include_partial_feature,
    learn_policy,
    resolve_custom_partial,
)
from .mujoco_env import (
    MuJoCoLearnedRewardRuntime,
    MuJoCoPreferenceRewardWrapper,
    collect_policy_trajectories as collect_mujoco_policy_trajectories,
    load_eval_env as load_mujoco_eval_env,
    make_eval_env as make_mujoco_eval_env,
    make_raw_env as make_mujoco_raw_env,
    make_trajectory_converter as make_mujoco_trajectory_converter,
    make_vecnormalize_env as make_mujoco_vecnormalize_env,
    ppo_hyperparams as mujoco_ppo_hyperparams,
)
from .mujoco_evaluation import (
    MuJoCoComponentEvalCallback,
    _component_keys as mujoco_component_keys,
    evaluate_mujoco_components,
    write_mujoco_component_summary,
)
from .reporting import (
    BackendRunPaths,
    report_eval_curve,
    select_final_policy,
)
from .rlhf import RlhfTrainer


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
            "reward_hidden_sizes": list(config.reward_hidden_sizes),
            "reward_model_lr": config.reward_model_lr if config.mode in self.preference_modes else None,
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

    def run(self) -> RunResult:
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
    ):
        self.spec = spec or get_mujoco_reward_spec(config.env_id).with_partial_profile(config.partial_profile)
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
        hyperparams = mujoco_ppo_hyperparams(config)

        if config.mode == "true":
            env_fn = lambda: make_mujoco_raw_env(config.env_id)
        elif config.mode == "partial":
            runtime = MuJoCoLearnedRewardRuntime(spec=self.spec, composition="partial", custom_partial=self.custom_partial)
            env_fn = lambda: MuJoCoPreferenceRewardWrapper(make_mujoco_raw_env(config.env_id), runtime)
        else:
            raise ValueError(f"Unsupported mode for this path: {config.mode}")

        train_env = make_mujoco_vecnormalize_env(env_fn, config.n_envs, run_dir / "monitor")
        eval_env = make_mujoco_eval_env(config.env_id, train_env)
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
        hyperparams = mujoco_ppo_hyperparams(config)

        runtime = MuJoCoLearnedRewardRuntime(
            spec=self.spec,
            composition=config.mode,
            custom_partial=self.custom_partial,
            target_mean=config.model_reward_target_mean,
            target_std=config.model_reward_target_std,
            reward_min=config.model_reward_min,
            reward_max=config.model_reward_max,
            reward_scale=config.model_reward_scale,
            normalize=config.normalize_model_reward,
            include_partial_feature=include_partial_feature(config),
        )
        train_env = make_mujoco_vecnormalize_env(
            lambda: MuJoCoPreferenceRewardWrapper(make_mujoco_raw_env(config.env_id), runtime),
            config.n_envs,
            run_dir / "monitor",
        )
        eval_env = make_mujoco_eval_env(config.env_id, train_env)
        callbacks = self.build_callbacks(run_dir, train_env, eval_env)
        model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **hyperparams)

        probe_env = make_mujoco_raw_env(config.env_id)
        action_shape = probe_env.action_space.shape
        input_size = probe_env.observation_space.shape[0] + action_shape[0] + 1
        probe_env.close()

        reward_model = RewardModel(input_size=input_size, hidden_sizes=config.reward_hidden_sizes)
        convert_traj = make_mujoco_trajectory_converter(runtime.include_partial_feature)
        total_queries = RlhfTrainer(
            config,
            model,
            runtime,
            callbacks,
            reward_model,
            convert_traj,
            lambda round_index, collection_steps: collect_mujoco_policy_trajectories(
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
        runtime: MuJoCoLearnedRewardRuntime | None = None,
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
            load_mujoco_eval_env,
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
