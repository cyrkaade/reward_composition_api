"""The single experiment runner for all suites and all five reward modes,
including the RLHF round loop for the preference modes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from gymnasium.spaces.utils import flatdim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from .config import PREFERENCE_MODES, ExperimentConfig, normalize_experiment_config
from .data import Trajectory
from .envs import TrajectoryCollector, load_eval_env, make_eval_env, make_train_env
from .evaluation import (
    ComponentEvalCallback,
    RunPaths,
    evaluate_components,
    component_fieldnames,
    report_eval_curve,
    select_final_policy,
    write_component_summary_csv,
)
from .partials import include_partial_feature, resolve_custom_partial
from .rewards.model import RewardModel
from .rewards.preferences import (
    choose_query_pairs,
    pretrain_reward_model,
    rate_pairs_from_true_reward,
    reward_model_io_stats,
    train_preference_reward_ensemble,
    train_preference_reward_model,
)
from .rewards.wrapper import LearnedRewardRuntime, PreferenceRewardWrapper
from .suites import Suite, get_suite


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    metadata_path: Path
    model_path: Path
    vecnormalize_path: Path | None = None
    synthetic_queries: int = 0
    metadata: dict = field(default_factory=dict)


class SaveVecNormalizeOnBest(BaseCallback):
    def __init__(self, env: VecNormalize, save_path: Path):
        super().__init__()
        self.env = env
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.env.save(self.save_path)
        return True


def query_schedule(query_budget: int, rounds: int) -> list[int]:
    unit = query_budget // rounds
    schedule = [unit] * rounds
    for i in range(query_budget - sum(schedule)):
        schedule[i % len(schedule)] += 1
    return schedule


def policy_training_schedule(total_timesteps: int, rounds: int, timesteps_per_round: int | None = None) -> list[int]:
    if timesteps_per_round is not None:
        return [timesteps_per_round] * rounds

    policy_steps_per_round = total_timesteps // rounds
    leftover_policy_steps = total_timesteps - policy_steps_per_round * rounds
    return [
        policy_steps_per_round + (leftover_policy_steps if round_index == rounds - 1 else 0)
        for round_index in range(rounds)
    ]


def learn_policy(
    model,
    total_timesteps: int,
    callback,
    progress_bar: bool,
    reset_num_timesteps: bool = True,
    log_interval: int | None = None,
) -> None:
    if total_timesteps <= 0:
        return

    learn_kwargs = {
        "total_timesteps": int(total_timesteps),
        "callback": callback,
        "progress_bar": progress_bar,
        "reset_num_timesteps": reset_num_timesteps,
    }
    if log_interval is not None:
        learn_kwargs["log_interval"] = log_interval
    model.learn(**learn_kwargs)


class RlhfTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        model,
        runtime,
        callbacks,
        reward_model: RewardModel | list[RewardModel],
        convert_traj: Callable[[Trajectory], list[list[float]]],
        collect_trajectories: Callable[[int, int], list[Trajectory]],
        collection_label: str,
    ):
        self.config = config
        self.model = model
        self.runtime = runtime
        self.callbacks = callbacks
        self.reward_models = reward_model if isinstance(reward_model, list) else [reward_model]
        self.reward_model = self.reward_models[0]
        self.convert_traj = convert_traj
        self.collect_trajectories = collect_trajectories
        self.collection_label = collection_label
        self.rated_train = []
        self.rated_val = []
        self.total_queries = 0
        self.pretraining_done = False
        self.partial_stat_count = 0
        self.partial_stat_mean = 0.0
        self.partial_stat_m2 = 0.0
        self.schedule = query_schedule(config.query_budget, config.rlhf_rounds)
        self.policy_steps_by_round = policy_training_schedule(
            config.timesteps,
            config.rlhf_rounds,
            config.policy_timesteps_per_round,
        )
        self.add_partial_to_predictions = config.mode in {"naive", "delta"}

    def run(self) -> int:
        self.train_initial_policy()
        for round_index, round_query_budget in enumerate(self.schedule):
            self.run_round(round_index, round_query_budget)
        self.train_final_policy()
        return self.total_queries

    def train_initial_policy(self) -> None:
        config = self.config
        if config.initial_timesteps:
            print(f"initial PPO training on {config.mode} reward for {config.initial_timesteps} timesteps")
            learn_policy(
                self.model,
                config.initial_timesteps,
                self.callbacks,
                progress_bar=config.progress_bar,
                reset_num_timesteps=False,
                log_interval=config.policy_log_interval,
            )

    def run_round(self, round_index: int, round_query_budget: int) -> None:
        config = self.config
        collection_steps = config.collection_timesteps * (2 if round_index == 0 else 1)
        print(
            f"\nPreference round {round_index}: "
            f"collecting {collection_steps} {self.collection_label} for {round_query_budget} queries"
        )
        if round_query_budget <= 0 and not self._needs_pretraining():
            print("skipping preference collection because no queries are scheduled")
            self.train_policy_round(round_index)
            return

        trajectories = self.collect_trajectories(round_index, collection_steps)
        self.update_partial_stats(trajectories)
        self.maybe_pretrain_reward_model(trajectories)
        self.add_query_pairs(trajectories, round_query_budget)
        self.maybe_train_reward_model()
        self.train_policy_round(round_index)
        if self.total_queries >= config.query_budget:
            print("synthetic query budget exhausted")

    def _needs_pretraining(self) -> bool:
        return bool(self.config.pretrain_reward_model and not self.pretraining_done)

    def update_partial_stats(self, trajectories: list[Trajectory]) -> None:
        """Welford running mean/std over every partial-reward step seen so far,
        shared by the model input feature, the delta loss, and the composed reward."""
        if not self.runtime.normalize_partial:
            return
        for trajectory in trajectories:
            for state in trajectory.states:
                value = float(state["partial_rew"])
                self.partial_stat_count += 1
                delta = value - self.partial_stat_mean
                self.partial_stat_mean += delta / self.partial_stat_count
                self.partial_stat_m2 += delta * (value - self.partial_stat_mean)
        if self.partial_stat_count >= 2:
            self.runtime.partial_mean = self.partial_stat_mean
            self.runtime.partial_std = max((self.partial_stat_m2 / self.partial_stat_count) ** 0.5, 1e-8)
            print(f"partial reward stats: mean={self.runtime.partial_mean:.4f}, std={self.runtime.partial_std:.4f}")

    def maybe_pretrain_reward_model(self, trajectories: list[Trajectory]) -> None:
        config = self.config
        if self._needs_pretraining():
            print(f"pretraining reward model on {config.pretrain_target} target")
            for model_index, reward_model in enumerate(self.reward_models):
                if len(self.reward_models) > 1:
                    print(f"pretraining reward ensemble member {model_index + 1}/{len(self.reward_models)}")
                pretrain_reward_model(
                    reward_model,
                    trajectories,
                    self.convert_traj,
                    target=config.pretrain_target,
                    epochs=config.pretrain_epochs,
                    batch_size=config.pretrain_batch_size,
                    learning_rate=config.pretrain_lr,
                )
            self.pretraining_done = True

    def add_query_pairs(self, trajectories: list[Trajectory], round_query_budget: int) -> None:
        config = self.config
        query_model = self.reward_models if (self.total_queries > 0 or self.pretraining_done) else None
        pairs = choose_query_pairs(
            trajectories,
            query_model,
            query_count=min(round_query_budget, config.query_budget - self.total_queries),
            fragment_length=config.fragment_length,
            active_learning=config.active_learning,
            convert_traj=self.convert_traj,
            add_partial_to_predictions=self.add_partial_to_predictions,
            dropout_samples=config.dropout_samples,
            dropout_p=config.dropout_p,
            active_learning_batches=config.active_learning_batches,
            active_query_strategy=config.active_query_strategy,
            transform_partial=self.runtime.composed_partial_reward,
        )
        rated_pairs = rate_pairs_from_true_reward(pairs)
        split = int(len(rated_pairs) * 0.8)
        self.rated_train.extend(rated_pairs[:split])
        self.rated_val.extend(rated_pairs[split:])
        self.total_queries += len(rated_pairs)
        print(f"rated {len(rated_pairs)} synthetic preference pairs; cumulative={self.total_queries}")

    def maybe_train_reward_model(self) -> None:
        config = self.config
        if self.rated_train:
            if len(self.reward_models) > 1:
                train_preference_reward_ensemble(
                    self.reward_models,
                    self.rated_train + self.rated_val,
                    convert_traj=self.convert_traj,
                    use_delta_loss=config.mode == "delta",
                    batch_size=config.reward_model_batch_size,
                    epochs=config.reward_model_epochs,
                    patience=config.reward_model_patience,
                    learning_rate=config.reward_model_lr,
                    partial_mean=self.runtime.partial_mean,
                    partial_std=self.runtime.partial_std,
                    partial_alpha=config.partial_alpha,
                    partial_alpha_penalty=config.partial_alpha_penalty,
                    partial_prediction_coef=config.partial_prediction_coef,
                )
                self.runtime.reward_model = None
                self.runtime.reward_models = self.reward_models
            else:
                train_preference_reward_model(
                    self.reward_model,
                    self.rated_train,
                    self.rated_val,
                    convert_traj=self.convert_traj,
                    use_delta_loss=config.mode == "delta",
                    batch_size=config.reward_model_batch_size,
                    epochs=config.reward_model_epochs,
                    patience=config.reward_model_patience,
                    learning_rate=config.reward_model_lr,
                    partial_mean=self.runtime.partial_mean,
                    partial_std=self.runtime.partial_std,
                    partial_alpha=config.partial_alpha,
                    partial_alpha_penalty=config.partial_alpha_penalty,
                    partial_prediction_coef=config.partial_prediction_coef,
                )
                self.runtime.reward_model = self.reward_model
                self.runtime.reward_models = None
            if config.learn_partial_alpha:
                alphas = [float(model.alpha.item()) for model in self.reward_models if model.alpha is not None]
                self.runtime.partial_alpha = sum(alphas) / len(alphas)
                print(f"learned partial alpha: {self.runtime.partial_alpha:.4f}")
            stat_trajectories = [pair.t1 for pair in self.rated_train + self.rated_val] + [
                pair.t2 for pair in self.rated_train + self.rated_val
            ]
            self.runtime.output_mean, self.runtime.output_std = reward_model_io_stats(
                self.reward_models,
                stat_trajectories,
                self.convert_traj,
            )
            print(f"reward model output stats: mean={self.runtime.output_mean}, std={self.runtime.output_std}")

    def train_policy_round(self, round_index: int) -> None:
        config = self.config
        policy_steps = self.policy_steps_by_round[round_index]
        print(f"training PPO on {config.mode} reward for {policy_steps} timesteps")
        learn_policy(
            self.model,
            policy_steps,
            self.callbacks,
            progress_bar=config.progress_bar,
            reset_num_timesteps=False,
            log_interval=config.policy_log_interval,
        )

    def train_final_policy(self) -> None:
        config = self.config
        if config.final_policy_timesteps:
            print(f"final PPO training on {config.mode} reward for {config.final_policy_timesteps} timesteps")
            learn_policy(
                self.model,
                config.final_policy_timesteps,
                self.callbacks,
                progress_bar=config.progress_bar,
                reset_num_timesteps=False,
                log_interval=config.policy_log_interval,
            )


def default_run_name(config: ExperimentConfig, suite: Suite) -> str:
    variant = config.variant_name or config.mode
    steps = f"{config.timesteps // 1_000_000}m" if config.timesteps >= 1_000_000 else f"{config.timesteps}"
    return f"{suite.slug(config.env_id)}_{variant}_{steps}_seed{config.seed}"


def make_reward_models(input_size: int, config: ExperimentConfig) -> RewardModel | list[RewardModel]:
    models = [
        RewardModel(
            input_size=input_size,
            hidden_sizes=config.reward_hidden_sizes,
            learn_alpha=config.learn_partial_alpha,
            alpha_init=config.partial_alpha,
            predict_partial=config.partial_prediction_coef > 0,
        )
        for _ in range(config.reward_model_ensemble_size)
    ]
    return models[0] if len(models) == 1 else models


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        config = normalize_experiment_config(config)
        self.suite = get_suite(config.suite)
        self.config = replace(
            config,
            run_name=config.run_name or default_run_name(config, self.suite),
            variant_name=config.variant_name or config.mode,
        )
        self.custom_partial = resolve_custom_partial(self.config)

    @property
    def run_dir(self) -> Path:
        return Path(self.config.log_dir) / self.config.run_name

    def run(self) -> RunResult:
        self.suite.setup(self.config)
        if self.config.mode in PREFERENCE_MODES:
            return self.train_preference_mode()
        return self.train_true_or_partial()

    def eval_freq(self) -> int:
        return max(self.config.eval_freq // self.config.n_envs, 1)

    def probe_spaces(self):
        probe_env = self.suite.make_raw_env(self.config.env_id)
        observation_space = probe_env.observation_space
        action_space = probe_env.action_space
        normalize = self.suite.should_normalize_observation(observation_space)
        hyperparams = self.suite.ppo_hyperparams(self.config, probe_env)
        probe_env.close()
        return observation_space, action_space, normalize, hyperparams

    def build_runtime(self, composition: str, observation_space, action_space, **kwargs) -> LearnedRewardRuntime:
        return LearnedRewardRuntime(
            env_id=self.config.env_id,
            composition=composition,
            observation_space=observation_space,
            action_space=action_space,
            observation_features=self.suite.observation_features,
            custom_partial=self.custom_partial,
            reset_info=dict(self.suite.wrapper_reset_info),
            cast_true_reward=self.suite.cast_true_reward_info,
            **kwargs,
        )

    def build_envs_and_callbacks(self, env_fn, run_dir: Path, normalize: bool):
        config = self.config
        train_env = make_train_env(env_fn, config.n_envs, run_dir / "monitor", normalize)
        eval_env = make_eval_env(self.suite.make_raw_env, config.env_id, train_env)

        best_callbacks: list[BaseCallback] = []
        if isinstance(train_env, VecNormalize):
            best_callbacks.append(SaveVecNormalizeOnBest(train_env, run_dir / "best_model" / "best_vecnormalize.pkl"))
        if config.stop_reward is not None:
            best_callbacks.append(StopTrainingOnRewardThreshold(reward_threshold=config.stop_reward, verbose=1))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(run_dir / "eval"),
            eval_freq=self.eval_freq(),
            n_eval_episodes=config.n_eval_episodes,
            deterministic=True,
            render=False,
            callback_on_new_best=CallbackList(best_callbacks) if best_callbacks else None,
        )
        component_callback = ComponentEvalCallback(
            run_dir / "eval" / "component_evaluations.csv",
            self.suite,
            config.env_id,
            custom_partial=self.custom_partial,
            eval_freq=self.eval_freq(),
            n_eval_episodes=config.n_eval_episodes,
            verbose=1,
        )
        return train_env, eval_env, CallbackList([eval_callback, component_callback])

    def train_true_or_partial(self) -> RunResult:
        config = self.config
        run_dir = self.run_dir
        run_dir.mkdir(exist_ok=True, parents=True)
        observation_space, action_space, normalize, hyperparams = self.probe_spaces()

        if config.mode == "true":
            env_fn = lambda: self.suite.make_raw_env(config.env_id)
        else:
            runtime = self.build_runtime("partial", observation_space, action_space)
            env_fn = lambda: PreferenceRewardWrapper(self.suite.make_raw_env(config.env_id), runtime)

        train_env, eval_env, callbacks = self.build_envs_and_callbacks(env_fn, run_dir, normalize)
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
        run_dir = self.run_dir
        run_dir.mkdir(exist_ok=True, parents=True)
        observation_space, action_space, normalize, hyperparams = self.probe_spaces()

        runtime = self.build_runtime(
            config.mode,
            observation_space,
            action_space,
            target_mean=config.model_reward_target_mean,
            target_std=config.model_reward_target_std,
            reward_min=config.model_reward_min,
            reward_max=config.model_reward_max,
            reward_scale=config.model_reward_scale,
            normalize=config.normalize_model_reward,
            normalize_partial=config.normalize_partial_reward,
            partial_alpha=config.partial_alpha,
            include_partial_feature=include_partial_feature(config),
        )
        train_env, eval_env, callbacks = self.build_envs_and_callbacks(
            lambda: PreferenceRewardWrapper(self.suite.make_raw_env(config.env_id), runtime),
            run_dir,
            normalize,
        )
        model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **hyperparams)

        input_size = flatdim(observation_space) + flatdim(action_space) + 1
        reward_model = make_reward_models(input_size, config)
        convert_traj = self.trajectory_converter(runtime)
        total_queries = RlhfTrainer(
            config,
            model,
            runtime,
            callbacks,
            reward_model,
            convert_traj,
            lambda round_index, collection_steps: TrajectoryCollector(vec_env=train_env, agent=model).rollout_trajectories(
                total_timesteps=collection_steps,
                seed=config.seed * 1000 + round_index * 100,
            ),
            collection_label=self.suite.collection_label,
        ).run()

        return self.save_and_report(model, train_env, eval_env, run_dir, synthetic_queries=total_queries, runtime=runtime)

    def trajectory_converter(self, runtime: LearnedRewardRuntime):
        def convert(trajectory: Trajectory):
            return [
                runtime.model_features(state["obs"], state["act"], state["partial_rew"]).tolist()
                for state in trajectory.states
            ]

        return convert

    def save_and_report(
        self,
        model: PPO,
        train_env,
        eval_env,
        run_dir: Path,
        synthetic_queries: int,
        runtime: LearnedRewardRuntime | None = None,
    ) -> RunResult:
        config = self.config
        suite = self.suite
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
            x_scale=suite.curve_x_scale,
            x_label=suite.curve_x_label,
            y_floor=suite.curve_y_floor,
        )

        final_stats = evaluate_components(
            model,
            suite,
            config.env_id,
            custom_partial=self.custom_partial,
            stats_source=train_env,
            n_eval_episodes=config.final_eval_episodes,
            seed=config.seed + 50_000,
        )
        write_component_summary_csv(
            paths.final_component_evaluation,
            actual_timesteps,
            final_stats,
            component_fieldnames(self.custom_partial),
        )

        final_policy, final_eval_env = select_final_policy(
            config,
            model,
            eval_env,
            run_dir,
            lambda env_id, stats_path: load_eval_env(suite.make_raw_env, env_id, stats_path),
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
        selected_stats = evaluate_components(
            final_policy,
            suite,
            config.env_id,
            custom_partial=self.custom_partial,
            stats_source=final_eval_env,
            n_eval_episodes=config.final_eval_episodes,
            seed=config.seed + 60_000,
        )

        metadata = {
            **self.common_metadata(actual_timesteps, synthetic_queries, best_logged_reward, best_logged_timestep),
            "env_slug": suite.slug(config.env_id),
            **suite.extra_metadata(config),
            "partial_keys": [self.custom_partial.name] if self.custom_partial else [],
            "component_keys": suite.metadata_component_keys(self.custom_partial),
            "selected_policy_true_reward_mean": float(mean_reward),
            "selected_policy_true_reward_std": float(std_reward),
            "selected_policy_components": selected_stats,
            **self.runtime_metadata(runtime),
        }

        paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.print_summary(float(mean_reward), float(std_reward), selected_stats, synthetic_queries)

        train_env.close()
        final_eval_env.close()
        return RunResult(
            run_dir=run_dir,
            metadata_path=paths.metadata,
            model_path=paths.final_model.with_suffix(".zip"),
            vecnormalize_path=vecnormalize_path,
            synthetic_queries=synthetic_queries,
            metadata=metadata,
        )

    def common_metadata(
        self,
        actual_timesteps: int,
        synthetic_queries: int,
        best_logged_reward: float | None,
        best_logged_timestep: int | None,
    ) -> dict:
        config = self.config
        is_preference = config.mode in PREFERENCE_MODES
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
            "query_budget": config.query_budget if is_preference else 0,
            "fragment_length": config.fragment_length if is_preference else None,
            "active_learning": config.active_learning if is_preference else None,
            "active_query_strategy": config.active_query_strategy if is_preference else None,
            "reward_hidden_sizes": list(config.reward_hidden_sizes),
            "reward_model_lr": config.reward_model_lr if is_preference else None,
            "reward_model_ensemble_size": config.reward_model_ensemble_size if is_preference else None,
            "pretrain_reward_model": config.pretrain_reward_model if is_preference else None,
            "pretrain_target": config.pretrain_target if config.pretrain_reward_model else None,
            "include_partial_feature": include_partial_feature(config) if is_preference else None,
            "normalize_partial_reward": config.normalize_partial_reward if is_preference else None,
            "partial_alpha": config.partial_alpha if is_preference else None,
            "learn_partial_alpha": config.learn_partial_alpha if is_preference else None,
            "partial_alpha_penalty": config.partial_alpha_penalty if config.learn_partial_alpha else None,
            "partial_prediction_coef": config.partial_prediction_coef if is_preference else None,
            "partial_reference": config.partial,
            "best_logged_true_reward": best_logged_reward,
            "best_logged_timestep": best_logged_timestep,
        }

    def runtime_metadata(self, runtime: LearnedRewardRuntime | None) -> dict:
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
            "partial_reward_mean": runtime.partial_mean,
            "partial_reward_std": runtime.partial_std,
            "final_partial_alpha": runtime.partial_alpha,
            "reward_composition": runtime.composition,
        }

    def print_summary(self, mean_reward: float, std_reward: float, selected_stats: dict, synthetic_queries: int) -> None:
        print(f"{self.config.final_policy.title()} deterministic true reward: {mean_reward:.3f} +/- {std_reward:.3f}")
        values = ", ".join(
            f"{key}={selected_stats.get(f'mean_{key}', 0.0):.3f}" for key in self.suite.summary_component_keys
        )
        print(f"Component means: {values}")
        print(f"Synthetic queries consumed: {synthetic_queries}")
        print(f"Saved model and logs to {self.run_dir}")


def run_experiment(config: ExperimentConfig) -> RunResult:
    return ExperimentRunner(config).run()
