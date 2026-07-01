from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import VecNormalize

from reward_composition_api.config import ExperimentConfig
from reward_composition_api.evaluation.reporting import RunPaths
from reward_composition_api.partial_reward import include_partial_feature, resolve_custom_partial
from reward_composition_api.registry import PartialSpec
from reward_composition_api.results import RunResult
from reward_composition_api.training import SaveVecNormalizeOnBest
from reward_composition_api.reward_models.reward_model import RewardModel


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

    def write_metadata(self, paths: RunPaths, metadata: dict) -> Path:
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


