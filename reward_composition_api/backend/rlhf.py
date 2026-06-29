from __future__ import annotations

from typing import Callable

from local_gym.wrappers.buffering_wrapper import Trajectory
from reward_model.reward_model import RewardModel

from .common import (
    choose_query_pairs,
    learn_policy,
    policy_training_schedule,
    pretrain_reward_model,
    query_schedule,
    rate_pairs_from_true_reward,
    reward_model_io_stats,
    train_preference_reward_ensemble,
    train_preference_reward_model,
)


class RlhfTrainer:
    def __init__(
        self,
        config,
        model,
        runtime,
        callbacks,
        reward_model: RewardModel | list[RewardModel],
        convert_traj: Callable[[Trajectory], list[list[float]]],
        collect_trajectories: Callable[[int, int], list[Trajectory]],
        continuous: bool,
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
        self.continuous = continuous
        self.collection_label = collection_label
        self.rated_train = []
        self.rated_val = []
        self.total_queries = 0
        self.pretraining_done = False
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
        trajectories = self.collect_trajectories(round_index, collection_steps)
        self.maybe_pretrain_reward_model(trajectories)
        self.add_query_pairs(trajectories, round_query_budget)
        self.maybe_train_reward_model()
        self.train_policy_round(round_index)
        if self.total_queries >= config.query_budget:
            print("synthetic query budget exhausted")

    def maybe_pretrain_reward_model(self, trajectories: list[Trajectory]) -> None:
        config = self.config
        if config.pretrain_reward_model and not self.pretraining_done:
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
            continuous=self.continuous,
            active_query_strategy=config.active_query_strategy,
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
                )
                self.runtime.reward_model = self.reward_model
                self.runtime.reward_models = None
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
