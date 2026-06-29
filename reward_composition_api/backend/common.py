from __future__ import annotations

import random
import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.optim import Adam

from local_gym.wrappers.buffering_wrapper import Trajectory
from reward_composition_api.partials import build_builtin_registry
from reward_composition_api.registry import PartialSpec, load_partial_reference
from reward_model.preferences.fragmenter import Fragmenter
from reward_model.preferences.preference import Preference
from reward_model.reward_model import (
    DeltaLoss,
    OutputRegularizationLoss,
    PairwiseLoss,
    RegularizationLoss,
    RewardModel,
    preference_prob,
)


class SaveVecNormalizeOnBest(BaseCallback):
    def __init__(self, env: VecNormalize, save_path: Path):
        super().__init__()
        self.env = env
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.env.save(self.save_path)
        return True


@dataclass(frozen=True)
class BackendRunPaths:
    run_dir: Path

    @property
    def final_model(self) -> Path:
        return self.run_dir / "final_model"

    @property
    def vecnormalize(self) -> Path:
        return self.run_dir / "vecnormalize.pkl"

    @property
    def eval_log(self) -> Path:
        return self.run_dir / "eval" / "evaluations.npz"

    @property
    def true_reward_curve(self) -> Path:
        return self.run_dir / "true_reward_curve.png"

    @property
    def final_component_evaluation(self) -> Path:
        return self.run_dir / "eval" / "final_component_evaluation.csv"

    @property
    def metadata(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def best_model(self) -> Path:
        return self.run_dir / "best_model" / "best_model.zip"

    @property
    def best_vecnormalize(self) -> Path:
        return self.run_dir / "best_model" / "best_vecnormalize.pkl"


class ComponentEvalCallback(BaseCallback):
    def __init__(
        self,
        log_path: Path,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.log_path = Path(log_path)
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed

    def component_fieldnames(self) -> list[str]:
        raise NotImplementedError

    def evaluate_components(self) -> dict:
        raise NotImplementedError

    def write_summary(self, stats: dict) -> None:
        raise NotImplementedError

    def log_message(self, stats: dict) -> str:
        return ""

    def _init_callback(self) -> None:
        self.log_path.parent.mkdir(exist_ok=True, parents=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.component_fieldnames())
                writer.writeheader()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        stats = self.evaluate_components()
        self.write_summary(stats)
        if self.verbose:
            message = self.log_message(stats)
            if message:
                print(message)
        return True


def normalize_obs(stats_source, observation):
    observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    if isinstance(stats_source, VecNormalize):
        return stats_source.normalize_obs(observation)
    return observation


def resolve_custom_partial(config) -> PartialSpec | None:
    if not config.partial:
        return None
    registry = build_builtin_registry()
    return load_partial_reference(config.partial, config.suite, registry)


def include_partial_feature(config) -> bool:
    if config.include_partial_feature is not None:
        return bool(config.include_partial_feature)
    return config.mode in {"naive", "delta"}


def make_raw_eval_env(make_raw_env, env_id: str):
    return DummyVecEnv([lambda: Monitor(make_raw_env(env_id))])


def load_vecnormalize_eval_env(env_id: str, stats_path: Path, make_raw_eval_env_fn) -> VecNormalize:
    env = VecNormalize.load(stats_path, make_raw_eval_env_fn(env_id))
    env.training = False
    env.norm_reward = False
    return env


def summarize_component_rows(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    stats = {}
    for key in keys:
        values = np.asarray([row.get(key, 0.0) for row in rows], dtype=np.float64)
        stats[f"mean_{key}"] = float(values.mean())
        stats[f"std_{key}"] = float(values.std())
    return stats


def write_component_summary_csv(path: Path, timestep: int, stats: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        row = {"timesteps": timestep}
        row.update({field: stats.get(field, "") for field in fieldnames if field != "timesteps"})
        writer.writerow(row)


def rate_pairs_from_true_reward(pairs: list[tuple[Trajectory, Trajectory]]) -> list[Preference]:
    rated_pairs = []
    for t1, t2 in pairs:
        rated_pairs.append(Preference(t1, t2, float(t1.get_summed_reward() > t2.get_summed_reward())))
    return rated_pairs


def fragment_trajectories(trajectories: list[Trajectory], fragment_length: int, continuous: bool) -> list[Trajectory]:
    fragmenter = Fragmenter(continuous=continuous, fragment_length=fragment_length, oversampling_rate=1.0)
    return fragmenter.fragment_trajectories(trajectories)


def random_query_pairs(fragments: list[Trajectory], query_count: int) -> list[tuple[Trajectory, Trajectory]]:
    shuffled = list(fragments)
    random.shuffle(shuffled)
    return list(zip(shuffled[::2], shuffled[1::2]))[:query_count]


def dropout_active_learning_pairs(
    fragments: list[Trajectory],
    reward_model: RewardModel,
    query_count: int,
    convert_traj: Callable[[Trajectory], list[list[float]]],
    add_partial_to_predictions: bool,
    k: int,
    dropout_p: float,
    n_batches: int,
) -> list[tuple[Trajectory, Trajectory]]:
    if len(fragments) < 2:
        return []

    mc_dropout_models = [deepcopy(reward_model).dropout(dropout_p) for _ in range(k)]
    fragment_tensor = th.as_tensor([convert_traj(fragment) for fragment in fragments], dtype=th.float32)
    partial_returns = [sum(state["partial_rew"] for state in fragment.states) for fragment in fragments]

    pred_returns = []
    with th.no_grad():
        for model in mc_dropout_models:
            returns = th.sum(model(fragment_tensor), dim=[1, 2]).detach().cpu().numpy().tolist()
            if add_partial_to_predictions:
                returns = [model_return + partial for model_return, partial in zip(returns, partial_returns)]
            pred_returns.append(returns)

    possible_indices = list(range(len(fragments)))
    best_indices_batch = None
    best_vars = None
    best_score = -float("inf")

    for _ in range(n_batches):
        random.shuffle(possible_indices)
        indices_batch = list(zip(possible_indices[::2], possible_indices[1::2]))
        if not indices_batch:
            continue
        pair_returns = th.as_tensor(
            [[(pred_returns[model_idx][i], pred_returns[model_idx][j]) for i, j in indices_batch] for model_idx in range(k)],
            dtype=th.float32,
        )
        pred_preferences = preference_prob(pair_returns, 2)
        variances = th.sum(th.var(pred_preferences, dim=0), dim=1).detach().cpu().numpy()
        score = float(np.sum(variances))
        if score > best_score:
            best_score = score
            best_indices_batch = indices_batch
            best_vars = variances

    if best_indices_batch is None or best_vars is None:
        return random_query_pairs(fragments, query_count)

    ranked = sorted(zip(best_indices_batch, best_vars), key=lambda item: item[1], reverse=True)
    return [(fragments[i], fragments[j]) for (i, j), _ in ranked[:query_count]]


def choose_query_pairs(
    trajectories: list[Trajectory],
    reward_model: RewardModel | None,
    query_count: int,
    fragment_length: int,
    active_learning: bool,
    convert_traj: Callable[[Trajectory], list[list[float]]],
    add_partial_to_predictions: bool,
    dropout_samples: int,
    dropout_p: float,
    active_learning_batches: int,
    continuous: bool,
) -> list[tuple[Trajectory, Trajectory]]:
    fragments = fragment_trajectories(trajectories, fragment_length, continuous=continuous)
    if len(fragments) < 2 or query_count <= 0:
        return []
    if reward_model is None or not active_learning:
        return random_query_pairs(fragments, query_count)
    return dropout_active_learning_pairs(
        fragments,
        reward_model,
        query_count,
        convert_traj,
        add_partial_to_predictions,
        dropout_samples,
        dropout_p,
        active_learning_batches,
    )


def rated_pairs_to_tensors(rated_pairs: list[Preference], convert_traj: Callable[[Trajectory], list[list[float]]]):
    t1s, t2s, ratings = [], [], []
    for pair in rated_pairs:
        t1s.append(convert_traj(pair.t1))
        t2s.append(convert_traj(pair.t2))
        ratings.append(pair.rating)
    return (
        th.as_tensor(t1s, dtype=th.float32),
        th.as_tensor(t2s, dtype=th.float32),
        th.as_tensor(ratings, dtype=th.float32),
    )


def partial_reward_tensor(rated_pairs: list[Preference], side: str):
    trajectories = [pair.t1 if side == "t1" else pair.t2 for pair in rated_pairs]
    rewards = [[[state["partial_rew"]] for state in trajectory.states] for trajectory in trajectories]
    return th.as_tensor(rewards, dtype=th.float32)


def reward_model_io_stats(
    reward_model: RewardModel,
    trajectories: list[Trajectory],
    convert_traj: Callable[[Trajectory], list[list[float]]],
):
    if not trajectories:
        return None, None
    with th.no_grad():
        trajectory_tensors = th.as_tensor([convert_traj(trajectory) for trajectory in trajectories], dtype=th.float32)
        outputs = reward_model(trajectory_tensors).reshape(-1)
    return float(outputs.mean().item()), float(outputs.std(unbiased=False).item())


def pretrain_reward_model(
    reward_model: RewardModel,
    trajectories: list[Trajectory],
    convert_traj: Callable[[Trajectory], list[list[float]]],
    target: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    rows = []
    targets = []
    for trajectory in trajectories:
        converted = convert_traj(trajectory)
        for converted_state, raw_state in zip(converted, trajectory.states):
            rows.append(converted_state)
            if target == "partial":
                targets.append([float(raw_state["partial_rew"])])
            elif target == "residual":
                targets.append([float(raw_state["rew"] - raw_state["partial_rew"])])
            elif target == "true":
                targets.append([float(raw_state["rew"])])
            else:
                raise ValueError(f"Unsupported pretrain target: {target}")

    if not rows:
        return

    x = th.as_tensor(rows, dtype=th.float32)
    y = th.as_tensor(targets, dtype=th.float32)
    optimizer = Adam(reward_model.parameters(), lr=learning_rate)
    loss_fn = th.nn.MSELoss()

    for epoch in range(epochs):
        order = th.randperm(x.shape[0])
        running_loss = 0.0
        batches = 0
        for batch_start in range(0, x.shape[0], batch_size):
            batch_indices = order[batch_start : batch_start + batch_size]
            pred = reward_model(x[batch_indices])
            loss = loss_fn(pred, y[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
        print(f"pretrain epoch {epoch}: mse={running_loss / max(batches, 1):.6f}")


def train_preference_reward_model(
    reward_model: RewardModel,
    train_pairs: list[Preference],
    val_pairs: list[Preference],
    convert_traj: Callable[[Trajectory], list[list[float]]],
    use_delta_loss: bool,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float = 0.01,
) -> None:
    if not train_pairs:
        return

    optimizer = Adam(reward_model.parameters(), lr=learning_rate)
    preference_loss = DeltaLoss() if use_delta_loss else PairwiseLoss()
    regularization_loss = RegularizationLoss(regularization_type="L1", lambda_reg=0.01)
    output_regularization_loss = OutputRegularizationLoss(regularization_type="L1", lambda_reg=0.001)
    best_state = deepcopy(reward_model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    no_improvement = 0

    for epoch in range(epochs):
        random.shuffle(train_pairs)
        t1_tensor, t2_tensor, ratings = rated_pairs_to_tensors(train_pairs, convert_traj)
        running_loss = 0.0
        batches = 0

        for batch_start in range(0, len(train_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(train_pairs))
            batch_pairs = train_pairs[batch_start:batch_end]
            y1 = reward_model(t1_tensor[batch_start:batch_end])
            y2 = reward_model(t2_tensor[batch_start:batch_end])
            rating_batch = ratings[batch_start:batch_end]

            if use_delta_loss:
                t1_partial = partial_reward_tensor(batch_pairs, "t1")
                t2_partial = partial_reward_tensor(batch_pairs, "t2")
                loss = preference_loss(y1, y2, t1_partial, t2_partial, rating_batch)
            else:
                loss = preference_loss(y1, y2, rating_batch)

            loss += (output_regularization_loss.forward(y1) + output_regularization_loss.forward(y2)) / 2
            total_loss = loss.sum() + regularization_loss.forward(reward_model)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += float(loss.mean().item())
            batches += 1

        val_loss = validate_preference_reward_model(reward_model, val_pairs, convert_traj, preference_loss, use_delta_loss)
        print(f"reward model epoch {epoch}: train_loss={running_loss / max(batches, 1):.4f}, val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(reward_model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"stopping reward model at epoch {epoch}; restoring epoch {best_epoch} val_loss={best_val:.4f}")
                reward_model.load_state_dict(best_state)
                break


class RlhfTrainer:
    def __init__(
        self,
        config,
        model,
        runtime,
        callbacks,
        reward_model: RewardModel,
        convert_traj: Callable[[Trajectory], list[list[float]]],
        collect_trajectories: Callable[[int, int], list[Trajectory]],
        continuous: bool,
        collection_label: str,
    ):
        self.config = config
        self.model = model
        self.runtime = runtime
        self.callbacks = callbacks
        self.reward_model = reward_model
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
            pretrain_reward_model(
                self.reward_model,
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
        query_model = self.reward_model if (self.total_queries > 0 or self.pretraining_done) else None
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
            stat_trajectories = [pair.t1 for pair in self.rated_train + self.rated_val] + [
                pair.t2 for pair in self.rated_train + self.rated_val
            ]
            self.runtime.output_mean, self.runtime.output_std = reward_model_io_stats(
                self.reward_model,
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


def run_preference_training_loop(
    config,
    model,
    runtime,
    callbacks,
    reward_model: RewardModel,
    convert_traj: Callable[[Trajectory], list[list[float]]],
    collect_trajectories: Callable[[int, int], list[Trajectory]],
    continuous: bool,
    collection_label: str,
) -> int:
    return RlhfTrainer(
        config,
        model,
        runtime,
        callbacks,
        reward_model,
        convert_traj,
        collect_trajectories,
        continuous,
        collection_label,
    ).run()


def validate_preference_reward_model(reward_model, val_pairs, convert_traj, preference_loss, use_delta_loss: bool) -> float:
    if not val_pairs:
        return 0.0
    with th.no_grad():
        t1_tensor, t2_tensor, ratings = rated_pairs_to_tensors(val_pairs, convert_traj)
        y1 = reward_model(t1_tensor)
        y2 = reward_model(t2_tensor)
        if use_delta_loss:
            t1_partial = partial_reward_tensor(val_pairs, "t1")
            t2_partial = partial_reward_tensor(val_pairs, "t2")
            return float(preference_loss(y1, y2, t1_partial, t2_partial, ratings).mean().item())
        return float(preference_loss(y1, y2, ratings).mean().item())


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


def load_eval_curve(evaluations_path: Path):
    if not evaluations_path.exists():
        raise FileNotFoundError(f"No evaluation log found at {evaluations_path}")
    eval_data = np.load(evaluations_path)
    return eval_data["timesteps"], eval_data["results"].mean(axis=1)


def smooth_curve(values, window):
    if window <= 1 or len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_true_reward_curve(
    evaluations_path: Path,
    output_path: Path,
    total_timesteps: int,
    plot_mode: str,
    smooth_window: int,
    x_scale: float,
    x_label: str,
    y_floor: float | None = None,
) -> None:
    raw_timesteps, raw_rewards = load_eval_curve(evaluations_path)
    timesteps = raw_timesteps / x_scale
    rewards = np.maximum.accumulate(raw_rewards) if plot_mode == "best" else raw_rewards
    rewards = smooth_curve(rewards, smooth_window)

    fig, ax = plt.subplots(figsize=(9, 5))
    label = "PPO best true reward" if plot_mode == "best" else "PPO true reward"
    ax.plot(timesteps, rewards, color="#2f6f9f", linewidth=2.2, label=label)
    ax.set_xlim(0, max(total_timesteps / x_scale, float(timesteps.max()) if len(timesteps) else 1.0, 1.0))
    ax.set_xlabel(x_label)

    if y_floor is not None:
        y_min = min(y_floor, int(np.floor(rewards.min() / 2.0) * 2))
        y_max = 0
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(0, y_min - 0.001, -2))
    else:
        y_min = float(np.min(rewards))
        y_max = float(np.max(rewards))
        if y_min == y_max:
            pad = max(abs(y_min) * 0.1, 1.0)
            y_min -= pad
            y_max += pad
        else:
            pad = max((y_max - y_min) * 0.08, 1.0)
            y_min -= pad
            y_max += pad
        ax.set_ylim(y_min, y_max)

    ax.set_ylabel("True reward")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def report_eval_curve(
    eval_log_path: Path,
    plot_path: Path,
    total_timesteps: int,
    plot_mode: str,
    smooth_window: int,
    x_scale: float,
    x_label: str,
    y_floor: float | None = None,
) -> tuple[float | None, int | None]:
    if not eval_log_path.exists():
        return None, None

    plot_true_reward_curve(
        eval_log_path,
        plot_path,
        total_timesteps,
        plot_mode,
        smooth_window,
        x_scale=x_scale,
        x_label=x_label,
        y_floor=y_floor,
    )
    eval_timesteps, eval_rewards = load_eval_curve(eval_log_path)
    best_idx = int(np.argmax(eval_rewards))
    best_logged_reward = float(eval_rewards[best_idx])
    best_logged_timestep = int(eval_timesteps[best_idx])
    print(f"Best logged true reward: {best_logged_reward:.3f} at {best_logged_timestep} timesteps")
    return best_logged_reward, best_logged_timestep


def select_final_policy(
    config,
    model,
    eval_env,
    run_dir: Path,
    load_eval_env_fn,
    load_policy_fn,
    load_best_stats: bool,
):
    paths = BackendRunPaths(run_dir)
    final_policy = model
    final_eval_env = eval_env
    if config.final_policy == "best" and paths.best_model.exists():
        if load_best_stats and paths.best_vecnormalize.exists():
            final_eval_env.close()
            final_eval_env = load_eval_env_fn(config.env_id, paths.best_vecnormalize)
        final_policy = load_policy_fn(paths.best_model, env=final_eval_env, device=config.device)
    return final_policy, final_eval_env
