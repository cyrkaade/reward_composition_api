"""Preference-based reward learning: trajectory fragmenting, synthetic
rating, query-pair selection (random / MC-dropout / ensemble variance), and
reward-model (pre)training with pairwise Bradley-Terry or delta losses."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Callable

import numpy as np
import torch as th
from torch.optim import Adam

from ..data import Preference, Trajectory
from .model import (
    DeltaLoss,
    OutputRegularizationLoss,
    PairwiseLoss,
    RegularizationLoss,
    RewardModel,
    preference_prob,
)


def rate_pairs_from_true_reward(pairs: list[tuple[Trajectory, Trajectory]]) -> list[Preference]:
    rated_pairs = []
    for t1, t2 in pairs:
        rated_pairs.append(Preference(t1, t2, float(t1.get_summed_reward() > t2.get_summed_reward())))
    return rated_pairs


def fragment_trajectories(trajectories: list[Trajectory], fragment_length: int) -> list[Trajectory]:
    fragments = []
    for trajectory in trajectories:
        states = trajectory.get_states()
        for start_idx in range(0, len(states), fragment_length):
            if start_idx + fragment_length <= len(states):
                fragments.append(Trajectory(states[start_idx : start_idx + fragment_length]))
    return fragments


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
    transform_partial: Callable[[float], float] | None = None,
) -> list[tuple[Trajectory, Trajectory]]:
    if len(fragments) < 2:
        return []

    mc_dropout_models = [deepcopy(reward_model).dropout(dropout_p) for _ in range(k)]
    fragment_tensor = th.as_tensor([convert_traj(fragment) for fragment in fragments], dtype=th.float32)
    partial_returns = _partial_fragment_returns(fragments, transform_partial)

    pred_returns = []
    with th.no_grad():
        for model in mc_dropout_models:
            returns = th.sum(model(fragment_tensor), dim=[1, 2]).detach().cpu().numpy().tolist()
            if add_partial_to_predictions:
                returns = [model_return + partial for model_return, partial in zip(returns, partial_returns)]
            pred_returns.append(returns)

    return _preference_variance_pairs(fragments, pred_returns, query_count, n_batches)


def ensemble_active_learning_pairs(
    fragments: list[Trajectory],
    reward_models: list[RewardModel],
    query_count: int,
    convert_traj: Callable[[Trajectory], list[list[float]]],
    add_partial_to_predictions: bool,
    n_batches: int,
    transform_partial: Callable[[float], float] | None = None,
) -> list[tuple[Trajectory, Trajectory]]:
    if len(fragments) < 2 or not reward_models:
        return []

    fragment_tensor = th.as_tensor([convert_traj(fragment) for fragment in fragments], dtype=th.float32)
    partial_returns = _partial_fragment_returns(fragments, transform_partial)

    pred_returns = []
    with th.no_grad():
        for model in reward_models:
            returns = th.sum(model(fragment_tensor), dim=[1, 2]).detach().cpu().numpy().tolist()
            if add_partial_to_predictions:
                returns = [model_return + partial for model_return, partial in zip(returns, partial_returns)]
            pred_returns.append(returns)

    return _preference_variance_pairs(fragments, pred_returns, query_count, n_batches)


def _partial_fragment_returns(fragments: list[Trajectory], transform_partial: Callable[[float], float] | None) -> list[float]:
    transform = transform_partial or (lambda value: value)
    return [sum(transform(state["partial_rew"]) for state in fragment.states) for fragment in fragments]


def _preference_variance_pairs(
    fragments: list[Trajectory],
    pred_returns: list[list[float]],
    query_count: int,
    n_batches: int,
) -> list[tuple[Trajectory, Trajectory]]:
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
            [
                [(pred_returns[model_idx][i], pred_returns[model_idx][j]) for i, j in indices_batch]
                for model_idx in range(len(pred_returns))
            ],
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
    reward_model: RewardModel | list[RewardModel] | None,
    query_count: int,
    fragment_length: int,
    active_learning: bool,
    convert_traj: Callable[[Trajectory], list[list[float]]],
    add_partial_to_predictions: bool,
    dropout_samples: int,
    dropout_p: float,
    active_learning_batches: int,
    active_query_strategy: str = "auto",
    transform_partial: Callable[[float], float] | None = None,
) -> list[tuple[Trajectory, Trajectory]]:
    fragments = fragment_trajectories(trajectories, fragment_length)
    if len(fragments) < 2 or query_count <= 0:
        return []
    if reward_model is None or not active_learning:
        return random_query_pairs(fragments, query_count)

    reward_models = reward_model if isinstance(reward_model, list) else [reward_model]
    if active_query_strategy == "auto":
        active_query_strategy = "ensemble" if len(reward_models) > 1 else "dropout"
    if active_query_strategy == "ensemble" and len(reward_models) > 1:
        return ensemble_active_learning_pairs(
            fragments,
            reward_models,
            query_count,
            convert_traj,
            add_partial_to_predictions,
            active_learning_batches,
            transform_partial,
        )

    return dropout_active_learning_pairs(
        fragments,
        reward_models[0],
        query_count,
        convert_traj,
        add_partial_to_predictions,
        dropout_samples,
        dropout_p,
        active_learning_batches,
        transform_partial,
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


def partial_reward_tensor(rated_pairs: list[Preference], side: str, partial_mean: float = 0.0, partial_std: float = 1.0):
    trajectories = [pair.t1 if side == "t1" else pair.t2 for pair in rated_pairs]
    rewards = [
        [[(state["partial_rew"] - partial_mean) / max(partial_std, 1e-8)] for state in trajectory.states]
        for trajectory in trajectories
    ]
    return th.as_tensor(rewards, dtype=th.float32)


def reward_model_io_stats(
    reward_model: RewardModel | list[RewardModel],
    trajectories: list[Trajectory],
    convert_traj: Callable[[Trajectory], list[list[float]]],
):
    if not trajectories:
        return None, None
    reward_models = reward_model if isinstance(reward_model, list) else [reward_model]
    with th.no_grad():
        trajectory_tensors = th.as_tensor([convert_traj(trajectory) for trajectory in trajectories], dtype=th.float32)
        outputs = th.stack([model(trajectory_tensors).reshape(-1) for model in reward_models]).mean(dim=0)
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
    partial_mean: float = 0.0,
    partial_std: float = 1.0,
    partial_alpha: float = 1.0,
    partial_alpha_penalty: float = 0.0,
    partial_prediction_coef: float = 0.0,
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
        # train()/eval() only affect the optional output batch-norm; they are
        # no-ops for the default model, which has no train/eval-sensitive layers.
        reward_model.train()
        random.shuffle(train_pairs)
        t1_tensor, t2_tensor, ratings = rated_pairs_to_tensors(train_pairs, convert_traj)
        running_loss = 0.0
        batches = 0

        for batch_start in range(0, len(train_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(train_pairs))
            batch_pairs = train_pairs[batch_start:batch_end]
            x1 = t1_tensor[batch_start:batch_end]
            x2 = t2_tensor[batch_start:batch_end]
            y1 = reward_model(x1)
            y2 = reward_model(x2)
            rating_batch = ratings[batch_start:batch_end]

            if use_delta_loss:
                alpha = reward_model.alpha if reward_model.alpha is not None else partial_alpha
                t1_partial = partial_reward_tensor(batch_pairs, "t1", partial_mean, partial_std)
                t2_partial = partial_reward_tensor(batch_pairs, "t2", partial_mean, partial_std)
                loss = preference_loss(y1, y2, t1_partial, t2_partial, rating_batch, alpha)
            else:
                loss = preference_loss(y1, y2, rating_batch)

            loss += (output_regularization_loss.forward(y1) + output_regularization_loss.forward(y2)) / 2
            total_loss = loss.sum() + regularization_loss.forward(reward_model)
            if use_delta_loss and reward_model.alpha is not None:
                total_loss = total_loss + partial_alpha_penalty * (reward_model.alpha - partial_alpha) ** 2
            if use_delta_loss and partial_prediction_coef > 0 and reward_model.partial_head is not None:
                prediction_loss = (
                    th.nn.functional.mse_loss(reward_model.predict_partial(x1), t1_partial)
                    + th.nn.functional.mse_loss(reward_model.predict_partial(x2), t2_partial)
                ) / 2
                total_loss = total_loss + partial_prediction_coef * prediction_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += float(loss.mean().item())
            batches += 1

        reward_model.eval()
        val_loss = validate_preference_reward_model(
            reward_model, val_pairs, convert_traj, preference_loss, use_delta_loss, partial_mean, partial_std, partial_alpha
        )
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


def split_preference_k_folds(rated_pairs: list[Preference], k: int) -> list[list[Preference]]:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    folds = [[] for _ in range(k)]
    shuffled = list(rated_pairs)
    random.shuffle(shuffled)
    for index, pair in enumerate(shuffled):
        folds[index % k].append(pair)
    return folds


def train_preference_reward_ensemble(
    reward_models: list[RewardModel],
    rated_pairs: list[Preference],
    convert_traj: Callable[[Trajectory], list[list[float]]],
    use_delta_loss: bool,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float = 0.01,
    partial_mean: float = 0.0,
    partial_std: float = 1.0,
    partial_alpha: float = 1.0,
    partial_alpha_penalty: float = 0.0,
    partial_prediction_coef: float = 0.0,
) -> None:
    if not rated_pairs:
        return
    if not reward_models:
        raise ValueError("reward_models must not be empty")

    folds = split_preference_k_folds(rated_pairs, len(reward_models))
    for fold_index, reward_model in enumerate(reward_models):
        val_pairs = folds[fold_index]
        train_pairs = [
            pair
            for other_fold_index, fold in enumerate(folds)
            if other_fold_index != fold_index
            for pair in fold
        ]
        if not train_pairs:
            train_pairs = list(rated_pairs)
        print(
            f"training reward ensemble member {fold_index + 1}/{len(reward_models)} "
            f"on {len(train_pairs)} pairs; validating on {len(val_pairs)} pairs"
        )
        train_preference_reward_model(
            reward_model,
            train_pairs,
            val_pairs,
            convert_traj=convert_traj,
            use_delta_loss=use_delta_loss,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            learning_rate=learning_rate,
            partial_mean=partial_mean,
            partial_std=partial_std,
            partial_alpha=partial_alpha,
            partial_alpha_penalty=partial_alpha_penalty,
            partial_prediction_coef=partial_prediction_coef,
        )


def validate_preference_reward_model(
    reward_model,
    val_pairs,
    convert_traj,
    preference_loss,
    use_delta_loss: bool,
    partial_mean: float = 0.0,
    partial_std: float = 1.0,
    partial_alpha: float = 1.0,
) -> float:
    if not val_pairs:
        return 0.0
    with th.no_grad():
        t1_tensor, t2_tensor, ratings = rated_pairs_to_tensors(val_pairs, convert_traj)
        y1 = reward_model(t1_tensor)
        y2 = reward_model(t2_tensor)
        if use_delta_loss:
            alpha = reward_model.alpha if reward_model.alpha is not None else partial_alpha
            t1_partial = partial_reward_tensor(val_pairs, "t1", partial_mean, partial_std)
            t2_partial = partial_reward_tensor(val_pairs, "t2", partial_mean, partial_std)
            return float(preference_loss(y1, y2, t1_partial, t2_partial, ratings, alpha).mean().item())
        return float(preference_loss(y1, y2, ratings).mean().item())
