from __future__ import annotations

import random

import numpy as np
import pytest
import torch as th

from rcomp.data import Preference, Trajectory
from rcomp.rewards.model import DeltaLoss, PairwiseLoss, RewardModel
from rcomp.rewards.preferences import (
    choose_query_pairs,
    fragment_trajectories,
    partial_reward_tensor,
    pretrain_reward_model,
    random_query_pairs,
    rate_pairs_from_true_reward,
    reward_model_io_stats,
    split_preference_k_folds,
    train_preference_reward_model,
)
from rcomp.trainer import policy_training_schedule, query_schedule

FEATURE_DIM = 3  # obs(1) + act(1) + partial(1)


def make_trajectory(length: int, reward: float, partial: float = 0.0) -> Trajectory:
    states = [
        {
            "obs": np.full(1, reward, dtype=np.float32),
            "act": np.zeros(1, dtype=np.float32),
            "done": False,
            "info": {},
            "rew": reward,
            "partial_rew": partial,
        }
        for _ in range(length)
    ]
    return Trajectory(states)


def convert_traj(trajectory: Trajectory) -> list[list[float]]:
    return [
        [*np.asarray(state["obs"], dtype=np.float32).tolist(), float(state["act"][0]), float(state["partial_rew"])]
        for state in trajectory.states
    ]


def test_fragment_trajectories_keeps_exact_length_fragments():
    states = [{"rew": float(index), "partial_rew": 0.0, "done": False} for index in range(5)]

    fragments = fragment_trajectories([Trajectory(states)], fragment_length=2)

    assert [len(fragment.states) for fragment in fragments] == [2, 2]
    assert [fragment.get_summed_reward() for fragment in fragments] == [1.0, 5.0]


def test_rate_pairs_from_true_reward():
    high = make_trajectory(2, reward=5.0)
    low = make_trajectory(2, reward=1.0)

    rated = rate_pairs_from_true_reward([(high, low), (low, high)])

    assert rated[0].rating == 1.0
    assert rated[1].rating == 0.0


def test_random_query_pairs_respects_count():
    fragments = [make_trajectory(1, reward=float(index)) for index in range(10)]

    pairs = random_query_pairs(fragments, query_count=3)

    assert len(pairs) == 3
    for t1, t2 in pairs:
        assert t1 is not t2


def test_choose_query_pairs_random_when_no_model():
    random.seed(0)
    trajectories = [make_trajectory(4, reward=float(index)) for index in range(4)]

    pairs = choose_query_pairs(
        trajectories,
        reward_model=None,
        query_count=3,
        fragment_length=2,
        active_learning=True,
        convert_traj=convert_traj,
        add_partial_to_predictions=False,
        dropout_samples=2,
        dropout_p=0.25,
        active_learning_batches=4,
    )

    assert 0 < len(pairs) <= 3
    assert all(len(t.states) == 2 for pair in pairs for t in pair)


def test_choose_query_pairs_dropout_strategy():
    random.seed(0)
    th.manual_seed(0)
    trajectories = [make_trajectory(4, reward=float(index), partial=0.5) for index in range(4)]
    model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,))

    pairs = choose_query_pairs(
        trajectories,
        reward_model=model,
        query_count=3,
        fragment_length=2,
        active_learning=True,
        convert_traj=convert_traj,
        add_partial_to_predictions=True,
        dropout_samples=3,
        dropout_p=0.25,
        active_learning_batches=4,
        active_query_strategy="dropout",
    )

    assert 0 < len(pairs) <= 3


def test_choose_query_pairs_ensemble_strategy():
    random.seed(0)
    th.manual_seed(0)
    trajectories = [make_trajectory(4, reward=float(index)) for index in range(4)]
    models = [RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,)) for _ in range(2)]

    pairs = choose_query_pairs(
        trajectories,
        reward_model=models,
        query_count=3,
        fragment_length=2,
        active_learning=True,
        convert_traj=convert_traj,
        add_partial_to_predictions=False,
        dropout_samples=2,
        dropout_p=0.25,
        active_learning_batches=4,
        active_query_strategy="ensemble",
    )

    assert 0 < len(pairs) <= 3


def make_rated_pairs(n: int) -> list[Preference]:
    pairs = []
    for index in range(n):
        better = make_trajectory(2, reward=2.0 + index, partial=1.0)
        worse = make_trajectory(2, reward=0.5, partial=0.2)
        pairs.append(Preference(better, worse, 1.0))
    return pairs


def test_train_preference_reward_model_pairwise_and_delta():
    for use_delta_loss in (False, True):
        th.manual_seed(0)
        model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,))
        pairs = make_rated_pairs(6)

        train_preference_reward_model(
            model,
            pairs[:4],
            pairs[4:],
            convert_traj=convert_traj,
            use_delta_loss=use_delta_loss,
            batch_size=2,
            epochs=2,
            patience=5,
        )

        output = model(th.zeros((1, FEATURE_DIM)))
        assert th.isfinite(output).all()


def test_partial_reward_tensor_normalization():
    pairs = [Preference(make_trajectory(2, reward=1.0, partial=3.0), make_trajectory(2, reward=0.0, partial=1.0), 1.0)]

    raw = partial_reward_tensor(pairs, "t1")
    normalized = partial_reward_tensor(pairs, "t1", partial_mean=2.0, partial_std=2.0)

    assert th.allclose(raw, th.full((1, 2, 1), 3.0))
    assert th.allclose(normalized, th.full((1, 2, 1), 0.5))


def test_delta_loss_prefers_matching_partials():
    loss = DeltaLoss()
    y1 = th.zeros((1, 2, 1))
    y2 = th.zeros((1, 2, 1))
    good_base = th.full((1, 2, 1), 5.0)
    bad_base = th.zeros((1, 2, 1))
    target = th.ones(1)

    aligned = loss(y1, y2, good_base, bad_base, target)
    misaligned = loss(y1, y2, bad_base, good_base, target)

    assert float(aligned) < float(misaligned)


def test_pairwise_loss_prefers_higher_first_input():
    loss = PairwiseLoss()
    high = th.full((1, 2, 1), 3.0)
    low = th.zeros((1, 2, 1))
    target = th.ones(1)

    assert float(loss(high, low, target)) < float(loss(low, high, target))


def test_pretrain_reward_model_targets():
    for target in ("partial", "residual", "true"):
        th.manual_seed(0)
        model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,))
        pretrain_reward_model(
            model,
            [make_trajectory(3, reward=1.0, partial=0.5)],
            convert_traj,
            target=target,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
        )

    with pytest.raises(ValueError, match="Unsupported pretrain target"):
        pretrain_reward_model(
            RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,)),
            [make_trajectory(1, reward=1.0)],
            convert_traj,
            target="bogus",
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
        )


def test_reward_model_io_stats():
    model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,))
    mean, std = reward_model_io_stats(model, [make_trajectory(2, reward=1.0)], convert_traj)

    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert reward_model_io_stats(model, [], convert_traj) == (None, None)


def test_split_preference_k_folds_uses_all_pairs_once():
    pairs = [Preference(None, None, float(index)) for index in range(11)]

    folds = split_preference_k_folds(pairs, 5)
    flattened = [pair for fold in folds for pair in fold]

    assert len(folds) == 5
    assert {id(pair) for pair in flattened} == {id(pair) for pair in pairs}
    assert max(len(fold) for fold in folds) - min(len(fold) for fold in folds) <= 1


def test_schedules():
    assert query_schedule(1400, 5) == [280, 280, 280, 280, 280]
    assert query_schedule(7, 3) == [3, 2, 2]
    assert policy_training_schedule(10, 3) == [3, 3, 4]
    assert policy_training_schedule(10, 3, timesteps_per_round=5) == [5, 5, 5]
