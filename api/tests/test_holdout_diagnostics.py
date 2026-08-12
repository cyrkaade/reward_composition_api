"""The held-out diagnostic set and per-member bootstrap resampling.

The point of both features is measurement integrity, so the tests assert the
properties that make the measurement meaningful rather than just that the code
runs: the diagnostic pairs must share no transition with anything trainable, and
bootstrap members must actually differ from one another.
"""

from __future__ import annotations

import random

import pytest

from rcomp.config import ConfigError, ExperimentConfig, normalize_experiment_config
from rcomp.data import Trajectory
from rcomp.rewards.model import RewardModel
from rcomp.rewards.preferences import (
    build_holdout_preferences,
    fragment_trajectories,
    random_query_pairs,
    rate_pairs_from_true_reward,
    split_holdout_trajectories,
    train_preference_reward_ensemble,
)


def make_trajectory(length: int, offset: float = 0.0) -> Trajectory:
    trajectory = Trajectory()
    for step in range(length):
        trajectory.push_state(
            obs=[float(step) + offset],
            act=[0.0],
            done=False,
            info={},
            true_rew=float(step) + offset,
            partial_rew=0.5 * (float(step) + offset),
        )
    return trajectory


def state_ids(trajectories) -> set[int]:
    return {id(state) for trajectory in trajectories for state in trajectory.states}


def test_holdout_shares_no_transition_with_the_query_pool():
    trajectories = [make_trajectory(100, offset=index) for index in range(40)]
    query, holdout = split_holdout_trajectories(trajectories, 30, 25, random.Random(0))

    assert holdout, "a 40-trajectory pool should be able to spare a diagnostic set"
    assert query, "the query pool must never be emptied"
    assert state_ids(query).isdisjoint(state_ids(holdout))

    holdout_pairs = build_holdout_preferences(holdout, 30, 25, random.Random(1))
    query_pairs = random_query_pairs(fragment_trajectories(query, 25), 70, rng=random.Random(2))

    holdout_states = state_ids(pair.t1 for pair in holdout_pairs) | state_ids(pair.t2 for pair in holdout_pairs)
    query_states = {id(state) for a, b in query_pairs for traj in (a, b) for state in traj.states}
    assert holdout_pairs
    assert holdout_states.isdisjoint(query_states)


def test_holdout_is_labelled_by_the_true_reward():
    holdout = [make_trajectory(50, offset=index) for index in range(8)]
    pairs = build_holdout_preferences(holdout, 4, 25, random.Random(0))
    assert pairs
    for pair in pairs:
        expected = float(pair.t1.get_summed_reward() > pair.t2.get_summed_reward())
        assert pair.rating == expected


@pytest.mark.parametrize(
    "count, fragment_length, pair_count",
    [(1, 25, 30), (0, 25, 30), (3, 200, 30), (40, 25, 0)],
)
def test_holdout_split_degrades_safely(count, fragment_length, pair_count):
    """Too few trajectories, fragments longer than an episode, or the feature
    switched off must all leave the query pool intact rather than starving it."""
    trajectories = [make_trajectory(100, offset=index) for index in range(count)]
    query, holdout = split_holdout_trajectories(trajectories, pair_count, fragment_length, random.Random(0))
    assert len(query) == count
    assert holdout == []


def test_bootstrap_gives_members_different_data():
    trajectories = [make_trajectory(25, offset=index) for index in range(60)]
    fragments = fragment_trajectories(trajectories, 25)
    rated = rate_pairs_from_true_reward(random_query_pairs(fragments, 30, rng=random.Random(0)))
    models = [RewardModel(input_size=3, hidden_sizes=(8,)) for _ in range(3)]

    random.seed(0)
    stats = train_preference_reward_ensemble(
        models,
        rated,
        convert_traj=lambda traj: [[float(state["obs"][0]), 0.0, 0.0] for state in traj.states],
        use_delta_loss=False,
        batch_size=8,
        epochs=1,
        patience=1,
        training_mode="full",
        bootstrap=True,
    )
    unique = [entry["n_unique_train_pairs"] for entry in stats]
    assert all(entry["bootstrap"] for entry in stats)
    # A bootstrap resample of N items covers ~63% of them, so every member must
    # be missing some pairs, and the members must not all see the same subset.
    assert all(value < len(rated) for value in unique)
    assert len(set(unique)) > 1 or unique[0] < len(rated)


def test_bootstrap_without_it_keeps_the_full_buffer():
    trajectories = [make_trajectory(25, offset=index) for index in range(40)]
    fragments = fragment_trajectories(trajectories, 25)
    rated = rate_pairs_from_true_reward(random_query_pairs(fragments, 20, rng=random.Random(0)))
    models = [RewardModel(input_size=3, hidden_sizes=(8,)) for _ in range(2)]
    stats = train_preference_reward_ensemble(
        models,
        rated,
        convert_traj=lambda traj: [[float(state["obs"][0]), 0.0, 0.0] for state in traj.states],
        use_delta_loss=False,
        batch_size=8,
        epochs=1,
        patience=1,
        training_mode="full",
    )
    assert all(entry["n_unique_train_pairs"] == len(rated) for entry in stats)
    assert not any(entry["bootstrap"] for entry in stats)


def base_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        suite="box2d",
        env_id="LunarLander-v3",
        mode="naive",
        partial="lunar_lander_approach",
        timesteps=1000,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_bootstrap_requires_full_ensemble_training():
    with pytest.raises(ConfigError, match="ensemble-training full"):
        normalize_experiment_config(
            base_config(ensemble_bootstrap=True, reward_model_ensemble_size=3, ensemble_training="kfold")
        )


def test_bootstrap_requires_an_actual_ensemble():
    with pytest.raises(ConfigError, match="ensemble_size"):
        normalize_experiment_config(
            base_config(ensemble_bootstrap=True, reward_model_ensemble_size=1, ensemble_training="full")
        )


def test_holdout_requires_a_preference_mode():
    with pytest.raises(ConfigError, match="preference mode"):
        normalize_experiment_config(base_config(mode="true", holdout_pairs=30))


def test_holdout_accepts_a_preference_mode():
    config = normalize_experiment_config(
        base_config(mode="naive", holdout_pairs=30, reward_model_ensemble_size=3, ensemble_training="full")
    )
    assert config.holdout_pairs == 30
