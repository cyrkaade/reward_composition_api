"""Bradley-Terry pretraining, the pretrain/query rollout split, and tanh bounding.

The point of the BT variant is calibration, not ranking: MSE pretraining fixes the
output magnitude to the partial's raw scale, so the model can end up confidently
wrong under the Bradley-Terry loss even while ranking better than chance. These
tests pin the property that motivated the flag - after BT pretraining the held-out
BT loss must be below chance (ln 2), which the MSE variant does not guarantee.
"""

from __future__ import annotations

import math

import pytest
import torch as th

from rcomp.config import ConfigError, ExperimentConfig, normalize_experiment_config
from rcomp.data import Trajectory
from rcomp.rewards.model import PairwiseLoss, RewardModel
from rcomp.rewards.preferences import (
    pretrain_reward_model,
    pretrain_reward_model_bt,
    rate_pairs_from_target,
    rated_pairs_to_tensors,
    fragment_trajectories,
    random_query_pairs,
)

CHANCE = math.log(2.0)


def make_trajectory(n_states: int, rng, scale: float = 30.0) -> Trajectory:
    """States whose partial reward is a large-scale linear function of obs[0], so
    an MSE fit lands on a big output magnitude - the case the BT variant fixes."""
    trajectory = Trajectory()
    for _ in range(n_states):
        obs = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)]
        partial = scale * obs[0]
        trajectory.push_state(obs, [0.0], False, {}, true_rew=partial + rng.gauss(0, 5), partial_rew=partial)
    return trajectory


def convert(trajectory: Trajectory):
    return [[*state["obs"], state["act"][0], state["partial_rew"] * 0.0] for state in trajectory.states]


def held_out_bt_loss(model, trajectories, fragment_length, target="partial"):
    fragments = fragment_trajectories(trajectories, fragment_length)
    pairs = rate_pairs_from_target(random_query_pairs(fragments, 200), target)
    with th.no_grad():
        x1, x2, ratings = rated_pairs_to_tensors(pairs, convert)
        return float(PairwiseLoss()(model(x1), model(x2), ratings).mean().item())


@pytest.fixture
def data():
    import random

    rng = random.Random(0)
    train = [make_trajectory(50, rng) for _ in range(20)]
    test = [make_trajectory(50, rng) for _ in range(20)]
    return train, test


def test_bt_pretraining_lands_below_chance(data):
    import random

    random.seed(0)
    th.manual_seed(0)
    train, test = data
    model = RewardModel(input_size=4, hidden_sizes=(64,))
    stats = pretrain_reward_model_bt(
        model, train, convert, target="partial", epochs=40, batch_size=32,
        learning_rate=1e-3, fragment_length=10,
    )
    assert stats["pretrain_pairs"] > 0
    assert held_out_bt_loss(model, test, 10) < CHANCE


def test_mse_pretraining_can_overshoot_the_bt_scale(data):
    """Regression guard for the failure the BT flag exists to fix: the MSE-fit
    model produces a much larger margin than BT-fit one on the same data."""
    import random

    random.seed(0)
    th.manual_seed(0)
    train, _ = data
    mse_model = RewardModel(input_size=4, hidden_sizes=(64,))
    pretrain_reward_model(mse_model, train, convert, target="partial", epochs=40, batch_size=32, learning_rate=1e-3)

    random.seed(0)
    th.manual_seed(0)
    bt_model = RewardModel(input_size=4, hidden_sizes=(64,))
    pretrain_reward_model_bt(
        bt_model, train, convert, target="partial", epochs=40, batch_size=32,
        learning_rate=1e-3, fragment_length=10,
    )

    with th.no_grad():
        x = th.as_tensor([convert(t) for t in train], dtype=th.float32)
        assert float(mse_model(x).std()) > 5 * float(bt_model(x).std())


def test_bt_labels_come_from_the_partial_not_the_true_reward():
    import random

    rng = random.Random(1)
    t1, t2 = make_trajectory(5, rng), make_trajectory(5, rng)
    # force a disagreement between partial and true ordering
    for state in t1.states:
        state["partial_rew"], state["rew"] = 10.0, -10.0
    for state in t2.states:
        state["partial_rew"], state["rew"] = -10.0, 10.0
    assert rate_pairs_from_target([(t1, t2)], "partial")[0].rating == 1.0
    assert rate_pairs_from_target([(t1, t2)], "true")[0].rating == 0.0
    assert rate_pairs_from_target([(t1, t2)], "residual")[0].rating == 0.0


def test_bt_pretraining_skips_when_too_few_fragments(data):
    train, _ = data
    model = RewardModel(input_size=4, hidden_sizes=(16,))
    assert pretrain_reward_model_bt(
        model, train[:1], convert, target="partial", epochs=2, batch_size=8,
        learning_rate=1e-3, fragment_length=1000,
    ) is None


def test_tanh_output_bounds_the_model():
    th.manual_seed(0)
    bounded = RewardModel(input_size=4, hidden_sizes=(32,), tanh_output=True)
    unbounded = RewardModel(input_size=4, hidden_sizes=(32,))
    x = th.randn(8, 5, 4) * 50.0
    assert float(bounded(x).abs().max()) <= 1.0
    assert not unbounded.tanh_output and bounded.tanh_output
    # sanity: the unbounded twin is not accidentally bounded by initialisation
    with th.no_grad():
        unbounded.head.weight.mul_(100.0)
    assert float(unbounded(x).abs().max()) > 1.0


class DummyRuntime:
    normalize_partial = False


class SplitHarness:
    """Minimal stand-in exercising RlhfTrainer.split_for_pretraining only."""

    def __init__(self, config):
        from rcomp.trainer import RlhfTrainer

        self.split = RlhfTrainer.split_for_pretraining.__get__(self)
        self._needs_pretraining = RlhfTrainer._needs_pretraining.__get__(self)
        self.config = config
        self.pretraining_done = False


def test_pretrain_holdout_splits_disjointly():
    trajectories = list(range(10))
    on = SplitHarness(ExperimentConfig(pretrain_holdout=True, pretrain_reward_model=True))
    fit, query = on.split(trajectories)
    assert fit == trajectories[:5] and query == trajectories[5:]
    assert not set(fit) & set(query)


def test_pretrain_holdout_is_off_by_default_and_after_pretraining():
    trajectories = list(range(10))
    off = SplitHarness(ExperimentConfig(pretrain_reward_model=True))
    assert off.split(trajectories) == (trajectories, trajectories)

    done = SplitHarness(ExperimentConfig(pretrain_holdout=True, pretrain_reward_model=True))
    done.pretraining_done = True
    assert done.split(trajectories) == (trajectories, trajectories)


def test_new_flags_default_to_previous_behaviour():
    config = ExperimentConfig()
    assert config.pretrain_loss == "mse"
    assert config.pretrain_holdout is False
    assert config.tanh_model_reward is False
    assert config.pretrain_pairs == 0


def test_config_validation():
    with pytest.raises(ConfigError):
        normalize_experiment_config(ExperimentConfig(suite="box2d", mode="feedback", pretrain_loss="huber"))
    with pytest.raises(ConfigError):
        normalize_experiment_config(
            ExperimentConfig(suite="box2d", mode="feedback", tanh_model_reward=True, batchnorm_model_reward=True)
        )
    with pytest.raises(ConfigError):
        normalize_experiment_config(ExperimentConfig(suite="box2d", mode="true", tanh_model_reward=True))
    ok = normalize_experiment_config(
        ExperimentConfig(suite="box2d", mode="feedback", pretrain_loss="bt", tanh_model_reward=True)
    )
    assert ok.pretrain_loss == "bt" and ok.tanh_model_reward is True
