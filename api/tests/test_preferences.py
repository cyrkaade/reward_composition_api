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


def test_delta_loss_alpha_scales_partial_base():
    loss = DeltaLoss()
    y1 = th.zeros((1, 2, 1))
    y2 = th.zeros((1, 2, 1))
    good_base = th.full((1, 2, 1), 2.0)
    bad_base = th.zeros((1, 2, 1))
    target = th.ones(1)

    weak = loss(y1, y2, good_base, bad_base, target, alpha=0.5)
    strong = loss(y1, y2, good_base, bad_base, target, alpha=2.0)

    assert float(strong) < float(weak)


def test_learned_alpha_stays_anchored():
    th.manual_seed(0)
    model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,), learn_alpha=True, alpha_init=1.0)
    pairs = make_rated_pairs(6)

    train_preference_reward_model(
        model,
        pairs[:4],
        pairs[4:],
        convert_traj=convert_traj,
        use_delta_loss=True,
        batch_size=2,
        epochs=3,
        patience=5,
        partial_alpha=1.0,
        partial_alpha_penalty=1.0,
    )

    assert model.alpha is not None
    assert abs(float(model.alpha) - 1.0) < 1.0


def test_dual_loss_trains_partial_head():
    th.manual_seed(0)
    model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,), predict_partial=True)
    pairs = make_rated_pairs(6)

    train_preference_reward_model(
        model,
        pairs[:4],
        pairs[4:],
        convert_traj=convert_traj,
        use_delta_loss=True,
        batch_size=2,
        epochs=2,
        patience=5,
        partial_prediction_coef=1.0,
    )

    x = th.zeros((1, FEATURE_DIM))
    assert model.partial_head is not None
    assert th.isfinite(model(x)).all()
    assert th.isfinite(model.predict_partial(x)).all()


def test_batchnorm_output_off_by_default_and_optional():
    # off by default: no batch-norm layer, forward unchanged
    plain = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,))
    assert plain.output_bn is None

    th.manual_seed(0)
    model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,), batchnorm_output=True)
    pairs = make_rated_pairs(6)
    train_preference_reward_model(
        model, pairs[:4], pairs[4:], convert_traj=convert_traj,
        use_delta_loss=True, batch_size=2, epochs=3, patience=5,
    )
    model.eval()
    # single-sample inference must work (running stats, not batch stats)
    out = model(th.zeros((1, FEATURE_DIM)))
    assert th.isfinite(out).all()
    # running stats moved away from their init during training
    assert not th.allclose(model.output_bn.running_var, th.ones(1))


def test_per_state_gate_trains_and_stays_in_unit_interval():
    plain = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,))
    assert plain.gate_head is None

    th.manual_seed(0)
    model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,), gate_partial=True)
    pairs = make_rated_pairs(6)
    train_preference_reward_model(
        model, pairs[:4], pairs[4:], convert_traj=convert_traj,
        use_delta_loss=True, batch_size=2, epochs=3, patience=5,
    )
    g = model.gate(th.zeros((5, FEATURE_DIM)))
    assert th.all(g >= 0) and th.all(g <= 1)  # sigmoid gate stays in [0,1]

    from rcomp.rewards.preferences import gate_statistics
    stats = gate_statistics(model, [make_trajectory(3, reward=1.0, partial=0.5)], convert_traj)
    assert 0.0 <= stats["mean"] <= 1.0


def test_gate_init_starts_at_the_requested_value():
    """gate_init=1 must start at 'trust the partial fully' (the naive baseline),
    so any shrinkage below 1 is something the preferences actually paid for."""
    for target in (0.5, 0.95):
        model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,), gate_partial=True, gate_init=target)
        g = model.gate(th.randn(7, FEATURE_DIM)).detach().reshape(-1)
        assert th.allclose(g, th.full_like(g, target), atol=1e-4)
        # weights are zeroed, so the init is a constant regardless of the input
        assert float(g.std()) < 1e-6


def test_gate_head_is_excluded_from_l1_regularization():
    """L1 on a bounded [0,1] gate does not regularize capacity, it just drags the
    gate toward 0.5 - and in naive mode it acts on a head with no gradient yet."""
    from rcomp.rewards.model import RegularizationLoss

    model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,), gate_partial=True, gate_init=0.95)
    before = float(RegularizationLoss(lambda_reg=1.0)(model))
    with th.no_grad():
        model.gate_head.bias.add_(100.0)
    after = float(RegularizationLoss(lambda_reg=1.0)(model))
    assert after == pytest.approx(before)


def test_gate_holdout_fits_on_pairs_the_model_did_not_train_on():
    """With gate_holdout the gate must be fit on held-out preferences: on the
    trunk's own training pairs the residual is memorized away and the gate
    collapses regardless of whether the partial is useful."""
    th.manual_seed(0)
    random.seed(0)
    model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,), gate_partial=True, gate_init=0.95)
    pairs = make_rated_pairs(20)
    train_preference_reward_model(
        model, pairs[:10], pairs[10:], convert_traj=convert_traj,
        use_delta_loss=False, batch_size=4, epochs=3, patience=5,
        gate_holdout=True, gate_learning_rate=1e-3, gate_patience=2,
    )
    g = model.gate(th.zeros((5, FEATURE_DIM)))
    assert th.all(g >= 0) and th.all(g <= 1)


def test_gate_prior_penalty_keeps_the_gate_closer_to_one():
    """The penalty makes 'use the partial fully' the null hypothesis."""
    results = {}
    for penalty in (0.0, 50.0):
        th.manual_seed(0)
        random.seed(0)
        model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,), gate_partial=True, gate_init=0.5)
        pairs = make_rated_pairs(16)
        train_preference_reward_model(
            model, pairs[:8], pairs[8:], convert_traj=convert_traj,
            use_delta_loss=True, batch_size=4, epochs=6, patience=10,
            gate_prior_penalty=penalty,
        )
        results[penalty] = float(model.gate(th.zeros((8, FEATURE_DIM))).detach().mean())
    assert results[50.0] > results[0.0]


def test_gate_partial_error_diagnostic_detects_a_gate_that_tracks_partial_error():
    """A gate that closes exactly where the partial disagrees with the true
    reward must produce a negative corr(g, |partial error|); the metric is what
    tells us whether the gate learned trust or just rescaled the partial."""
    from rcomp.rewards.preferences import gate_partial_error_stats

    class IdealGate:
        """Stub standing in for a gate that perfectly knows where the partial is
        wrong: open when the partial matches the true reward, closed otherwise."""

        gate_head = object()

        def gate(self, x):  # x is (n_traj, traj_len, [obs, act, partial])
            obs, partial = x[..., 0], x[..., 2]
            return th.where(th.abs(obs - partial) < 0.5, 0.9, 0.1).unsqueeze(-1)

    # true reward and partial both have to vary, or the error is constant and the
    # correlation is undefined. Two trustworthy groups, two untrustworthy ones.
    trajectories = (
        [make_trajectory(4, reward=2.0, partial=2.0) for _ in range(2)]
        + [make_trajectory(4, reward=0.0, partial=0.0) for _ in range(2)]
        + [make_trajectory(4, reward=2.0, partial=0.0) for _ in range(2)]
        + [make_trajectory(4, reward=0.0, partial=2.0) for _ in range(2)]
    )

    stats = gate_partial_error_stats(IdealGate(), trajectories, convert_traj)
    assert stats["n_states"] == 32
    assert stats["corr_gate_partial_error"] < -0.9
    assert stats["mean_gate_high_error"] < stats["mean_gate_low_error"]

    # A gate that has collapsed to a constant carries no per-state information,
    # so the correlation is undefined (None) rather than a spurious number. This
    # is the case a fixed alpha would reproduce exactly.
    class ConstantGate:
        gate_head = object()

        def gate(self, x):
            return th.full((*x.shape[:-1], 1), 0.3)

    flat = gate_partial_error_stats(ConstantGate(), trajectories, convert_traj)
    assert flat["corr_gate_partial_error"] is None


def test_gate_diagnostic_returns_json_safe_values():
    """A degenerate partial (no variance in the error) must not leak NaN into
    metadata.json, which would make the file invalid JSON."""
    from rcomp.rewards.preferences import gate_partial_error_stats

    model = RewardModel(input_size=FEATURE_DIM, hidden_sizes=(8,), gate_partial=True)
    identical = [make_trajectory(4, reward=1.0, partial=1.0) for _ in range(4)]
    stats = gate_partial_error_stats(model, identical, convert_traj)
    for key, value in stats.items():
        assert value is None or np.isfinite(value), f"{key} is not JSON-safe: {value}"


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
