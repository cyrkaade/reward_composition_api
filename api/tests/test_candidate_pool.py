"""B-Pref's disagreement sampling draws an INDEPENDENT candidate pool and keeps
the per-pair top-k. The legacy selector instead scores whole perfect matchings
and keeps the best one, which is a different estimator: inside a matching every
fragment appears at most once, so the chosen set is constrained by a global
matching rather than by per-pair informativeness. These tests pin the difference
so the active-learning comparison is against the literature's method.
"""

from __future__ import annotations

import random

import pytest

from rcomp.config import ConfigError, ExperimentConfig, normalize_experiment_config
from rcomp.data import Trajectory
from rcomp.rewards.preferences import (
    _candidate_pool_pairs,
    _preference_variance_pairs,
    choose_query_pairs,
)


def fragments(n: int) -> list[Trajectory]:
    out = []
    for i in range(n):
        t = Trajectory()
        t.push_state([float(i)], [0.0], False, {}, true_rew=float(i), partial_rew=0.0)
        out.append(t)
    return out


def ensemble_returns(n: int, disagree_on: set[int]) -> list[list[float]]:
    """Two members that agree everywhere except on the listed fragments."""
    a = [0.0] * n
    b = [0.0] * n
    for i in disagree_on:
        a[i], b[i] = 50.0, -50.0
    return [a, b]


def test_pool_selects_the_pairs_the_ensemble_disagrees_about():
    n = 20
    frags = fragments(n)
    # ONE contested fragment: a pair of two equally contested fragments would
    # cancel (see test_opposed_fragments_cancel), so a single one keeps the
    # property unambiguous.
    picked = _candidate_pool_pairs(frags, ensemble_returns(n, {7}), query_count=3,
                                   pool_multiplier=30, rng=random.Random(0))
    assert len(picked) == 3
    index = {id(f): i for i, f in enumerate(frags)}
    for x, y in picked:
        assert 7 in (index[id(x)], index[id(y)])


def test_opposed_fragments_cancel():
    """Two fragments the ensemble is equally split on carry NO disagreement about
    which is preferred: member A ranks both high, member B ranks both low, and
    both agree the comparison is a coin flip. Disagreement sampling should rank
    such a pair below one contested fragment against a settled one."""
    frags = fragments(20)
    returns = ensemble_returns(20, {4, 9})
    picked = _candidate_pool_pairs(frags, returns, query_count=1, pool_multiplier=60,
                                   rng=random.Random(5))
    index = {id(f): i for i, f in enumerate(frags)}
    (a, b), = picked
    chosen = {index[id(a)], index[id(b)]}
    assert len(chosen & {4, 9}) == 1, "the top pair should mix a contested fragment with a settled one"


def test_pool_returns_exactly_query_count_and_no_self_pairs():
    rng = random.Random(1)
    frags = fragments(30)
    picked = _candidate_pool_pairs(frags, ensemble_returns(30, {1}), query_count=7,
                                   pool_multiplier=10, rng=rng)
    assert len(picked) == 7
    assert all(a is not b for a, b in picked)


def test_pool_can_reuse_a_fragment_but_a_matching_cannot():
    """The substantive difference. With one clearly contested fragment, an
    independent pool may query it repeatedly against different partners; a
    perfect matching can use it at most once."""
    frags = fragments(40)
    returns = ensemble_returns(40, {5})
    pool = _candidate_pool_pairs(frags, returns, query_count=5, pool_multiplier=20,
                                 rng=random.Random(2))
    index = {id(f): i for i, f in enumerate(frags)}
    uses = sum(1 for a, b in pool if 5 in (index[id(a)], index[id(b)]))
    assert uses > 1, "an independent pool should be able to re-query a contested fragment"

    matching = _preference_variance_pairs(frags, returns, query_count=5, n_batches=32,
                                          rng=random.Random(2))
    uses_m = sum(1 for a, b in matching if 5 in (index[id(a)], index[id(b)]))
    assert uses_m <= 1, "a perfect matching can use each fragment at most once"


def test_pool_degrades_gracefully_on_tiny_fragment_sets():
    frags = fragments(2)
    picked = _candidate_pool_pairs(frags, ensemble_returns(2, set()), query_count=5,
                                   pool_multiplier=10, rng=random.Random(3))
    assert 0 < len(picked) <= 5
    assert _candidate_pool_pairs(fragments(1), [[0.0]], 3, 10, rng=random.Random(3)) == []


def test_choose_query_pairs_routes_to_the_requested_protocol():
    from rcomp.rewards.model import RewardModel

    trajectory = Trajectory()
    for i in range(200):
        trajectory.push_state([float(i % 7)], [0.0], False, {}, true_rew=float(i), partial_rew=0.0)
    models = [RewardModel(input_size=3, hidden_sizes=(8,)) for _ in range(2)]

    def convert(t):
        return [[s["obs"][0], s["act"][0], s["partial_rew"]] for s in t.states]

    for protocol in ("matching", "pool"):
        pairs = choose_query_pairs(
            [trajectory], models, query_count=5, fragment_length=10, active_learning=True,
            convert_traj=convert, add_partial_to_predictions=False, dropout_samples=2,
            dropout_p=0.1, active_learning_batches=8, active_query_strategy="ensemble",
            rng=random.Random(0), candidate_protocol=protocol, pool_multiplier=10,
        )
        assert len(pairs) == 5, protocol


def test_protocol_defaults_to_legacy_and_validates():
    assert ExperimentConfig().active_candidate_protocol == "matching"
    assert ExperimentConfig().active_pool_multiplier == 10
    with pytest.raises(ConfigError):
        normalize_experiment_config(
            ExperimentConfig(suite="box2d", mode="feedback", active_candidate_protocol="topk")
        )
    with pytest.raises(ConfigError):
        normalize_experiment_config(
            ExperimentConfig(suite="box2d", mode="feedback", active_pool_multiplier=0)
        )
    ok = normalize_experiment_config(
        ExperimentConfig(suite="box2d", mode="feedback", active_candidate_protocol="pool")
    )
    assert ok.active_candidate_protocol == "pool"
