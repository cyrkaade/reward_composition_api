"""The STARC ladder must keep its calibrated weights and its ordering.

The alignment values are a calibration, so they are pinned loosely (the
estimator has ~0.02 of seed noise and ~0.03 of hyperparameter spread at the
bottom of the ladder).  The *ordering* is the property the grid experiment
depends on, so that is pinned strictly.
"""

from __future__ import annotations

import numpy as np
import pytest

from rcomp.partials import PartialRegistry, load_partial_reference
from partials.lunar_lander_starc_alignment import STARC_LADDER


def _partial(name: str):
    return load_partial_reference(
        f"lunar_lander_starc_alignment:{name}",
        "box2d",
        PartialRegistry(),
    ).create("LunarLander-v3")


@pytest.mark.parametrize("name", list(STARC_LADDER))
def test_only_the_terminal_weight_differs_between_rungs(name):
    """Shaping and fuel are exactly the true reward's at every rung."""
    underlying = _partial(name).obj
    _, terminal_weight, _ = STARC_LADDER[name]

    assert underlying.w_distance == 1.0
    assert underlying.w_speed == 1.0
    assert underlying.w_tilt == 1.0
    assert underlying.w_leg == 1.0
    assert underlying.w_side == 1.0
    assert underlying.w_main == 1.0
    assert underlying.w_game_over == terminal_weight
    assert underlying.w_landed == terminal_weight


@pytest.mark.parametrize("name", list(STARC_LADDER))
def test_terminal_payoffs_follow_the_calibrated_weight(name):
    partial = _partial(name)
    _, terminal_weight, _ = STARC_LADDER[name]
    resting = np.zeros(8, dtype=np.float64)
    crashing = resting.copy()
    crashing[2] = 1.0

    landed = partial.step(resting, 0, resting, 100.0, True, False, {}).partial
    partial.reset({})
    crashed = partial.step(resting, 0, crashing, -100.0, True, False, {}).partial

    assert landed == pytest.approx(100.0 * terminal_weight)
    assert crashed == pytest.approx(-100.0 * terminal_weight)


def test_the_ladder_is_strictly_increasing_in_both_target_and_weight():
    """A grid indexed by alignment is meaningless if the rungs are not ordered."""
    targets = [entry[0] for entry in STARC_LADDER.values()]
    weights = [entry[1] for entry in STARC_LADDER.values()]
    measured = [entry[2] for entry in STARC_LADDER.values()]

    assert targets == sorted(targets)
    assert weights == sorted(weights)
    assert measured == sorted(measured)
    assert targets == [0.2, 0.4, 0.6, 0.8, 1.0]


def test_held_out_measurements_sit_near_their_targets():
    for name, (target, _, measured) in STARC_LADDER.items():
        assert abs(measured - target) < 0.05, name


def test_the_top_rung_is_the_true_reward_exactly():
    """lls_a100's 1.0 is an identity, so it must really be the identity."""
    _, weight, _ = STARC_LADDER["lls_a100"]
    assert weight == 1.0
    underlying = _partial("lls_a100").obj
    assert all(
        getattr(underlying, attr) == 1.0
        for attr in ("w_distance", "w_speed", "w_tilt", "w_leg",
                     "w_side", "w_main", "w_game_over", "w_landed")
    )


def test_the_bottom_rungs_really_do_invert_the_terminal():
    """Documented consequence of asking for alignment below 0.293."""
    for name in ("lls_a20", "lls_a40"):
        assert STARC_LADDER[name][1] < 0.0, f"{name} should reward crashing"
    for name in ("lls_a60", "lls_a80", "lls_a100"):
        assert STARC_LADDER[name][1] > 0.0
