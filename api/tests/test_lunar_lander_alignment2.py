"""Pin the five supplied alignment-table partials and all displayed weights."""

from __future__ import annotations

import pytest

from partials.lunar_lander_alignment2 import ALIGNMENT_ROWS
from rcomp.partials import PartialRegistry, load_partial_reference


EXPECTED = {
    "partial_0.5_0.0_0.0_1.0": (
        0.297506305575371,
        0.212763247747556,
        (0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ),
    "partial_1.0_0.0_1.0_1.0": (
        0.240131175518036,
        0.747968668313665,
        (1.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0),
    ),
    "partial_0.5_0.5_0.0_0.5": (
        0.181834864616394,
        0.955797103877229,
        (0.5, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0),
    ),
    "partial_0.0_1.0_0.0_1.0": (
        0.117974838614464,
        0.899949398128207,
        (0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ),
    "partial_0.5_1.0_0.5_1.0": (
        0.0596099875867367,
        0.982347842944448,
        (0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0),
    ),
}

ATTRS = (
    "w_distance",
    "w_speed",
    "w_tilt",
    "w_leg",
    "w_side",
    "w_main",
    "w_game_over",
    "w_landed",
)


def test_table_metadata_and_rows_are_pinned_exactly():
    assert list(ALIGNMENT_ROWS) == list(EXPECTED)
    for name, (alignment, pcc, _) in EXPECTED.items():
        assert ALIGNMENT_ROWS[name]["alignment"] == pytest.approx(alignment)
        assert ALIGNMENT_ROWS[name]["pcc"] == pytest.approx(pcc)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_every_table_partial_resolves_with_all_displayed_weights(name):
    partial = load_partial_reference(
        f"lunar_lander_alignment2:{name}",
        "box2d",
        PartialRegistry(),
    ).create("LunarLander-v3")

    expected_weights = EXPECTED[name][2]
    assert tuple(getattr(partial.obj, attr) for attr in ATTRS) == expected_weights
