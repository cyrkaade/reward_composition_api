"""The five LunarLander partial rewards from the supplied alignment table.

The table reports both an ``alignment`` score and a per-step PCC for every
row.  Those measurements are metadata only; the reward itself is defined by
the eight displayed term weights.  The implementation reuses the faithful
term-by-term LunarLander decomposition in :mod:`lunar_lander_alignment`.

The registered names intentionally match the table's ``function name`` column,
even where that compact name does not encode every displayed engine weight.
"""

from __future__ import annotations

from partials.lunar_lander_alignment import LunarLanderWeightedPartial


# name -> table measurements and the eight displayed reward-term weights.
# Insertion order matches the supplied image (highest alignment first).
ALIGNMENT_ROWS = {
    "partial_0.5_0.0_0.0_1.0": {
        "alignment": 0.297506305575371,
        "pcc": 0.212763247747556,
        "weights": dict(
            distance=0.5,
            speed=0.0,
            tilt=0.0,
            leg=1.0,
            side_engine=1.0,
            main_engine=1.0,
            game_over=1.0,
            landed=1.0,
        ),
    },
    "partial_1.0_0.0_1.0_1.0": {
        "alignment": 0.240131175518036,
        "pcc": 0.747968668313665,
        "weights": dict(
            distance=1.0,
            speed=0.0,
            tilt=1.0,
            leg=1.0,
            side_engine=0.5,
            main_engine=0.5,
            game_over=1.0,
            landed=1.0,
        ),
    },
    "partial_0.5_0.5_0.0_0.5": {
        "alignment": 0.181834864616394,
        "pcc": 0.955797103877229,
        "weights": dict(
            distance=0.5,
            speed=0.5,
            tilt=0.0,
            leg=1.0,
            side_engine=0.5,
            main_engine=0.5,
            game_over=1.0,
            landed=1.0,
        ),
    },
    "partial_0.0_1.0_0.0_1.0": {
        "alignment": 0.117974838614464,
        "pcc": 0.899949398128207,
        "weights": dict(
            distance=0.0,
            speed=1.0,
            tilt=0.0,
            leg=1.0,
            side_engine=1.0,
            main_engine=1.0,
            game_over=1.0,
            landed=1.0,
        ),
    },
    "partial_0.5_1.0_0.5_1.0": {
        "alignment": 0.0596099875867367,
        "pcc": 0.982347842944448,
        "weights": dict(
            distance=0.5,
            speed=1.0,
            tilt=0.5,
            leg=1.0,
            side_engine=1.0,
            main_engine=1.0,
            game_over=1.0,
            landed=1.0,
        ),
    },
}


def register(registry) -> None:
    for name, row in ALIGNMENT_ROWS.items():
        weights = dict(row["weights"])
        registry.register(
            name=name,
            suite="box2d",
            factory=(
                lambda env_id, _weights=weights: LunarLanderWeightedPartial(
                    **_weights,
                    continuous="Continuous" in env_id,
                )
            ),
            description=(
                "LunarLander table partial "
                f"(alignment={row['alignment']:.6f}, PCC={row['pcc']:.6f})"
            ),
            env_ids=("LunarLander-v3", "LunarLanderContinuous-v3"),
            component_keys=LunarLanderWeightedPartial.component_keys,
        )
