"""Emit params_wsum_atari2.txt: the weighted-sum alpha sweep on Breakout and Pong.

The second Atari pair, run on exactly the parameters that produced the
MsPacman/Qbert result -- same 13 arms, same two budgets, same four alphas, same
1M timesteps -- so the four games sit in one comparable grid.  Only the cells
and their priors differ.

Seeds default to 5 because that is what the first Atari pair actually finished,
and an unmatched seed count would make the four cells incomparable.

Row format:  CELL ARM SEED PARTIAL BUDGET ALPHA
"""

import os
from pathlib import Path

# Two priors per game, chosen to vary the one property that separated Qbert
# (big win) from MsPacman (null): how often the prior fires relative to the true
# reward.  Alignment alone does not explain the split -- Qbert's rqb_tiles has
# the same PCC as the rqb_visit02 the grid used (0.68 vs 0.67) but only 1.0x the
# true reward's fragment coverage, while visit02 has 2.8x.
#
# Measured per fragment of 25 over 60k random steps (seeds 0-2):
#
#   cell        prior          PCC    fires vs true reward
#   breakout    rbo_bricks     0.98   1.00x   <- aligned but adds no coverage
#   breakoutd   rbo_track05    0.81   4.88x   <- Qbert's winning profile
#   pong        rpong_score    0.38   0.03x   <- silent on 98.4% of fragments
#   pongd       rpong_track05  0.27   1.71x
#
# Running both per game turns "will the weighted sum win here" into "does prior
# density predict whether it wins", which is answerable either way.
CELLS = {
    "breakout": "reasonable_atari_partials:rbo_bricks",
    "breakoutd": "reasonable_atari_partials:rbo_track05",
    "pong": "reasonable_atari_partials:rpong_score",
    "pongd": "reasonable_atari_partials:rpong_track05",
}

BUDGETS = (2800, 5600)
ALPHAS = (0.20, 0.40, 0.60, 0.80)
SEEDS = range(int(os.environ.get("SEEDS", "5")))


def arms():
    """The 13 arms per cell: true, then vanilla/naive/4 alphas at each budget."""
    yield ("true", 0, 1.0)
    for budget in BUDGETS:
        yield (f"vanilla_q{budget}", budget, 1.0)
        yield (f"naive_q{budget}", budget, 1.0)
        for alpha in ALPHAS:
            yield (f"ws{int(alpha * 100):03d}_q{budget}", budget, alpha)


def main() -> None:
    rows = []
    for seed in SEEDS:  # seed-major: a truncated run still gives whole seed layers
        for cell, partial in CELLS.items():
            for arm, budget, alpha in arms():
                reference = "-" if arm == "true" else partial
                rows.append(f"{cell} {arm} {seed} {reference} {budget} {alpha:.2f}")

    out = Path(__file__).with_name("params_wsum_atari2.txt")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")

    per_seed = len(rows) // len(SEEDS)
    print(f"wrote {out} with {len(rows)} rows ({per_seed} per seed layer)")
    print(f"  cells={len(CELLS)} arms={per_seed // len(CELLS)} seeds={len(SEEDS)}")
    print(f"  first row: {rows[0]}")
    print(f"  row {per_seed + 1} (start of seed 1): {rows[per_seed]}")


if __name__ == "__main__":
    main()
