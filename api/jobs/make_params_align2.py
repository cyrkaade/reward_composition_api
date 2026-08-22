"""Emit the 5 x 5 x 5 weighted-sum grid for the supplied partial table.

This is the requested follow-up to the experiment that generated
``starc_grid_heatmap.png``.  It keeps q400, the 1M-step horizon, seeds 0--4,
and weighted-sum alphas 0.2/0.4/0.5/0.6/0.8, changing only the five partial
functions.  The already-completed ``starc`` true and vanilla arms are invariant
to the partial and can be reused as plot references, so this file emits exactly
the 125 requested weighted-sum runs rather than rerunning ten controls.

Row format:  CELL ARM SEED PARTIAL BUDGET ALPHA
"""

import os
from pathlib import Path


# Cell labels carry the table's alignment rounded to three decimal places.
# Order matches the supplied image.
RUNGS = {
    "a298": "lunar_lander_alignment2:partial_0.5_0.0_0.0_1.0",
    "a240": "lunar_lander_alignment2:partial_1.0_0.0_1.0_1.0",
    "a182": "lunar_lander_alignment2:partial_0.5_0.5_0.0_0.5",
    "a118": "lunar_lander_alignment2:partial_0.0_1.0_0.0_1.0",
    "a060": "lunar_lander_alignment2:partial_0.5_1.0_0.5_1.0",
}

BUDGET = int(os.environ.get("BUDGET", "400"))
ALPHAS = (0.20, 0.40, 0.50, 0.60, 0.80)
SEEDS = range(int(os.environ.get("SEEDS", "5")))


def main() -> None:
    rows = []
    for seed in SEEDS:  # seed-major: partial completion gives whole seed layers
        for cell, partial in RUNGS.items():
            for alpha in ALPHAS:
                arm = f"ws{int(alpha * 100):03d}_q{BUDGET}"
                rows.append(f"{cell} {arm} {seed} {partial} {BUDGET} {alpha:.2f}")

    out = Path(__file__).with_name("params_align2.txt")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")

    per_seed = len(rows) // len(SEEDS)
    print(f"wrote {out} with {len(rows)} rows ({per_seed} per seed layer)")
    print(
        f"  grid: {len(RUNGS)} partials x {len(ALPHAS)} alphas "
        f"x {len(SEEDS)} seeds = {len(rows)}"
    )
    print(f"  first row: {rows[0]}")
    print(f"  row {per_seed + 1} (start of seed 1): {rows[per_seed]}")


if __name__ == "__main__":
    main()
