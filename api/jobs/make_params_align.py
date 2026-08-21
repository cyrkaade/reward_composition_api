"""Emit params_align.txt: alignment x alpha on LunarLander.

The 5x5 the experiment is about: five hand-written priors spanning an alignment
ladder (``partials/lunar_lander_alignment.py``) crossed with five weighted-sum
alphas, at one query budget and one horizon.

Two reference arms ride along in a `ref` cell -- `true` (ceiling) and
`vanilla_q400`, i.e. feedback with no prior at all (floor).  They are NOT part of
the 5x5; they cost 10 runs out of 135 and without them no panel in the grid has a
scale.  The existing wsum LunarLander arms cannot serve: those ran 2M timesteps,
this grid runs 1M.  Set REFS=0 to drop them.

Row format:  CELL ARM SEED PARTIAL BUDGET ALPHA
"""

import os
from pathlib import Path

# Cell name -> partial reference.  Cell doubles as the panel label in the figure,
# and the digits are the alignment score the grid reported for that weight row.
RUNGS = {
    "a00": "lunar_lander_alignment:lla_a00",
    "a30": "lunar_lander_alignment:lla_a30",
    "a50": "lunar_lander_alignment:lla_a50",
    "a75": "lunar_lander_alignment:lla_a75",
    "a95": "lunar_lander_alignment:lla_a95",
}

BUDGET = int(os.environ.get("BUDGET", "400"))
ALPHAS = (0.20, 0.40, 0.50, 0.60, 0.80)
SEEDS = range(int(os.environ.get("SEEDS", "5")))
WITH_REFS = os.environ.get("REFS", "1") != "0"


def main() -> None:
    rows = []
    for seed in SEEDS:  # seed-major: a truncated run still gives whole seed layers
        if WITH_REFS:
            rows.append(f"ref true {seed} - 0 1.00")
            rows.append(f"ref vanilla_q{BUDGET} {seed} - {BUDGET} 1.00")
        for cell, partial in RUNGS.items():
            for alpha in ALPHAS:
                arm = f"ws{int(alpha * 100):03d}_q{BUDGET}"
                rows.append(f"{cell} {arm} {seed} {partial} {BUDGET} {alpha:.2f}")

    out = Path(__file__).with_name("params_align.txt")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")

    per_seed = len(rows) // len(SEEDS)
    grid = len(RUNGS) * len(ALPHAS) * len(SEEDS)
    print(f"wrote {out} with {len(rows)} rows ({per_seed} per seed layer)")
    print(f"  grid: {len(RUNGS)} rungs x {len(ALPHAS)} alphas x {len(SEEDS)} seeds = {grid}")
    print(f"  refs: {len(rows) - grid}")
    print(f"  first row: {rows[0]}")
    print(f"  row {per_seed + 1} (start of seed 1): {rows[per_seed]}")


if __name__ == "__main__":
    main()
