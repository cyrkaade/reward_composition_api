"""Emit params_starc.txt: STARC alignment x alpha on LunarLander.

Same shape as the PCC grid in make_params_align.py -- five prior rungs crossed
with five weighted-sum alphas, one budget, one horizon -- but the rungs are the
STARC ladder in ``partials/lunar_lander_starc_alignment.py``, calibrated to
0.2 / 0.4 / 0.6 / 0.8 / 1.0 and verified on held-out seeds.

The two reference arms ride along in a `ref` cell: `true` (ceiling) and
`vanilla_q400`, feedback with no prior at all (floor).  They are not part of the
5x5 and cost 10 runs out of 135, but without them no panel has a scale.  The
align grid's refs cannot be reused -- it ran different priors but also landed on
whatever nodes Slurm had free, so they are not a matched control.  Set REFS=0 to
drop them.

Row format:  CELL ARM SEED PARTIAL BUDGET ALPHA
"""

import os
from pathlib import Path

# Cell name -> partial reference.  Cell doubles as the panel label in the
# figure, and the digits are the STARC alignment target for that rung.
RUNGS = {
    "s20": "lunar_lander_starc_alignment:lls_a20",
    "s40": "lunar_lander_starc_alignment:lls_a40",
    "s60": "lunar_lander_starc_alignment:lls_a60",
    "s80": "lunar_lander_starc_alignment:lls_a80",
    "s100": "lunar_lander_starc_alignment:lls_a100",
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

    out = Path(__file__).with_name("params_starc.txt")
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
