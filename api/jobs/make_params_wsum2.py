"""Emit params_wsum2.txt: the second wave of the weighted-sum alpha sweep.

Two additions, kept in one file so they schedule together:

* three new environments (Pusher, Swimmer, Walker2d) get the FULL arm set;
* the five environments already running get only the new alpha=0.5 arms, since
  everything else there is already done or in flight.

Deliberately written to a NEW params file rather than regenerating
params_wsum.txt: run_wsum.sh selects its row with `sed -n "${TASK_ID}p"`, so
rewriting the file the first grid is still reading would silently repoint any
requeued task at a different configuration.

Row format:  CELL ARM SEED PARTIAL BUDGET ALPHA
"""

import os
from pathlib import Path

# The five already running, with the partials they were launched under.
EXISTING = {
    "ll": "reasonable_partials:rll_tilt50",
    "ant": "reasonable_partials:rant_cap30_ctrl05",
    "hopper": "reasonable_partials:rhop_cap80",
    "reacher": "reasonable_partials:rrch_x_ctrl050",
    "bipedal": "reasonable_partials:rbw_cap75",
}
NEW = {
    "pusher": "reasonable_partials:rpsh_cap20_ctrl10",
    "swimmer": "reasonable_partials:rswm_cap28_ctrl001",
    "walker": "reasonable_partials:rwalk_cap175",
}

BUDGETS = (200, 400)
ALPHAS = (0.20, 0.40, 0.50, 0.60, 0.80)  # 0.50 is the addition
NEW_ONLY_ALPHAS = (0.50,)
SEEDS = range(int(os.environ.get("SEEDS", "10")))


def full_arms():
    """Every arm a fresh cell needs: true, then vanilla/naive/5 alphas x 2 budgets."""
    yield ("true", 0, 1.0)
    for budget in BUDGETS:
        yield (f"vanilla_q{budget}", budget, 1.0)
        yield (f"naive_q{budget}", budget, 1.0)
        for alpha in ALPHAS:
            yield (f"ws{int(alpha * 100):03d}_q{budget}", budget, alpha)


def topup_arms():
    """Only the arms an already-running cell is missing."""
    for budget in BUDGETS:
        for alpha in NEW_ONLY_ALPHAS:
            yield (f"ws{int(alpha * 100):03d}_q{budget}", budget, alpha)


def main() -> None:
    rows = []
    for seed in SEEDS:  # seed-major: a truncated run still gives whole seed layers
        for cell, partial in NEW.items():
            for arm, budget, alpha in full_arms():
                reference = "-" if arm == "true" else partial
                rows.append(f"{cell} {arm} {seed} {reference} {budget} {alpha:.2f}")
        for cell, partial in EXISTING.items():
            for arm, budget, alpha in topup_arms():
                rows.append(f"{cell} {arm} {seed} {partial} {budget} {alpha:.2f}")

    out = Path(__file__).with_name("params_wsum2.txt")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")

    per_seed = len(rows) // len(SEEDS)
    new_rows = sum(1 for r in rows if r.split()[0] in NEW)
    print(f"wrote {out} with {len(rows)} rows ({per_seed} per seed layer)")
    print(f"  {len(NEW)} new cells x 15 arms x {len(SEEDS)} seeds = {new_rows}")
    print(f"  {len(EXISTING)} existing cells x 2 alpha-0.5 arms x {len(SEEDS)} seeds = {len(rows) - new_rows}")
    print(f"  first row: {rows[0]}")
    print(f"  row {per_seed + 1} (start of seed 1): {rows[per_seed]}")


if __name__ == "__main__":
    main()
