"""Generate jobs/params_hopper5m.txt (20 rows: VARIANT SEED).

Hopper-v5, ground-truth reward, 5M steps, 10 seeds, two hyperparameter arms:

    ours    the config in rcomp/ppo_presets.py  (--tuned-hyperparams)
    stock   SB3 out-of-the-box defaults

BOTH ARMS RUN AT --n-envs 1. That is not a detail: the preset has only ever been
measured at 1, and the archived stock runs used 8. Holding n_envs equal is what
makes the two arms differ in hyperparameters ALONE. The archived n_envs=8 stock
result (peak 3532 @3.45M, final 1677, 10 seeds) stays available as a third
reference point, but it is not the comparison.

Why 5M and not 1M: the preset's validation was 3 seeds x 1M, which is thin for a
config the project now depends on, and 1M cannot show whether its 35% peak-to-
final drawdown keeps growing. Evaluation runs every 50K steps, so the 1M numbers
are recoverable from the same runs - there is no need for a separate short job.

Adding the partial arm later costs nothing in this script: `ours_partial` and
`stock_partial` are already wired into run_hopper5m.sh. Uncomment the second
`add(...)` below and the job grows to 40 rows.
"""

from pathlib import Path

SEEDS = range(10)

lines: list[str] = []


def add(variants, seeds=SEEDS):
    for variant in variants:
        for seed in seeds:
            lines.append(f"{variant} {seed}")


# Ordered arm-major so that a partially-finished array still yields whole cells.
add(("ours", "stock"))

# The pending E0 question - does the partial beat the true reward once the true
# arm can actually learn? - at the same budget. 20 more runs.
# add(("ours_partial", "stock_partial"))

out = Path(__file__).with_name("params_hopper5m.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(lines)} rows)")
