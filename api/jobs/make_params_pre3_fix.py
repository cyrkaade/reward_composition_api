"""Generate jobs/params_pre3_fix.txt (90 rows: CELL VARIANT SEED).

PRE3 STAGE 1: find the one configuration change that stops the policy farming
the episode time limit, and do it in a single submission that already contains
the matched references and feedback partners for whichever lever wins.

Ordered so a prefix is usable:
    rows  1 - 40   repair levers on naive_bounded (the four candidates)
    rows 41 - 60   feedback partners for the two most likely winners
    rows 61 - 80   true / partial references at gamma 0.99
    rows 81 - 90   the partial-design prediction, at the unrepaired gate config
"""

from pathlib import Path

SEEDS = range(10)
lines: list[str] = []


def add(variants, seeds=SEEDS):
    for variant in variants:
        for seed in seeds:
            lines.append(f"ll {variant} {seed}")


# Repair levers. Baseline is the EXISTING logs/pre3g_ll_naive_bounded, so it
# costs nothing and every arm here is a one- or two-flag delta from it. 40
add(("naive_g99", "naive_l1", "naive_stop", "naive_g99_l1"))

# Feedback partners. M1 (naive vs feedback) has to be measured at ONE working
# configuration; these cover the two most likely winners so no second round is
# needed. feedback_stop also subsumes run_pre3_repair.sh rows 1-20. 20
add(("feedback_g99_l1", "feedback_stop"))

# Matched floor and ceiling at gamma 0.99. The gate's true/partial ran at
# gamma 0.999, so if the discount is the fix its references have to move too. 20
add(("true_g99", "partial_g99"))

# The partial-design prediction, deliberately run at the UNREPAIRED gate config
# because that is where hovering happens. lunar_lander_approach is pure
# potential-based shaping (Ng, Harada & Russell 1999): it cannot change the
# optimal policy, and therefore carries no incentive to terminate.
# lunarlander_p50 adds a non-potential terminal term. Prediction: p50 suppresses
# hovering where the potential-based partial cannot. 10
add(("naive_p50",))

out = Path(__file__).with_name("params_pre3_fix.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
