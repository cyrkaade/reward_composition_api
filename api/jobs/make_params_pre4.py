"""Generate jobs/params_pre4.txt (480 rows: CELL VARIANT BUDGET SEED).

ONE submission covering every open question, ordered so the only configuration
risk sits at the front of the array and is isolated to a single environment.

    rows   1 -  30   A  LunarLander config repair (the three single levers)
    rows  31 - 110   B  references: true + partial, 4 envs
    rows 111 - 270   C  M1 + query efficiency: feedback vs naive, budget ladder
    rows 271 - 390   D  pretraining objective: none vs MSE vs hard-BT
    rows 391 - 470   E  active learning with the corrected candidate pool
    rows 471 - 480   F  partial design: potential-based vs terminal-term partial

Why the risk is confined: gamma 0.999 is a LunarLander-only property of the
tuned presets (Walker2d 0.99, Pusher 0.99, Reacher 0.9), and the output-L1 term
the gate switched off is simply the package default. So Walker/Pusher/Reacher
have no open configuration question and rows 31+ are safe for them regardless of
how block A turns out. Only the LunarLander cells depend on it, and they use the
setting 6,000 archived runs already validated.
"""

from pathlib import Path

SEEDS = range(10)
COLLAPSE = ("ll", "walker")      # variable-horizon: the termination channel is farmable
CONTROL = ("pusher", "reacher")  # fixed horizon (100 and 50 steps): immune by construction
ALL_ENVS = COLLAPSE + CONTROL
LADDER = (175, 350, 700)

lines: list[str] = []


def add(cells, variants, budgets, seeds=SEEDS):
    for cell in cells:
        for variant in variants:
            for budget in budgets:
                for seed in seeds:
                    lines.append(f"{cell} {variant} {budget} {seed}")


# --- A. repair, LunarLander only --------------------------------------------
# Baseline is the EXISTING logs/pre3g_ll_naive_bounded (gamma .999, no output
# L1), so it costs no runs. The fourth arm -- both levers together -- is
# `naive_std` at ll/q350 in block C, so it is not duplicated here. 30
add(("ll",), ("naive_g99", "naive_l1", "naive_stop"), (350,))

# --- B. references, all four environments ------------------------------------
# Needed because every arm below runs tuned PPO plus the standardized reward
# model, so no archived floor/ceiling applies. BUDGET is ignored. 80
add(ALL_ENVS, ("true_std", "partial_std"), (350,))

# --- C. M1 and query efficiency ----------------------------------------------
# Full ladder on the two variable-horizon envs, one budget on the two
# fixed-horizon controls (their role is to show nothing happens, which does not
# need a ladder). 120 + 40 = 160
add(COLLAPSE, ("feedback_std", "naive_std"), LADDER)
add(CONTROL, ("feedback_std", "naive_std"), (350,))

# --- D. pretraining objective -------------------------------------------------
# "none" is block C's q175/q350 cells, so only mse and bt are new. MSE is run at
# the low budget only: it is the archived, broken objective and the point is to
# show BT fixes it where the prior matters most. 40 + 80 = 120
add(COLLAPSE, ("mse_feedback", "mse_naive"), (175,))
add(COLLAPSE, ("bt_feedback", "bt_naive"), (175, 350))

# --- E. active learning ------------------------------------------------------
# Uniform baselines come from blocks C and D. These use the corrected candidate
# pool (--active-candidate-protocol pool): an independent 10x pool scored
# per-pair, as B-Pref does, rather than the best of 512 whole perfect matchings. 80
add(COLLAPSE, ("al_feedback", "al_naive", "al_bt_feedback", "al_bt_naive"), (350,))

# --- F. partial design -------------------------------------------------------
# Deliberately at the UNREPAIRED gate config, because that is where hovering
# happens. lunar_lander_approach is pure potential-based shaping (Ng, Harada &
# Russell 1999): it cannot change the optimal policy and therefore carries no
# incentive to terminate. lunarlander_p50 adds a non-potential terminal term. 10
add(("ll",), ("naive_p50",), (350,))

out = Path(__file__).with_name("params_pre4.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
