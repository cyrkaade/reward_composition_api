"""Generate jobs/params_final.txt (CELL VARIANT BUDGET SEED).

THE FINAL GRID. Five stages, ordered so every cheap measurement that could
invalidate an expensive one runs first.

    stage 0  rows   1 -  50   gates: hyperparameters, env qualification, prior strength
    stage 1  rows  51 - 270   the headline: true / partial / feedback / naive
    stage 2  rows 271 - 300   reward SCALE: alpha below 1 on LunarLander
    stage 3  rows 301 - 340   cold start: pretraining x active learning, 2x2
    stage 4  rows 341 - 360   prior INFORMATION: naive on the weakened priors

PPO HYPERPARAMETERS: STOCK SB3, NO --tuned-hyperparams
------------------------------------------------------
Decided 2026-08-12 and recorded in CLAUDE.md; this grid follows it. Measured over
the archive on the two surviving environments, tuned buys nothing:

    LunarLander  stock 15 seeds final 281.8 peak 291.8 solved 15/15
                 rl-zoo 10 seeds final 282.7 peak 289.1 solved 10/10
    Pusher       stock 15 seeds final -23.6 peak -22.0
                 rl-zoo (inert, no block exists) -23.9 / -22.2

Verified by diffing Suite.ppo_hyperparams with the flag on and off:
  - LunarLander 5 keys differ, and stock is ALREADY gamma .99, so the old
    `--policy-learning-kwargs '{gamma:0.99}'` override existed only to undo the
    rl-zoo block's .999. Dropping the flag drops the need for the override.
  - Pusher 0 keys differ. The flag was always inert there.
  - HalfCheetah 10 keys differ and Swimmer 5 - both untested here, so stage 0
    measures them instead of guessing. Swimmer's stock gamma is .99 against the
    zoo's .9999, and Swimmer's reward arrives many steps after the stroke that
    earned it, so that one key may be load-bearing.

This buys one provenance for the whole paper - "SB3 defaults, plus one
documented override on Swimmer if stage 0 says it is needed" - instead of
"rl-zoo on two envs, rl-zoo-minus-an-override on one, defaults on a fourth".

TWO SEPARATE THINGS THAT WERE CONFLATED
---------------------------------------
PRE4 found naive beat vanilla RLHF everywhere but beat the PRIOR ALONE on only
one environment of four, because three priors already recover 89-116% of the
task. There are two independent ways to leave the prior less room, and they are
NOT the same lever:

  stage 2, SCALE       alpha multiplies the prior. The composed reward is
                       alpha*prior + model, and the model is tanh-bounded to
                       ~0.5/step in every env, so alpha sets who is louder.
  stage 4, INFORMATION delete components from the prior so it genuinely knows
                       less about the task.

Scaling every weight inside a partial by c is EXACTLY alpha=c, since the partial
is linear in its weights - so a "weaker" partial built by rescaling would just be
stage 2 under another name. The stage 4 ladder therefore removes whole terms:
p10 knows only where the pad is, p25 adds speed, lunar_lander_approach adds
orientation and leg contact (and recovers 89% alone, which is the problem).

Stage 0 runs the PARTIAL-ONLY arms of that ladder first, because they need no
reward model and no labels. A level is worth a naive arm in stage 4 only if
training on it alone lands well short of the true arm. Guessing which level is
weak from random-policy statistics does not work here: measured over 40 random
episodes the fragment-ranking agreement is p10 62%, p25 52%, approach 60%, i.e.
not even monotone. Random-policy partiality is a known-unreliable number in this
project. Only the partial arm answers it.
"""

from pathlib import Path

SEEDS = range(10)
PILOT = range(5)
REST = range(5, 10)   # stage 1 tops the stage-0 pilots up to 10 rather than repeating them
LADDER = (175, 350, 700)

lines: list[str] = []


def add(cells, variants, budgets, seeds=SEEDS):
    for cell in cells:
        for variant in variants:
            for budget in budgets:
                for seed in seeds:
                    lines.append(f"{cell} {variant} {budget} {seed}")


# --- stage 0: everything cheap that could invalidate stage 1 -----------------
# 0a. Which hyperparameters for the two new environments? Ground truth only, so
#     no reward model is involved and the comparison is clean. 20
add(("cheetah",), ("true_stock", "true_zoo"), (0,), PILOT)
add(("swimmer",), ("true_stock", "true_g9999"), (0,), PILOT)

# 0b. Do the new environments qualify at all, and is their prior incomplete?
#     Walker2d was never asked this and cost 480 runs. 10
add(("cheetah", "swimmer"), ("partial_std",), (0,), PILOT)

# 0c. How much does each weakened LunarLander prior leave on the table? These
#     arms cost zero labels. A level only earns a naive arm in stage 4 if
#     training on it alone lands well below the true arm. 20
add(("ll",), ("partial_p10", "partial_p25"), (0,))

# --- stage 1: the headline ---------------------------------------------------
# LunarLander and Pusher carry the full budget ladder; they are the two envs
# already known to qualify, so query efficiency is answerable there as soon as
# the held-out diagnostic works. The new envs run at q350 until stage 0 clears
# them. Their true/partial arms take seeds 5-9 only: run_final.sh skips a run
# whose metadata.json exists, so the stage-0 pilots are reused. 20 + 40 + 80 + 80 = 220
add(("cheetah", "swimmer"), ("true_std", "partial_std"), (0,), REST)
add(("cheetah", "swimmer"), ("feedback_std", "naive_std"), (350,))
add(("ll", "pusher"), ("true_std", "partial_std"), (0,))
add(("ll", "pusher"), ("feedback_std", "naive_std"), LADDER)

# --- stage 2: reward SCALE ---------------------------------------------------
# Measured prior:model per-step scale in PRE4 - Walker 5.19, LunarLander 1.24,
# Pusher 0.53, Reacher 0.29 - ordered the four environments exactly by whether
# the composition beat the prior alone. alpha 1.0 is naive_std at ll/q350.
#
# ALPHA < 1 HAS BEEN RUN BEFORE and it LOWERED final return. The `ga` family has
# 180 runs at alpha in {0.1, 0.25, 0.5} with the gate OFF, 15 seeds each, paired
# against alpha 1.0 (curve-final, exact two-sided Wilcoxon):
#     LunarLander  0.1 -79.3 (3/15, p=0.008)   0.25 -114.6 (2/15, p=0.008)   0.5 -54.8 (p=0.121)
#     Pusher       0.1  -3.0 (3/15, p=0.015)   0.25   -1.3 (p=0.389)         0.5  -2.0 (p=0.095)
#     Reacher      0.1  -0.7 (p=0.330)         0.25   -2.3 (p=0.095)         0.5  -9.0 (4/15, p=0.010)
#
# Those runs used the LEGACY reward model - hidden [200], ensemble 1, NO tanh,
# q700 only - so they are not a test of the scale hypothesis, which is a claim
# about a tanh-BOUNDED model whose per-step size is pinned near 0.5 regardless of
# environment. Without tanh the model's scale is free to grow and the ratio alpha
# is supposed to control does not exist. This stage re-runs alpha under the
# standardized model. Given the prior evidence, treat a positive result as the
# surprising outcome, not the expected one. 30
add(("ll",), ("alpha010", "alpha025", "alpha050"), (350,))

# --- stage 3: cold start, 2x2 ------------------------------------------------
# Scored on the held-out set, not on final return. Cells:
#   A none+uniform (= feedback_std, not duplicated)   B pretrained+uniform
#   C none+AL                                          D pretrained+AL
# plus one naive AL cell, because every PRE4 AL null was measured with an
# ensemble whose three members were the same function. 40
add(("ll",), ("bt_uniform", "al_none", "bt_al", "al_none_naive"), (350,))

# --- stage 4: prior INFORMATION ----------------------------------------------
# The naive arms of the ladder whose partial arms stage 0 showed leave headroom.
# Pair each against its own partial arm from stage 0c to get naive-minus-prior
# WITHIN one environment - a dose-response curve, which is far stronger evidence
# than comparing across four environments that differ in many other ways. 20
add(("ll",), ("naive_p10", "naive_p25"), (350,))

out = Path(__file__).with_name("params_final.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
