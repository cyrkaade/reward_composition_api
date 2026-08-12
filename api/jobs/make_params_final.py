"""Generate jobs/params_final.txt (CELL VARIANT BUDGET SEED).

THE FINAL GRID. Four stages, ordered so the cheap gate runs first and nothing
downstream is wasted if an environment fails to qualify.

    stage 0  rows   1 -  20   qualify HalfCheetah and Swimmer (5 seeds, true+partial)
    stage 1  rows  21 - 240   the headline: true / partial / feedback / naive
    stage 2  rows 241 - 270   the scale law: alpha below 1 on LunarLander
    stage 3  rows 271 - 310   cold start: pretraining x active learning, 2x2

WHAT CHANGED FROM PRE4, AND WHY
-------------------------------
1. --holdout-pairs 100. PRE4 could not measure query efficiency because nothing
   measured generalisation: --ensemble-training full leaves n_val_pairs=0, and
   training accuracy hit exactly 1.0000 on Pusher and Walker2d. Every run here
   reserves 100 pairs per round from trajectories no query is drawn from, rates
   them with the true reward, and scores the ensemble before and after each
   round. That is the primary readout now, not a side diagnostic.
2. --ensemble-bootstrap. In PRE4 the max-min training accuracy ACROSS the three
   ensemble members was exactly 0.000 on Pusher and Walker2d - the members were
   the same function, so disagreement-based active learning had nothing to
   measure and stage 3 would have been meaningless without this.
3. --reward-model-train-accuracy-stop 0.97. B-Pref's actual rule (verified in
   train_PEBBLE.py: `if total_acc > 0.97: break`). Without it the model runs a
   fixed 100 epochs and memorises.
4. Walker2d and Reacher are gone. Walker's ceiling is bimodal (799-6263 over 10
   seeds, 3 under 2500), it is unconverged at 2M, and all 16 of its cells were
   query-starved. Reacher's entire start-to-ceiling range is 2.4 reward units.
5. Two new environments, both of which CANNOT terminate, so none of the
   Walker/Hopper failure modes are expressible. Their partials are the shape
   that actually worked: pusher_honest is the only PRE4 prior that is
   catastrophic on its own (-99% of the true range), and both new ones match it.

THE QUESTION STAGE 1 EXISTS TO ANSWER
-------------------------------------
PRE4 showed naive beats feedback everywhere, but naive beat PARTIAL-ONLY on only
one environment of four - because three of the four priors recover 89-116% of
the task by themselves. Deleting a penalty term only leaves headroom when the
penalty is large relative to the reward. Pusher (control weight 0.1 on 7
actuators) leaves headroom; LunarLander (fuel, ~-0.3/step) does not. HalfCheetah
copies Pusher's structure and Swimmer breaks the direction of the reward
outright, so stage 1 is a real replication rather than a fourth null.
"""

from pathlib import Path

SEEDS = range(10)
PILOT_SEEDS = range(5)          # stage 0
REMAINING_SEEDS = range(5, 10)  # stage 1 tops the pilot up to 10 rather than repeating it
LADDER = (175, 350, 700)

lines: list[str] = []


def add(cells, variants, budgets, seeds=SEEDS):
    for cell in cells:
        for variant in variants:
            for budget in budgets:
                for seed in seeds:
                    lines.append(f"{cell} {variant} {budget} {seed}")


# --- stage 0: qualify the new environments -----------------------------------
# 5 seeds each, ground truth and prior only, no reward model involved. This is
# the check Walker2d never got: is the ceiling unimodal, does it converge, and
# is the prior actually incomplete? HalfCheetah's known risk is bimodality
# (~45% of seeds learn to run) but that was only ever measured on STOCK SB3 -
# all 399 archived HalfCheetah runs have tuned_hyperparams unset and not one is
# a mode=true run. DO NOT SUBMIT STAGE 1 FOR AN ENV THAT FAILS THIS. 20
add(("cheetah", "swimmer"), ("true_std", "partial_std"), (0,), PILOT_SEEDS)

# --- stage 1: the headline ---------------------------------------------------
# LunarLander and Pusher carry the full budget ladder because they are the two
# environments already known to qualify, so query efficiency is answerable there
# the moment the holdout works. The new environments run at q350 only until
# stage 0 clears them. Their true/partial arms take seeds 5-9 only: stage 0
# already ran seeds 0-4 of the identical cell, and run_final.sh skips a run whose
# metadata.json exists, so the pilot is reused instead of repeated. 20 + 40 + 80 + 80 = 220
add(("cheetah", "swimmer"), ("true_std", "partial_std"), (0,), REMAINING_SEEDS)
add(("cheetah", "swimmer"), ("feedback_std", "naive_std"), (350,))
add(("ll", "pusher"), ("true_std", "partial_std"), (0,))
add(("ll", "pusher"), ("feedback_std", "naive_std"), LADDER)

# --- stage 2: the scale law --------------------------------------------------
# The learned reward is tanh-bounded, so its per-step size is ~0.5 in EVERY
# environment, while the prior's is whatever the env's reward scale happens to
# be. Measured prior:model per step - Walker 5.19, LunarLander 1.24, Pusher
# 0.53, Reacher 0.29 - orders the four environments exactly by whether the
# composition beat the prior alone. alpha is the knob that moves LunarLander
# into Pusher's regime, and alpha < 1 has never been run: all 3,574 historical
# runs used alpha in {1, 2, 3}. alpha 1.0 is naive_std at ll/q350 above. 30
add(("ll",), ("alpha010", "alpha025", "alpha050"), (350,))

# --- stage 3: cold start, 2x2 ------------------------------------------------
# Factors: pretraining on the partial (on/off) x active learning (on/off),
# scored on the held-out set rather than on final return. Reading:
#   pretrain effect = (pre, uniform) - (none, uniform)
#   AL effect       = (none, AL)     - (none, uniform)
#   interaction     = (pre, AL) - (pre, uniform) - (none, AL) + (none, uniform)
# The interaction needs ~4x the sample size of a main effect, so treat a null
# there as uninformative at n=10 rather than as evidence of no interaction.
# (none, uniform) is feedback_std at ll/q350 above, so it is not duplicated. 40
add(("ll",), ("bt_uniform", "al_none", "bt_al"), (350,))
add(("ll",), ("al_none_naive",), (350,))

out = Path(__file__).with_name("params_final.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
