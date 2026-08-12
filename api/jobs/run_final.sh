#!/bin/bash
#SBATCH --job-name=final
#SBATCH --output=logs/slurm/final_%A_%a.out
#SBATCH --error=logs/slurm/final_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-50
#SBATCH --requeue
set -euo pipefail

# ============================================================================
# THE FINAL GRID. Submit in stages; stage 0 gates the rest.
#
#   sbatch --array=1-50    jobs/run_final.sh    # stage 0: the gates
#   python jobs/analyze_final.py --only 0       # <-- READ, THEN SET THE TWO
#                                               #     VARIABLES MARKED BELOW
#   sbatch --array=51-270  jobs/run_final.sh    # stage 1: the headline
#   sbatch --array=271-300 jobs/run_final.sh    # stage 2: reward scale (alpha)
#   sbatch --array=301-340 jobs/run_final.sh    # stage 3: cold start 2x2
#   sbatch --array=341-360 jobs/run_final.sh    # stage 4: prior information
#
# Stages 2 and 3 are LunarLander-only on settled hyperparameters, so they can go
# in alongside stage 1. Stage 4 depends on stage 0c.
#
# ============================================================================
# PPO HYPERPARAMETERS: STOCK SB3. --tuned-hyperparams IS NOT PASSED.
# ============================================================================
# Decided 2026-08-12 (CLAUDE.md). Measured over the archive, tuned buys nothing
# on the two surviving environments:
#     LunarLander  stock 15 seeds -> final 281.8, peak 291.8, solved 15/15
#                  rl-zoo 10 seeds -> final 282.7, peak 289.1, solved 10/10
#     Pusher       stock 15 seeds -> final -23.6; rl-zoo has no Pusher block at
#                  all, so the flag was always inert there (verified: 0 keys)
# Stock LunarLander is ALREADY gamma .99, so the old
# `--policy-learning-kwargs '{gamma:0.99}'` override is gone too - it existed
# only to undo the rl-zoo block's .999.
#
# The two new environments have never been measured either way (10 keys differ
# on HalfCheetah, 5 on Swimmer), so stage 0a measures instead of guessing.
# ============================================================================

# ---------------------------------------------------------------------------
# SET THESE TWO AFTER READING `analyze_final.py --only 0`. They are the only
# hyperparameter decisions this grid leaves open.
#
# CHEETAH_ARM: "stock" or "zoo". Stock is the default because it keeps one
#   provenance for the paper. Switch to "zoo" only if stage 0a shows the zoo
#   block clearly wins AND its true arm is unimodal.
# SWIMMER_GAMMA: Swimmer's reward arrives many steps after the stroke that earns
#   it, and the zoo uses gamma .9999 against SB3's .99. Stage 0a runs both. If
#   stock .99 learns to swim, blank this out for a pure-SB3 story.
# ---------------------------------------------------------------------------
CHEETAH_ARM="stock"
SWIMMER_GAMMA=(--policy-learning-kwargs '{gamma:0.9999}')

if [ ! -f jobs/params_final.txt ]; then
  echo "jobs/params_final.txt missing; run: python jobs/make_params_final.py" >&2
  exit 2
fi
if [ "$(wc -l < jobs/params_final.txt)" -ne 360 ]; then
  echo "jobs/params_final.txt must contain exactly 360 lines" >&2
  exit 2
fi

HELP_TEXT="$(python -m rcomp train --help)"
for FLAG in --holdout-pairs --ensemble-bootstrap --reward-model-train-accuracy-stop \
            --ensemble-training --round0-data-protocol --tanh-model-reward \
            --dedicated-query-rng --active-candidate-protocol --pretrain-loss \
            --policy-learning-kwargs --partial-alpha --n-envs; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required flag: $FLAG -- git pull" >&2
    exit 2
  fi
done

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_final.txt | tr -d '\r')"
read -r CELL VARIANT BUDGET SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${BUDGET:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# PER-ENVIRONMENT SETTINGS, all verified on 2026-08-12 rather than remembered:
#   FRAGMENT vs MEASURED episode length. Walker2d's 16 starved cells came from
#     fragment 50 meeting ~18-step early episodes, so round 0 bought 2-3 pairs
#     instead of 70. Measured: Pusher exactly 100, HalfCheetah and Swimmer
#     exactly 1000 (neither can terminate), LunarLander 100 random to ~1000.
#   COLLECTION big enough for the queries AND the 100-pair holdout, checked by
#     running round 0 at the highest per-round demand: LunarLander delivered
#     140/140 with 1379 fragments spare, HalfCheetah and Swimmer 70/70 with 1000.
#   NENV 8 everywhere. With stock SB3 there is no `reference` block to match -
#     that rule applies to zoo presets, and ignoring it is what broke Hopper.
#     The archive's stock evidence for LunarLander and Pusher is at n_envs 8.
# ---------------------------------------------------------------------------
EXTRA=()
case "$CELL" in
  ll)
    SUITE=box2d;  ENV=LunarLander-v3; PARTIAL=lunar_lander_approach
    STEPS=2000000; FRAGMENT=25; COLLECTION=40000; NENV=8 ;;
  pusher)
    SUITE=mujoco; ENV=Pusher-v5;      PARTIAL=pusher_honest
    STEPS=3000000; FRAGMENT=25; COLLECTION=20000; NENV=8 ;;
  cheetah)
    # Never terminates. Its prior drops a 0.1-weight control cost on 6 actuators
    # - the same shape as pusher_honest, the only PRE4 prior that was genuinely
    # incomplete (-99% of the true range alone). Known risk is bimodality, but
    # that was only ever measured on stock SB3 with no ground-truth arm at all:
    # all 399 archived HalfCheetah runs have tuned_hyperparams unset and not one
    # is mode=true. Stage 0 is that missing measurement.
    SUITE=mujoco; ENV=HalfCheetah-v5; PARTIAL=halfcheetah_forward
    STEPS=3000000; FRAGMENT=50; COLLECTION=50000; NENV=8
    if [ "$CHEETAH_ARM" = "zoo" ]; then EXTRA=(--tuned-hyperparams); NENV=1; fi ;;
  swimmer)
    # The only environment here that CANNOT terminate at all, so the lowest
    # variance available. Deleting the control cost does NOT give an incomplete
    # prior here (weight 1e-4); swimmer_speed rewards |velocity| and is blind to
    # direction instead - measured, it correlates 0.23 with reward_forward while
    # x-velocity alone correlates 0.97, and 12k steps of optimising it drives the
    # true reward to -36.
    SUITE=mujoco; ENV=Swimmer-v5;     PARTIAL=swimmer_speed
    STEPS=2000000; FRAGMENT=50; COLLECTION=50000; NENV=8
    EXTRA=("${SWIMMER_GAMMA[@]}") ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

# Stage 0a overrides the hyperparameter arm explicitly, whatever the variables above say.
# The zoo arm also drops to n_envs 1 because the zoo blocks are tuned for it, and
# ignoring that `reference` is exactly what pinned Hopper at the survive-only
# optimum: 488 updates over 2M instead of the intended ~3,900.
case "$VARIANT" in
  true_stock)  EXTRA=() ;;
  true_zoo)    EXTRA=(--tuned-hyperparams); NENV=1 ;;
  true_g9999)  EXTRA=(--policy-learning-kwargs '{gamma:0.9999}') ;;
esac

# Stages 0c and 4 swap the prior. MUST be the file:name form -- `--partial`
# resolves a bare name as a MODULE, and these live inside lunar_lander_levels.py,
# which is how block F died instantly in PRE4 with "No module named".
case "$VARIANT" in
  partial_p10|naive_p10) PARTIAL=lunar_lander_levels:lunarlander_p10 ;;
  partial_p25|naive_p25) PARTIAL=lunar_lander_levels:lunarlander_p25 ;;
esac

LOGDIR="logs/final_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"
# Guards only against re-running a FINISHED run. It does NOT stop two concurrent
# submissions writing into one directory -- check `squeue -u $USER` first.
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

# Resolve the partial before burning a queue slot; this is the exact call the
# trainer makes. NOT `rcomp validate-partial` (feeds a 4-element dummy obs and
# fails on every 8-dim LunarLander partial) and NOT `rcomp list-partials` (prints
# bare names even for partials that need the file:name form).
if ! python -c "
import sys
from rcomp.partials import PartialRegistry, load_partial_reference
try:
    load_partial_reference('$PARTIAL', '$SUITE', PartialRegistry())
except Exception as exc:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    sys.exit(1)
"; then
  echo "partial '$PARTIAL' does not resolve for suite $SUITE" >&2
  exit 2
fi

ARGS=(
  --suite "$SUITE" --env-id "$ENV" --partial "$PARTIAL"
  --n-envs "$NENV" --final-policy last
  --timesteps "$STEPS" --seed "$SEED"
  --eval-freq 50000 --n-eval-episodes 20
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
  "${EXTRA[@]}"
)

# The standardized reward model. Identical to PRE4 except the three NEW lines,
# so PRE4 stays the direct comparison.
RM=(
  --query-budget "$BUDGET" --rlhf-rounds 5
  --collection-timesteps "$COLLECTION" --fragment-length "$FRAGMENT"
  --reward-hidden-sizes 256,256,256
  --reward-model-ensemble-size 3
  --reward-model-lr 0.0003
  --reward-model-batch-size 128
  --reward-model-loss-reduction mean
  --reward-model-l1 0
  --reward-output-l1 0.001
  --ensemble-training full
  --round0-data-protocol separate
  --dedicated-query-rng
  --tanh-model-reward
  --holdout-pairs 100                        # NEW: 100 pairs/round no member ever trains on
  --ensemble-bootstrap                       # NEW: PRE4's 3 members had spread exactly 0.000
  --reward-model-train-accuracy-stop 0.97    # NEW: B-Pref's actual stop, from train_PEBBLE.py
  --save-reward-model
)

UNIFORM=(--no-active-learning)
ACTIVE=(--active-learning --active-query-strategy ensemble --active-candidate-protocol pool --active-pool-multiplier 10)
NAIVE=(--mode naive --partial-alpha 1.0 --no-include-partial-feature)
BT=(--pretrain-reward-model --pretrain-target partial --pretrain-loss bt)

case "$VARIANT" in
  # --- stage 0a: which hyperparameters for the new environments? -------------
  true_stock|true_zoo|true_g9999) ARGS+=(--mode true) ;;

  # --- stages 0b, 0c, 1: references and the headline -------------------------
  true_std)                   ARGS+=(--mode true) ;;
  partial_std|partial_p10|partial_p25) ARGS+=(--mode partial) ;;
  feedback_std)               ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode feedback) ;;
  naive_std)                  ARGS+=("${RM[@]}" "${UNIFORM[@]}" "${NAIVE[@]}") ;;

  # --- stage 2: reward SCALE (alpha 1.0 is naive_std at ll/q350) -------------
  alpha010) ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode naive --partial-alpha 0.10 --no-include-partial-feature) ;;
  alpha025) ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode naive --partial-alpha 0.25 --no-include-partial-feature) ;;
  alpha050) ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode naive --partial-alpha 0.50 --no-include-partial-feature) ;;

  # --- stage 3: cold start 2x2 (A = feedback_std, not duplicated) ------------
  bt_uniform)    ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode feedback "${BT[@]}") ;;
  al_none)       ARGS+=("${RM[@]}" "${ACTIVE[@]}"  --mode feedback) ;;
  bt_al)         ARGS+=("${RM[@]}" "${ACTIVE[@]}"  --mode feedback "${BT[@]}") ;;
  al_none_naive) ARGS+=("${RM[@]}" "${ACTIVE[@]}"  "${NAIVE[@]}") ;;

  # --- stage 4: prior INFORMATION (partial already swapped above) ------------
  naive_p10|naive_p25) ARGS+=("${RM[@]}" "${UNIFORM[@]}" "${NAIVE[@]}") ;;

  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

srun python -m rcomp train "${ARGS[@]}"
