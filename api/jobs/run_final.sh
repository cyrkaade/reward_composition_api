#!/bin/bash
#SBATCH --job-name=final
#SBATCH --output=logs/slurm/final_%A_%a.out
#SBATCH --error=logs/slurm/final_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-20
#SBATCH --requeue
set -euo pipefail

# ============================================================================
# THE FINAL GRID. Submit in stages; each stage is gated on the previous one.
#
#   sbatch --array=1-20    jobs/run_final.sh    # stage 0: qualify the new envs
#   python jobs/analyze_final.py --only 0       # <-- READ THIS BEFORE STAGE 1
#   sbatch --array=21-240  jobs/run_final.sh    # stage 1: the headline
#   sbatch --array=241-270 jobs/run_final.sh    # stage 2: the scale law
#   sbatch --array=271-310 jobs/run_final.sh    # stage 3: cold start 2x2
#
# Stage 2 and 3 are LunarLander-only and depend on nothing in stage 0, so they
# may be submitted alongside stage 1.
#
# ============================================================================
# WHAT MAKES THIS DIFFERENT FROM PRE4
# ============================================================================
# PRE4 answered "does the prior beat vanilla RLHF" (yes, +202/+214/+182 on
# LunarLander, 8-9/10 seeds) but could not answer either follow-up:
#
#   1. Does COMBINING beat just using the prior? Only on Pusher (+19.0, 10/10,
#      p=0.002). Null on the other three, because their priors already recover
#      89-116% of the task alone. Fixed here by replacing the two disqualified
#      environments with two whose priors are genuinely incomplete.
#
#   2. How many labels does it take? Unanswerable, because nothing measured
#      generalisation: --ensemble-training full sets val_pairs=[] for every
#      member, so n_val_pairs was 0 in all 480 runs and reward-model training
#      accuracy reached exactly 1.0000 on Pusher and Walker2d. A model that is
#      100% correct on its own labels cannot respond to getting more of them.
#      Fixed here by --holdout-pairs 100.
#
# Three settings below are the fix, and they are the reason this grid is not
# just PRE4 again:
#   --holdout-pairs 100                       100 pairs/round from trajectories
#                                             no query touches, rated by the
#                                             true reward, never trained on,
#                                             scored before AND after each round
#   --ensemble-bootstrap                      PRE4's three ensemble members had
#                                             max-min training accuracy of
#                                             exactly 0.000 on Pusher and
#                                             Walker2d: one function, not three
#   --reward-model-train-accuracy-stop 0.97   B-Pref's actual rule, verified in
#                                             train_PEBBLE.py
# ============================================================================

if [ ! -f jobs/params_final.txt ]; then
  echo "jobs/params_final.txt missing; run: python jobs/make_params_final.py" >&2
  exit 2
fi
if [ "$(wc -l < jobs/params_final.txt)" -ne 310 ]; then
  echo "jobs/params_final.txt must contain exactly 310 lines" >&2
  exit 2
fi

# Fail before burning a queue slot if the checked-out CLI predates these flags.
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
# PER-ENVIRONMENT SETTINGS. Every number here was verified on 2026-08-12, not
# remembered:
#   - FRAGMENT vs measured episode length. Walker2d's 16 starved cells came from
#     fragment 50 against ~18-step early episodes, so round 0 bought 2-3 pairs
#     instead of 70. Measured episode lengths: Pusher exactly 100, HalfCheetah
#     and Swimmer exactly 1000 (neither can terminate), LunarLander 100 random
#     to ~1000 trained.
#   - COLLECTION large enough to supply BOTH the queries and the 100-pair
#     holdout. Verified by running round 0 at the highest per-round demand:
#     LunarLander delivered 140/140 with 1379 fragments available; HalfCheetah
#     and Swimmer delivered 70/70 with 1000 available.
#   - NENV matches the preset's own `reference` block wherever the preset is
#     tuned. Ignoring that is exactly what broke Hopper: the rl-zoo block at
#     n_envs 8 gets 488 updates over 2M instead of the intended ~3,900.
# ---------------------------------------------------------------------------
GAMMA=()
case "$CELL" in
  ll)
    # gamma pinned to .99: the tuned LunarLander preset uses .999, which is the
    # time limit itself as an effective horizon and is what broke the PRE3 gate.
    # Block A confirmed gamma is the whole fix (the output-L1 lever was +7.9,
    # 6/10 seeds, p=0.49 - nothing).
    SUITE=box2d;  ENV=LunarLander-v3; PARTIAL=lunar_lander_approach
    STEPS=2000000; FRAGMENT=25; COLLECTION=40000; NENV=8
    GAMMA=(--policy-learning-kwargs '{gamma:0.99}') ;;
  pusher)
    # rl-zoo has no Pusher block, so --tuned-hyperparams is inert here and this
    # is stock SB3. Say so in the paper: the strongest PRE4 result is on untuned
    # PPO. It converges by ~1.5M with a tight true arm (-26.0..-21.9, 0/10
    # failures), so the defaults are adequate.
    SUITE=mujoco; ENV=Pusher-v5;      PARTIAL=pusher_honest
    STEPS=3000000; FRAGMENT=25; COLLECTION=20000; NENV=8 ;;
  cheetah)
    # Never terminates. Its partial drops a 0.1-weight control cost on 6
    # actuators - the same shape as pusher_honest, the only PRE4 prior that was
    # genuinely incomplete. Known risk is bimodality, but that was measured on
    # STOCK SB3: all 399 archived HalfCheetah runs have tuned_hyperparams unset
    # and not one is a mode=true run. Stage 0 is that missing measurement.
    SUITE=mujoco; ENV=HalfCheetah-v5; PARTIAL=halfcheetah_forward
    STEPS=3000000; FRAGMENT=50; COLLECTION=50000; NENV=1 ;;
  swimmer)
    # The only environment here that CANNOT terminate at all, so the lowest
    # variance available. gamma 0.9999 comes from the preset and is load-bearing
    # - do not override it. Deleting the control cost does NOT give an
    # incomplete prior here (weight 1e-4); swimmer_speed rewards |velocity| and
    # is blind to direction instead (correlates 0.23 with reward_forward against
    # 0.97 for x-velocity alone).
    SUITE=mujoco; ENV=Swimmer-v5;     PARTIAL=swimmer_speed
    STEPS=2000000; FRAGMENT=50; COLLECTION=50000; NENV=4 ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

LOGDIR="logs/final_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"
# Only guards against re-running a FINISHED run. It does NOT stop two concurrent
# submissions writing into the same directory -- check `squeue -u $USER` first.
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

# Resolve the partial before burning a queue slot. This is the exact call the
# trainer makes. NOT `rcomp validate-partial`: that feeds a 4-element dummy
# observation and fails on every 8-dim LunarLander partial, including working
# ones. And `rcomp list-partials` prints bare names even for partials that live
# inside a multi-partial file and need the file:name form.
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
  --tuned-hyperparams --n-envs "$NENV" --final-policy last
  --timesteps "$STEPS" --seed "$SEED"
  --eval-freq 50000 --n-eval-episodes 20
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
)

# Everything the standardized reward model shares. Identical to PRE4 except the
# three lines marked NEW, so PRE4 remains the direct comparison.
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
  --holdout-pairs 100                        # NEW: the real measurement
  --ensemble-bootstrap                       # NEW: three members, not one
  --reward-model-train-accuracy-stop 0.97    # NEW: B-Pref's actual stop
  --save-reward-model
)

UNIFORM=(--no-active-learning)
ACTIVE=(--active-learning --active-query-strategy ensemble --active-candidate-protocol pool --active-pool-multiplier 10)
NAIVE=(--mode naive --partial-alpha 1.0 --no-include-partial-feature)
BT=(--pretrain-reward-model --pretrain-target partial --pretrain-loss bt)

case "$VARIANT" in
  # --- stages 0 and 1: references and the headline ---------------------------
  true_std)     ARGS+=(--mode true "${GAMMA[@]}") ;;
  partial_std)  ARGS+=(--mode partial "${GAMMA[@]}") ;;
  feedback_std) ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode feedback "${GAMMA[@]}") ;;
  naive_std)    ARGS+=("${RM[@]}" "${UNIFORM[@]}" "${NAIVE[@]}" "${GAMMA[@]}") ;;

  # --- stage 2: the scale law ------------------------------------------------
  # alpha 1.0 is naive_std at ll/q350, so it is not duplicated.
  alpha010) ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode naive --partial-alpha 0.10 --no-include-partial-feature "${GAMMA[@]}") ;;
  alpha025) ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode naive --partial-alpha 0.25 --no-include-partial-feature "${GAMMA[@]}") ;;
  alpha050) ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode naive --partial-alpha 0.50 --no-include-partial-feature "${GAMMA[@]}") ;;

  # --- stage 3: cold start 2x2 -----------------------------------------------
  # (none, uniform) is feedback_std at ll/q350, so it is not duplicated.
  bt_uniform)    ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode feedback "${BT[@]}" "${GAMMA[@]}") ;;
  al_none)       ARGS+=("${RM[@]}" "${ACTIVE[@]}"  --mode feedback "${GAMMA[@]}") ;;
  bt_al)         ARGS+=("${RM[@]}" "${ACTIVE[@]}"  --mode feedback "${BT[@]}" "${GAMMA[@]}") ;;
  # One naive AL cell: PRE4's AL nulls were all measured with a collapsed
  # ensemble, so the whole question is reopened, and the prior is the condition
  # under which better queries are most likely to matter.
  al_none_naive) ARGS+=("${RM[@]}" "${ACTIVE[@]}"  "${NAIVE[@]}" "${GAMMA[@]}") ;;

  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

srun python -m rcomp train "${ARGS[@]}"
