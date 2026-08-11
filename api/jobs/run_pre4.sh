#!/bin/bash
#SBATCH --job-name=pre4
#SBATCH --output=logs/slurm/pre4_%A_%a.out
#SBATCH --error=logs/slurm/pre4_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-480
#SBATCH --requeue

# PRE4: one submission for every open question, on four environments with tuned
# PPO and the standardized (B-Pref) reward model throughout.
#
# ---------------------------------------------------------------------------
# WHY THE GATE FAILED, AND WHAT THIS FIXES
# ---------------------------------------------------------------------------
# The PRE3 gate did not fail ambiguously; it failed by EPISODE-LENGTH FARMING.
# LunarLander's time limit is 1000 steps. Median final mean episode length:
# true 202, partial 525, bounded feedback 964, bounded naive 992. The failing
# policies stopped landing and hovered until timeout. Per seed in
# pre3g_ll_naive_bounded, all six seeds that lost solved status grew their mean
# episode length by 175-620 steps (median +476); the one seed that kept it grew
# by 12. Runs ending at the limit have median final -74.2 (n=23) against +170.8
# for runs still terminating (n=17). Hovering went from 11/90 (12%) across the
# archive to 23/40 (58%) in the gate.
#
# Two causes, introduced together, neither sufficient alone:
#
#   1. DISCOUNT. --tuned-hyperparams takes LunarLander from gamma .99 to .999,
#      so the effective horizon 1/(1-gamma) goes 100 -> 1000 steps: exactly the
#      time limit. Isolated evidence with no reward model in the loop -
#      mode=partial runs 163-184 steps at .99 and 525 at .999, while mode=true
#      is immune (185 -> 202) because its landing bonus pays for terminating.
#      THIS IS LUNARLANDER-ONLY: Walker2d's tuned preset is gamma .99, Pusher
#      .99, Reacher .9. So the risk is confined to one environment.
#
#   2. REWARD LEVEL. With equal-length fragments the Bradley-Terry loss is
#      EXACTLY invariant to adding a constant c to every state - both sides gain
#      L*c and the softmax is unchanged - so the reward's additive level is set
#      by initialisation and drift, not by data. In a variable-horizon MDP that
#      level decides whether the policy wants to live forever. The archive
#      pinned it with the output L1 term (still the package default, 0.001);
#      the gate set --reward-output-l1 0.
#
# Every B-Pref environment is fixed-horizon (DMControl 1000 steps, Meta-world
# 500, no early termination), which is why the standard recipe never needed a
# level term and why importing it wholesale broke a variable-horizon task.
#
# THE STANDARD CONFIG here is therefore: tuned PPO, gamma pinned to .99 on
# LunarLander only, output L1 restored to the default 0.001, and otherwise the
# gate's standardized reward model (3x256, ensemble 3, lr 3e-4, batch 128, mean
# BT loss, full-data ensemble training, independently collected round-0 data,
# tanh-bounded output).
#
# ---------------------------------------------------------------------------
# ORDERING AND THE ONE RISK
# ---------------------------------------------------------------------------
# Rows 1-30 are the LunarLander repair. Analyze them AS SOON AS THEY LAND:
#
#     python jobs/analyze_pre4.py --only A
#
# If gamma+L1 is not the winning lever, `scancel` and adjust the LunarLander
# cells only - Walker/Pusher/Reacher carry no configuration risk and their runs
# stay valid either way. Slurm dispatches array tasks roughly in index order, so
# the repair block finishes first.
#
# ---------------------------------------------------------------------------
# WHAT EACH BLOCK ANSWERS
# ---------------------------------------------------------------------------
#   A  rows   1- 30  which lever stops the hovering (gamma / output-L1 / stop)
#   B  rows  31-110  does each env still qualify (true > partial), and where
#                    does collapse happen at all
#   C  rows 111-270  M1 (prior vs vanilla RLHF) and query efficiency, full
#                    ladder 175/350/700 on the two variable-horizon envs
#   D  rows 271-390  pretraining objective: none vs MSE vs hard-BT. MSE pins the
#                    output magnitude to the partial's scale, which BT then
#                    reads as a logit; on LunarLander the MSE-pretrained model
#                    ranks 66% of held-out pairs correctly yet scores BT loss
#                    2.81 against a chance level of 0.69. Verified locally on
#                    one seed: BT_before 2.416 (mse) -> 1.021 (bt).
#   E  rows 391-470  active learning with the CORRECTED candidate pool. The
#                    legacy selector scores whole perfect matchings and keeps
#                    the best; B-Pref draws an independent 10x pool and takes
#                    the per-pair top-k. Any AL null measured with the old
#                    selector was not a test of B-Pref's method.
#   F  rows 471-480  partial design: a pure potential-based prior cannot change
#                    the optimal policy and so carries no incentive to
#                    terminate; does a non-potential terminal term suppress the
#                    farming where it cannot?
#
# Fragment length is 25-50 everywhere (B-Pref uses 50 DMControl / 25 Meta-world)
# with collection raised to match, which also retires the fragment_length=1
# deviation that affected ~3,240 archived MuJoCo runs.
#
# PREFLIGHT:
#   python jobs/make_params_pre4.py
#   wc -l jobs/params_pre4.txt        # must be 480
#   mkdir -p logs/slurm
#   sbatch jobs/run_pre4.sh

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api

if [ ! -f jobs/params_pre4.txt ]; then
  echo "missing jobs/params_pre4.txt; run make_params_pre4.py first" >&2
  exit 2
fi
if [ "$(wc -l < jobs/params_pre4.txt)" -ne 480 ]; then
  echo "jobs/params_pre4.txt must contain exactly 480 lines" >&2
  exit 2
fi

HELP_TEXT="$(python -m rcomp train --help)"
for FLAG in --reward-model-loss-reduction --reward-output-l1 --reward-model-train-accuracy-stop \
            --ensemble-training --round0-data-protocol --tanh-model-reward --dedicated-query-rng \
            --active-candidate-protocol --pretrain-loss --pretrain-holdout --policy-learning-kwargs; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required flag: $FLAG" >&2
    exit 2
  fi
done

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_pre4.txt | tr -d '\r')"
read -r CELL VARIANT BUDGET SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${BUDGET:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

# GAMMA is set only where the tuned preset would otherwise use .999. Verified by
# diffing Suite.ppo_hyperparams with the flag on: LunarLander .999, Walker2d
# .99, Pusher .99, Reacher .9.
GAMMA=()
case "$CELL" in
  ll)
    SUITE=box2d;  ENV=LunarLander-v3; PARTIAL=lunar_lander_approach
    STEPS=2000000; FRAGMENT=25; COLLECTION=30000
    GAMMA=(--policy-learning-kwargs '{gamma:0.99}') ;;
  walker)
    SUITE=mujoco; ENV=Walker2d-v5;    PARTIAL=walker2d_survive_forward
    STEPS=2000000; FRAGMENT=50; COLLECTION=25000 ;;
  pusher)
    SUITE=mujoco; ENV=Pusher-v5;      PARTIAL=pusher_honest
    STEPS=3000000; FRAGMENT=25; COLLECTION=15000 ;;
  reacher)
    SUITE=mujoco; ENV=Reacher-v5;     PARTIAL=reacher_distance_partial
    STEPS=5000000; FRAGMENT=25; COLLECTION=15000 ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

LOGDIR="logs/pre4_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

# Block F swaps the partial; everything else uses the environment's default one.
# MUST be the file:name form. `--partial` resolves as a MODULE name, and
# lunarlander_p50 lives inside partials/lunar_lander_levels.py alongside p25 and
# p75, so the bare name dies at startup with "No module named 'lunarlander_p50'".
# The other four partials happen to sit in files named after themselves, which is
# why only this variant was affected -- and why the first submission produced ten
# instant failures and no runs. `rcomp list-partials` prints the bare name because
# it scans every file in partials/, so it is NOT a check that --partial will work.
if [ "$VARIANT" = "naive_p50" ]; then
  PARTIAL=lunar_lander_levels:lunarlander_p50
fi

# Resolve the partial before burning a queue slot. This is the exact call the
# trainer makes, so it catches the file:name mistake above. NOT `rcomp
# validate-partial`: that feeds a 4-element dummy observation and so fails on
# every 8-dim LunarLander partial, including the four that work.
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
  echo "multi-partial files must be referenced as file:name" >&2
  exit 2
fi

ARGS=(
  --suite "$SUITE" --env-id "$ENV" --partial "$PARTIAL"
  --tuned-hyperparams --final-policy last
  --timesteps "$STEPS" --seed "$SEED"
  --eval-freq 50000 --n-eval-episodes 20
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
)

# Everything the standardized reward model shares, in one place. The ONLY
# difference between the two blocks below is the output L1 term: RM restores the
# package default 0.001, RM_GATE keeps the gate's 0. Spelled out rather than
# patched by index so the difference is readable and cannot drift.
RM_COMMON=(
  --query-budget "$BUDGET" --rlhf-rounds 5
  --collection-timesteps "$COLLECTION" --fragment-length "$FRAGMENT"
  --reward-hidden-sizes 256,256,256
  --reward-model-ensemble-size 3
  --reward-model-lr 0.0003
  --reward-model-batch-size 128
  --reward-model-loss-reduction mean
  --reward-model-l1 0
  --ensemble-training full
  --round0-data-protocol separate
  --dedicated-query-rng
  --tanh-model-reward
)
RM=("${RM_COMMON[@]}" --reward-output-l1 0.001)
RM_GATE=("${RM_COMMON[@]}" --reward-output-l1 0)

UNIFORM=(--no-active-learning)
# B-Pref's disagreement sampling: independent 10x candidate pool, per-pair top-k.
ACTIVE=(--active-learning --active-query-strategy ensemble --active-candidate-protocol pool --active-pool-multiplier 10)
NAIVE=(--mode naive --partial-alpha 1.0 --no-include-partial-feature)
# No --pretrain-holdout: --round0-data-protocol separate above already collects
# independently seeded A/B streams, so pretraining and the round-0 queries are
# drawn from disjoint rollouts. Combining the two is rejected by config
# validation because --pretrain-holdout is the older ordered split.
BT=(--pretrain-reward-model --pretrain-target partial --pretrain-loss bt)
MSE=(--pretrain-reward-model --pretrain-target partial --pretrain-loss mse)

case "$VARIANT" in
  # --- A. repair levers, LunarLander (baseline = logs/pre3g_ll_naive_bounded) -
  # Each is the GATE config plus exactly one lever. The both-levers arm is
  # naive_std at ll/q350 below, so it is not duplicated.
  naive_g99)   ARGS+=("${RM_GATE[@]}" "${UNIFORM[@]}" "${NAIVE[@]}" "${GAMMA[@]}") ;;
  naive_l1)    ARGS+=("${RM[@]}"      "${UNIFORM[@]}" "${NAIVE[@]}") ;;
  naive_stop)  ARGS+=("${RM_GATE[@]}" "${UNIFORM[@]}" "${NAIVE[@]}" --reward-model-train-accuracy-stop 0.97) ;;

  # --- B. references ---------------------------------------------------------
  true_std)    ARGS+=(--mode true "${GAMMA[@]}") ;;
  partial_std) ARGS+=(--mode partial "${GAMMA[@]}") ;;

  # --- C. M1 and query efficiency -------------------------------------------
  feedback_std) ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode feedback "${GAMMA[@]}") ;;
  naive_std)    ARGS+=("${RM[@]}" "${UNIFORM[@]}" "${NAIVE[@]}" "${GAMMA[@]}") ;;

  # --- D. pretraining objective ---------------------------------------------
  mse_feedback) ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode feedback "${MSE[@]}" "${GAMMA[@]}") ;;
  mse_naive)    ARGS+=("${RM[@]}" "${UNIFORM[@]}" "${NAIVE[@]}"    "${MSE[@]}" "${GAMMA[@]}") ;;
  bt_feedback)  ARGS+=("${RM[@]}" "${UNIFORM[@]}" --mode feedback "${BT[@]}"  "${GAMMA[@]}") ;;
  bt_naive)     ARGS+=("${RM[@]}" "${UNIFORM[@]}" "${NAIVE[@]}"    "${BT[@]}"  "${GAMMA[@]}") ;;

  # --- E. active learning ----------------------------------------------------
  al_feedback)    ARGS+=("${RM[@]}" "${ACTIVE[@]}" --mode feedback "${GAMMA[@]}") ;;
  al_naive)       ARGS+=("${RM[@]}" "${ACTIVE[@]}" "${NAIVE[@]}"   "${GAMMA[@]}") ;;
  al_bt_feedback) ARGS+=("${RM[@]}" "${ACTIVE[@]}" --mode feedback "${BT[@]}" "${GAMMA[@]}") ;;
  al_bt_naive)    ARGS+=("${RM[@]}" "${ACTIVE[@]}" "${NAIVE[@]}"   "${BT[@]}" "${GAMMA[@]}") ;;

  # --- F. partial design, at the UNREPAIRED gate config ----------------------
  # PARTIAL was already swapped to lunarlander_p50 above, so this is the gate's
  # naive_bounded arm with the partial as the only difference.
  naive_p50) ARGS+=("${RM_GATE[@]}" "${UNIFORM[@]}" "${NAIVE[@]}") ;;

  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

srun python -m rcomp train "${ARGS[@]}"
