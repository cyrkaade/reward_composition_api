#!/bin/bash
#SBATCH --job-name=wsuma
#SBATCH --account=aalto_users
#SBATCH --output=logs/slurm/wsuma_%A_%a.out
#SBATCH --error=logs/slurm/wsuma_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=1-130
#SBATCH --requeue
set -euo pipefail

# Atari half of the weighted-sum alpha sweep. Same 13 arms and 10 seeds as
# run_wsum.sh; the reward-model block below is identical except for the two
# Atari-specific flags and the budget-dependent round-0 collection.
export PATH="/scratch/work/akishea1/envs/rcomp/bin:${PATH}"

PARAMS_FILE="${PARAMS_FILE:-jobs/params_wsum_atari.txt}"
LAUNCHER="${LAUNCHER-srun}"
EXPECTED_ROWS="${EXPECTED_ROWS:-130}"
TIMESTEPS="${TIMESTEPS:-400000}"
# The pixel CNN reward model is the whole reason Atari was infeasible before.
# cuda + batched env inference measured 2.66x on an H200 (CLAUDE.md, 2026-08-21).
# The ensemble vmap flag is deliberately NOT set: it measured net-negative.
RM_DEVICE="${RM_DEVICE:-cuda}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

if [ ! -f "$PARAMS_FILE" ]; then
  echo "$PARAMS_FILE missing; run: python jobs/make_params_wsum_atari.py" >&2
  exit 2
fi
if [ "$(wc -l < "$PARAMS_FILE")" -ne "$EXPECTED_ROWS" ]; then
  echo "$PARAMS_FILE must contain exactly $EXPECTED_ROWS lines" >&2
  exit 2
fi
if ! command -v python >/dev/null 2>&1; then
  echo "python not on PATH -- activate /scratch/work/akishea1/envs/rcomp" >&2
  exit 2
fi

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE" | tr -d '\r')"
read -r CELL ARM SEED PARTIAL BUDGET ALPHA <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${ARM:-}" ] || [ -z "${SEED:-}" ] || [ -z "${PARTIAL:-}" ] \
   || [ -z "${BUDGET:-}" ] || [ -z "${ALPHA:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

case "$CELL" in
  mspacman) SUITE=atari; ENV=ALE/MsPacman-v5 ;;
  qbert)    SUITE=atari; ENV=ALE/Qbert-v5 ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

# Fragment 10 = 40 raw frames at frameskip 4. Shorter than the 25 the non-Atari
# cells mostly use (Hopper already uses 10), and chosen for cost: fragment
# length multiplies BOTH the round-0 collection needed for label delivery AND
# the per-pair cost of every reward-model gradient step, so 25 -> 10 is a 2.5x
# cut on the two dominant terms at once.
FRAGMENT="${FRAGMENT:-10}"
COLLECTION="${COLLECTION:-20000}"
# The reward model trains on cumulative pairs each round with batch 32. At the
# 100-epoch default the final q5600 round is 175 batches x 100 epochs x 3
# members, each batch a 640-image CNN forward+backward -- hours per run. The
# 0.97 train-accuracy stop is not guaranteed to fire on pixels, so cap it.
RM_EPOCHS="${RM_EPOCHS:-10}"
# Round 0 is the binding constraint on label delivery. query_model is None
# there, so queries are drawn UNIFORMLY even with --active-learning on, and
# random_query_pairs shuffles and zips -- no fragment is reused, so round 0
# needs 2 fragments per pair. Rounds 1-4 use the candidate pool, which samples
# pairs independently and may reuse a fragment, so they need far fewer.
#
# round0 = 2 * (budget/5) * fragment / 0.946, rounded up for margin. The 0.946
# is measured, not assumed: 56000 steps at fragment 50 on MsPacman yielded 1060
# fragments per stream against an ideal 1120.
ROUND0=$(python -c "
import math
budget, fragment = $BUDGET, $FRAGMENT
need = 2 * (budget / 5) * fragment / 0.946
print(int(math.ceil(need * 1.10 / 1000.0)) * 1000)
")
if [ -z "$ROUND0" ]; then echo "failed to compute ROUND0" >&2; exit 2; fi

LOGDIR="logs/wsuma_${CELL}_${ARM}"
RUNNAME="wsuma_${CELL}_${ARM}_seed${SEED}"
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

if [ "$PARTIAL" != "-" ]; then
  if ! python -c "
import sys
from rcomp.partials import PartialRegistry, load_partial_reference
try:
    spec = load_partial_reference('$PARTIAL', '$SUITE', PartialRegistry())
    spec.create('$ENV')
except Exception as exc:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    sys.exit(1)
"; then
    echo "partial '$PARTIAL' does not resolve/instantiate for $ENV" >&2
    exit 2
  fi
fi

ARGS=(
  --suite "$SUITE" --env-id "$ENV"
  --n-envs 8 --device "$POLICY_DEVICE"
  --timesteps "$TIMESTEPS" --seed "$SEED"
  --final-policy last
  --eval-freq "${EVAL_FREQ:-25000}" --n-eval-episodes "${EVAL_EPS:-5}" --final-eval-episodes "${FINAL_EVAL_EPS:-20}"
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
)

RM=(
  --query-budget "$BUDGET" --rlhf-rounds 5
  --collection-timesteps "$COLLECTION" --fragment-length "$FRAGMENT"
  --round0-collection-timesteps "$ROUND0"
  --reward-hidden-sizes 256,256,256
  --reward-model-ensemble-size 3
  --reward-model-lr 0.0003
  --reward-model-batch-size 32
  --reward-model-epochs "$RM_EPOCHS"
  --reward-model-loss-reduction mean
  --reward-model-l1 0
  --reward-output-l1 0.001
  --ensemble-training full
  --ensemble-bootstrap
  --reward-model-train-accuracy-stop 0.97
  --round0-data-protocol separate
  --dedicated-query-rng
  --tanh-model-reward --tanh-scale 5
  --active-learning --active-query-strategy ensemble --active-candidate-protocol pool
  --reward-model-device "$RM_DEVICE"
  --batch-env-reward-inference
)

case "$ARM" in
  true)
    ARGS+=(--mode true) ;;
  vanilla_*)
    ARGS+=("${RM[@]}" --mode feedback) ;;
  naive_*)
    ARGS+=("${RM[@]}" --mode naive --partial "$PARTIAL") ;;
  ws*)
    ARGS+=("${RM[@]}" --mode weighted_sum --partial "$PARTIAL"
            --partial-alpha "$ALPHA"
            --normalize-partial-reward --normalize-model-reward) ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

${LAUNCHER} python -m rcomp train "${ARGS[@]}"
