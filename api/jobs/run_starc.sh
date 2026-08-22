#!/bin/bash
#SBATCH --job-name=starcgrid
#SBATCH --account=aalto_users
#SBATCH --output=logs/slurm/starcgrid_%A_%a.out
#SBATCH --error=logs/slurm/starcgrid_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-135
#SBATCH --requeue
set -euo pipefail

# STARC alignment x alpha on LunarLander: 5 prior rungs x 5 weighted-sum
# alphas, q400, 1M timesteps, 5 seeds, plus true/vanilla references.
#
# The rungs are calibrated to STARC alignment 0.2/0.4/0.6/0.8/1.0 (see
# partials/lunar_lander_starc_alignment.py).  Identical in every other respect
# to run_align.sh, so the two grids are directly comparable.
#
# Plain CPU array job, one rcomp process per task -- the same shape as
# run_wsum.sh, deliberately NOT the GPU-bundled shape used for Atari.  Nothing
# here touches a GPU: LunarLander is an 8-dim observation and the reward model
# is a 3x256 MLP, both of which run faster on CPU than they schedule on one.
#
# Every setting below except --timesteps and --query-budget is copied verbatim
# from run_wsum.sh's `ll` cell, so these runs sit on the same footing as that
# grid's LunarLander arms.

export PATH="/scratch/work/akishea1/envs/rcomp/bin:${PATH}"

PARAMS_FILE="${PARAMS_FILE:-jobs/params_starc.txt}"
LAUNCHER="${LAUNCHER-srun}"
EXPECTED_ROWS="${EXPECTED_ROWS:-135}"
TIMESTEPS="${TIMESTEPS:-1000000}"

if [ ! -f "$PARAMS_FILE" ]; then
  echo "$PARAMS_FILE missing; run: python jobs/make_params_starc.py" >&2
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

SUITE=box2d
ENV=LunarLander-v3
FRAGMENT=25
COLLECTION=40000
EVALEP=10
PPO_KWARGS='{"n_steps":256,"ent_coef":0.01,"target_kl":0.03}'

LOGDIR="logs/starc_${CELL}_${ARM}"
RUNNAME="starc_${CELL}_${ARM}_seed${SEED}"
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
  --n-envs 8 --policy-learning-kwargs "$PPO_KWARGS" --device cpu
  --timesteps "$TIMESTEPS" --seed "$SEED"
  --final-policy last
  --eval-freq 20000 --n-eval-episodes "$EVALEP" --final-eval-episodes 30
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
)

RM=(
  --query-budget "$BUDGET" --rlhf-rounds 5
  --collection-timesteps "$COLLECTION" --fragment-length "$FRAGMENT"
  --round0-collection-timesteps 50000
  --reward-hidden-sizes 256,256,256
  --reward-model-ensemble-size 3
  --reward-model-lr 0.0003
  --reward-model-batch-size 32
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
