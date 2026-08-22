#!/bin/bash
#SBATCH --job-name=align2
#SBATCH --account=aalto_users
#SBATCH --output=logs/slurm/align2_%A_%a.out
#SBATCH --error=logs/slurm/align2_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-125
#SBATCH --requeue
set -euo pipefail

# Follow-up to the grid that generated starc_grid_heatmap.png: five supplied
# LunarLander partials x five weighted-sum alphas, q400, 1M timesteps, 5 seeds.
# The requested grid is exactly 125 rows and contains no extra control arms.
# All training settings below are copied verbatim from run_starc.sh.

export PATH="/scratch/work/akishea1/envs/rcomp/bin:${PATH}"

PARAMS_FILE="${PARAMS_FILE:-jobs/params_align2.txt}"
LAUNCHER="${LAUNCHER-srun}"
EXPECTED_ROWS="${EXPECTED_ROWS:-125}"
TIMESTEPS="${TIMESTEPS:-1000000}"

if [ ! -f "$PARAMS_FILE" ]; then
  echo "$PARAMS_FILE missing; run: python jobs/make_params_align2.py" >&2
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

LOGDIR="logs/align2_${CELL}_${ARM}"
RUNNAME="align2_${CELL}_${ARM}_seed${SEED}"
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

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

ARGS=(
  --suite "$SUITE" --env-id "$ENV"
  --n-envs 8 --policy-learning-kwargs "$PPO_KWARGS" --device cpu
  --timesteps "$TIMESTEPS" --seed "$SEED"
  --final-policy last
  --eval-freq 20000 --n-eval-episodes "$EVALEP" --final-eval-episodes 30
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
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
  --mode weighted_sum --partial "$PARTIAL"
  --partial-alpha "$ALPHA"
  --normalize-partial-reward --normalize-model-reward
)

${LAUNCHER} python -m rcomp train "${ARGS[@]}"
