#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=logs/slurm/baselines_%A_%a.out
#SBATCH --error=logs/slurm/baselines_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-100
#SBATCH --requeue

# E0: the floor and the ceiling for all five environments.
#
#   partial : train on the hand-written partial alone. If a policy trained on it
#             reaches the ceiling, the partial is not actually incomplete and the
#             environment cannot show anything about composition.
#   true    : train on the ground-truth reward. The best any method could do.
#
# No preferences are involved, so there is no query budget here. 10 seeds.
# These two numbers are what decide which environments enter the paper, and they
# currently exist only for LunarLander and Pusher (at n=5, on the old metric).

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api
source jobs/env_common.sh

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_baselines.txt)
read -r CELL MODE SEED <<< "$LINE"
set_env_vars "$CELL"

if [ "$MODE" = "true" ]; then
  PARTIAL_FLAG=""
else
  PARTIAL_FLAG="--partial $PARTIAL"
fi

LOGDIR="logs/base_${CELL}_${MODE}"
RUNNAME="${CELL}_${MODE}_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train \
  --suite "$SUITE" --env-id "$ENV" \
  --mode "$MODE" $PARTIAL_FLAG $EXTRA \
  --final-policy last \
  --timesteps "$STEPS" --seed "$SEED" \
  --eval-freq 50000 --n-eval-episodes 20 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
