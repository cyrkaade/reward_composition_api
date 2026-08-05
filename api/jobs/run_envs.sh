#!/bin/bash
#SBATCH --job-name=envs
#SBATCH --output=logs/slurm/envs_%A_%a.out
#SBATCH --error=logs/slurm/envs_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-180
#SBATCH --requeue

# Job A: extend the headline table to two more environments.
#
# run_main.sh covers LunarLander / Pusher / Reacher. This adds Hopper and
# Walker2d, both early-termination, which is the regime where RLHF collapses -
# so they are where the "the prior suppresses overoptimization" claim should
# show up most clearly. Pusher stays in as the negative control: it has no
# collapse (5-7% peak-to-final for every method), which is exactly why nothing
# subtle ever appears there.
#
# 15 seeds is sufficient here: naive-vs-feedback is an ~85%-vs-30% effect, which
# has power 0.81 at n=15 (paired McNemar). Do not reuse this seed count for the
# smaller pretraining contrasts - see run_pretrain.sh.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_envs.txt)
read -r CELL VARIANT BUDGET SEED <<< "$LINE"

case "$CELL" in
  hopper)
    ENV=Hopper-v5;   PARTIAL=hopper_capped_forward_survive; STEPS=5000000 ;;
  walker)
    ENV=Walker2d-v5; PARTIAL=walker2d_survive_forward;      STEPS=5000000 ;;
  *) echo "unknown cell: $CELL" >&2; exit 1 ;;
esac

case "$VARIANT" in
  feedback) MODEFLAGS="--mode feedback"; PARTIAL_FLAG="" ;;
  naive)    MODEFLAGS="--mode naive --partial-alpha 1.0"; PARTIAL_FLAG="--partial $PARTIAL" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

LOGDIR="logs/env_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train \
  --suite mujoco --env-id "$ENV" \
  $PARTIAL_FLAG $MODEFLAGS \
  --final-policy last \
  --query-budget "$BUDGET" --rlhf-rounds 5 \
  --timesteps "$STEPS" --seed "$SEED" \
  --eval-freq 50000 --n-eval-episodes 20 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
