#!/bin/bash
#SBATCH --job-name=mainlast
#SBATCH --output=logs/slurm/mainlast_%A_%a.out
#SBATCH --error=logs/slurm/mainlast_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-270
#SBATCH --requeue

# Experiment 2: the headline table, scored on the FINAL policy.
#
# The default (--final-policy best) picks the best of ~20 checkpoints using the
# ground-truth reward, which no real RLHF pipeline can do. It also hides the
# peak-then-collapse behaviour that is the most interesting result in the data,
# because it always rescues the checkpoint from just before the collapse.
#
# HalfCheetah is dropped: ~45% of its seeds land in a "learned to run" mode and
# the rest do not, so its median is mode-counting rather than an effect. Reacher
# replaces it.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_main.txt)
read -r CELL VARIANT BUDGET SEED <<< "$LINE"

case "$CELL" in
  ll_approach)
    SUITE=box2d;  ENV=LunarLander-v3; PARTIAL=lunar_lander_approach
    STEPS=2000000; EXTRA="--collection-timesteps 30000" ;;
  pusher)
    SUITE=mujoco; ENV=Pusher-v5;      PARTIAL=pusher_honest
    STEPS=3000000; EXTRA="" ;;
  reacher)
    SUITE=mujoco; ENV=Reacher-v5;     PARTIAL=reacher_distance_partial
    STEPS=5000000; EXTRA="" ;;
  *) echo "unknown cell: $CELL" >&2; exit 1 ;;
esac

case "$VARIANT" in
  # pure RLHF: no partial anywhere, the baseline the prior has to beat.
  # --partial is omitted entirely so nothing about the partial can leak in.
  feedback) MODEFLAGS="--mode feedback"; PARTIAL_FLAG="" ;;
  # the simple additive composition
  naive)    MODEFLAGS="--mode naive --partial-alpha 1.0"; PARTIAL_FLAG="--partial $PARTIAL" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

LOGDIR="logs/last_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train \
  --suite "$SUITE" --env-id "$ENV" \
  $PARTIAL_FLAG $MODEFLAGS $EXTRA \
  --final-policy last \
  --query-budget "$BUDGET" --rlhf-rounds 5 \
  --timesteps "$STEPS" --seed "$SEED" \
  --eval-freq 50000 --n-eval-episodes 20 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
