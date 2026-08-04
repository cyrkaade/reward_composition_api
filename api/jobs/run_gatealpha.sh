#!/bin/bash
#SBATCH --job-name=gatealpha
#SBATCH --output=logs/slurm/gatealpha_%A_%a.out
#SBATCH --error=logs/slurm/gatealpha_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-300
#SBATCH --requeue

# Experiment 1: does the per-state gate beat a plain constant alpha?
#
# The gate can only shrink the partial (g is in [0,1]), so a gate that helps may
# simply be acting as a smaller alpha. Every gate arm therefore has fixed-alpha
# controls at 1.0 / 0.5 / 0.25 / 0.1 run under otherwise identical settings.
#
# All arms use --no-include-partial-feature. If the partial is also a model input
# then h already contains it and g measures redundancy rather than trust, and the
# gate and its controls would not be comparable.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_gatealpha.txt)
read -r CELL VARIANT SEED <<< "$LINE"

BUDGET=700

case "$CELL" in
  ll_approach)
    SUITE=box2d;  ENV=LunarLander-v3; PARTIAL=lunar_lander_approach
    STEPS=2000000; EXTRA="--collection-timesteps 30000" ;;
  ll_p50)
    SUITE=box2d;  ENV=LunarLander-v3; PARTIAL=lunar_lander_levels:lunarlander_p50
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
  a100) MODEFLAGS="--mode naive --partial-alpha 1.0" ;;
  a050) MODEFLAGS="--mode naive --partial-alpha 0.5" ;;
  a025) MODEFLAGS="--mode naive --partial-alpha 0.25" ;;
  a010) MODEFLAGS="--mode naive --partial-alpha 0.1" ;;
  gate) MODEFLAGS="--mode naive --partial-alpha 1.0 --gate-partial --gate-holdout \
                   --gate-init 0.95 --gate-lr 0.001 --gate-prior-penalty 0.01 \
                   --gate-patience 10 --gate-diagnostic" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

LOGDIR="logs/ga_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"

# Idempotent: a finished run is skipped, so re-running the array only fills gaps.
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train \
  --suite "$SUITE" --env-id "$ENV" --partial "$PARTIAL" \
  $MODEFLAGS $EXTRA \
  --no-include-partial-feature \
  --final-policy last \
  --query-budget "$BUDGET" --rlhf-rounds 5 \
  --timesteps "$STEPS" --seed "$SEED" \
  --eval-freq 50000 --n-eval-episodes 20 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
