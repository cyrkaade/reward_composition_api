#!/bin/bash
#SBATCH --job-name=combo
#SBATCH --output=logs/slurm/combo_%A_%a.out
#SBATCH --error=logs/slurm/combo_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-600
#SBATCH --requeue

# E2: do M2 and M3 combine, or does each one alone already get the whole gain?
#
# Every arm sums the partial into the reward (M1 = --mode naive), because M1 is
# established. On top of that, two factors are crossed:
#
#                     no partial input      partial as RM input (M2)
#   no pretrain       f0p0  (naive only)    f1p0  (M1+M2)
#   pretrain (M3)     f0p1  (M1+M3)         f1p1  (all three)
#
# 5 envs x 4 cells x {350,700} x 15 seeds. Budgets stop at 700 because those are
# the budgets where vanilla RLHF actually fails - at 1400 it starts working and
# there is no headroom left to tell the mechanisms apart.
#
# --reward-model-diagnostics gives the mechanism measurements out of these same
# runs, no extra jobs:
#   M2: accuracy drop when the partial input feature is ablated = how much the
#       reward model actually relies on it. (Weight magnitudes cannot answer
#       this - the partial is on a different numeric scale from the obs.)
#   M3: Bradley-Terry loss on held-out pairs BEFORE preference training, so a
#       pretrained model can be compared against a randomly initialised one.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api
source jobs/env_common.sh

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_combo.txt)
read -r CELL VARIANT BUDGET SEED <<< "$LINE"
set_env_vars "$CELL"

case "$VARIANT" in
  f0p0) FEAT="--no-include-partial-feature"; PRE="" ;;
  f1p0) FEAT="--include-partial-feature";    PRE="" ;;
  f0p1) FEAT="--no-include-partial-feature"; PRE="--pretrain-reward-model --pretrain-target partial" ;;
  f1p1) FEAT="--include-partial-feature";    PRE="--pretrain-reward-model --pretrain-target partial" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

LOGDIR="logs/combo_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train \
  --suite "$SUITE" --env-id "$ENV" --partial "$PARTIAL" \
  --mode naive --partial-alpha 1.0 \
  $FEAT $PRE $EXTRA \
  --reward-model-diagnostics --save-reward-model \
  --final-policy last \
  --query-budget "$BUDGET" --rlhf-rounds 5 \
  --timesteps "$STEPS" --seed "$SEED" \
  --eval-freq 50000 --n-eval-episodes 20 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
