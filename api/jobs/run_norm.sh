#!/bin/bash
#SBATCH --job-name=norm
#SBATCH --output=logs/slurm/norm_%A_%a.out
#SBATCH --error=logs/slurm/norm_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-300
#SBATCH --requeue

# N1: is the hand-written prior simply too QUIET on most environments?
#
# The composed reward is alpha*partial + model_output. Nothing ties those two to
# the same scale - the Bradley-Terry loss only cares about differences between
# clips, so the model output can settle anywhere. Measured per step in the
# current runs (partial : model output std):
#
#     LunarLander  2.84x   <- the partial is LOUDER than the model
#     Walker2d     0.14x
#     Pusher       0.07x
#     Reacher      0.05x
#     Hopper       0.03x   <- the model is 38x louder than the partial
#
# A ~100x spread. And LunarLander, the one environment where the partial is the
# louder signal, is also the one where naive helps most (+105 to +154) and where
# it removes the collapse. On the four MuJoCo environments, where the model
# drowns the partial out, naive's benefit is small (Pusher +8, Reacher +3).
#
# --normalize-model-reward standardises the model output to unit variance using
# running statistics, which puts every environment within ~7x instead of ~100x.
#
# PREDICTION, so this is falsifiable: normalisation should HELP the four MuJoCo
# environments (the partial stops being drowned out) and may HURT LunarLander
# (the partial currently dominates at 2.84x, and normalising drops it to ~0.5x).
# If instead it changes nothing anywhere, scale is not the issue and alpha is
# purely about how much the partial can be trusted - which is what the alpha
# sweep already suggests, since ll_approach and ll_p50 share an environment and
# a scale but want alpha 1.0 and 0.1 respectively.
#
# Only run N2 (an alpha sweep under normalisation, asking whether a single alpha
# then transfers across environments) if this job shows scale matters at all.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api
source jobs/env_common.sh

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_norm.txt)
read -r CELL VARIANT BUDGET SEED <<< "$LINE"
set_env_vars "$CELL"

case "$VARIANT" in
  raw)  NORM="" ;;
  norm) NORM="--normalize-model-reward" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

LOGDIR="logs/nm_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train \
  --suite "$SUITE" --env-id "$ENV" --partial "$PARTIAL" \
  --mode naive --partial-alpha 1.0 \
  $NORM $EXTRA \
  --reward-model-diagnostics \
  --final-policy last \
  --query-budget "$BUDGET" --rlhf-rounds 5 \
  --timesteps "$STEPS" --seed "$SEED" \
  --eval-freq 50000 --n-eval-episodes 20 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
