#!/bin/bash
#SBATCH --job-name=fisher
#SBATCH --output=logs/slurm/fisher_%A_%a.out
#SBATCH --error=logs/slurm/fisher_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-200
#SBATCH --requeue

# E4: M3's second explanation - pretraining bridges the active-learning cold start.
#
# To pick an informative query you need a reward model, but to get a reward model
# you need queries. At round 0 there is no model, so query selection falls back to
# random. Pretraining on the partial puts a model there in time to choose.
#
#                     ensemble 1            ensemble 5
#   random init       e1_nopre              e5_nopre
#   pretrained        e1_pre                e5_pre
#
# The measurement is --query-fisher-diagnostic: for each round it records the
# Bradley-Terry Fisher information of the queries the selector actually chose,
# p(1-p)*||dR_A - dR_B||^2, i.e. how much their answers are expected to pin the
# reward parameters down. The claim predicts a gap at ROUND 0 specifically, which
# should shrink in later rounds once everyone has a trained model.
#
# Ensembles have never been run in this project (size was 1 in all 3,549 previous
# runs), so the e5 arms are also the first real exercise of that code path. Note
# ensemble size changes the active-learning strategy as a side effect: `auto`
# resolves to ensemble-disagreement above size 1 and MC-dropout at size 1. Read
# pretrain-vs-random WITHIN an ensemble size; treat e1-vs-e5 as secondary.
#
# --no-include-partial-feature is required: with the partial as a model input,
# pretraining it to predict the partial is trivial and the comparison is void.
#
# Budget 350 only - the cold start is a scarce-query problem, and 10 seeds is
# ample because the Fisher number is itself an average over hundreds of queries.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api
source jobs/env_common.sh

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_fisher.txt)
read -r CELL VARIANT SEED <<< "$LINE"
set_env_vars "$CELL"

BUDGET=350

case "$VARIANT" in
  e1_nopre) ENS="--reward-model-ensemble-size 1"; PRE="" ;;
  e1_pre)   ENS="--reward-model-ensemble-size 1"; PRE="--pretrain-reward-model --pretrain-target partial" ;;
  e5_nopre) ENS="--reward-model-ensemble-size 5"; PRE="" ;;
  e5_pre)   ENS="--reward-model-ensemble-size 5"; PRE="--pretrain-reward-model --pretrain-target partial" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

LOGDIR="logs/fi_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train \
  --suite "$SUITE" --env-id "$ENV" --partial "$PARTIAL" \
  --mode naive --partial-alpha 1.0 \
  --no-include-partial-feature \
  $ENS $PRE $EXTRA \
  --query-fisher-diagnostic --reward-model-diagnostics --save-reward-model \
  --final-policy last \
  --query-budget "$BUDGET" --rlhf-rounds 5 \
  --timesteps "$STEPS" --seed "$SEED" \
  --eval-freq 50000 --n-eval-episodes 20 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
