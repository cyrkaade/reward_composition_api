#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=logs/slurm/pretrain_%A_%a.out
#SBATCH --error=logs/slurm/pretrain_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-480
#SBATCH --requeue

# Job B: does pretraining on the partial AUGMENT naive, and where does any gain
# come from?
#
# The previous batch only ever ran pretraining on top of `feedback`, which
# answered "can pretraining replace naive" (no). It never ran naive+pretrain,
# which is the combination worth having. This is that 2x2:
#
#                     active learning ON        active learning OFF
#   pretrain ON       pre_al                    pre_noal
#   pretrain OFF      nopre_al                  nopre_noal
#
# The active-learning factor isolates the cold-start effect. In trainer.py the
# round-0 query selector is:
#     query_model = self.reward_models if (total_queries > 0 or pretraining_done) else None
# so WITHOUT pretraining round 0 has no model and picks queries at random, while
# WITH pretraining the partial-trained model drives query selection from the very
# first round. Crossing with --no-active-learning separates "better initial
# reward model" from "better round-0 queries".
#
# --no-include-partial-feature is REQUIRED, not optional: in naive mode the
# partial is a model input by default, and pretraining a network to predict an
# input it can already see is trivial. All four arms use it so they stay
# comparable.
#
# 30 seeds, not 15. This contrast is much smaller than naive-vs-feedback; a
# 60%-vs-30% effect has power 0.23 at n=15 and 0.57 at n=30 (paired McNemar).
# Budgets stop at 700 because the pretraining benefit decays as labels get
# cheap (it recovered 88% / 65% / 53% of naive's gain at 350 / 700 / 1400).

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_pretrain.txt)
read -r CELL VARIANT BUDGET SEED <<< "$LINE"

case "$CELL" in
  ll)
    SUITE=box2d;  ENV=LunarLander-v3; PARTIAL=lunar_lander_approach
    STEPS=2000000; EXTRA="--collection-timesteps 30000" ;;
  hopper)
    SUITE=mujoco; ENV=Hopper-v5;      PARTIAL=hopper_capped_forward_survive
    STEPS=5000000; EXTRA="" ;;
  *) echo "unknown cell: $CELL" >&2; exit 1 ;;
esac

case "$VARIANT" in
  pre_al)      PRE="--pretrain-reward-model --pretrain-target partial"; AL="--active-learning" ;;
  pre_noal)    PRE="--pretrain-reward-model --pretrain-target partial"; AL="--no-active-learning" ;;
  nopre_al)    PRE="";                                                  AL="--active-learning" ;;
  nopre_noal)  PRE="";                                                  AL="--no-active-learning" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

LOGDIR="logs/pt_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train \
  --suite "$SUITE" --env-id "$ENV" --partial "$PARTIAL" \
  --mode naive --partial-alpha 1.0 \
  --no-include-partial-feature \
  $PRE $AL $EXTRA \
  --final-policy last \
  --query-budget "$BUDGET" --rlhf-rounds 5 \
  --timesteps "$STEPS" --seed "$SEED" \
  --eval-freq 50000 --n-eval-episodes 20 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
