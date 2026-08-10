#!/bin/bash
#SBATCH --job-name=pre3r
#SBATCH --output=logs/slurm/pre3r_%A_%a.out
#SBATCH --error=logs/slurm/pre3r_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-20
#SBATCH --requeue

# PRE3 REPAIR: staged LunarLander reward-model bridge (40 parameter rows).
#
# SUBMIT ONLY LINES 1--20 FIRST:
#   python jobs/make_params_pre3_repair.py
#   wc -l jobs/params_pre3_repair.txt       # must be 40
#   mkdir -p logs/slurm
#   sbatch --array=1-20%20 jobs/run_pre3_repair.sh
#
# Analyze those runs with:
#   python jobs/analyze_pre3_repair.py --stage primary
#
# Lines 1--20 retain the bounded pre3 standard model (3x256, ensemble 3,
# full-buffer training) but stop each member when its full-epoch training
# ranking accuracy exceeds 0.97.  This is a narrow B-Pref-inspired repair, not
# an exact B-Pref reproduction: this code recreates optimizers each RLHF round
# and stopping is independent per ensemble member.
#
# Lines 21--40 are a fallback only.  They restore the exact historical reward
# model (1x200, ensemble 1, lr .01, summed loss, both L1 penalties, validation
# early stopping) under the same tuned PPO and fair data/query protocol.  Run
# them only if the primary analyzer prints the fallback submission command.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api

if [ ! -f jobs/params_pre3_repair.txt ]; then
  echo "missing jobs/params_pre3_repair.txt; run make_params_pre3_repair.py first" >&2
  exit 2
fi

if [ "$(wc -l < jobs/params_pre3_repair.txt)" -ne 40 ]; then
  echo "jobs/params_pre3_repair.txt must contain exactly 40 lines" >&2
  exit 2
fi

HELP_TEXT="$(python -m rcomp train --help)"
REQUIRED_FLAGS=(
  --reward-hidden-sizes
  --reward-model-ensemble-size
  --reward-model-lr
  --reward-model-batch-size
  --reward-model-epochs
  --reward-model-patience
  --reward-model-train-accuracy-stop
  --reward-model-loss-reduction
  --reward-model-l1
  --reward-output-l1
  --ensemble-training
  --round0-data-protocol
  --tanh-model-reward
  --dedicated-query-rng
  --reward-model-diagnostics
  --save-reward-model
)
for FLAG in "${REQUIRED_FLAGS[@]}"; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required repair flag: $FLAG" >&2
    echo "pull the complete pre3-repair revision before submitting this job" >&2
    exit 2
  fi
done

# tr removes carriage returns if the params file was generated on Windows.
LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_pre3_repair.txt | tr -d '\r')"
read -r CELL VARIANT SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
if [ "$CELL" != "ll" ]; then
  echo "unknown cell: $CELL" >&2
  exit 2
fi

LOGDIR="logs/pre3r_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q350_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

ARGS=(
  --suite box2d
  --env-id LunarLander-v3
  --partial lunar_lander_approach
  --tuned-hyperparams
  --final-policy last
  --timesteps 2000000
  --seed "$SEED"
  --eval-freq 50000
  --n-eval-episodes 20
  --run-name "$RUNNAME"
  --log-dir "$LOGDIR"
  --query-budget 350
  --rlhf-rounds 5
  --no-active-learning
  --collection-timesteps 30000
  --fragment-length 25
  --round0-data-protocol separate
  --dedicated-query-rng
  --reward-model-diagnostics
  --save-reward-model
  --reward-model-epochs 100
  --reward-model-patience 10
)

case "$VARIANT" in
  adaptive_feedback|adaptive_naive)
    ARGS+=(
      --reward-hidden-sizes 256,256,256
      --reward-model-ensemble-size 3
      --reward-model-lr 0.0003
      --reward-model-batch-size 128
      --reward-model-loss-reduction mean
      --reward-model-l1 0
      --reward-output-l1 0
      --ensemble-training full
      --tanh-model-reward
      --reward-model-train-accuracy-stop 0.97
    )
    ;;
  legacy_feedback|legacy_naive)
    ARGS+=(
      --reward-hidden-sizes 200
      --reward-model-ensemble-size 1
      --reward-model-lr 0.01
      --reward-model-batch-size 32
      --reward-model-loss-reduction sum
      --reward-model-l1 0.01
      --reward-output-l1 0.001
      --ensemble-training kfold
    )
    ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

if [[ "$VARIANT" == *_feedback ]]; then
  ARGS+=(--mode feedback)
else
  # M1 is addition at inference; do not also hand the partial to the model.
  ARGS+=(--mode naive --partial-alpha 1.0 --no-include-partial-feature)
fi

srun python -m rcomp train "${ARGS[@]}"
