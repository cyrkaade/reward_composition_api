#!/bin/bash
#SBATCH --job-name=pre3g
#SBATCH --output=logs/slurm/pre3g_%A_%a.out
#SBATCH --error=logs/slurm/pre3g_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-100
#SBATCH --requeue

# PRE3 STAGE 0: standard-reward-model / bounding gate (100 runs).
#
# This job deliberately precedes the pretraining and active-learning grids.  It
# asks whether their base reward model should use the PEBBLE/B-Pref-style tanh
# output and whether the paper's collapse mechanism survives that standard
# choice.  Apart from --tanh-model-reward, bounded and unbounded arms are
# identical: 3x256 leaky-ReLU reward model, ensemble 3, lr 3e-4, batch 128,
# mean-reduced BT loss, no L1 penalties, every ensemble member trained on all
# labels, uniform queries, independently collected round-0 data, tuned PPO,
# and last policy.
#
# DECISION RULE (make this decision before launching later stages):
#   1. Stop if any cell has fewer than 10 completed seeds or any preference run
#      delivers other than 350/350 queries.
#   2. LunarLander qualifies only if tuned true > partial on BOTH median peak
#      and median final.  Walker2d qualifies separately only if run_e0tuned.sh
#      gives true > partial on both metrics.
#   3. Use the bounded standard model in later stages.  As a go/no-go PILOT
#      heuristic (not an inferential conclusion), call the collapse contrast
#      worth confirming only if bounded feedback has >=3/10 seeds with >20%
#      peak-to-final drop, at least two more collapsed seeds than bounded naive,
#      and a larger median drawdown.  If bounding removes that contrast, do not
#      build the mechanism claim around reward-model collapse.
#   4. If the bounded model is plainly non-learning (its feedback AND naive
#      median finals are each below half their matched unbounded medians), stop
#      and diagnose scaling before running any pretraining/AL grid.
#
# PREFLIGHT: the job fails before training if the checked-out rcomp CLI is not
# the pre3-capable revision and lacks any standardization/protocol flag below.
# Generate the ordered parameter file before submitting:
#   python jobs/make_params_pre3_gate.py
#   wc -l jobs/params_pre3_gate.txt       # must be 100
#   mkdir -p logs/slurm
#   sbatch jobs/run_pre3_gate.sh
# Analyze the first stage with `python jobs/analyze_pre3_gate.py --cell ll`.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api

if [ ! -f jobs/params_pre3_gate.txt ]; then
  echo "missing jobs/params_pre3_gate.txt; run make_params_pre3_gate.py first" >&2
  exit 2
fi

if [ "$(wc -l < jobs/params_pre3_gate.txt)" -ne 100 ]; then
  echo "jobs/params_pre3_gate.txt must contain exactly 100 lines" >&2
  exit 2
fi

# Do not discover a missing anticipated flag only after allocating hours of
# work.  Bash substring matching avoids a fragile grep pipeline under pipefail.
HELP_TEXT="$(python -m rcomp train --help)"
REQUIRED_FLAGS=(
  --reward-hidden-sizes
  --reward-model-ensemble-size
  --reward-model-lr
  --reward-model-batch-size
  --reward-model-loss-reduction
  --reward-model-l1
  --reward-output-l1
  --ensemble-training
  --round0-data-protocol
  --tanh-model-reward
)
for FLAG in "${REQUIRED_FLAGS[@]}"; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required pre3 flag: $FLAG" >&2
    echo "pull/merge the pre3 core patch before submitting this job" >&2
    exit 2
  fi
done

# tr removes carriage returns if the params file was generated on Windows.
LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_pre3_gate.txt | tr -d '\r')"
read -r CELL VARIANT SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

case "$CELL" in
  ll)
    SUITE=box2d
    ENV=LunarLander-v3
    PARTIAL=lunar_lander_approach
    FRAGMENT=25
    COLLECTION=30000
    ;;
  walker)
    SUITE=mujoco
    ENV=Walker2d-v5
    PARTIAL=walker2d_survive_forward
    FRAGMENT=50
    COLLECTION=15000
    ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

LOGDIR="logs/pre3g_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q350_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

ARGS=(
  --suite "$SUITE"
  --env-id "$ENV"
  --partial "$PARTIAL"
  --tuned-hyperparams
  --final-policy last
  --timesteps 2000000
  --seed "$SEED"
  --eval-freq 50000
  --n-eval-episodes 20
  --run-name "$RUNNAME"
  --log-dir "$LOGDIR"
)

case "$VARIANT" in
  true)
    ARGS+=(--mode true)
    ;;
  partial)
    ARGS+=(--mode partial)
    ;;
  feedback_unbounded|feedback_bounded|naive_unbounded|naive_bounded)
    ARGS+=(
      --query-budget 350
      --rlhf-rounds 5
      --no-active-learning
      --collection-timesteps "$COLLECTION"
      --fragment-length "$FRAGMENT"
      --reward-hidden-sizes 256,256,256
      --reward-model-ensemble-size 3
      --reward-model-lr 0.0003
      --reward-model-batch-size 128
      --reward-model-loss-reduction mean
      --reward-model-l1 0
      --reward-output-l1 0
      --ensemble-training full
      --round0-data-protocol separate
    )
    if [[ "$VARIANT" == feedback_* ]]; then
      ARGS+=(--mode feedback)
    else
      # The reward model must not be handed the partial as an input feature:
      # naive is the persistent additive prior, not the M2 feature mechanism.
      ARGS+=(--mode naive --partial-alpha 1.0 --no-include-partial-feature)
    fi
    if [[ "$VARIANT" == *_bounded ]]; then
      ARGS+=(--tanh-model-reward)
    fi
    ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

srun python -m rcomp train "${ARGS[@]}"
