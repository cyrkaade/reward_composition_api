#!/bin/bash
#SBATCH --job-name=alphaup
#SBATCH --output=logs/slurm/alphaup_%A_%a.out
#SBATCH --error=logs/slurm/alphaup_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-120
#SBATCH --requeue

# Does the MuJoCo null come from the prior being invisible next to the model?
#
# alpha has only ever been tested at <= 1 (gatealpha) or in {1,2,3} historically.
# But the two signals are on wildly different scales, and what the composed reward
# actually depends on is the ratio std(model) / std(alpha * partial). Measured from
# partial_reward_std / model_reward_output_std at alpha = 1:
#
#   LunarLander approach   partial 4.56  model ~0.9   ratio ~0.2   prior dominates
#   Reacher                partial 0.097 model 2.76   ratio ~28    model dominates
#   Pusher honest          partial 0.196 model ~10    ratio ~51    model dominates
#
# So on Reacher/Pusher alpha=1 is not "the prior at full strength", it is the prior
# turned almost off - which is a plausible alternative explanation for why those
# envs show no effect from any composition method. Even the biased-partial optimum
# on LunarLander (p50, alpha 0.1) sits at ratio ~1.8; Reacher needs alpha ~15 and
# Pusher alpha ~30 just to reach that.
#
# This job extends the gatealpha curve upward. Everything else is byte-identical to
# run_gatealpha.sh (budget 700, naive, --no-include-partial-feature, --final-policy
# last, same steps, same seeds) and it writes into the SAME logs/ga_<cell>_<variant>
# tree, so the existing a010/a025/a050/a100 arms are the low end of one curve.
#
# Variant names follow gatealpha: aNNN = alpha * 100.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_alphaup.txt)
read -r CELL VARIANT SEED <<< "$LINE"

BUDGET=700

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
  a300)   ALPHA=3.0 ;;
  a1000)  ALPHA=10.0 ;;
  a3000)  ALPHA=30.0 ;;
  a10000) ALPHA=100.0 ;;
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
  --mode naive --partial-alpha "$ALPHA" \
  $EXTRA \
  --no-include-partial-feature \
  --final-policy last \
  --query-budget "$BUDGET" --rlhf-rounds 5 \
  --timesteps "$STEPS" --seed "$SEED" \
  --eval-freq 50000 --n-eval-episodes 20 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
