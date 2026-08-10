#!/bin/bash
#SBATCH --job-name=e0v2
#SBATCH --output=logs/slurm/e0v2_%A_%a.out
#SBATCH --error=logs/slurm/e0v2_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-40
#SBATCH --requeue

# E0, third attempt. Both previous attempts were wrong for different reasons.
#
# WHAT HAPPENED
#   run_e0tuned.sh    tuned preset, 2M, n_envs=8. Hopper pinned at ~1011 (both
#                     arms), i.e. the survive-only local optimum. No verdict.
#   my n_envs theory  refuted. rl-zoo preset at --n-envs 1 sits at 989 @260k.
#   my gamma theory   refuted. gamma 0.99 alone: 1026 @300k.
#   my n_epochs theory refuted. n_epochs 20 alone: 980 @140k.
#   log_std_init      necessary but not sufficient. Forcing -2 onto the WORKING
#                     config collapses it 2835 -> 452 by 160k, but flipping the
#                     rl-zoo block to 0 still leaves it at 1012 @300k.
#
# The rl-zoo Hopper-v4 block is mismatched in several ways at once for
# Hopper-v5 in this codebase. The Hopper preset in rcomp/ppo_presets.py has been
# replaced with a gSDE-derived config validated here (3 seeds x 1M, n_envs=1):
#
#   median peak 2995   median final 2127   median drawdown 35%
#   per seed: 3285->2127 (-35%) | 2599->418 (-84%) | 2995->2518 (-16%)
#
# vs the rl-zoo block's flat ~1000, and stock SB3's 1873 @1M.
#
# WALKER is unchanged and still uses its rl-zoo block, which does work here
# (4986 vs 3038 for stock at 2M) and survives n_envs=8. It only needs a longer
# budget: both arms were still climbing +15%/+11% in the last quarter at 2M.
#
# WHAT THIS JOB MEASURES
#   Hopper  1M, --n-envs 1, validated preset, true vs partial, 10 seeds
#   Walker  5M, n_envs=8,   rl-zoo preset,    true vs partial, 10 seeds
#
# EXPECT HOPPER TO BE UNUSABLE FOR THE COLLAPSE CLAIM EITHER WAY. It collapses
# under every configuration tried: stock 20%, validated preset 35% median with
# one seed of three losing 84%. That is an environment property, not a setup
# bug, so Hopper cannot support "the prior suppresses reward-model
# overoptimization" - PPO on the GROUND-TRUTH reward already collapses there.
# This job answers only the narrower E0 question: does the partial beat the true
# reward once the true arm can actually learn?
#
# READ WITH jobs/peak_vs_final.py - median peak next to median final.
#
# DECISION RULE
#   true > partial on BOTH final and peak  -> qualifies; keep original partials;
#                                             do NOT submit run_fix.sh
#   true > partial on peak, loses on final -> stability only. Report peak, and
#                                             drop Hopper from any collapse claim.
#   true <= partial on PEAK                -> the partial genuinely wins; only
#                                             then is run_fix.sh justified
#   arms tied at a floor, or still climbing at the last eval -> no verdict

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api
source jobs/env_common.sh

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_e0nenv.txt)
read -r CELL MODE SEED <<< "$LINE"
set_env_vars "$CELL"

# Per-env corrections to the shared table.
case "$CELL" in
  # n_envs=1 matches the validated Hopper measurement above. Do not raise it
  # without re-measuring: this preset has only ever been checked at 1.
  hopper) NENV="--n-envs 1"; STEPS=1000000 ;;
  walker) NENV="";           STEPS=5000000 ;;
  *) echo "this job only covers hopper and walker" >&2; exit 1 ;;
esac

if [ "$MODE" = "true" ]; then
  PARTIAL_FLAG=""
else
  PARTIAL_FLAG="--partial $PARTIAL"
fi

LOGDIR="logs/e0v2_${CELL}_${MODE}"
RUNNAME="${CELL}_${MODE}_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train \
  --suite "$SUITE" --env-id "$ENV" \
  --mode "$MODE" $PARTIAL_FLAG $EXTRA $TUNED $NENV \
  --final-policy last \
  --timesteps "$STEPS" --seed "$SEED" \
  --eval-freq 20000 --n-eval-episodes 20 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
