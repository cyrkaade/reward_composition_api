#!/bin/bash
#SBATCH --job-name=e0tuned
#SBATCH --output=logs/slurm/e0tuned_%A_%a.out
#SBATCH --error=logs/slurm/e0tuned_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-40
#SBATCH --requeue

# E0-tuned: does Hopper/Walker2d still fail the premise once PPO is set up the
# way the literature sets it up?
#
# The original E0 said the partial beats the true reward on both (Hopper 2436 vs
# 1677, Walker2d 5854 vs 5723). Reading the eval curves rather than the endpoint
# shows why:
#
#   Hopper true    peak 3532 @3.45M -> final 1830   drawdown 20%, 5/10 seeds >20%
#   Hopper partial peak 2627 @2.35M -> final 2440   drawdown  3%, 0/10 seeds
#   Walker true    peak 6285        -> final 5945   (true WINS on both)
#   Walker partial peak 5922        -> final 5793
#
# The true arm reaches the BETTER policy on Hopper and then loses it, and on
# Walker it wins outright and had not even converged (at 1M: 728 vs the tuned
# reference's 3479). Neither inversion is significant: Hopper Mann-Whitney
# p=0.076 / paired-seed Wilcoxon p=0.131, Walker p=1.000 / p=0.695.
#
# So this job changes exactly two things, both via jobs/env_common.sh:
#   --tuned-hyperparams   per-env PPO block from rcomp/ppo_presets.py
#                         (Hopper: n_steps 512, lr 9.8e-5, gae 0.99, ReLU 256x256
#                          instead of n_steps 2048, lr 3e-4, gae 0.95, Tanh)
#   2M steps              instead of 5M; rl-zoo budgets 1M and we run 8 envs
#
# and re-measures the E0 floor/ceiling for the two broken environments only.
# LunarLander/Pusher/Reacher are untouched: their true arms converge with no
# drawdown (LL 3.0%, Pusher and Reacher flat), so there is nothing to fix and
# changing them would break comparability with the paper's core results.
#
# The ORIGINAL partials are used (hopper_capped_forward_survive,
# walker2d_survive_forward), because they are what the premise is being tested
# on. The low-cap replacements stay available as $PARTIAL_LOWCAP.
#
# No extra control arm is needed for "tuned vs untuned at the same budget": the
# existing 5M runs can be read at their 2M checkpoint (Hopper true 2073, Walker
# true 3038), which isolates the hyperparameter change from the budget change.
#
# READ THE RESULT AS: median FINAL and median PEAK side by side, per arm.
# metadata.json already carries both - selected_policy_true_reward_mean (final,
# because --final-policy last) and best_logged_true_reward / best_logged_timestep
# (peak). A large gap between them is the diagnostic that mattered here.
#
# DECISION RULE
#   true > partial on BOTH final and peak  -> environments qualify, keep the
#                                             original partials, do NOT submit
#                                             run_fix.sh, re-run E1/E2 at 2M
#   true still <= partial on final only    -> still a stability problem; try
#                                             --policy-learning-kwargs
#                                             '{clip_range_vf:0.5}' before
#                                             touching the partials
#   true <= partial on PEAK                -> the partial genuinely beats the
#                                             true reward; now run_fix.sh is
#                                             justified

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api
source jobs/env_common.sh

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_e0tuned.txt)
read -r CELL MODE SEED <<< "$LINE"
set_env_vars "$CELL"

if [ "$MODE" = "true" ]; then
  PARTIAL_FLAG=""
else
  PARTIAL_FLAG="--partial $PARTIAL"
fi

LOGDIR="logs/e0t_${CELL}_${MODE}"
RUNNAME="${CELL}_${MODE}_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train \
  --suite "$SUITE" --env-id "$ENV" \
  --mode "$MODE" $PARTIAL_FLAG $EXTRA $TUNED \
  --final-policy last \
  --timesteps "$STEPS" --seed "$SEED" \
  --eval-freq 20000 --n-eval-episodes 20 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
