#!/bin/bash
#SBATCH --job-name=hop5m
#SBATCH --output=logs/slurm/hop5m_%A_%a.out
#SBATCH --error=logs/slurm/hop5m_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-40
#SBATCH --requeue

# Hopper-v5 on the GROUND-TRUTH reward: our preset vs SB3 out-of-the-box.
# 10 seeds each, 5M steps, --n-envs 1 for both arms.
#
# WHY THIS EXISTS
# Three Hopper configs have been measured and they are not comparable to each
# other - different budgets, different n_envs, different seed counts:
#
#   SB3 stock     peak 3532 @3.45M -> final 1677    5M, n_envs 8, 10 seeds
#   rl-zoo v4     peak 1033        -> final 1011    2M, n_envs 8, 10 seeds
#   our preset    peak 2995        -> final 2127    1M, n_envs 1,  3 seeds
#
# rl-zoo is settled and excluded: it never learns at all. Its block is tuned for
# Hopper-v4, and v5 stopped paying healthy_reward on unhealthy steps, which makes
# standing still a stronger attractor; combined with log_std_init -2 (action std
# 0.135 vs 1.0) every seed parked on the survive-only optimum - reward ~1000,
# episode length 1000.0, total reward_forward 1.0 - and never left.
#
# That leaves ours vs stock, which have never been run head to head. This does
# that, at equal budget, equal n_envs, equal seeds, on one machine.
#
# WHAT THE ARCHIVED STOCK RUN CANNOT TELL US
# It used n_envs=8. Comparing it against a preset only ever validated at n_envs=1
# would confound the hyperparameters with the batch size, which is exactly the
# mistake that produced the earlier tied-at-the-floor Hopper result. So stock is
# re-run here at n_envs=1 rather than read out of logs/base_hopper_true.
#
# WHAT TO EXPECT
# Our preset should win on FINAL. It reached stock's 1M score (1873) roughly five
# times sooner, and it carries clip_range_vf 0.5, which stock lacks - stock's
# value estimate has no brake, and stock is the arm that loses a third of its
# peak. Both are expected to collapse to some degree; Hopper always has. The open
# questions are how big the gap is at 5M, whether our drawdown stays near 35% or
# grows, and whether the 84%-losing seed seen at 1M was a one-off.
#
# READ WITH  python jobs/analyze_hopper5m.py
# It reports the 1M values, the peak WITH ITS TIMESTEP, the final, and the
# drawdown - per seed and as medians. Evaluation runs every 50K steps, so the
# whole curve is in eval/evaluations.npz and no separate 1M job is needed.
#
# WALLTIME: 36h. This arm is gradient-heavy (n_epochs 20, batch 32, n_steps 512
# at n_envs=1 is ~3.1M minibatch updates over 5M steps) and there is NO mid-run
# checkpointing - a timeout loses that run. Completed runs are skipped on
# resubmission, so re-running the array fills gaps.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_hopper5m.txt)
LINE=${LINE%$'\r'}
read -r VARIANT SEED <<< "$LINE"

if [ -z "${VARIANT:-}" ] || [ -z "${SEED:-}" ]; then
  echo "could not parse params line ${SLURM_ARRAY_TASK_ID}: '${LINE}'" >&2
  exit 2
fi

# hopper_capped_forward_survive is the environment's standard partial from
# jobs/env_common.sh - the true reward with the control cost dropped. Only the
# *_partial variants use it.
PARTIAL=hopper_capped_forward_survive

# NORM is the VecNormalize column. 'auto' is what every archived run used and
# what the MuJoCo suite decides on its own (always on); 'off' removes the
# wrapper, which is the only way to run SB3 defaults literally out-of-the-box.
case "$VARIANT" in
  ours)          MODE=true;    TUNED="--tuned-hyperparams"; NORM=auto ;;
  stock)         MODE=true;    TUNED="";                    NORM=auto ;;
  ours_nonorm)   MODE=true;    TUNED="--tuned-hyperparams"; NORM=off  ;;
  stock_nonorm)  MODE=true;    TUNED="";                    NORM=off  ;;
  ours_partial)  MODE=partial; TUNED="--tuned-hyperparams"; NORM=auto ;;
  stock_partial) MODE=partial; TUNED="";                    NORM=auto ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

# Fail loudly if the installed package predates --env-normalize, rather than
# running 40 silently-normalized jobs and calling them an ablation.
if ! python -m rcomp train --help 2>&1 | grep -q -- "--env-normalize"; then
  echo "this rcomp has no --env-normalize; git pull and reinstall before submitting" >&2
  exit 2
fi

LOGDIR="logs/hop5m_${VARIANT}"
RUNNAME="hopper_${VARIANT}_seed${SEED}"
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

ARGS=(
  --suite mujoco --env-id Hopper-v5
  --mode "$MODE"
  --partial "$PARTIAL"
  --timesteps 5000000
  --n-envs 1
  --env-normalize "$NORM"
  --seed "$SEED"
  --final-policy last
  --eval-freq 50000
  --n-eval-episodes 20
  --run-name "$RUNNAME"
  --log-dir "$LOGDIR"
)
if [ -n "$TUNED" ]; then
  ARGS+=("$TUNED")
fi

echo "=== ${RUNNAME} (variant=${VARIANT} mode=${MODE} tuned='${TUNED}' normalize=${NORM}) ==="
python -m rcomp train "${ARGS[@]}"
