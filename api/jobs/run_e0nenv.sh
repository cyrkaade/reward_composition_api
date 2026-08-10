#!/bin/bash
#SBATCH --job-name=e0nenv
#SBATCH --output=logs/slurm/e0nenv_%A_%a.out
#SBATCH --error=logs/slurm/e0nenv_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-40
#SBATCH --requeue

# E0-tuned, second attempt. Fixes the two things run_e0tuned.sh got wrong.
#
# HOPPER: --n-envs 1, because the preset is tuned for it.
#   run_e0tuned.sh kept n_envs=8, so n_steps 512 became 4096-sample updates and
#   only 488 of them over 2M steps, instead of the source's ~3,900. Combined
#   with lr 9.8e-5, log_std_init=-2 and gamma 0.999, every seed locked onto the
#   survive-only local optimum inside 0.2M steps and stayed there for the
#   remaining 1.8M:
#
#     e0t_hopper_true     final 1011  peak 1033   ep_len 1000.0
#     e0t_hopper_partial  final 1016  peak 1024   ep_len 1000.0, reward_forward 1.0
#     stock SB3 @2M       final 2073                            (p=0.014 worse)
#
#   i.e. the agent stands perfectly still, banks +1/step for the full episode
#   and never moves. Both arms tied (p=0.678), so that run gives no verdict.
#   --n-envs 1 restores 512-sample updates and ~3,900 of them. DummyVecEnv
#   steps serially, so this is close to wall-clock neutral.
#
# WALKER: 5M instead of 2M, n_envs left at 8.
#   The preset worked there - 4986 at 2M against 3038 for stock SB3 (+1949,
#   p=0.064) - but both arms were still climbing at the buzzer (true +15.3%,
#   partial +11.2% over the last quarter), so the partial's 370-point lead is
#   the same unconverged-true-arm confound as before. Untuned Walker at 5M
#   already qualifies (true 5945 > partial 5793, peak 6285 > 5922), so the
#   tuned arm needs the same budget to be comparable.
#
# Everything else is unchanged from run_e0tuned.sh: original partials, tuned
# preset, --final-policy last, 10 seeds.
#
# READ THE RESULT WITH jobs/peak_vs_final.py - median peak next to median final.
#
# DECISION RULE (unchanged)
#   true > partial on BOTH final and peak -> envs qualify, keep the original
#                                            partials, do NOT submit run_fix.sh
#   true <= partial on final only         -> stability; try
#                                            --policy-learning-kwargs '{clip_range_vf:0.5}'
#   true <= partial on PEAK               -> the partial genuinely wins; only
#                                            then is run_fix.sh justified
#   both arms tied at a floor / still     -> the run gives no verdict; do not
#   climbing at the last eval                read anything into the sign

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
  hopper) NENV="--n-envs 1" ;;
  walker) NENV=""; STEPS=5000000 ;;
  *) echo "this job only covers hopper and walker" >&2; exit 1 ;;
esac

if [ "$MODE" = "true" ]; then
  PARTIAL_FLAG=""
else
  PARTIAL_FLAG="--partial $PARTIAL"
fi

LOGDIR="logs/e0n_${CELL}_${MODE}"
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
