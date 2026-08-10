#!/bin/bash
#SBATCH --job-name=fix
#SBATCH --output=logs/slurm/fix_%A_%a.out
#SBATCH --error=logs/slurm/fix_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-460
#SBATCH --requeue

# Re-run Hopper and Walker2d with partials that are actually incomplete.
#
# !! DO NOT SUBMIT THIS UNTIL run_e0tuned.sh HAS BEEN CHECKED (2026-08-10) !!
#
# The premise below is probably wrong. The inversion it describes was measured
# with stock SB3 PPO defaults at 5M steps, where Hopper's TRUE arm peaks at 3532
# and then collapses to 1830 (the partial peaks lower, at 2627, but holds it),
# and Walker's true arm had not converged. Neither inversion is statistically
# significant (Hopper p=0.076, Walker p=1.000, n=10). run_e0tuned.sh re-measures
# both with tuned hyperparameters at 2M steps. If the true arm then sits cleanly
# above the partial, these 460 runs are unnecessary and the ORIGINAL partials
# stay. See jobs/env_common.sh for the full numbers.
#
# Original rationale, retained for the record:
#
# The first attempt used hopper_capped_forward_survive and
# walker2d_survive_forward. Both produced BETTER policies than training on the
# ground-truth reward (Hopper 2436 vs 1677, Walker2d 5854 vs 5723, scored on
# true reward), so there was no gap for preference learning to fill and neither
# environment could test the mechanisms. Their forward-progress term was
# effectively unbounded, and removing the temptation to run fast happens to be
# good advice when falling ends the episode.
#
# The replacements cap forward credit low (partials/mujoco_capped_low.py), which
# should give a slow stable gait that the true reward comfortably beats.
#
# This job re-runs all three experiments for these two environments only:
#   base  (E0)  partial-only and true-reward floors/ceilings, 10 seeds
#   main  (E1)  naive vs feedback, 3 budgets, 15 seeds
#   combo (E2)  M2 x M3 on top of naive, 2 budgets, 15 seeds
#
# CHECK E0 FIRST. If partial still beats true on either environment, stop and
# lower the cap further rather than spending the main/combo runs.
#
# Not re-run here: the pretrain (E3), fisher (E4) and normalisation (N1) jobs
# also include Hopper. Redo those only if the environment qualifies this time.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api
source jobs/env_common.sh

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_fix.txt)
read -r CELL GROUP VARIANT BUDGET SEED <<< "$LINE"
set_env_vars "$CELL"

COMMON="--suite $SUITE --env-id $ENV --final-policy last --timesteps $STEPS \
        --seed $SEED --eval-freq 50000 --n-eval-episodes 20"
RLHF="--query-budget $BUDGET --rlhf-rounds 5 --reward-model-diagnostics"

case "$GROUP" in
  base)
    LOGDIR="logs/fx_base_${CELL}_${VARIANT}"
    RUNNAME="${CELL}_${VARIANT}_seed${SEED}"
    if [ "$VARIANT" = "true" ]; then ARGS="--mode true"; else ARGS="--mode partial --partial $PARTIAL_LOWCAP"; fi
    ARGS="$ARGS $EXTRA"
    ;;
  main)
    LOGDIR="logs/fx_main_${CELL}_${VARIANT}"
    RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"
    if [ "$VARIANT" = "feedback" ]; then
      ARGS="--mode feedback $RLHF $EXTRA"
    else
      ARGS="--mode naive --partial-alpha 1.0 --partial $PARTIAL_LOWCAP $RLHF $EXTRA"
    fi
    ;;
  combo)
    LOGDIR="logs/fx_combo_${CELL}_${VARIANT}"
    RUNNAME="${CELL}_${VARIANT}_q${BUDGET}_seed${SEED}"
    case "$VARIANT" in
      f0p0) FEAT="--no-include-partial-feature"; PRE="" ;;
      f1p0) FEAT="--include-partial-feature";    PRE="" ;;
      f0p1) FEAT="--no-include-partial-feature"; PRE="--pretrain-reward-model --pretrain-target partial" ;;
      f1p1) FEAT="--include-partial-feature";    PRE="--pretrain-reward-model --pretrain-target partial" ;;
      *) echo "unknown combo variant: $VARIANT" >&2; exit 1 ;;
    esac
    ARGS="--mode naive --partial-alpha 1.0 --partial $PARTIAL_LOWCAP $FEAT $PRE $RLHF --save-reward-model $EXTRA"
    ;;
  *) echo "unknown group: $GROUP" >&2; exit 1 ;;
esac

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train $COMMON $ARGS --run-name "$RUNNAME" --log-dir "$LOGDIR"
