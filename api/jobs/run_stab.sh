#!/bin/bash
#SBATCH --job-name=stab
#SBATCH --output=logs/slurm/stab_%A_%a.out
#SBATCH --error=logs/slurm/stab_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-140
#SBATCH --requeue
set -euo pipefail

# ============================================================================
# THE STABILITY PILOT: can either of two fixes repair a broken true-reward
# ceiling on HalfCheetah, Walker2d and Ant, at the true/partial level, before
# the composition experiment is built on top of it.
#
#   python jobs/make_params_stab.py
#   wc -l jobs/params_stab.txt        # must print 140
#   mkdir -p logs/slurm
#   sbatch --array=1-140 jobs/run_stab.sh
#
# See jobs/make_params_stab.py for the measured failure in each cell, why the
# controls are re-run here instead of taken from logs/sel_*, and why Ant is
# kept despite having no ceiling to stabilize.
# ============================================================================

# ---------------------------------------------------------------------------
# ONE BASE CONFIG, SHARED BY ALL FOURTEEN ARMS. Only the treatment varies.
#
#   1M timesteps, n_envs 8, --preset generic, --final-policy last,
#   --policy-learning-kwargs {n_steps:256,ent_coef:0.01}
#
# n_steps 256 at n_envs 8 restores SB3's default 2048-sample rollout and 488 PPO
# iterations per million; leaving the 2048 default at 8 envs gives 61 iterations
# per million and under-trains the CEILING specifically, which is the one error
# that would manufacture the result this pilot exists to measure.
#
# ent_coef 0.01 is the base the 2M selection-grid rerun will ship. It is NOT the
# base the archived logs/sel_* runs used (their metadata records
# policy_learning_kwargs {n_steps: 256} and no ent_coef), which is exactly why
# every control here is re-run rather than read off the archive.
#
# --preset generic on every cell: MuJoCoSuite.default_ppo_hyperparams applies a
# tuned rl-zoo block to Reacher-v5 whenever preset is "auto" (the default) and
# NOT gated behind --tuned-hyperparams. No Reacher cell is in this pilot, but
# passing `generic` explicitly is what makes "stock SB3" a checked claim rather
# than an assumption. --tuned-hyperparams is not passed anywhere.
#
# No RLHF flags at all: every arm is mode=true or mode=partial, zero labels.
# ---------------------------------------------------------------------------

if [ ! -f jobs/params_stab.txt ]; then
  echo "jobs/params_stab.txt missing; run: python jobs/make_params_stab.py" >&2
  exit 2
fi
if [ "$(wc -l < jobs/params_stab.txt)" -ne 140 ]; then
  echo "jobs/params_stab.txt must contain exactly 140 lines" >&2
  exit 2
fi

# --unhealthy-penalty is new; an old checkout would silently run the control
# config under a treatment log directory.
HELP_TEXT="$(python -m rcomp train --help)"
for FLAG in --unhealthy-penalty --policy-learning-kwargs --preset --final-policy --n-envs; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required flag: $FLAG -- git pull" >&2
    exit 2
  fi
done

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_stab.txt | tr -d '\r')"
read -r CELL TREATMENT ARM SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${TREATMENT:-}" ] || [ -z "${ARM:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# PER-CELL SETTINGS. EVALEP matches the selection grid: fewer episodes on the
# env that cannot terminate (only the initial state varies), more where the
# episode length is itself random. EvalCallback and ComponentEvalCallback each
# run EVALEP episodes at every eval point.
# ---------------------------------------------------------------------------
case "$CELL" in
  cheetah) ENV=HalfCheetah-v5; MODULE=sel_halfcheetah; EVALEP=5  ;;
  walker)  ENV=Walker2d-v5;    MODULE=sel_walker2d;    EVALEP=10 ;;
  ant)     ENV=Ant-v5;         MODULE=sel_ant;         EVALEP=10 ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

PPO_KWARGS='{n_steps:256,ent_coef:0.01}'
TREAT=()
case "$TREATMENT" in
  ctrl) ;;
  sde)
    # The ONLY change: independent per-step exploration noise becomes gSDE's
    # temporally correlated, state-dependent noise. sde_sample_freq stays at
    # SB3's default -1 (resample once per rollout = every n_steps=256 steps).
    # Nothing about the environment, the reward or the episode structure moves,
    # so evaluation is untouched and there is no train/eval mismatch.
    PPO_KWARGS='{n_steps:256,ent_coef:0.01,use_sde:true}' ;;
  noterm)
    # TRAINING ENV ONLY: no termination when unhealthy, and the boolean +1/0
    # healthy bonus becomes a persistent +1/-1. Every evaluation path still
    # scores the STANDARD environment, and the final policy is additionally
    # scored on the modified one as selected_policy_modified_env_reward_mean.
    if [ "$CELL" = "cheetah" ]; then
      echo "noterm is undefined on $ENV: it never terminates and has no is_healthy rule" >&2
      exit 2
    fi
    TREAT=(--unhealthy-penalty 1.0) ;;
  *) echo "unknown treatment: $TREATMENT" >&2; exit 2 ;;
esac

LOGDIR="logs/stab_${CELL}_${TREATMENT}_${ARM}"
RUNNAME="stab_${CELL}_${TREATMENT}_${ARM}_seed${SEED}"
# Guards only against re-running a FINISHED run. It does NOT stop two concurrent
# submissions writing into one directory -- check `squeue -u $USER` first.
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

MODE=(--mode true)
if [ "$ARM" != "true" ]; then
  PARTIAL="${MODULE}:${ARM}"
  # Resolve the prior before burning a queue slot, using the exact call the
  # trainer makes. NOT `rcomp validate-partial` (feeds a 4-element dummy obs)
  # and NOT `rcomp list-partials` (prints bare names even where the file:name
  # form is required).
  if ! python -c "
import sys
from rcomp.partials import PartialRegistry, load_partial_reference
try:
    spec = load_partial_reference('$PARTIAL', 'mujoco', PartialRegistry())
    spec.create('$ENV')
except Exception as exc:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    sys.exit(1)
"; then
    echo "partial '$PARTIAL' does not resolve/instantiate for $ENV" >&2
    exit 2
  fi
  MODE=(--mode partial --partial "$PARTIAL")
fi

srun python -m rcomp train \
  --suite mujoco --env-id "$ENV" --preset generic \
  "${MODE[@]}" "${TREAT[@]}" \
  --n-envs 8 --policy-learning-kwargs "$PPO_KWARGS" \
  --timesteps 1000000 --seed "$SEED" \
  --final-policy last \
  --eval-freq 20000 --n-eval-episodes "$EVALEP" --final-eval-episodes 30 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
