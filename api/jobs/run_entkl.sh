#!/bin/bash
#SBATCH --job-name=entkl
#SBATCH --output=logs/slurm/entkl_%A_%a.out
#SBATCH --error=logs/slurm/entkl_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-80
#SBATCH --requeue
set -euo pipefail

# ============================================================================
# ENT/KL CEILING REPAIR: can the true arms be made stable again?
#
#   python jobs/make_params_entkl.py
#   wc -l jobs/params_entkl.txt      # must print 80
#   mkdir -p logs/slurm
#   squeue -u $USER                  # nothing already writing to logs/entkl_*
#   sbatch --array=1-80 jobs/run_entkl.sh
#
# mode=true ONLY, 8 cells x 2 arms x 5 seeds, 1M timesteps, zero labels.
#   ent0   ent_coef 0.01 -> 0
#   kl     ent_coef 0.01 kept, + target_kl 0.03
#
# THE CONTROL IS NOT RE-RUN. The first 1M of the existing 2M logs/sel_*_true
# runs is identical to a 1M run (constant lr and clip_range, single learn()
# call), and evaluations.npz index 49 is exactly 1000000. analyze_entkl.py
# reads it from there at matched seeds. See jobs/make_params_entkl.py.
#
# Everything below is copied from run_sel.sh's true-arm path unchanged: same
# suite, env, --preset generic on MuJoCo (Reacher's tuned rl-zoo block applies
# at preset=auto WITHOUT --tuned-hyperparams, so `generic` is what makes "stock
# SB3" true of all eight cells), n_envs 8, n_steps 256, --final-policy last,
# --eval-freq 20000, per-cell --n-eval-episodes, --final-eval-episodes 30.
# Swimmer keeps its gamma 0.9999 override. --tuned-hyperparams is NOT passed.
# The ONLY differences from that path are --timesteps (1M, not 2M) and the arm.
# ============================================================================

if [ ! -f jobs/params_entkl.txt ]; then
  echo "jobs/params_entkl.txt missing; run: python jobs/make_params_entkl.py" >&2
  exit 2
fi
if [ "$(wc -l < jobs/params_entkl.txt)" -ne 80 ]; then
  echo "jobs/params_entkl.txt must contain exactly 80 lines" >&2
  exit 2
fi

HELP_TEXT="$(python -m rcomp train --help)"
for FLAG in --policy-learning-kwargs --preset --final-policy --n-envs \
            --eval-freq --n-eval-episodes --final-eval-episodes; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required flag: $FLAG -- git pull" >&2
    exit 2
  fi
done

# HORIZON. Default 1M. Override at submit time with
#     ENTKL_TIMESTEPS=2000000 sbatch --array=1-80 jobs/run_entkl.sh
# (sbatch exports the submitting environment by default).
#
# READ THIS BEFORE CHOOSING. Walker2d and HumanoidStandup are still HEALTHY at
# 1M -- measured on the archived control: walker final 2120, drawdown 0.28,
# last-quarter trend +137 (still rising); standup final 140073, drawdown 0.11.
# Their collapse to 904 and 85735 happens in the SECOND million. So at 1M this
# experiment cannot test whether either arm prevents that collapse on the two
# cells that motivated it; it can only show the log_std trajectory and catch
# Ant, which is already broken by 1M. At 2M the archived control also needs no
# truncation -- it is the run as it stands.
TIMESTEPS="${ENTKL_TIMESTEPS:-1000000}"
if [[ ! "$TIMESTEPS" =~ ^[0-9]+$ ]]; then
  echo "ENTKL_TIMESTEPS must be an integer, got: $TIMESTEPS" >&2
  exit 2
fi

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_entkl.txt | tr -d '\r')"
read -r CELL ARM SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${ARM:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# PER-CELL SETTINGS. EVALEP matches run_sel.sh exactly: fewer episodes on the
# three environments that cannot terminate (only the initial state varies),
# more where the episode length itself is random.
# ---------------------------------------------------------------------------
PRESET=(--preset generic)
GAMMA=""

case "$CELL" in
  ll)
    SUITE=box2d;  ENV=LunarLander-v3;      EVALEP=10
    PRESET=() ;;                       # box2d has no preset knob; pure SB3 defaults
  reacher)
    SUITE=mujoco; ENV=Reacher-v5;          EVALEP=20 ;;
  pusher)
    SUITE=mujoco; ENV=Pusher-v5;           EVALEP=10 ;;
  swimmer)
    # Swimmer's reward arrives many steps after the stroke that earns it; left at
    # stock gamma .99 PPO parks on the ~35-return local optimum, so the ceiling
    # would be an artifact of the discount rather than the task. Same override
    # run_sel.sh applies.
    SUITE=mujoco; ENV=Swimmer-v5;          EVALEP=5
    GAMMA="gamma:0.9999," ;;
  cheetah)
    SUITE=mujoco; ENV=HalfCheetah-v5;      EVALEP=5 ;;
  walker)
    SUITE=mujoco; ENV=Walker2d-v5;         EVALEP=10 ;;
  ant)
    SUITE=mujoco; ENV=Ant-v5;              EVALEP=10 ;;
  standup)
    SUITE=mujoco; ENV=HumanoidStandup-v5;  EVALEP=5 ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

# ent_coef is written out explicitly in BOTH arms rather than relying on the
# suite default, so the metadata records the tested value either way.
case "$ARM" in
  ent0) PPO_KWARGS="{n_steps:256,${GAMMA}ent_coef:0}" ;;
  kl)   PPO_KWARGS="{n_steps:256,${GAMMA}ent_coef:0.01,target_kl:0.03}" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

LOGDIR="logs/entkl_${CELL}_${ARM}"
RUNNAME="entkl_${CELL}_${ARM}_seed${SEED}"
# Guards only against re-running a FINISHED run. It does NOT stop two concurrent
# submissions writing into one directory -- check `squeue -u $USER` first.
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

ARGS=(
  --suite "$SUITE" --env-id "$ENV"
  --mode true
  --n-envs 8 --policy-learning-kwargs "$PPO_KWARGS"
  --timesteps "$TIMESTEPS" --seed "$SEED"
  --final-policy last
  --eval-freq 20000 --n-eval-episodes "$EVALEP" --final-eval-episodes 30
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
  "${PRESET[@]}"
)

srun python -m rcomp train "${ARGS[@]}"
