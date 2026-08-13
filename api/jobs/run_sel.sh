#!/bin/bash
#SBATCH --job-name=sel
#SBATCH --output=logs/slurm/sel_%A_%a.out
#SBATCH --error=logs/slurm/sel_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-280
#SBATCH --requeue
set -euo pipefail

# ============================================================================
# THE SELECTION GRID: which 8 environments, and which prior on each.
#
#   python jobs/make_params_sel.py
#   wc -l jobs/params_sel.txt        # must print 280
#   mkdir -p logs/slurm
#   sbatch --array=1-280 jobs/run_sel.sh
#
# Three arms per environment, 5 seeds, 1M timesteps each:
#   true      ground-truth reward, 0 labels      -> the ceiling (black)
#   vanilla   preference RLHF, no prior, q350    -> the floor   (green)
#   <prior>   the hand-written prior alone       -> the candidate (red)
# A prior qualifies when its curve sits clearly under the ceiling and at or
# above the floor. Nothing here composes a prior with a reward model.
#
# See jobs/make_params_sel.py for why five priors per environment and why
# rescaling a prior is not one of the five.
# ============================================================================

# ---------------------------------------------------------------------------
# PPO: STOCK SB3, WITH TWO DOCUMENTED OVERRIDES APPLIED TO EVERY ARM ALIKE.
#
# 1. n_steps 2048 -> 256, with n_envs 8.
#    SB3's default n_steps=2048 assumes ONE env, i.e. a 2048-sample rollout and
#    488 PPO iterations per million steps. At n_envs=8 the same setting makes a
#    16384-sample rollout and only 61 iterations per million - and this grid runs
#    1M, not the 2-3M the archive used. 61 policy updates is not a converged run
#    on any of these environments, and under-training the CEILING is the one
#    error that would silently manufacture the result we are looking for.
#    n_steps=256 x n_envs=8 restores exactly the SB3 default rollout of 2048
#    samples and 488 iterations, while keeping the 8-way wall-clock speedup.
#    Ignoring the reference n_envs is what pinned Hopper at its survive-only
#    optimum in an earlier grid; this is that lesson applied in the other
#    direction.
#
# 2. --preset generic on every MuJoCo cell.
#    MuJoCoSuite.default_ppo_hyperparams applies an Optuna-tuned rl-zoo block to
#    Reacher-v5 whenever preset is "auto", which is the DEFAULT - it is not gated
#    behind --tuned-hyperparams. Verified by diffing Suite.ppo_hyperparams:
#    preset auto gives Reacher gamma 0.9, lr 1.04e-4, n_steps 512, batch 32,
#    n_epochs 5, gae 1.0, clip 0.3; preset generic gives the stock 0.99 / 3e-4 /
#    2048 / 64 / 10 / 0.95 / 0.2 that every other MuJoCo cell gets. Passing
#    `generic` explicitly is what makes "stock SB3" true of all eight cells.
#    (The archived Reacher runs all used the tuned block without saying so.)
#
# --tuned-hyperparams is NOT passed anywhere. Swimmer additionally overrides
# gamma, for the reason given in its case block below.
# ---------------------------------------------------------------------------

if [ ! -f jobs/params_sel.txt ]; then
  echo "jobs/params_sel.txt missing; run: python jobs/make_params_sel.py" >&2
  exit 2
fi
if [ "$(wc -l < jobs/params_sel.txt)" -ne 280 ]; then
  echo "jobs/params_sel.txt must contain exactly 280 lines" >&2
  exit 2
fi

HELP_TEXT="$(python -m rcomp train --help)"
for FLAG in --policy-learning-kwargs --preset --ensemble-training --round0-data-protocol \
            --tanh-model-reward --dedicated-query-rng --ensemble-bootstrap \
            --reward-model-train-accuracy-stop --final-policy --n-envs; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required flag: $FLAG -- git pull" >&2
    exit 2
  fi
done

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_sel.txt | tr -d '\r')"
read -r CELL VARIANT SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# PER-CELL SETTINGS.
#
# FRAGMENT is set against the MEASURED random-policy episode length, because
# round 0 collects with an untrained policy: LunarLander 104, Reacher exactly 50,
# Pusher exactly 100, Swimmer / HalfCheetah / HumanoidStandup exactly 1000
# (none of the three can terminate), Ant 76, Walker2d 18. fragment_trajectories
# cuts non-overlapping blocks and DISCARDS the remainder, so an episode shorter
# than the fragment yields nothing at all - that is how 16 Walker2d cells in PRE4
# starved to 2-3 pairs a round on fragment 50. Walker2d therefore gets 10.
#
# COLLECTION is sized so round 0 can still buy its 70 pairs (140 fragments) from
# the worst case. With --round0-data-protocol separate, round 0 collects two
# independent streams of COLLECTION steps and queries only the second.
#   ll        40000 / 104 = 384 eps x 4 frags  = ~1500
#   reacher   20000 /  50 = 400 eps x 2        =   800
#   pusher    20000 / 100 = 200 eps x 4        =   800
#   walker    30000 /  18 = 1666 eps x 1       =  1666
#   ant       40000 /  76 = 526 eps x 3        =  1578
#   swimmer / cheetah / standup  50000 / 1000 x 20 = 1000
#
# EVALEP: fewer episodes on the three environments that cannot terminate, where
# only the initial state varies; more where the episode length itself is random.
# EvalCallback and ComponentEvalCallback each run EVALEP episodes at every
# eval point, so this is the dominant cost on the 1000-step environments.
# ---------------------------------------------------------------------------
PARTIAL=""
PPO_KWARGS='{n_steps:256}'
PRESET=(--preset generic)

case "$CELL" in
  ll)
    SUITE=box2d;  ENV=LunarLander-v3;      MODULE=sel_lunarlander
    FRAGMENT=25; COLLECTION=40000; EVALEP=10
    PRESET=() ;;                       # box2d has no preset knob; pure SB3 defaults
  reacher)
    SUITE=mujoco; ENV=Reacher-v5;          MODULE=sel_reacher
    FRAGMENT=25; COLLECTION=20000; EVALEP=20 ;;
  pusher)
    SUITE=mujoco; ENV=Pusher-v5;           MODULE=sel_pusher
    FRAGMENT=25; COLLECTION=20000; EVALEP=10 ;;
  swimmer)
    # Swimmer's reward arrives many steps after the stroke that earns it and the
    # rl-zoo block overrides gamma .99 -> .9999 for exactly that reason. Left at
    # stock .99, PPO parks on the ~35-return local optimum and the "ceiling" this
    # grid is measuring would be an artifact of the discount, not the task.
    SUITE=mujoco; ENV=Swimmer-v5;          MODULE=sel_swimmer
    FRAGMENT=50; COLLECTION=50000; EVALEP=5
    PPO_KWARGS='{n_steps:256,gamma:0.9999}' ;;
  cheetah)
    SUITE=mujoco; ENV=HalfCheetah-v5;      MODULE=sel_halfcheetah
    FRAGMENT=50; COLLECTION=50000; EVALEP=5 ;;
  walker)
    SUITE=mujoco; ENV=Walker2d-v5;         MODULE=sel_walker2d
    FRAGMENT=10; COLLECTION=30000; EVALEP=10 ;;
  ant)
    SUITE=mujoco; ENV=Ant-v5;              MODULE=sel_ant
    FRAGMENT=25; COLLECTION=40000; EVALEP=10 ;;
  standup)
    SUITE=mujoco; ENV=HumanoidStandup-v5;  MODULE=sel_humanoidstandup
    FRAGMENT=50; COLLECTION=50000; EVALEP=5 ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

if [ "$VARIANT" != "true" ] && [ "$VARIANT" != "vanilla" ]; then
  PARTIAL="${MODULE}:${VARIANT}"
fi

LOGDIR="logs/sel_${CELL}_${VARIANT}"
RUNNAME="sel_${CELL}_${VARIANT}_seed${SEED}"
# Guards only against re-running a FINISHED run. It does NOT stop two concurrent
# submissions writing into one directory -- check `squeue -u $USER` first.
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

# Resolve the prior before burning a queue slot, using the exact call the trainer
# makes. NOT `rcomp validate-partial` (feeds a 4-element dummy obs and fails on
# every 8-dim LunarLander partial) and NOT `rcomp list-partials` (prints bare
# names even for partials that need the file:name form).
if [ -n "$PARTIAL" ]; then
  if ! python -c "
import sys
from rcomp.partials import PartialRegistry, load_partial_reference
try:
    spec = load_partial_reference('$PARTIAL', '$SUITE', PartialRegistry())
    spec.create('$ENV')
except Exception as exc:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    sys.exit(1)
"; then
    echo "partial '$PARTIAL' does not resolve/instantiate for $ENV in suite $SUITE" >&2
    exit 2
  fi
fi

ARGS=(
  --suite "$SUITE" --env-id "$ENV"
  --n-envs 8 --policy-learning-kwargs "$PPO_KWARGS"
  --timesteps 1000000 --seed "$SEED"
  --final-policy last
  --eval-freq 20000 --n-eval-episodes "$EVALEP" --final-eval-episodes 30
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
  "${PRESET[@]}"
)

# The standardized (B-Pref-shaped) reward model, identical to the final grid
# minus --holdout-pairs. The holdout is an oracle diagnostic for composition
# experiments; here the vanilla arm is only a reference line and the fragments
# are better spent on queries.
RM=(
  --query-budget 350 --rlhf-rounds 5
  --collection-timesteps "$COLLECTION" --fragment-length "$FRAGMENT"
  --reward-hidden-sizes 256,256,256
  --reward-model-ensemble-size 3
  --reward-model-lr 0.0003
  --reward-model-batch-size 32
  --reward-model-loss-reduction mean
  --reward-model-l1 0
  --reward-output-l1 0.001
  --ensemble-training full
  --ensemble-bootstrap
  --reward-model-train-accuracy-stop 0.97
  --round0-data-protocol separate
  --dedicated-query-rng
  --tanh-model-reward
  --no-active-learning
)

case "$VARIANT" in
  true)    ARGS+=(--mode true) ;;
  vanilla) ARGS+=("${RM[@]}" --mode feedback) ;;
  *)       ARGS+=(--mode partial --partial "$PARTIAL") ;;
esac

srun python -m rcomp train "${ARGS[@]}"
