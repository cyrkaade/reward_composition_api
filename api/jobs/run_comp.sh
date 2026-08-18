#!/bin/bash
#SBATCH --job-name=comp
#SBATCH --output=logs/slurm/comp_%A_%a.out
#SBATCH --error=logs/slurm/comp_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-960
#SBATCH --requeue
set -euo pipefail

# ============================================================================
# THE COMPOSITION GRID: vanilla vs naive vs weighted_sum, 8 cells, 10 seeds.
#
#   module load mamba && source activate /scratch/work/akishea1/envs/rcomp
#   python jobs/make_params_comp.py
#   wc -l jobs/params_comp.txt        # must print 960
#   mkdir -p logs/slurm
#   squeue -u $USER                   # nothing already writing to logs/comp_*
#   sbatch --array=1-960 jobs/run_comp.sh
#
# 12 variants per cell: true, partial, and then vanilla / naive / ws05 / ws10 /
# ws15 at each of two label budgets. Writes to logs/comp_*.
#
# naive       --mode naive, alpha 1.0, no normalization         = partial + RM
# ws<alpha>   --mode naive + --normalize-partial-reward
#                          + --normalize-model-reward
#                          + --partial-alpha <alpha>  = norm(partial)*a + norm(RM)
#
# so naive vs ws10 is a single-variable contrast (normalization on or off at the
# same weight), and ws05/ws10/ws15 is the mixing-ratio sweep.
#
# ATARI RUNS 10x LONGER AND BUYS 8x THE LABELS.
#   mujoco/box2d   2M steps,  350 / 700  queries, eval every 20000
#   atari         20M steps, 2800 / 5600 queries, eval every 200000
# eval_freq is scaled so both give 100 evaluation points; leaving it at 20000 on
# a 20M run would produce 1000 eval points and spend about half the job
# evaluating.
#
# target_kl 0.03 on every cell and every arm.
# --tuned-hyperparams is NOT passed anywhere.
# ============================================================================

if [ ! -f jobs/params_comp.txt ]; then
  echo "jobs/params_comp.txt missing; run: python jobs/make_params_comp.py" >&2
  exit 2
fi
if [ "$(wc -l < jobs/params_comp.txt)" -ne 960 ]; then
  echo "jobs/params_comp.txt must contain exactly 960 lines" >&2
  exit 2
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not on PATH -- activate the environment before submitting:" >&2
  echo "  module load mamba && source activate /scratch/work/akishea1/envs/rcomp" >&2
  exit 2
fi

HELP_TEXT="$(python -m rcomp train --help)"
for FLAG in --policy-learning-kwargs --preset --ensemble-training --round0-data-protocol \
            --tanh-model-reward --dedicated-query-rng --ensemble-bootstrap \
            --reward-model-train-accuracy-stop --final-policy --n-envs \
            --round0-collection-timesteps --tanh-scale --partial-alpha \
            --normalize-partial-reward --normalize-model-reward \
            --active-query-strategy --active-candidate-protocol; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required flag: $FLAG -- git pull" >&2
    exit 2
  fi
done

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_comp.txt | tr -d '\r')"
read -r CELL VARIANT SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# PER-CELL SETTINGS.
#
# FRAGMENT is set against the MEASURED random-policy episode length, because
# round 0 collects with an untrained policy. fragment_trajectories cuts
# non-overlapping blocks and DISCARDS the remainder, so an episode shorter than
# FRAGMENT yields nothing at all.
#
# COLLECTION is sized so the HIGH budget can be delivered. 5600 queries over 5
# rounds is 1120 pairs = 2240 fragments a round, which is why the Atari cells
# collect 250000 rather than the 50000 the earlier grids used. Qbert is the
# binding case: 250000 / 312 steps = ~801 episodes x 4 fragments = ~3205,
# against the 2240 needed. Pong: 250000 / 910 = ~275 episodes x 14 = ~3846.
# Non-Atari at 700 queries needs only 280 fragments a round, which every cell
# clears by a wide margin.
#
# ROUND0 is the round-0 override. The earlier grids forced 50000 because their
# collections were smaller than that; here the Atari cells collect far more, so
# forcing 50000 would SHRINK round 0. It is therefore set per cell.
# ---------------------------------------------------------------------------
PRESET=(--preset generic)
STEPS=2000000
EVALFREQ=20000
BUDGET_LOW=350
BUDGET_HIGH=700
ROUND0=50000
GAMMA=""

case "$CELL" in
  ll)
    SUITE=box2d;  ENV=LunarLander-v3;    PARTIAL=sel_lunarlander:sll_pad_speed
    FRAGMENT=25; COLLECTION=40000; EVALEP=10
    PRESET=() ;;                       # box2d has no preset knob; pure SB3
  pusher)
    SUITE=mujoco; ENV=Pusher-v5;         PARTIAL=sel_pusher:spsh_goal
    FRAGMENT=25; COLLECTION=20000; EVALEP=10 ;;
  reacher)
    SUITE=mujoco; ENV=Reacher-v5;        PARTIAL=sel_reacher:srch_dist
    FRAGMENT=25; COLLECTION=20000; EVALEP=20 ;;
  swimmer)
    # Swimmer's reward arrives many steps after the stroke that earns it. Left
    # at stock gamma .99, PPO parks on the ~35-return local optimum and the
    # ceiling would be an artifact of the discount rather than the task.
    SUITE=mujoco; ENV=Swimmer-v5;        PARTIAL=sel_swimmer:sswm_straight
    FRAGMENT=50; COLLECTION=50000; EVALEP=5
    GAMMA="gamma:0.9999," ;;
  hopper)
    SUITE=mujoco; ENV=Hopper-v5;         PARTIAL=hopper_levels:hopper_p75
    FRAGMENT=10; COLLECTION=30000; EVALEP=10 ;;
  bipedal)
    SUITE=box2d;  ENV=BipedalWalker-v3;  PARTIAL=sel_bipedal:sbw_target
    FRAGMENT=50; COLLECTION=30000; EVALEP=10
    PRESET=() ;;
  qbert)
    SUITE=atari;  ENV=ALE/Qbert-v5;      PARTIAL=sel_qbert:sqb_life_heavy
    FRAGMENT=64; COLLECTION=250000; EVALEP=10
    STEPS=20000000; EVALFREQ=200000; BUDGET_LOW=2800; BUDGET_HIGH=5600
    ROUND0=250000
    PRESET=() ;;                       # AtariSuite supplies its own block
  pong)
    SUITE=atari;  ENV=ALE/Pong-v5;       PARTIAL=sel_pong:spg_rally
    FRAGMENT=64; COLLECTION=250000; EVALEP=10
    STEPS=20000000; EVALFREQ=200000; BUDGET_LOW=2800; BUDGET_HIGH=5600
    ROUND0=250000
    PRESET=() ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

# PPO. The Atari suite's own block (n_steps 128, batch 256, lr 2.5e-4, clip 0.1,
# n_epochs 4) is Atari-specific tuning and is left alone; n_steps 256 is forced
# only on the MuJoCo/Box2D cells, where it restores SB3's 2048-sample rollout at
# n_envs 8.
if [ "$SUITE" = "atari" ]; then
  PPO_KWARGS="{ent_coef:0.01,target_kl:0.03}"
else
  PPO_KWARGS="{n_steps:256,${GAMMA}ent_coef:0.01,target_kl:0.03}"
fi

# Decode the variant into method + budget.
BUDGET=""
METHOD="$VARIANT"
case "$VARIANT" in
  *_low)  METHOD="${VARIANT%_low}";  BUDGET="$BUDGET_LOW" ;;
  *_high) METHOD="${VARIANT%_high}"; BUDGET="$BUDGET_HIGH" ;;
esac

LOGDIR="logs/comp_${CELL}_${VARIANT}"
RUNNAME="comp_${CELL}_${VARIANT}_seed${SEED}"
# Guards only against re-running a FINISHED run. It does NOT stop two concurrent
# submissions writing into one directory -- check `squeue -u $USER` first.
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

# Resolve the prior before burning a queue slot, using the exact call the
# trainer makes. NOT `rcomp validate-partial` (feeds a 4-element dummy obs) and
# NOT `rcomp list-partials` (prints bare names even where file:name is needed).
if [ "$METHOD" != "vanilla" ]; then
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
  --timesteps "$STEPS" --seed "$SEED"
  --final-policy last
  --eval-freq "$EVALFREQ" --n-eval-episodes "$EVALEP" --final-eval-episodes 30
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
  "${PRESET[@]}"
)

# The standardized (B-Pref-shaped) reward model -- identical to run_sel2.sh
# apart from budget, collection and round-0 size, which are per cell.
RM=(
  --query-budget "$BUDGET" --rlhf-rounds 5
  --collection-timesteps "$COLLECTION" --fragment-length "$FRAGMENT"
  --round0-collection-timesteps "$ROUND0"
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
  --tanh-scale 5
  --active-learning
  --active-query-strategy ensemble
  --active-candidate-protocol pool
)

case "$METHOD" in
  true)    ARGS+=(--mode true) ;;
  partial) ARGS+=(--mode partial --partial "$PARTIAL") ;;
  vanilla) ARGS+=("${RM[@]}" --mode feedback) ;;
  naive)   ARGS+=("${RM[@]}" --mode naive --partial "$PARTIAL" --partial-alpha 1.0) ;;
  ws05)    ARGS+=("${RM[@]}" --mode naive --partial "$PARTIAL" --partial-alpha 0.5 \
                  --normalize-partial-reward --normalize-model-reward) ;;
  ws10)    ARGS+=("${RM[@]}" --mode naive --partial "$PARTIAL" --partial-alpha 1.0 \
                  --normalize-partial-reward --normalize-model-reward) ;;
  ws15)    ARGS+=("${RM[@]}" --mode naive --partial "$PARTIAL" --partial-alpha 1.5 \
                  --normalize-partial-reward --normalize-model-reward) ;;
  *) echo "unknown method: $METHOD (from variant $VARIANT)" >&2; exit 2 ;;
esac

srun python -m rcomp train "${ARGS[@]}"
