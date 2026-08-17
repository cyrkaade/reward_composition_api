#!/bin/bash
#SBATCH --job-name=sel2
#SBATCH --output=logs/slurm/sel2_%A_%a.out
#SBATCH --error=logs/slurm/sel2_%A_%a.err
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-280
#SBATCH --requeue
set -euo pipefail

# ============================================================================
# THE SELECTION GRID, RERUN with the PPO settings the entkl experiment picked.
#
#   python jobs/make_params_sel2.py
#   wc -l jobs/params_sel2.txt       # must print 280
#   mkdir -p logs/slurm
#   squeue -u $USER                  # nothing already writing to logs/sel2_*
#   sbatch --array=1-280 jobs/run_sel2.sh
#
# Three arms per environment, 5 seeds, 2M timesteps each:
#   true      ground-truth reward, 0 labels      -> the ceiling (black)
#   vanilla   preference RLHF, no prior, q350    -> the floor   (green)
#   <prior>   the hand-written prior alone       -> the candidate (red)
#
# Writes to logs/sel2_* . The original logs/sel_* are left alone so the two
# grids can be compared; analyze_sel.py --prefix sel2 reads this one.
#
# ============================================================================
# WHAT CHANGED FROM run_sel.sh, AND ONLY THIS
# ============================================================================
#
# 1. target_kl 0.03, EVERY cell and EVERY arm alike.
#    PPO stops its epoch loop once the approximate KL between the old and new
#    policy passes this. Measured by the entkl experiment (80 runs, 8 cells x 2
#    arms x 5 seeds at 2M, mode=true, against the archived sel_*_true control):
#
#      cell      ctrl final -> kl final   drawdown ctrl -> kl   final log_std
#      cheetah   1539 -> 3675   (5/5)     0.08 -> 0.03          +2.86 -> +0.06
#      walker     904 -> 3675   (5/5)     0.72 -> 0.23          +3.10 -> +0.89
#      standup  85735 -> 168227 (5/5)     0.48 -> 0.00         +11.18 -> +0.43
#      ll         246 -> 273    (4/5)     0.16 -> 0.06          n/a, discrete
#      reacher/pusher/swimmer: flat, within noise, no harm
#
#    log_std starts at 0.0 in every arm. On the old settings it ran away; with
#    target_kl it stays put even though ent_coef 0.01 is still on. So the
#    runaway was oversized policy updates, not the entropy bonus, which is why
#    ent_coef 0.01 is KEPT here. The other tested arm (ent_coef 0.01 -> 0) did
#    not fix it and hurt cheetah, pusher and ant.
#
# 2. Ant only: log_std_init -2 via policy_kwargs.
#    target_kl alone leaves Ant at +13 with its peak still at the FIRST eval
#    point and 98% of it given back -- it stops hurting itself without ever
#    learning. At the stock log_std_init 0 the expected control cost is
#    2.06/step (0.5 * E[sum a^2], 8 actuators, sigma=1) against a +1.00/step
#    alive bonus, so every step alive loses money from the first rollout, and
#    Ant-v5 charges NO terminal penalty, so ending the episode is the cheapest
#    way out. At sigma=0.135 the control cost is 0.07/step and the sign flips.
#    A local 2x2 probe at 300k: 676-724 final, 1279-1528 peak, ~760-step
#    episodes, against -14.5 and 8.5-step episodes on the old settings.
#
#    --policy-learning-kwargs REPLACES policy_kwargs wholesale (suites.py does a
#    top-level dict.update), so the MuJoCo suite's own net_arch and activation
#    MUST be repeated inside the block or Ant silently gets SB3's default 64x64
#    ReLU net while every other cell keeps 256x256 Tanh. Verified end to end:
#    the saved model reports log_std -1.999, net_arch {pi:[256,256],
#    vf:[256,256]}, activation Tanh, target_kl 0.03.
#
#    LunarLander is DISCRETE, so log_std does not exist there and no
#    policy_kwargs block is passed -- box2d stays pure SB3 including its 64x64
#    net, exactly as in run_sel.sh.
#
# Everything else below is run_sel.sh unchanged: n_steps 256 at n_envs 8,
# --preset generic on every MuJoCo cell, Swimmer's gamma 0.9999, the per-cell
# FRAGMENT / COLLECTION / EVALEP table, and the whole reward-model block.
# --tuned-hyperparams is NOT passed anywhere.
# ============================================================================

if [ ! -f jobs/params_sel2.txt ]; then
  echo "jobs/params_sel2.txt missing; run: python jobs/make_params_sel2.py" >&2
  exit 2
fi
if [ "$(wc -l < jobs/params_sel2.txt)" -ne 280 ]; then
  echo "jobs/params_sel2.txt must contain exactly 280 lines" >&2
  exit 2
fi

HELP_TEXT="$(python -m rcomp train --help)"
for FLAG in --policy-learning-kwargs --preset --ensemble-training --round0-data-protocol \
            --tanh-model-reward --dedicated-query-rng --ensemble-bootstrap \
            --reward-model-train-accuracy-stop --final-policy --n-envs \
            --round0-collection-timesteps --tanh-scale \
            --active-query-strategy --active-candidate-protocol; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required flag: $FLAG -- git pull" >&2
    exit 2
  fi
done

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_sel2.txt | tr -d '\r')"
read -r CELL VARIANT SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# PER-CELL SETTINGS -- unchanged from run_sel.sh.
#
# FRAGMENT is set against the MEASURED random-policy episode length, because
# round 0 collects with an untrained policy: LunarLander 104, Reacher exactly 50,
# Pusher exactly 100, Swimmer / HalfCheetah / HumanoidStandup exactly 1000
# (none of the three can terminate), Ant 76, Walker2d 18. fragment_trajectories
# cuts non-overlapping blocks and DISCARDS the remainder, so an episode shorter
# than the fragment yields nothing at all.
#
# COLLECTION is sized so EVERY round can buy its 70 pairs (140 fragments) from
# the worst case. ROUND 0 is overridden to 50000 everywhere.
#
# EVALEP: fewer episodes on the three environments that cannot terminate, where
# only the initial state varies; more where the episode length itself is random.
# ---------------------------------------------------------------------------
PARTIAL=""
PRESET=(--preset generic)
# JSON rather than the loose k:v form, because Ant needs a nested mapping and
# parse_key_value_mapping tries json.loads first. Both paths were smoke-tested.
PPO_KWARGS='{"n_steps":256,"ent_coef":0.01,"target_kl":0.03}'

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
    # Swimmer's reward arrives many steps after the stroke that earns it. Left at
    # stock gamma .99, PPO parks on the ~35-return local optimum and the
    # "ceiling" would be an artifact of the discount, not the task.
    SUITE=mujoco; ENV=Swimmer-v5;          MODULE=sel_swimmer
    FRAGMENT=50; COLLECTION=50000; EVALEP=5
    PPO_KWARGS='{"n_steps":256,"gamma":0.9999,"ent_coef":0.01,"target_kl":0.03}' ;;
  cheetah)
    SUITE=mujoco; ENV=HalfCheetah-v5;      MODULE=sel_halfcheetah
    FRAGMENT=50; COLLECTION=50000; EVALEP=5 ;;
  walker)
    SUITE=mujoco; ENV=Walker2d-v5;         MODULE=sel_walker2d
    FRAGMENT=10; COLLECTION=30000; EVALEP=10 ;;
  ant)
    # The policy_kwargs block repeats MuJoCoSuite's own net_arch and
    # activation_fn on purpose -- see the header. Dropping them would give Ant
    # SB3's default 64x64 ReLU net while every other cell runs 256x256 Tanh.
    SUITE=mujoco; ENV=Ant-v5;              MODULE=sel_ant
    FRAGMENT=25; COLLECTION=40000; EVALEP=10
    PPO_KWARGS='{"n_steps":256,"ent_coef":0.01,"target_kl":0.03,"policy_kwargs":{"log_std_init":-2,"activation_fn":"Tanh","net_arch":{"pi":[256,256],"vf":[256,256]}}}' ;;
  standup)
    SUITE=mujoco; ENV=HumanoidStandup-v5;  MODULE=sel_humanoidstandup
    FRAGMENT=50; COLLECTION=50000; EVALEP=5 ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

if [ "$VARIANT" != "true" ] && [ "$VARIANT" != "vanilla" ]; then
  PARTIAL="${MODULE}:${VARIANT}"
fi

LOGDIR="logs/sel2_${CELL}_${VARIANT}"
RUNNAME="sel2_${CELL}_${VARIANT}_seed${SEED}"
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
  --timesteps 2000000 --seed "$SEED"
  --final-policy last
  --eval-freq 20000 --n-eval-episodes "$EVALEP" --final-eval-episodes 30
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
  "${PRESET[@]}"
)

# The standardized (B-Pref-shaped) reward model -- identical to run_sel.sh.
RM=(
  --query-budget 350 --rlhf-rounds 5
  --collection-timesteps "$COLLECTION" --fragment-length "$FRAGMENT"
  --round0-collection-timesteps 50000
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

case "$VARIANT" in
  true)    ARGS+=(--mode true) ;;
  vanilla) ARGS+=("${RM[@]}" --mode feedback) ;;
  *)       ARGS+=(--mode partial --partial "$PARTIAL") ;;
esac

srun python -m rcomp train "${ARGS[@]}"
