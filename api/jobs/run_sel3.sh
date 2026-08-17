#!/bin/bash
#SBATCH --job-name=sel3
#SBATCH --output=logs/slurm/sel3_%A_%a.out
#SBATCH --error=logs/slurm/sel3_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-105
#SBATCH --requeue
set -euo pipefail

# ============================================================================
# SELECTION SCREEN, THREE NEW CELLS: BipedalWalker-v3, MsPacman, Qbert.
#
#   module load mamba && source activate /scratch/work/akishea1/envs/rcomp
#   python jobs/make_params_sel3.py
#   wc -l jobs/params_sel3.txt        # must print 105
#   mkdir -p logs/slurm
#   squeue -u $USER                   # nothing already writing to logs/sel3_*
#   sbatch --array=1-105 jobs/run_sel3.sh
#
#   python jobs/analyze_sel.py --prefix sel3
#
# Three arms per cell, 5 seeds, 2M timesteps, q350: true / vanilla / 5 priors.
# Writes to logs/sel3_* ; the sel_* and sel2_* grids are untouched.
#
# target_kl 0.03 IS applied here, on every cell and arm, matching sel2 and
# every experiment from here on.
#
# ---------------------------------------------------------------------------
# PPO PER SUITE -- the two new suites are NOT MuJoCo and do not take the same
# treatment, so this is spelled out rather than copied.
#
# box2d: no preset knob (Box2DSuite inherits the base defaults), so BipedalWalker
#   runs pure SB3 -- n_steps 256 at n_envs 8, batch 64, gamma .99, lr 3e-4,
#   clip .2, n_epochs 10, 64x64 net -- exactly as LunarLander does in run_sel.sh.
#
# atari: AtariSuite.default_ppo_hyperparams supplies its OWN block and does not
#   branch on --preset, so passing one would be inert. Verified by resolving
#   Suite.ppo_hyperparams: n_steps 128, batch 256, lr 2.5e-4, clip 0.1,
#   n_epochs 4, ent_coef 0.01, 256x256 ReLU. n_steps is left at the suite's 128
#   (x8 envs = 1024 per rollout) rather than forced to 256, because that block is
#   the Atari-specific tuning and is not the thing under test here. ent_coef 0.01
#   is passed explicitly even though the block already sets it, so metadata
#   records the tested value.
#
# Atari envs are built by the suite with obs_type="ram" (128 bytes), frameskip 4
# and repeat_action_probability 0.25. The priors read `true_reward`,
# info["lives"] and info["episode_frame_number"] -- never the RAM bytes, which
# have no documented meaning. See partials/sel_mspacman.py.
#
# AtariSuite sets default_active_learning = False; --active-learning below
# overrides that, so all three cells use the same query selection as the rest
# of the grid.
# ---------------------------------------------------------------------------
#
# KNOWN RISK: 2M is SHORT for Atari (this project's own convention is 25M). The
# true arm may not separate from a random policy, which would fail the ceiling
# premise for those two cells. Random-policy reference, measured on the
# installed build: MsPacman ~190, Qbert ~200. Check the true-arm column against
# those before reading any prior contrast.
# ============================================================================

if [ ! -f jobs/params_sel3.txt ]; then
  echo "jobs/params_sel3.txt missing; run: python jobs/make_params_sel3.py" >&2
  exit 2
fi
if [ "$(wc -l < jobs/params_sel3.txt)" -ne 105 ]; then
  echo "jobs/params_sel3.txt must contain exactly 105 lines" >&2
  exit 2
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not on PATH -- activate the environment before submitting:" >&2
  echo "  module load mamba && source activate /scratch/work/akishea1/envs/rcomp" >&2
  exit 2
fi

HELP_TEXT="$(python -m rcomp train --help)"
for FLAG in --policy-learning-kwargs --ensemble-training --round0-data-protocol \
            --tanh-model-reward --dedicated-query-rng --ensemble-bootstrap \
            --reward-model-train-accuracy-stop --final-policy --n-envs \
            --round0-collection-timesteps --tanh-scale \
            --active-query-strategy --active-candidate-protocol; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required flag: $FLAG -- git pull" >&2
    exit 2
  fi
done

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_sel3.txt | tr -d '\r')"
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
#   bipedal   median 856 steps (min 50) -> fragment 50 gives ~17 per episode;
#             collection 30000 = ~35 episodes = ~595 fragments a round
#   mspacman  median 449 steps -> fragment 64 gives 7 per episode;
#             collection 50000 = ~111 episodes = ~777 fragments a round
#   qbert     median 312 steps -> fragment 64 gives 4 per episode;
#             collection 50000 = ~160 episodes = ~640 fragments a round
#
# Every round needs 140 fragments to buy its 70 pairs, so all three clear it
# comfortably. ROUND 0 is overridden to 50000 everywhere, as in run_sel.sh.
# ---------------------------------------------------------------------------
case "$CELL" in
  bipedal)
    SUITE=box2d;  ENV=BipedalWalker-v3;  MODULE=sel_bipedal
    FRAGMENT=50; COLLECTION=30000; EVALEP=10
    PPO_KWARGS='{n_steps:256,ent_coef:0.01,target_kl:0.03}' ;;
  mspacman)
    SUITE=atari;  ENV=ALE/MsPacman-v5;   MODULE=sel_mspacman
    FRAGMENT=64; COLLECTION=50000; EVALEP=10
    PPO_KWARGS='{ent_coef:0.01,target_kl:0.03}' ;;
  qbert)
    SUITE=atari;  ENV=ALE/Qbert-v5;      MODULE=sel_qbert
    FRAGMENT=64; COLLECTION=50000; EVALEP=10
    PPO_KWARGS='{ent_coef:0.01,target_kl:0.03}' ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

PARTIAL=""
if [ "$VARIANT" != "true" ] && [ "$VARIANT" != "vanilla" ]; then
  PARTIAL="${MODULE}:${VARIANT}"
fi

LOGDIR="logs/sel3_${CELL}_${VARIANT}"
RUNNAME="sel3_${CELL}_${VARIANT}_seed${SEED}"
# Guards only against re-running a FINISHED run. It does NOT stop two concurrent
# submissions writing into one directory -- check `squeue -u $USER` first.
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

# Resolve the prior before burning a queue slot, using the exact call the
# trainer makes. NOT `rcomp validate-partial` and NOT `rcomp list-partials`.
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
)

# The standardized (B-Pref-shaped) reward model -- identical to run_sel2.sh.
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
