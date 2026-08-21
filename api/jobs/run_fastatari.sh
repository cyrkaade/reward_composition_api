#!/bin/bash
#SBATCH --job-name=fastatari
#SBATCH --output=logs/slurm/fastatari_%A_%a.out
#SBATCH --error=logs/slurm/fastatari_%A_%a.err
#SBATCH --account=ellis_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --array=1-21
#SBATCH --requeue
set -euo pipefail

# Fast pixel-Atari pilot: 1 seed, 1M policy timesteps, three arms per game
# (true / zero-label RLHF / five hand-written RAM partials alone).
#
# Differences from jobs/run_reasonable_atari.sh, all deliberate and all about
# wall-clock cost -- see jobs/bench_atari.sh for the measurements behind them:
#   * 1M timesteps instead of 2M, 1 seed instead of 5, 5 partials instead of 8.
#   * eval-freq 50000 instead of 20000, keeping 20 curve points at 1M.  The Atari suite
#     sets record_stochastic_evaluation, so EVERY eval point runs three separate
#     single-env rollouts (deterministic, stochastic, component).  At 20k that
#     was ~19.5k single-env eval steps per 20k training steps.
#   * n-eval-episodes 5 instead of 10.
#   * Thread count left at the Slurm default (one per allocated CPU), which
#     measured faster than OMP_NUM_THREADS=1 despite the reward model being
#     called at batch size 1 once per env per step.  OMP=<n> overrides it.
#   * ellis_users account: fairshare factor 0.50 against aalto_users' 0.0056,
#     which is the difference between starting now and starting in three days.

PARAMS_FILE="${PARAMS_FILE:-jobs/params_fastatari.txt}"
EXPECTED_ROWS="${EXPECTED_ROWS:-21}"
# The vanilla arm is the zero-prior FLOOR of this screen, so starving it of
# labels would flatter every partial.  1400 pairs of 25-step clips is four
# times the reasonable grid's 350 and still fits: preferences.py builds the
# whole training tensor before the epoch loop (rated_pairs_to_tensors, ~L522),
# at pairs * 2 * fragment * 28225 * 4 bytes.  1400x25 = 7.9 GB; the 5600 pairs
# that Christiano et al. used on Atari would be 31.6 GB at 25 and 81 GB at 64.
BUDGET="${BUDGET:-1400}"
FRAGMENT="${FRAGMENT:-25}"
TIMESTEPS="${TIMESTEPS:-1000000}"
# Torch defaults to one thread per allocated CPU, which measured fastest here;
# OMP=<n> overrides it for comparison (OMP=1 came out ~20% slower on vanilla).
if [ -n "${OMP:-}" ]; then
  export OMP_NUM_THREADS="$OMP"
  export MKL_NUM_THREADS="$OMP"
fi

if [ ! -f "$PARAMS_FILE" ]; then
  echo "$PARAMS_FILE missing; run: python jobs/make_params_fastatari.py" >&2
  exit 2
fi
if [ "$(wc -l < "$PARAMS_FILE")" -ne "$EXPECTED_ROWS" ]; then
  echo "$PARAMS_FILE must contain exactly $EXPECTED_ROWS lines" >&2
  exit 2
fi

PY="${PY:-/scratch/work/akishea1/envs/rcomp/bin/python}"
if [ ! -x "$PY" ]; then
  echo "python not found at $PY -- set PY=... or activate /scratch/work/akishea1/envs/rcomp" >&2
  exit 2
fi

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE" | tr -d '\r')"
read -r CELL VARIANT SEED PARTIAL <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${SEED:-}" ] || [ -z "${PARTIAL:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

case "$CELL" in
  mspacman) ENV=ALE/MsPacman-v5 ;;
  qbert)    ENV=ALE/Qbert-v5 ;;
  pong)     ENV=ALE/Pong-v5 ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

case "$VARIANT" in
  true|vanilla)
    [ "$PARTIAL" = "-" ] || { echo "$VARIANT row unexpectedly carries a partial" >&2; exit 2; } ;;
  *)
    [[ "$PARTIAL" == reasonable_atari_partials:* ]] || {
      echo "candidate must come from reasonable_atari_partials, got: $PARTIAL" >&2
      exit 2
    } ;;
esac

LOGDIR="logs/fastatari_${CELL}_${VARIANT}"
RUNNAME="fastatari_${CELL}_${VARIANT}_seed${SEED}"
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

if [ "$PARTIAL" != "-" ]; then
  if ! $PY -c "
import sys
from rcomp.partials import PartialRegistry, load_partial_reference
try:
    spec = load_partial_reference('$PARTIAL', 'atari', PartialRegistry())
    spec.create('$ENV')
except Exception as exc:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    sys.exit(1)
"; then
    echo "partial '$PARTIAL' does not resolve/instantiate for $ENV" >&2
    exit 2
  fi
fi

ARGS=(
  --suite atari --env-id "$ENV"
  --n-envs 8 --policy-learning-kwargs '{"n_steps":128,"ent_coef":0.01,"target_kl":0.03}'
  --timesteps "$TIMESTEPS" --seed "$SEED"
  --final-policy last
  --eval-freq 50000 --n-eval-episodes 5 --final-eval-episodes 20
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
)

RM=(
  --query-budget "$BUDGET" --rlhf-rounds 5
  --collection-timesteps 50000 --fragment-length "$FRAGMENT"
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
  --holdout-pairs 100
  --reward-model-train-accuracy-stop 0.97
  --round0-data-protocol separate
  --dedicated-query-rng
  --tanh-model-reward --tanh-scale 5
  --active-learning --active-query-strategy ensemble --active-candidate-protocol pool
)

case "$VARIANT" in
  true) ARGS+=(--mode true) ;;
  vanilla) ARGS+=("${RM[@]}" --mode feedback) ;;
  *) ARGS+=(--mode partial --partial "$PARTIAL") ;;
esac

echo "task=${SLURM_ARRAY_TASK_ID} cell=$CELL variant=$VARIANT seed=$SEED omp=${OMP_NUM_THREADS:-default} on $(hostname)"
srun $PY -m rcomp train "${ARGS[@]}"
