#!/bin/bash
#SBATCH --job-name=wsum
#SBATCH --account=aalto_users
#SBATCH --output=logs/slurm/wsum_%A_%a.out
#SBATCH --error=logs/slurm/wsum_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-650
#SBATCH --requeue
set -euo pipefail

# Make the job self-contained: sbatch exports the submitting shell's PATH, which
# does not necessarily have the rcomp env activated.
export PATH="/scratch/work/akishea1/envs/rcomp/bin:${PATH}"

# Alpha sweep for the normalized weighted sum, against true / vanilla / naive.
# Environment blocks are copied verbatim from run_reasonable.sh so the partials
# chosen by that screen are evaluated under the configuration that screened them.

PARAMS_FILE="${PARAMS_FILE:-jobs/params_wsum.txt}"
# a bundling wrapper already holds the allocation and sets LAUNCHER=""
LAUNCHER="${LAUNCHER-srun}"
EXPECTED_ROWS="${EXPECTED_ROWS:-650}"
TIMESTEPS="${TIMESTEPS:-2000000}"
if [ ! -f "$PARAMS_FILE" ]; then
  echo "$PARAMS_FILE missing; run: python jobs/make_params_wsum.py" >&2
  exit 2
fi
if [ "$(wc -l < "$PARAMS_FILE")" -ne "$EXPECTED_ROWS" ]; then
  echo "$PARAMS_FILE must contain exactly $EXPECTED_ROWS lines" >&2
  exit 2
fi
if ! command -v python >/dev/null 2>&1; then
  echo "python not on PATH -- activate /scratch/work/akishea1/envs/rcomp" >&2
  exit 2
fi

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE" | tr -d '\r')"
read -r CELL ARM SEED PARTIAL BUDGET ALPHA <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${ARM:-}" ] || [ -z "${SEED:-}" ] || [ -z "${PARTIAL:-}" ] \
   || [ -z "${BUDGET:-}" ] || [ -z "${ALPHA:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

PRESET=(--preset generic)
PPO_KWARGS='{"n_steps":256,"ent_coef":0.01,"target_kl":0.03}'
case "$CELL" in
  ll)
    SUITE=box2d; ENV=LunarLander-v3
    FRAGMENT=25; COLLECTION=40000; EVALEP=10; PRESET=() ;;
  bipedal)
    SUITE=box2d; ENV=BipedalWalker-v3
    FRAGMENT=50; COLLECTION=30000; EVALEP=10; PRESET=() ;;
  ant)
    SUITE=mujoco; ENV=Ant-v5
    FRAGMENT=25; COLLECTION=40000; EVALEP=10
    PPO_KWARGS='{"n_steps":256,"ent_coef":0.01,"target_kl":0.03,"policy_kwargs":{"log_std_init":-2,"activation_fn":"Tanh","net_arch":{"pi":[256,256],"vf":[256,256]}}}' ;;
  reacher)
    SUITE=mujoco; ENV=Reacher-v5
    FRAGMENT=25; COLLECTION=20000; EVALEP=20 ;;
  hopper)
    SUITE=mujoco; ENV=Hopper-v5
    FRAGMENT=10; COLLECTION=30000; EVALEP=10 ;;
  pusher)
    SUITE=mujoco; ENV=Pusher-v5
    FRAGMENT=25; COLLECTION=20000; EVALEP=10 ;;
  swimmer)
    SUITE=mujoco; ENV=Swimmer-v5
    FRAGMENT=50; COLLECTION=50000; EVALEP=5
    PPO_KWARGS='{"n_steps":256,"gamma":0.9999,"ent_coef":0.01,"target_kl":0.03}' ;;
  walker)
    SUITE=mujoco; ENV=Walker2d-v5
    FRAGMENT=10; COLLECTION=30000; EVALEP=10 ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

LOGDIR="logs/wsum_${CELL}_${ARM}"
RUNNAME="wsum_${CELL}_${ARM}_seed${SEED}"
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

if [ "$PARTIAL" != "-" ]; then
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
    echo "partial '$PARTIAL' does not resolve/instantiate for $ENV" >&2
    exit 2
  fi
fi

ARGS=(
  --suite "$SUITE" --env-id "$ENV"
  --n-envs 8 --policy-learning-kwargs "$PPO_KWARGS" --device cpu
  --timesteps "$TIMESTEPS" --seed "$SEED"
  --final-policy last
  --eval-freq 20000 --n-eval-episodes "$EVALEP" --final-eval-episodes 30
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
  "${PRESET[@]}"
)

# Identical reward-model block to run_reasonable.sh; only the budget varies.
RM=(
  --query-budget "$BUDGET" --rlhf-rounds 5
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
  --tanh-model-reward --tanh-scale 5
  --active-learning --active-query-strategy ensemble --active-candidate-protocol pool
)

case "$ARM" in
  true)
    ARGS+=(--mode true) ;;
  vanilla_*)
    ARGS+=("${RM[@]}" --mode feedback) ;;
  naive_*)
    ARGS+=("${RM[@]}" --mode naive --partial "$PARTIAL") ;;
  ws*)
    ARGS+=("${RM[@]}" --mode weighted_sum --partial "$PARTIAL"
            --partial-alpha "$ALPHA"
            --normalize-partial-reward --normalize-model-reward) ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

${LAUNCHER} python -m rcomp train "${ARGS[@]}"
