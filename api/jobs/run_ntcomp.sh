#!/bin/bash
#SBATCH --job-name=ntcomp
#SBATCH --output=logs/slurm/ntcomp_%A_%a.out
#SBATCH --error=logs/slurm/ntcomp_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-1360
#SBATCH --requeue
set -euo pipefail

# Confirmatory grid.  jobs/params_ntcomp.txt can only be generated from a
# complete, frozen ntscreen selection.  Seeds 10..19 are independent of the
# screen.  Do not hand-edit the params file to bypass that gate.

PARAMS_FILE="${PARAMS_FILE:-jobs/params_ntcomp_cpu.txt}"
EXPECTED_ROWS="${EXPECTED_ROWS:-1360}"
if [ ! -f "$PARAMS_FILE" ]; then
  echo "$PARAMS_FILE missing; run screen analysis, then python jobs/make_params_ntcomp.py" >&2
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

HELP_TEXT="$(python -m rcomp train --help)"
for FLAG in --policy-learning-kwargs --preset --ensemble-training --round0-data-protocol \
            --dedicated-query-rng --ensemble-bootstrap \
            --reward-model-train-accuracy-stop --final-policy --n-envs \
            --round0-collection-timesteps --partial-alpha \
            --normalize-partial-reward --normalize-model-reward \
            --active-query-strategy --active-candidate-protocol; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required flag: $FLAG -- git pull" >&2
    exit 2
  fi
done

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE" | tr -d '\r')"
read -r CELL VARIANT SEED PARTIAL <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${SEED:-}" ] || [ -z "${PARTIAL:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
PARTIAL_NAME="${PARTIAL##*:}"
if [[ "$PARTIAL_NAME" == *timid* ]]; then
  echo "confirmatory grid refuses timid partial: $PARTIAL" >&2
  exit 2
fi

PRESET=(--preset generic)
PPO_KWARGS='{"n_steps":256,"ent_coef":0.01,"target_kl":0.03}'

case "$CELL" in
  ll)
    SUITE=box2d; ENV=LunarLander-v3
    FRAGMENT=25; COLLECTION=40000; EVALEP=10
    PRESET=() ;;
  reacher)
    SUITE=mujoco; ENV=Reacher-v5
    FRAGMENT=25; COLLECTION=20000; EVALEP=20 ;;
  pusher)
    SUITE=mujoco; ENV=Pusher-v5
    FRAGMENT=25; COLLECTION=20000; EVALEP=10 ;;
  swimmer)
    SUITE=mujoco; ENV=Swimmer-v5
    FRAGMENT=50; COLLECTION=50000; EVALEP=5
    PPO_KWARGS='{"n_steps":256,"gamma":0.9999,"ent_coef":0.01,"target_kl":0.03}' ;;
  hopper)
    SUITE=mujoco; ENV=Hopper-v5
    FRAGMENT=10; COLLECTION=30000; EVALEP=10 ;;
  bipedal)
    SUITE=box2d; ENV=BipedalWalker-v3
    FRAGMENT=50; COLLECTION=30000; EVALEP=10
    PRESET=() ;;
  walker)
    SUITE=mujoco; ENV=Walker2d-v5
    FRAGMENT=10; COLLECTION=30000; EVALEP=10 ;;
  ant)
    SUITE=mujoco; ENV=Ant-v5
    FRAGMENT=25; COLLECTION=40000; EVALEP=10
    PPO_KWARGS='{"n_steps":256,"ent_coef":0.01,"target_kl":0.03,"policy_kwargs":{"log_std_init":-2,"activation_fn":"Tanh","net_arch":{"pi":[256,256],"vf":[256,256]}}}' ;;
  mspacman)
    SUITE=atari; ENV=ALE/MsPacman-v5
    FRAGMENT=64; COLLECTION=50000; EVALEP=10
    PRESET=()
    PPO_KWARGS='{"n_steps":128,"ent_coef":0.01,"target_kl":0.03}' ;;
  qbert)
    SUITE=atari; ENV=ALE/Qbert-v5
    FRAGMENT=64; COLLECTION=50000; EVALEP=10
    PRESET=()
    PPO_KWARGS='{"n_steps":128,"ent_coef":0.01,"target_kl":0.03}' ;;
  pong)
    SUITE=atari; ENV=ALE/Pong-v5
    FRAGMENT=64; COLLECTION=50000; EVALEP=10
    PRESET=()
    PPO_KWARGS='{"n_steps":128,"ent_coef":0.01,"target_kl":0.03}' ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

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

METHOD="$VARIANT"
BUDGET=""
if [[ "$VARIANT" == *_q* ]]; then
  METHOD="${VARIANT%_q*}"
  BUDGET="${VARIANT##*_q}"
fi

LOGDIR="logs/ntcomp_${CELL}_${VARIANT}"
RUNNAME="ntcomp_${CELL}_${VARIANT}_seed${SEED}"
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
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
  --active-learning --active-query-strategy ensemble --active-candidate-protocol pool
)

case "$METHOD" in
  true)    ARGS+=(--mode true) ;;
  partial) ARGS+=(--mode partial --partial "$PARTIAL") ;;
  vanilla) ARGS+=("${RM[@]}" --mode feedback) ;;
  naive)   ARGS+=("${RM[@]}" --mode naive --partial "$PARTIAL" --partial-alpha 1.0) ;;
  ws25)    ARGS+=("${RM[@]}" --mode weighted_sum --partial "$PARTIAL" --partial-alpha 0.25 \
                  --normalize-partial-reward --normalize-model-reward) ;;
  ws50)    ARGS+=("${RM[@]}" --mode weighted_sum --partial "$PARTIAL" --partial-alpha 0.50 \
                  --normalize-partial-reward --normalize-model-reward) ;;
  ws75)    ARGS+=("${RM[@]}" --mode weighted_sum --partial "$PARTIAL" --partial-alpha 0.75 \
                  --normalize-partial-reward --normalize-model-reward) ;;
  *) echo "unknown method: $METHOD (from $VARIANT)" >&2; exit 2 ;;
esac

srun python -m rcomp train "${ARGS[@]}"
