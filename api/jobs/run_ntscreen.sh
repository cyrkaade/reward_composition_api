#!/bin/bash
#SBATCH --job-name=ntscreen
#SBATCH --output=logs/slurm/ntscreen_%A_%a.out
#SBATCH --error=logs/slurm/ntscreen_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-240
#SBATCH --requeue
set -euo pipefail

# Stage one is deliberately only true reward versus five partials. The CPU
# array contains eight Box2D/MuJoCo cells; run_ntscreen_atari.sh reuses this
# body for three pixel-policy/RAM-partial Atari cells on GPU nodes.

PARAMS_FILE="${PARAMS_FILE:-jobs/params_ntscreen_cpu.txt}"
EXPECTED_ROWS="${EXPECTED_ROWS:-240}"
if [ ! -f "$PARAMS_FILE" ]; then
  echo "$PARAMS_FILE missing; run: python jobs/make_params_ntscreen.py" >&2
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
for FLAG in --policy-learning-kwargs --preset --final-policy --n-envs; do
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
  echo "screen refuses timid partial reference: $PARTIAL" >&2
  exit 2
fi

PRESET=(--preset generic)
PPO_KWARGS='{"n_steps":256,"ent_coef":0.01,"target_kl":0.03}'

case "$CELL" in
  ll)
    SUITE=box2d; ENV=LunarLander-v3; EVALEP=10; PRESET=() ;;
  reacher)
    SUITE=mujoco; ENV=Reacher-v5; EVALEP=20 ;;
  pusher)
    SUITE=mujoco; ENV=Pusher-v5; EVALEP=10 ;;
  swimmer)
    SUITE=mujoco; ENV=Swimmer-v5; EVALEP=5
    PPO_KWARGS='{"n_steps":256,"gamma":0.9999,"ent_coef":0.01,"target_kl":0.03}' ;;
  hopper)
    SUITE=mujoco; ENV=Hopper-v5; EVALEP=10 ;;
  bipedal)
    SUITE=box2d; ENV=BipedalWalker-v3; EVALEP=10; PRESET=() ;;
  walker)
    SUITE=mujoco; ENV=Walker2d-v5; EVALEP=10 ;;
  ant)
    SUITE=mujoco; ENV=Ant-v5; EVALEP=10
    PPO_KWARGS='{"n_steps":256,"ent_coef":0.01,"target_kl":0.03,"policy_kwargs":{"log_std_init":-2,"activation_fn":"Tanh","net_arch":{"pi":[256,256],"vf":[256,256]}}}' ;;
  mspacman)
    SUITE=atari; ENV=ALE/MsPacman-v5; EVALEP=10; PRESET=()
    PPO_KWARGS='{"n_steps":128,"ent_coef":0.01,"target_kl":0.03}' ;;
  qbert)
    SUITE=atari; ENV=ALE/Qbert-v5; EVALEP=10; PRESET=()
    PPO_KWARGS='{"n_steps":128,"ent_coef":0.01,"target_kl":0.03}' ;;
  pong)
    SUITE=atari; ENV=ALE/Pong-v5; EVALEP=10; PRESET=()
    PPO_KWARGS='{"n_steps":128,"ent_coef":0.01,"target_kl":0.03}' ;;
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

LOGDIR="logs/ntscreen_${CELL}_${VARIANT}"
RUNNAME="ntscreen_${CELL}_${VARIANT}_seed${SEED}"
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
  --n-envs 8 --policy-learning-kwargs "$PPO_KWARGS"
  --timesteps 2000000 --seed "$SEED"
  --final-policy last
  --eval-freq 20000 --n-eval-episodes "$EVALEP" --final-eval-episodes 30
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
  "${PRESET[@]}"
)

case "$VARIANT" in
  true)
    [ "$PARTIAL" = "-" ] || { echo "true row unexpectedly carries a partial" >&2; exit 2; }
    ARGS+=(--mode true) ;;
  *)
    [ "$PARTIAL" != "-" ] || { echo "candidate row has no partial" >&2; exit 2; }
    ARGS+=(--mode partial --partial "$PARTIAL") ;;
esac

srun python -m rcomp train "${ARGS[@]}"
