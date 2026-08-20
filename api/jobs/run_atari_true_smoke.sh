#!/bin/bash
#SBATCH --job-name=atari_true_smoke
#SBATCH --output=logs/slurm/atari_true_smoke_%A_%a.out
#SBATCH --error=logs/slurm/atari_true_smoke_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --array=1-15
#SBATCH --requeue
set -euo pipefail

# Fifteen independent true-reward runs: three Atari games x seeds 0..4.
# This intentionally contains no partial, reward model, or preference query.

PARAMS_FILE="${PARAMS_FILE:-jobs/params_atari_true_smoke.txt}"
EXPECTED_ROWS=15
if [ ! -f "$PARAMS_FILE" ]; then
  echo "$PARAMS_FILE missing; run: python jobs/make_params_atari_true_smoke.py" >&2
  exit 2
fi
if [ "$(wc -l < "$PARAMS_FILE")" -ne "$EXPECTED_ROWS" ]; then
  echo "$PARAMS_FILE must contain exactly $EXPECTED_ROWS lines" >&2
  exit 2
fi

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE" | tr -d '\r')"
read -r CELL SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

case "$CELL" in
  mspacman) ENV=ALE/MsPacman-v5 ;;
  qbert)    ENV=ALE/Qbert-v5 ;;
  pong)     ENV=ALE/Pong-v5 ;;
  *) echo "unknown Atari cell: $CELL" >&2; exit 2 ;;
esac

LOGDIR="logs/atari_true_smoke_${CELL}"
RUNNAME="atari_true_smoke_${CELL}_seed${SEED}"
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

srun python -m rcomp train \
  --suite atari --env-id "$ENV" --mode true \
  --n-envs 8 \
  --policy-learning-kwargs '{"n_steps":128,"ent_coef":0.01,"target_kl":0.03}' \
  --timesteps 400000 --seed "$SEED" \
  --final-policy last \
  --eval-freq 10000 --n-eval-episodes 10 --final-eval-episodes 30 \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
