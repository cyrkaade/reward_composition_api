#!/bin/bash
#SBATCH --job-name=wsumag
#SBATCH --account=ellis_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/wsumag_%A_%a.out
#SBATCH --error=logs/slurm/wsumag_%A_%a.err
#SBATCH --requeue
set -uo pipefail

# Bundling wrapper for the Atari weighted-sum grid.
#
# Unlike run_wsum_gpu.sh, these runs genuinely WANT the GPU: both the PPO policy
# and the 3-member preference reward model are Nature CNNs. Measured on an H200
# (CLAUDE.md, 2026-08-21): cuda + --batch-env-reward-inference is 2.66x the
# historical CPU per-env path. So PER_TASK is small -- 4 runs share one H200 --
# rather than the 16 that made sense for MLP runs that never touched it.
export PATH="/scratch/work/akishea1/envs/rcomp/bin:${PATH}"

PER_TASK="${PER_TASK:-4}"
THREADS="${THREADS:-8}"
TOTAL_ROWS="${TOTAL_ROWS:-260}"
JOB="${SLURM_ARRAY_JOB_ID:-manual}"

first=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_TASK + 1 ))
last=$(( first + PER_TASK - 1 ))
[ "$last" -gt "$TOTAL_ROWS" ] && last="$TOTAL_ROWS"
if [ "$first" -gt "$TOTAL_ROWS" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID}: rows beyond ${TOTAL_ROWS}, nothing to do"
  exit 0
fi

echo "task ${SLURM_ARRAY_TASK_ID} on $(hostname) cores=$(nproc) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  rows ${first}..${last} (${THREADS} threads each)"

pids=()
rows=()
for row in $(seq "$first" "$last"); do
  (
    export SLURM_ARRAY_TASK_ID="$row"
    export LAUNCHER=""
    export OMP_NUM_THREADS="$THREADS"
    export MKL_NUM_THREADS="$THREADS"
    export OPENBLAS_NUM_THREADS="$THREADS"
    exec bash jobs/run_wsum_atari.sh
  ) > "logs/slurm/wsumag_${JOB}_row${row}.log" 2>&1 &
  pids+=($!)
  rows+=("$row")
done

# Wait for every child before reporting, so one failure does not hide the rest.
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "row ${rows[$i]}: ok"
  else
    echo "row ${rows[$i]}: FAILED (see logs/slurm/wsumag_${JOB}_row${rows[$i]}.log)" >&2
    failed=$((failed + 1))
  fi
done

echo "task ${SLURM_ARRAY_TASK_ID} done: $(( ${#pids[@]} - failed ))/${#pids[@]} rows ok"
exit $(( failed > 0 ))
