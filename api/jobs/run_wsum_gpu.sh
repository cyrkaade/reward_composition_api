#!/bin/bash
#SBATCH --job-name=wsumg
#SBATCH --account=ellis_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/wsumg_%A_%a.out
#SBATCH --error=logs/slurm/wsumg_%A_%a.err
#SBATCH --requeue
set -uo pipefail

# Bundling wrapper for the weighted-sum grid.
#
# gpu-h200-141g-ellis is the only partition where this account has usable
# priority (fairshare 0.478 against 0.0056 on aalto_users, whose CPU jobs sat
# at Reason=Priority indefinitely), and its QOS rejects any job that does not
# request a GPU (QOSMinGRES).  These are MLP-policy runs that do not need one,
# so rather than burn a whole H200 per run each task takes a GPU slot and runs
# PER_TASK rows of params_wsum.txt side by side on its 16 cores.
#
# Each child is the ordinary run_wsum.sh body with LAUNCHER="" so it execs
# python directly inside this allocation instead of creating a job step.

export PATH="/scratch/work/akishea1/envs/rcomp/bin:${PATH}"

PER_TASK="${PER_TASK:-4}"
THREADS="${THREADS:-4}"
TOTAL_ROWS="${TOTAL_ROWS:-650}"
JOB="${SLURM_ARRAY_JOB_ID:-manual}"

first=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_TASK + 1 ))
last=$(( first + PER_TASK - 1 ))
[ "$last" -gt "$TOTAL_ROWS" ] && last="$TOTAL_ROWS"
if [ "$first" -gt "$TOTAL_ROWS" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID}: rows beyond ${TOTAL_ROWS}, nothing to do"
  exit 0
fi

echo "task ${SLURM_ARRAY_TASK_ID} on $(hostname) cores=$(nproc): rows ${first}..${last} (${THREADS} threads each)"

pids=()
rows=()
for row in $(seq "$first" "$last"); do
  (
    export SLURM_ARRAY_TASK_ID="$row"
    export LAUNCHER=""
    export OMP_NUM_THREADS="$THREADS"
    export MKL_NUM_THREADS="$THREADS"
    export OPENBLAS_NUM_THREADS="$THREADS"
    exec bash jobs/run_wsum.sh
  ) > "logs/slurm/wsumg_${JOB}_row${row}.log" 2>&1 &
  pids+=($!)
  rows+=("$row")
done

# Wait for every child before reporting, so one failure does not hide the rest.
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "row ${rows[$i]}: ok"
  else
    echo "row ${rows[$i]}: FAILED (see logs/slurm/wsumg_${JOB}_row${rows[$i]}.log)" >&2
    failed=$((failed + 1))
  fi
done

echo "task ${SLURM_ARRAY_TASK_ID} done: $(( ${#pids[@]} - failed ))/${#pids[@]} rows ok"
exit $(( failed > 0 ))
