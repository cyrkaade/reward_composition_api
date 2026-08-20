#!/bin/bash
#SBATCH --job-name=ntcomp_atari
#SBATCH --output=logs/slurm/ntcomp_atari_%A_%a.out
#SBATCH --error=logs/slurm/ntcomp_atari_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --array=1-510
#SBATCH --requeue
set -euo pipefail

export PARAMS_FILE=jobs/params_ntcomp_atari.txt
export EXPECTED_ROWS=510
bash jobs/run_ntcomp.sh
