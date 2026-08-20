#!/bin/bash
#SBATCH --job-name=ntscreen_atari
#SBATCH --output=logs/slurm/ntscreen_atari_%A_%a.out
#SBATCH --error=logs/slurm/ntscreen_atari_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --array=1-90
#SBATCH --requeue
set -euo pipefail

export PARAMS_FILE=jobs/params_ntscreen_atari.txt
export EXPECTED_ROWS=90
bash jobs/run_ntscreen.sh
