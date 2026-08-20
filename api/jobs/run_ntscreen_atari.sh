#!/bin/bash
#SBATCH --job-name=ntscreen_atari
#SBATCH --output=logs/slurm/ntscreen_atari_%A_%a.out
#SBATCH --error=logs/slurm/ntscreen_atari_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --array=1-90
#SBATCH --requeue
set -euo pipefail

nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.cuda.is_available(); print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0)); print('cuda smoke', torch.ones(1, device='cuda').item())"

export PARAMS_FILE=jobs/params_ntscreen_atari.txt
export EXPECTED_ROWS=90
bash jobs/run_ntscreen.sh
