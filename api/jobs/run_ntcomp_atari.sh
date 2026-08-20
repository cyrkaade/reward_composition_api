#!/bin/bash
#SBATCH --job-name=ntcomp_atari
#SBATCH --output=logs/slurm/ntcomp_atari_%A_%a.out
#SBATCH --error=logs/slurm/ntcomp_atari_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --array=1-510
#SBATCH --requeue
set -euo pipefail

nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.cuda.is_available(); print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0)); print('cuda smoke', torch.ones(1, device='cuda').item())"

export PARAMS_FILE=jobs/params_ntcomp_atari.txt
export EXPECTED_ROWS=510
bash jobs/run_ntcomp.sh
