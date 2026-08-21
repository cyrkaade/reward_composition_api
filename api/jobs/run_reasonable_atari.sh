#!/bin/bash
#SBATCH --job-name=reasonable_atari
#SBATCH --output=logs/slurm/reasonable_atari_%A_%a.out
#SBATCH --error=logs/slurm/reasonable_atari_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --array=1-150
#SBATCH --requeue
set -euo pipefail

# Independent A100 array for MsPacman, Qbert, and Pong.  Policy/reward-model
# inputs are stacked grayscale pixels; hand-written partials receive only the
# synchronized RAM snapshot supplied by AtariSuite.

nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.cuda.is_available(); print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0)); print('cuda smoke', torch.ones(1, device='cuda').item())"

export PARAMS_FILE=jobs/params_reasonable_atari.txt
export EXPECTED_ROWS=150
bash jobs/run_reasonable.sh
