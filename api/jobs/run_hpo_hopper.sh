#!/bin/bash
#SBATCH --job-name=hpohop
#SBATCH --output=logs/slurm/hpohop_%A_%a.out
#SBATCH --error=logs/slurm/hpohop_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-48
#SBATCH --requeue

# Stage 1 of a hyperparameter search for Hopper-v5, because nobody has published
# one. Verified 2026-08-11: rl-baselines3-zoo's ppo.yml has NO v5 keys at all
# (only Hopper-v4, Walker2d-v4, HalfCheetah-v4, Humanoid-v4, Ant-v4, Swimmer-v4,
# Reacher-v2, InvertedPendulum-v2, InvertedDoublePendulum-v2, HumanoidStandup-v2).
# The one public Hopper-v5 PPO number is a bare "baseline_ppo_return 3183" with
# no configuration attached. So the v4 block is the closest published thing, and
# it does not transfer - v5 changed the reward (healthy_reward is no longer paid
# on unhealthy steps), which makes the stand-still optimum a stronger attractor.
#
# DESIGN: coordinate sweep, one factor at a time around the config validated in
# rcomp/ppo_presets.py (median peak 2995 / final 2127 over 3 seeds x 1M). A full
# grid over these 8 knobs is thousands of runs; this is 16 configs x 3 seeds = 48
# and tells us WHICH knobs matter. Stage 2 then grids only those, at more seeds.
#
# OBJECTIVE IS **FINAL**, NOT PEAK. Every experiment in this project runs with
# --final-policy last, and Hopper's whole problem is that it collapses after
# peaking. A config that peaks at 3500 and ends at 800 is useless here. The
# analyzer reports both so the gap stays visible.
#
# Baseline (variant `base`) = the current preset:
#   n_steps 512, batch 32, lr 3e-5, gamma 0.99, gae_lambda 0.9, clip_range 0.4,
#   clip_range_vf 0.5, n_epochs 20, ent_coef 0, net_arch [256,256],
#   SB3-default Tanh / ortho_init True / log_std_init 0, n_envs 1
#
# Analyze with:  python jobs/analyze_hpo_hopper.py
#
# NOTE ON WALKER: do not copy the winner across. Walker2d's rl-zoo block already
# works here (4986 vs 3038 for stock at 2M). Only Hopper needs this.

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_hpo_hopper.txt)
read -r VARIANT SEED <<< "$LINE"

# Baseline kwargs, then one field overridden per variant.
BASE='"n_steps":512,"batch_size":32,"learning_rate":3e-5,"gamma":0.99,"gae_lambda":0.9,"clip_range":0.4,"clip_range_vf":0.5,"n_epochs":20,"ent_coef":0'
PK='"policy_kwargs":{"net_arch":[256,256]}'
NENV=1

case "$VARIANT" in
  base)        KW="{$BASE,$PK}" ;;
  # learning rate: the single most common cause of both slow learning and late collapse
  lr1e5)       KW="{$BASE,$PK}"; KW=${KW/\"learning_rate\":3e-5/\"learning_rate\":1e-5} ;;
  lr1e4)       KW="{$BASE,$PK}"; KW=${KW/\"learning_rate\":3e-5/\"learning_rate\":1e-4} ;;
  lr3e4)       KW="{$BASE,$PK}"; KW=${KW/\"learning_rate\":3e-5/\"learning_rate\":3e-4} ;;
  # optimisation strength per batch
  ep5)         KW="{$BASE,$PK}"; KW=${KW/\"n_epochs\":20/\"n_epochs\":5} ;;
  ep10)        KW="{$BASE,$PK}"; KW=${KW/\"n_epochs\":20/\"n_epochs\":10} ;;
  # trust region
  clip02)      KW="{$BASE,$PK}"; KW=${KW/\"clip_range\":0.4/\"clip_range\":0.2} ;;
  novfclip)    KW="{$BASE,$PK}"; KW=${KW/\"clip_range_vf\":0.5/\"clip_range_vf\":null} ;;
  # horizon: 0.999 is what the rl-zoo block uses and is my prime suspect for
  # over-valuing the +1/step survival bonus (effective horizon 1000 vs 100)
  gam999)      KW="{$BASE,$PK}"; KW=${KW/\"gamma\":0.99,/\"gamma\":0.999,} ;;
  gae95)       KW="{$BASE,$PK}"; KW=${KW/\"gae_lambda\":0.9/\"gae_lambda\":0.95} ;;
  # exploration: proven load-bearing (forcing -2 collapsed the baseline 2835->452)
  logstd_m1)   KW="{$BASE,\"policy_kwargs\":{\"net_arch\":[256,256],\"log_std_init\":-1}}" ;;
  ent001)      KW="{$BASE,$PK}"; KW=${KW/\"ent_coef\":0/\"ent_coef\":0.001} ;;
  # batch geometry
  steps1024)   KW="{$BASE,$PK}"; KW=${KW/\"n_steps\":512/\"n_steps\":1024} ;;
  batch64)     KW="{$BASE,$PK}"; KW=${KW/\"batch_size\":32/\"batch_size\":64} ;;
  # does the config survive parallel envs? the preset has only ever been checked
  # at n_envs=1, and every other job in this project runs 8
  nenv8)       KW="{$BASE,$PK}"; NENV=8 ;;
  # ReLU instead of SB3-default Tanh (the rl-zoo/gSDE tables both use ReLU)
  relu)        KW="{$BASE,\"policy_kwargs\":{\"net_arch\":[256,256],\"activation_fn\":\"ReLU\"}}" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

LOGDIR="logs/hpo_hopper_${VARIANT}"
RUNNAME="${VARIANT}_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

echo "variant=$VARIANT seed=$SEED n_envs=$NENV kwargs=$KW"

srun python -m rcomp train \
  --suite mujoco --env-id Hopper-v5 --mode true --preset generic \
  --n-envs "$NENV" --final-policy last \
  --timesteps 1000000 --seed "$SEED" \
  --eval-freq 20000 --n-eval-episodes 20 \
  --policy-learning-kwargs "$KW" \
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
