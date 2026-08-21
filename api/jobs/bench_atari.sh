#!/bin/bash
#SBATCH --job-name=bench_atari
#SBATCH --output=logs/slurm/bench_atari_%A_%a.out
#SBATCH --error=logs/slurm/bench_atari_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
set -euo pipefail

# Throughput benchmark for the pixel-Atari configuration used by
# jobs/run_reasonable_atari.sh.  Every task runs MsPacman for 60k policy steps
# and reports wall-clock steps/s, varying ONE thing at a time:
#   evaluation frequency, mode, torch thread count, ensemble size, device.

PY=/scratch/work/akishea1/envs/rcomp/bin/python
STEPS=60000
NOEVAL=100000000

ARGS=(--suite atari --env-id ALE/MsPacman-v5 --n-envs 8
      --policy-learning-kwargs '{"n_steps":128,"ent_coef":0.01,"target_kl":0.03}'
      --timesteps "$STEPS" --seed 0 --final-policy last
      --final-eval-episodes 5 --log-dir logs/bench_atari)

RM=(--query-budget 40 --rlhf-rounds 2
    --collection-timesteps 10000 --fragment-length 64
    --round0-collection-timesteps 10000
    --reward-hidden-sizes 256,256,256
    --reward-model-ensemble-size 3
    --reward-model-lr 0.0003 --reward-model-batch-size 32
    --reward-model-loss-reduction mean
    --reward-model-l1 0 --reward-output-l1 0.001
    --ensemble-training full --ensemble-bootstrap
    --reward-model-train-accuracy-stop 0.97
    --round0-data-protocol separate --dedicated-query-rng
    --tanh-model-reward --tanh-scale 5
    --active-learning --active-query-strategy ensemble --active-candidate-protocol pool)

PARTIAL=(--mode partial --partial reasonable_atari_partials:rmsp_pellets)

case "$SLURM_ARRAY_TASK_ID" in
  1) NAME=true_eval20k;    MODE=(--mode true);      EF=20000;  NE=10 ;;
  2) NAME=true_noeval;     MODE=(--mode true);      EF=$NOEVAL; NE=1 ;;
  3) NAME=partial_eval20k; MODE=("${PARTIAL[@]}");  EF=20000;  NE=10 ;;
  4) NAME=partial_noeval;  MODE=("${PARTIAL[@]}");  EF=$NOEVAL; NE=1 ;;
  5) NAME=vanilla_eval20k; MODE=(--mode feedback);  EF=20000;  NE=10 ;;
  6) NAME=vanilla_noeval;  MODE=(--mode feedback);  EF=$NOEVAL; NE=1 ;;
  7) NAME=vanilla_1thread; MODE=(--mode feedback);  EF=$NOEVAL; NE=1
     export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ;;
  8) NAME=vanilla_ens1;    MODE=(--mode feedback);  EF=$NOEVAL; NE=1
     RM=(--query-budget 40 --rlhf-rounds 2 --collection-timesteps 10000
         --fragment-length 64 --round0-collection-timesteps 10000
         --reward-hidden-sizes 256,256,256 --reward-model-ensemble-size 1
         --reward-model-lr 0.0003 --reward-model-batch-size 32
         --reward-model-loss-reduction mean --reward-model-l1 0
         --reward-output-l1 0.001 --ensemble-training full
         --reward-model-train-accuracy-stop 0.97
         --round0-data-protocol separate --dedicated-query-rng
         --tanh-model-reward --tanh-scale 5) ;;
  9) NAME=true_omp8;       MODE=(--mode true);      EF=$NOEVAL; NE=1
     export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ;;
  *) echo "unknown task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

RUN=(--run-name "bench_${NAME}" --eval-freq "$EF" --n-eval-episodes "$NE")
[ "${MODE[1]:-}" = feedback ] && RUN+=("${RM[@]}")

rm -rf "logs/bench_atari/bench_${NAME}"
echo "=== $NAME on $(hostname) omp=${OMP_NUM_THREADS:-unset} ==="
$PY -c "import torch; print('torch threads', torch.get_num_threads(), 'cuda', torch.cuda.is_available())"
T0=$(date +%s)
srun $PY -m rcomp train "${ARGS[@]}" "${MODE[@]}" "${RUN[@]}"
T1=$(date +%s)
echo "BENCH $NAME wall=$((T1-T0))s steps=$STEPS fps=$(( STEPS / (T1-T0+1) ))"
