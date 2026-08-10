#!/bin/bash
#SBATCH --job-name=pre3f
#SBATCH --output=logs/slurm/pre3f_%A_%a.out
#SBATCH --error=logs/slurm/pre3f_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-90
#SBATCH --requeue

# PRE3 STAGE 1: repair the gate by isolating WHY it failed (90 runs).
#
# WHAT THE GATE ACTUALLY DID.  It did not fail ambiguously; it failed by
# episode-length farming.  LunarLander's time limit is 1000 steps.  Median final
# mean episode length, read from each run's eval/evaluations.npz:
#
#     true              202 steps    final  278.9
#     partial           525          final  225.8
#     feedback bounded  964          final  -69.8
#     naive    bounded  992          final   -2.7
#
# The failing policies stopped landing and hovered until timeout.  Per seed, in
# pre3g_ll_naive_bounded, all six seeds that lost solved status grew their mean
# episode length by 175-620 steps (median +476); the single seed that kept it
# (seed3, 228 -> 221) grew by 12.  Pooling all 40 gate preference runs, those
# ending at the limit (n=23) have median final -74.2 and those still terminating
# (n=17) have median +170.8.  Hovering rate went from 11/90 (12%) across the
# archive to 23/40 (58%) in the gate.
#
# TWO CANDIDATE CAUSES, both introduced by the gate, neither sufficient alone:
#
#   1. DISCOUNT.  --tuned-hyperparams takes LunarLander from gamma .99 to .999,
#      so the effective horizon 1/(1-gamma) goes 100 -> 1000 steps: exactly the
#      time limit.  Isolated evidence with no reward model in the loop at all -
#      mode=partial runs 163-184 steps at gamma .99 and 525 at gamma .999, while
#      mode=true is immune (185 -> 202) because the landing bonus explicitly pays
#      for terminating.  rl-zoo tuned that gamma against the ground-truth reward,
#      where hovering is punished; under a learned reward it opens an exploit.
#
#   2. REWARD LEVEL.  With equal-length fragments the Bradley-Terry loss is
#      EXACTLY invariant to adding a constant c to every state: both sides of the
#      comparison gain 25c and the softmax is unchanged.  The additive level is
#      therefore fixed by initialisation and drift, not by data - and in a
#      variable-horizon MDP that level decides whether the policy wants to live
#      forever.  The archive pinned it with the output L1 term (still the default,
#      0.001); the gate set --reward-output-l1 0.
#
# Every B-Pref environment is fixed-horizon (DMControl 1000 steps, Meta-world
# 500, no early termination), which is why the standard recipe never needed a
# level term and why importing it wholesale broke a variable-horizon task.
#
# DESIGN.  Baseline is the EXISTING logs/pre3g_ll_naive_bounded, so it costs no
# runs and every arm below is a one- or two-flag delta from it.  All other
# settings are copied verbatim from run_pre3_gate.sh so each arm isolates one
# thing.  The job also carries the matched references and the feedback partners
# for the two most likely winners, so no second round is needed whichever lever
# wins.  It subsumes run_pre3_repair.sh rows 1-20 (that is the *_stop arms).
#
# PRIMARY READOUT is the HOVERING RATE - the fraction of seeds whose final mean
# eval episode length is >= 950 - not a median return.  It is a binary with a
# 12%-vs-58% effect, so at n=10 it has far more power than medians of a
# heavy-tailed return.  Use jobs/analyze_pre3_fix.py.
#
# DECISION RULE:
#   * A lever is a fix if it cuts naive's hovering rate to <= 2/10 AND leaves
#     >= 5/10 seeds finishing solved (>= 200).
#   * Prefer the SMALLEST fix that clears that bar; prefer naive_g99_l1 only if
#     neither single lever clears it.
#   * Then read M1 (naive vs feedback) at the winning configuration and carry
#     that configuration into every later stage.
#   * If all four levers fail, the standardized reward model is not viable on a
#     variable-horizon task; fall back to run_pre3_repair.sh rows 21-40 (legacy
#     reward model) before concluding anything about pretraining or AL.
#
# PREFLIGHT:
#   python jobs/make_params_pre3_fix.py
#   wc -l jobs/params_pre3_fix.txt        # must be 90
#   mkdir -p logs/slurm
#   sbatch jobs/run_pre3_fix.sh

set -euo pipefail

module load mamba
source activate /scratch/work/akishea1/envs/rcomp
cd /scratch/work/akishea1/reward_composition_api/api

if [ ! -f jobs/params_pre3_fix.txt ]; then
  echo "missing jobs/params_pre3_fix.txt; run make_params_pre3_fix.py first" >&2
  exit 2
fi

if [ "$(wc -l < jobs/params_pre3_fix.txt)" -ne 90 ]; then
  echo "jobs/params_pre3_fix.txt must contain exactly 90 lines" >&2
  exit 2
fi

# Fail before allocating hours of work if the checked-out CLI predates a flag
# this job depends on.
HELP_TEXT="$(python -m rcomp train --help)"
REQUIRED_FLAGS=(
  --reward-model-loss-reduction
  --reward-output-l1
  --reward-model-train-accuracy-stop
  --ensemble-training
  --round0-data-protocol
  --tanh-model-reward
  --policy-learning-kwargs
)
for FLAG in "${REQUIRED_FLAGS[@]}"; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required flag: $FLAG" >&2
    exit 2
  fi
done

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_pre3_fix.txt | tr -d '\r')"
read -r CELL VARIANT SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

if [ "$CELL" != "ll" ]; then
  echo "unknown cell: $CELL" >&2
  exit 2
fi

SUITE=box2d
ENV=LunarLander-v3
PARTIAL=lunar_lander_approach
FRAGMENT=25
COLLECTION=30000

LOGDIR="logs/pre3f_${CELL}_${VARIANT}"
RUNNAME="${CELL}_${VARIANT}_q350_seed${SEED}"

if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

# Identical to run_pre3_gate.sh so each arm below is a clean one-variable delta.
ARGS=(
  --suite "$SUITE"
  --env-id "$ENV"
  --tuned-hyperparams
  --final-policy last
  --timesteps 2000000
  --seed "$SEED"
  --eval-freq 50000
  --n-eval-episodes 20
  --run-name "$RUNNAME"
  --log-dir "$LOGDIR"
)

# The gate's standardized reward-model block, verbatim.
GATE_RM=(
  --query-budget 350
  --rlhf-rounds 5
  --no-active-learning
  --collection-timesteps "$COLLECTION"
  --fragment-length "$FRAGMENT"
  --reward-hidden-sizes 256,256,256
  --reward-model-ensemble-size 3
  --reward-model-lr 0.0003
  --reward-model-batch-size 128
  --reward-model-loss-reduction mean
  --reward-model-l1 0
  --reward-output-l1 0
  --ensemble-training full
  --round0-data-protocol separate
  --tanh-model-reward
)
NAIVE=(--mode naive --partial-alpha 1.0 --no-include-partial-feature)
G99=(--policy-learning-kwargs '{gamma:0.99}')
LEVEL=(--reward-output-l1 0.001)   # overrides the 0 in GATE_RM
STOP=(--reward-model-train-accuracy-stop 0.97)

case "$VARIANT" in
  # --- repair levers on naive (baseline = logs/pre3g_ll_naive_bounded) -------
  naive_g99)     ARGS+=(--partial "$PARTIAL" "${GATE_RM[@]}" "${NAIVE[@]}" "${G99[@]}") ;;
  naive_l1)      ARGS+=(--partial "$PARTIAL" "${GATE_RM[@]}" "${NAIVE[@]}" "${LEVEL[@]}") ;;
  naive_stop)    ARGS+=(--partial "$PARTIAL" "${GATE_RM[@]}" "${NAIVE[@]}" "${STOP[@]}") ;;
  naive_g99_l1)  ARGS+=(--partial "$PARTIAL" "${GATE_RM[@]}" "${NAIVE[@]}" "${G99[@]}" "${LEVEL[@]}") ;;

  # --- feedback partners, so M1 is measurable at a working configuration ----
  feedback_g99_l1) ARGS+=(--partial "$PARTIAL" "${GATE_RM[@]}" --mode feedback "${G99[@]}" "${LEVEL[@]}") ;;
  feedback_stop)   ARGS+=(--partial "$PARTIAL" "${GATE_RM[@]}" --mode feedback "${STOP[@]}") ;;

  # --- matched floor and ceiling at gamma 0.99 ------------------------------
  # The gate's true/partial ran at gamma .999. If the discount is the fix, the
  # references have to move with it or the qualification check is mismatched.
  # --partial is inert in mode=true but the gate passed it, so pass it here too
  # and keep partial_reference identical in the metadata.
  true_g99)      ARGS+=(--partial "$PARTIAL" --mode true "${G99[@]}") ;;
  partial_g99)   ARGS+=(--partial "$PARTIAL" --mode partial "${G99[@]}") ;;

  # --- the partial-design prediction, at the UNREPAIRED gate config ---------
  # Same as pre3g_ll_naive_bounded except the partial. lunar_lander_approach is
  # pure potential-based shaping, so it cannot change the optimal policy and
  # carries no incentive to terminate; lunarlander_p50 adds a non-potential
  # terminal term. Prediction: p50 suppresses hovering where approach cannot.
  naive_p50)     ARGS+=(--partial lunarlander_p50 "${GATE_RM[@]}" "${NAIVE[@]}") ;;

  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

srun python -m rcomp train "${ARGS[@]}"
