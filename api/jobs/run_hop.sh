#!/bin/bash
#SBATCH --job-name=hop
#SBATCH --output=logs/slurm/hop_%A_%a.out
#SBATCH --error=logs/slurm/hop_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-40
#SBATCH --requeue
set -euo pipefail

# ============================================================================
# HOPPER PRIOR SCREEN -- pick the one prior Hopper contributes to the
# composition experiment.
#
#   module load mamba && source activate /scratch/work/akishea1/envs/rcomp
#   python jobs/make_params_hop.py
#   wc -l jobs/params_hop.txt         # must print 40
#   mkdir -p logs/slurm
#   squeue -u $USER                   # nothing already writing to logs/hop_*
#   sbatch --array=1-40 jobs/run_hop.sh
#
#   python jobs/analyze_sel.py --prefix hop --cell hopper
#
# Hopper replaces HalfCheetah in the composition experiment. It was excluded by
# request from the selection grid, so nothing is known about where its priors
# land against its own vanilla floor. This measures that and nothing else.
#
# SELECTION RULE: pick the prior at or BELOW the vanilla floor. That is not the
# sel grid's 0.2-0.8 partiality band -- the composition experiment needs the
# reward model to have something left to add, so read the raw finals against
# vanilla rather than the partiality column.
#
# ============================================================================
# PPO SETTINGS: THE ORIGINAL SELECTION-GRID ONES. NO target_kl.
#
# ent_coef 0.01, n_steps 256 at n_envs 8, --preset generic, stock SB3
# otherwise. This deliberately does NOT carry the target_kl 0.03 from sel2,
# because the composition experiment will not use it either, and a prior
# screened under different optimizer settings does not transfer: on sel2,
# target_kl moved Walker2d's vanilla floor from 398 to 3671 and reordered every
# prior in that cell. Screen under the settings you will run on.
#
# Consequence to expect, from the sel grid measured under these same settings:
# Hopper's true arm will probably be unstable and may end well below its own
# peak, the way Walker2d's did (drawdown 0.72 there). That is a known cost of
# dropping target_kl. Read peak alongside final in the analyzer output; if the
# true arm collapses, the vanilla floor is still a valid target for this
# screen, since the rule here is stated against vanilla, not against true.
# ============================================================================

if [ ! -f jobs/params_hop.txt ]; then
  echo "jobs/params_hop.txt missing; run: python jobs/make_params_hop.py" >&2
  exit 2
fi
if [ "$(wc -l < jobs/params_hop.txt)" -ne 40 ]; then
  echo "jobs/params_hop.txt must contain exactly 40 lines" >&2
  exit 2
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not on PATH -- activate the environment before submitting:" >&2
  echo "  module load mamba && source activate /scratch/work/akishea1/envs/rcomp" >&2
  exit 2
fi

HELP_TEXT="$(python -m rcomp train --help)"
for FLAG in --policy-learning-kwargs --preset --ensemble-training --round0-data-protocol \
            --tanh-model-reward --dedicated-query-rng --ensemble-bootstrap \
            --reward-model-train-accuracy-stop --final-policy --n-envs \
            --round0-collection-timesteps --tanh-scale \
            --active-query-strategy --active-candidate-protocol; do
  if [[ "$HELP_TEXT" != *"$FLAG"* ]]; then
    echo "checked-out rcomp CLI is missing required flag: $FLAG -- git pull" >&2
    exit 2
  fi
done

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/params_hop.txt | tr -d '\r')"
read -r CELL VARIANT SEED <<< "$LINE"
if [ -z "${CELL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${SEED:-}" ]; then
  echo "no valid parameter row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

SUITE=mujoco
ENV=Hopper-v5
# Measured on installed gymnasium 1.2.3: Hopper-v5 random-policy episode length
# is median 17 (min 9, max 80), essentially the same profile as Walker2d's 18,
# so Walker2d's fragment/collection pair carries over unchanged.
# fragment_trajectories cuts non-overlapping blocks and DISCARDS the remainder,
# so an episode shorter than FRAGMENT yields nothing at all.
# COLLECTION 30000 / 17 = ~1765 episodes, far above the 140 fragments a round
# needs to buy its 70 pairs.
FRAGMENT=10
COLLECTION=30000
EVALEP=10
PPO_KWARGS='{n_steps:256,ent_coef:0.01}'

# The params file carries the bare variant name so log dirs stay readable; map
# it back to the file:name form --partial actually requires. A bare name
# resolves the MODULE, which is why hopper_capped_low and the three levels
# entries must be qualified (verified: the bare forms raise PartialRegistryError).
case "$VARIANT" in
  true|vanilla)                  PARTIAL="" ;;
  hopper_capped_low)             PARTIAL="mujoco_capped_low:hopper_capped_low" ;;
  hopper_p25)                    PARTIAL="hopper_levels:hopper_p25" ;;
  hopper_p50)                    PARTIAL="hopper_levels:hopper_p50" ;;
  hopper_p75)                    PARTIAL="hopper_levels:hopper_p75" ;;
  hopper_capped_forward_survive) PARTIAL="hopper_capped_forward_survive" ;;
  hopper_gameable_bounce)        PARTIAL="hopper_gameable_bounce" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

LOGDIR="logs/hop_${CELL}_${VARIANT}"
RUNNAME="hop_${CELL}_${VARIANT}_seed${SEED}"
# Guards only against re-running a FINISHED run. It does NOT stop two concurrent
# submissions writing into one directory -- check `squeue -u $USER` first.
if [ -f "${LOGDIR}/${RUNNAME}/metadata.json" ]; then
  echo "already complete: ${LOGDIR}/${RUNNAME}"
  exit 0
fi

# Resolve the prior before burning a queue slot, using the exact call the
# trainer makes. NOT `rcomp validate-partial` and NOT `rcomp list-partials`.
if [ -n "$PARTIAL" ]; then
  if ! python -c "
import sys
from rcomp.partials import PartialRegistry, load_partial_reference
try:
    spec = load_partial_reference('$PARTIAL', '$SUITE', PartialRegistry())
    spec.create('$ENV')
except Exception as exc:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    sys.exit(1)
"; then
    echo "partial '$PARTIAL' does not resolve/instantiate for $ENV in suite $SUITE" >&2
    exit 2
  fi
fi

ARGS=(
  --suite "$SUITE" --env-id "$ENV"
  --n-envs 8 --policy-learning-kwargs "$PPO_KWARGS"
  --timesteps 2000000 --seed "$SEED"
  --final-policy last
  --eval-freq 20000 --n-eval-episodes "$EVALEP" --final-eval-episodes 30
  --run-name "$RUNNAME" --log-dir "$LOGDIR"
  --preset generic
)

# The standardized (B-Pref-shaped) reward model -- identical to run_sel.sh.
RM=(
  --query-budget 350 --rlhf-rounds 5
  --collection-timesteps "$COLLECTION" --fragment-length "$FRAGMENT"
  --round0-collection-timesteps 50000
  --reward-hidden-sizes 256,256,256
  --reward-model-ensemble-size 3
  --reward-model-lr 0.0003
  --reward-model-batch-size 32
  --reward-model-loss-reduction mean
  --reward-model-l1 0
  --reward-output-l1 0.001
  --ensemble-training full
  --ensemble-bootstrap
  --reward-model-train-accuracy-stop 0.97
  --round0-data-protocol separate
  --dedicated-query-rng
  --tanh-model-reward
  --tanh-scale 5
  --active-learning
  --active-query-strategy ensemble
  --active-candidate-protocol pool
)

case "$VARIANT" in
  true)    ARGS+=(--mode true) ;;
  vanilla) ARGS+=("${RM[@]}" --mode feedback) ;;
  *)       ARGS+=(--mode partial --partial "$PARTIAL") ;;
esac

srun python -m rcomp train "${ARGS[@]}"
