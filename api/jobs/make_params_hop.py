"""Generate jobs/params_hop.txt (CELL VARIANT SEED) for the HOPPER PRIOR SCREEN.

Hopper replaces HalfCheetah in the composition experiment, but Hopper has never
been screened the way the other eight cells were -- it was excluded by request
from the selection grid, so there is no measurement of where its priors land
relative to its own vanilla floor.

This screen supplies exactly that, and nothing else. One cell, three kinds of
arm, 5 seeds, 2M timesteps:

    true      ground-truth reward, 0 labels        the ceiling
    vanilla   preference RLHF, no prior, q350      the floor
    <prior>   the hand-written prior alone         the candidates

The PPO settings are the ORIGINAL selection-grid ones -- ent_coef 0.01 and NO
target_kl -- because that is what the composition experiment will run on, and a
prior screened under different optimizer settings tells you nothing about where
it will land under these. That matters more than usual here: on the sel2 rerun
target_kl moved Walker2d's vanilla floor from 398 to 3671, which reordered every
prior in the cell. Screen under the settings you will use.

    rows 1 - 40   8 arms x 5 seeds

WHY THESE SIX PRIORS
--------------------
All six Hopper priors already in the registry, no new ones written. They span
the same kinds of wrongness the sel_* families use, and the archive says Hopper
priors tend to come in HIGH -- "partial beats the true reward" was recorded on
Hopper and Walker2d both -- so the weak end of the ladder is where the usable
candidate is most likely to be:

    hopper_capped_low               survive + min(forward, 0.2)   weakest
    hopper_p25                      levels ladder, ~25%
    hopper_capped_forward_survive   survive + min(forward, 1.0)
    hopper_gameable_bounce          survive + 2*|z-velocity|, no forward term
    hopper_p50                      levels ladder, ~50%
    hopper_p75                      levels ladder, ~75%

THE SELECTION RULE for this screen is the one asked for: pick the prior whose
curve lands AT OR BELOW the vanilla floor. That is the opposite of the sel grid's
0.2-0.8 band, and deliberately so -- the composition experiment needs headroom
for the reward model to add something, which a prior that already matches the
true reward would not leave.

Read the raw finals against vanilla, not the partiality column: partiality is
scored against the true arm and answers a different question.

FIVE SEEDS CANNOT TEST ANYTHING (exact two-sided Wilcoxon floor at n=5 is
0.0625). This is a screen for ordering, not a comparison.
"""

from pathlib import Path

SEEDS = range(5)

# Bare names resolve the MODULE, so a partial living inside a multi-partial file
# needs the file:name form. hopper_capped_low sits in mujoco_capped_low.py; the
# other five are one-per-module. Every one of these is checked at submit time by
# run_hop.sh before a queue slot is burned.
PRIORS = (
    "mujoco_capped_low:hopper_capped_low",
    "hopper_levels:hopper_p25",
    "hopper_capped_forward_survive",
    "hopper_gameable_bounce",
    "hopper_levels:hopper_p50",
    "hopper_levels:hopper_p75",
)

lines: list[str] = []
for seed in SEEDS:
    lines.append(f"hopper true {seed}")
for seed in SEEDS:
    lines.append(f"hopper vanilla {seed}")
for prior in PRIORS:
    # the params file carries the bare variant name; run_hop.sh maps it back to
    # the full file:name reference, so log dir names stay readable
    variant = prior.split(":")[-1]
    for seed in SEEDS:
        lines.append(f"hopper {variant} {seed}")

out = Path(__file__).with_name("params_hop.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
print(f"  rows 1-{len(SEEDS)}: true")
print(f"  rows {len(SEEDS) + 1}-{2 * len(SEEDS)}: vanilla")
print(f"  rows {2 * len(SEEDS) + 1}-{len(lines)}: {len(PRIORS)} priors x {len(SEEDS)} seeds")
