"""Generate jobs/params_comp.txt (CELL VARIANT SEED) for the COMPOSITION grid.

The main experiment: does adding a hand-written prior to a learned reward model
beat the model alone, and does normalizing the two terms before adding them
change the answer.

    true            ground-truth reward, 0 labels          the ceiling
    partial         the hand-written prior alone, 0 labels the prior's own level
    vanilla         learned reward model only              the floor
    naive           partial + RM                           raw sum
    ws<alpha>       norm(partial)*alpha + norm(RM)         normalized sum

Two label budgets per cell, three alphas on the weighted sum, 10 seeds.

    12 variants x 8 cells x 10 seeds = 960 runs

    rows    1 -  80   true      8 cells x 10 seeds
    rows   81 - 160   partial
    rows  161 - 320   vanilla   low and high budget
    rows  321 - 480   naive     low and high budget
    rows  481 - 960   ws05/ws10/ws15 x low and high budget

WHAT IS AND IS NOT VARIED
-------------------------
alpha is varied ONLY on the weighted sum, which is what was asked for. naive
runs at its natural alpha 1.0 with no normalization, so `naive` vs `ws10` is a
clean single-variable contrast: same weight, normalization on or off. `ws05` and
`ws15` then say whether the mixing ratio matters once both terms are on a
comparable scale.

weighted_sum NEEDS NO NEW CODE. rewards/wrapper.py already composes naive as
partial_alpha * transform_partial_reward(p) + transform_model_output(m), so
--mode naive with --normalize-partial-reward --normalize-model-reward
--partial-alpha A is exactly norm(partial)*alpha + norm(RM). Verified end to end:
metadata records partial_reward_mean/std and model_reward_output_mean/std.

THE PRIORS, one per cell, chosen from the sel/sel2/sel3/hop screens
    ll        sel_lunarlander:sll_pad_speed
    pusher    sel_pusher:spsh_goal
    reacher   sel_reacher:srch_dist
    swimmer   sel_swimmer:sswm_straight
    hopper    hopper_levels:hopper_p75
    bipedal   sel_bipedal:sbw_target
    qbert     sel_qbert:sqb_life_heavy
    pong      sel_pong:spg_rally

TWO OF THESE ARE UNSCREENED AND SHOULD BE READ WITH THAT IN MIND.

  pong   spg_rally is new and was never run as a partial-only arm. It was built
         for this grid because Pong is the sparsest cell here: measured on the
         installed build, a random policy scores -21 and only 2.3% of its steps
         carry any reward at all. spg_rally pays a dense bonus per step of rally
         survived and credits winning a point, while never mentioning that
         CONCEDING is bad -- that omitted half is what the reward model has to
         supply. Measured density: signal on 98% of steps against the true
         reward's 2.3%. The `partial` arm in this grid is what screens it.

  hopper hopper_p75 was screened, but WITHOUT target_kl, which this grid uses.
         In that screen Hopper's vanilla arm peaked at 3439 and collapsed to
         1018 (drawdown 0.66), so the floor it was chosen against is the
         collapsed one. target_kl removed exactly that collapse elsewhere
         (Walker2d vanilla went 398 -> 3671), so Hopper's floor will very likely
         come in far higher here and p75 may land below it rather than near it.

TEN SEEDS is the smallest n at which the exact two-sided Wilcoxon can reach
p<0.05 at all: its floor is 2/2^10 = 0.002. Five could not have tested anything.
"""

from pathlib import Path

SEEDS = range(10)

CELLS = ("ll", "pusher", "reacher", "swimmer", "hopper", "bipedal", "qbert", "pong")

# Budget is carried as low/high and mapped per cell in run_comp.sh, because the
# Atari cells run at 2800/5600 and everything else at 350/700.
VARIANTS = (
    "true",
    "partial",
    "vanilla_low", "vanilla_high",
    "naive_low", "naive_high",
    "ws05_low", "ws10_low", "ws15_low",
    "ws05_high", "ws10_high", "ws15_high",
)

lines: list[str] = []
for variant in VARIANTS:
    for cell in CELLS:
        for seed in SEEDS:
            lines.append(f"{cell} {variant} {seed}")

out = Path(__file__).with_name("params_comp.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
block = len(CELLS) * len(SEEDS)
for index, variant in enumerate(VARIANTS):
    print(f"  rows {index * block + 1:4d}-{(index + 1) * block:4d}: {variant}")
