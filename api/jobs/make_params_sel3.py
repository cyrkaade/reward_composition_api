"""Generate jobs/params_sel3.txt (CELL VARIANT SEED) for the THREE NEW CELLS.

Extends the selection screen to one more Box2D environment and two Atari games,
so the composition experiment is not built entirely on MuJoCo. Same three arms
per cell as the original grid, 5 seeds, 2M timesteps, q350:

    true      ground-truth reward, 0 labels        the ceiling
    vanilla   preference RLHF, no prior, q350      the floor
    <prior>   the hand-written prior alone         the candidate

    rows   1 -  15   true     3 cells x 5 seeds
    rows  16 -  30   vanilla  3 cells x 5 seeds
    rows  31 - 105   priors   3 cells x 5 priors x 5 seeds

WHY THESE THREE ENVIRONMENTS
----------------------------
BipedalWalker-v3 is the only Box2D environment left that can support a graded
prior. LunarLander is already in the grid; CarRacing is image-based, which the
suite is not set up for. Its reward decomposes into exactly three things a
designer can get wrong (forward progress, the upright term, the torque cost)
plus a -100 fall penalty that no prior here reproduces -- see sel_bipedal.py.

MsPacman and Qbert are the two Atari games picked to give the prior the most to
do, and both are new here (the archive already holds Breakout and SpaceInvaders,
20-25 runs each). The argument for both is the same: their true reward is the
raw score, and the score says NOTHING about dying, even though losing a life
costs the agent the entire rest of the episode. That gap is exactly what a
hand-written prior can state and a reward model must infer from 350 fragment
comparisons. Qbert's version is the sharper one -- falling off the pyramid ends
a life instantly -- and MsPacman's is the denser one, so between them they cover
both regimes.

MEASURED on the installed build (ale-py 0.10.1, obs_type="ram", frameskip 4,
sticky actions 0.25), random policy: MsPacman scores a median 190 over a median
449 steps; Qbert 200 over 312. BipedalWalker-v3 runs a median 856 steps (min 50,
max 1600).

KNOWN RISK, STATED UP FRONT: 2M steps is SHORT for Atari. The archive's own
convention for Atari is 25M, and at 2M the true arm may not separate from a
random policy, in which case the ceiling premise fails for those two cells the
way HalfCheetah's did. Read the true-arm rows in the environment table against
the random-policy scores above before trusting any prior contrast here.
"""

from pathlib import Path

SEEDS = range(5)

# cell -> (partial module, the five priors). Bare names resolve the MODULE, so
# every reference run_sel3.sh builds carries the file:name form.
PRIORS = {
    "bipedal": ("sel_bipedal", ("sbw_speed", "sbw_speed_cap", "sbw_upright", "sbw_speed_effort", "sbw_target")),
    "mspacman": ("sel_mspacman", ("smp_score", "smp_score_cap", "smp_life_heavy", "smp_survive", "smp_score_life_time")),
    "qbert": ("sel_qbert", ("sqb_score", "sqb_score_cap", "sqb_life_heavy", "sqb_survive", "sqb_score_life_time")),
}

CELLS = tuple(PRIORS)

lines: list[str] = []

for cell in CELLS:
    for seed in SEEDS:
        lines.append(f"{cell} true {seed}")

for cell in CELLS:
    for seed in SEEDS:
        lines.append(f"{cell} vanilla {seed}")

for cell in CELLS:
    _, names = PRIORS[cell]
    for name in names:
        for seed in SEEDS:
            lines.append(f"{cell} {name} {seed}")

out = Path(__file__).with_name("params_sel3.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
block = len(CELLS) * len(SEEDS)
print(f"  rows 1-{block}: true")
print(f"  rows {block + 1}-{2 * block}: vanilla")
print(f"  rows {2 * block + 1}-{len(lines)}: priors")
