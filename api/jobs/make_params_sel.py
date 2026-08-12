"""Generate jobs/params_sel.txt (CELL VARIANT SEED) for the SELECTION grid.

WHAT THIS GRID IS FOR
---------------------
Pick the environments and the hand-written priors that later experiments will run
on. It answers one question per (env, prior) pair, with three curves per env:

    true      training on the ground-truth reward           = the ceiling
    vanilla   preference-based RLHF, no prior, q350         = the floor
    <prior>   training on the hand-written prior alone      = the candidate

A prior qualifies if its true-reward curve sits CLEARLY BELOW the ceiling and AT
OR ABOVE the floor. Nothing here composes a prior with a reward model - that is
the next experiment; this one only measures where the candidate lines land.

WHY FIVE PRIORS PER ENVIRONMENT
-------------------------------
Guessing which prior is weak does not work in this project. Measured over 40
random episodes the fragment-ranking agreement of the LunarLander ladder was p10
62%, p25 52%, approach 60% - not even monotone, so random-policy statistics
cannot order priors by strength. Worse, two archived priors (hopper and walker2d
"survive + forward") turned out to produce BETTER true-reward policies than
training on the true reward. The only reliable readout is the partial-only arm
itself, and it costs no labels, so all five run and the choice is made from the
result rather than in advance.

The five per environment are deliberately different KINDS of wrong, not five
strengths of one thing:

    saturation      credit stops accruing past a cap        (two cap levels)
    omission        a whole true-reward term deleted
    axis / proxy    a surrogate objective, or one axis of the real one
    over-specified  a target the designer picked, penalized on both sides
    over-priced     a real cost term charged far above its true weight

SCALING IS NOT A LEVER. Multiplying every weight in a prior by c is EXACTLY
`--partial-alpha c`, because the prior is linear in its weights and naive mode
computes alpha * partial. On top of that the training env is wrapped in
VecNormalize(norm_reward=True), so a uniform rescale is normalized straight back
out in mode=partial. Every candidate here is therefore a nonlinear or
structurally different function, never a rescale.

THE EIGHT ENVIRONMENTS
----------------------
LunarLander-v3 is required. The other seven are what is left of the MuJoCo suite
after removing Hopper (excluded by request) and three that cannot support a
graded prior at all:

    InvertedPendulum-v5        ceiling is "stay alive"; the alive bonus IS the
                               reward, so any prior that works scores ~100%
    InvertedDoublePendulum-v5  same shape - measured, the alive bonus is 10/step
                               against a 9.35/step optimum, so the reward is
                               effectively binary and no prior lands mid-band
    Humanoid-v5                PPO barely leaves the alive bonus at 1M steps

Two of the eight carry known risks, and the mode=true arm here IS the test of
them rather than an assumption:

    HalfCheetah-v5   the archive records ~45% of seeds learning to run and the
                     rest not, which makes its medians mode-counting. All 399
                     archived HalfCheetah runs have tuned_hyperparams unset and
                     NOT ONE is a mode=true run, so the ceiling has never been
                     measured on this environment at all.
    Walker2d-v5      PRE4 true-arm range was 799-6263 over 10 seeds with 3 below
                     2500, and its archived prior beat the true reward.

    HumanoidStandup-v5 is a milder version of the same caution: it never
                     terminates, so it is the lowest-variance cell here, but a
                     random policy already scores 32.6k of a ceiling around
                     70-100k, so its dynamic range is narrow.

Read the true-arm seed spread in analyze_sel.py before trusting any contrast on
those three.

LAYOUT
------
    rows   1 -  40   true     8 cells x 5 seeds
    rows  41 -  80   vanilla  8 cells x 5 seeds
    rows  81 - 280   priors   8 cells x 5 priors x 5 seeds

The references come first so `--array=1-80` gives both bounding curves on their
own if the queue is tight; `--array=1-280` runs everything at once.
"""

from pathlib import Path

SEEDS = range(5)

# cell -> the five candidate priors, as the file:name form --partial requires.
# A bare name resolves the MODULE, so every one of these MUST carry the colon.
PRIORS = {
    "ll": ("sel_lunarlander", ("sll_pad", "sll_pad_speed", "sll_pad_fuel", "sll_touchdown", "sll_upright")),
    "reacher": ("sel_reacher", ("srch_dist", "srch_dist_cap", "srch_x", "srch_quad", "srch_hit")),
    "pusher": ("sel_pusher", ("spsh_goal_reach", "spsh_reach", "spsh_goal_cap", "spsh_goal", "spsh_timid")),
    "swimmer": ("sel_swimmer", ("sswm_cap10", "sswm_cap18", "sswm_speed", "sswm_straight", "sswm_timid")),
    "cheetah": ("sel_halfcheetah", ("shc_cap10", "shc_cap20", "shc_forward", "shc_timid", "shc_level")),
    "walker": ("sel_walker2d", ("swk_cap05", "swk_cap15", "swk_stand", "swk_target", "swk_timid")),
    "ant": ("sel_ant", ("sant_cap03", "sant_cap10", "sant_radial", "sant_stand", "sant_timid")),
    "standup": ("sel_humanoidstandup", ("shs_cap60", "shs_cap120", "shs_rise", "shs_crouch", "shs_timid")),
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

out = Path(__file__).with_name("params_sel.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
print(f"  rows 1-{len(CELLS) * len(SEEDS)}: true")
print(f"  rows {len(CELLS) * len(SEEDS) + 1}-{2 * len(CELLS) * len(SEEDS)}: vanilla")
print(f"  rows {2 * len(CELLS) * len(SEEDS) + 1}-{len(lines)}: priors")
