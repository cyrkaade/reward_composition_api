"""Generate jobs/params_sel2.txt (CELL VARIANT SEED) for the SELECTION GRID RERUN.

Same 280 rows as the original selection grid -- 8 cells x (true + vanilla + 5
priors) x 5 seeds -- rerun at 2M with the PPO settings the entkl experiment
picked out. See jobs/run_sel2.sh for what changed and why.

The row layout is deliberately identical to params_sel.txt so a row number means
the same thing in both grids and the two can be compared line by line.

    rows   1 -  40   true     8 cells x 5 seeds
    rows  41 -  80   vanilla  8 cells x 5 seeds
    rows  81 - 280   priors   8 cells x 5 priors x 5 seeds

WHY THE RERUN
-------------
All 280 runs of the first grid carried ent_coef 0.01 and stock log_std_init 0.
Measured on the finished runs, the policy's action noise ran away in six of the
seven continuous-action cells (final log_std: standup +11.2, ant +5.8, walker
+3.1, cheetah +2.9, against an init of 0.0), and the true arms on Walker2d,
HumanoidStandup and Ant lost most of their peak before the end. Every prior in
those cells was judged against a ceiling that had fallen over, so the verdicts
cannot be trusted -- including "prior beats the true arm", which was the first
grid's headline finding on Walker2d and HalfCheetah.

The entkl experiment (80 runs, 8 cells x 2 arms x 5 seeds at 2M, mode=true)
tested two candidate repairs against the archived control:

    ent_coef 0.01 -> 0     did NOT fix it, and hurt cheetah, pusher and ant
    + target_kl 0.03       fixed it, 5/5 seeds on all three broken cells

    cell        ctrl final -> kl final    drawdown ctrl -> kl   log_std kl
    cheetah     1539 -> 3675  (5/5)       0.08 -> 0.03          +0.06
    walker       904 -> 3675  (5/5)       0.72 -> 0.23          +0.89
    standup    85735 -> 168227 (5/5)      0.48 -> 0.00          +0.43
    ll           246 -> 273   (4/5)       0.16 -> 0.06          n/a discrete

target_kl also held log_std near its initial value WITH ent_coef 0.01 still on,
which is why ent_coef is kept: the runaway was driven by oversized policy
updates, not by the entropy bonus itself. Turning the bonus off instead drove
log_std to about -2 and under-explored.

ANT IS THE EXCEPTION and keeps a second change. target_kl moved it from -14 to
+13, but its peak is still at the FIRST eval point and it still gives up 98% of
it, i.e. it stops harming itself without ever learning. Ant's problem is that
at the stock log_std_init 0 the expected control cost is 2.06/step against a
+1.00/step alive bonus, so staying alive loses money from the first rollout and
- Ant-v5 charging no terminal penalty - dying is the cheapest way out. Measured
at sigma=0.135 (log_std_init -2) the control cost is 0.07/step and the sign
flips. A local 2x2 probe at 300k reached 676-724 final and 1279-1528 peak with
~760-step episodes, against -14.5 and 8.5-step episodes on the old settings.
"""

from pathlib import Path

SEEDS = range(5)

# cell -> the five candidate priors, as the file:name form --partial requires.
# A bare name resolves the MODULE, so every one of these MUST carry the colon.
# Identical to make_params_sel.py; the priors are not what is being changed.
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

out = Path(__file__).with_name("params_sel2.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
print(f"  rows 1-{len(CELLS) * len(SEEDS)}: true")
print(f"  rows {len(CELLS) * len(SEEDS) + 1}-{2 * len(CELLS) * len(SEEDS)}: vanilla")
print(f"  rows {2 * len(CELLS) * len(SEEDS) + 1}-{len(lines)}: priors")

reference = Path(__file__).with_name("params_sel.txt")
if reference.exists():
    same = reference.read_text(encoding="utf-8").split() == out.read_text(encoding="utf-8").split()
    print(f"  rows identical to params_sel.txt: {same}")
