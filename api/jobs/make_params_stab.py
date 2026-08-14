"""Generate jobs/params_stab.txt (CELL TREATMENT ARM SEED) for the STABILITY pilot.

WHAT THIS PILOT IS FOR
----------------------
Three of the selection grid's eight environments have a broken ceiling, and the
break has nothing to do with the priors being tested - it is there in `mode=true`
with no reward model involved at all. Everything downstream is measured against
that ceiling, so a difference between composition arms could just be reporting
which failure mode each one landed in. Two candidate fixes are tested here, each
isolated to one variable, at the true/partial level, before the much larger
composition experiment is built on top.

MEASURED FROM THE 1M SELECTION GRID (api/logs/sel_*, 5 seeds each), which is what
the fixes are aimed at:

    HalfCheetah-v5  true  1044 1046 1326 1498 1853   (final, per seed)
                    -> UNIMODAL AND STUCK, not the 45/55 mode split the archive
                       recorded for its non-true arms. It is BELOW its own prior
                       arm (shc_forward median 1586) and below its own vanilla
                       arm (median 2058), so the premise "true is the ceiling"
                       fails outright here. The plateau is reached by ~200k and
                       the last quarter is 163 BELOW the quarter before it in 5/5
                       seeds, so this is a converged local optimum rather than a
                       run that needs more time; the gait is real but slow (1.24
                       and 1.73 m/s on seeds 3 and 0) and action noise settled at
                       std 0.13-0.16, so exploration did not collapse either.
                       The mode split is real but lives in the OTHER arms:
                       shc_forward 1324/1560/1586 | 3143/3543.

    Walker2d-v5     true  peak 1809-4603 -> final 1204-2228, three of five seeds
                    losing more than half their peak (-52%, -61%, -65%).
                    swk_cap15 beats it on final in 5/5 seeds (2152-3037) while
                    its peaks are nearly equal (3355 vs 3140), so the premise
                    fails through DRAWDOWN specifically. This is the one cell
                    where the measured failure matches the cliff story.

    Ant-v5          true  peak 273-461 AT THE FIRST EVAL POINT (20k steps), then
                    monotone decay to a NEGATIVE final (-14 to -55) in 5/5 seeds.
                    There is no ceiling to stabilize: the "peak" is the stand-
                    still local optimum found before any real learning. Ant is
                    kept because removing termination changes it more than any
                    other cell (episodes go from 76 steps to a fixed 1000), but
                    it is a repair attempt on a dead arm, not a stabilization of
                    a working one, and must be reported as such.

THE TWO TREATMENTS
------------------
    sde     use_sde=True, nothing else. Exploration noise goes from independent
            per-step samples to gSDE's temporally correlated, state-dependent
            noise. sde_sample_freq is left at SB3's default -1, which resamples
            the noise matrix once per rollout, i.e. every n_steps=256 steps.
            No environment or reward change, so evaluation is untouched and no
            train/eval mismatch exists. Note the preset file pairs use_sde
            with sde_sample_freq 4 in every block where it uses it at all, and
            has no gSDE block for Walker2d or HalfCheetah - -1 is SB3's default,
            not a tuned value.

    noterm  --unhealthy-penalty 1.0 on the TRAINING env only: stop terminating
            when unhealthy, and replace the boolean +1/0 healthy bonus with a
            persistent +1/-1. Only for the two envs with an is_healthy rule.
            HalfCheetah never terminates, so the treatment is undefined there.

WHY EVERY CONTROL IS RE-RUN HERE RATHER THAN TAKEN FROM logs/sel_*
------------------------------------------------------------------
The archived sel_* arms ran at 1M with policy_learning_kwargs {n_steps: 256} and
NO ent_coef (verified from their metadata.json); the grid has since been
reconfigured to 2M with ent_coef 0.01. Reusing them would put ent_coef in the
comparison alongside the variable under test, at 5 seeds against 10. PRE4 was
already read wrong once this way - two of its blocks were compared against a
baseline that also differed in reward_model_batch_size and dedicated_query_rng -
so the controls are re-run here at 10 seeds, in the same submission, from the
same checkout. The base config is the one the 2M rerun will ship:
{n_steps: 256, ent_coef: 0.01}.

Ten seeds because five cannot test anything: the exact two-sided Wilcoxon floor
at n=5 is 1/2^5 = 0.031 one-sided, and CLAUDE.md's own rule is that a 5-seed
pilot estimates an effect size but cannot test one.

LAYOUT
------
    rows   1 -  60   ctrl    6 arms x 10 seeds   (the baseline all contrasts use)
    rows  61 - 100   sde     4 arms x 10 seeds   (experiment 1)
    rows 101 - 140   noterm  4 arms x 10 seeds   (experiment 2)

Controls come first so `--array=1-60` gives every baseline on its own if the
queue is tight; `--array=1-140` runs everything.
"""

from pathlib import Path

SEEDS = range(10)

# cell -> the single prior that cell's partial arm uses, in the file:name form
# --partial requires (a bare name resolves the MODULE, not the partial).
PARTIAL = {
    "cheetah": "sel_halfcheetah:shc_forward",
    "walker": "sel_walker2d:swk_cap15",
    "ant": "sel_ant:sant_cap10",
}

# treatment -> the cells it applies to. `noterm` needs an is_healthy rule, which
# HalfCheetah-v5 does not have (it never terminates), so it is walker/ant only.
TREATMENTS = {
    "ctrl": ("cheetah", "walker", "ant"),
    "sde": ("cheetah", "walker"),
    "noterm": ("walker", "ant"),
}

lines: list[str] = []
for treatment, cells in TREATMENTS.items():
    for cell in cells:
        for arm in ("true", PARTIAL[cell].split(":")[1]):
            for seed in SEEDS:
                lines.append(f"{cell} {treatment} {arm} {seed}")

out = Path(__file__).with_name("params_stab.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"wrote {out} with {len(lines)} lines")
start = 1
for treatment, cells in TREATMENTS.items():
    count = len(cells) * 2 * len(SEEDS)
    print(f"  rows {start}-{start + count - 1}: {treatment} ({', '.join(cells)})")
    start += count
