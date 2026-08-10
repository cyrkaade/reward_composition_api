"""Generate jobs/params_pre2.txt.

One line per run: CELL VARIANT BUDGET SEED. Run this, check `wc -l`, then sbatch
jobs/run_pre2.sh. See run_pre2.sh for what each variant means.

The file is ORDERED so a prefix of it is a usable experiment on its own:

    lines    1 - 200   E0  references: true / partial / vanilla feedback, 4 envs
    lines  201 - 800   E1+E2+E3  the main grid (pretraining x active learning x budget)
    lines  801 - 1000  E4..E7    ablations

so `sbatch --array=1-800 jobs/run_pre2.sh` runs everything the paper needs and
defers the ablations, and a later `sbatch jobs/run_pre2.sh` fills the rest in
(every arm skips runs whose metadata.json already exists).
"""

from pathlib import Path

SEEDS = range(10)
LADDER = (175, 350, 700)
ALL_ENVS = ("ll", "walker", "pusher", "reacher")
# LunarLander and Walker2d are the two envs where PPO-on-true is stable but the
# learned reward collapses, so they are the ones the mechanism ablations need.
COLLAPSE_ENVS = ("ll", "walker")

lines: list[str] = []


def add(cells, variants, budgets, seeds=SEEDS):
    for cell in cells:
        for variant in variants:
            for budget in budgets:
                for seed in seeds:
                    lines.append(f"{cell} {variant} {budget} {seed}")


# --------------------------------------------------------------- E0 references
# Everything now runs --tuned-hyperparams, which materially changes LunarLander
# (5 keys) and Walker2d (8 keys), so the archived baselines for those two no
# longer apply and the floor/ceiling has to be re-measured here. Reacher and
# Pusher are unchanged by the flag (0 differing keys, verified) but are included
# so the whole comparison lives in one place under one config.
# The BUDGET column is ignored for true/partial; 350 is a placeholder. 80
add(ALL_ENVS, ("true", "partial"), (350,))

# Vanilla RLHF, the arm the prior is supposed to beat. Full ladder. 120
add(ALL_ENVS, ("fb_al",), LADDER)

# ------------------------------------------- E1 + E2 + E3: one grid, three reads
# pretraining objective {none, mse, bt} x active learning {on, off} x budget.
# mse_noal is dropped: MSE is the broken baseline, and whether active learning
# rescues a broken initialisation is not a question the paper asks. 600
add(ALL_ENVS, ("none_al", "none_noal", "mse_al", "bt_al", "bt_noal"), LADDER)

# ----------------------------------------------------------------- E4..E7
# E4 PEBBLE's real active learning: ensemble of 3 + disagreement, all 4 envs
# because this is the gate for whether the AL claim is worth pursuing at all. 80
add(ALL_ENVS, ("none_ens3", "bt_ens3"), (350,))

# E5 the pretrain/query data leak, A/B against the holdout arms. 40
add(COLLAPSE_ENVS, ("bt_leak",), (175, 350))

# E6 cheating ceiling: pretrain on TRUE-reward labels. Bounds how much of M3's
# failure is "the partial ranks badly" vs "an init cannot beat a persistent
# prior". 40
add(COLLAPSE_ENVS, ("true_al",), (175, 350))

# E7 tanh-bounded reward model, on the two envs where collapse actually happens
# (a bounding ablation is meaningless where nothing runs away). 40
add(COLLAPSE_ENVS, ("tanh_naive", "tanh_feedback"), (350,))

out = Path(__file__).with_name("params_pre2.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
