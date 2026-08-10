"""Generate jobs/params_pre2.txt.

One line per run: CELL VARIANT BUDGET SEED. Run this, check `wc -l`, then sbatch
jobs/run_pre2.sh. See run_pre2.sh for what each variant means.
"""

from pathlib import Path

SEEDS = range(10)
LADDER = (175, 350, 700)

lines: list[str] = []


def add(cells, variants, budgets, seeds=SEEDS):
    for cell in cells:
        for variant in variants:
            for budget in budgets:
                for seed in seeds:
                    lines.append(f"{cell} {variant} {budget} {seed}")


# E1+E2+E3 share one grid: pretraining objective x active learning x budget.
# 2 envs x 6 variants x 3 budgets x 10 seeds = 360
add(("ll", "pusher"), ("none_al", "none_noal", "mse_al", "mse_noal", "bt_al", "bt_noal"), LADDER)

# E4 real ensemble disagreement (3 models), vs the size-1 MC-dropout default. 40
add(("ll", "pusher"), ("none_ens3", "bt_ens3"), (350,))

# E5 the pretrain/query data leak, A/B against the holdout arms above. 40
add(("ll", "pusher"), ("bt_leak",), (175, 350))

# E6 cheating ceiling: pretrain on TRUE-reward labels. Bounds how much of M3's
# failure is "the partial ranks badly" vs "an init cannot beat a persistent
# prior". LunarLander only. 20
add(("ll",), ("true_al",), (175, 350))

# E7 tanh-bounded reward model (PEBBLE's). LunarLander only. 20
add(("ll",), ("tanh_naive", "tanh_feedback"), (350,))

# Vanilla RLHF reference. 175 is the new bottom rung; 350 is also run here rather
# than reused from logs/last_* so the tanh_feedback arm has a same-code, same-seed
# unbounded partner to be paired against. 40
add(("ll", "pusher"), ("fb_al",), (175, 350))

# E8 hyperparameter control. The whole grid runs the STOCK PPO config, because
# that is what the ~1,500 archived LunarLander runs use and what the collapse
# numbers are measured on. This cell re-runs the three-arm collapse comparison
# with rl-zoo's tuned LunarLander block (--tuned-hyperparams changes gamma
# 0.99->0.999, n_steps 2048->1024, n_epochs 10->4, gae_lambda 0.95->0.98,
# ent_coef 0->0.01) so "the collapse is a hyperparameter artifact" can be
# answered with data instead of an argument. Pusher is excluded: its preset is
# tuned=False, so the flag is a verified no-op there (0 differing keys). 30
add(("ll",), ("tuned_true", "tuned_feedback", "tuned_naive"), (350,))

out = Path(__file__).with_name("params_pre2.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
