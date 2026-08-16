"""Generate jobs/params_entkl.txt (CELL ARM SEED) for the ENT/KL ceiling repair.

WHY THIS EXISTS
---------------
Every one of the 280 runs in the 2M selection grid carries `ent_coef 0.01`, which
was added to keep exploration alive near Walker2d's instability boundary. Measured
on the finished runs, it does the opposite: the Gaussian policy's log_std runs away
in six of the seven continuous-action cells.

    cell        ctrl weight   final log_std (sigma)      true drawdown
    reacher     1.0           -3.3  (0.04)  shrinks      0.33
    pusher      0.1 x7        -1.3  (0.27)  shrinks      0.09
    swimmer     1e-4          +0.8..+1.8 (2-6)           0.06
    cheetah     0.1 x6        +2.2..+3.8 (9-46)          0.08
    walker      1e-3          +2.4..+4.7 (11-107)        0.72
    ant         0.5 x8        +4.4..+6.9 (81-1022)       1.03
    standup     0.1 x17       +11.0..+11.3 (60k-79k)     0.48

log_std_init is 0.0 everywhere, so every positive number above is drift during
training. The mechanism: diagonal-Gaussian entropy is sum(log sigma), unbounded
above; SB3 clips actions to the action space before stepping the env, so once
sigma is large the environment reward stops responding to further growth while the
entropy gradient stays constant at +ent_coef. Nothing pulls it back. Only Reacher
and Pusher resist, because their control-cost term is large enough relative to the
reward scale to fight the entropy bonus -- and those two are the only cells whose
true arms are clean and converged.

LunarLander-v3 is DISCRETE. Categorical entropy is capped at log(4) = 1.39, so it
cannot diverge there. It is included anyway as the negative control: if the ent0
arm moves LunarLander, the effect is not the runaway mechanism.

THE TWO ARMS
------------
    ent0    ent_coef 0.01 -> 0        removes the entropy bonus outright
    kl      ent_coef 0.01 + target_kl 0.03
            keeps the bonus, and instead caps how far a single PPO update may
            move the policy. SB3 aborts the epoch loop once approximate KL
            exceeds target_kl. This tests whether the damage is the entropy
            bonus itself or the destructive updates it enables.

Each is a SINGLE-VARIABLE change from the control. Nothing else moves.

THE CONTROL IS FREE -- DO NOT RE-RUN IT
---------------------------------------
learning_rate (3e-4) and clip_range (0.2) resolve to plain floats, not schedules
(verified by resolving Suite.ppo_hyperparams), and mode=true is a single learn()
call. So the first 1,000,000 steps of an existing 2M sel_*_true run are identical
to a 1M run at the same seed. evaluations.npz has 100 points at eval_freq 20000,
and index 49 is exactly 1000000 (verified). analyze_entkl.py reads the control
from logs/sel_*_true truncated to index 49 -- matched seeds, matched config,
matched machine. Re-running it would only add machine-to-machine noise.

    rows  1 - 40   ent0   8 cells x 5 seeds
    rows 41 - 80   kl     8 cells x 5 seeds

FIVE SEEDS CANNOT TEST ANYTHING (exact two-sided Wilcoxon floor at n=5 is 0.0625).
This is a screen for effect size and direction, same as the selection grid.
"""

from pathlib import Path

SEEDS = range(5)

CELLS = ("ll", "reacher", "pusher", "swimmer", "cheetah", "walker", "ant", "standup")
ARMS = ("ent0", "kl")

lines: list[str] = []
for arm in ARMS:
    for cell in CELLS:
        for seed in SEEDS:
            lines.append(f"{cell} {arm} {seed}")

out = Path(__file__).with_name("params_entkl.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
block = len(CELLS) * len(SEEDS)
for index, arm in enumerate(ARMS):
    print(f"  rows {index * block + 1}-{(index + 1) * block}: {arm}")
