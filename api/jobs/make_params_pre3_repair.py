"""Generate the ordered 40-run parameter file for ``run_pre3_repair.sh``.

Run from ``api/``::

    python jobs/make_params_pre3_repair.py
    wc -l jobs/params_pre3_repair.txt   # must print 40

Lines 1--20 are the primary, bounded standard-model repair.  Lines 21--40
are a fallback bridge to the historical reward-model recipe and must not be
submitted unless the primary analyzer says to do so.
"""

from pathlib import Path


SEEDS = range(10)

lines: list[str] = []


def add(variants: tuple[str, ...]) -> None:
    for variant in variants:
        for seed in SEEDS:
            lines.append(f"ll {variant} {seed}")


# Lines 1--20: standard 3x256/e3/tanh model with B-Pref's 97% training-
# accuracy stopping rule.  Feedback and naive differ only by composition.
add(("adaptive_feedback", "adaptive_naive"))

# Lines 21--40: exact historical reward-model architecture/training fallback,
# while retaining the fair tuned-PPO, separate-data, dedicated-query protocol.
add(("legacy_feedback", "legacy_naive"))

assert len(lines) == 40
assert lines[0] == "ll adaptive_feedback 0"
assert lines[9] == "ll adaptive_feedback 9"
assert lines[10] == "ll adaptive_naive 0"
assert lines[19] == "ll adaptive_naive 9"
assert lines[20] == "ll legacy_feedback 0"
assert lines[-1] == "ll legacy_naive 9"

out = Path(__file__).with_name("params_pre3_repair.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
