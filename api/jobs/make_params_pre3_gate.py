"""Generate the ordered 100-run parameter file for ``run_pre3_gate.sh``.

Run from ``api/``::

    python jobs/make_params_pre3_gate.py
    wc -l jobs/params_pre3_gate.txt   # must print 100

The order is intentional: lines 1--20 are the LunarLander premise references,
21--60 are the LunarLander reward-model gate, and 61--100 are the Walker2d
reward-model gate.  Every experimental arm has seeds 0--9.
"""

from pathlib import Path


SEEDS = range(10)

lines: list[str] = []


def add(cell: str, variants: tuple[str, ...]) -> None:
    for variant in variants:
        for seed in SEEDS:
            lines.append(f"{cell} {variant} {seed}")


# Lines 1--20: the LunarLander premise under tuned PPO.
add("ll", ("true", "partial"))

# Lines 21--60: change only the learned reward's output bound within each mode.
add(
    "ll",
    (
        "feedback_unbounded",
        "feedback_bounded",
        "naive_unbounded",
        "naive_bounded",
    ),
)

# Lines 61--100: the same standard-reward-model gate on Walker2d.  Walker's
# tuned true/partial premise is being measured by run_e0tuned.sh, not duplicated.
add(
    "walker",
    (
        "feedback_unbounded",
        "feedback_bounded",
        "naive_unbounded",
        "naive_bounded",
    ),
)

assert len(lines) == 100
assert lines[0] == "ll true 0"
assert lines[19] == "ll partial 9"
assert lines[20] == "ll feedback_unbounded 0"
assert lines[59] == "ll naive_bounded 9"
assert lines[60] == "walker feedback_unbounded 0"
assert lines[-1] == "walker naive_bounded 9"

out = Path(__file__).with_name("params_pre3_gate.txt")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(lines)} lines")
