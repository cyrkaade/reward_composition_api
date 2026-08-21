"""Emit params_wsum_atari.txt: the weighted-sum alpha sweep on the two Atari cells.

Mirrors make_params_wsum.py exactly -- same 13 arms, same 10 seeds, seed-major
so a truncated run still yields a complete figure at fewer seeds rather than a
few finished environments and empty panels.  Only the cells, the partials and
the two query budgets differ.

Row format:  CELL ARM SEED PARTIAL BUDGET ALPHA
"""

from pathlib import Path

CELLS = {
    "mspacman": "reasonable_atari_partials:rmsp_pellets",
    "qbert": "reasonable_atari_partials:rqb_visit02",
}

# Atari budgets, an order of magnitude above the non-Atari grid: pixel
# observations give the reward model far less to generalize from per label.
BUDGETS = (2800, 5600)
ALPHAS = (0.20, 0.40, 0.60, 0.80)
SEEDS = range(10)


def arms():
    """The 13 arms per cell: true, then vanilla/naive/4 alphas at each budget."""
    yield ("true", 0, 1.0)
    for budget in BUDGETS:
        yield (f"vanilla_q{budget}", budget, 1.0)
        yield (f"naive_q{budget}", budget, 1.0)
        for alpha in ALPHAS:
            yield (f"ws{int(alpha * 100):03d}_q{budget}", budget, alpha)


def main() -> None:
    rows = []
    for seed in SEEDS:  # seed-major
        for cell, partial in CELLS.items():
            for arm, budget, alpha in arms():
                reference = "-" if arm == "true" else partial
                rows.append(f"{cell} {arm} {seed} {reference} {budget} {alpha:.2f}")

    out = Path(__file__).with_name("params_wsum_atari.txt")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")

    per_seed = len(rows) // len(SEEDS)
    print(f"wrote {out} with {len(rows)} rows ({per_seed} per seed layer)")
    print(f"  cells={len(CELLS)} arms={per_seed // len(CELLS)} seeds={len(SEEDS)}")
    print(f"  first row: {rows[0]}")
    print(f"  row {per_seed + 1} (start of seed 1): {rows[per_seed]}")


if __name__ == "__main__":
    main()
