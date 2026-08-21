"""Parameter file for the weighted-sum alpha sweep.

Four training rewards are compared on every environment:
  true          ground-truth env reward, no labels (the ceiling)
  vanilla       learned reward model only (mode=feedback), the zero-prior floor
  naive         partial + learned model, unnormalized (the existing baseline)
  ws<alpha>     alpha*norm(partial) + (1-alpha)*norm(model), alpha in {.2,.4,.6,.8}

`true` carries no query budget, so it is emitted once per cell rather than once
per budget.  Every other arm is emitted at both budgets.  Seeds are shared
across all arms so every contrast is paired.
"""

from pathlib import Path

SEEDS = list(range(10))
ALPHAS = ["0.20", "0.40", "0.60", "0.80"]

# cell -> (partial reference, low budget, high budget)
CELLS = {
    "ll":      ("reasonable_partials:rll_tilt50",      200, 400),
    "ant":     ("reasonable_partials:rant_cap30_ctrl05", 200, 400),
    "hopper":  ("reasonable_partials:rhop_cap80",      200, 400),
    "reacher": ("reasonable_partials:rrch_x_ctrl050",  200, 400),
    "bipedal": ("reasonable_partials:rbw_cap75",       200, 400),
}


def rows():
    for cell, (partial, low, high) in CELLS.items():
        for seed in SEEDS:
            # the ceiling uses no labels, so it is budget-independent
            yield cell, "true", seed, "-", 0, "1.0"
        for budget in (low, high):
            for seed in SEEDS:
                yield cell, f"vanilla_q{budget}", seed, "-", budget, "1.0"
                yield cell, f"naive_q{budget}", seed, partial, budget, "1.0"
                for alpha in ALPHAS:
                    tag = f"ws{alpha.replace('.', '')[:3]}"
                    yield cell, f"{tag}_q{budget}", seed, partial, budget, alpha


def main():
    out = Path(__file__).with_name("params_wsum.txt")
    lines = [" ".join(str(x) for x in row) for row in rows()]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} with {len(lines)} rows")
    arms = sorted({l.split()[1] for l in lines})
    print(f"{len(CELLS)} cells x {len(arms)} arms x {len(SEEDS)} seeds")
    print("arms:", " ".join(arms))


if __name__ == "__main__":
    main()
