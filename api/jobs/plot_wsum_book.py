"""Re-lay the weighted-sum figure as two landscape pages.

Same curves, same colors, same data as `analyze_wsum.py` -- only the grid is
different.  The 8x2 portrait figure (one row per environment, one column per
budget) is split in half and transposed, so each page is 2 rows x 4 columns:
rows are the query budgets (200, 400), columns are four environments.

Usage:
    python jobs/plot_wsum_book.py [--logs logs] [--out logs/wsum_book]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from analyze_wsum import (
    ALPHA_COLOR,
    ALPHA_LABEL,
    ALPHAS,
    BASE_STYLE,
    CELL_TITLES,
    CELLS,
    band,
    load_runs,
    split_arm,
)


def draw_page(grouped, cells, budgets, out_png, page_no, n_pages):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = len(budgets), len(cells)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.6 * ncols, 3.9 * nrows), squeeze=False
    )

    for r, budget in enumerate(budgets):
        for c, cell in enumerate(cells):
            ax = axes[r][c]
            for base, (label, color, ls, lw) in BASE_STYLE.items():
                key = (cell, base, None if base == "true" else budget)
                got = band([(x["steps"], x["curve"]) for x in grouped.get(key, [])])
                if not got:
                    continue
                steps, med, lo, hi, n = got
                ax.plot(steps, med, color=color, linestyle=ls, linewidth=lw,
                        label=f"{label} (n={n})")
                ax.fill_between(steps, lo, hi, color=color, alpha=0.12, linewidth=0)
            for a in ALPHAS:
                got = band([(x["steps"], x["curve"])
                            for x in grouped.get((cell, f"ws{a}", budget), [])])
                if not got:
                    continue
                steps, med, lo, hi, n = got
                ax.plot(steps, med, color=ALPHA_COLOR[a], linewidth=1.7,
                        label=f"weighted {ALPHA_LABEL[a]} (n={n})")
                ax.fill_between(steps, lo, hi, color=ALPHA_COLOR[a], alpha=0.10,
                                linewidth=0)
            ax.set_title(CELL_TITLES.get(cell, cell), fontsize=11)
            ax.grid(alpha=0.25)
            ax.set_xlabel("policy timesteps")
            if c == 0:
                # the row is the budget, so it is named on the left edge
                ax.set_ylabel(f"{budget} queries\ntrue episode return", fontsize=10)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Normalized weighted sum: alpha*norm(partial) + (1-alpha)*norm(model)\n"
        f"median over seeds, shaded IQR  (page {page_no} of {n_pages})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}  [{nrows} rows x {ncols} cols: {', '.join(cells)}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="logs/wsum_book")
    ap.add_argument("--prefix", default="wsum")
    ap.add_argument("--per-page", type=int, default=4,
                    help="environments per page (columns)")
    args = ap.parse_args()

    root = Path(args.logs)
    runs = load_runs(root, args.prefix)
    if not runs:
        raise SystemExit(f"no wsum runs found under {root}")

    grouped = defaultdict(list)
    for r in runs:
        base, budget = split_arm(r["arm"])
        grouped[(r["cell"], base, budget)].append(r)

    cells = [c for c in CELLS if any(k[0] == c for k in grouped)]
    budgets = sorted({k[2] for k in grouped if k[2] is not None}) or [0]
    print(f"loaded {len(runs)} runs | cells={cells} | budgets={budgets}")

    pages = [cells[i:i + args.per_page] for i in range(0, len(cells), args.per_page)]
    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for i, page_cells in enumerate(pages, start=1):
        draw_page(grouped, page_cells, budgets,
                  f"{out_base}_{i}.png", i, len(pages))


if __name__ == "__main__":
    main()
