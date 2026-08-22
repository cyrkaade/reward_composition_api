"""Re-lay the weighted-sum figure as two landscape pages.

Same curves, same colors, same data as `analyze_wsum.py` -- only the grid is
different.  The 8x2 portrait figure (one row per environment, one column per
budget) is split in half and transposed, so each page is 2 rows x 4 columns:
rows are the query budgets (200, 400), columns are four environments.

With `--with-partial-only` each panel also gets the hand-written prior trained
alone, with zero preference labels, taken from the screening family that chose
that prior.  See `load_partial_only` for how the arm is matched and what differs.

Usage:
    python jobs/plot_wsum_book.py [--logs logs] [--out logs/wsum_book]
    python jobs/plot_wsum_book.py --with-partial-only     # adds the prior-alone line
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

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

# Cyan is the one hue the existing figure does not already use.  Checked against
# every colour in BASE_STYLE and ALPHA_COLOR with the dataviz palette validator:
# worst-case CVD separation 12.8 (target >= 8).  It sits below the 3:1 contrast
# line on white, which the thick dash-dot stroke and the legend entry cover.
PARTIAL_ONLY_COLOR = "#00B3C6"
PARTIAL_ONLY_STYLE = (0, (5, 1.4, 1.4, 1.4))


def load_partial_only(root: Path, cells, grouped, prefix: str = "wsum"):
    """the same hand-written prior, trained alone with zero labels

    Matched, not assumed: the `partial_reference` of each wsum cell is read from
    that cell's own metadata, and the prior-alone runs are the ones that carry
    the identical (env_id, partial_reference) with `mode == "partial"`.

    The PPO configuration of the two families is identical -- n_envs 8,
    n_steps 256, ent_coef 0.01, target_kl 0.03, env_normalize auto, final_policy
    last -- and both evaluate every 20k steps from 20k, so the x grids coincide.
    Two things do differ and are handled here or reported by the caller:
      * the screening runs are 2M timesteps, so curves are TRUNCATED to the
        wsum x-range before being drawn;
      * they have 5 seeds (4 on Swimmer) against wsum's 10, so the panel legend
        prints each arm's own n.
    """
    want = {}
    for cell in cells:
        for meta_path in sorted(root.glob(f"{prefix}_{cell}_*/*/metadata.json")):
            try:
                d = json.loads(meta_path.read_text())
            except Exception:
                continue
            if d.get("partial_reference") and d.get("env_id"):
                want[cell] = (d["env_id"], d["partial_reference"])
                break

    # how far the wsum curves of that cell actually run, so the prior-alone arm
    # is cut to the same window rather than stretching the axis
    xmax = {}
    for cell in cells:
        ends = [r["steps"][-1] for k, rs in grouped.items() if k[0] == cell
                for r in rs if r["steps"] is not None and len(r["steps"])]
        xmax[cell] = max(ends) if ends else None

    found = defaultdict(list)
    for meta_path in sorted(root.glob("*/*/metadata.json")):
        fam = meta_path.parent.parent.name
        if fam.startswith(prefix):
            continue
        try:
            d = json.loads(meta_path.read_text())
        except Exception:
            continue
        if d.get("mode") != "partial":
            continue
        key = (d.get("env_id"), d.get("partial_reference"))
        cell = next((c for c in cells if want.get(c) == key), None)
        if cell is None:
            continue
        npz = meta_path.parent / "eval" / "evaluations.npz"
        if not npz.exists():
            continue
        try:
            data = np.load(npz)
            steps = np.asarray(data["timesteps"], dtype=float)
            curve = np.asarray(data["results"], dtype=float).mean(axis=1)
        except Exception:
            continue
        if xmax[cell] is not None:
            m = steps <= xmax[cell] + 1e-6
            steps, curve = steps[m], curve[m]
        found[cell].append((steps, curve, fam))

    for cell in cells:
        ref = want.get(cell, (None, None))[1]
        fams = sorted({f for _, _, f in found.get(cell, [])})
        print(f"  {cell:9s} partial={ref}  n={len(found.get(cell, []))}  {fams}")
    return {c: [(s, cu) for s, cu, _ in v] for c, v in found.items()}, want


def draw_page(grouped, cells, budgets, out_png, page_no, n_pages, partial_only=None):
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
            # the prior alone, zero labels -- one arm, so the same line is drawn
            # on both budget rows
            got = band(partial_only.get(cell, [])) if partial_only else None
            if got:
                steps, med, lo, hi, n = got
                ax.plot(steps, med, color=PARTIAL_ONLY_COLOR,
                        linestyle=PARTIAL_ONLY_STYLE, linewidth=2.4,
                        label=f"partial only, 0 labels (n={n})", zorder=5)
                ax.fill_between(steps, lo, hi, color=PARTIAL_ONLY_COLOR,
                                alpha=0.12, linewidth=0)
            ax.set_title(CELL_TITLES.get(cell, cell), fontsize=11)
            ax.grid(alpha=0.25)
            ax.set_xlabel("policy timesteps")
            if c == 0:
                # the row is the budget, so it is named on the left edge
                ax.set_ylabel(f"{budget} queries\ntrue episode return", fontsize=10)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best")

    sub = "median over seeds, shaded IQR"
    if partial_only:
        sub += ("   |   partial only = the same hand-written prior trained alone with "
                "zero labels, from the screening runs\n(2M truncated to this window, "
                "5 seeds vs 10; PPO config matched)")
    fig.suptitle(
        "Normalized weighted sum: alpha*norm(partial) + (1-alpha)*norm(model)\n"
        f"{sub}  (page {page_no} of {n_pages})",
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
    ap.add_argument("--with-partial-only", action="store_true",
                    help="also draw the hand-written prior trained alone (0 labels); "
                         "writes to <out>_with_partial_only_<page>.png")
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
    partial_only = None
    suffix = ""
    if args.with_partial_only:
        print("matching the prior-alone arm for each cell:")
        partial_only, _ = load_partial_only(root, cells, grouped, args.prefix)
        missing = [c for c in cells if not partial_only.get(c)]
        if missing:
            print(f"WARNING: no prior-alone run matched for {missing}")
        suffix = "_with_partial_only"
    for i, page_cells in enumerate(pages, start=1):
        draw_page(grouped, page_cells, budgets,
                  f"{out_base}{suffix}_{i}.png", i, len(pages),
                  partial_only=partial_only)


if __name__ == "__main__":
    main()
