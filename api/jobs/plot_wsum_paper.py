"""Three paper figures from the weighted-sum grid.

Reads exactly the same runs as `analyze_wsum.py` and changes none of them.
Everything here is derived from two per-run scalars:

  tail5  mean of the last 5 evaluation points (the analyzer's own final metric)
  peak5  max over rolling 5-point means of the evaluation curve, so peak and
         final are computed the same way and a flat noisy curve does not
         manufacture a drawdown

Per environment those are put on a common scale:

  normalized = (x - floor) / (ceiling - floor)

  floor    median first eval point over every run of that cell (untrained policy)
  ceiling  median tail5 of the `true` arm (training on the ground-truth reward)

so 0 = untrained and 1 = the ground-truth-reward ceiling, and the eight
environments can be read on one axis despite spanning -50 to 4000 raw return.

Figures
  1  <out>_alpha_law.png     normalized return vs alpha, one panel per env
  2  <out>_summary.png       env x arm heatmap of normalized return, both budgets
  3  <out>_label_efficiency  prior at 200 labels vs zero-prior RLHF at 400

Usage:
    python jobs/plot_wsum_paper.py [--logs logs] [--out logs/wsum_paper]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from analyze_wsum import ALPHAS, CELL_TITLES, CELLS, load_runs, split_arm

# Validated all-pairs against the dataviz palette checker (light surface):
# CVD separation, normal-vision floor, chroma and contrast all PASS.
# Keeps the green=zero-prior / orange=naive roles of the existing curve figure.
C_WS = "#005F8C"       # weighted sum
C_WS_LIGHT = "#6BAED6"  # weighted sum, smaller budget (same hue, lighter step)
C_NAIVE = "#D55E00"    # naive (partial + model)
C_VANILLA = "#009E73"  # vanilla RLHF, no prior
C_TRUE = "#111111"     # ground-truth reward ceiling
INK = "#222222"
MUTED = "#666666"

ALPHA_VALS = {"020": 0.2, "040": 0.4, "050": 0.5, "060": 0.6, "080": 0.8}
ARM_ORDER = ["true", "vanilla", "naive"] + [f"ws{a}" for a in ALPHAS]
ARM_LABEL = {
    "true": "true\nreward",
    "vanilla": "vanilla\nRLHF",
    "naive": "naive",
    **{f"ws{a}": "ws\n" + r"$\alpha$" + f"={ALPHA_VALS[a]:g}" for a in ALPHAS},
}
# the alpha we quote in figure 3, fixed up front rather than picked per env
HEADLINE_ALPHA = "080"


def peak5(curve):
    """max over rolling 5-point means, so peak is measured like tail5"""
    if curve is None or len(curve) < 5:
        return None
    k = np.convolve(np.asarray(curve, dtype=float), np.ones(5) / 5.0, mode="valid")
    return float(k.max())


def collect(logs: Path, prefix: str):
    runs = load_runs(logs, prefix)
    if not runs:
        raise SystemExit(f"no runs found under {logs} with prefix {prefix}")
    g = defaultdict(list)
    for r in runs:
        base, budget = split_arm(r["arm"])
        r["peak5"] = peak5(r["curve"])
        g[(r["cell"], base, budget)].append(r)
    cells = [c for c in CELLS if any(k[0] == c for k in g)]
    budgets = sorted({k[2] for k in g if k[2] is not None})

    scale = {}
    for cell in cells:
        allr = [r for k, rs in g.items() if k[0] == cell for r in rs]
        floor = float(np.median([float(r["curve"][0]) for r in allr
                                 if r["curve"] is not None and len(r["curve"])]))
        ceil = float(np.median([r["tail5"] for r in g[(cell, "true", None)]
                                if r["tail5"] is not None]))
        scale[cell] = (floor, ceil)
        assert ceil > floor, f"{cell}: ceiling {ceil} not above floor {floor}"
    return g, cells, budgets, scale


def series(g, scale, cell, arm, budget, field="tail5"):
    """per-seed normalized values, ordered by seed, or None if the arm is absent"""
    key = (cell, arm, None if arm == "true" else budget)
    rs = sorted(g.get(key, []), key=lambda r: (r["seed"] is None, r["seed"]))
    vals = [(r["seed"], r[field]) for r in rs if r.get(field) is not None]
    if not vals:
        return None
    floor, ceil = scale[cell]
    return {s: (v - floor) / (ceil - floor) for s, v in vals}


def paired(a, b):
    """median paired difference, win count, exact two-sided Wilcoxon p"""
    if not a or not b:
        return None
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    d = np.array([a[s] - b[s] for s in shared])
    if np.allclose(d, 0):
        return float(np.median(d)), int((d > 0).sum()), len(d), 1.0
    try:
        p = float(wilcoxon(d, alternative="two-sided", method="exact").pvalue)
    except Exception:
        p = float(wilcoxon(d, alternative="two-sided").pvalue)
    return float(np.median(d)), int((d > 0).sum()), len(d), p


def med(s):
    return float(np.median(list(s.values()))) if s else np.nan


# --------------------------------------------------------------------------
# figure 1 -- the alpha law
# --------------------------------------------------------------------------
def fig_alpha_law(g, cells, budgets, scale, out_png):
    import matplotlib.pyplot as plt

    xs = [ALPHA_VALS[a] for a in ALPHAS]
    panels = cells + ["__aggregate__"]
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 10.2))
    style = {budgets[0]: (C_WS_LIGHT, "--", "o"), budgets[-1]: (C_WS, "-", "s")}

    agg = {b: [] for b in budgets}
    agg_ref = {b: {"vanilla": [], "naive": []} for b in budgets}

    for i, panel in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        if panel == "__aggregate__":
            for b in budgets:
                col, ls, mk = style[b]
                ys = np.median(np.vstack(agg[b]), axis=0)
                ax.plot(xs, ys, color=col, linestyle=ls, marker=mk, markersize=6,
                        linewidth=2.4, label=f"weighted sum, {b} labels", zorder=3)
                for name, c in (("vanilla", C_VANILLA), ("naive", C_NAIVE)):
                    ax.axhline(float(np.median(agg_ref[b][name])), color=c,
                               linestyle=ls, linewidth=1.6, alpha=0.9)
            ax.set_title("median across the 8 environments",
                         fontsize=11, fontweight="bold")
        else:
            keep, offscale = [0.0, 1.0], []
            for b in budgets:
                col, ls, mk = style[b]
                ys, seeds = [], []
                for a in ALPHAS:
                    s = series(g, scale, panel, f"ws{a}", b)
                    ys.append(med(s))
                    seeds.append(s)
                agg[b].append(ys)
                ax.plot(xs, ys, color=col, linestyle=ls, marker=mk, markersize=5.5,
                        linewidth=2.0, zorder=3)
                lo = [np.percentile(list(s.values()), 25) for s in seeds]
                hi = [np.percentile(list(s.values()), 75) for s in seeds]
                ax.fill_between(xs, lo, hi, color=col, alpha=0.14, linewidth=0, zorder=1)
                keep += lo + hi
                for name, c in (("vanilla", C_VANILLA), ("naive", C_NAIVE)):
                    v = med(series(g, scale, panel, name, b))
                    agg_ref[b][name].append(v)
                    ax.axhline(v, color=c, linestyle=ls, linewidth=1.4, alpha=0.85)
                    if name == "naive":
                        keep.append(v)
                    else:
                        offscale.append((v, c, b))
            # vanilla collapses far below the curves on some envs; letting it set the
            # axis flattens the alpha response into a sliver, so the axis follows the
            # curves and any reference below it is labelled at the bottom edge instead
            y0, y1 = min(keep), max(keep)
            pad = 0.14 * max(y1 - y0, 1e-6)
            y0, y1 = y0 - pad, y1 + pad
            ax.set_ylim(y0, y1)
            below = [(v, c, b) for v, c, b in offscale if v < y0]
            for j, (v, c, b) in enumerate(sorted(below)):
                ax.annotate(
                    f"vanilla@{b} = {v:.1f}",
                    xy=(0.30 + 0.28 * j, y0), xytext=(0.30 + 0.28 * j, y0 + 0.10 * (y1 - y0)),
                    color=c, fontsize=7.5, ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1, shrinkA=1, shrinkB=0))
            ax.set_title(CELL_TITLES.get(panel, panel), fontsize=11)

        ax.axhline(1.0, color=C_TRUE, linestyle=(0, (1, 2)), linewidth=1.5, zorder=2)
        ax.axhline(0.0, color=MUTED, linewidth=0.8, alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xlim(0.13, 0.87)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i % ncols == 0:
            ax.set_ylabel("normalized return", fontsize=9.5)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"$\alpha$   (weight on the hand-written prior)", fontsize=9.5)

    # one figure-level legend, so no panel carries a box
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=C_TRUE, ls=(0, (1, 2)), lw=1.5,
               label="true-reward ceiling (= 1.0)"),
        Line2D([], [], color=C_WS, ls="-", marker="s", ms=5.5, lw=2.0,
               label=f"weighted sum, {budgets[-1]} labels"),
        Line2D([], [], color=C_WS_LIGHT, ls="--", marker="o", ms=5.5, lw=2.0,
               label=f"weighted sum, {budgets[0]} labels"),
        Line2D([], [], color=C_NAIVE, ls="-", lw=1.6, label=f"naive, {budgets[-1]} labels"),
        Line2D([], [], color=C_NAIVE, ls="--", lw=1.6, label=f"naive, {budgets[0]} labels"),
        Line2D([], [], color=C_VANILLA, ls="-", lw=1.6, label=f"vanilla RLHF, {budgets[-1]} labels"),
        Line2D([], [], color=C_VANILLA, ls="--", lw=1.6, label=f"vanilla RLHF, {budgets[0]} labels"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        "How much weight should the hand-written prior carry?\n"
        r"normalized return vs $\alpha$ in $\alpha\cdot$norm(partial) + $(1-\alpha)\cdot$norm(model)"
        "   |   0 = untrained policy, 1 = training on the true reward"
        "   |   median of 10 seeds, shaded IQR",
        fontsize=12.5, y=0.985)
    fig.tight_layout(rect=[0, 0.055, 1, 0.945])
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


# --------------------------------------------------------------------------
# figure 2 -- env x arm summary, both budgets
# --------------------------------------------------------------------------
def fig_summary(g, cells, budgets, scale, out_png, report):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    # sequential, single hue, light -> dark: magnitude, not polarity
    cmap = LinearSegmentedColormap.from_list(
        "wsblue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])

    fig, axes = plt.subplots(1, len(budgets), figsize=(7.6 * len(budgets), 6.4))
    axes = np.atleast_1d(axes)

    for bi, budget in enumerate(budgets):
        ax = axes[bi]
        M = np.full((len(cells), len(ARM_ORDER)), np.nan)
        txt = [["" for _ in ARM_ORDER] for _ in cells]
        sig = np.zeros_like(M, dtype=bool)
        for r, cell in enumerate(cells):
            van = series(g, scale, cell, "vanilla", budget)
            for c, arm in enumerate(ARM_ORDER):
                s = series(g, scale, cell, arm, budget)
                if not s:
                    continue
                M[r, c] = med(s)
                txt[r][c] = f"{M[r, c]:.2f}"
                if arm != "vanilla":
                    pr = paired(s, van)
                    if pr and pr[3] < 0.05:
                        sig[r, c] = True
                    report[f"{cell}|{budget}|{arm}|vs_vanilla"] = (
                        None if not pr else
                        {"median_delta_norm": pr[0], "wins": f"{pr[1]}/{pr[2]}", "p_exact": pr[3]})
                report[f"{cell}|{budget}|{arm}|norm"] = M[r, c]

        im = ax.imshow(np.clip(M, 0.0, 1.15), cmap=cmap, vmin=0.0, vmax=1.15,
                       aspect="auto")
        for r in range(len(cells)):
            for c in range(len(ARM_ORDER)):
                if np.isnan(M[r, c]):
                    continue
                shade = cmap(np.clip(M[r, c], 0, 1.15) / 1.15)
                lum = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
                ax.text(c, r, txt[r][c], ha="center", va="center", fontsize=9,
                        color="white" if lum < 0.55 else INK,
                        fontweight="bold" if sig[r, c] else "normal")
                if sig[r, c]:
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                           edgecolor="#111111", linewidth=1.6, zorder=4))
        ax.set_xticks(range(len(ARM_ORDER)))
        ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER], fontsize=9)
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([CELL_TITLES.get(c, c) for c in cells], fontsize=9.5)
        ax.set_title(f"{budget} preference labels", fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(-0.5, len(ARM_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(cells), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        if bi:
            ax.set_yticklabels([])

    cb = fig.colorbar(im, ax=list(axes), fraction=0.022, pad=0.015)
    cb.set_label("normalized return   (0 = untrained,  1 = true-reward ceiling)",
                 fontsize=9.5)
    cb.ax.axhline(1.0, color=C_TRUE, linewidth=1.4)
    fig.suptitle(
        "Every arm on one scale: fraction of the true-reward ceiling recovered\n"
        "median of 10 paired seeds   |   boxed + bold = beats vanilla RLHF, "
        "exact two-sided Wilcoxon p < 0.05   |   values clipped to 1.15 for color, printed exactly",
        fontsize=12.5, y=0.99)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


# --------------------------------------------------------------------------
# figure 3 -- label efficiency
# --------------------------------------------------------------------------
def fig_label_efficiency(g, cells, budgets, scale, out_png, report):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    lo_b, hi_b = budgets[0], budgets[-1]
    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    ys = np.arange(len(cells))[::-1]
    rows = []

    # vanilla falls far below the untrained policy on several envs; the axis stops
    # at XMIN and anything past it is pinned to the edge as a triangle with its value,
    # so an off-scale point is visible rather than silently clipped away
    XMIN, XMAX, XWIN = -0.75, 1.20, 1.36

    def clamp(v):
        return max(v, XMIN + 0.02)

    for y, cell in zip(ys, cells):
        van_hi = series(g, scale, cell, "vanilla", hi_b)
        van_lo = series(g, scale, cell, "vanilla", lo_b)
        ws_lo = series(g, scale, cell, f"ws{HEADLINE_ALPHA}", lo_b)
        naive_lo = series(g, scale, cell, "naive", lo_b)
        span = [med(series(g, scale, cell, f"ws{a}", lo_b)) for a in ALPHAS]

        # the full alpha range at the small budget, so alpha=0.8 is not read as a pick
        ax.plot([clamp(min(span)), clamp(max(span))], [y, y], color=C_WS_LIGHT,
                linewidth=7, alpha=0.35, solid_capstyle="round", zorder=1)
        ax.plot([clamp(med(van_lo)), clamp(med(van_hi))], [y, y], color=C_VANILLA,
                linewidth=1.4, alpha=0.55, zorder=2)
        off = []
        for v, kw in ((med(van_lo), dict(s=52, facecolor="white", edgecolor=C_VANILLA,
                                         linewidth=1.8)),
                      (med(van_hi), dict(s=72, color=C_VANILLA))):
            if v < XMIN:
                off.append(v)
            else:
                ax.scatter(v, y, **kw, zorder=4)
        if off:
            # both budgets can land off-scale on the same row, so they share one label
            ax.scatter(XMIN + 0.02, y, marker="<", s=58, color=C_VANILLA, zorder=4)
            ax.text(XMIN + 0.055, y + 0.24,
                    "vanilla " + " / ".join(f"{v:.1f}" for v in sorted(off)),
                    fontsize=7.5, color=C_VANILLA, va="bottom", ha="left")
        ax.scatter(clamp(med(naive_lo)), y, s=72, color=C_NAIVE, marker="D", zorder=4)
        ax.scatter(clamp(med(ws_lo)), y, s=95, color=C_WS, marker="s", zorder=5)

        pr = paired(ws_lo, van_hi)
        prn = paired(naive_lo, van_hi)
        rows.append((cell, med(ws_lo), med(van_hi), pr, prn))
        report[f"{cell}|ws{HEADLINE_ALPHA}_q{lo_b}_vs_vanilla_q{hi_b}"] = (
            None if not pr else {"median_delta_norm": pr[0], "wins": f"{pr[1]}/{pr[2]}",
                                 "p_exact": pr[3]})
        stars = "" if not pr else ("**" if pr[3] < 0.01 else ("*" if pr[3] < 0.05 else ""))
        ax.text(XWIN, y, f"{pr[1]}/{pr[2]}{stars}" if pr else "-", fontsize=9.5,
                va="center", ha="center", color=INK)

    ax.axvline(1.0, color=C_TRUE, linestyle=(0, (1, 2)), linewidth=1.5, zorder=3)
    ax.axvline(0.0, color=MUTED, linewidth=0.9, alpha=0.6)
    ax.axvline(XMAX + 0.06, color="#CCCCCC", linewidth=1.0)
    top = len(cells) - 0.42
    ax.text(1.0, top, "true-reward\nceiling", fontsize=8.5, color=C_TRUE,
            va="bottom", ha="center")
    ax.text(XWIN, top, f"seeds won\nvs vanilla@{hi_b}", fontsize=8.5,
            color=INK, va="bottom", ha="center")

    ax.set_yticks(ys)
    ax.set_yticklabels([CELL_TITLES.get(c, c) for c in cells], fontsize=10)
    ax.set_ylim(-0.65, len(cells) + 0.15)
    ax.set_xlim(XMIN, XWIN + 0.10)
    ax.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xlabel("normalized return   (0 = untrained,  1 = true-reward ceiling)", fontsize=10)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    handles = [
        Line2D([], [], color=C_WS, marker="s", ls="", ms=9,
               label=rf"weighted sum $\alpha$={ALPHA_VALS[HEADLINE_ALPHA]:g}, {lo_b} labels"),
        Line2D([], [], color=C_WS_LIGHT, lw=7, alpha=0.35,
               label=rf"range over all $\alpha$, {lo_b} labels"),
        Line2D([], [], color=C_NAIVE, marker="D", ls="", ms=8,
               label=f"naive, {lo_b} labels"),
        Line2D([], [], color=C_VANILLA, marker="o", ls="", ms=8.5,
               label=f"vanilla RLHF, {hi_b} labels  (2x the labels)"),
        Line2D([], [], marker="o", ls="", ms=8, markerfacecolor="white",
               markeredgecolor=C_VANILLA, markeredgewidth=1.8,
               label=f"vanilla RLHF, {lo_b} labels"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=9.5, frameon=False,
               ncol=3, bbox_to_anchor=(0.5, -0.02))
    n_win = sum(1 for _, w, v, _, _ in rows if w > v)
    fig.suptitle(
        f"Half the labels, and still ahead: the prior at {lo_b} labels vs zero-prior RLHF at {hi_b}\n"
        rf"weighted sum $\alpha$={ALPHA_VALS[HEADLINE_ALPHA]:g} beats vanilla RLHF on twice the budget in "
        f"{n_win} of {len(cells)} environments   |   median of 10 paired seeds, "
        "* p<0.05  ** p<0.01 exact Wilcoxon",
        fontsize=12.5, y=0.985)
    fig.tight_layout(rect=[0, 0.055, 1, 0.9])
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="logs/wsum_paper")
    ap.add_argument("--prefix", default="wsum")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#BBBBBB",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    g, cells, budgets, scale = collect(Path(args.logs), args.prefix)
    print(f"cells={cells}  budgets={budgets}")
    for c in cells:
        print(f"  {c:9s} floor {scale[c][0]:9.1f}  ceiling(true) {scale[c][1]:9.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {}
    fig_alpha_law(g, cells, budgets, scale, f"{out}_alpha_law.png")
    fig_summary(g, cells, budgets, scale, f"{out}_summary.png", report)
    rows = fig_label_efficiency(g, cells, budgets, scale, f"{out}_label_efficiency.png", report)

    print(f"\n{'cell':16s} {'ws@'+str(budgets[0]):>9s} {'van@'+str(budgets[-1]):>9s} "
          f"{'delta':>8s} {'wins':>6s} {'p':>8s}")
    for cell, w, v, pr, _ in rows:
        print(f"{CELL_TITLES.get(cell, cell):16s} {w:9.2f} {v:9.2f} "
              f"{pr[0]:8.2f} {str(pr[1])+'/'+str(pr[2]):>6s} {pr[3]:8.4f}")
    Path(f"{out}_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out}_report.json")


if __name__ == "__main__":
    main()
