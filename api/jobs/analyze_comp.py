"""Read back the composition grid: does the prior help, and does normalizing help.

    python jobs/analyze_comp.py                  # tables + CSV + figure
    python jobs/analyze_comp.py --cell pong
    python jobs/analyze_comp.py --no-plot

THE ARMS
--------
    true        ground-truth reward, 0 labels           the ceiling
    partial     the hand-written prior alone, 0 labels  what the prior is worth
    vanilla     learned reward model only               the floor
    naive       partial + RM, alpha 1.0, raw sum
    ws05/10/15  norm(partial)*alpha + norm(RM)

vanilla / naive / ws* each run at two budgets (350 and 700; 2800 and 5600 on the
Atari cells).

THE FOUR CONTRASTS, all paired by seed within a budget
------------------------------------------------------
    naive  - vanilla    does adding the prior to the model beat the model alone?
                        THIS IS THE HEADLINE.
    ws10   - naive      does normalizing both terms help, at the same weight?
                        Single-variable: alpha is 1.0 on both sides.
    ws05   - ws10       does the mixing ratio matter once both are normalized?
    ws15   - ws10
    partial - vanilla   does the prior ALONE already beat the model alone? If it
                        does, any gain from naive may be the prior carrying it.

STATISTICS. Exact two-sided Wilcoxon signed-rank via scipy, paired on seed. At
n=10 the exact floor is 2/2^10 = 0.002, so p<0.05 is reachable -- unlike the
5-seed screens. The normal approximation is NOT used: jobs/analyze_pre4.py's
hand-rolled version prints 0.005 where the exact value is 0.002.

READ LABEL DELIVERY FIRST. The table prints delivered/budget per arm. A shortfall
correlates with how badly an arm is doing, so a contrast against a starved arm is
not measuring what it claims to.

READ PEAK NEXT TO FINAL. They disagree in sign often enough in this project that
final alone has been misleading before.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover
    wilcoxon = None

CELLS = ("ll", "pusher", "reacher", "swimmer", "hopper", "bipedal", "qbert", "pong")

ENV_LABEL = {
    "ll": "LunarLander-v3",
    "pusher": "Pusher-v5",
    "reacher": "Reacher-v5",
    "swimmer": "Swimmer-v5",
    "hopper": "Hopper-v5",
    "bipedal": "BipedalWalker-v3",
    "qbert": "ALE/Qbert-v5",
    "pong": "ALE/Pong-v5",
}

METHODS = ("vanilla", "naive", "ws05", "ws10", "ws15")
BUDGETS = ("low", "high")

CONTRASTS = (
    ("naive", "vanilla", "naive - vanilla    [HEADLINE]"),
    ("ws10", "naive", "ws10 - naive       [normalization]"),
    ("ws05", "ws10", "ws05 - ws10        [alpha 0.5 v 1]"),
    ("ws15", "ws10", "ws15 - ws10        [alpha 1.5 v 1]"),
    ("partial", "vanilla", "partial - vanilla  [prior alone]"),
)


def load(root: Path) -> list[dict]:
    runs = []
    for meta_path in sorted(root.glob("comp_*/*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_dir = meta_path.parent
        _, cell, variant = run_dir.parent.name.split("_", 2)

        curve_path = run_dir / "eval" / "evaluations.npz"
        if not curve_path.exists():
            print(f"warning: no evaluations.npz for {run_dir}")
            continue
        data = np.load(curve_path)
        results = np.asarray(data["results"], dtype=np.float64)
        if results.size == 0:
            print(f"warning: empty curve for {run_dir}")
            continue
        point_means = results.mean(axis=1)
        timesteps = np.asarray(data["timesteps"], dtype=np.int64)
        peak_index = int(np.argmax(point_means))

        # variant is "true" / "partial" / "<method>_<budget>"
        if "_" in variant:
            method, budget = variant.rsplit("_", 1)
        else:
            method, budget = variant, ""

        runs.append(
            {
                "cell": cell,
                "variant": variant,
                "method": method,
                "budget": budget,
                "seed": int(meta["seed"]),
                "env_id": meta["env_id"],
                "mode": meta["mode"],
                "partial_reference": meta.get("partial_reference") or "",
                "partial_alpha": meta.get("partial_alpha"),
                "normalize_partial": bool(meta.get("normalize_partial_reward")),
                "normalize_model": bool(meta.get("normalize_model_reward")),
                "query_budget": int(meta.get("query_budget") or 0),
                "synthetic_queries": int(meta.get("synthetic_queries") or 0),
                "final": float(point_means[-1]),
                "peak": float(point_means[peak_index]),
                "peak_at": int(timesteps[peak_index]),
                "start": float(point_means[0]),
                "drawdown": (point_means[peak_index] - point_means[-1]) / abs(point_means[peak_index])
                if point_means[peak_index] else float("nan"),
                "timesteps": timesteps,
                "point_means": point_means,
            }
        )
    return runs


def med(values) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def paired(by_variant, a_variant, b_variant):
    """Return (deltas, wins, n) paired on seed, or (None, 0, 0)."""
    a = {run["seed"]: run["final"] for run in by_variant.get(a_variant, [])}
    b = {run["seed"]: run["final"] for run in by_variant.get(b_variant, [])}
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None, 0, 0
    deltas = np.array([a[s] - b[s] for s in seeds], dtype=np.float64)
    return deltas, int((deltas > 0).sum()), len(seeds)


def exact_p(deltas) -> float:
    """Exact two-sided Wilcoxon signed-rank. nan when undefined."""
    if wilcoxon is None or deltas is None or len(deltas) == 0:
        return float("nan")
    if np.all(deltas == 0):
        return 1.0
    try:
        return float(wilcoxon(deltas, alternative="two-sided", method="exact").pvalue)
    except (ValueError, TypeError):
        try:  # ties/zeros force the approximation; say so by returning it anyway
            return float(wilcoxon(deltas, alternative="two-sided").pvalue)
        except Exception:
            return float("nan")


def report(runs, cells) -> None:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["cell"], run["variant"]), []).append(run)

    for cell in cells:
        variants = {v: r for (c, v), r in grouped.items() if c == cell}
        if not variants:
            continue
        print("\n" + "=" * 104)
        print(f"{cell} / {ENV_LABEL.get(cell, cell)}")
        print("=" * 104)

        for reference in ("true", "partial"):
            rows = variants.get(reference, [])
            if rows:
                print(f"  {reference:8s} n={len(rows):2d}  final {med([r['final'] for r in rows]):10.1f}"
                      f"   peak {med([r['peak'] for r in rows]):10.1f}"
                      f"   drawdown {med([r['drawdown'] for r in rows]):5.2f}")
        prior = next((r["partial_reference"] for r in variants.get("partial", []) if r["partial_reference"]), "")
        if prior:
            print(f"  prior: {prior}")

        for budget in BUDGETS:
            present = [m for m in METHODS if f"{m}_{budget}" in variants]
            if not present:
                continue
            example = variants[f"{present[0]}_{budget}"][0]
            print(f"\n  --- budget {budget} (q{example['query_budget']}) ---")
            print(f"      {'arm':9s} {'n':>2s} {'final':>10s} {'peak':>10s} {'drawdown':>9s} "
                  f"{'labels':>12s} {'alpha':>6s} {'norm':>5s}")
            for method in present:
                rows = variants[f"{method}_{budget}"]
                delivery = f"{med([r['synthetic_queries'] for r in rows]):.0f}/{rows[0]['query_budget']}"
                alpha = rows[0]["partial_alpha"]
                norm = "yes" if rows[0]["normalize_partial"] else "no"
                if method == "vanilla":
                    alpha, norm = None, "-"
                print(f"      {method:9s} {len(rows):2d} {med([r['final'] for r in rows]):10.1f} "
                      f"{med([r['peak'] for r in rows]):10.1f} {med([r['drawdown'] for r in rows]):9.2f} "
                      f"{delivery:>12s} {('-' if alpha is None else f'{alpha:.1f}'):>6s} {norm:>5s}")

            print(f"      {'contrast':34s} {'delta':>10s} {'wins':>7s} {'p':>9s}")
            for a_method, b_method, label in CONTRASTS:
                a_variant = a_method if a_method in ("true", "partial") else f"{a_method}_{budget}"
                b_variant = b_method if b_method in ("true", "partial") else f"{b_method}_{budget}"
                if a_variant not in variants or b_variant not in variants:
                    continue
                by_variant = {a_variant: variants[a_variant], b_variant: variants[b_variant]}
                deltas, wins, n = paired(by_variant, a_variant, b_variant)
                if deltas is None:
                    continue
                p = exact_p(deltas)
                star = " *" if p < 0.05 else ""
                print(f"      {label:34s} {med(deltas):+10.1f} {wins:3d}/{n:<3d} {p:9.3f}{star}")

    print("\n" + "-" * 104)
    print("delta is the median PAIRED difference on final return, positive favouring the first arm.")
    print("wins counts seeds where the first arm was higher. p is EXACT two-sided Wilcoxon (n=10 floor 0.002).")
    print("* marks p<0.05. Check labels before reading any row: a starved arm is not a fair comparison.")
    print("Reacher and Pusher are negative-scale (closer to 0 is better); a positive delta is still better there,")
    print("because the contrast is computed on the raw return, not on distance from zero.")


def write_csv(runs, path: Path) -> None:
    fields = ["cell", "env_id", "variant", "method", "budget", "seed", "mode", "partial_reference",
              "partial_alpha", "normalize_partial", "normalize_model", "query_budget",
              "synthetic_queries", "final", "peak", "peak_at", "start", "drawdown"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda r: (r["cell"], r["variant"], r["seed"])):
            writer.writerow({f: run[f] for f in fields})
    print(f"wrote per-seed CSV: {path}  ({len(runs)} runs)")


def plot(runs, cells, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["cell"], run["variant"]), []).append(run)
    live = [c for c in cells if any(k[0] == c for k in grouped)]
    if not live:
        print("nothing to plot")
        return

    colors = {"true": "black", "partial": "tab:grey", "vanilla": "seagreen",
              "naive": "tab:red", "ws05": "tab:orange", "ws10": "tab:blue", "ws15": "tab:purple"}
    ncols = 2
    nrows = len(live)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3.8 * nrows), squeeze=False)

    for row, cell in enumerate(live):
        for col, budget in enumerate(BUDGETS):
            ax = axes[row][col]

            def draw(variant, label, color, **kw):
                rows = grouped.get((cell, variant), [])
                if not rows:
                    return
                length = min(len(r["point_means"]) for r in rows)
                stack = np.vstack([r["point_means"][:length] for r in rows])
                ax.plot(rows[0]["timesteps"][:length], np.median(stack, axis=0),
                        color=color, label=label, **kw)

            draw("true", "true", colors["true"], linewidth=2.2)
            draw("partial", "partial", colors["partial"], linewidth=1.6, linestyle=":")
            for method in METHODS:
                draw(f"{method}_{budget}", method, colors[method], linewidth=1.5,
                     linestyle="--" if method.startswith("ws") else "-")
            example = grouped.get((cell, f"vanilla_{budget}"), [{}])[0]
            ax.set_title(f"{cell} / {ENV_LABEL.get(cell, cell)} -- q{example.get('query_budget', '?')}")
            ax.set_xlabel("timesteps")
            ax.set_ylabel("true reward")
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False, ncol=2)

    fig.suptitle("Composition grid: true / partial / vanilla / naive / weighted sum", y=1.0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figure: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs")
    parser.add_argument("--cell", action="append", choices=CELLS)
    parser.add_argument("--csv", default="logs/comp_summary.csv")
    parser.add_argument("--figure", default="logs/comp_curves.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    runs = load(Path(args.root))
    if not runs:
        print(f"no comp_* runs with metadata.json under {Path(args.root).resolve()}")
        return
    cells = tuple(args.cell) if args.cell else CELLS
    runs = [r for r in runs if r["cell"] in cells]

    counts: dict[tuple[str, str], int] = {}
    for run in runs:
        counts[(run["cell"], run["variant"])] = counts.get((run["cell"], run["variant"]), 0) + 1
    short = [f"{c}/{v} ({n}/10)" for (c, v), n in sorted(counts.items()) if n != 10]
    if short:
        print(f"INCOMPLETE ARMS (not 10 seeds): {', '.join(short)}\n")
    if wilcoxon is None:
        print("scipy not available -- p-values will be nan\n")

    write_csv(runs, Path(args.csv))
    report(runs, cells)
    if not args.no_plot:
        plot(runs, cells, Path(args.figure))


if __name__ == "__main__":
    main()
