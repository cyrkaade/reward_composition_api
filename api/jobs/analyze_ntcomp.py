"""Analyze the gated non-timid confirmatory composition grid.

Primary endpoint (declared before runs): q200 naive minus vanilla. The global
test uses one environment-level normalized effect per cell,

    median_seed(naive - vanilla) / (median(true) - median(vanilla)),

and an exact two-sided Wilcoxon test across the eleven environments. A cell where
q200 vanilla reaches/exceeds true fails the headroom premise and makes the
global endpoint invalid rather than being silently dropped. Individual q200
cell tests are Holm-corrected across all eleven. q100 is supportive and q400 is
the high-label saturation check.  Normalization/alpha contrasts are secondary.
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

CELLS = ("ll", "reacher", "pusher", "swimmer", "hopper", "bipedal", "walker", "ant", "mspacman", "qbert", "pong")
ATARI_CELLS = {"mspacman", "qbert", "pong"}
BUDGETS = (100, 200, 400)
METHODS = ("vanilla", "naive", "ws25", "ws50", "ws75")


def med(values) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def exact_p(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    if wilcoxon is None or len(values) == 0:
        return float("nan")
    if np.all(values == 0):
        return 1.0
    try:
        return float(wilcoxon(values, alternative="two-sided", method="exact").pvalue)
    except (TypeError, ValueError):
        return float(wilcoxon(values, alternative="two-sided").pvalue)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    finite = sorted(((key, value) for key, value in pvalues.items() if np.isfinite(value)), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    count = len(finite)
    for rank, (key, value) in enumerate(finite):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[key] = running
    return {key: adjusted.get(key, float("nan")) for key in pvalues}


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    data = np.load(path)
    results = np.asarray(data["results"], dtype=np.float64)
    if results.size == 0:
        return None
    return np.asarray(data["timesteps"], dtype=np.int64), results.mean(axis=1)


def load(root: Path) -> list[dict]:
    runs = []
    for meta_path in sorted(root.glob("ntcomp_*/*/metadata.json")):
        run_dir = meta_path.parent
        _, cell, variant = run_dir.parent.name.split("_", 2)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        deterministic_curve = load_curve(run_dir / "eval" / "evaluations.npz")
        stochastic_curve = load_curve(run_dir / "eval_stochastic" / "evaluations.npz")
        if deterministic_curve is None or (cell in ATARI_CELLS and stochastic_curve is None):
            continue
        det_timesteps, det_means = deterministic_curve
        timesteps, means = stochastic_curve if cell in ATARI_CELLS else deterministic_curve
        peak_index = int(np.argmax(means))
        if "_q" in variant:
            method, budget_text = variant.rsplit("_q", 1)
            budget = int(budget_text)
        else:
            method, budget = variant, 0
        runs.append({
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
            "final": float(means[-1]),
            "peak": float(means[peak_index]),
            "peak_at": int(timesteps[peak_index]),
            "start": float(means[0]),
            "timesteps": timesteps,
            "means": means,
            "det_final": float(det_means[-1]),
            "det_peak": float(det_means.max()),
            "det_peak_at": int(det_timesteps[int(np.argmax(det_means))]),
            "det_start": float(det_means[0]),
            "det_timesteps": det_timesteps,
            "det_means": det_means,
        })
    return runs


def validate_complete(runs: list[dict], params: Path) -> None:
    expected = set()
    for line in params.read_text(encoding="utf-8").splitlines():
        cell, variant, seed, _ = line.split()
        expected.add((cell, variant, int(seed)))
    actual = {(run["cell"], run["variant"], run["seed"]) for run in runs}
    missing = sorted(expected - actual)
    extras = sorted(actual - expected)
    if len(expected) != 1870 or missing or extras or len(actual) != len(runs):
        print(f"invalid/incomplete: expected={len(expected)} actual={len(actual)} missing={len(missing)} extras={len(extras)}")
        if missing:
            print(f"  missing: {missing[:15]}")
        if extras:
            print(f"  extras: {extras[:15]}")
        raise SystemExit(2)


def write_csv(runs: list[dict], path: Path) -> None:
    fields = (
        "cell", "env_id", "variant", "method", "budget", "seed", "mode", "partial_reference",
        "partial_alpha", "normalize_partial", "normalize_model", "query_budget", "synthetic_queries",
        "final", "peak", "peak_at", "start", "det_final", "det_peak", "det_peak_at", "det_start",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda item: (item["cell"], item["variant"], item["seed"])):
            writer.writerow({field: run[field] for field in fields})
    print(f"wrote {path} ({len(runs)} runs)")


def deltas(grouped, cell: str, a: str, b: str) -> np.ndarray:
    aa = {run["seed"]: run["final"] for run in grouped[(cell, a)]}
    bb = {run["seed"]: run["final"] for run in grouped[(cell, b)]}
    seeds = sorted(set(aa) & set(bb))
    return np.asarray([aa[seed] - bb[seed] for seed in seeds], dtype=np.float64)


def report(runs: list[dict]) -> None:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["cell"], run["variant"]), []).append(run)

    primary_p = {}
    normalized_effects = []
    invalid_headroom = []
    rows = []
    for cell in CELLS:
        true_final = med([run["final"] for run in grouped[(cell, "true")]])
        partial_final = med([run["final"] for run in grouped[(cell, "partial")]])
        reference = grouped[(cell, "partial")][0]["partial_reference"]
        print("\n" + "=" * 115)
        print(f"{cell}: true {true_final:.1f}, partial {partial_final:.1f}, prior {reference}")
        print(f"{'budget':>7s} {'vanilla':>10s} {'naive':>10s} {'delta':>10s} {'wins':>7s} {'p':>8s} {'labels V/N':>17s} {'ws50-naive':>12s}")
        for budget in BUDGETS:
            vanilla_variant = f"vanilla_q{budget}"
            naive_variant = f"naive_q{budget}"
            vanilla = grouped[(cell, vanilla_variant)]
            naive = grouped[(cell, naive_variant)]
            effect = deltas(grouped, cell, naive_variant, vanilla_variant)
            norm_effect = deltas(grouped, cell, f"ws50_q{budget}", naive_variant)
            p = exact_p(effect)
            labels = (
                f"{med([r['synthetic_queries'] for r in vanilla]):.0f}/{budget},"
                f"{med([r['synthetic_queries'] for r in naive]):.0f}/{budget}"
            )
            print(f"q{budget:>5d} {med([r['final'] for r in vanilla]):10.1f} {med([r['final'] for r in naive]):10.1f} "
                  f"{med(effect):+10.1f} {int((effect > 0).sum()):3d}/{len(effect):<3d} {p:8.3f} {labels:>17s} {med(norm_effect):+12.1f}")
            rows.append((cell, budget, med(effect), p))
            if budget == 200:
                primary_p[cell] = p
                vanilla_final = med([run["final"] for run in vanilla])
                headroom = true_final - vanilla_final
                if headroom <= 0:
                    invalid_headroom.append(cell)
                else:
                    normalized_effects.append(med(effect) / headroom)

    adjusted = holm(primary_p)
    print("\n" + "=" * 115)
    print("PRIMARY q200 CELL TESTS (exact Wilcoxon, Holm across 11 environments)")
    for cell in CELLS:
        effect = next(value for this_cell, budget, value, _ in rows if this_cell == cell and budget == 200)
        print(f"  {cell:9s} delta {effect:+10.2f}  raw p={primary_p[cell]:.4f}  Holm p={adjusted[cell]:.4f}")
    if invalid_headroom:
        print(f"GLOBAL PRIMARY INVALID: q200 vanilla reached/exceeded true in {', '.join(invalid_headroom)}")
    else:
        global_p = exact_p(normalized_effects)
        print(f"GLOBAL PRIMARY: median normalized q200 effect {med(normalized_effects):+.3f}, "
              f"positive cells {sum(value > 0 for value in normalized_effects)}/11, exact p={global_p:.4f}")


def plot(runs: list[dict], output: Path, deterministic: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["cell"], run["variant"]), []).append(run)
    fig, axes = plt.subplots(len(CELLS), len(BUDGETS), figsize=(19, 41), squeeze=False)
    colors = {"true": "black", "partial": "grey", "vanilla": "seagreen", "naive": "tab:red",
              "ws25": "tab:orange", "ws50": "tab:blue", "ws75": "tab:purple"}
    for row, cell in enumerate(CELLS):
        for col, budget in enumerate(BUDGETS):
            ax = axes[row][col]
            variants = ("true", "partial") + tuple(f"{method}_q{budget}" for method in METHODS)
            for variant in variants:
                rows = grouped[(cell, variant)]
                means_key = "det_means" if deterministic else "means"
                timesteps_key = "det_timesteps" if deterministic else "timesteps"
                length = min(len(run[means_key]) for run in rows)
                stack = np.vstack([run[means_key][:length] for run in rows])
                method = variant.split("_q", 1)[0]
                ax.plot(rows[0][timesteps_key][:length], np.median(stack, axis=0),
                        color=colors[method], label=method,
                        linestyle="--" if method.startswith("ws") else (":" if method == "partial" else "-"))
            ax.set_title(f"{cell} / q{budget}")
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False, ncol=2)
    policy_label = "deterministic diagnostic" if deterministic else "stochastic primary for Atari"
    fig.suptitle(f"Non-timid confirmatory composition grid ({policy_label})")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs")
    parser.add_argument("--params", default="jobs/params_ntcomp.txt")
    parser.add_argument("--csv", default="logs/ntcomp_summary.csv")
    parser.add_argument("--figure", default="logs/ntcomp_curves.png")
    parser.add_argument("--deterministic-figure", default="logs/ntcomp_curves_deterministic.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    runs = load(Path(args.root))
    validate_complete(runs, Path(args.params))
    write_csv(runs, Path(args.csv))
    report(runs)
    if not args.no_plot:
        plot(runs, Path(args.figure))
        plot(runs, Path(args.deterministic_figure), deterministic=True)


if __name__ == "__main__":
    main()
