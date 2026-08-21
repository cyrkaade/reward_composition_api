"""Analyze the reasonable-prior grid with paired seeds and stable endpoints.

Primary score = the mean of the last five periodic checkpoints for each seed.
For Atari those checkpoints use sampled-policy (stochastic) returns; argmax
returns are retained only in a separate diagnostic figure.  The report never
uses a peak or a single final checkpoint to choose a candidate.

Five seeds are descriptive, not a significance test.  Candidates are ranked by
their paired distance from vanilla RLHF, and only when the true arm actually
beats vanilla.  No tiny positive change is labelled a pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CELLS = (
    "ll", "bipedal", "ant", "reacher", "pusher", "hopper", "swimmer", "walker",
    "mspacman", "qbert", "pong",
)
ATARI_CELLS = {"mspacman", "qbert", "pong"}
TAIL_CHECKPOINTS = 5


def med(values) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def read_expected(path: Path) -> tuple[dict[tuple[str, str, int], str], dict[str, list[str]]]:
    expected: dict[tuple[str, str, int], str] = {}
    variants: dict[str, list[str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 4:
            raise SystemExit(f"{path}:{line_number}: expected four fields, got {len(fields)}")
        cell, variant, seed_text, reference = fields
        seed = int(seed_text)
        key = (cell, variant, seed)
        if key in expected:
            raise SystemExit(f"duplicate parameter row: {key}")
        expected[key] = reference
        variants.setdefault(cell, [])
        if variant not in variants[cell]:
            variants[cell].append(variant)
    if tuple(variants) != CELLS:
        raise SystemExit(f"parameter cell order/content differs from {CELLS}: {tuple(variants)}")
    for cell, cell_variants in variants.items():
        if cell_variants[:2] != ["true", "vanilla"]:
            raise SystemExit(f"{cell}: true and vanilla must be the first two variants")
        for variant in cell_variants:
            seeds = sorted(seed for c, v, seed in expected if c == cell and v == variant)
            if seeds != list(range(5)):
                raise SystemExit(f"{cell}/{variant}: expected paired seeds 0..4, got {seeds}")
    return expected, variants


def curve(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    payload = np.load(path)
    results = np.asarray(payload["results"], dtype=np.float64)
    if results.size == 0:
        return None
    return np.asarray(payload["timesteps"], dtype=np.int64), results.mean(axis=1)


def load_runs(root: Path) -> list[dict]:
    runs = []
    for metadata_path in sorted(root.glob("reasonable_*/*/metadata.json")):
        run_dir = metadata_path.parent
        try:
            _, cell, variant = run_dir.parent.name.split("_", 2)
        except ValueError:
            continue
        if cell not in CELLS:
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        deterministic = curve(run_dir / "eval" / "evaluations.npz")
        stochastic = curve(run_dir / "eval_stochastic" / "evaluations.npz")
        primary = stochastic if cell in ATARI_CELLS else deterministic
        if primary is None or deterministic is None:
            continue
        timesteps, means = primary
        det_timesteps, det_means = deterministic
        if len(means) < TAIL_CHECKPOINTS or len(det_means) < TAIL_CHECKPOINTS:
            continue
        runs.append({
            "cell": cell,
            "variant": variant,
            "seed": int(metadata["seed"]),
            "env_id": metadata["env_id"],
            "mode": metadata["mode"],
            "partial_reference": metadata.get("partial_reference") or "-",
            "primary_policy": "stochastic" if cell in ATARI_CELLS else "deterministic",
            "head5": float(means[:TAIL_CHECKPOINTS].mean()),
            "tail5": float(means[-TAIL_CHECKPOINTS:].mean()),
            "last": float(means[-1]),
            "peak": float(means.max()),
            "det_head5": float(det_means[:TAIL_CHECKPOINTS].mean()),
            "det_tail5": float(det_means[-TAIL_CHECKPOINTS:].mean()),
            "det_last": float(det_means[-1]),
            "synthetic_queries": int(metadata.get("synthetic_queries") or 0),
            "query_budget": int(metadata.get("query_budget") or 0),
            "timesteps": timesteps,
            "means": means,
            "det_timesteps": det_timesteps,
            "det_means": det_means,
            "run_dir": str(run_dir),
        })
    return runs


def validate(
    runs: list[dict], expected: dict[tuple[str, str, int], str]
) -> dict[tuple[str, str, int], dict]:
    actual: dict[tuple[str, str, int], dict] = {}
    duplicates = []
    for run in runs:
        key = (run["cell"], run["variant"], run["seed"])
        if key in actual:
            duplicates.append(key)
        actual[key] = run
    missing = sorted(set(expected) - set(actual))
    extras = sorted(set(actual) - set(expected))
    mismatched = [
        (key, expected[key], actual[key]["partial_reference"])
        for key in sorted(set(expected) & set(actual))
        if expected[key] != actual[key]["partial_reference"]
    ]
    if missing or extras or duplicates or mismatched:
        print(
            f"INCOMPLETE/INVALID GRID: missing={len(missing)} extras={len(extras)} "
            f"duplicates={len(duplicates)} refs={len(mismatched)}"
        )
        for label, values in (
            ("missing", missing), ("extras", extras),
            ("duplicates", duplicates), ("references", mismatched),
        ):
            if values:
                print(f"  {label}: {values[:15]}")
        raise SystemExit(2)
    return actual


def group_runs(runs: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["cell"], run["variant"]), []).append(run)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["seed"])
    return grouped


def paired_delta(left: list[dict], right: list[dict], field: str = "tail5") -> np.ndarray:
    left_by_seed = {row["seed"]: row[field] for row in left}
    right_by_seed = {row["seed"]: row[field] for row in right}
    if set(left_by_seed) != set(right_by_seed):
        raise ValueError("paired comparison has different seeds")
    return np.asarray([left_by_seed[seed] - right_by_seed[seed] for seed in sorted(left_by_seed)])


def classify(position: float, valid_bracket: bool) -> str:
    if not valid_bracket or not np.isfinite(position):
        return "NO TRUE>VANILLA BRACKET"
    if position < -0.10:
        return "below vanilla"
    if position <= 0.25:
        return "vanilla-like"
    if position <= 0.80:
        return "between vanilla and true"
    if position <= 1.10:
        return "near true"
    return "above true (premise warning)"


def summarize(grouped, variants: dict[str, list[str]]) -> dict:
    report = {
        "endpoint": f"mean of last {TAIL_CHECKPOINTS} periodic checkpoints per seed",
        "atari_primary_policy": "stochastic sampled policy",
        "atari_diagnostic_policy": "deterministic argmax policy",
        "seeds": list(range(5)),
        "environments": {},
    }
    print("\nPrimary endpoint: per-seed mean of the last five checkpoints; paired seeds 0..4.")
    print("Atari uses stochastic sampled-policy return. Five seeds are descriptive; no p-values or pass flags.\n")
    for cell in CELLS:
        true_rows = grouped[(cell, "true")]
        vanilla_rows = grouped[(cell, "vanilla")]
        true_value = med([row["tail5"] for row in true_rows])
        vanilla_value = med([row["tail5"] for row in vanilla_rows])
        bracket = true_value - vanilla_value
        true_vanilla_deltas = paired_delta(true_rows, vanilla_rows)
        bracket_threshold = max(1.0, 0.05 * max(abs(true_value), abs(vanilla_value)))
        true_wins = int(np.count_nonzero(true_vanilla_deltas > 0))
        # A tiny positive gap is not a usable ceiling/floor bracket.  This is the
        # same practical floor that prevents the old Pong +0.2 smoke wobble from
        # being called learning, plus a 4/5 paired-seed consistency requirement.
        valid_bracket = bracket >= bracket_threshold and true_wins >= 4
        labels = med([row["synthetic_queries"] for row in vanilla_rows])
        print("=" * 118)
        print(
            f"{cell}: true={true_value:.2f}, vanilla={vanilla_value:.2f}, "
            f"true-vanilla={bracket:+.2f} (need {bracket_threshold:.2f}, wins {true_wins}/5), "
            f"labels={labels:.0f}/350, "
            f"primary={true_rows[0]['primary_policy']}"
        )
        print(
            f"{'candidate':28s} {'tail5':>11s} {'paired-v':>11s} {'wins':>7s} "
            f"{'position':>10s} {'learned':>11s}  interpretation"
        )
        candidate_payload = []
        for variant in variants[cell][2:]:
            rows = grouped[(cell, variant)]
            value = med([row["tail5"] for row in rows])
            delta = paired_delta(rows, vanilla_rows)
            position = (value - vanilla_value) / bracket if valid_bracket else float("nan")
            learning = med([row["tail5"] - row["head5"] for row in rows])
            verdict = classify(position, valid_bracket)
            candidate = {
                "variant": variant,
                "partial_reference": rows[0]["partial_reference"],
                "median_tail5": value,
                "median_paired_delta_vs_vanilla": float(np.median(delta)),
                "wins_vs_vanilla": int(np.count_nonzero(delta > 0)),
                "position_vanilla_0_true_1": position if np.isfinite(position) else None,
                "median_tail5_minus_head5": learning,
                "interpretation": verdict,
            }
            candidate_payload.append(candidate)
            position_text = f"{position:10.2f}" if np.isfinite(position) else f"{'n/a':>10s}"
            print(
                f"{variant:28s} {value:11.2f} {np.median(delta):+11.2f} "
                f"{np.count_nonzero(delta > 0):>3d}/5 {position_text} {learning:+11.2f}  {verdict}"
            )
        ranked = sorted(
            (item for item in candidate_payload if item["position_vanilla_0_true_1"] is not None),
            key=lambda item: abs(item["position_vanilla_0_true_1"]),
        )
        report["environments"][cell] = {
            "true_median_tail5": true_value,
            "vanilla_median_tail5": vanilla_value,
            "true_minus_vanilla": bracket,
            "practical_bracket_threshold": bracket_threshold,
            "true_wins_vs_vanilla": true_wins,
            "valid_true_vanilla_bracket": valid_bracket,
            "vanilla_labels_median": labels,
            "candidates": candidate_payload,
            "closest_to_vanilla_descriptive": [item["variant"] for item in ranked[:3]],
        }
    return report


def write_csv(runs: list[dict], path: Path) -> None:
    fields = (
        "cell", "env_id", "variant", "mode", "partial_reference", "seed", "primary_policy",
        "head5", "tail5", "last", "peak", "det_head5", "det_tail5", "det_last",
        "synthetic_queries", "query_budget", "run_dir",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda row: (CELLS.index(row["cell"]), row["variant"], row["seed"])):
            writer.writerow({field: run[field] for field in fields})
    print(f"wrote {path} ({len(runs)} runs)")


def plot_curves(grouped, variants: dict[str, list[str]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(17, 25), squeeze=False)
    candidate_colors = plt.cm.tab10(np.linspace(0.0, 0.9, 8))
    for index, cell in enumerate(CELLS):
        ax = axes[index // 2][index % 2]
        for variant_index, variant in enumerate(variants[cell]):
            rows = grouped[(cell, variant)]
            length = min(len(row["means"]) for row in rows)
            stack = np.vstack([row["means"][:length] for row in rows])
            center = np.median(stack, axis=0)
            low, high = np.percentile(stack, [25, 75], axis=0)
            if variant == "true":
                color, width, style = "black", 2.5, "-"
            elif variant == "vanilla":
                color, width, style = "seagreen", 2.3, "-"
            else:
                color, width, style = candidate_colors[variant_index - 2], 1.35, "--"
            ax.plot(rows[0]["timesteps"][:length], center, color=color, linewidth=width, linestyle=style, label=variant)
            ax.fill_between(rows[0]["timesteps"][:length], low, high, color=color, alpha=0.08, linewidth=0)
        policy = "stochastic primary" if cell in ATARI_CELLS else "deterministic"
        ax.set_title(f"{cell} ({policy})")
        ax.set_xlabel("policy timesteps")
        ax.set_ylabel("true return")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=5.8, ncol=2, frameon=False)
    axes[-1][-1].axis("off")
    fig.suptitle("Reasonable partial screen: true vs vanilla RLHF vs partial-only (median and IQR over 5 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_tail5(grouped, variants: dict[str, list[str]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(17, 25), squeeze=False)
    for index, cell in enumerate(CELLS):
        ax = axes[index // 2][index % 2]
        cell_variants = variants[cell]
        for y, variant in enumerate(cell_variants):
            values = np.asarray([row["tail5"] for row in grouped[(cell, variant)]])
            color = "black" if variant == "true" else "seagreen" if variant == "vanilla" else "tab:orange"
            ax.scatter(values, np.full(values.shape, y), color=color, s=18, alpha=0.75)
            ax.scatter([np.median(values)], [y], color=color, marker="|", s=180, linewidth=2.5)
        ax.set_yticks(range(len(cell_variants)), cell_variants, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(cell)
        ax.set_xlabel("per-seed mean of last 5 true-return checkpoints")
        ax.grid(axis="x", alpha=0.25)
    axes[-1][-1].axis("off")
    fig.suptitle("Stable endpoints by paired seed (vertical mark = median)")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_atari_diagnostic(grouped, variants: dict[str, list[str]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    candidate_colors = plt.cm.tab10(np.linspace(0.0, 0.9, 8))
    for ax, cell in zip(axes.flat, sorted(ATARI_CELLS)):
        for variant_index, variant in enumerate(variants[cell]):
            rows = grouped[(cell, variant)]
            length = min(len(row["det_means"]) for row in rows)
            stack = np.vstack([row["det_means"][:length] for row in rows])
            center = np.median(stack, axis=0)
            low, high = np.percentile(stack, [25, 75], axis=0)
            color = "black" if variant == "true" else "seagreen" if variant == "vanilla" else candidate_colors[variant_index - 2]
            style = "-" if variant in ("true", "vanilla") else "--"
            ax.plot(rows[0]["det_timesteps"][:length], center, color=color, linestyle=style, label=variant)
            ax.fill_between(rows[0]["det_timesteps"][:length], low, high, color=color, alpha=0.07, linewidth=0)
        ax.set_title(cell)
        ax.set_xlabel("policy timesteps")
        ax.set_ylabel("true return")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=2, frameon=False)
    axes[-1][-1].axis("off")
    fig.suptitle("Atari deterministic argmax-policy diagnostic (not used for selection)")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs")
    parser.add_argument("--params", default="jobs/params_reasonable.txt")
    parser.add_argument("--output-prefix", default="logs/reasonable_screen")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    expected, variants = read_expected(Path(args.params))
    runs = load_runs(Path(args.root))
    validate(runs, expected)
    grouped = group_runs(runs)
    prefix = Path(args.output_prefix)
    write_csv(runs, prefix.with_name(prefix.name + "_runs.csv"))
    report = summarize(grouped, variants)
    report_path = prefix.with_name(prefix.name + "_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    if not args.no_plot:
        plot_curves(grouped, variants, prefix.with_name(prefix.name + "_curves.png"))
        plot_tail5(grouped, variants, prefix.with_name(prefix.name + "_tail5.png"))
        plot_atari_diagnostic(grouped, variants, prefix.with_name(prefix.name + "_atari_deterministic.png"))


if __name__ == "__main__":
    main()
