"""Analyze the 330-run true-vs-five-partials calibration screen.

This stage describes performance and draws curves; it deliberately does not
select a prior or inspect any RLHF result. Choose one candidate per environment
after reviewing this report, then record those choices for the confirmatory
composition experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CELLS = (
    "ll", "reacher", "pusher", "swimmer", "hopper", "bipedal",
    "walker", "ant", "mspacman", "qbert", "pong",
)


def med(values) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def read_expected(path: Path) -> dict[tuple[str, str, int], str]:
    expected = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 4:
            raise SystemExit(f"{path}:{line_number}: expected four fields, got {len(fields)}")
        cell, variant, seed_text, reference = fields
        key = (cell, variant, int(seed_text))
        if key in expected:
            raise SystemExit(f"duplicate expected run: {key}")
        expected[key] = reference
    if len(expected) != 330:
        raise SystemExit(f"{path} must describe 330 runs, found {len(expected)}")
    return expected


def component_pcc(run_dir: Path) -> float:
    path = run_dir / "eval" / "component_evaluations.csv"
    if not path.exists():
        return float("nan")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("mean_total") not in (None, "") and row.get("mean_partial") not in (None, "")
        ]
    if len(rows) < 3:
        return float("nan")
    true_values = np.asarray([float(row["mean_total"]) for row in rows])
    partial_values = np.asarray([float(row["mean_partial"]) for row in rows])
    if true_values.std() == 0 or partial_values.std() == 0:
        return float("nan")
    return float(np.corrcoef(true_values, partial_values)[0, 1])


def load(root: Path) -> list[dict]:
    runs = []
    for meta_path in sorted(root.glob("ntscreen_*/*/metadata.json")):
        run_dir = meta_path.parent
        _, cell, variant = run_dir.parent.name.split("_", 2)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        curve_path = run_dir / "eval" / "evaluations.npz"
        if not curve_path.exists():
            continue
        data = np.load(curve_path)
        results = np.asarray(data["results"], dtype=np.float64)
        timesteps = np.asarray(data["timesteps"], dtype=np.int64)
        if results.size == 0:
            continue
        means = results.mean(axis=1)
        peak_index = int(np.argmax(means))
        runs.append({
            "cell": cell,
            "variant": variant,
            "seed": int(meta["seed"]),
            "env_id": meta["env_id"],
            "mode": meta["mode"],
            "partial_reference": meta.get("partial_reference") or "-",
            "final": float(means[-1]),
            "peak": float(means[peak_index]),
            "peak_at": int(timesteps[peak_index]),
            "start": float(means[0]),
            "pcc": component_pcc(run_dir),
            "run_dir": str(run_dir),
            "timesteps": timesteps,
            "means": means,
        })
    return runs


def validate_complete(runs: list[dict], expected: dict[tuple[str, str, int], str]) -> None:
    actual = {}
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
    if duplicates or missing or extras or mismatched:
        print(
            f"INCOMPLETE/INVALID SCREEN: missing={len(missing)} extras={len(extras)} "
            f"duplicates={len(duplicates)} refs={len(mismatched)}"
        )
        for label, values in (("missing", missing), ("extras", extras), ("duplicates", duplicates), ("references", mismatched)):
            if values:
                print(f"  {label}: {values[:12]}")
        raise SystemExit(2)


def write_csv(runs: list[dict], path: Path) -> None:
    fields = (
        "cell", "env_id", "variant", "mode", "partial_reference", "seed",
        "final", "peak", "peak_at", "start", "pcc", "run_dir",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda item: (item["cell"], item["variant"], item["seed"])):
            writer.writerow({field: run[field] for field in fields})
    print(f"wrote {path} ({len(runs)} runs)")


def report(runs: list[dict], expected: dict[tuple[str, str, int], str]) -> None:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["cell"], run["variant"]), []).append(run)

    for cell in CELLS:
        true = grouped[(cell, "true")]
        true_final = med([run["final"] for run in true])
        true_start = med([run["start"] for run in true])
        span = true_final - true_start
        variants = []
        for expected_cell, variant, _ in expected:
            if expected_cell == cell and variant != "true" and variant not in variants:
                variants.append(variant)

        print("\n" + "=" * 102)
        print(f"{cell}: true start {true_start:.1f}, true final {true_final:.1f}, span {span:.1f}")
        print(f"{'candidate':30s} {'final':>11s} {'peak':>11s} {'range frac':>11s} {'PCC':>8s}  reference")
        for variant in variants:
            rows = grouped[(cell, variant)]
            final = med([run["final"] for run in rows])
            peak = med([run["peak"] for run in rows])
            fraction = (final - true_start) / span if span > 0 else float("nan")
            pcc = med([run["pcc"] for run in rows if np.isfinite(run["pcc"])])
            reference = rows[0]["partial_reference"]
            print(f"{variant:30s} {final:11.1f} {peak:11.1f} {fraction:11.2f} {pcc:8.2f}  {reference}")


def plot(runs: list[dict], expected: dict[tuple[str, str, int], str], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["cell"], run["variant"]), []).append(run)
    fig, axes = plt.subplots(6, 2, figsize=(16, 24), squeeze=False)
    for index, cell in enumerate(CELLS):
        ax = axes[index // 2][index % 2]
        variants = ["true"]
        for expected_cell, variant, _ in expected:
            if expected_cell == cell and variant not in variants:
                variants.append(variant)
        for variant in variants:
            rows = grouped[(cell, variant)]
            length = min(len(run["means"]) for run in rows)
            stack = np.vstack([run["means"][:length] for run in rows])
            center = np.median(stack, axis=0)
            low, high = np.percentile(stack, [25, 75], axis=0)
            color = "black" if variant == "true" else None
            line = ax.plot(
                rows[0]["timesteps"][:length], center, label=variant,
                color=color, linewidth=2.2 if variant == "true" else 1.5,
                linestyle="-" if variant == "true" else "--",
            )[0]
            ax.fill_between(rows[0]["timesteps"][:length], low, high, color=line.get_color(), alpha=0.10)
        ax.set_title(cell)
        ax.set_xlabel("timesteps")
        ax.set_ylabel("true reward")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=2, frameon=False)
    axes[-1][-1].axis("off")
    fig.suptitle("Partial calibration: true reward (black) vs five hand-written partials")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs")
    parser.add_argument("--params", default="jobs/params_ntscreen.txt")
    parser.add_argument("--csv", default="logs/ntscreen_summary.csv")
    parser.add_argument("--figure", default="logs/ntscreen_curves.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    expected = read_expected(Path(args.params))
    runs = load(Path(args.root))
    validate_complete(runs, expected)
    write_csv(runs, Path(args.csv))
    report(runs, expected)
    if not args.no_plot:
        plot(runs, expected, Path(args.figure))
    print("\nNo candidate was auto-selected; review the curves and summary before the confirmatory grid.")


if __name__ == "__main__":
    main()
