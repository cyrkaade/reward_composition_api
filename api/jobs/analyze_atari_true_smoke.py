"""Summarize whether true-reward pixel PPO improves in the Atari smoke grid."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


CELLS = ("mspacman", "qbert", "pong")
SEEDS = range(5)
ROOT = Path("logs")


def load_runs() -> list[dict]:
    runs = []
    for cell in CELLS:
        for seed in SEEDS:
            run_dir = ROOT / f"atari_true_smoke_{cell}" / f"atari_true_smoke_{cell}_seed{seed}"
            metadata_path = run_dir / "metadata.json"
            curve_path = run_dir / "eval" / "evaluations.npz"
            if not metadata_path.exists() or not curve_path.exists():
                raise SystemExit(f"missing completed smoke run: {run_dir}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            data = np.load(curve_path)
            timesteps = np.asarray(data["timesteps"], dtype=np.int64)
            means = np.asarray(data["results"], dtype=np.float64).mean(axis=1)
            if means.size < 2:
                raise SystemExit(f"too few evaluations in {run_dir}")
            runs.append({
                "cell": cell,
                "env_id": metadata["env_id"],
                "seed": seed,
                "start": float(means[0]),
                "final": float(means[-1]),
                "peak": float(means.max()),
                "delta": float(means[-1] - means[0]),
                "timesteps": timesteps,
                "means": means,
            })
    return runs


def write_outputs(runs: list[dict]) -> None:
    csv_path = ROOT / "atari_true_smoke_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("cell", "env_id", "seed", "start", "final", "peak", "delta")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: run[field] for field in fields})

    report = {"timesteps": 400_000, "seeds": list(SEEDS), "environments": {}}
    for cell in CELLS:
        rows = [run for run in runs if run["cell"] == cell]
        starts = np.asarray([run["start"] for run in rows])
        finals = np.asarray([run["final"] for run in rows])
        deltas = finals - starts
        passed = bool(np.median(deltas) > 0 and np.count_nonzero(deltas > 0) >= 3)
        report["environments"][cell] = {
            "median_start": float(np.median(starts)),
            "median_final": float(np.median(finals)),
            "median_delta": float(np.median(deltas)),
            "improved_seeds": int(np.count_nonzero(deltas > 0)),
            "passed": passed,
        }
        print(
            f"{cell:9s} start={np.median(starts):8.1f} final={np.median(finals):8.1f} "
            f"delta={np.median(deltas):+8.1f} improved={np.count_nonzero(deltas > 0)}/5 "
            f"{'PASS' if passed else 'NO CLEAR IMPROVEMENT'}"
        )
    report["all_passed"] = all(item["passed"] for item in report["environments"].values())
    json_path = ROOT / "atari_true_smoke_report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, cell in zip(axes.flat, CELLS):
        rows = [run for run in runs if run["cell"] == cell]
        length = min(len(run["means"]) for run in rows)
        stack = np.vstack([run["means"][:length] for run in rows])
        x = rows[0]["timesteps"][:length]
        center = np.median(stack, axis=0)
        low, high = np.percentile(stack, [25, 75], axis=0)
        ax.plot(x, center, color="black", linewidth=2, label="median true reward")
        ax.fill_between(x, low, high, color="black", alpha=0.15, label="IQR")
        ax.set_title(cell)
        ax.set_xlabel("timesteps")
        ax.set_ylabel("true reward")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[-1][-1].axis("off")
    fig.suptitle("Atari pixel-policy true-reward smoke test (5 seeds, 400k steps)")
    fig.tight_layout()
    figure_path = ROOT / "atari_true_smoke_curves.png"
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {csv_path}, {json_path}, and {figure_path}")


if __name__ == "__main__":
    write_outputs(load_runs())
