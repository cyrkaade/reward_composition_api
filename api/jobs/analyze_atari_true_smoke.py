"""Summarize whether true-reward pixel PPO improves in the Atari smoke grid."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


CELLS = ("mspacman", "qbert", "pong")
SEEDS = range(5)
ROOT = Path("logs")


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    timesteps = np.asarray(data["timesteps"], dtype=np.int64)
    means = np.asarray(data["results"], dtype=np.float64).mean(axis=1)
    return timesteps, means


def load_runs() -> list[dict]:
    runs = []
    for cell in CELLS:
        for seed in SEEDS:
            run_dir = ROOT / f"atari_true_smoke_{cell}" / f"atari_true_smoke_{cell}_seed{seed}"
            metadata_path = run_dir / "metadata.json"
            deterministic_path = run_dir / "eval" / "evaluations.npz"
            stochastic_path = run_dir / "eval_stochastic" / "evaluations.npz"
            if not metadata_path.exists() or not deterministic_path.exists() or not stochastic_path.exists():
                raise SystemExit(f"missing completed dual-evaluation smoke run: {run_dir}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            timesteps, means = load_curve(stochastic_path)
            det_timesteps, det_means = load_curve(deterministic_path)
            if means.size < 2 or det_means.size < 2:
                raise SystemExit(f"too few evaluations in {run_dir}")
            if means.size < 5 or det_means.size < 5:
                raise SystemExit(f"need at least five evaluations for stable endpoints in {run_dir}")
            head = float(means[:5].mean())
            tail = float(means[-5:].mean())
            det_head = float(det_means[:5].mean())
            det_tail = float(det_means[-5:].mean())
            runs.append({
                "cell": cell,
                "env_id": metadata["env_id"],
                "seed": seed,
                "start": head,
                "final": tail,
                "peak": float(means.max()),
                "delta": tail - head,
                "timesteps": timesteps,
                "means": means,
                "det_start": det_head,
                "det_final": det_tail,
                "det_peak": float(det_means.max()),
                "det_delta": det_tail - det_head,
                "det_timesteps": det_timesteps,
                "det_means": det_means,
            })
    return runs


def write_outputs(runs: list[dict]) -> None:
    csv_path = ROOT / "atari_true_smoke_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "cell", "env_id", "seed", "start", "final", "peak", "delta",
            "det_start", "det_final", "det_peak", "det_delta",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: run[field] for field in fields})

    report = {
        "timesteps": 400_000,
        "seeds": list(SEEDS),
        "endpoint": "mean of first five versus mean of last five checkpoints per seed",
        "environments": {},
    }
    for cell in CELLS:
        rows = [run for run in runs if run["cell"] == cell]
        starts = np.asarray([run["start"] for run in rows])
        finals = np.asarray([run["final"] for run in rows])
        deltas = finals - starts
        det_starts = np.asarray([run["det_start"] for run in rows])
        det_finals = np.asarray([run["det_final"] for run in rows])
        det_deltas = det_finals - det_starts
        median_start = float(np.median(starts))
        median_delta = float(np.median(deltas))
        # A one-point absolute and five-percent relative floor prevents a tiny
        # score wobble (Pong's +0.2) from being called learning.  Five seeds are
        # still a descriptive smoke check, so require consistency in 4/5 seeds.
        practical_threshold = max(1.0, 0.05 * abs(median_start))
        passed = bool(median_delta >= practical_threshold and np.count_nonzero(deltas > 0) >= 4)
        report["environments"][cell] = {
            "primary_evaluation_policy": "stochastic",
            "median_start": median_start,
            "median_final": float(np.median(finals)),
            "median_delta": median_delta,
            "practical_improvement_threshold": practical_threshold,
            "improved_seeds": int(np.count_nonzero(deltas > 0)),
            "deterministic_median_start": float(np.median(det_starts)),
            "deterministic_median_final": float(np.median(det_finals)),
            "deterministic_median_delta": float(np.median(det_deltas)),
            "deterministic_improved_seeds": int(np.count_nonzero(det_deltas > 0)),
            "passed": passed,
        }
        print(
            f"{cell:9s} stochastic start={np.median(starts):8.1f} final={np.median(finals):8.1f} "
            f"delta={median_delta:+8.1f} threshold={practical_threshold:6.1f} "
            f"improved={np.count_nonzero(deltas > 0)}/5 "
            f"{'PASS' if passed else 'NO CLEAR IMPROVEMENT'}; "
            f"deterministic final={np.median(det_finals):8.1f} delta={np.median(det_deltas):+8.1f}"
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
        det_length = min(len(run["det_means"]) for run in rows)
        det_stack = np.vstack([run["det_means"][:det_length] for run in rows])
        det_x = rows[0]["det_timesteps"][:det_length]
        det_center = np.median(det_stack, axis=0)
        det_low, det_high = np.percentile(det_stack, [25, 75], axis=0)
        ax.plot(det_x, det_center, color="tab:gray", linewidth=1.6, linestyle="--", label="deterministic diagnostic")
        ax.fill_between(det_x, det_low, det_high, color="tab:gray", alpha=0.10)
        ax.set_title(cell)
        ax.set_xlabel("timesteps")
        ax.set_ylabel("true reward")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[-1][-1].axis("off")
    fig.suptitle("Atari pixel-policy true reward: stochastic primary vs deterministic diagnostic")
    fig.tight_layout()
    figure_path = ROOT / "atari_true_smoke_curves.png"
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {csv_path}, {json_path}, and {figure_path}")


if __name__ == "__main__":
    write_outputs(load_runs())
