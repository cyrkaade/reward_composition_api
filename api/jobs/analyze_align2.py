"""Analyze the supplied-partial grid with the existing STARC-grid renderer.

The 125 new runs live under ``logs/align2_*``.  The true and vanilla reference
arms do not depend on the partial, so this analyzer reads the already-completed
``logs/starc_ref_*`` controls and combines them with the new grid.  It produces
``logs/align2_grid_{curves,heatmap}.png`` and a JSON report by default.

Usage:
    python jobs/analyze_align2.py [--logs logs] [--out logs/align2_grid]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import analyze_starc as renderer


# Plot bottom-to-top in increasing table alignment.  The generator uses the
# opposite (image) order only to preserve the supplied row order in params.
renderer.RUNGS = ["a060", "a118", "a182", "a240", "a298"]
renderer.RUNG_ALIGN = {
    "a060": 0.0596099875867367,
    "a118": 0.117974838614464,
    "a182": 0.181834864616394,
    "a240": 0.240131175518036,
    "a298": 0.297506305575371,
}

_load_starc_runs = renderer.load_runs


def _load_align2_runs(root: Path):
    runs = []
    for meta_path in sorted(root.glob("align2_*/*/metadata.json")):
        cell_arm = meta_path.parent.parent.name[len("align2_"):]
        cell, _, arm = cell_arm.partition("_")
        if not arm:
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        steps = curve = None
        npz = meta_path.parent / "eval" / "evaluations.npz"
        if npz.exists():
            try:
                data = np.load(npz)
                steps = np.asarray(data["timesteps"], dtype=float)
                curve = np.asarray(data["results"], dtype=float).mean(axis=1)
            except Exception:
                steps = curve = None
        runs.append({
            "cell": cell,
            "arm": arm,
            "base": arm.rpartition("_q")[0] or arm,
            "seed": meta.get("seed"),
            "final": meta.get("selected_policy_true_reward_mean"),
            "peak": meta.get("best_logged_true_reward"),
            "queries": meta.get("synthetic_queries"),
            "budget": meta.get("query_budget") or 0,
            "alpha": meta.get("partial_alpha"),
            "mode": meta.get("mode"),
            "steps": steps,
            "curve": curve,
            "tail5": float(np.mean(curve[-5:])) if curve is not None and len(curve) else None,
        })
    return runs


def load_runs(root: Path):
    grid = _load_align2_runs(root)
    refs = [run for run in _load_starc_runs(root) if run["cell"] == "ref"]
    return grid + refs


renderer.load_runs = load_runs


if __name__ == "__main__":
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", "logs/align2_grid"])
    renderer.main()
