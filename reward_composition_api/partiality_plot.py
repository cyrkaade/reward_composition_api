from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from reward_composition_api.errors import ConfigError


@dataclass(frozen=True)
class PartialityGridConfig:
    runs_root: str | Path = "logs"
    partiality_root: str | Path = Path("logs") / "partiality"
    output: str | Path = Path("logs") / "partiality" / "partiality_grid.png"
    env_id: str | None = None
    title: str = "Partiality vs RLHF queries"


def plot_partiality_grid(config: PartialityGridConfig) -> Path:
    partiality_by_key = load_partiality_index(Path(config.partiality_root))
    rows = load_grid_rows(Path(config.runs_root), partiality_by_key, config.env_id)
    if not rows:
        raise ConfigError("No matching run metadata could be joined with partiality results")

    table = build_grid(rows)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_grid(table, output, config.title)
    return output


def load_partiality_index(root: Path) -> dict[tuple[str, str], float]:
    index = {}
    if not root.exists():
        return index
    for path in sorted(root.rglob("*.json")):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        env_id = metrics.get("env_id")
        partial = metrics.get("partial")
        partiality = metrics.get("partiality_clipped_0_1", metrics.get("partiality"))
        if env_id and partial and partiality is not None:
            index[(str(env_id), _partial_name(str(partial)))] = float(partiality)
    return index


def load_grid_rows(runs_root: Path, partiality_by_key: dict[tuple[str, str], float], env_id: str | None = None) -> list[dict[str, Any]]:
    rows = []
    if not runs_root.exists():
        return rows
    for metadata_path in sorted(runs_root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        row_env = metadata.get("env_id")
        if env_id is not None and row_env != env_id:
            continue

        reward = metadata.get("selected_policy_true_reward_mean")
        if reward is None:
            continue

        partiality = _metadata_partiality(metadata, partiality_by_key)
        if partiality is None:
            continue

        rows.append(
            {
                "env_id": row_env,
                "partiality": float(partiality),
                "queries": int(metadata.get("query_budget") or 0),
                "reward": float(reward),
                "variant": metadata.get("variant", metadata.get("mode")),
                "run_dir": str(metadata_path.parent),
            }
        )
    return rows


def build_grid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        partiality = round(float(row["partiality"]), 2)
        queries = int(row["queries"])
        grouped[(queries, partiality)].append(float(row["reward"]))

    query_values = sorted({query for query, _partiality in grouped}, reverse=True)
    partiality_values = sorted({_partiality for _query, _partiality in grouped})
    values = np.full((len(query_values), len(partiality_values)), np.nan, dtype=np.float64)
    counts = np.zeros_like(values, dtype=np.int64)

    for row_idx, query in enumerate(query_values):
        for col_idx, partiality in enumerate(partiality_values):
            rewards = grouped.get((query, partiality), [])
            if rewards:
                values[row_idx, col_idx] = float(sum(rewards) / len(rewards))
                counts[row_idx, col_idx] = len(rewards)

    return {
        "queries": query_values,
        "partialities": partiality_values,
        "values": values,
        "counts": counts,
    }


def render_grid(table: dict[str, Any], output: Path, title: str) -> None:
    values = table["values"]
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("rainbow_r").copy()
    cmap.set_bad("#eeeeee")

    size = max(4.5, 0.7 * max(values.shape))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(masked, cmap=cmap, aspect="equal")

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Partiality")
    ax.set_ylabel("No. RLHF queries")
    ax.set_xticks(range(len(table["partialities"])))
    ax.set_xticklabels([_format_axis_value(value) for value in table["partialities"]])
    ax.set_yticks(range(len(table["queries"])))
    ax.set_yticklabels([_format_query_value(value) for value in table["queries"]])

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            if np.isfinite(values[row_idx, col_idx]):
                ax.text(col_idx, row_idx, f"{values[row_idx, col_idx]:.0f}", ha="center", va="center", color="white", fontsize=9)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Final reward")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _metadata_partiality(metadata: dict[str, Any], partiality_by_key: dict[tuple[str, str], float]) -> float | None:
    if metadata.get("mode") == "true":
        return 1.0
    partial_ref = metadata.get("partial_reference")
    if not partial_ref:
        return None
    return partiality_by_key.get((str(metadata.get("env_id")), _partial_name(str(partial_ref))))


def _partial_name(reference: str) -> str:
    if ":" in reference:
        return reference.rsplit(":", 1)[1]
    if "/" in reference:
        return reference.rsplit("/", 1)[1]
    return Path(reference).stem


def _format_axis_value(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_query_value(value: int) -> str:
    if value >= 1000 and value % 1000 == 0:
        return f"{value // 1000}k"
    return str(value)
