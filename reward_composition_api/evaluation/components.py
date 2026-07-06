from __future__ import annotations

import numpy as np


def summarize_component_rows(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    stats = {}
    for key in keys:
        values = np.asarray([row.get(key, 0.0) for row in rows], dtype=np.float64)
        stats[f"mean_{key}"] = float(values.mean())
        stats[f"std_{key}"] = float(values.std())
    return stats
