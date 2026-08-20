"""Generate the gated 1,870-run confirmatory composition grid.

The manually reviewed eleven-cell screen selects every partial. Confirmatory seeds 10..19 do
not overlap screen seeds 0..4. Per cell: true + partial, then
vanilla/naive/ws25/ws50/ws75 at q100/q200/q400. CPU and Atari/GPU subsets are
written alongside the master analysis file.
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS = ("ll", "reacher", "pusher", "swimmer", "hopper", "bipedal", "walker", "ant", "mspacman", "qbert", "pong")
ATARI_CELLS = {"mspacman", "qbert", "pong"}
SEEDS = range(10, 20)
BUDGETS = (100, 200, 400)
METHODS = ("vanilla", "naive", "ws25", "ws50", "ws75")
VARIANTS = ("true", "partial") + tuple(f"{method}_q{budget}" for budget in BUDGETS for method in METHODS)

selection_path = Path("logs/ntscreen_selection.json")
if not selection_path.exists():
    raise SystemExit(
        "logs/ntscreen_selection.json is missing; review the completed screen, then "
        "record one partial_reference per cell in that file"
    )
payload = json.loads(selection_path.read_text(encoding="utf-8"))
if payload.get("status") != "complete":
    failures = ", ".join(sorted(payload.get("failures", {}))) or "unknown"
    raise SystemExit(f"screen selection is not complete; ineligible cells: {failures}")
selections = payload.get("selections", {})
if set(selections) != set(CELLS):
    raise SystemExit(f"selection must contain exactly {CELLS}; got {tuple(sorted(selections))}")

references = {}
for cell in CELLS:
    reference = str(selections[cell].get("partial_reference") or "")
    if not reference or "timid" in reference.rsplit(":", 1)[-1].lower():
        raise SystemExit(f"invalid/non-independent selection for {cell}: {reference!r}")
    references[cell] = reference


def rows(cells) -> list[str]:
    return [f"{cell} {variant} {seed} {references[cell]}" for variant in VARIANTS for cell in cells for seed in SEEDS]


all_lines = rows(CELLS)
cpu_lines = rows(tuple(cell for cell in CELLS if cell not in ATARI_CELLS))
atari_lines = rows(tuple(cell for cell in CELLS if cell in ATARI_CELLS))
outputs = {
    "params_ntcomp.txt": (all_lines, 1870),
    "params_ntcomp_cpu.txt": (cpu_lines, 1360),
    "params_ntcomp_atari.txt": (atari_lines, 510),
}
for filename, (lines, expected) in outputs.items():
    path = Path(__file__).with_name(filename)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if len(lines) != expected:
        raise SystemExit(f"internal error: expected {expected} rows for {filename}, generated {len(lines)}")
    print(f"wrote {path} with {len(lines)} lines")
for cell in CELLS:
    print(f"  {cell:9s} {references[cell]}")
