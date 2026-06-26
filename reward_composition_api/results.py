from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    metadata_path: Path
    model_path: Path
    vecnormalize_path: Path | None = None
    synthetic_queries: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedRun:
    env_id: str
    seed: int
    variant: str
    mode: str
    run_dir: Path
    completed: bool
    command: list[str]

    def as_manifest_row(self) -> dict:
        return {
            "env_id": self.env_id,
            "seed": self.seed,
            "variant": self.variant,
            "mode": self.mode,
            "run_dir": str(self.run_dir),
            "completed": self.completed,
            "command": self.command,
        }


@dataclass(frozen=True)
class SweepResult:
    manifest_path: Path
    planned_runs: list[PlannedRun]
    pending_runs: list[PlannedRun]
    executed: bool


@dataclass(frozen=True)
class SummaryResult:
    summary_csv: Path
    aggregate_csv: Path
    rows: list[dict]
    aggregate_rows: list[dict]
