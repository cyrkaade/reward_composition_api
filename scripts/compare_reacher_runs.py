from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "env_id",
    "seed",
    "timesteps",
    "mode",
    "query_budget",
    "current_selected_true_reward_mean",
    "main_selected_true_reward_mean",
    "current_minus_main",
    "winner",
    "current_selected_true_reward_std",
    "main_selected_true_reward_std",
    "current_best_logged_true_reward",
    "main_best_logged_true_reward",
    "current_synthetic_queries",
    "main_synthetic_queries",
    "current_run_dir",
    "main_run_dir",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare current and main Reacher experiment metadata.")
    parser.add_argument("--root", required=True, help="Log root containing current/ and main/ subfolders.")
    parser.add_argument("--out", default=None, help="CSV output path. Defaults to <root>/comparison.csv.")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "comparison.csv"
    current = load_version(root / "current")
    main_rows = load_version(root / "main")
    rows = compare_rows(current, main_rows, args.tolerance)
    write_csv(out, rows)
    print_summary(rows, out)
    return 0


def load_version(root: Path) -> dict[tuple, dict]:
    rows = {}
    for metadata_path in sorted(root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        key = (
            metadata.get("env_id"),
            as_int(metadata.get("seed")),
            metadata.get("mode"),
            as_int(metadata.get("query_budget")),
            as_int(metadata.get("requested_timesteps")),
        )
        rows[key] = {
            "metadata": metadata,
            "run_dir": str(metadata_path.parent),
        }
    return rows


def compare_rows(current: dict[tuple, dict], main_rows: dict[tuple, dict], tolerance: float) -> list[dict]:
    rows = []
    for key in sorted(set(current) | set(main_rows), key=sort_key):
        env_id, seed, mode, query_budget, timesteps = key
        current_item = current.get(key)
        main_item = main_rows.get(key)
        current_metadata = (current_item or {}).get("metadata", {})
        main_metadata = (main_item or {}).get("metadata", {})
        current_mean = as_float(current_metadata.get("selected_policy_true_reward_mean"))
        main_mean = as_float(main_metadata.get("selected_policy_true_reward_mean"))
        delta = None if current_mean is None or main_mean is None else current_mean - main_mean
        rows.append(
            {
                "env_id": env_id,
                "seed": seed,
                "timesteps": timesteps,
                "mode": "vanilla" if mode == "feedback" else mode,
                "query_budget": query_budget,
                "current_selected_true_reward_mean": current_mean,
                "main_selected_true_reward_mean": main_mean,
                "current_minus_main": delta,
                "winner": winner(delta, tolerance),
                "current_selected_true_reward_std": as_float(current_metadata.get("selected_policy_true_reward_std")),
                "main_selected_true_reward_std": as_float(main_metadata.get("selected_policy_true_reward_std")),
                "current_best_logged_true_reward": as_float(current_metadata.get("best_logged_true_reward")),
                "main_best_logged_true_reward": as_float(main_metadata.get("best_logged_true_reward")),
                "current_synthetic_queries": as_int(current_metadata.get("synthetic_queries")),
                "main_synthetic_queries": as_int(main_metadata.get("synthetic_queries")),
                "current_run_dir": (current_item or {}).get("run_dir"),
                "main_run_dir": (main_item or {}).get("run_dir"),
            }
        )
    return rows


def sort_key(key: tuple) -> tuple:
    env_id, seed, mode, query_budget, timesteps = key
    mode_order = {"feedback": 0, "naive": 1, "delta": 2}
    return (str(env_id), seed or -1, timesteps or -1, mode_order.get(mode, 99), query_budget or -1)


def winner(delta: float | None, tolerance: float) -> str:
    if delta is None:
        return "missing"
    if abs(delta) <= tolerance:
        return "tie"
    return "current" if delta > 0 else "main"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def print_summary(rows: list[dict], out: Path) -> None:
    print(f"wrote comparison to {out}")
    if not rows:
        print("no metadata rows found yet")
        return
    print("mode      q     current        main          delta       winner")
    for row in rows:
        print(
            f"{row['mode']:<8} "
            f"{row['query_budget']:>4} "
            f"{format_float(row['current_selected_true_reward_mean']):>12} "
            f"{format_float(row['main_selected_true_reward_mean']):>12} "
            f"{format_float(row['current_minus_main']):>12} "
            f"{row['winner']}"
        )
    if any(row["winner"] not in {"tie", "missing"} for row in rows):
        print("different final true rewards detected; higher selected true reward is treated as better")


def format_float(value) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def as_float(value):
    if value is None or value == "":
        return None
    return float(value)


def as_int(value):
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
