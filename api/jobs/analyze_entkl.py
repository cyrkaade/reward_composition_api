"""Read back the ent/KL ceiling repair against the free 1M control.

    python jobs/analyze_entkl.py                 # tables + CSV + figure
    python jobs/analyze_entkl.py --cell walker
    python jobs/analyze_entkl.py --no-plot

THE CONTROL
-----------
Not a re-run. logs/sel_*_true holds 2M mode=true runs whose first 1M is identical
to a 1M run: learning_rate and clip_range resolve to floats, not schedules, and
mode=true is a single learn() call, so total_timesteps changes nothing before the
1M mark. evaluations.npz carries 100 points at eval_freq 20000 and index 49 is
exactly 1000000, so the control curve is points 0..49 of the archived run at the
same seed. If that assumption ever breaks the arms become uninterpretable, so it
is asserted at load time rather than trusted.

WHAT TO READ
------------
drawdown = (peak - final) / |peak| is the headline. The grid's failure on Walker2d
(0.72), Ant (1.03) and HumanoidStandup (0.48) is a RETENTION failure -- those cells
reached good peaks and then lost them -- so an arm that lowers drawdown while
holding peak is the win condition, and an arm that lowers drawdown by never
climbing is not.

log_std is the mechanism readout. Init is 0.0 in every arm; the control ends at
+11.3 on standup, +6.9 on ant, +4.7 on walker. If ent0 keeps it near or below 0
and kl does not, the damage is the entropy bonus itself rather than update size.
The control's log_std is only available at 2M (its saved final_model.zip), so it
is printed as a reference endpoint, NOT as a matched 1M comparison.

FIVE SEEDS CANNOT TEST ANYTHING -- the exact two-sided Wilcoxon floor at n=5 is
0.0625. Medians, per-seed spreads and win counts only; no p-values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CELLS = ("ll", "reacher", "pusher", "swimmer", "cheetah", "walker", "ant", "standup")
ARMS = ("ctrl", "ent0", "kl")

ENV_LABEL = {
    "ll": "LunarLander-v3",
    "reacher": "Reacher-v5",
    "pusher": "Pusher-v5",
    "swimmer": "Swimmer-v5",
    "cheetah": "HalfCheetah-v5",
    "walker": "Walker2d-v5",
    "ant": "Ant-v5",
    "standup": "HumanoidStandup-v5",
}

# The control is the archived 2M true arm truncated to wherever the new arms
# actually stopped. Derived from the runs themselves rather than hard-coded, so
# the same analyzer works whether run_entkl.sh was submitted at 1M or 2M.
DEFAULT_CUTOFF = 1_000_000


def final_log_std(run_dir: Path) -> float:
    """Max log_std of the saved policy, or nan. Torch load is optional here so the
    tables still print on a machine without the model files fetched."""
    model_path = run_dir / "final_model.zip"
    if not model_path.exists():
        return float("nan")
    try:
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path), device="cpu")
        return float(model.policy.log_std.detach().max())
    except Exception:
        return float("nan")


def curve_stats(point_means: np.ndarray, timesteps: np.ndarray) -> dict:
    peak_index = int(np.argmax(point_means))
    peak = float(point_means[peak_index])
    final = float(point_means[-1])
    quarter = len(point_means) // 4
    q3 = float(point_means[len(point_means) // 2 : 3 * quarter].mean()) if quarter else float("nan")
    q4 = float(point_means[3 * quarter :].mean()) if quarter else float("nan")
    return {
        "final": final,
        "peak": peak,
        "peak_at": int(timesteps[peak_index]),
        "peak_frac": float(peak_index) / len(point_means),
        "drawdown": (peak - final) / abs(peak) if peak else float("nan"),
        "trend": q4 - q3,
    }


def load(root: Path, want_log_std: bool) -> list[dict]:
    runs: list[dict] = []

    # --- the two new arms -------------------------------------------------
    for meta_path in sorted(root.glob("entkl_*/*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_dir = meta_path.parent
        _, cell, arm = run_dir.parent.name.split("_", 2)
        curve_path = run_dir / "eval" / "evaluations.npz"
        if not curve_path.exists():
            print(f"warning: no evaluations.npz for {run_dir}")
            continue
        data = np.load(curve_path)
        point_means = np.asarray(data["results"], dtype=np.float64).mean(axis=1)
        timesteps = np.asarray(data["timesteps"], dtype=np.int64)
        row = {
            "cell": cell,
            "arm": arm,
            "seed": int(meta["seed"]),
            "env_id": meta["env_id"],
            "ent_coef": meta["policy_learning_kwargs"].get("ent_coef"),
            "target_kl": meta["policy_learning_kwargs"].get("target_kl"),
            "log_std_max": final_log_std(run_dir) if want_log_std else float("nan"),
            "log_std_at": "1M",
            "point_means": point_means,
            "timesteps": timesteps,
        }
        row.update(curve_stats(point_means, timesteps))
        runs.append(row)

    # Truncate the control to wherever the new arms actually stopped, so the two
    # curves cover the same horizon. With no new arms yet, fall back to 1M.
    cutoff = max((int(r["timesteps"][-1]) for r in runs), default=DEFAULT_CUTOFF)
    print(f"control truncated at {cutoff} timesteps (from the entkl_* arms' own horizon)")

    # --- the control, truncated out of the archived 2M runs ---------------
    for meta_path in sorted(root.glob("sel_*_true/*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_dir = meta_path.parent
        _, cell, _ = run_dir.parent.name.split("_", 2)
        curve_path = run_dir / "eval" / "evaluations.npz"
        if not curve_path.exists():
            continue
        data = np.load(curve_path)
        timesteps = np.asarray(data["timesteps"], dtype=np.int64)
        hits = np.where(timesteps == cutoff)[0]
        if hits.size == 0:
            raise SystemExit(
                f"{run_dir}: no eval point at exactly {cutoff}. The free-control "
                "assumption does not hold for this run; re-run the control explicitly."
            )
        cut = int(hits[0]) + 1
        point_means = np.asarray(data["results"], dtype=np.float64).mean(axis=1)[:cut]
        row = {
            "cell": cell,
            "arm": "ctrl",
            "seed": int(meta["seed"]),
            "env_id": meta["env_id"],
            "ent_coef": meta["policy_learning_kwargs"].get("ent_coef"),
            "target_kl": meta["policy_learning_kwargs"].get("target_kl"),
            # the archived model is the 2M endpoint, NOT the 1M state
            "log_std_max": final_log_std(run_dir) if want_log_std else float("nan"),
            "log_std_at": "2M",
            "point_means": point_means,
            "timesteps": timesteps[:cut],
        }
        row.update(curve_stats(point_means, timesteps[:cut]))
        runs.append(row)

    return runs


def med(values) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def report(runs: list[dict], cells) -> None:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["cell"], run["arm"]), []).append(run)

    print("\n" + "=" * 112)
    print("ENT/KL REPAIR -- ctrl is the archived 2M true arm truncated at 1M, matched seeds")
    print("=" * 112)
    for cell in cells:
        ctrl = grouped.get((cell, "ctrl"), [])
        if not ctrl:
            continue
        base_final, base_peak = med([r["final"] for r in ctrl]), med([r["peak"] for r in ctrl])
        print(f"\n--- {cell} / {ENV_LABEL[cell]}")
        print(
            f"    {'arm':6s} {'n':>2s} {'final':>10s} {'d final':>9s} {'peak':>10s} {'d peak':>9s} "
            f"{'peak@':>8s} {'drawdown':>9s} {'trend':>10s} {'log_std':>9s} {'better':>7s}"
        )
        for arm in ARMS:
            arm_runs = grouped.get((cell, arm), [])
            if not arm_runs:
                continue
            finals = [r["final"] for r in arm_runs]
            peaks = [r["peak"] for r in arm_runs]
            dd = [r["drawdown"] for r in arm_runs]
            ls = [r["log_std_max"] for r in arm_runs if np.isfinite(r["log_std_max"])]
            better = "-" if arm == "ctrl" else f"{sum(1 for v in finals if v > base_final)}/{len(finals)}"
            ls_text = f"{med(ls):+7.2f}{arm_runs[0]['log_std_at']:>2s}" if ls else "      nan"
            print(
                f"    {arm:6s} {len(arm_runs):2d} {med(finals):10.1f} "
                f"{'' if arm == 'ctrl' else format(med(finals) - base_final, '+9.1f'):>9s} "
                f"{med(peaks):10.1f} "
                f"{'' if arm == 'ctrl' else format(med(peaks) - base_peak, '+9.1f'):>9s} "
                f"{med([r['peak_frac'] for r in arm_runs]):8.2f} {med(dd):9.2f} "
                f"{med([r['trend'] for r in arm_runs]):+10.1f} {ls_text:>9s} {better:>7s}"
            )
        for arm in ("ent0", "kl"):
            arm_runs = sorted(grouped.get((cell, arm), []), key=lambda r: r["seed"])
            if arm_runs:
                print(f"      {arm:5s} per-seed final: {[round(r['final'], 1) for r in arm_runs]}")
    print(
        "\ndrawdown = (peak-final)/|peak|; the grid's failures are RETENTION failures, so a\n"
        "win lowers drawdown while HOLDING peak. trend = last quarter minus the quarter\n"
        "before it: negative means still falling at 1M. log_std is tagged 1M for the new\n"
        "arms and 2M for ctrl (only its 2M model was saved) -- not a matched comparison.\n"
        "n=5: medians and win counts only, no p-values."
    )


def write_csv(runs: list[dict], path: Path) -> None:
    fields = ["cell", "env_id", "arm", "seed", "ent_coef", "target_kl", "final", "peak",
              "peak_at", "peak_frac", "drawdown", "trend", "log_std_max", "log_std_at"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda r: (r["cell"], r["arm"], r["seed"])):
            writer.writerow({field: run[field] for field in fields})
    print(f"wrote per-seed CSV: {path}  ({len(runs)} runs)")


def plot(runs: list[dict], cells, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["cell"], run["arm"]), []).append(run)
    live = [c for c in cells if (c, "ctrl") in grouped]
    if not live:
        print("nothing to plot")
        return
    ncols = 2
    nrows = (len(live) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 4.2 * nrows), squeeze=False)
    colors = {"ctrl": "black", "ent0": "tab:blue", "kl": "tab:orange"}
    for index, cell in enumerate(live):
        ax = axes[index // ncols][index % ncols]
        for arm in ARMS:
            arm_runs = grouped.get((cell, arm), [])
            if not arm_runs:
                continue
            length = min(len(r["point_means"]) for r in arm_runs)
            stack = np.vstack([r["point_means"][:length] for r in arm_runs])
            x = arm_runs[0]["timesteps"][:length]
            ax.plot(x, np.median(stack, axis=0), color=colors[arm], label=arm,
                    linewidth=2.2 if arm == "ctrl" else 1.6)
            if len(arm_runs) > 2:
                ax.fill_between(x, np.percentile(stack, 25, axis=0), np.percentile(stack, 75, axis=0),
                                color=colors[arm], alpha=0.12, linewidth=0)
        ax.set_title(f"{cell} / {ENV_LABEL[cell]}")
        ax.set_xlabel("timesteps")
        ax.set_ylabel("true reward")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    for spare in range(len(live), nrows * ncols):
        axes[spare // ncols][spare % ncols].axis("off")
    fig.suptitle("ent/KL repair at 1M: ctrl (black) vs ent_coef 0 (blue) vs target_kl 0.03 (orange)", y=1.0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figure: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs")
    parser.add_argument("--cell", action="append", choices=CELLS)
    parser.add_argument("--csv", default="logs/entkl_summary.csv")
    parser.add_argument("--figure", default="logs/entkl_curves.png")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-log-std", action="store_true",
                        help="skip loading final_model.zip (faster; log_std column becomes nan)")
    args = parser.parse_args()

    runs = load(Path(args.root), want_log_std=not args.no_log_std)
    if not runs:
        print(f"no entkl_* or sel_*_true runs under {Path(args.root).resolve()}")
        return
    cells = tuple(args.cell) if args.cell else CELLS
    runs = [r for r in runs if r["cell"] in cells]

    short = {}
    for run in runs:
        short[(run["cell"], run["arm"])] = short.get((run["cell"], run["arm"]), 0) + 1
    incomplete = [f"{c}/{a} ({n}/5)" for (c, a), n in sorted(short.items()) if n != 5]
    if incomplete:
        print(f"\nINCOMPLETE ARMS (not 5 seeds): {', '.join(incomplete)}")

    write_csv(runs, Path(args.csv))
    report(runs, cells)
    if not args.no_plot:
        plot(runs, cells, Path(args.figure))


if __name__ == "__main__":
    main()
