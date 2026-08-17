"""Read back the selection grid: which environments qualify, which priors land
in the band, and the three-curve figure per environment.

    python jobs/analyze_sel.py                     # tables + CSV + figure
    python jobs/analyze_sel.py --cell ll           # one environment
    python jobs/analyze_sel.py --no-plot

WHAT QUALIFIES
--------------
Per environment the grid produces three kinds of curve, all scored on the TRUE
reward by an EvalCallback that runs on a raw env:

    true      the ceiling
    vanilla   preference RLHF with no prior, q350 -- the floor
    <prior>   the hand-written prior alone, 0 labels

A prior is a candidate when its curve sits clearly under the ceiling and at or
above the floor. "Clearly under" is scored as RANGE RECOVERED,

    partiality = (arm - start) / (true - start)

with `start` the first evaluation point of the cell's true arm, i.e. what an
essentially untrained policy scores. 1.0 means the prior is the true reward,
0.0 means it bought nothing. The target band is 0.2 to 0.8.

A second, post-training readout is printed next to it: PCC, the Pearson
correlation between the policy's partial return and its true return across the
run's evaluation checkpoints (eval/component_evaluations.csv, columns
mean_partial and mean_total). It is computed AFTER training from states the
trained policy actually visits, so unlike the random-policy ranking agreement
in the archive it reflects the on-distribution alignment of the prior: near
+1.0 means optimizing the prior moved the true reward with it the whole way,
near 0 or negative means the two objectives decoupled. It is a per-seed number;
the table shows the median over seeds. Only prior arms have it (the true and
vanilla arms carry no partial, their column is nan).

FIVE SEEDS CANNOT TEST ANYTHING. The exact two-sided Wilcoxon floor at n=5 is
0.0625, so no contrast here can reach p<0.05 however large it is. This script
therefore prints medians, per-seed spreads and win counts, and no p-values. It is
a screen; the chosen cells get seeds later.

READ THE ENVIRONMENT TABLE FIRST. Three cells carry known risks (HalfCheetah's
bimodality, Walker2d's true-arm spread, HumanoidStandup's narrow dynamic range),
and a prior contrast against an unstable ceiling means nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CELLS = ("ll", "reacher", "pusher", "swimmer", "cheetah", "hopper", "walker", "ant", "standup",
         "bipedal", "mspacman", "qbert")

ENV_LABEL = {
    "ll": "LunarLander-v3",
    "reacher": "Reacher-v5",
    "pusher": "Pusher-v5",
    "swimmer": "Swimmer-v5",
    "cheetah": "HalfCheetah-v5",
    "hopper": "Hopper-v5",
    "walker": "Walker2d-v5",
    "ant": "Ant-v5",
    "standup": "HumanoidStandup-v5",
    "bipedal": "BipedalWalker-v3",
    "mspacman": "ALE/MsPacman-v5",
    "qbert": "ALE/Qbert-v5",
}

BAND_LOW, BAND_HIGH = 0.2, 0.8


def load_runs(root: Path, prefix: str = "sel") -> list[dict]:
    runs = []
    for meta_path in sorted(root.glob(f"{prefix}_*/*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_dir = meta_path.parent
        # log dir is logs/sel_<cell>_<variant>; the variant can contain '_', the
        # cell cannot, so split once from the left.
        _, cell, variant = run_dir.parent.name.split("_", 2)

        curve_path = run_dir / "eval" / "evaluations.npz"
        if not curve_path.exists():
            print(f"warning: no evaluations.npz for {run_dir}")
            continue
        data = np.load(curve_path)
        timesteps = np.asarray(data["timesteps"], dtype=np.int64)
        results = np.asarray(data["results"], dtype=np.float64)  # (n_evals, n_episodes)
        if results.size == 0:
            print(f"warning: empty curve for {run_dir}")
            continue
        point_means = results.mean(axis=1)
        peak_index = int(np.argmax(point_means))

        # Post-training PCC partiality: corr(partial return, true return) over
        # the periodic component evaluations. nan when the arm logs no partial
        # (true/vanilla) or either series is constant.
        pcc = float("nan")
        comp_path = run_dir / "eval" / "component_evaluations.csv"
        if comp_path.exists():
            with comp_path.open(newline="", encoding="utf-8") as handle:
                comp_rows = [
                    row for row in csv.DictReader(handle)
                    if row.get("mean_total") not in (None, "") and row.get("mean_partial") not in (None, "")
                ]
            true_series = np.asarray([float(row["mean_total"]) for row in comp_rows], dtype=np.float64)
            partial_series = np.asarray([float(row["mean_partial"]) for row in comp_rows], dtype=np.float64)
            if len(true_series) >= 3 and true_series.std() > 0 and partial_series.std() > 0:
                pcc = float(np.corrcoef(partial_series, true_series)[0, 1])

        runs.append(
            {
                "cell": cell,
                "variant": variant,
                "seed": int(meta["seed"]),
                "env_id": meta["env_id"],
                "mode": meta["mode"],
                "partial_reference": meta.get("partial_reference") or "",
                "total_timesteps": int(meta["actual_timesteps"]),
                # final = the LAST policy (--final-policy last), scored over
                # --final-eval-episodes fresh episodes at the end of the run
                "final_reward": float(meta["selected_policy_true_reward_mean"]),
                "final_std": float(meta["selected_policy_true_reward_std"]),
                # curve statistics, from the periodic evaluations
                "curve_start": float(point_means[0]),
                "curve_final_mean": float(results[-1].mean()),
                "curve_final_std": float(results[-1].std()),
                "curve_final_median": float(np.median(results[-1])),
                "max_reward": float(point_means[peak_index]),
                "max_reward_timestep": int(timesteps[peak_index]),
                "pcc_partial_true": pcc,
                "query_budget": int(meta.get("query_budget") or 0),
                "synthetic_queries": int(meta.get("synthetic_queries") or 0),
                "timesteps": timesteps,
                "point_means": point_means,
                "run_dir": str(run_dir),
            }
        )
    return runs


def by_arm(runs: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["cell"], run["variant"]), []).append(run)
    for key in grouped:
        grouped[key].sort(key=lambda item: item["seed"])
    return grouped


def med(values) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def write_csv(runs: list[dict], path: Path) -> None:
    fields = [
        "cell", "env_id", "variant", "mode", "partial_reference", "seed",
        "total_timesteps", "final_reward", "final_std",
        "curve_final_mean", "curve_final_std", "curve_final_median",
        "max_reward", "max_reward_timestep", "curve_start", "pcc_partial_true",
        "query_budget", "synthetic_queries", "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda item: (item["cell"], item["variant"], item["seed"])):
            writer.writerow({field: run[field] for field in fields})
    print(f"wrote per-seed CSV: {path}  ({len(runs)} runs)")


def report_environments(grouped, cells) -> tuple[dict[str, tuple[float, float]], set[str]]:
    """Print the ceiling/floor table.

    Returns (anchors, broken). `anchors` maps cell -> (start, true_final) for
    cells where partiality is defined at all; `broken` holds the cells where it
    is NOT, i.e. where the true arm ENDS BELOW where it STARTED.

    That case is not hypothetical. In the first grid Ant's true arm started at
    +511.8 (a stand-still local optimum found before any real learning) and
    ended at -14.5, so span = -526.3. Dividing by a negative span flips the sign
    of every partiality in the cell: sant_cap10, which finished at -1275.9
    against a true arm at -14.5, scored partiality 3.40 and was labelled
    "BEATS THE TRUE ARM". It does not; it is 89x worse. Any cell whose span is
    <= 0 has no ceiling to measure against and is refused rather than scored.
    """
    print("\n" + "=" * 108)
    print("ENVIRONMENT QUALIFICATION -- read this before any prior contrast")
    print("=" * 108)
    print(
        f"{'cell':9s} {'env':22s} {'n':>2s} {'true final':>11s} {'seed min':>10s} "
        f"{'seed max':>10s} {'spread/med':>11s} {'start':>10s} {'vanilla':>10s} {'labels':>9s}"
    )
    anchors = {}
    broken: set[str] = set()
    for cell in cells:
        true_runs = grouped.get((cell, "true"), [])
        vanilla_runs = grouped.get((cell, "vanilla"), [])
        if not true_runs:
            print(f"{cell:9s} {ENV_LABEL[cell]:22s}  -- no true arm --")
            continue
        finals = [run["curve_final_mean"] for run in true_runs]
        start = med([run["curve_start"] for run in true_runs])
        true_final = med(finals)
        spread = (max(finals) - min(finals)) / abs(true_final) if true_final else float("nan")
        vanilla_final = med([run["curve_final_mean"] for run in vanilla_runs]) if vanilla_runs else float("nan")
        delivery = (
            f"{med([r['synthetic_queries'] for r in vanilla_runs]):.0f}/"
            f"{vanilla_runs[0]['query_budget']}"
            if vanilla_runs
            else "-"
        )
        print(
            f"{cell:9s} {ENV_LABEL[cell]:22s} {len(true_runs):2d} {true_final:11.1f} "
            f"{min(finals):10.1f} {max(finals):10.1f} {spread:11.2f} {start:10.1f} "
            f"{vanilla_final:10.1f} {delivery:>9s}"
        )
        if true_final - start <= 0:
            broken.add(cell)
            print(
                f"{'':9s} ^^ NO CEILING: the true arm ends at or below its own starting point "
                f"(span {true_final - start:.1f}).\n"
                f"{'':9s}    partiality is undefined here and every prior in this cell is refused,\n"
                f"{'':9s}    not scored -- a negative span would invert the sign of the whole column."
            )
            continue
        anchors[cell] = (start, true_final)
    print(
        "\nspread/med is (max-min)/|median| over the 5 true-arm seeds. A cell whose ceiling\n"
        "is not reproducible cannot support a prior contrast, whatever the priors do.\n"
        "labels is delivered/budget on the vanilla arm; a shortfall means the floor is\n"
        "not the floor it claims to be."
    )
    return anchors, broken


def report_priors(grouped, anchors, cells, broken: set[str] | None = None) -> None:
    broken = broken or set()
    print("\n" + "=" * 108)
    print("PRIOR CANDIDATES -- partiality = (arm - start) / (true - start), target band 0.2 to 0.8")
    print("=" * 108)
    for cell in cells:
        if cell in broken:
            priors = sorted(
                variant for (this_cell, variant) in grouped
                if this_cell == cell and variant not in ("true", "vanilla")
            )
            print(f"\n--- {cell} / {ENV_LABEL[cell]}   REFUSED: no ceiling, partiality undefined")
            print(f"    raw finals only, in the environment's own units:")
            for variant in priors:
                finals = [run["curve_final_mean"] for run in grouped[(cell, variant)]]
                print(f"    {variant:18s} {med(finals):10.1f}   per-seed {[round(v, 1) for v in finals]}")
            continue
        if cell not in anchors:
            continue
        start, true_final = anchors[cell]
        span = true_final - start
        vanilla_runs = grouped.get((cell, "vanilla"), [])
        vanilla_final = med([run["curve_final_mean"] for run in vanilla_runs]) if vanilla_runs else float("nan")
        vanilla_part = (vanilla_final - start) / span if span else float("nan")

        priors = sorted(
            variant for (this_cell, variant) in grouped
            if this_cell == cell and variant not in ("true", "vanilla")
        )
        print(f"\n--- {cell} / {ENV_LABEL[cell]}   true {true_final:.1f}   start {start:.1f}   "
              f"vanilla {vanilla_final:.1f} (partiality {vanilla_part:.2f})")
        print(
            f"    {'prior':18s} {'n':>2s} {'final':>10s} {'seed min':>10s} {'seed max':>10s} "
            f"{'peak':>10s} {'peak @':>9s} {'partiality':>11s} {'PCC':>6s} {'>vanilla':>9s}  verdict"
        )
        for variant in priors:
            runs = grouped[(cell, variant)]
            finals = [run["curve_final_mean"] for run in runs]
            final = med(finals)
            partiality = (final - start) / span if span else float("nan")
            beat = sum(1 for value in finals if value >= vanilla_final)
            in_band = BAND_LOW <= partiality <= BAND_HIGH
            above_floor = final >= vanilla_final
            if partiality > 1.0:
                verdict = "BEATS THE TRUE ARM -- premise failure"
            elif not above_floor:
                verdict = "below vanilla"
            elif in_band:
                verdict = "CANDIDATE"
            elif partiality > BAND_HIGH:
                verdict = "too strong (~true reward)"
            else:
                verdict = "too weak"
            pccs = [r["pcc_partial_true"] for r in runs if np.isfinite(r["pcc_partial_true"])]
            pcc_text = f"{med(pccs):6.2f}" if pccs else "   nan"
            print(
                f"    {variant:18s} {len(runs):2d} {final:10.1f} {min(finals):10.1f} "
                f"{max(finals):10.1f} {med([r['max_reward'] for r in runs]):10.1f} "
                f"{med([r['max_reward_timestep'] for r in runs]):9.0f} {partiality:11.2f} "
                f"{pcc_text} {beat:>4d}/{len(runs):<4d}  {verdict}"
            )


def plot(grouped, cells, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    live = [cell for cell in cells if (cell, "true") in grouped]
    if not live:
        print("nothing to plot")
        return
    ncols = 2 if len(live) > 1 else 1
    nrows = (len(live) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 4.2 * nrows), squeeze=False)

    for index, cell in enumerate(live):
        ax = axes[index // ncols][index % ncols]
        priors = sorted(
            variant for (this_cell, variant) in grouped
            if this_cell == cell and variant not in ("true", "vanilla")
        )
        reds = plt.cm.autumn(np.linspace(0.0, 0.65, max(len(priors), 1)))

        def draw(variant, color, label, **kwargs):
            runs = grouped.get((cell, variant), [])
            if not runs:
                return
            # Seeds can stop on slightly different eval counts; trim to the shortest.
            length = min(len(run["point_means"]) for run in runs)
            x = runs[0]["timesteps"][:length]
            stack = np.vstack([run["point_means"][:length] for run in runs])
            middle = np.median(stack, axis=0)
            ax.plot(x, middle, color=color, label=label, **kwargs)
            if len(runs) > 2:
                ax.fill_between(
                    x,
                    np.percentile(stack, 25, axis=0),
                    np.percentile(stack, 75, axis=0),
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )

        draw("true", "black", "true (ceiling)", linewidth=2.4)
        draw("vanilla", "seagreen", "vanilla RLHF (floor)", linewidth=2.0)
        for prior, color in zip(priors, reds):
            draw(prior, color, prior, linewidth=1.5, linestyle="--")

        ax.set_title(f"{cell} / {ENV_LABEL[cell]}")
        ax.set_xlabel("timesteps")
        ax.set_ylabel("true reward")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False, ncol=2)

    for spare in range(len(live), nrows * ncols):
        axes[spare // ncols][spare % ncols].axis("off")

    fig.suptitle("Selection grid: true (black) vs vanilla RLHF (green) vs hand-written priors (red)", y=1.0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figure: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs", help="directory holding the <prefix>_* log dirs")
    parser.add_argument("--prefix", default="sel",
                        help="log-dir prefix: 'sel' for the original grid, 'sel2' for the "
                             "target_kl rerun. Defaults for --csv/--figure follow it.")
    parser.add_argument("--cell", action="append", choices=CELLS, help="restrict to these cells (repeatable)")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--figure", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    csv_path = args.csv or f"logs/{args.prefix}_summary.csv"
    figure_path = args.figure or f"logs/{args.prefix}_curves.png"

    root = Path(args.root)
    runs = load_runs(root, args.prefix)
    if not runs:
        print(f"no {args.prefix}_* runs with metadata.json under {root.resolve()}")
        return

    cells = tuple(args.cell) if args.cell else CELLS
    runs = [run for run in runs if run["cell"] in cells]
    grouped = by_arm(runs)

    write_csv(runs, Path(csv_path))
    expected = {(cell, variant) for (cell, variant) in grouped}
    short = [f"{cell}/{variant} ({len(grouped[(cell, variant)])}/5)" for cell, variant in sorted(expected)
             if len(grouped[(cell, variant)]) != 5]
    if short:
        print(f"\nINCOMPLETE ARMS (not 5 seeds): {', '.join(short)}")

    anchors, broken = report_environments(grouped, cells)
    report_priors(grouped, anchors, cells, broken)
    if not args.no_plot:
        plot(grouped, cells, Path(figure_path))


if __name__ == "__main__":
    main()
