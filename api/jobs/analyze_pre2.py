#!/usr/bin/env python
"""One-shot analysis of the run_pre2.sh grid: seven questions, one verdict each.

Usage (from api/):
    python jobs/analyze_pre2.py                    # everything
    python jobs/analyze_pre2.py --root logs        # different log root
    python jobs/analyze_pre2.py --only E1 E3       # a subset

Every policy comparison is PAIRED BY SEED inside an (env, budget) cell and
reported as a median difference with a two-sided Wilcoxon signed-rank p-value.
Means are never reported: the archive shows the distributions are heavy-tailed
and bimodal, so a mean reports which tail a seed landed in.

Metric direction is per-env. LunarLander is higher-is-better (solved >= 200);
Pusher is closer-to-zero-is-better and its rewards are negative, so a "win" on
Pusher is still a larger number, but drawdown ratios are meaningless there and
are suppressed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

CHANCE_BT = math.log(2.0)
POSITIVE_SCALE = {"lunarlander"}  # envs where a drawdown ratio makes sense
EXPECTED_RUNS = 550               # keep in sync with jobs/make_params_pre2.py


# --------------------------------------------------------------------------- io
def load_runs(root: Path) -> list[dict]:
    runs = []
    for meta_path in sorted(root.glob("pre2_*/*/metadata.json")):
        run_dir = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cell = run_dir.parent.name           # pre2_<env>_<variant>
        variant = cell[len("pre2_"):].split("_", 1)[1]
        env = meta.get("env_slug") or meta.get("env_id")

        final, peak = meta.get("selected_policy_true_reward_mean"), meta.get("best_logged_true_reward")
        curve = run_dir / "eval" / "evaluations.npz"
        if curve.exists():
            data = np.load(curve)
            rewards = data["results"].mean(axis=1)
            final, peak = float(rewards[-1]), float(rewards.max())

        before = meta.get("rm_diagnostics_before_training") or {}
        after = meta.get("rm_diagnostics") or {}
        fisher = meta.get("query_fisher") or []
        round0 = next((f for f in fisher if f.get("round") == 0), {})

        runs.append({
            "cell": cell, "variant": variant, "env": env, "seed": meta.get("seed"),
            "budget": meta.get("query_budget"), "queries": meta.get("synthetic_queries"),
            "final": final, "peak": peak,
            "bt_before": before.get("bt_loss"), "acc_before": before.get("accuracy"),
            "bt_after": after.get("bt_loss"), "acc_after": after.get("accuracy"),
            "fisher0": round0.get("fisher_median"), "selector_trained": round0.get("selector_was_trained"),
            "pretrain_loss": meta.get("pretrain_loss"), "holdout": meta.get("pretrain_holdout"),
            "tanh": meta.get("tanh_model_reward"), "ens": meta.get("reward_model_ensemble_size"),
            "al": meta.get("active_learning"), "mode": meta.get("mode"),
        })
    return runs


# ------------------------------------------------------------------- statistics
def wilcoxon_p(diffs: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank, normal approximation with tie-averaged ranks."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 6:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mu, sigma = n * (n + 1) / 4, (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    return math.erfc(abs((w_plus - mu) / sigma) / math.sqrt(2)) if sigma > 0 else float("nan")


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, so a family of cells is not read as
    independent tests. Reported alongside the raw p, never instead of it."""
    usable = {k: v for k, v in pvalues.items() if v == v}
    ordered = sorted(usable.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * p)
        out[key] = min(running, 1.0)
    return out


def paired(runs, a_variant, b_variant, key="final", envs=None, budgets=None):
    """Median a-minus-b per (env, budget), paired by seed."""
    index = defaultdict(dict)
    for r in runs:
        if r[key] is None:
            continue
        index[(r["env"], r["budget"], r["seed"])][r["variant"]] = r[key]
    cells = defaultdict(list)
    for (env, budget, _), by_variant in index.items():
        if envs and env not in envs:
            continue
        if budgets and budget not in budgets:
            continue
        if a_variant in by_variant and b_variant in by_variant:
            cells[(env, budget)].append(by_variant[a_variant] - by_variant[b_variant])
    return {k: v for k, v in sorted(cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))}


def report(title, cells, note="", better="higher"):
    print(f"\n{title}")
    if not cells:
        print("   (no paired runs yet)")
        return
    pvalues = {f"{env}/{budget}": wilcoxon_p(d) for (env, budget), d in cells.items()}
    adjusted = holm(pvalues)
    for (env, budget), diffs in cells.items():
        key = f"{env}/{budget}"
        p, p_adj = pvalues[key], adjusted.get(key, float("nan"))
        median = st.median(diffs)
        wins = sum(1 for x in diffs if x > 0)
        mark = "  <-- significant" if p == p and p_adj == p_adj and p_adj < 0.05 else ""
        print(f"   {env:>12} q={budget:<5} n={len(diffs):<3} median={median:+10.2f} "
              f"wins={wins}/{len(diffs)}  p={p:.3f} holm={p_adj:.3f}{mark}")
    if note:
        print(f"   note: {note}")


def median_of(runs, variant, key, env=None, budget=None):
    vals = [r[key] for r in runs
            if r["variant"] == variant and r[key] is not None
            and (env is None or r["env"] == env) and (budget is None or r["budget"] == budget)]
    return st.median(vals) if vals else float("nan")


def drawdown_table(runs, variants, budget=None):
    print(f"\n   {'variant':>16} {'env':>12} {'n':>4} {'peak':>10} {'final':>10} {'drawdown':>9} {'>20%':>8}")
    for variant in variants:
        for env in sorted({r["env"] for r in runs if r["variant"] == variant}):
            rows = [r for r in runs if r["variant"] == variant and r["env"] == env
                    and r["peak"] is not None and r["final"] is not None
                    and (budget is None or r["budget"] == budget)]
            if not rows:
                continue
            peaks = [r["peak"] for r in rows]
            finals = [r["final"] for r in rows]
            if env in POSITIVE_SCALE:
                drops = [1 - f / p for f, p in zip(finals, peaks) if p > 0]
                dd = f"{100 * st.median(drops):8.1f}%" if drops else "     n/a"
                over = f"{sum(1 for d in drops if d > 0.2)}/{len(drops)}"
            else:
                dd, over = "     n/a", "n/a"
            print(f"   {variant:>16} {env:>12} {len(rows):>4} {st.median(peaks):>10.1f} "
                  f"{st.median(finals):>10.1f} {dd:>9} {over:>8}")


# ---------------------------------------------------------------- the questions
def coverage(runs):
    print("=" * 100)
    print("COVERAGE  (starved queries invalidate a budget comparison, so check this first)")
    print("=" * 100)
    by_cell = defaultdict(list)
    for r in runs:
        by_cell[(r["env"], r["variant"], r["budget"])].append(r)
    starved = []
    for key in sorted(by_cell, key=lambda k: (str(k[0]), k[1], k[2])):
        rows = by_cell[key]
        delivered = [r["queries"] for r in rows if r["queries"] is not None]
        if delivered and st.median(delivered) < key[2] * 0.95:
            starved.append((key, st.median(delivered)))
    print(f"   completed runs: {len(runs)} / {EXPECTED_RUNS}")
    missing = EXPECTED_RUNS - len(runs)
    if missing > 0:
        print(f"   still missing:  {missing}  (resubmit the same sbatch; completed runs are skipped)")
    if starved:
        print("   QUERY STARVATION - these cells did not get the queries they asked for:")
        for (env, variant, budget), got in starved:
            print(f"      {env}/{variant} q={budget} delivered {got:.0f}")
    else:
        print("   query delivery: OK in every cell")


def e1(runs):
    print("\n" + "=" * 100)
    print("E1  PRETRAINING OBJECTIVE: does Bradley-Terry pretraining beat MSE, and does either beat none?")
    print("=" * 100)
    print("\n-- calibration, measured directly on held-out preferences BEFORE any preference training --")
    print("   (chance BT loss = 0.693; the MSE failure is HIGH loss at ABOVE-chance accuracy = confidently wrong)")
    print(f"\n   {'variant':>10} {'env':>12} {'BT before':>11} {'acc before':>11} {'BT after':>10} {'acc after':>10}")
    for variant in ("none_al", "mse_al", "bt_al", "true_al"):
        for env in sorted({r["env"] for r in runs if r["variant"] == variant}):
            print(f"   {variant:>10} {env:>12} {median_of(runs, variant, 'bt_before', env):>11.3f} "
                  f"{median_of(runs, variant, 'acc_before', env):>11.3f} "
                  f"{median_of(runs, variant, 'bt_after', env):>10.3f} "
                  f"{median_of(runs, variant, 'acc_after', env):>10.3f}")
    print("\n   READ: bt_al's BT-before must be far below mse_al's. If it is not, the objective fix")
    print("   did not take and nothing downstream is interpretable. Accuracy separates the two")
    print("   halves of the problem: calibration (BT loss) vs information (accuracy).")

    report("-- policy: BT pretraining minus MSE pretraining (both with AL) --",
           paired(runs, "bt_al", "mse_al"))
    report("-- policy: BT pretraining minus no pretraining (both with AL) --",
           paired(runs, "bt_al", "none_al"),
           note="this is the headline for M3. Positive and significant = pretraining is revived.")
    report("-- policy: MSE pretraining minus no pretraining (the archived, broken version) --",
           paired(runs, "mse_al", "none_al"),
           note="expected negative on LunarLander, reproducing the archive at the new budgets.")


def e2(runs):
    print("\n" + "=" * 100)
    print("E2  ACTIVE LEARNING: does it help, and does it help MORE once the model has a real prior?")
    print("=" * 100)
    print("   In trainer.py the round-0 selector is None unless a model exists, so without")
    print("   pretraining round 0 is always random. Pretraining is what makes round-0 AL possible.")
    report("-- AL on minus AL off, WITHOUT pretraining --", paired(runs, "none_al", "none_noal"))
    report("-- AL on minus AL off, WITH BT pretraining --", paired(runs, "bt_al", "bt_noal"),
           note="bigger here than above = the prior is what makes active learning pay off.")
    report("-- AL on minus AL off, WITH MSE pretraining --", paired(runs, "mse_al", "mse_noal"))

    print("\n-- cold start: median Fisher information of the ROUND-0 queries --")
    print("   (how much the first batch of answers is expected to pin down the reward parameters)")
    print(f"\n   {'variant':>10} {'env':>12} {'fisher round 0':>16} {'selector had a model':>22}")
    for variant in ("none_al", "mse_al", "bt_al"):
        for env in sorted({r["env"] for r in runs if r["variant"] == variant}):
            trained = [r["selector_trained"] for r in runs if r["variant"] == variant and r["env"] == env]
            print(f"   {variant:>10} {env:>12} {median_of(runs, variant, 'fisher0', env):>16.4g} "
                  f"{str(any(trained)):>22}")


def e3(runs):
    print("\n" + "=" * 100)
    print("E3  BUDGET: does the pretraining effect decay as labels get cheaper?")
    print("=" * 100)
    print("   The archive says it roughly HALVES from 350 to 700 in 8 of 10 cells, and 350 was")
    print("   the lowest budget ever run. If the effect is real it should be LARGEST at 175.")
    for label, (a, b) in {"BT minus none": ("bt_al", "none_al"), "MSE minus none": ("mse_al", "none_al")}.items():
        cells = paired(runs, a, b)
        print(f"\n   {label}: median difference by budget")
        by_env = defaultdict(dict)
        for (env, budget), diffs in cells.items():
            by_env[env][budget] = st.median(diffs)
        for env, ladder in sorted(by_env.items()):
            trend = "  ".join(f"q={b}: {v:+9.2f}" for b, v in sorted(ladder.items()))
            monotone = len(ladder) == 3 and abs(ladder[175]) > abs(ladder[350]) > abs(ladder[700])
            print(f"      {env:>12}  {trend}   {'DECAYS with budget' if monotone else ''}")

    print("\n-- absolute level per arm, to check nothing is saturated at the top --")
    print(f"\n   {'variant':>10} {'env':>12} " + " ".join(f"{'q=' + str(b):>10}" for b in (175, 350, 700)))
    for variant in ("fb_al", "none_al", "bt_al", "mse_al"):
        for env in sorted({r["env"] for r in runs if r["variant"] == variant}):
            cells = " ".join(f"{median_of(runs, variant, 'final', env, b):>10.1f}" for b in (175, 350, 700))
            print(f"   {variant:>10} {env:>12} {cells}")


def e4(runs):
    print("\n" + "=" * 100)
    print("E4  ENSEMBLE: PEBBLE's real active learning (3 models + disagreement) vs our size-1 MC-dropout")
    print("=" * 100)
    print("   At ensemble size 1 the 'auto' strategy silently becomes MC-dropout, which is NOT")
    print("   what PEBBLE does, so the archived active-learning null never tested their method.")
    report("-- ensemble 3 + disagreement minus size 1, no pretraining --", paired(runs, "none_ens3", "none_al"))
    report("-- ensemble 3 + disagreement minus size 1, BT pretraining --", paired(runs, "bt_ens3", "bt_al"))
    print("\n   CAVEAT: train_preference_reward_ensemble splits the pairs into DISJOINT folds, so")
    print("   each of the 3 members sees only 2/3 of the data (PEBBLE trains all members on all")
    print("   of it). A null here is partly a statement about that split, not only about")
    print("   disagreement sampling.")


def e5(runs):
    print("\n" + "=" * 100)
    print("E5  DATA LEAK: pretraining and round-0 queries on the same rollout vs disjoint halves")
    print("=" * 100)
    report("-- leak (shared rollout) minus holdout (disjoint) --", paired(runs, "bt_leak", "bt_al"),
           note="positive = the old shared-rollout number was inflated by memorisation.")
    print(f"\n   {'variant':>10} {'env':>12} {'BT before':>11} {'acc before':>11} {'fisher round 0':>16}")
    for variant in ("bt_leak", "bt_al"):
        for env in sorted({r["env"] for r in runs if r["variant"] == variant}):
            print(f"   {variant:>10} {env:>12} {median_of(runs, variant, 'bt_before', env):>11.3f} "
                  f"{median_of(runs, variant, 'acc_before', env):>11.3f} "
                  f"{median_of(runs, variant, 'fisher0', env):>16.4g}")
    print("\n   READ: if the leak arm looks better ONLY on the diagnostics and not on the policy,")
    print("   the leak was inflating the measurement rather than the method.")


def e6(runs):
    print("\n" + "=" * 100)
    print("E6  CEILING: pretraining on TRUE-reward labels (cheating) - can ANY initialisation win?")
    print("=" * 100)
    report("-- true-label pretraining minus no pretraining --", paired(runs, "true_al", "none_al"),
           note="if even this does not beat naive, an init cannot substitute for a persistent "
                "prior, and M3 is dead for a stronger reason than 'our partial was bad'.")
    report("-- true-label pretraining minus partial-label BT pretraining --", paired(runs, "true_al", "bt_al"),
           note="the gap here is how much of M3's shortfall is the partial's own ranking quality.")


def e7(runs):
    print("\n" + "=" * 100)
    print("E7  TANH BOUNDING: does the collapse survive PEBBLE's bounded reward model?")
    print("=" * 100)
    print("   An unbounded learned reward is what lets a policy run away, and the headline is")
    print("   that policies run away. model_reward_min/max only clip in the wrapper AFTER")
    print("   training; tanh bounds inside the model, where it shapes the BT loss.")
    report("-- naive: tanh minus unbounded --", paired(runs, "tanh_naive", "none_al", budgets=[350]))
    report("-- feedback: tanh minus unbounded --", paired(runs, "tanh_feedback", "fb_al", budgets=[350]))
    print("\n-- collapse, peak vs final (LunarLander only; drawdown is meaningless on Pusher) --")
    drawdown_table(runs, ("fb_al", "tanh_feedback", "none_al", "tanh_naive", "bt_al"), budget=350)
    print("\n   READ: if feedback still collapses with tanh and naive still does not, the")
    print("   mechanism claim survives the standard bounded model and gets much stronger.")


def e8(runs):
    print("\n" + "=" * 100)
    print("E8  HYPERPARAMETER CONTROL: is the collapse an artifact of the untuned PPO config?")
    print("=" * 100)
    print("   The whole grid runs stock PPO, matching the ~1,500 archived LunarLander runs.")
    print("   These three arms re-run the collapse comparison under rl-zoo's tuned LunarLander")
    print("   block (gamma 0.99->0.999, n_steps 2048->1024, n_epochs 10->4, gae_lambda")
    print("   0.95->0.98, ent_coef 0->0.01). Pusher is absent because the flag is a verified")
    print("   no-op there: its preset is tuned=False, 0 differing keys.")
    print("\n-- collapse under each config (LunarLander, q=350) --")
    drawdown_table(runs, ("fb_al", "tuned_feedback", "none_al", "tuned_naive", "tuned_true"), budget=350)
    print("\n   READ: the claim needs the TRUE arm stable and FEEDBACK collapsing under BOTH")
    print("   configs. If tuned_true is stable and tuned_feedback still collapses while")
    print("   tuned_naive does not, the mechanism is not a hyperparameter artifact and the")
    print("   'why didn't you tune PPO' review comment is answered with data.")
    print("   If tuned_true itself collapses, LunarLander joins Hopper as contaminated under")
    print("   that config - report the stock numbers and say why.")
    report("-- policy: tuned naive minus tuned feedback (the M1 gap under the tuned config) --",
           paired(runs, "tuned_naive", "tuned_feedback", budgets=[350]),
           note="compare with fb_al vs none_al above; the gap should survive, not necessarily match.")


def verdict(runs):
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    checks = []

    def med_all(cells):
        pooled = [d for diffs in cells.values() for d in diffs]
        return (st.median(pooled), wilcoxon_p(pooled), len(pooled)) if pooled else (float("nan"),) * 2 + (0,)

    for label, cells in (
        ("BT pretraining beats MSE", paired(runs, "bt_al", "mse_al")),
        ("BT pretraining beats no pretraining", paired(runs, "bt_al", "none_al")),
        ("active learning helps (with prior)", paired(runs, "bt_al", "bt_noal")),
        ("active learning helps (no prior)", paired(runs, "none_al", "none_noal")),
        ("ensemble-3 disagreement helps", paired(runs, "bt_ens3", "bt_al")),
        ("the data leak was inflating results", paired(runs, "bt_leak", "bt_al")),
        ("even true-label pretraining helps", paired(runs, "true_al", "none_al")),
    ):
        median, p, n = med_all(cells)
        if n == 0:
            checks.append((label, "no data", float("nan"), 0))
            continue
        call = "YES" if (p == p and p < 0.05 and median > 0) else ("NO (worse)" if median < 0 and p == p and p < 0.05 else "no effect")
        checks.append((label, call, median, n))

    for label, call, median, n in checks:
        print(f"   {label:<40} {call:<12} pooled median={median:+9.2f}  n={n}")
    print("\n   Pooled across envs and budgets - directional only. Trust the per-cell tables above.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs", help="log root containing pre2_* cells")
    parser.add_argument("--only", nargs="*", default=None, help="subset, e.g. E1 E3")
    args = parser.parse_args()

    runs = load_runs(Path(args.root))
    if not runs:
        print(f"no pre2_* runs found under {args.root}/ - has the job finished and been scp'd back?")
        return 1

    coverage(runs)
    stages = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E7": e7, "E8": e8}
    for name, fn in stages.items():
        if args.only is None or name in args.only:
            fn(runs)
    verdict(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
