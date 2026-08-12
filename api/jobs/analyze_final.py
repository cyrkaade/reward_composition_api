#!/usr/bin/env python
"""Read the final grid (jobs/run_final.sh). Five stages, one verdict each.

    python jobs/analyze_final.py            # everything present
    python jobs/analyze_final.py --only 0   # the gates, first

Uses scipy's EXACT Wilcoxon and an exact McNemar. jobs/analyze_pre4.py used a
normal approximation without continuity or tie correction and its output was
quoted in notes as "exact"; at n=10 what it prints as 0.005 is exactly 0.002.
The approximation is conservative, so no PRE4 conclusion changes, but do not
mix the two families of numbers in one table.

THE PRIMARY READOUT IS NO LONGER FINAL RETURN. It is `holdout_accuracy`: how
often the reward-model ensemble ranks a pair of fragments the same way the true
reward does, on 100 pairs per round drawn from trajectories no query touched and
that no member trained on. PRE4 had nothing like this - n_val_pairs was 0 in all
480 runs - which is why its budget ladder was flat and uninterpretable.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover - analysis-only dependency
    wilcoxon = None

ENV_ORDER = ["ll", "pusher", "cheetah", "swimmer"]
ENV_NAME = {"ll": "LunarLander-v3", "pusher": "Pusher-v5",
            "cheetah": "HalfCheetah-v5", "swimmer": "Swimmer-v5"}
# Terminating is the FAILURE on every env here except LunarLander, and Pusher /
# HalfCheetah / Swimmer cannot terminate at all, so the hovering readout applies
# to LunarLander alone. Scoring Walker2d with it in PRE4 reported successful
# walkers as failures: spearman(ep_len, return) was +0.913 there against -0.705
# on LunarLander.
TIME_LIMIT = {"ll": 1000}
SOLVED = {"ll": 200.0}


def env_key(env):
    return (ENV_ORDER.index(env) if env in ENV_ORDER else 9, str(env))


def load(root: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(root.glob("final_*/*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cell_dir = meta_path.parent.parent.name          # final_<cell>_<variant>
        _, cell, variant = cell_dir.split("_", 2)
        curve_path = meta_path.parent / "eval" / "evaluations.npz"
        if not curve_path.exists():
            continue
        curve = np.load(curve_path)
        rewards = curve["results"].mean(axis=1)
        lengths = curve["ep_lengths"].mean(axis=1)

        holdout = meta.get("holdout_diagnostics") or []
        after = [entry["after"] for entry in holdout if entry.get("after")]
        before = [entry["before"] for entry in holdout if entry.get("before")]
        accuracies = [entry["accuracy"] for entry in after]
        composed = [entry["accuracy_composed"] for entry in after]

        members = [m for rd in (meta.get("reward_model_training") or []) for m in rd.get("members", [])]
        train_acc = [m["final_train_accuracy"] for m in members if m.get("final_train_accuracy") is not None]
        last_round = (meta.get("reward_model_training") or [{}])[-1].get("members") or []
        member_spread = (
            max(m["final_train_accuracy"] for m in last_round) - min(m["final_train_accuracy"] for m in last_round)
            if last_round else None
        )

        rows.append({
            "cell": cell, "variant": variant, "seed": meta.get("seed"),
            "budget": meta.get("query_budget"), "delivered": meta.get("synthetic_queries"),
            "final": float(rewards[-1]), "peak": float(rewards.max()),
            "start": float(rewards[0]),
            "ep_len_final": float(lengths[-1]),
            "curve": rewards,
            # holdout_accuracy is the LAST round's, i.e. the finished model
            "holdout_accuracy": accuracies[-1] if accuracies else None,
            "holdout_accuracy_first": accuracies[0] if accuracies else None,
            "holdout_composed": composed[-1] if composed else None,
            "holdout_before_first": before[0]["accuracy"] if before else None,
            "holdout_curve": accuracies,
            "n_holdout": holdout[-1]["n_holdout_pairs"] if holdout else 0,
            "train_acc": st.median(train_acc) if train_acc else None,
            "member_spread": member_spread,
            "alpha": meta.get("partial_alpha"),
        })
    return rows


def index(rows, variant, cell, budget=None):
    return {r["seed"]: r for r in rows
            if r["variant"] == variant and r["cell"] == cell
            and (budget is None or r["budget"] == budget)}


def exact_p(diffs) -> float:
    values = [d for d in diffs if d != 0]
    if not values or wilcoxon is None:
        return float("nan")
    return float(wilcoxon(diffs, alternative="two-sided", zero_method="wilcox", method="exact").pvalue)


def mcnemar(pairs):
    stopped = sum(1 for a, b in pairs if b and not a)
    caused = sum(1 for a, b in pairs if a and not b)
    total = stopped + caused
    if total < 1:
        return stopped, caused, float("nan")
    pmf = lambda i: math.comb(total, i) * 0.5 ** total
    observed = pmf(stopped)
    return stopped, caused, min(1.0, sum(pmf(i) for i in range(total + 1) if pmf(i) <= observed + 1e-12))


def compare(rows, a, b, cell, budget=None, b_budget="same", key="final", label=None):
    ia = index(rows, a, cell, budget)
    ib = index(rows, b, cell, budget if b_budget == "same" else b_budget)
    seeds = sorted(set(ia) & set(ib))
    usable = [s for s in seeds if ia[s].get(key) is not None and ib[s].get(key) is not None]
    if not usable:
        return None
    diffs = [ia[s][key] - ib[s][key] for s in usable]
    return {"n": len(usable), "median": st.median(diffs), "p": exact_p(diffs),
            "wins": sum(1 for d in diffs if d > 0), "label": label or f"{a} - {b}",
            "diffs": diffs, "seeds": usable}


def show(res, indent="      "):
    if res is None:
        print(f"{indent}(incomplete)")
        return
    print(f"{indent}{res['label']:<40} n={res['n']:<3} median={res['median']:+9.2f} "
          f"wins={res['wins']}/{res['n']:<3} exact p={res['p']:.4f}")


def describe(label, runs):
    if not runs:
        return f"   {label:>26}   (none)"
    values = list(runs.values()) if isinstance(runs, dict) else runs
    cell = values[0]["cell"]
    limit = TIME_LIMIT.get(cell)
    hover = (f"hover={sum(1 for r in values if r['ep_len_final'] >= 0.95 * limit)}/{len(values)}"
             if limit else "hover=  n/a")
    threshold = SOLVED.get(cell)
    solved = (f"solved={sum(1 for r in values if r['final'] >= threshold)}/{len(values)}"
              if threshold else "solved=   n/a")
    accs = [r["holdout_accuracy"] for r in values if r["holdout_accuracy"] is not None]
    acc = f"holdoutAcc={st.median(accs):.3f}" if accs else "holdoutAcc=  n/a"
    return (f"   {label:>26} n={len(values):<3} {hover:<12} {solved:<12} "
            f"final={st.median(r['final'] for r in values):>9.1f} "
            f"peak={st.median(r['peak'] for r in values):>9.1f} {acc}")


# --------------------------------------------------------------------- stages
def qualify(rows, cell, true_variant, partial_variant, label):
    """The three tests Walker2d failed and was never given before 480 runs."""
    true_runs = index(rows, true_variant, cell)
    if not true_runs:
        print(f"\n   {label}: (no runs yet)")
        return
    finals = sorted(r["final"] for r in true_runs.values())
    spread = finals[-1] - finals[0]
    gains = [float(r["curve"][-1] - r["curve"][-1 - max(len(r["curve"]) // 4, 1)]) for r in true_runs.values()]
    print(f"\n   {label}   n={len(finals)}")
    print(f"      (a) true arm     min={finals[0]:9.1f}  median={st.median(finals):9.1f}  max={finals[-1]:9.1f}"
          f"   spread/median={spread / max(abs(st.median(finals)), 1e-9):5.2f}")
    print(f"      (b) last-quarter gain median={st.median(gains):+9.1f}")
    partial_runs = index(rows, partial_variant, cell)
    if not partial_runs:
        return
    seeds = sorted(set(true_runs) & set(partial_runs))
    if not seeds:
        return
    start = st.median(r["start"] for r in true_runs.values())
    true_med = st.median(true_runs[s]["final"] for s in seeds)
    partial_med = st.median(partial_runs[s]["final"] for s in seeds)
    span = true_med - start
    recovery = 100 * (partial_med - start) / span if abs(span) > 1e-9 else float("nan")
    gap = compare(rows, true_variant, partial_variant, cell, label="true - prior")
    print(f"      (c) untrained={start:9.1f}  true={true_med:9.1f}  prior alone={partial_med:9.1f}"
          f"   prior recovers {recovery:7.1f}% of the range")
    show(gap, "          ")
    if recovery > 70:
        print("          -> PRIOR TOO COMPLETE. Same failure as lunar_lander_approach (89%)")
        print("             and Walker2d (116%): nothing left for the learned reward to add,")
        print("             so naive-minus-prior cannot be significant here.")
    elif gap and gap["median"] > 0:
        print("          -> incomplete prior AND true > prior. This is Pusher's shape (-99%),")
        print("             the only PRE4 environment where the composition beat both parts.")


def stage0(rows):
    print("\n" + "=" * 112)
    print("STAGE 0  THE GATES")
    print("=" * 112)

    print("\n   0a. HYPERPARAMETERS for the two new environments. Everything else in this")
    print("       grid runs STOCK SB3 with no --tuned-hyperparams, which the archive shows")
    print("       is free on LunarLander (281.8 vs 282.7) and inert on Pusher (no zoo block")
    print("       exists). These two have never been measured either way.")
    for cell, arms in (("cheetah", ("true_stock", "true_zoo")),
                       ("swimmer", ("true_stock", "true_g9999"))):
        print(f"\n      {ENV_NAME.get(cell, cell)}")
        for arm in arms:
            runs = index(rows, arm, cell)
            if not runs:
                print(f"         {arm:<12} (none)")
                continue
            finals = sorted(r["final"] for r in runs.values())
            print(f"         {arm:<12} n={len(finals)}  median={st.median(finals):9.1f}  "
                  f"min={finals[0]:9.1f}  max={finals[-1]:9.1f}  "
                  f"peak={st.median(r['peak'] for r in runs.values()):9.1f}")
        show(compare(rows, arms[1], arms[0], cell, label=f"{arms[1]} - {arms[0]}"), "         ")
    print("\n      DECIDE: keep stock unless the alternative clearly wins AND stays unimodal.")
    print("      Then set CHEETAH_ARM / SWIMMER_GAMMA at the top of run_final.sh.")

    print("\n   0b. DO THE NEW ENVIRONMENTS QUALIFY?")
    print("       (a) UNIMODAL ceiling -- a true arm that collapses in a third of seeds cannot")
    print("           anchor 'how close to the ceiling'. Walker2d: 799-6263, 3/10 under 2500.")
    print("       (b) CONVERGED -- last-quarter gain near zero. Walker2d was still +492 at 2M.")
    print("       (c) INCOMPLETE PRIOR -- the prior alone must NOT already solve the task.")
    for cell in ("cheetah", "swimmer"):
        # fall back to the stage-0a stock arm before the stage-1 true arm exists
        true_variant = "true_std" if index(rows, "true_std", cell) else "true_stock"
        qualify(rows, cell, true_variant, "partial_std", ENV_NAME.get(cell, cell))

    print("\n   0c. HOW MUCH ROOM DOES EACH WEAKENED LunarLander PRIOR LEAVE?")
    print("       lunar_lander_approach recovers 89% of the range alone, which is why")
    print("       naive-minus-prior was null on LunarLander at every PRE4 budget. These")
    print("       levels DELETE components rather than rescale, because rescaling every")
    print("       weight by c is exactly --partial-alpha c and is already stage 2:")
    print("         p10  distance only          p25  + speed")
    print("         approach  + orientation and leg contact (the PRE4 prior)")
    print("       Only run a stage-4 naive arm for a level that leaves real headroom.")
    true_runs = index(rows, "true_std", "ll")
    if true_runs:
        start = st.median(r["start"] for r in true_runs.values())
        true_med = st.median(r["final"] for r in true_runs.values())
        span = true_med - start
        print(f"\n      LunarLander untrained={start:.1f}  true={true_med:.1f}  (range {span:.1f})")
        print(f"      {'prior':<26} {'final':>9} {'recovers':>10}   true - prior")
        for variant, name in (("partial_p10", "p10 (distance only)"),
                              ("partial_p25", "p25 (+ speed)"),
                              ("partial_std", "approach (PRE4 prior)")):
            runs = index(rows, variant, "ll")
            if not runs:
                print(f"      {name:<26}    (none)")
                continue
            med = st.median(r["final"] for r in runs.values())
            recovery = 100 * (med - start) / span if abs(span) > 1e-9 else float("nan")
            gap = compare(rows, "true_std", variant, "ll", label="")
            summary = f"{gap['median']:+8.1f}  {gap['wins']}/{gap['n']}  p={gap['p']:.4f}" if gap else ""
            print(f"      {name:<26} {med:>9.1f} {recovery:>9.1f}%   {summary}")
        print("\n      READ: a level recovering well under ~70% has headroom a learned reward")
        print("      can fill. One recovering 89% like the PRE4 prior does not.")


def stage1(rows):
    print("\n" + "=" * 112)
    print("STAGE 1  THE HEADLINE")
    print("=" * 112)
    for cell in sorted({r["cell"] for r in rows}, key=env_key):
        budgets = sorted({r["budget"] for r in rows if r["cell"] == cell and r["variant"] == "naive_std"})
        if not budgets:
            continue
        print(f"\n   {ENV_NAME.get(cell, cell)}")
        for variant, budget in (("true_std", None), ("partial_std", None),
                                ("feedback_std", 350), ("naive_std", 350)):
            print(describe(f"{cell}/{variant.replace('_std', '')}", index(rows, variant, cell, budget)))

        print("\n      A. vs vanilla RLHF -- PRE4 answered this: +202/+214/+182 on LunarLander")
        for budget in budgets:
            show(compare(rows, "naive_std", "feedback_std", cell, budget, label=f"naive - feedback  q={budget}"))
            show(compare(rows, "naive_std", "feedback_std", cell, budget, key="peak",
                         label=f"naive - feedback  q={budget} PEAK"))

        print("\n      B. vs the PRIOR ALONE -- the control PRE4 under-weighted, and the one")
        print("         that decides whether this is a paper about composition or about priors.")
        print("         Watch PEAK as well as FINAL: on PRE4 no env beat the prior at its own")
        print("         peak, so Pusher's win was entirely 'does not throw the peak away'.")
        for budget in budgets:
            show(compare(rows, "naive_std", "partial_std", cell, budget, b_budget=0,
                         label=f"naive - prior      q={budget}"))
            show(compare(rows, "naive_std", "partial_std", cell, budget, b_budget=0, key="peak",
                         label=f"naive - prior      q={budget} PEAK"))

        print("\n      C. reward-model quality on the never-trained holdout (the new measurement)")
        for variant in ("feedback_std", "naive_std"):
            for budget in budgets:
                runs = index(rows, variant, cell, budget)
                accs = [r["holdout_accuracy"] for r in runs.values() if r["holdout_accuracy"] is not None]
                if not accs:
                    continue
                trains = [r["train_acc"] for r in runs.values() if r["train_acc"] is not None]
                spreads = [r["member_spread"] for r in runs.values() if r["member_spread"] is not None]
                comp = [r["holdout_composed"] for r in runs.values() if r["holdout_composed"] is not None]
                print(f"         {variant:<14} q{budget:<5} holdout={st.median(accs):.3f}  "
                      f"train={st.median(trains) if trains else float('nan'):.3f}  "
                      f"gap={st.median(trains) - st.median(accs) if trains else float('nan'):+.3f}  "
                      f"composed={st.median(comp) if comp else float('nan'):.3f}  "
                      f"memberSpread={st.median(spreads) if spreads else float('nan'):.3f}")
        print("         READ: train-minus-holdout is the overfitting gap PRE4 could not see")
        print("         (n_val_pairs was 0). memberSpread near 0 means the ensemble collapsed to")
        print("         one function -- if that happens again, active learning is untestable.")
        if len(budgets) > 1:
            print("\n      D. does the holdout accuracy respond to the budget? (PRE4 was flat)")
            for variant in ("feedback_std", "naive_std"):
                show(compare(rows, variant, variant, cell, budgets[-1], b_budget=budgets[0],
                             key="holdout_accuracy",
                             label=f"{variant} q{budgets[-1]}-q{budgets[0]} holdoutAcc"))


def stage2(rows):
    print("\n" + "=" * 112)
    print("STAGE 2  THE SCALE LAW: does turning alpha down let the composition beat the prior?")
    print("=" * 112)
    print("   The learned reward is tanh-bounded, so its per-step size is ~0.5 everywhere while")
    print("   the prior's is the environment's own reward scale. Measured prior:model per step")
    print("   in PRE4 -- Walker 5.19, LunarLander 1.24, Pusher 0.53, Reacher 0.29 -- orders the")
    print("   four environments EXACTLY by whether the composition beat the prior alone.")
    print("   PRIOR EVIDENCE POINTS THE OTHER WAY. The `ga` family already ran alpha in")
    print("   {0.1, 0.25, 0.5} with the gate off, 15 seeds, and lowering alpha LOWERED final")
    print("   return: LunarLander -79.3 (3/15, p=0.008) at 0.1 and -114.6 (2/15, p=0.008) at")
    print("   0.25; Pusher -3.0 (p=0.015) at 0.1; Reacher -9.0 (p=0.010) at 0.5. Those runs")
    print("   used the LEGACY model (hidden [200], ensemble 1, NO tanh), so they do not test")
    print("   the scale hypothesis - that claim is specifically about a tanh-BOUNDED model,")
    print("   and without tanh the ratio alpha is meant to control does not exist. Still,")
    print("   treat 'naive - prior turns positive as alpha falls' as the surprising outcome.")
    print(f"\n   {'alpha':>8} {'final':>10} {'peak':>10} {'holdoutAcc':>11}   naive - prior (final)")
    for variant, alpha in (("alpha010", 0.10), ("alpha025", 0.25), ("alpha050", 0.50), ("naive_std", 1.00)):
        runs = index(rows, variant, "ll", 350)
        if not runs:
            print(f"   {alpha:>8.2f}   (none)")
            continue
        accs = [r["holdout_accuracy"] for r in runs.values() if r["holdout_accuracy"] is not None]
        gap = compare(rows, variant, "partial_std", "ll", 350, b_budget=0, label="")
        summary = (f"{gap['median']:+8.1f}  {gap['wins']}/{gap['n']}  p={gap['p']:.4f}" if gap else "(no prior arm)")
        print(f"   {alpha:>8.2f} {st.median(r['final'] for r in runs.values()):>10.1f} "
              f"{st.median(r['peak'] for r in runs.values()):>10.1f} "
              f"{st.median(accs) if accs else float('nan'):>11.3f}   {summary}")


def stage3(rows):
    print("\n" + "=" * 112)
    print("STAGE 3  COLD START: pretraining x active learning, scored on the holdout")
    print("=" * 112)
    print("   PRE4 scored both on final return, which is heavy-tailed and needs many seeds.")
    print("   Holdout accuracy is a proportion over 100 pairs, so it is far lower variance and")
    print("   it measures the thing the mechanism is about: does the model rank pairs the way")
    print("   the true reward does. Cells (all LunarLander feedback, q350):")
    print("      A none+uniform = feedback_std   B pretrained+uniform = bt_uniform")
    print("      C none+AL      = al_none        D pretrained+AL      = bt_al")
    for key, name in (("holdout_accuracy", "holdout accuracy"), ("final", "final return")):
        print(f"\n   -- {name} --")
        pretrain = compare(rows, "bt_uniform", "feedback_std", "ll", 350, key=key,
                           label="B - A  pretraining effect")
        active = compare(rows, "al_none", "feedback_std", "ll", 350, key=key,
                         label="C - A  active-learning effect")
        joint = compare(rows, "bt_al", "feedback_std", "ll", 350, key=key,
                        label="D - A  both together")
        for res in (pretrain, active, joint):
            show(res)
        cells = {name: index(rows, variant, "ll", 350)
                 for name, variant in (("A", "feedback_std"), ("B", "bt_uniform"),
                                       ("C", "al_none"), ("D", "bt_al"))}
        seeds = sorted(set.intersection(*(set(c) for c in cells.values()))) if all(cells.values()) else []
        seeds = [s for s in seeds if all(cells[n][s].get(key) is not None for n in cells)]
        if seeds:
            interaction = [cells["D"][s][key] - cells["B"][s][key] - cells["C"][s][key] + cells["A"][s][key]
                           for s in seeds]
            print(f"      {'D - B - C + A  interaction':<40} n={len(seeds):<3} "
                  f"median={st.median(interaction):+9.2f} "
                  f"wins={sum(1 for d in interaction if d > 0)}/{len(seeds):<3} "
                  f"exact p={exact_p(interaction):.4f}")
            print("      An interaction needs ~4x the sample size of a main effect, so a null")
            print("      here at n=10 is uninformative, NOT evidence that they do not interact.")
    naive_al = compare(rows, "al_none_naive", "naive_std", "ll", 350, key="holdout_accuracy",
                       label="AL - uniform, WITH the prior (holdoutAcc)")
    print()
    show(naive_al, "   ")
    print("   PRE4's AL nulls were all measured with a collapsed ensemble (member spread exactly")
    print("   0.000 on two envs), so the question is genuinely reopened, not replicated.")


def stage4(rows):
    print("\n" + "=" * 112)
    print("STAGE 4  PRIOR INFORMATION: does the composition's gain grow as the prior knows less?")
    print("=" * 112)
    print("   This is the dose-response PRE4 lacked. It is WITHIN one environment, so unlike")
    print("   the cross-environment comparison it is not confounded by horizon, reward scale,")
    print("   dynamics or convergence. Each level DELETES components rather than rescaling -")
    print("   rescaling every weight by c is exactly --partial-alpha c and is stage 2.")
    print("   PREDICTION: naive - prior is largest for p10 and shrinks to nothing at the")
    print("   PRE4 prior, which recovers 89% of the task on its own.")
    print(f"\n   {'prior':<26} {'prior alone':>12} {'naive':>10} {'holdoutAcc':>11}   naive - prior")
    for naive_variant, partial_variant, name in (
        ("naive_p10", "partial_p10", "p10 (distance only)"),
        ("naive_p25", "partial_p25", "p25 (+ speed)"),
        ("naive_std", "partial_std", "approach (PRE4 prior)"),
    ):
        naive_runs = index(rows, naive_variant, "ll", 350)
        partial_runs = index(rows, partial_variant, "ll")
        if not naive_runs or not partial_runs:
            print(f"   {name:<26}   (incomplete)")
            continue
        accs = [r["holdout_accuracy"] for r in naive_runs.values() if r["holdout_accuracy"] is not None]
        gap = compare(rows, naive_variant, partial_variant, "ll", 350, b_budget=0, label="")
        summary = f"{gap['median']:+8.1f}  {gap['wins']}/{gap['n']}  p={gap['p']:.4f}" if gap else ""
        print(f"   {name:<26} {st.median(r['final'] for r in partial_runs.values()):>12.1f} "
              f"{st.median(r['final'] for r in naive_runs.values()):>10.1f} "
              f"{st.median(accs) if accs else float('nan'):>11.3f}   {summary}")
    print("\n   Also report naive - feedback at each level: if the prior is weak enough that")
    print("   the composition stops beating vanilla RLHF too, the level is past the useful")
    print("   range and only says 'a bad prior does not help'.")
    for naive_variant, name in (("naive_p10", "p10"), ("naive_p25", "p25"), ("naive_std", "approach")):
        show(compare(rows, naive_variant, "feedback_std", "ll", 350,
                     label=f"naive({name}) - feedback"), "   ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs")
    parser.add_argument("--only", nargs="*", default=None, help="subset, e.g. 0 1")
    args = parser.parse_args()

    rows = load(Path(args.root))
    if not rows:
        print(f"no final_* runs under {args.root}/ - has the job finished and been scp'd back?")
        return 1
    if wilcoxon is None:
        print("WARNING: scipy is not installed; every p-value below will be nan.\n")

    print("=" * 112)
    print(f"COVERAGE  {len(rows)}/360 runs")
    print("=" * 112)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["cell"], row["variant"], row["budget"])].append(row)
    starved = []
    print("   LABEL DELIVERY -- below 95% is not a matched-budget comparison. All 16 Walker2d")
    print("   cells in PRE4 sat at 42-81% because fragment_length 50 met ~18-step episodes,")
    print("   and the deficit tracked how badly the arm was doing, which always favours the")
    print("   winner. Verified before submission: LunarLander 140/140, HalfCheetah and Swimmer")
    print("   70/70 at the highest per-round demand, with the 100-pair holdout also reserved.")
    for key in sorted(grouped, key=lambda k: (env_key(k[0]), k[1], k[2] or 0)):
        delivered = [r["delivered"] for r in grouped[key] if r["delivered"] is not None]
        if not delivered or not key[2]:
            continue
        fraction = st.median(delivered) / float(key[2])
        if fraction < 0.95:
            starved.append(key)
        print(f"      {key[0]:>9} {key[1]:>16} q{key[2]:<5} delivered med={st.median(delivered):>6.0f} "
              f"min={min(delivered):>5} ({100 * fraction:5.1f}%)"
              f"{'   <== STARVED' if fraction < 0.95 else ''}")
    missing = {k: len(v) for k, v in grouped.items() if len(v) < 10 and k[1] not in ("true_std", "partial_std")}
    if missing:
        print(f"\n   cells with fewer than 10 seeds: {missing}")
    holdouts = [r["n_holdout"] for r in rows if r["n_holdout"]]
    print(f"\n   holdout pairs per round: {sorted(set(holdouts)) or 'NONE -- the run predates --holdout-pairs'}")

    stages = {"0": stage0, "1": stage1, "2": stage2, "3": stage3, "4": stage4}
    for name, fn in stages.items():
        if args.only is None or name in args.only:
            fn(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
