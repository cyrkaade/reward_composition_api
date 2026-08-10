#!/usr/bin/env python
"""Read the PRE3 stage-1 repair (jobs/run_pre3_fix.sh).

    python jobs/analyze_pre3_fix.py

PRIMARY READOUT is the HOVERING RATE, not a median return: the fraction of seeds
whose FINAL mean eval episode length is at or near LunarLander's 1000-step time
limit. The gate failed by episode-length farming, and that binary carries a
12%-vs-58% effect, so at n=10 it has far more power than a median of a
heavy-tailed return. Returns are reported alongside, never alone.

Every comparison is paired by seed against logs/pre3g_ll_naive_bounded (the
unrepaired gate arm), which is why this job needed no baseline runs of its own.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

HOVER_STEPS = 950          # LunarLander-v3's time limit is 1000
SOLVED = 200.0             # LunarLander's conventional solved threshold

BASELINE = "pre3g_ll_naive_bounded"
LEVERS = ["naive_g99", "naive_l1", "naive_stop", "naive_g99_l1"]


def load_cell(root: Path, cell: str) -> dict[int, dict]:
    """seed -> {final, peak, ep_len_final, ep_len_peak} for one log directory."""
    out = {}
    for meta_path in sorted(root.glob(f"{cell}/*/metadata.json")):
        run = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        curve = run / "eval" / "evaluations.npz"
        if not curve.exists():
            continue
        z = np.load(curve)
        rewards = z["results"].mean(axis=1)
        lengths = z["ep_lengths"].mean(axis=1)
        best = int(np.argmax(rewards))
        out[meta.get("seed")] = {
            "final": float(rewards[-1]), "peak": float(rewards[best]),
            "ep_len_final": float(lengths[-1]), "ep_len_peak": float(lengths[best]),
            "queries": meta.get("synthetic_queries"), "budget": meta.get("query_budget"),
        }
    return out


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test against p=0.5 -- the exact McNemar test on
    discordant pairs. n < 1 returns NaN rather than a fake certainty."""
    if n < 1:
        return float("nan")
    def pmf(i):
        return math.comb(n, i) * 0.5 ** n
    observed = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed + 1e-12))


def mcnemar(pairs: list[tuple[bool, bool]]) -> tuple[int, int, float]:
    """pairs of (arm_hovered, baseline_hovered) -> (fixed, broke, p)."""
    fixed = sum(1 for a, b in pairs if b and not a)     # baseline hovered, arm did not
    broke = sum(1 for a, b in pairs if a and not b)
    return fixed, broke, binom_two_sided(fixed, fixed + broke)


def wilcoxon_p(diffs: list[float]) -> float:
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


def summarize(name: str, runs: dict[int, dict]) -> str:
    if not runs:
        return f"   {name:>22}   (no completed runs)"
    v = list(runs.values())
    hover = sum(1 for r in v if r["ep_len_final"] >= HOVER_STEPS)
    ever = sum(1 for r in v if r["peak"] >= SOLVED)
    fin = sum(1 for r in v if r["final"] >= SOLVED)
    return (f"   {name:>22} n={len(v):<3} hover={hover}/{len(v):<3} "
            f"everSolved={ever}/{len(v):<3} finalSolved={fin}/{len(v):<3} "
            f"final={st.median(r['final'] for r in v):>8.1f} "
            f"peak={st.median(r['peak'] for r in v):>7.1f} "
            f"epLen={st.median(r['ep_len_final'] for r in v):>6.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="logs")
    args = ap.parse_args()
    root = Path(args.root)

    cells = {"BASELINE (gate)": load_cell(root, BASELINE)}
    for v in LEVERS + ["feedback_g99_l1", "feedback_stop", "true_g99", "partial_g99", "naive_p50"]:
        cells[v] = load_cell(root, f"pre3f_ll_{v}")
    gate_feedback = load_cell(root, "pre3g_ll_feedback_bounded")
    gate_true = load_cell(root, "pre3g_ll_true")
    gate_partial = load_cell(root, "pre3g_ll_partial")

    done = sum(len(c) for k, c in cells.items() if k != "BASELINE (gate)")
    print("=" * 104)
    print(f"COVERAGE   {done}/90 new runs complete   (hovering = final mean episode length >= {HOVER_STEPS} of 1000)")
    print("=" * 104)
    starved = [(k, r["queries"], r["budget"]) for k, c in cells.items() for r in c.values()
               if r["budget"] and r["queries"] and r["queries"] < 0.95 * r["budget"]]
    print(f"   query starvation: {'none' if not starved else starved[:5]}")

    print("\n" + "=" * 104)
    print("ALL ARMS")
    print("=" * 104)
    print(summarize("gate true", gate_true))
    print(summarize("gate partial", gate_partial))
    print(summarize("gate feedback", gate_feedback))
    for name, runs in cells.items():
        print(summarize(name, runs))

    print("\n" + "=" * 104)
    print("Q1  WHICH LEVER STOPS THE HOVERING?   (paired by seed vs the unrepaired gate arm)")
    print("=" * 104)
    print("   A lever is a FIX if hovering drops to <=2/10 and >=5/10 seeds finish solved.")
    base = cells["BASELINE (gate)"]
    verdicts = {}
    for lever in LEVERS:
        arm = cells[lever]
        seeds = sorted(set(arm) & set(base))
        if not seeds:
            print(f"   {lever:>14}   (no paired seeds yet)")
            continue
        pairs = [(arm[s]["ep_len_final"] >= HOVER_STEPS, base[s]["ep_len_final"] >= HOVER_STEPS) for s in seeds]
        fixed, broke, p = mcnemar(pairs)
        hover = sum(1 for a, _ in pairs if a)
        fin = sum(1 for s in seeds if arm[s]["final"] >= SOLVED)
        dr = [arm[s]["final"] - base[s]["final"] for s in seeds]
        ok = hover <= 2 and fin >= 5
        verdicts[lever] = ok
        print(f"   {lever:>14} n={len(seeds):<3} hover {hover}/{len(seeds)}  "
              f"(stopped {fixed}, caused {broke}, McNemar p={p:.3f})   "
              f"finalSolved={fin}/{len(seeds)}   dFinal={st.median(dr):+8.1f} (p={wilcoxon_p(dr):.3f})"
              f"   {'<== FIX' if ok else ''}")

    winners = [k for k, ok in verdicts.items() if ok]
    print(f"\n   verdict: {'no lever cleared the bar - fall back to run_pre3_repair.sh rows 21-40 (legacy RM)' if not winners else 'fix(es): ' + ', '.join(winners)}")
    if winners:
        smallest = min(winners, key=lambda k: len(k.split('_')))
        print(f"   prefer the SMALLEST fix that clears the bar: {smallest}")

    print("\n" + "=" * 104)
    print("Q2  M1 AT A WORKING CONFIGURATION   (does the prior still beat vanilla RLHF?)")
    print("=" * 104)
    for naive_cell, fb_cell, label in (("naive_g99_l1", "feedback_g99_l1", "gamma .99 + output L1"),
                                       ("naive_stop", "feedback_stop", "accuracy stop")):
        a, b = cells[naive_cell], cells[fb_cell]
        seeds = sorted(set(a) & set(b))
        if not seeds:
            print(f"   {label:>24}   (incomplete)")
            continue
        d = [a[s]["final"] - b[s]["final"] for s in seeds]
        ha = sum(1 for s in seeds if a[s]["ep_len_final"] >= HOVER_STEPS)
        hb = sum(1 for s in seeds if b[s]["ep_len_final"] >= HOVER_STEPS)
        print(f"   {label:>24} n={len(seeds):<3} naive-minus-feedback median={st.median(d):+8.1f} "
              f"(p={wilcoxon_p(d):.3f})   hovering: naive {ha}/{len(seeds)} vs feedback {hb}/{len(seeds)}")
    print("\n   READ: the mechanism claim is that the prior suppresses episode-length farming,")
    print("   so naive's hovering count must be clearly below feedback's, not just its return higher.")

    print("\n" + "=" * 104)
    print("Q3  DOES THE DISCOUNT MOVE THE REFERENCES?   (true/partial need to match the winner)")
    print("=" * 104)
    print(summarize("gate true   (g.999)", gate_true))
    print(summarize("true_g99", cells["true_g99"]))
    print(summarize("gate partial(g.999)", gate_partial))
    print(summarize("partial_g99", cells["partial_g99"]))
    for label, tc, pc in (("gamma .999", gate_true, gate_partial),
                          ("gamma .99 ", cells["true_g99"], cells["partial_g99"])):
        seeds = sorted(set(tc) & set(pc))
        if not seeds:
            continue
        d = [tc[s]["final"] - pc[s]["final"] for s in seeds]
        print(f"   premise at {label}: true-minus-partial median={st.median(d):+8.1f} "
              f"on {sum(1 for x in d if x > 0)}/{len(d)} seeds (p={wilcoxon_p(d):.3f})")
    print("\n   READ: the env qualifies only if true > partial. Also watch partial's episode")
    print("   length: a pure potential-based partial has NO incentive to terminate, so it")
    print("   lengthens episodes on its own at gamma .999 (525 steps vs true's 202).")

    print("\n" + "=" * 104)
    print("Q4  PARTIAL DESIGN: does a non-potential terminal term stop the farming?")
    print("=" * 104)
    print("   Pre-registered prediction. lunar_lander_approach is pure potential-based shaping")
    print("   (Ng, Harada & Russell 1999), so it cannot change the optimal policy and carries no")
    print("   incentive to terminate. lunarlander_p50 adds a non-potential terminal term. Both")
    print("   arms run the UNREPAIRED gate config, so the partial is the only difference.")
    a, b = cells["naive_p50"], base
    seeds = sorted(set(a) & set(b))
    if seeds:
        pairs = [(a[s]["ep_len_final"] >= HOVER_STEPS, b[s]["ep_len_final"] >= HOVER_STEPS) for s in seeds]
        fixed, broke, p = mcnemar(pairs)
        d = [a[s]["final"] - b[s]["final"] for s in seeds]
        print(f"\n   n={len(seeds)}  hovering {sum(1 for x, _ in pairs if x)}/{len(seeds)} vs baseline "
              f"{sum(1 for _, y in pairs if y)}/{len(seeds)}  (stopped {fixed}, caused {broke}, p={p:.3f})")
        print(f"   final: {st.median(d):+.1f} vs the potential-based partial (p={wilcoxon_p(d):.3f})")
        print("\n   If p50 hovers less: a designer-facing law - a potential-based prior is SAFE")
        print("   (cannot change the optimum) but POWERLESS against the termination channel.")
    else:
        print("   (incomplete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
