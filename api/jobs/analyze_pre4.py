#!/usr/bin/env python
"""Read the PRE4 grid (jobs/run_pre4.sh): six blocks, one verdict each.

    python jobs/analyze_pre4.py              # everything
    python jobs/analyze_pre4.py --only A     # the repair block, as soon as it lands

HOVERING is the primary readout wherever the environment has a time limit it can
farm. The PRE3 gate failed because policies stopped terminating and ran to the
limit, and that binary carries a far larger, lower-variance signal at n=10 than a
median of a heavy-tailed return. Fixed-horizon environments (Pusher 100 steps,
Reacher 50) cannot farm it by construction, so it is suppressed for them and
they act as the negative control for the whole mechanism.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

# Fraction of the env's time limit above which a policy is treated as refusing to
# terminate. None = fixed horizon, so the concept does not apply.
TIME_LIMIT = {"lunarlander": 1000, "walker2d": 1000, "pusher": None, "reacher": None}
HOVER_FRAC = 0.95
SOLVED = {"lunarlander": 200.0}          # envs with a conventional solved threshold
ENV_ORDER = ["lunarlander", "walker2d", "pusher", "reacher"]
GATE_BASELINE = "pre3g_ll_naive_bounded"  # gamma .999, output-L1 0


def env_key(env):
    return (ENV_ORDER.index(env) if env in ENV_ORDER else 9, str(env))


def hovering(env: str, ep_len: float | None) -> bool | None:
    limit = TIME_LIMIT.get(env)
    if limit is None or ep_len is None:
        return None
    return ep_len >= HOVER_FRAC * limit


def load(root: Path, pattern: str) -> list[dict]:
    rows = []
    for meta_path in sorted(root.glob(pattern + "/*/metadata.json")):
        run = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cell = run.parent.name
        variant = cell.split("_", 2)[2] if cell.startswith("pre4_") else cell
        env = meta.get("env_slug") or meta.get("env_id")
        curve = run / "eval" / "evaluations.npz"
        if not curve.exists():
            continue
        z = np.load(curve)
        rewards = z["results"].mean(axis=1)
        lengths = z["ep_lengths"].mean(axis=1)
        best = int(np.argmax(rewards))
        rows.append({
            "variant": variant, "env": env, "seed": meta.get("seed"),
            "budget": meta.get("query_budget"), "queries": meta.get("synthetic_queries"),
            "final": float(rewards[-1]), "peak": float(rewards[best]),
            "ep_len_final": float(lengths[-1]), "ep_len_peak": float(lengths[best]),
            "hover": hovering(env, float(lengths[-1])),
        })
    return rows


def index(rows, variant, env=None, budget=None):
    return {r["seed"]: r for r in rows
            if r["variant"] == variant and (env is None or r["env"] == env)
            and (budget is None or r["budget"] == budget)}


def wilcoxon_p(diffs):
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
    w = sum(r for r, x in zip(ranks, d) if x > 0)
    mu, sd = n * (n + 1) / 4, (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    return math.erfc(abs((w - mu) / sd) / math.sqrt(2)) if sd else float("nan")


def mcnemar(pairs):
    """pairs of (arm_hovered, baseline_hovered) -> (stopped, caused, exact p)."""
    stopped = sum(1 for a, b in pairs if b and not a)
    caused = sum(1 for a, b in pairs if a and not b)
    n = stopped + caused
    if n < 1:
        return stopped, caused, float("nan")
    pmf = lambda i: math.comb(n, i) * 0.5 ** n
    obs = pmf(stopped)
    return stopped, caused, min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


def line(label, runs):
    if not runs:
        return f"   {label:>22}   (none)"
    v = list(runs.values()) if isinstance(runs, dict) else runs
    env = v[0]["env"]
    hov = [r["hover"] for r in v if r["hover"] is not None]
    hov_s = f"hover={sum(hov)}/{len(hov)}" if hov else "hover=  n/a"
    thr = SOLVED.get(env)
    sol = f"solved={sum(1 for r in v if r['final'] >= thr)}/{len(v)}" if thr else "solved=   n/a"
    return (f"   {label:>22} n={len(v):<3} {hov_s:<12} {sol:<12} "
            f"final={st.median(r['final'] for r in v):>9.1f} "
            f"peak={st.median(r['peak'] for r in v):>9.1f} "
            f"epLen={st.median(r['ep_len_final'] for r in v):>6.0f}")


def compare(rows, a, b, env, budget=None, label=None):
    """Paired a-minus-b within one (env, budget)."""
    ia, ib = index(rows, a, env, budget), index(rows, b, env, budget)
    seeds = sorted(set(ia) & set(ib))
    if not seeds:
        return None
    d = [ia[s]["final"] - ib[s]["final"] for s in seeds]
    out = {"n": len(seeds), "median": st.median(d), "p": wilcoxon_p(d),
           "wins": sum(1 for x in d if x > 0), "label": label or f"{a} - {b}"}
    hp = [(ia[s]["hover"], ib[s]["hover"]) for s in seeds if ia[s]["hover"] is not None]
    if hp:
        out["hover_a"] = sum(1 for x, _ in hp if x)
        out["hover_b"] = sum(1 for _, y in hp if y)
        out["stopped"], out["caused"], out["hover_p"] = mcnemar(hp)
    return out


def show(res, indent="   "):
    if res is None:
        print(f"{indent}(incomplete)")
        return
    extra = ""
    if "hover_a" in res:
        extra = (f"   hovering {res['hover_a']}/{res['n']} vs {res['hover_b']}/{res['n']} "
                 f"(stopped {res['stopped']}, caused {res['caused']}, p={res['hover_p']:.3f})")
    print(f"{indent}{res['label']:<34} n={res['n']:<3} median={res['median']:+9.1f} "
          f"wins={res['wins']}/{res['n']:<3} p={res['p']:.3f}{extra}")


# ---------------------------------------------------------------------- blocks
def block_a(rows, gate):
    print("\n" + "=" * 108)
    print("A  CONFIG REPAIR (LunarLander): which lever stops the episode-length farming?")
    print("=" * 108)
    print("   Baseline is the PRE3 gate arm (gamma .999, output-L1 0). A lever is a FIX if")
    print("   hovering drops to <=2/10 and >=5/10 seeds finish solved. Prefer the smallest.")
    if not gate:
        print(f"   WARNING: logs/{GATE_BASELINE} not found. The pass/fail bar below still")
        print("   applies, but the paired McNemar against the unrepaired arm cannot be run.")
    print(line("gate baseline", gate))
    verdicts = {}
    for lever, label in (("naive_g99", "gamma .99"), ("naive_l1", "output-L1 .001"),
                         ("naive_stop", "accuracy stop"), ("naive_std", "gamma .99 + L1")):
        arm = index(rows, lever, "lunarlander", 350)
        if not arm:
            print(f"   {label:>22}   (incomplete)")
            continue
        print(line(label, arm))
        seeds = sorted(set(arm) & set(gate))
        pairs = [(arm[s]["hover"], gate[s]["hover"]) for s in seeds]
        stopped, caused, p = mcnemar(pairs)
        hov = sum(1 for a, _ in pairs if a)
        sol = sum(1 for s in arm if arm[s]["final"] >= 200.0)
        ok = hov <= 2 and sol >= 5
        verdicts[label] = ok
        print(f"   {'':>22}   vs gate: stopped {stopped}, caused {caused}, McNemar p={p:.3f}"
              f"{'    <== FIX' if ok else ''}")
    winners = [k for k, v in verdicts.items() if v]
    print(f"\n   VERDICT: {', '.join(winners) if winners else 'no lever cleared the bar'}")
    if not winners:
        print("   -> the standardized reward model is not viable on a variable-horizon task;")
        print("      fall back to the legacy reward model before reading blocks C-F on LunarLander.")
        print("      Walker/Pusher/Reacher are unaffected: gamma .999 is LunarLander-only.")


def block_b(rows):
    print("\n" + "=" * 108)
    print("B  REFERENCES: does each environment qualify, and where is collapse even possible?")
    print("=" * 108)
    print(f"\n   {'env':>12} {'true':>10} {'partial':>10} {'gap':>10}  {'horizon':>10}  premise")
    for env in sorted({r["env"] for r in rows}, key=env_key):
        t, p = index(rows, "true_std", env), index(rows, "partial_std", env)
        seeds = sorted(set(t) & set(p))
        if not seeds:
            continue
        d = [t[s]["final"] - p[s]["final"] for s in seeds]
        horizon = "variable" if TIME_LIMIT.get(env) else "fixed"
        print(f"   {env:>12} {st.median(t[s]['final'] for s in seeds):>10.1f} "
              f"{st.median(p[s]['final'] for s in seeds):>10.1f} {st.median(d):>+10.1f}  {horizon:>10}  "
              f"{'qualifies' if st.median(d) > 0 else 'PREMISE FAILS'} "
              f"({sum(1 for x in d if x > 0)}/{len(d)} seeds, p={wilcoxon_p(d):.3f})")
    print("\n   collapse table (q350):")
    for env in sorted({r["env"] for r in rows}, key=env_key):
        for v in ("true_std", "partial_std", "feedback_std", "naive_std"):
            runs = index(rows, v, env, None if v in ("true_std", "partial_std") else 350)
            if runs:
                print(line(f"{env}/{v.replace('_std','')}", runs))
    print("\n   READ: the mechanism claim needs the TRUE arm stable while feedback runs away.")
    print("   Fixed-horizon envs cannot farm the time limit, so they are the negative control.")


def block_c(rows):
    print("\n" + "=" * 108)
    print("C  M1 AND QUERY EFFICIENCY: does the prior beat vanilla RLHF, and with how many labels?")
    print("=" * 108)
    for env in sorted({r["env"] for r in rows}, key=env_key):
        budgets = sorted({r["budget"] for r in rows if r["env"] == env and r["variant"] == "naive_std"})
        if not budgets:
            continue
        print(f"\n   {env}:")
        for b in budgets:
            show(compare(rows, "naive_std", "feedback_std", env, b, f"naive - feedback  q={b}"), "      ")
    print("\n   READ: query efficiency = the budget at which naive matches feedback's best budget.")
    print("   On a variable-horizon env the hovering counts should separate the arms too, not")
    print("   just the returns - that is the mechanism, not a correlate.")


def block_d(rows):
    print("\n" + "=" * 108)
    print("D  PRETRAINING OBJECTIVE: MSE pins the output scale, Bradley-Terry calibrates it")
    print("=" * 108)
    print("   MSE regresses onto the partial's raw per-state values, so the model's margin is")
    print("   set by the partial's scale, which BT then reads as a logit. On LunarLander the")
    print("   MSE-pretrained model ranks 66% of held-out pairs correctly yet scores BT loss")
    print("   2.81 against chance 0.69. Verified on one seed: BT_before 2.416 -> 1.021.")
    for env in ("lunarlander", "walker2d"):
        print(f"\n   {env}:")
        for mode in ("feedback", "naive"):
            base = f"{mode}_std"
            for b in (175, 350):
                show(compare(rows, f"bt_{mode}", base, env, b, f"BT  - none   {mode:8} q={b}"), "      ")
            show(compare(rows, f"mse_{mode}", base, env, 175, f"MSE - none   {mode:8} q=175"), "      ")
            show(compare(rows, f"bt_{mode}", f"mse_{mode}", env, 175, f"BT  - MSE    {mode:8} q=175"), "      ")
    print("\n   READ: BT-minus-MSE positive confirms the objective fix. BT-minus-none is the")
    print("   real M3 question, and it should be largest at q175 if the prior is doing work.")


def block_e(rows):
    print("\n" + "=" * 108)
    print("E  ACTIVE LEARNING with the corrected candidate pool (B-Pref's, not the old selector)")
    print("=" * 108)
    print("   The legacy selector scored whole perfect matchings and kept the best; B-Pref")
    print("   draws an independent 10x pool and takes the per-pair top-k. Any earlier AL null")
    print("   was therefore not a test of their method.")
    for env in ("lunarlander", "walker2d"):
        print(f"\n   {env}:")
        for mode in ("feedback", "naive"):
            show(compare(rows, f"al_{mode}", f"{mode}_std", env, 350, f"AL - uniform      {mode:8}"), "      ")
            show(compare(rows, f"al_bt_{mode}", f"bt_{mode}", env, 350, f"AL - uniform (+BT) {mode:8}"), "      ")
    print("\n   READ: PEBBLE reports uncertainty sampling helping only on Quadruped-walk and")
    print("   explicitly not on 'relatively simple environments, like Walker and Cheetah'.")
    print("   A null here replicates that; a win with the prior (+BT rows) would be new.")


def block_f(rows, gate):
    print("\n" + "=" * 108)
    print("F  PARTIAL DESIGN: can a non-potential terminal term stop the farming?")
    print("=" * 108)
    print("   Pre-registered. lunar_lander_approach is pure potential-based shaping (Ng, Harada")
    print("   & Russell 1999): it cannot change the optimal policy, and therefore carries no")
    print("   incentive to terminate. lunarlander_p50 adds a non-potential terminal term. Both")
    print("   arms run the UNREPAIRED gate config, so the partial is the only difference.")
    arm = index(rows, "naive_p50", "lunarlander", 350)
    print(line("gate (approach)", gate))
    print(line("gate (p50)", arm))
    seeds = sorted(set(arm) & set(gate))
    if seeds:
        pairs = [(arm[s]["hover"], gate[s]["hover"]) for s in seeds]
        stopped, caused, p = mcnemar(pairs)
        d = [arm[s]["final"] - gate[s]["final"] for s in seeds]
        print(f"\n   n={len(seeds)}  stopped {stopped}, caused {caused}, McNemar p={p:.3f}   "
              f"final {st.median(d):+.1f} (p={wilcoxon_p(d):.3f})")
        print("\n   If p50 hovers less: a designer-facing law - a potential-based prior is SAFE")
        print("   (provably cannot change the optimum) but POWERLESS against the termination")
        print("   channel, which is exactly the failure mode the learned reward introduces.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="logs")
    ap.add_argument("--only", nargs="*", default=None, help="subset, e.g. A C")
    args = ap.parse_args()
    root = Path(args.root)

    rows = load(root, "pre4_*")
    gate = {r["seed"]: r for r in load(root, GATE_BASELINE)}
    if not rows:
        print(f"no pre4_* runs under {root}/ - has the job finished and been scp'd back?")
        return 1

    print("=" * 108)
    print(f"COVERAGE  {len(rows)}/480 runs complete")
    print("=" * 108)
    by = defaultdict(int)
    for r in rows:
        by[(r["env"], r["variant"], r["budget"])] += 1
    starved = [(k, v) for k, v in ((f"{r['env']}/{r['variant']}/q{r['budget']}", r["queries"])
                                   for r in rows) if v is not None and v < 0.95 * float(k.split("q")[-1])]
    print(f"   cells present: {len(by)}   query starvation: {'none' if not starved else starved[:5]}")

    stages = {"A": lambda: block_a(rows, gate), "B": lambda: block_b(rows), "C": lambda: block_c(rows),
              "D": lambda: block_d(rows), "E": lambda: block_e(rows), "F": lambda: block_f(rows, gate)}
    for name, fn in stages.items():
        if args.only is None or name in args.only:
            fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
