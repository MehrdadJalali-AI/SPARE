"""Run all SPARE experiments (E1-E6), write CSVs to results/ and figures to
manuscript/figures/. Usage:  python experiments/run_all.py [--quick]

All reported coverage values are computed with the reasoner (closure of T ∪ S),
not with the capped justification lists, so they are exact.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spare import (TYPE, build_instance, closure_of, ilp, justification_greedy,  # noqa
                   robust_greedy, total_weight, triple_greedy)
from spare import baselines as bl  # noqa
from spare.generators import dks_instance, synth_j, ub_synth  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def true_cov(inst, S):
    cl = closure_of(inst.tbox, S)
    return sum(inst.weight[t] for t in inst.targets if t in cl) / total_weight(inst)


def spare_hybrid(inst, B, time_limit=60):
    """SPARE default for f_E: A1, A2, time-limited A3; keep the best by exact coverage."""
    from spare import spare_hybrid as _h
    return _h(inst, B, r=1, time_limit=time_limit, evaluate=lambda S: true_cov(inst, S))


METHODS = {
    "SPARE (ILP+greedy)": lambda inst, B: spare_hybrid(inst, B),
    "SPARE-J (A2)": justification_greedy,
    "SPARE-T (A1)": triple_greedy,
    "JustFreq": bl.justfreq_baseline,
    "Degree": bl.degree_baseline,
    "PageRank": bl.pagerank_baseline,
    "Entity ablation": bl.entity_ablation_baseline,
    "Random": lambda inst, B: bl.random_baseline(inst, B, seed=0),
}


# --------------------------------------------------------------------- E1
def e1(quick):
    rows = []
    seeds = [0] if quick else [0, 1, 2]
    fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    for prof in ["rdfs", "rl"]:
        for seed in seeds:
            T, A = ub_synth(2, 4, 1.0, seed=seed, profile=prof)
            inst = build_instance(T, A)
            for f in fracs:
                B = max(1, int(f * len(A)))
                for name, fn in METHODS.items():
                    t0 = time.time()
                    if name == "Random":
                        vals = [true_cov(inst, bl.random_baseline(inst, B, seed=s)) for s in range(5)]
                        v = float(np.mean(vals))
                    else:
                        v = true_cov(inst, fn(inst, B))
                    rows.append(dict(profile=prof, seed=seed, frac=f, budget=B, method=name,
                                     coverage=v, seconds=time.time() - t0,
                                     n_abox=len(A), n_targets=len(inst.targets)))
                print(f"E1 {prof} seed{seed} {f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e1_coverage_vs_budget.csv"), index=False)
    return df


# --------------------------------------------------------------------- E2
def e2(quick):
    rows = []
    seeds = range(2 if quick else 5)
    for p in [0.0, 0.25, 0.5, 0.75, 0.9]:
        for seed in seeds:
            T, A = synth_j(p_multi=p, seed=seed)
            inst = build_instance(T, A)
            B = int(0.1 * len(A))
            r = ilp(inst, B, time_limit=30)
            opt = true_cov(inst, r["S"])
            a1 = true_cov(inst, triple_greedy(inst, B))
            a2 = true_cov(inst, justification_greedy(inst, B))
            rows.append(dict(kind="synth", p_multi=p, seed=seed, max_just=inst.max_just_size,
                             ilp=opt, ilp_optimal=(r["status"] == 0), A1=a1, A2=a2,
                             jf=true_cov(inst, bl.justfreq_baseline(inst, B))))
        print("E2 p", p, flush=True)
    for k in [5, 10, 15]:
        for seed in seeds:
            T, A, edges = dks_instance(n=40, p=0.15, seed=seed)
            inst = build_instance(T, A)
            r = ilp(inst, k, time_limit=30)
            rows.append(dict(kind="dks", p_multi=1.0, k=k, seed=seed, max_just=2,
                             ilp=true_cov(inst, r["S"]), ilp_optimal=(r["status"] == 0),
                             A1=true_cov(inst, triple_greedy(inst, k)),
                             A2=true_cov(inst, justification_greedy(inst, k)),
                             jf=true_cov(inst, bl.justfreq_baseline(inst, k))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e2_optimality_gap.csv"), index=False)
    return df


# --------------------------------------------------------------------- E3
def e3(quick):
    T, A = ub_synth(2, 4, 1.0, seed=0, profile="rl")
    inst = build_instance(T, A)
    rows = []
    for f in [0.2, 0.3]:
        B = int(f * len(A))
        sols = {"r=1 (ILP, f_E)": ilp(inst, B, time_limit=60)["S"],
                "r=2 (ILP, f_R)": ilp(inst, B, time_limit=60, r=2)["S"],
                "r=1 (greedy)": justification_greedy(inst, B),
                "r=2 (greedy)": robust_greedy(inst, B, r=2),
                "Degree": bl.degree_baseline(inst, B)}
        for name, S in sols.items():
            base = true_cov(inst, S)
            for q in [0.0, 0.1, 0.2, 0.3, 0.4]:
                vals = []
                for s in range(5 if quick else 20):
                    rng = random.Random(s)
                    Sl = sorted(S)
                    keep = set(rng.sample(Sl, k=int(round((1 - q) * len(Sl)))))
                    vals.append(true_cov(inst, keep))
                rows.append(dict(frac=f, method=name, q=q, coverage=float(np.mean(vals)),
                                 std=float(np.std(vals)), base=base))
        print("E3", f, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e3_robustness.csv"), index=False)
    return df


# --------------------------------------------------------------------- E4
def query_recall(inst, full_cl, S):
    cl = closure_of(inst.tbox, S)
    ans_full, ans_s = defaultdict(set), defaultdict(set)
    for (s, p, o) in full_cl:
        key = ("C", o) if p == TYPE else ("P", p)
        ans_full[key].add((s, o))
    for (s, p, o) in cl:
        key = ("C", o) if p == TYPE else ("P", p)
        ans_s[key].add((s, o))
    rec = [len(ans_s[k] & v) / len(v) for k, v in ans_full.items() if v]
    return float(np.mean(rec)), len(rec)


def e4(quick):
    T, A = ub_synth(2, 4, 1.0, seed=0, profile="rl")
    inst = build_instance(T, A)
    instw = build_instance(T, A, weight="macro-query", include_asserted=True)
    full = inst.closure
    rows = []
    for f in [0.05, 0.1, 0.2, 0.3]:
        B = int(f * len(A))
        sols = {name: fn(inst, B) for name, fn in METHODS.items()}
        r = ilp(instw, B, time_limit=60)
        sols["SPARE-W (ILP, macro-query)"] = max(
            [justification_greedy(instw, B), triple_greedy(instw, B)] + ([r["S"]] if r["S"] else []),
            key=lambda S: query_recall(inst, full, S)[0])
        for name, S in sols.items():
            rec, nq = query_recall(inst, full, S)
            rows.append(dict(frac=f, method=name, macro_recall=rec, n_queries=nq,
                             entail_cov=true_cov(inst, S)))
        print("E4", f, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e4_query_recall.csv"), index=False)
    return df


# --------------------------------------------------------------------- E5
def e5(quick):
    rows = []
    for nu in ([1, 2] if quick else [1, 2, 4, 8]):
        T, A = ub_synth(nu, 4, 1.0, seed=0, profile="rl")
        t0 = time.time()
        inst = build_instance(T, A)
        tb = time.time() - t0
        B = int(0.1 * len(A))
        t0 = time.time(); S1 = triple_greedy(inst, B); t1 = time.time() - t0
        t0 = time.time(); S2 = justification_greedy(inst, B); t2 = time.time() - t0
        r = ilp(inst, B, time_limit=120)
        n_j = sum(len(v) for v in inst.just.values())
        rows.append(dict(n_univ=nu, n_abox=len(A), n_targets=len(inst.targets), n_just=n_j,
                         build_s=tb, A1_s=t1, A2_s=t2, ilp_s=r["seconds"],
                         ilp_build_s=r["build_seconds"], ilp_solve_s=r["solve_seconds"],
                         ilp_time_limit=120, ilp_status=r["status"],
                         ilp_optimal=(r["status"] == 0), A1=true_cov(inst, S1),
                         A2=true_cov(inst, S2), ilp=true_cov(inst, r["S"])))
        print("E5", nu, rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e5_scalability.csv"), index=False)
    return df


# --------------------------------------------------------------------- E6
def e6(quick):
    T, A = synth_j(p_multi=0.75, seed=1)
    rows = []
    B = int(0.1 * len(A))
    ref = build_instance(T, A, m=64)
    for m in [1, 2, 4, 8, 16, 32]:
        inst = build_instance(T, A, m=m)
        n_j = sum(len(v) for v in inst.just.values())
        S2 = justification_greedy(inst, B)
        r = ilp(inst, B, time_limit=30)
        rows.append(dict(m=m, n_just=n_j, A2=true_cov(ref, S2), ilp=true_cov(ref, r["S"]),
                         ilp_optimal=(r["status"] == 0)))
        print("E6", rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e6_cap_ablation.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    for name, fn in [("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6)]:
        if a.only is None or name in a.only:
            t0 = time.time()
            fn(a.quick)
            print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)
