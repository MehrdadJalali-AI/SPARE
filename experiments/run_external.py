"""E8: SPARE on external benchmarks and a real-world KG.

Datasets (prepared by experiments/prepare_real_data.py, files in data/external):
  LUBM-1         original UBA 1.7 generator, univ-bench.owl
  OWL2Bench-QL   official generator, 1 university, seed 1 (linear after loading)
  OWL2Bench-RL-  same, RL profile, without the transitivity of hasSameHomeTownWith
  OWL2Bench-EL-  same, EL profile, without the transitivity of hasSameHomeTownWith
  DBpedia        real DBpedia instance triples (JOIE DB111K, subject-sampled to
                 30k triples) + official DBpedia ontology

Usage: python experiments/run_external.py --datasets LUBM-1 DBpedia ...
Writes results/e8_<dataset>.csv and results/e8_datasets_<dataset>.json.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from spare import build_instance, closure_of, ilp, justification_greedy, total_weight, triple_greedy  # noqa
from spare import baselines as bl  # noqa
from spare.io import load_kb  # noqa

EXT = os.path.join(ROOT, "data", "external")
RES = os.path.join(ROOT, "results")
O2B_NS = {"https://kracr.iiitd.edu.in/OWL2Bench#": "http://benchmark/OWL2Bench#"}
DROP_TRANS = "http://benchmark/OWL2Bench#hasSameHomeTownWith"


def load(name):
    if name == "LUBM-1":
        t, a, r = load_kb(os.path.join(EXT, "univ-bench.owl"), sorted(glob.glob(os.path.join(EXT, "lubm1", "*.owl"))),
                          abox_format="xml")
    elif name.startswith("OWL2Bench"):
        prof = name.split("-")[1][:2]
        t, a, r = load_kb(os.path.join(EXT, "owl2bench", f"UNIV-BENCH-OWL2{prof}.owl"),
                          os.path.join(EXT, "owl2bench", f"OWL2{prof}-1.owl"), abox_format="turtle",
                          namespace_map=O2B_NS)
        if name.endswith("-"):
            before = len(t.transitive)
            t.transitive = [x for x in t.transitive if x != DROP_TRANS]
            r["dropped_transitive_axioms"] = before - len(t.transitive)
    elif name == "DBpedia":
        t, a, r = load_kb(os.path.join(EXT, "dbpedia_ontology.owl"), os.path.join(EXT, "dbpedia_joie_sample.nt"),
                          abox_format="nt")
    else:
        raise ValueError(name)
    return t, a, r


def exact_cov(inst, S, W):
    cl = closure_of(inst.tbox, S)
    return sum(inst.weight[x] for x in inst.targets if x in cl) / W


def run(name, fracs, time_limit):
    t0 = time.time()
    T, A, report = load(name)
    tl = time.time() - t0
    t0 = time.time()
    inst = build_instance(T, A, m=8)
    tb = time.time() - t0
    W = total_weight(inst)
    ms = [min(len(j) for j in inst.just[x]) for x in inst.targets]
    stats = dict(dataset=name, n_abox=len(A), n_targets=len(inst.targets),
                 n_just=int(sum(len(v) for v in inst.just.values())),
                 multi_share=float(np.mean([m > 1 for m in ms])), max_just=inst.max_just_size,
                 linear=T.is_linear(), tbox=T.size(), load_s=tl, build_s=tb,
                 loader_report={k: int(v) for k, v in report.items()})
    with open(os.path.join(RES, f"e8_datasets_{name}.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(stats, flush=True)
    rows = []
    for f in fracs:
        B = max(1, int(f * len(A)))
        sols, secs = {}, {}
        for mname, fn in [("SPARE-T (A1)", triple_greedy), ("SPARE-J (A2)", justification_greedy),
                          ("JustFreq", bl.justfreq_baseline), ("Degree", bl.degree_baseline),
                          ("PageRank", bl.pagerank_baseline), ("Entity ablation", bl.entity_ablation_baseline)]:
            t1 = time.time()
            sols[mname] = fn(inst, B)
            secs[mname] = time.time() - t1
        r = ilp(inst, B, time_limit=time_limit)
        covs = {m: exact_cov(inst, S, W) for m, S in sols.items()}
        ilp_cov = exact_cov(inst, r["S"], W) if r["S"] else float("nan")
        cands = [covs["SPARE-T (A1)"], covs["SPARE-J (A2)"]] + ([ilp_cov] if r["S"] else [])
        covs["SPARE (ILP+greedy)"] = max(cands)
        secs["SPARE (ILP+greedy)"] = secs["SPARE-T (A1)"] + secs["SPARE-J (A2)"] + r["seconds"]
        rnd = [exact_cov(inst, bl.random_baseline(inst, B, seed=s), W) for s in range(3)]
        covs["Random"] = float(np.mean(rnd))
        secs["Random"] = float("nan")
        for m, v in covs.items():
            rows.append(dict(dataset=name, frac=f, budget=B, method=m, coverage=v, seconds=secs[m],
                             ilp_status=r["status"], ilp_capped_optimal=(r["status"] == 0),
                             ilp_exact_cov=ilp_cov, ilp_build_s=r["build_seconds"],
                             ilp_solve_s=r["solve_seconds"]))
        print(name, f, {m: round(v, 4) for m, v in covs.items()}, "ILP status", r["status"],
              round(r["solve_seconds"], 1), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"e8_{name}.csv"), index=False)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["LUBM-1", "OWL2Bench-QL", "OWL2Bench-RL-", "OWL2Bench-EL-", "DBpedia"])
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--time-limit", type=float, default=60)
    a = ap.parse_args()
    for d in a.datasets:
        run(d, a.fracs, a.time_limit)
