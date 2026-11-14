"""E2b: for SynthJ instances where the ILP does not finish, report greedy objective
relative to the HiGHS dual (upper) bound on the capped objective."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spare import build_instance, coverage, ilp, justification_greedy, triple_greedy, total_weight
from spare.generators import synth_j
rows = []
for p in [0.5, 0.75]:
    for seed in range(5):
        T, A = synth_j(p_multi=p, seed=seed)
        inst = build_instance(T, A)
        B = int(0.1 * len(A))
        r = ilp(inst, B, time_limit=60)
        W = total_weight(inst)
        rows.append(dict(p_multi=p, seed=seed, optimal=(r["status"] == 0), bound=r["bound"] / W,
                         ilp=(r["objective"] or 0) / W, A1=coverage(inst, triple_greedy(inst, B)) / W,
                         A2=coverage(inst, justification_greedy(inst, B)) / W))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "e2b_bounds.csv"), index=False)
