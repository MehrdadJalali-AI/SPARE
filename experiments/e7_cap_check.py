"""E7 (verification): effect of the justification cap on A1 for a linear TBox.

On the UB-Synth RDFS profile every ABox justification is a singleton
(Lemma 3), so with an effectively unbounded cap (m = 10^6) the extracted
lists are complete and the ILP optimises the FULL objective f_E; a proven
optimal status is then a certificate for the full objective.
Writes results/e7_cap_check.csv.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spare import build_instance, ilp, triple_greedy  # noqa: E402
from spare.generators import ub_synth  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from run_all import true_cov  # noqa: E402

T, A = ub_synth(2, 4, 1.0, seed=0, profile="rdfs")
i8 = build_instance(T, A, m=8)
iu = build_instance(T, A, m=10**6)
max_j = max(len(iu.just[t]) for t in iu.targets)
assert max_j < 10**6, "cap must not bind for the full-objective certificate"
n_bind = sum(1 for t in iu.targets if len(iu.just[t]) > 8)
rows = []
for f in [0.01, 0.02, 0.05, 0.1, 0.2]:
    B = int(f * len(A))
    r8 = ilp(i8, B, time_limit=120)
    ru = ilp(iu, B, time_limit=120)
    rows.append(dict(frac=f, budget=B, n_targets=len(iu.targets), targets_capped_at_8=n_bind,
                     max_just_uncapped=max_j,
                     A1_m8_exact=true_cov(i8, triple_greedy(i8, B)),
                     A1_uncapped_exact=true_cov(iu, triple_greedy(iu, B)),
                     ILP_m8_exact=true_cov(i8, r8["S"]), ILP_m8_status=r8["status"],
                     ILP_full_exact=true_cov(iu, ru["S"]), ILP_full_status=ru["status"],
                     ILP_full_solve_s=ru["solve_seconds"]))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "e7_cap_check.csv"),
                          index=False)
