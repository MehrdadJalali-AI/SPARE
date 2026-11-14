import itertools
import random

import networkx as nx
import pytest

from spare import (TBox, TYPE, build_instance, closure_of, coverage, ilp,
                   justification_greedy, triple_greedy)
from spare.generators import dks_instance, synth_j, ub_synth


def test_subclass_chain_singleton_justification():
    t = TBox(subclass=[("A", "B"), ("B", "C")])
    inst = build_instance(t, [("x", TYPE, "A")])
    assert ("x", TYPE, "C") in inst.targets
    assert inst.just[("x", TYPE, "C")] == [frozenset([("x", TYPE, "A")])]


def test_conjunction_needs_both():
    t = TBox(conj=[("A", "B", "D")])
    inst = build_instance(t, [("x", TYPE, "A"), ("x", TYPE, "B")])
    tgt = ("x", TYPE, "D")
    assert inst.just[tgt] == [frozenset([("x", TYPE, "A"), ("x", TYPE, "B")])]
    assert coverage(inst, {("x", TYPE, "A")}) == 0
    assert coverage(inst, set(inst.abox)) == 1


def test_transitive_multiple_justifications():
    t = TBox(transitive=["p"])
    A = [("a", "p", "b"), ("b", "p", "c"), ("a", "p", "c2"), ("c2", "p", "c")]
    inst = build_instance(t, A)
    js = inst.just[("a", "p", "c")]
    assert len(js) == 2 and all(len(j) == 2 for j in js)


def test_lemma1_coverage_matches_reasoner():
    """Lemma 1: T ∪ S ⊨ α  iff  some J ⊆ S (checked by re-running the reasoner)."""
    t, A = synth_j(p_multi=0.5, n_ind=25, n_edges=40, seed=3)
    inst = build_instance(t, A, m=64)
    rng = random.Random(0)
    for _ in range(20):
        S = set(rng.sample(inst.abox, k=len(inst.abox) // 2))
        cl = closure_of(t, S)
        for tgt in inst.targets:
            assert (tgt in cl) == any(j <= S for j in inst.just[tgt]) or len(inst.just[tgt]) == 64


def test_lemma2_soundness():
    t, A = ub_synth(scale=0.2, seed=1)
    inst = build_instance(t, A)
    S = justification_greedy(inst, budget=len(A) // 5)
    assert closure_of(t, S) <= inst.closure


def test_dks_reduction_equivalence():
    """Theorem 1: BJPS optimum on the reduction = densest-k-subgraph edge count."""
    t, A, edges = dks_instance(n=9, p=0.4, seed=2)
    inst = build_instance(t, A)
    assert inst.max_just_size == 2
    G = nx.Graph(edges)
    G.add_nodes_from(range(9))
    k = 4
    best = max(G.subgraph(c).number_of_edges() for c in itertools.combinations(range(9), k))
    res = ilp(inst, budget=k)
    assert round(res["objective"]) == best


def test_greedy_optimal_ratio_linear():
    """Theorem 2 sanity check: on linear TBoxes greedy >= (1-1/e) OPT."""
    t, A = ub_synth(scale=0.15, seed=4, profile="rdfs")
    inst = build_instance(t, A)
    assert inst.max_just_size == 1
    for B in [5, 20, 60]:
        g = coverage(inst, triple_greedy(inst, B))
        opt = ilp(inst, B)["objective"]
        assert g >= (1 - 1 / 2.718281828) * opt - 1e-6


def test_running_example_figure_numbers():
    """Numbers shown in Figure 2 (running example): Degree 4/12, SPARE 7/12 = optimum at B=3."""
    from spare import baselines as bl
    t = TBox(subclass=[("GradStudent", "Student"), ("Student", "Person"), ("Dept", "Org"), ("Employee", "Person")],
             domain=[("memberOf", "Person")], range=[("memberOf", "Org")],
             subproperty=[("headOf", "worksFor"), ("worksFor", "memberOf")],
             transitive=["subOrgOf"], chain=[("memberOf", "subOrgOf", "memberOf")],
             exists=[("headOf", "Dept", "Chair")], conj=[("Student", "Employee", "WorkingStudent")])
    A = [("ann", TYPE, "GradStudent"), ("ann", "memberOf", "cs"), ("cs", "subOrgOf", "kit"), ("ann", TYPE, "Employee"),
         ("bob", "headOf", "cs"), ("cs", TYPE, "Dept"), ("bob", "memberOf", "cs"), ("eve", "memberOf", "cs")]
    inst = build_instance(t, A)
    assert len(inst.targets) == 12
    opt = max(coverage(inst, set(S)) for S in itertools.combinations(inst.abox, 3))
    assert opt == 7
    assert coverage(inst, bl.degree_baseline(inst, 3)) == 4
    assert coverage(inst, justification_greedy(inst, 3)) == 7
    assert coverage(inst, ilp(inst, 3)["S"]) == 7


def test_fR_overlapping_justifications_shared_triple_deletion():
    """Two justifications J1={a,b}, J2={a,c} of the same target share triple a.
    f_R (r=2) counts both distinct justifications; deleting the shared triple a
    invalidates both, so f_E and f_R drop to 0 together."""
    from spare import robust_coverage
    t = TBox(conj=[("A", "B", "D"), ("A", "C", "D")])
    a, b, c = ("x", TYPE, "A"), ("x", TYPE, "B"), ("x", TYPE, "C")
    inst = build_instance(t, [a, b, c])
    tgt = ("x", TYPE, "D")
    assert sorted(map(sorted, inst.just[tgt])) == sorted([sorted([a, b]), sorted([a, c])])
    S = {a, b, c}
    assert robust_coverage(inst, S, r=2) == 1.0 and coverage(inst, S) == 1.0
    S_del = S - {a}
    assert robust_coverage(inst, S_del, r=2) == 0.0 and coverage(inst, S_del) == 0.0
    assert tgt not in closure_of(t, S_del)
    S_b = S - {b}                       # deleting a non-shared triple keeps J2
    assert coverage(inst, S_b) == 1.0 and robust_coverage(inst, S_b, r=2) == 0.5


def test_robust_greedy_keeps_target_eligible_until_r():
    """With r=2 and budget 3 the robust greedy must add the second
    justification of the same target (a target with one justification stays
    eligible), whereas r=1 has no reason to."""
    from spare import robust_coverage, robust_greedy
    t = TBox(conj=[("A", "B", "D"), ("A", "C", "D")])
    A = [("x", TYPE, "A"), ("x", TYPE, "B"), ("x", TYPE, "C")]
    inst = build_instance(t, A)
    S = robust_greedy(inst, 3, r=2)
    assert S == set(A) and robust_coverage(inst, S, r=2) == 1.0


def test_robust_greedy_spends_budget_and_matches_ilp_on_small_instance():
    from spare import robust_coverage, robust_greedy
    t, A = synth_j(p_multi=0.5, n_ind=30, n_edges=50, seed=5)
    inst = build_instance(t, A)
    B = len(A) // 5
    S = robust_greedy(inst, B, r=2)
    assert len(S) <= B
    useful = {a for tg in inst.targets for j in inst.just[tg] for a in j}
    assert len(S) == min(B, len(useful))       # fill spends the budget while gains exist or potential remains
    opt = ilp(inst, B, r=2)
    assert robust_coverage(inst, S, 2) <= opt["objective"] + 1e-6


def test_proposition_fw_equals_macro_recall():
    """Proposition 1: with targets = all certain answers of nonempty atomic
    queries and w = Σ 1/|cert(q,K)| over the queries an answer belongs to,
    f_E(S)/|W| equals macro-averaged certain-answer recall."""
    from collections import defaultdict
    from spare.kb import query_key
    t, A = ub_synth(scale=0.2, seed=2)
    inst = build_instance(t, A, weight="macro-query", include_asserted=True)
    full = defaultdict(set)
    for f in inst.closure:
        full[query_key(f)].add(f)
    W = [q for q, ans in full.items() if ans]
    assert set(inst.targets) == set(inst.closure)
    rng = random.Random(1)
    for frac in [0.05, 0.2, 0.5]:
        S = set(rng.sample(inst.abox, k=int(frac * len(inst.abox))))
        cl = closure_of(t, S)
        macro = sum(len(full[q] & cl) / len(full[q]) for q in W) / len(W)
        fe = sum(inst.weight[x] for x in inst.targets if x in cl)
        assert abs(fe / len(W) - macro) < 1e-9
