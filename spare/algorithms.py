"""Solvers for budgeted justification-preserving sparsification (BJPS).

A1  triple_greedy          marginal-gain greedy over single triples
                           ((1-1/e) guarantee when every justification is a singleton)
A2  justification_greedy   greedy over whole justifications + fill + prune
A3  ilp                    exact MILP (scipy/HiGHS), small instances
    robust_greedy          A2 variant for the r-robust objective f_R
"""
from __future__ import annotations

import heapq
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np

from .kb import Fact, Instance, Just


# ---------------------------------------------------------------- objectives
def coverage(inst: Instance, S: Set[Fact]) -> float:
    """f_E(S): weighted targets with at least one justification inside S."""
    tot = 0.0
    for t in inst.targets:
        if any(j <= S for j in inst.just[t]):
            tot += inst.weight[t]
    return tot


def robust_coverage(inst: Instance, S: Set[Fact], r: int = 2) -> float:
    """f_R(S): weighted fraction of up to r justifications kept per target."""
    tot = 0.0
    for t in inst.targets:
        k = sum(1 for j in inst.just[t] if j <= S)
        tot += inst.weight[t] * min(r, k) / r
    return tot


def total_weight(inst: Instance) -> float:
    return sum(inst.weight.values())


# ------------------------------------------------------------ shared state
class _State:
    """Incremental bookkeeping of missing triples per justification."""

    def __init__(self, inst: Instance):
        self.inst = inst
        self.jlist: List[Just] = []
        self.jtarget: List[Fact] = []
        self.containing = defaultdict(list)   # triple -> [j index]
        for t in inst.targets:
            for j in inst.just[t]:
                k = len(self.jlist)
                self.jlist.append(j)
                self.jtarget.append(t)
                for a in j:
                    self.containing[a].append(k)
        self.missing = [len(j) for j in self.jlist]
        self.S: Set[Fact] = set()
        self.covered: Set[Fact] = set()
        self.ncov = defaultdict(int)          # target -> #justifications fully kept
        self.spent = 0.0

    def add(self, a: Fact):
        if a in self.S:
            return
        self.S.add(a)
        self.spent += self.inst.cost[a]
        for k in self.containing.get(a, ()):
            self.missing[k] -= 1
            if self.missing[k] == 0:
                t = self.jtarget[k]
                self.ncov[t] += 1
                self.covered.add(t)

    def gain(self, a: Fact, r: int = 1) -> float:
        """Exact marginal gain of the (capped) objective for adding triple a.

        r = 1: f_E (a target counts once it has one retained justification).
        r > 1: f_R (each distinct retained justification counts, up to r per
        target). Justifications that share triple a are counted separately,
        exactly as f_R counts them.
        """
        if a in self.S:
            return 0.0
        newly = defaultdict(int)
        for k in self.containing.get(a, ()):
            if self.missing[k] == 1:
                newly[self.jtarget[k]] += 1
        g = 0.0
        for t, n in newly.items():
            have = self.ncov[t]
            if have < r:
                g += self.inst.weight[t] * (min(r, have + n) - have) / r
        return g

    def potential(self, a: Fact, r: int = 1) -> float:
        """Fractional credit Σ w/(r·missing) over justifications of targets
        that still have fewer than r retained justifications (tie-breaker)."""
        g = 0.0
        for k in self.containing.get(a, ()):
            t = self.jtarget[k]
            if self.ncov[t] < r and self.missing[k] > 0:
                g += self.inst.weight[t] / (r * self.missing[k])
        return g


# ------------------------------------------------------------------- A1
def _greedy_fill(st: "_State", budget: float, use_potential: bool = True,
                 require_gain: bool = False, r: int = 1) -> None:
    """Exact greedy on key (gain/cost, potential/cost) with incremental updates.

    Keys of a triple change only when a justification containing it changes
    or the retained-justification count of one of its targets changes, so we
    re-key exactly those triples after each pick. Works for submodular and
    non-submodular objectives; r selects f_E (r = 1) or f_R (r > 1).
    """
    inst = st.inst
    tj = defaultdict(list)
    for k, t in enumerate(st.jtarget):
        tj[t].append(k)
    ver = defaultdict(int)

    def key(a):
        c = inst.cost[a]
        p = st.potential(a, r) / c if use_potential else 0.0
        return (st.gain(a, r) / c, p)

    heap = []
    for a in st.containing:
        if a not in st.S:
            g, p = key(a)
            heap.append((-g, -p, ver[a], a))
    heapq.heapify(heap)
    while heap:
        ng, np_, v, a = heapq.heappop(heap)
        if a in st.S or v != ver[a]:
            continue
        if (ng, np_) == (0.0, 0.0) or (require_gain and ng >= 0):
            break
        if st.spent + inst.cost[a] > budget + 1e-9:
            continue
        affected_t = {st.jtarget[k] for k in st.containing[a]}
        before = {t: st.ncov[t] for t in affected_t}
        st.add(a)
        touched = set()
        for k in st.containing[a]:
            touched |= st.jlist[k]
        for t in affected_t:
            if st.ncov[t] != before[t]:
                for k in tj[t]:
                    touched |= st.jlist[k]
        for b in touched:
            if b not in st.S:
                ver[b] += 1
                g, p = key(b)
                heapq.heappush(heap, (-g, -p, ver[b], b))


def triple_greedy(inst: Instance, budget: float) -> Set[Fact]:
    """A1. Pick the triple with the best (gain, potential) per cost.

    For linear TBoxes f_E is monotone submodular and this is the classic
    greedy with the (1-1/e) guarantee (unit costs). For non-linear TBoxes
    gains can grow (complementarity); zero-gain ties fall back to the
    fractional potential Σ w/missing so the method does not stall.
    """
    st = _State(inst)
    _greedy_fill(st, budget, use_potential=True)
    return st.S


# ------------------------------------------------------------------- A2
def justification_greedy(inst: Instance, budget: float, r: int = 1,
                         fill: bool = True, prune: bool = True) -> Set[Fact]:
    """A2. Greedy over whole justifications with lazy re-evaluation.

    Score of candidate J = weight of all targets for which adding J \\ S
    completes new justifications (each counted towards at most r retained
    justifications, i.e. the marginal gain in f_E for r = 1 or f_R for r > 1)
    divided by cost(J \\ S). A target stays eligible until it has r retained
    justifications. After the budget is exhausted: prune triples that lie in
    no complete justification, then fill the remaining budget with
    single-triple marginal gains of the same objective.
    """
    st = _State(inst)

    def score(k):
        """Weight of all targets that adding J_k \\ S would newly satisfy
        (towards r justifications each), per unit of added cost."""
        add = [a for a in st.jlist[k] if a not in st.S]
        c = sum(inst.cost[a] for a in add)
        if c == 0:
            return 0.0, 0.0
        addset = set(add)
        newly = defaultdict(int)
        seen = set()
        for a in add:
            for kk in st.containing.get(a, ()):
                if kk in seen or st.missing[kk] == 0:
                    continue
                seen.add(kk)
                if st.missing[kk] == len(st.jlist[kk] & addset):
                    newly[st.jtarget[kk]] += 1
        g = 0.0
        for t, n in newly.items():
            have = st.ncov[t]
            if have < r:
                g += inst.weight[t] * (min(r, have + n) - have) / r
        return g / c, c

    heap = [(-score(k)[0], k) for k in range(len(st.jlist))]
    heapq.heapify(heap)
    while heap:
        negs, k = heapq.heappop(heap)
        t = st.jtarget[k]
        if st.missing[k] == 0 or st.ncov[t] >= r:
            continue
        s, c = score(k)
        if s <= 0:
            continue
        if heap and s < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-s, k))       # lazy re-evaluation
            continue
        if st.spent + c > budget + 1e-9:
            continue
        for a in st.jlist[k]:
            st.add(a)

    S = st.S
    if prune:
        useful = set()
        for k, j in enumerate(st.jlist):
            if st.missing[k] == 0:
                useful |= j
        if useful != S:
            st2 = _State(inst)
            for a in useful:
                st2.add(a)
            st = st2
    if fill:
        _greedy_fill(st, budget, use_potential=True, require_gain=True, r=r)
    return st.S


def robust_greedy(inst: Instance, budget: float, r: int = 2) -> Set[Fact]:
    """A2 for f_R: targets stay eligible until they have r retained
    justifications; pruning keeps every triple of every complete
    justification, and the budget fill uses f_R marginal gains."""
    return justification_greedy(inst, budget, r=r, fill=True, prune=True)


# ------------------------------------------------------------------- A3
def ilp(inst: Instance, budget: float, time_limit: float = 60.0,
        r: Optional[int] = None) -> Dict:
    """A3. Exact MILP with scipy.optimize.milp (HiGHS).

    max Σ w_t y_t  s.t.  Σ c_a x_a <= B ;  z_J <= x_a (a ∈ J) ;
                          y_t <= Σ_J z_J  (or r·y_t <= Σ z_J, y ∈ [0,1], for f_R)
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    t_build0 = time.time()
    triples = sorted({a for t in inst.targets for j in inst.just[t] for a in j})
    xi = {a: i for i, a in enumerate(triples)}
    jl = [(t, j) for t in inst.targets for j in inst.just[t]]
    nx, nz, ny = len(triples), len(jl), len(inst.targets)
    yi = {t: nx + nz + i for i, t in enumerate(inst.targets)}
    n = nx + nz + ny
    rows = 1 + sum(len(j) for _, j in jl) + ny
    A = lil_matrix((rows, n))
    lb = np.full(rows, -np.inf)
    ub = np.zeros(rows)
    for a, i in xi.items():
        A[0, i] = inst.cost[a]
    ub[0] = budget
    row = 1
    for k, (t, j) in enumerate(jl):
        for a in j:
            A[row, nx + k] = 1
            A[row, xi[a]] = -1
            row += 1
    rr = 1 if r is None else r
    tj = defaultdict(list)
    for k, (t, j) in enumerate(jl):
        tj[t].append(k)
    for t in inst.targets:
        A[row, yi[t]] = rr
        for k in tj[t]:
            A[row, nx + k] = -1
        row += 1
    cvec = np.zeros(n)
    for t in inst.targets:
        cvec[yi[t]] = -inst.weight[t]
    integrality = np.ones(n)
    if r is not None:
        integrality[nx + nz:] = 0
    build_s = time.time() - t_build0
    t0 = time.time()
    res = milp(cvec, constraints=LinearConstraint(A.tocsr(), lb, ub),
               integrality=integrality, bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False})
    S = set()
    if res.x is not None:
        S = {a for a, i in xi.items() if res.x[i] > 0.5}
    return {"S": S, "status": res.status, "message": res.message,
            "objective": -res.fun if res.fun is not None else None,
            "bound": -getattr(res, "mip_dual_bound", np.nan) if res.x is not None else None,
            "gap": getattr(res, "mip_gap", None),
            # solve_seconds is bounded by time_limit (HiGHS); build_seconds is
            # model construction before the solver starts; seconds = both.
            "solve_seconds": time.time() - t0, "build_seconds": build_s,
            "seconds": build_s + (time.time() - t0)}


# ------------------------------------------------------------- hybrid
def spare_hybrid(inst: Instance, budget: float, r: int = 1, time_limit: float = 60.0,
                 evaluate=None) -> Set[Fact]:
    """SPARE default: A1, A2 and the time-limited A3 for the same objective;
    return the candidate with the highest value of `evaluate`.

    r = 1 (f_E): `evaluate` should be exact entailment coverage (closure of
    T ∪ S); the experiments pass it. r > 1 (f_R): candidates are compared by
    f_R on the extracted (capped) justification lists, because the exact
    number of distinct justifications is not available beyond the cap.
    """
    if r == 1:
        cands = [justification_greedy(inst, budget), triple_greedy(inst, budget)]
    else:
        cands = [robust_greedy(inst, budget, r=r)]
    res = ilp(inst, budget, time_limit=time_limit, r=None if r == 1 else r)
    if res["S"]:
        cands.append(res["S"])
    if evaluate is None:
        evaluate = (lambda S: coverage(inst, S)) if r == 1 else (lambda S: robust_coverage(inst, S, r))
    return max(cands, key=evaluate)
