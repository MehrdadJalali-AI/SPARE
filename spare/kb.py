"""Knowledge base, OWL 2 RL-lite forward-chaining reasoner with why-provenance,
and ABox-justification extraction.

Facts are triples (s, p, o). Concept assertions use p == TYPE.
The TBox is held fixed; only ABox assertions are candidates for sparsification.

Supported TBox axioms (each maps to one OWL 2 RL rule):

  single-premise ("linear") rules -> every ABox justification is a singleton
    subclass       C ⊑ D                 cax-sco
    subproperty    P ⊑ Q                 prp-spo1
    domain         dom(P) = C            prp-dom
    range          rng(P) = C            prp-rng
    inverse        P ≡ Q⁻                prp-inv1/2
    symmetric      P ≡ P⁻                prp-symp

  multi-premise rules -> justifications of size >= 2 can arise
    transitive     P ∘ P ⊑ P             prp-trp
    chain          P1 ∘ P2 ⊑ Q           prp-spo2
    conjunction    C1 ⊓ C2 ⊑ D           cls-int1 + cax-sco (intersection on the left-hand side)
    exists         ∃P.C ⊑ D              cls-svf1 (C may be THING)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

TYPE = "rdf:type"
THING = "owl:Thing"
AUX = "spare-aux:"   # auxiliary classes introduced by the loader; never targets

Fact = Tuple[str, str, str]
Just = FrozenSet[Fact]


@dataclass
class TBox:
    subclass: List[Tuple[str, str]] = field(default_factory=list)
    subproperty: List[Tuple[str, str]] = field(default_factory=list)
    domain: List[Tuple[str, str]] = field(default_factory=list)
    range: List[Tuple[str, str]] = field(default_factory=list)
    inverse: List[Tuple[str, str]] = field(default_factory=list)
    symmetric: List[str] = field(default_factory=list)
    transitive: List[str] = field(default_factory=list)
    chain: List[Tuple[str, str, str]] = field(default_factory=list)
    conj: List[Tuple[str, str, str]] = field(default_factory=list)
    exists: List[Tuple[str, str, str]] = field(default_factory=list)

    def is_linear(self) -> bool:
        """True iff only single-premise rules are present (Theorem 2 regime)."""
        return not (self.transitive or self.chain or self.conj or self.exists)

    def restrict_linear(self) -> "TBox":
        return TBox(list(self.subclass), list(self.subproperty), list(self.domain),
                    list(self.range), list(self.inverse), list(self.symmetric))

    def size(self) -> int:
        return sum(len(getattr(self, f)) for f in self.__dataclass_fields__)


class Reasoner:
    """Semi-naive forward chaining that records every rule instance.

    After `materialise`, `self.derivations[f]` is a list of premise tuples, one
    per rule instance that derives f. Asserted facts have no premises recorded
    as derivations; they are their own base case.
    """

    def __init__(self, tbox: TBox):
        self.t = tbox
        self.sub_c = defaultdict(list)
        for c, d in tbox.subclass:
            self.sub_c[c].append(d)
        self.sub_p = defaultdict(list)
        for p, q in tbox.subproperty:
            self.sub_p[p].append(q)
        self.dom = defaultdict(list)
        for p, c in tbox.domain:
            self.dom[p].append(c)
        self.rng = defaultdict(list)
        for p, c in tbox.range:
            self.rng[p].append(c)
        self.inv = defaultdict(list)
        for p, q in tbox.inverse:
            self.inv[p].append(q)
            self.inv[q].append(p)
        self.sym = set(tbox.symmetric)
        self.trans = set(tbox.transitive)
        self.chain1 = defaultdict(list)  # p1 -> (p2, q)
        self.chain2 = defaultdict(list)  # p2 -> (p1, q)
        for p1, p2, q in tbox.chain:
            self.chain1[p1].append((p2, q))
            self.chain2[p2].append((p1, q))
        self.conj = defaultdict(list)  # c -> (other, d)
        for c1, c2, d in tbox.conj:
            self.conj[c1].append((c2, d))
            self.conj[c2].append((c1, d))
        self.ex_p = defaultdict(list)  # p -> (c, d)
        self.ex_c = defaultdict(list)  # c -> (p, d)
        for p, c, d in tbox.exists:
            self.ex_p[p].append((c, d))
            self.ex_c[c].append((p, d))

    def materialise(self, abox: Iterable[Fact], record: bool = True):
        facts: Set[Fact] = set(abox)
        self.asserted = set(facts)
        self.derivations: Dict[Fact, List[Tuple[Fact, ...]]] = defaultdict(list)
        types = defaultdict(set)         # x -> {C}
        out = defaultdict(lambda: defaultdict(set))   # p -> s -> {o}
        inn = defaultdict(lambda: defaultdict(set))   # p -> o -> {s}

        def index(f):
            s, p, o = f
            if p == TYPE:
                types[s].add(o)
            else:
                out[p][s].add(o)
                inn[p][o].add(s)

        for f in facts:
            index(f)
        delta = set(facts)
        seen_inst = set()

        def emit(new, prem, buf):
            if record:
                key = (new, prem)
                if key not in seen_inst:
                    seen_inst.add(key)
                    self.derivations[new].append(prem)
            if new not in facts:
                buf.add(new)

        while delta:
            nxt: Set[Fact] = set()
            for f in delta:
                s, p, o = f
                if p == TYPE:
                    c = o
                    for d in self.sub_c.get(c, ()):
                        emit((s, TYPE, d), (f,), nxt)
                    for c2, d in self.conj.get(c, ()):
                        if c2 in types[s]:
                            emit((s, TYPE, d), tuple(sorted((f, (s, TYPE, c2)))), nxt)
                    for q, d in self.ex_c.get(c, ()):
                        for x in inn[q].get(s, ()):
                            emit((x, TYPE, d), tuple(sorted(((x, q, s), f))), nxt)
                else:
                    for q in self.sub_p.get(p, ()):
                        emit((s, q, o), (f,), nxt)
                    for c in self.dom.get(p, ()):
                        emit((s, TYPE, c), (f,), nxt)
                    for c in self.rng.get(p, ()):
                        emit((o, TYPE, c), (f,), nxt)
                    for q in self.inv.get(p, ()):
                        emit((o, q, s), (f,), nxt)
                    if p in self.sym:
                        emit((o, p, s), (f,), nxt)
                    if p in self.trans:
                        for z in list(out[p].get(o, ())):
                            if z != s:
                                emit((s, p, z), tuple(sorted((f, (o, p, z)))), nxt)
                        for w in list(inn[p].get(s, ())):
                            if w != o:
                                emit((w, p, o), tuple(sorted(((w, p, s), f))), nxt)
                    for p2, q in self.chain1.get(p, ()):
                        for z in list(out[p2].get(o, ())):
                            emit((s, q, z), tuple(sorted((f, (o, p2, z)))), nxt)
                    for p1, q in self.chain2.get(p, ()):
                        for w in list(inn[p1].get(s, ())):
                            emit((w, q, o), tuple(sorted(((w, p1, s), f))), nxt)
                    for c, d in self.ex_p.get(p, ()):
                        if c == THING:
                            emit((s, TYPE, d), (f,), nxt)
                        elif c in types[o]:
                            emit((s, TYPE, d), tuple(sorted((f, (o, TYPE, c)))), nxt)
            for g in nxt:
                facts.add(g)
                index(g)
            delta = nxt
        self.closure = facts
        return facts


def _minimise(cands: List[Just], m: int) -> List[Just]:
    cands = sorted(set(cands), key=lambda j: (len(j), sorted(j)))
    keep: List[Just] = []
    for j in cands:
        if not any(k <= j for k in keep):
            keep.append(j)
            if len(keep) >= m:
                break
    return keep


def justifications(reasoner: Reasoner, m: int = 8, max_rounds: int = 50
                   ) -> Dict[Fact, List[Just]]:
    """ABox justifications for every fact in the closure (why-provenance fixpoint).

    Returns up to m subset-minimal justifications per fact, smallest first.
    With the cap, the result is a sound under-approximation of Just(α):
    every returned set J satisfies T ∪ J ⊨ α and is minimal among the
    candidates found. For linear TBoxes it is exact up to the cap.
    """
    J: Dict[Fact, List[Just]] = {f: [frozenset([f])] for f in reasoner.asserted}
    derived = [f for f in reasoner.derivations]
    for _ in range(max_rounds):
        changed = False
        for f in derived:
            cands = list(J.get(f, []))
            for prem in reasoner.derivations[f]:
                lists = [J.get(p) for p in prem]
                if any(not l for l in lists):
                    continue
                for combo in product(*lists):
                    cands.append(frozenset().union(*combo))
            new = _minimise(cands, m)
            if new != J.get(f):
                J[f] = new
                changed = True
        if not changed:
            break
    return J


@dataclass
class Instance:
    """A BJPS instance: TBox, ABox, targets with weights, justifications, costs."""
    tbox: TBox
    abox: List[Fact]
    targets: List[Fact]
    weight: Dict[Fact, float]
    just: Dict[Fact, List[Just]]
    cost: Dict[Fact, float]
    closure: Set[Fact]

    @property
    def max_just_size(self) -> int:
        return max((len(j) for t in self.targets for j in self.just[t]), default=0)


def query_key(f: Fact):
    """Atomic query a fact answers: instance retrieval C(?x) or role P(?x,?y)."""
    return ("C", f[2]) if f[1] == TYPE else ("P", f[1])


def build_instance(tbox: TBox, abox: Iterable[Fact], m: int = 8,
                   weight: str = "uniform", prov_cost: Dict[Fact, int] | None = None,
                   include_asserted: bool = False) -> Instance:
    """weight: 'uniform' | 'type-heavy' | 'macro-query'.

    'macro-query' (variant W) gives each fact weight 1/|answers of its atomic
    query| and, with include_asserted=True, also targets asserted facts, so
    that f_E(S)/#queries equals the macro-averaged certain-answer recall.
    """
    abox = sorted(set(abox))
    r = Reasoner(tbox)
    closure = r.materialise(abox)
    J = justifications(r, m=m)
    aset = set(abox)
    targets = sorted(f for f in closure if (include_asserted or f not in aset) and J.get(f)
                     and not (f[1] == TYPE and f[2].startswith(AUX)))
    if weight == "uniform":
        w = {t: 1.0 for t in targets}
    elif weight == "type-heavy":
        w = {t: (2.0 if t[1] == TYPE else 1.0) for t in targets}
    elif weight == "macro-query":
        cnt = defaultdict(int)
        for f in closure:
            if not (f[1] == TYPE and f[2].startswith(AUX)):
                cnt[query_key(f)] += 1
        w = {t: 1.0 / cnt[query_key(t)] for t in targets}
    else:
        raise ValueError(weight)
    cost = {a: 1.0 + (prov_cost.get(a, 0) if prov_cost else 0) for a in abox}
    return Instance(tbox, abox, targets, w, {t: J[t] for t in targets}, cost, closure)


def closure_of(tbox: TBox, facts: Iterable[Fact]) -> Set[Fact]:
    return Reasoner(tbox).materialise(facts, record=False)
