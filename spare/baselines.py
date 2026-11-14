"""Semantics-blind and semantics-light baselines (all return S ⊆ A with |S| <= B,
unit costs assumed; for non-unit costs the ranking is consumed until the budget)."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import List, Set

import networkx as nx

from .kb import TYPE, Fact, Instance


def _take(ranked: List[Fact], inst: Instance, budget: float) -> Set[Fact]:
    S, spent = set(), 0.0
    for a in ranked:
        c = inst.cost[a]
        if spent + c <= budget + 1e-9:
            S.add(a)
            spent += c
    return S


def random_baseline(inst: Instance, budget: float, seed: int = 0) -> Set[Fact]:
    rng = random.Random(seed)
    ranked = list(inst.abox)
    rng.shuffle(ranked)
    return _take(ranked, inst, budget)


def _graph(inst: Instance) -> nx.Graph:
    G = nx.Graph()
    for s, p, o in inst.abox:
        G.add_edge(s, o if p != TYPE else "class:" + o)
    return G


def degree_baseline(inst: Instance, budget: float) -> Set[Fact]:
    G = _graph(inst)
    deg = dict(G.degree())
    key = lambda f: deg.get(f[0], 0) + deg.get(f[2] if f[1] != TYPE else "class:" + f[2], 0)
    return _take(sorted(inst.abox, key=lambda f: (-key(f), f)), inst, budget)


def pagerank_baseline(inst: Instance, budget: float) -> Set[Fact]:
    G = _graph(inst)
    pr = nx.pagerank(G, alpha=0.85, max_iter=200)
    key = lambda f: pr.get(f[0], 0) + pr.get(f[2] if f[1] != TYPE else "class:" + f[2], 0)
    return _take(sorted(inst.abox, key=lambda f: (-key(f), f)), inst, budget)


def entity_ablation_baseline(inst: Instance, budget: float) -> Set[Fact]:
    """Entity-level selection in the spirit of Ramalli et al. (FGCS 2026):
    remove low-degree entities first, i.e. keep triples of the highest-degree
    entities until the budget is reached (a triple is kept only when all of
    its individuals are kept)."""
    G = _graph(inst)
    deg = dict(G.degree())
    ents = sorted({f[0] for f in inst.abox} | {f[2] for f in inst.abox if f[1] != TYPE},
                  key=lambda e: (-deg.get(e, 0), e))
    rank = {e: i for i, e in enumerate(ents)}
    # a triple becomes available once its latest-ranked individual is kept
    avail = defaultdict(list)
    for f in inst.abox:
        inds = [f[0]] + ([f[2]] if f[1] != TYPE else [])
        avail[max(rank[e] for e in inds)].append(f)
    ranked = []
    for i in range(len(ents)):
        ranked.extend(sorted(avail.get(i, [])))
    return _take(ranked, inst, budget)


def justfreq_baseline(inst: Instance, budget: float) -> Set[Fact]:
    """Semantics-aware but non-optimising: rank triples by how many target
    justifications they occur in (ablation: is optimisation needed at all?)."""
    cnt = Counter()
    for t in inst.targets:
        for j in inst.just[t]:
            for a in j:
                cnt[a] += inst.weight[t]
    return _take(sorted(inst.abox, key=lambda f: (-cnt[f], f)), inst, budget)
