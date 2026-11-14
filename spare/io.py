"""Load real RDF/OWL data into SPARE (requires `rdflib`).

    from spare.io import load_kb
    tbox, abox = load_kb("ontology.ttl", "data.nt")

TBox axioms outside the supported OWL 2 RL-lite fragment are skipped and
counted in the returned report (so the fragment actually used is explicit).
ABox = all triples whose predicate is not an RDFS/OWL schema term.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from .kb import AUX, THING, TYPE, TBox, Fact

SCHEMA_NS = ("http://www.w3.org/2000/01/rdf-schema#", "http://www.w3.org/2002/07/owl#")


def _short(term) -> str:
    return str(term)


def load_kb(tbox_path: str, abox_path=None, abox_format: str | None = None,
            namespace_map: dict | None = None, tbox_format: str | None = None
            ) -> Tuple[TBox, List[Fact], Counter]:
    """namespace_map: {old_prefix: new_prefix} applied to ABox IRIs, e.g. to align
    OWL2Bench ABox terms (https://kracr.iiitd.edu.in/OWL2Bench#) with its TBox
    namespace (http://benchmark/OWL2Bench#)."""
    import rdflib
    from rdflib import OWL, RDF, RDFS
    from rdflib.collection import Collection

    g = rdflib.Graph()
    g.parse(tbox_path, format=tbox_format)
    t = TBox()
    report = Counter()
    named = lambda x: isinstance(x, rdflib.URIRef)

    aux = [0]

    def fresh() -> str:
        aux[0] += 1
        return f"{AUX}{aux[0]}"

    def svf(node):
        if (node, RDF.type, OWL.Restriction) in g:
            p = g.value(node, OWL.onProperty)
            c = g.value(node, OWL.someValuesFrom)
            if named(p) and c is not None and named(c):
                return _short(p), _short(c)
        return None

    def lhs_class(node):
        """Return a class name standing for a left-hand-side expression
        (named class, ∃P.C, or a binary/n-ary intersection of these), adding
        auxiliary rules as needed; None if unsupported. Sound for atomic
        entailments: auxiliary classes only abbreviate sub-expressions."""
        if named(node):
            return _short(node)
        r = svf(node)
        if r:
            x = fresh()
            t.exists.append((r[0], r[1], x))
            return x
        inter = g.value(node, OWL.intersectionOf)
        if inter is not None:
            parts = [lhs_class(i) for i in Collection(g, inter)]
            if not parts or any(q is None for q in parts):
                return None
            cur = parts[0]
            for q in parts[1:]:
                x = fresh()
                t.conj.append((cur, q, x))
                cur = x
            return cur
        return None

    def add_lhs(node, d: str) -> bool:
        """node ⊑ d for a left-hand-side expression node."""
        union = g.value(node, OWL.unionOf)
        if union is not None:
            items = list(Collection(g, union))
            cs = [lhs_class(i) for i in items]
            if any(c is None for c in cs):
                return False
            for c in cs:
                t.subclass.append((c, d))
            return True
        c = lhs_class(node)
        if c is None:
            return False
        if c.startswith(AUX) and t.conj and t.conj[-1][2] == c:
            a, b, _ = t.conj.pop()          # name the final conjunction d directly
            t.conj.append((a, b, d))
        elif c.startswith(AUX) and t.exists and t.exists[-1][2] == c:
            pp, cc, _ = t.exists.pop()
            t.exists.append((pp, cc, d))
        else:
            t.subclass.append((c, d))
        return True

    def add_rhs(c: str, node) -> bool:
        """c ⊑ node for a right-hand-side expression: keep the named
        conjuncts (OWL 2 RL cls-int2); existential/other conjuncts would
        need new individuals or negation and are dropped (reported)."""
        if named(node):
            t.subclass.append((c, _short(node)))
            return True
        inter = g.value(node, OWL.intersectionOf)
        if inter is not None:
            kept = False
            for i in Collection(g, inter):
                if named(i):
                    t.subclass.append((c, _short(i)))
                    kept = True
                else:
                    report["dropped_rhs_conjunct"] += 1
            return kept
        return False

    for c, d in g.subject_objects(RDFS.subClassOf):
        if named(c) and named(d):
            t.subclass.append((_short(c), _short(d)))
        elif named(d):
            if not add_lhs(c, _short(d)):
                report["skipped_subClassOf"] += 1
        elif named(c):
            if not add_rhs(_short(c), d):
                report["skipped_subClassOf"] += 1
        else:
            report["skipped_subClassOf"] += 1
    for c, d in g.subject_objects(OWL.equivalentClass):
        if named(c) and named(d):
            t.subclass += [(_short(c), _short(d)), (_short(d), _short(c))]
            continue
        if not named(c):
            c, d = d, c
        if not named(c):
            report["skipped_equivalentClass"] += 1
            continue
        ok1 = add_lhs(d, _short(c))       # d ⊑ c
        ok2 = add_rhs(_short(c), d)       # c ⊑ d (named conjuncts only)
        if not (ok1 or ok2):
            report["skipped_equivalentClass"] += 1
        elif not ok1:
            report["equivalentClass_rhs_only"] += 1
    # OWL 1 style definitions: a named class with owl:intersectionOf is
    # equivalent to that intersection (e.g. LUBM's univ-bench.owl).
    for c, lst in g.subject_objects(OWL.intersectionOf):
        if not named(c):
            continue
        items = list(Collection(g, lst))
        parts = [lhs_class(i) for i in items]
        if parts and all(q is not None for q in parts):
            cur = parts[0]
            for i, q in enumerate(parts[1:]):
                nxt = _short(c) if i == len(parts) - 2 else fresh()
                t.conj.append((cur, q, nxt))
                cur = nxt
            if len(parts) == 1:
                t.subclass.append((cur, _short(c)))
        else:
            report["skipped_named_intersection"] += 1
        for i in items:                          # c ⊑ named conjuncts
            if named(i):
                t.subclass.append((_short(c), _short(i)))
        report["named_intersection_definitions"] += 1

    for p, q in g.subject_objects(RDFS.subPropertyOf):
        if named(p) and named(q):
            t.subproperty.append((_short(p), _short(q)))
    for p, c in g.subject_objects(RDFS.domain):
        if named(c):
            t.domain.append((_short(p), _short(c)))
    for p, c in g.subject_objects(RDFS.range):
        if named(c) and not str(c).startswith("http://www.w3.org/2001/XMLSchema#"):
            t.range.append((_short(p), _short(c)))
    for p, q in g.subject_objects(OWL.inverseOf):
        t.inverse.append((_short(p), _short(q)))
    for p in g.subjects(RDF.type, OWL.SymmetricProperty):
        t.symmetric.append(_short(p))
    for p in g.subjects(RDF.type, OWL.TransitiveProperty):
        t.transitive.append(_short(p))
    for q, lst in g.subject_objects(OWL.propertyChainAxiom):
        items = list(Collection(g, lst))
        if len(items) == 2:
            t.chain.append((_short(items[0]), _short(items[1]), _short(q)))
        else:
            report["skipped_chain_len!=2"] += 1

    abox: List[Fact] = []
    if abox_path:
        h = rdflib.Graph()
        for ap in ([abox_path] if isinstance(abox_path, str) else list(abox_path)):
            h.parse(ap, format=abox_format)
        nm = namespace_map or {}

        def m(x: str) -> str:
            for old, new in nm.items():
                if x.startswith(old):
                    report["namespace_mapped"] += 1
                    return new + x[len(old):]
            return x
        for s, p, o in h:
            if isinstance(o, rdflib.Literal):
                report["skipped_literal"] += 1
                continue
            if p == RDF.type:
                if str(o).startswith(SCHEMA_NS):
                    continue
                abox.append((m(_short(s)), TYPE, m(_short(o))))
            elif not str(p).startswith(SCHEMA_NS):
                abox.append((m(_short(s)), m(_short(p)), m(_short(o))))
    report["abox"] = len(abox)
    report["tbox_axioms"] = t.size()
    return t, sorted(set(abox)), report
