"""Prepare the external benchmarks used in E8 (all sources are public GitHub
repositories; commits pinned below). Run once; writes data/external/*.

  LUBM(1)      original UBA 1.7 generator (zzxx-husky/LUBM-COPY, seed 0) +
               univ-bench.owl (wouterbeek/lubm)
  OWL2Bench    official generator jar (kracr/owl2bench), 1 university, seed 1,
               profiles QL, RL, EL; ABox namespace aligned to the TBox namespace
  DBpedia      real DBpedia instance triples and types from the JOIE DB111K
               release (JunhengH/joie-kdd19) + the official DBpedia ontology
               (dbpedia/ontology-tracker); subject-sampled to ~30k triples

Usage: python experiments/prepare_real_data.py --ext /path/to/clones
(the clones are expected under --ext with the directory names used below).
"""
import argparse
import glob
import json
import os
import random
import subprocess

SOURCES = {
    "owl2bench": ("https://github.com/kracr/owl2bench.git", "5a27536c8e7f7621b34c61c8a57ed9eb63e7829a"),
    "joie": ("https://github.com/JunhengH/joie-kdd19.git", "6ce57ae30fae67bf6e2f48e7e6db35dbab3e0201"),
    "zzxx-husky_LUBM-COPY": ("https://github.com/zzxx-husky/LUBM-COPY.git", "3bdf5e65d6518cb30e299d74156f4bb27ed586e8"),
    "wouterbeek_lubm": ("https://github.com/wouterbeek/lubm.git", "003eada5de5ed1919b83acd9443942adb8532248"),
    "dbo-tracker": ("https://github.com/dbpedia/ontology-tracker.git", "32f457a42b910e39ef087b26f1664bd8d507518b"),
}
DBR = "http://dbpedia.org/resource/"
DBO = "http://dbpedia.org/ontology/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def camel(s: str) -> str:
    return "".join(w[:1].upper() + w[1:] for w in s.split("_"))


def iri(x: str) -> str:
    from urllib.parse import quote
    return "<" + DBR + quote(x, safe="_(),.'-!:~*") + ">"


def dbpedia(ext: str, out: str, target_triples: int = 30000, seed: int = 0):
    import rdflib
    from rdflib import OWL, RDF
    g = rdflib.Graph()
    g.parse(os.path.join(ext, "dbo-tracker/ontology/ontology.owl"))
    props = {str(p)[len(DBO):] for t in (OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property)
             for p in g.subjects(RDF.type, t) if str(p).startswith(DBO)}
    classes = {str(c)[len(DBO):] for c in g.subjects(RDF.type, OWL.Class) if str(c).startswith(DBO)}
    D = os.path.join(ext, "joie/data/dbpedia")
    ins = [l.rstrip("\n").split("\t") for l in open(os.path.join(D, "db_insnet.txt"), encoding="utf-8")]
    ins = [(s, p, o) for s, p, o in ins if p in props]
    typ = [l.rstrip("\n").split("\t") for f in ("db_InsType_train.txt", "db_InsType_test.txt")
           for l in open(os.path.join(D, f), encoding="utf-8")]
    typ = [(s, camel(c)) for s, _, c in typ if camel(c) in classes]
    by_subj = {}
    for s, p, o in ins:
        by_subj.setdefault(s, []).append((s, p, o))
    tby = {}
    for s, c in typ:
        tby.setdefault(s, []).append(c)
    subjects = sorted(by_subj)
    random.Random(seed).shuffle(subjects)
    lines, chosen = [], 0
    for s in subjects:
        if len(lines) >= target_triples:
            break
        chosen += 1
        for _, p, o in by_subj[s]:
            lines.append(f"{iri(s)} <{DBO}{p}> {iri(o)} .")
        for c in tby.get(s, []):
            lines.append(f"{iri(s)} <{RDF_TYPE}> <{DBO}{c}> .")
    lines = sorted(set(lines))
    with open(os.path.join(out, "dbpedia_joie_sample.nt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"subjects_sampled": chosen, "triples": len(lines), "relations_matched": len({p for _, p, _ in ins}),
            "types_matched": len({c for _, c in typ})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "external"))
    ap.add_argument("--skip-generators", action="store_true",
                    help="skip running the LUBM and OWL2Bench generators (Java)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = {"sources": SOURCES}
    meta["dbpedia"] = dbpedia(a.ext, a.out)
    print(meta["dbpedia"])
    if not a.skip_generators:
        print("Generators: see README (java -jar OWL2Bench.jar 1 <QL|RL|EL> 1; "
              "javac + java edu.lehigh.swat.bench.uba.Generator -univ 1 -index 0 -seed 0)")
    with open(os.path.join(a.out, "SOURCES.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main()
