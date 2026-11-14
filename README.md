# SPARE — Sparsification Preserving Assertions, Reasoning and Evidence

Budgeted justification-preserving sparsification (BJPS) of OWL knowledge graphs.
Given an OWL knowledge base (T, A), a budget B and weighted target entailments,
SPARE selects S ⊆ A with cost(S) ≤ B that maximises the weight of entailments
retained with at least one complete justification. The TBox is fixed, S consists
of real asserted triples, and every inference from T ∪ S is sound.

This repository contains the implementation, experiment scripts, data and raw
results for the article *SPARE: Justification-Preserving Sparsification of OWL
Knowledge Graphs under a Budget* (M. Jalali, E. Kheirkhah, B. Vu).

## Repository layout

| Path | Contents |
| --- | --- |
| `spare/kb.py` | TBox model, OWL 2 RL-style reasoner with why-provenance, justification extraction, instance builder |
| `spare/algorithms.py` | Objectives f_E / f_R, A1 triple greedy, A2 justification greedy, A3 integer program (SciPy/HiGHS), hybrid |
| `spare/baselines.py` | Random, Degree, PageRank, Entity ablation, JustFreq |
| `spare/generators.py` | UB-Synth (LUBM-style, RDFS and RL profiles), SynthJ (controlled multi-premise share), densest-k-subgraph gadget |
| `spare/io.py` | Loader for real RDF/OWL data (rdflib) |
| `tests/test_core.py` | 12 tests: lemmas, hardness reduction, greedy ratio, running example, robustness, query-recall objective |
| `experiments/run_all.py` | Experiments E1–E6 on synthetic data → `results/*.csv` |
| `experiments/e2_bounds.py` | Dual bounds for the capped objective on SynthJ → `results/e2b_bounds.csv` |
| `experiments/e7_cap_check.py` | Effect of the justification cap → `results/e7_cap_check.csv` |
| `experiments/prepare_real_data.py` | Builds the DBpedia sample and records pinned source commits in `data/external/SOURCES.json` |
| `experiments/run_external.py` | E8: LUBM(1), OWL2Bench QL/RL⁻/EL⁻ and DBpedia → `results/e8_*.csv` |
| `experiments/e9_kgrag_llm.py` | E9: KG-RAG question answering with an LLM (Gemini or Anthropic API; key read from an environment variable) |
| `experiments/make_figures.py` | Figures and LaTeX tables from the result files (written to `manuscript/figures`, `manuscript/tables`) |
| `data/external/` | Original LUBM and OWL2Bench data, DBpedia ontology and sample (see `data/external/README.md`) |
| `results/` | Raw result files of all experiments reported in the article |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests                  # 12 tests, a few seconds
python experiments/run_all.py --quick      # fast sanity run
python experiments/run_all.py              # synthetic experiments E1–E6
python experiments/run_external.py         # external benchmarks and DBpedia (E8)
python experiments/make_figures.py
```

For E9, set `GEMINI_API_KEY` (or `ANTHROPIC_API_KEY`) in your environment and run
`python experiments/e9_kgrag_llm.py --check` first.

## Using SPARE on your own knowledge graph

```python
from spare import build_instance, justification_greedy, triple_greedy, ilp, coverage, total_weight
from spare.io import load_kb

tbox, abox, report = load_kb("ontology.ttl", "data.nt")
print(report)                                   # what was loaded / skipped
inst = build_instance(tbox, abox, m=8)          # m = justification cap per fact
B = int(0.1 * len(abox))
S = justification_greedy(inst, B)               # or triple_greedy / ilp(inst, B)["S"]
print(coverage(inst, S) / total_weight(inst))
```

For query recall instead of entailment coverage, use
`build_instance(tbox, abox, weight="macro-query", include_asserted=True)`.

## Supported OWL 2 RL fragment

Single-premise rules: `rdfs:subClassOf`, `rdfs:subPropertyOf`, `rdfs:domain`,
`rdfs:range`, `owl:inverseOf`, `owl:SymmetricProperty`.
Multi-premise rules: `owl:TransitiveProperty`, `owl:propertyChainAxiom` (length 2),
`owl:intersectionOf` (OWL 2 RL `cls-int1`, `cls-int2`) and `owl:someValuesFrom`
restrictions on the left-hand side of `rdfs:subClassOf` / `owl:equivalentClass`
(translated with auxiliary classes).

## Licence

Code: MIT (see `LICENSE`). Files in `data/external/` keep the licences of their
original sources.
