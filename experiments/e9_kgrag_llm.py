"""E9 (to be run by the authors with their own API key): KG-RAG question
answering over sparsified DBpedia.

Protocol
--------
* KB: the real DBpedia sample of E8 (JOIE DB111K triples + official DBpedia
  ontology), sparsified at budget B (default 10 % of the ABox) by SPARE, Degree,
  JustFreq and Random. Every method keeps the same number of triples.
* Questions: yes/no type questions "Is <entity> a <dbo:Class>?" for
  - positives: inferred (not asserted) dbo: type facts entailed by the FULL KB;
  - negatives: dbo: classes that the full KB does not entail for that entity
    (sampled from classes entailed for other entities), to measure false "yes".
* Context for a question: the retained triples mentioning the entity
  (as subject or object), truncated to --max-context triples, rendered as
  "subject predicate object" lines with DBpedia local names. No TBox is given;
  the model must rely on its own knowledge of the DBpedia vocabulary.
* The model answers YES, NO or UNKNOWN. We report, per method:
    accuracy_full      answer agrees with the full KB (YES on positives, NO/UNKNOWN on negatives)
    supported_yes_rate on positives that ARE entailed by T ∪ S, share answered YES
    unsupported_yes    on positives NOT entailed by T ∪ S, share answered YES
                       (answers not grounded in the retained context)
    false_yes          share of YES on negatives (hallucination)
    context_tokens     mean number of context triples per question
* Reasoner labels (entailed by T ∪ S or not) come from SPARE's reasoner, so the
  faithfulness numbers are exact.

Usage (on a machine with network access to the API; never commit keys):
  Gemini (no extra package needed):
    export GEMINI_API_KEY=...
    python experiments/e9_kgrag_llm.py --provider gemini --model gemini-2.5-flash
  Anthropic:
    pip install anthropic; export ANTHROPIC_API_KEY=...
    python experiments/e9_kgrag_llm.py --provider anthropic --model claude-haiku-5-5
  List the Gemini models your key can use:
    python experiments/e9_kgrag_llm.py --provider gemini --list-models
Writes results/e9_kgrag_<model>.csv (per question) and e9_kgrag_<model>_summary.csv.
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from spare import TYPE, build_instance, closure_of, justification_greedy, triple_greedy  # noqa
from spare import baselines as bl  # noqa
from spare.io import load_kb  # noqa

EXT = os.path.join(ROOT, "data", "external")
RES = os.path.join(ROOT, "results")
DBO = "http://dbpedia.org/ontology/"


def local(x: str) -> str:
    from urllib.parse import unquote
    return unquote(x.rsplit("/", 1)[-1]).replace("_", " ")


def make_questions(full_cl, abox, n_pos, n_neg, seed):
    aset = set(abox)
    rng = random.Random(seed)
    types = defaultdict(set)
    for s, p, o in full_cl:
        if p == TYPE and o.startswith(DBO):
            types[s].add(o)
    pos = sorted((s, c) for s, cs in types.items() for c in cs if (s, TYPE, c) not in aset)
    rng.shuffle(pos)   # sorted first: set order varies between processes
    pos = pos[:n_pos]
    all_classes = sorted({c for cs in types.values() for c in cs})
    ents = sorted(types)
    neg, seen = [], set()
    while len(neg) < n_neg:
        s = rng.choice(ents)
        c = rng.choice(all_classes)
        if c not in types[s] and (s, c) not in seen:
            seen.add((s, c))
            neg.append((s, c))
    return [(s, c, True) for s, c in pos] + [(s, c, False) for s, c in neg]


def context(S, ent, k):
    lines = [f"{local(s)} | {local(p) if p != TYPE else 'type'} | {local(o)}"
             for s, p, o in sorted(S) if s == ent or o == ent]
    return lines[:k]


PROMPT = ("You are given facts from a knowledge graph, one per line as subject | predicate | object.\n"
          "Answer the question using these facts and your knowledge of what the DBpedia predicates and "
          "classes mean. Answer with exactly one word: YES, NO, or UNKNOWN (if the facts do not allow a "
          "conclusion).\n\nFacts:\n{facts}\n\nQuestion: Is '{ent}' a {cls} (DBpedia class dbo:{cid})?")


def build_prompt(ctx, ent, cls):
    return PROMPT.format(facts="\n".join(ctx) if ctx else "(no facts)", ent=local(ent), cls=local(cls),
                         cid=cls.rsplit("/", 1)[-1])


def parse_answer(text):
    """First standalone YES / NO / UNKNOWN in the reply (tolerates 'Answer: **YES**', etc.)."""
    import re
    m = re.search(r"\b(YES|NO|UNKNOWN)\b", (text or "").upper())
    return m.group(1) if m else "UNPARSED"


class GeminiClient:
    """Minimal REST client for Gemini (urllib only). Two endpoints:
      * Gemini Developer API (keys starting 'AIza'):
          generativelanguage.googleapis.com/v1beta/models/<model>:generateContent
      * Vertex AI express mode (keys starting 'AQ.'):
          aiplatform.googleapis.com/v1/publishers/google/models/<model>:generateContent
    The endpoint is chosen from the key format unless --endpoint is given.
    The key is read from GEMINI_API_KEY and never printed."""

    def __init__(self, model, endpoint="auto"):
        self.key = os.environ["GEMINI_API_KEY"].strip()
        if endpoint == "auto":
            endpoint = "vertex" if self.key.startswith("AQ.") else "developer"
        self.endpoint = endpoint
        name = model.split("/", 1)[1] if model.startswith("models/") else model
        if endpoint == "vertex":
            self.url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{name}:generateContent"
        else:
            self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent"

    def _req(self, url, body=None):
        import json, urllib.error, urllib.request
        from urllib.parse import quote
        data = None if body is None else json.dumps(body).encode()
        full = url + ("&" if "?" in url else "?") + "key=" + quote(self.key, safe="")
        req = urllib.request.Request(full, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace").replace(self.key, "<key>")
            raise RuntimeError(f"HTTP {e.code} from {self.endpoint} endpoint: {msg[:800]}") from None

    def list_models(self):
        if self.endpoint == "vertex":
            return ["(model listing is not available in Vertex AI express mode; try gemini-2.5-flash, "
                    "gemini-2.5-flash-lite or gemini-2.5-pro)"]
        d = self._req("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200")
        return [m["name"] for m in d.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]

    def answer(self, prompt):
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 1024,
                                     "thinkingConfig": {"thinkingBudget": 0}}}
        try:
            d = self._req(self.url, body)
        except RuntimeError as e:                    # model without thinkingConfig support
            if "HTTP 400" not in str(e) or "thinking" not in str(e).lower():
                raise
            body["generationConfig"].pop("thinkingConfig", None)
            d = self._req(self.url, body)
        parts = d.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    def check(self):
        """One test call; raises with Google's error message if the key/model is wrong."""
        return self.answer("Reply with the single word YES.")


class AnthropicClient:
    def __init__(self, model):
        import anthropic
        self.c = anthropic.Anthropic()               # reads ANTHROPIC_API_KEY
        self.model = model

    def answer(self, prompt):
        r = self.c.messages.create(model=self.model, max_tokens=5, temperature=0,
                                   messages=[{"role": "user", "content": prompt}])
        return r.content[0].text


def ask(client, ctx, ent, cls, retries=4):
    prompt = build_prompt(ctx, ent, cls)
    for i in range(retries):
        try:
            raw = client.answer(prompt)
            return parse_answer(raw), (raw or "").strip().replace("\n", " ")[:200]
        except Exception as e:                       # rate limits, transient errors
            err = e
            time.sleep(3 * (i + 1))
    raise err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--endpoint", choices=["auto", "developer", "vertex"], default="auto",
                    help="Gemini endpoint; auto = vertex for 'AQ.' keys, developer otherwise")
    ap.add_argument("--check", action="store_true", help="one test call, then exit")
    ap.add_argument("--sleep", type=float, default=0.0, help="pause between calls (free-tier rate limits)")
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--n-pos", type=int, default=300)
    ap.add_argument("--n-neg", type=int, default=300)
    ap.add_argument("--max-context", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    client = GeminiClient(a.model, a.endpoint) if a.provider == "gemini" else AnthropicClient(a.model)
    if a.list_models:
        print("\n".join(client.list_models()))
        return
    if a.check:
        print("endpoint:", getattr(client, "endpoint", a.provider), "| model:", a.model,
              "| test answer:", repr(client.check()))
        return
    print("endpoint:", getattr(client, "endpoint", a.provider), "| test answer:", repr(client.check()), flush=True)

    T, A, _ = load_kb(os.path.join(EXT, "dbpedia_ontology.owl"), os.path.join(EXT, "dbpedia_joie_sample.nt"),
                      abox_format="nt")
    inst = build_instance(T, A, m=8)
    B = int(a.frac * len(A))
    sols = {"SPARE": max([justification_greedy(inst, B), triple_greedy(inst, B)],
                         key=lambda S: len(closure_of(T, S))),
            "Degree": bl.degree_baseline(inst, B), "JustFreq": bl.justfreq_baseline(inst, B),
            "Random": bl.random_baseline(inst, B, seed=a.seed)}
    qs = make_questions(inst.closure, A, a.n_pos, a.n_neg, a.seed)
    tag = (a.provider + "_" + a.model).replace("/", "_")
    part = os.path.join(RES, f"e9_kgrag_{tag}_partial_v2.csv")   # v2: answer parsing fixed
    cols = ["method", "entity", "cls", "positive", "entailed_by_S", "answer", "n_context", "raw"]
    done = set()
    if os.path.exists(part):                      # resume an interrupted run
        prev = pd.read_csv(part)
        done = set(zip(prev.method, prev.entity, prev.cls))
        print(f"resuming: {len(done)} answers already in {os.path.basename(part)}", flush=True)
    else:
        pd.DataFrame(columns=cols).to_csv(part, index=False)
    total = len(sols) * len(qs)
    n_done, t0, n_new = len(done), time.time(), 0
    tally = defaultdict(int)
    print(f"{total} questions to answer in total ({len(qs)} per method)", flush=True)
    for m, S in sols.items():
        cl = closure_of(T, S)
        k = 0
        for ent, cls, is_pos in qs:
            k += 1
            if (m, ent, cls) in done:
                continue
            ctx = context(S, ent, a.max_context)
            ans, raw = ask(client, ctx, ent, cls)
            if a.sleep:
                time.sleep(a.sleep)
            pd.DataFrame([dict(method=m, entity=ent, cls=cls, positive=is_pos,
                               entailed_by_S=(ent, TYPE, cls) in cl, answer=ans,
                               n_context=len(ctx), raw=raw)])[cols].to_csv(part, mode="a", header=False, index=False)
            n_done += 1
            n_new += 1
            tally[ans] += 1
            if k % 25 == 0:
                eta = (time.time() - t0) / n_new * (total - n_done) / 60
                print(f"{m}: {k}/{len(qs)} | overall {n_done}/{total} | about {eta:.0f} min left | answers so far {dict(tally)}", flush=True)
        print(m, "done", flush=True)
    df = pd.read_csv(part)
    df = df.drop_duplicates(["method", "entity", "cls"])
    assert len(df) == len(sols) * len(qs), f"incomplete: {len(df)} of {len(sols) * len(qs)} answers"
    df.to_csv(os.path.join(RES, f"e9_kgrag_{tag}.csv"), index=False)
    out = []
    for m, g in df.groupby("method"):
        p, n = g[g.positive], g[~g.positive]
        ps, pu = p[p.entailed_by_S], p[~p.entailed_by_S]
        out.append(dict(method=m, budget_frac=a.frac,
                        accuracy_full=((p.answer == "YES").sum() + (n.answer != "YES").sum()) / len(g),
                        supported_yes_rate=(ps.answer == "YES").mean() if len(ps) else float("nan"),
                        unsupported_yes=(pu.answer == "YES").mean() if len(pu) else float("nan"),
                        false_yes=(n.answer == "YES").mean(), entailed_share=p.entailed_by_S.mean(),
                        unknown_rate=(g.answer == "UNKNOWN").mean(), unparsed_rate=(g.answer == "UNPARSED").mean(),
                        context_tokens=g.n_context.mean()))
    pd.DataFrame(out).to_csv(os.path.join(RES, f"e9_kgrag_{tag}_summary.csv"), index=False)
    print(pd.DataFrame(out).round(3).to_string())
    print("all methods done:", len(df), "answers", flush=True)


if __name__ == "__main__":
    main()
