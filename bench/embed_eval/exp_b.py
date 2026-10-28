"""GMA-443 Exp B: google/embeddinggemma-2 page embeddings for retrieval and a low-weight boundary signal.

  exp_b.py dump                      run every set once with a recording embedder; write P3/pages.jsonl (page text per set)
  (embed with embed_pages.py in the sandbox venv -> P3/emb_{doc,query}.npy + P3/emb_keys.json)
  exp_b.py retrieval                 page->same-document retrieval (NDCG@5, MRR, R@10) per set; lexical baseline beside it
  exp_b.py boundary SET T|off        boundary eval with embed_shift at cosine threshold T (off = v0 detector)
  exp_b.py tune                      grid T on politax-train only; writes P3/tune.json
Sets: builtin, dlnet-whole, politax-train (trump PDFs), politax-holdout (obama/harris/biden; reporting only).
PoliTax scan pages come from the LightOn OCR cache (--ocr-cache path), so the false-OCR hard-fail gate stays on (run_fixture).
No Packet default changes: embedder is None unless this harness passes one."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from packet.boundary import _FURNITURE  # noqa: E402
from packet.eval.run import (DEFAULT_POLITAX_CACHE, DEFAULT_WHOLEDOC_CACHE, OCR_CACHE_FLOORS, evaluate_gate,  # noqa: E402
                             run_fixture)
from packet.pipeline import Pipeline, PipelineConfig  # noqa: E402

P3 = Path(os.environ.get("P3", Path.home() / ".cache/packet/model-eval/p3"))
SETS = ["builtin", "dlnet-whole", "politax-train", "politax-holdout"]
MAX_CHARS = 6000


def page_text(blocks):
    return " ".join(b.text for b in blocks if b.kind not in _FURNITURE and b.text).strip()[:MAX_CHARS]


def key(text):
    return hashlib.sha1(text.encode()).hexdigest()


class Embeddings:
    """Lookup of precomputed unit vectors by page-text key; records every page seen (fixture, index, text)."""

    def __init__(self, load=True):
        self.vec, self.seen, self.fixture = {}, [], None
        if load and (P3 / "emb_keys.json").exists():
            import numpy as np
            keys = json.loads((P3 / "emb_keys.json").read_text()); arr = np.load(P3 / "emb_doc.npy")
            self.vec = {k: arr[i].tolist() for i, k in enumerate(keys)}

    def __call__(self, ref, blocks):
        t = page_text(blocks); self.seen.append((self.fixture, ref.index, t))
        return self.vec.get(key(t)) if t else None


def fixtures(name):
    """(fixtures, runner) for a set; runner(fixture, config) -> EvalResult."""
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "gma443"
    if name == "builtin":
        from packet.eval.fixtures import build_builtin
        return build_builtin(tmp / "builtin"), lambda fx, cfg: run_fixture(fx, Pipeline(config=cfg))
    if name == "dlnet-whole":
        from packet.eval import wholedoc
        m = wholedoc.load_manifest()
        return wholedoc.build_fixtures(m, DEFAULT_WHOLEDOC_CACHE, tmp / "dlnet"), lambda fx, cfg: run_fixture(fx, Pipeline(config=cfg))
    from packet.eval.politax_ocr_eval import CacheStats, ocr_cache_pipeline
    from packet.eval.politax_split import holdout_file_names
    from packet.eval.run import prepare_politax
    m, fxs = prepare_politax(DEFAULT_POLITAX_CACHE); hold = holdout_file_names(m)
    want = (lambda f: f in hold) if name == "politax-holdout" else (lambda f: f not in hold)
    sha = {d.file_name: d.sha256 for d in m.docs}; stats = CacheStats()
    fx = [f for f in fxs if want(f.notes or f.pdf.name)]
    return fx, lambda f, cfg: run_fixture(f, ocr_cache_pipeline(DEFAULT_POLITAX_CACHE, sha[f.notes or f.pdf.name], stats, cfg))


def run_set(name, emb, t=None):
    fxs, run = fixtures(name); cfg = PipelineConfig(embedder=emb, embed_shift_max=t if t is not None else -2.0)
    out = []
    for fx in fxs:
        emb.fixture = fx.name; out.append((fx, run(fx, cfg)))
    return out


def dump():
    P3.mkdir(parents=True, exist_ok=True); rows = []
    for s in SETS:
        e = Embeddings(load=False); res = run_set(s, e)
        gold = {fx.name: [(d.page_start, d.page_end) for d in fx.gold.documents] for fx, _ in res}
        for fxn, idx, t in e.seen:
            doc = next((i for i, (a, b) in enumerate(gold[fxn]) if a <= idx <= b), None)
            rows.append({"set": s, "fixture": fxn, "page": idx, "doc": doc, "key": key(t) if t else None, "text": t})
        print(s, len(res), "fixtures", sum(1 for r in rows if r["set"] == s), "pages", flush=True)
    (P3 / "pages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _metrics(ranked_rel, n_rel):
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(ranked_rel[:5]) if r)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(5, n_rel)))
    rr = next((1 / (i + 1) for i, r in enumerate(ranked_rel) if r), 0.0)
    return dcg / idcg, rr, sum(ranked_rel[:10]) / n_rel


def retrieval():
    import numpy as np
    from collections import Counter
    from packet.topic import topic_similarity
    rows = [json.loads(l) for l in (P3 / "pages.jsonl").read_text().splitlines()]
    keys = json.loads((P3 / "emb_keys.json").read_text()); ix = {k: i for i, k in enumerate(keys)}
    D, Q = np.load(P3 / "emb_doc.npy"), np.load(P3 / "emb_query.npy"); out = {}
    for s in SETS:
        by = {}
        for r in rows:
            if r["set"] == s and r["key"] and r["doc"] is not None: by.setdefault(r["fixture"], []).append(r)
        m = {"emb": [], "lex": []}
        for fx, pages in by.items():
            words = [Counter(w for w in p["text"].lower().split() if len(w) > 2) for p in pages]
            for qi, q in enumerate(pages):
                cand = [j for j in range(len(pages)) if j != qi]
                n_rel = sum(pages[j]["doc"] == q["doc"] for j in cand)
                if not n_rel: continue
                es = [float(Q[ix[q["key"]]] @ D[ix[pages[j]["key"]]]) for j in cand]
                ls = [topic_similarity(words[qi], words[j]) for j in cand]
                for nm, sc in (("emb", es), ("lex", ls)):
                    order = sorted(range(len(cand)), key=lambda k: -sc[k])
                    m[nm].append(_metrics([pages[cand[k]]["doc"] == q["doc"] for k in order], n_rel))
        out[s] = {"queries": len(m["emb"]), "fixtures": len(by)}
        for nm in m:
            if m[nm]:
                out[s][nm] = {k: round(sum(x[i] for x in m[nm]) / len(m[nm]), 4) for i, k in enumerate(("ndcg@5", "mrr", "r@10"))}
    (P3 / "retrieval.json").write_text(json.dumps(out, indent=1)); return out


def boundary(s, t):
    emb = Embeddings(); t0 = time.perf_counter(); res = run_set(s, emb, None if t == "off" else float(t))
    rs = [r for _, r in res]
    lo, macro = OCR_CACHE_FLOORS["politax"] if s == "politax-holdout" else (0.0, 0.0)
    gate = evaluate_gate(rs, min_f1=lo, min_macro_f1=macro)
    fp = sum(len(set(r.boundary.predicted) - set(r.boundary.gold)) for r in rs)
    fn = sum(len(set(r.boundary.gold) - set(r.boundary.predicted)) for r in rs)
    tp = sum(len(set(r.boundary.predicted) & set(r.boundary.gold)) for r in rs)
    return {"set": s, "t": t, "macro_f1": round(sum(r.boundary.f1 for r in rs) / len(rs), 4), "false_splits": fp, "missed": fn,
            "micro_precision": round(tp / (tp + fp), 4) if tp + fp else None, "micro_recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "false_ocr_pages": sum(len(r.false_ocr_pages) for r in rs), "gate_passed": gate.passed, "gate_failures": gate.failures,
            "fixtures": len(rs), "wall_s": round(time.perf_counter() - t0, 1),
            "per_fixture": [(r.name, round(r.boundary.f1, 3)) for r in rs]}


GRID = ["off", 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def tune():
    rows = [boundary("politax-train", t) for t in GRID]
    base = rows[0]
    ok = [r for r in rows[1:] if r["false_splits"] <= base["false_splits"]]
    best = max(ok, key=lambda r: (r["macro_f1"], -r["false_splits"])) if ok else None
    if best is not None and best["macro_f1"] <= base["macro_f1"]:
        best = None  # no lift without extra false splits: keep the signal off
    # Secondary rule (declared before any holdout run): max train macro F1 with micro precision >= v0's.
    prec_ok = [r for r in rows[1:] if (r["micro_precision"] or 0) >= (base["micro_precision"] or 0)]
    alt = max(prec_ok, key=lambda r: (r["macro_f1"], -r["false_splits"])) if prec_ok else None
    loose = max(rows[1:], key=lambda r: (r["macro_f1"], -r["false_splits"]))  # unconstrained, reported for context
    out = {"split": "politax-train (trump) only", "grid": [{k: r[k] for k in ("t", "macro_f1", "false_splits", "missed", "micro_precision", "micro_recall")} for r in rows],
           "rule": "max train macro F1 with false_splits <= off", "chosen": best and best["t"],
           "lift_over_off": best and round(best["macro_f1"] - base["macro_f1"], 4),
           "unconstrained_best": {k: loose[k] for k in ("t", "macro_f1", "false_splits", "missed")},
           "precision_rule": "max train macro F1 with micro precision >= off", "precision_rule_chosen": alt and alt["t"]}
    (P3 / "tune.json").write_text(json.dumps(out, indent=1)); return out


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "dump": dump()
    elif cmd == "retrieval": print(json.dumps(retrieval(), indent=1))
    elif cmd == "boundary":
        r = boundary(a[0], a[1]); (P3 / f"boundary_{a[0]}_{a[1]}.json").write_text(json.dumps(r, indent=1)); print(json.dumps(r))
    elif cmd == "tune": print(json.dumps(tune(), indent=1))
    else: sys.exit(__doc__)
