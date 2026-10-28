"""GMA-448 Exp B2: Hunch-pattern teacher-student boundary signal (teacher = PoliTax gold labels).

The student is EmbeddingGemma 2 page-head embeddings + logistic regression. It enters Packet through the existing
``decision_backend`` hook (bench-only adapter, no src/ change), so its probability goes through
``combined_confidence`` (floor 0.40, cap 0.74) and the 0.74 no-reset cap, then the real LookaheadSplitter.

  b2.py labels                 label table + split checks -> B2/labels.json
  b2.py dump                   record per-page text for every (set, backend) -> B2/pages.jsonl
  (embed with embed_cpu.py in the sandbox venv -> B2/emb.npy, B2/emb_keys.json, B2/latency.json)
  b2.py ceiling                oracle student (0.74 on gold starts) on train + holdout -> B2/ceiling.json
  b2.py train                  grouped 5-fold CV on politax-train only; threshold rule -> B2/student.json
  b2.py ablate                 frozen student on every arm x backend + regressions -> B2/ablation.json
Holdout is never used by `train`. No Packet defaults change: the backend is None unless this harness passes one."""
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from packet import boundary as B
from packet.decision import DecisionResult
from packet.eval.run import DEFAULT_POLITAX_CACHE, DEFAULT_WHOLEDOC_CACHE, run_fixture
from packet.pipeline import Pipeline, PipelineConfig

B2 = Path(os.environ.get("B2", Path.home() / ".cache/packet/model-eval/b2"))
P1 = Path.home() / ".cache/packet/model-eval/p1"
BACKENDS = {  # holdout OCR caches per backend; LightOn is the only one with train pages
    "lighton": DEFAULT_POLITAX_CACHE,
    "lighton-untagged": P1 / "politax-lighton-untagged",
    "lfm25x": P1 / "politax-lfm25x",
    "minicpm1": B2 / "politax-minicpm1",
}
MAX_CHARS = 1500  # page head: boundaries show at the top of a page; keeps CPU embedding short
CAP = 0.74


def page_text(blocks):
    """All block text, no kind filter: backend-agnostic (untagged OCR has no furniture kinds)."""
    return " ".join(b.text.strip() for b in blocks if b.text and b.text.strip())[:MAX_CHARS]


def key(t):
    return hashlib.sha1(t.encode()).hexdigest()


class Capabilities:
    images = False


class StudentBackend:
    """Bench-only DecisionBackend. mode: record | oracle | student. Called once per page, in order."""
    capabilities = Capabilities()
    send_images = False

    def __init__(self, mode, probs=None, threshold=None, gold=None):
        self.mode, self.probs, self.t, self.gold = mode, probs or {}, threshold, gold or {}
        self.fixture, self.seen = None, []

    def decide(self, req, render=None):
        t = page_text(req.blocks)
        if self.mode == "record":
            self.seen.append((self.fixture, req.page_index, t)); return DecisionResult(probability=None)
        if self.mode == "oracle":
            return DecisionResult(probability=CAP if req.page_index in self.gold.get(self.fixture, ()) else 0.0)
        p = self.probs.get((self.fixture, req.page_index))
        if p is None:
            return DecisionResult(probability=None)
        return DecisionResult(probability=CAP if p >= self.t else 0.0)


@contextmanager
def heuristics_off():
    """Student-alone arm: zero every heuristic weight (look-ahead page-number veto and the 0.74 cap stay)."""
    names = [n for n in dir(B) if n.startswith("W_")] + ["COMBINED"]
    old = {n: getattr(B, n) for n in names}
    try:
        for n in names: setattr(B, n, 0.0)
        yield
    finally:
        for n, v in old.items(): setattr(B, n, v)


def fixtures(set_name, backend="lighton"):
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "gma448"
    if set_name == "builtin":
        from packet.eval.fixtures import build_builtin
        return build_builtin(tmp / "builtin"), lambda fx, cfg: run_fixture(fx, Pipeline(config=cfg))
    if set_name == "dlnet-whole":
        from packet.eval import wholedoc
        m = wholedoc.load_manifest()
        return wholedoc.build_fixtures(m, DEFAULT_WHOLEDOC_CACHE, tmp / "dlnet"), lambda fx, cfg: run_fixture(fx, Pipeline(config=cfg))
    from packet.eval.politax_ocr_eval import CacheStats, ocr_cache_pipeline
    from packet.eval.politax_split import holdout_file_names
    from packet.eval.run import prepare_politax
    data = BACKENDS[backend]
    m, fxs = prepare_politax(data); hold = holdout_file_names(m)
    want = (lambda f: f in hold) if set_name == "politax-holdout" else (lambda f: f not in hold)
    sha = {d.file_name: d.sha256 for d in m.docs}; stats = CacheStats()
    fx = [f for f in fxs if want(f.notes or f.pdf.name)]
    only = os.environ.get("ONLY_FILES")  # MiniCPM subset (whole docs); SUBSET_ALL=1 applies it to every backend
    if only and set_name == "politax-holdout" and (backend == "minicpm1" or os.environ.get("SUBSET_ALL")):
        keep = set(only.split(",")); fx = [f for f in fx if (f.notes or f.pdf.name) in keep]
    return fx, lambda f, cfg: run_fixture(f, ocr_cache_pipeline(data, sha[f.notes or f.pdf.name], stats, cfg))


def gold_starts(fx):
    return {d.page_start for d in fx.gold.documents if d.page_start > 0}


def score(rs, wall):
    fp = sum(len(set(r.boundary.predicted) - set(r.boundary.gold)) for r in rs)
    fn = sum(len(set(r.boundary.gold) - set(r.boundary.predicted)) for r in rs)
    tp = sum(len(set(r.boundary.predicted) & set(r.boundary.gold)) for r in rs)
    return {"macro_f1": round(sum(r.boundary.f1 for r in rs) / len(rs), 4), "false_splits": fp, "missed": fn, "tp": tp,
            "micro_precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "micro_recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "false_ocr_pages": sum(len(r.false_ocr_pages) for r in rs), "fixtures": len(rs), "wall_s": round(wall, 1),
            "per_fixture": [(r.name, round(r.boundary.f1, 3)) for r in rs]}


def run(set_name, backend, be, alone=False):
    fxs, runner = fixtures(set_name, backend); out = []; t0 = time.perf_counter()
    cfg = PipelineConfig(decision_backend=be) if be is not None else PipelineConfig()
    ctx = heuristics_off() if alone else _null()
    with ctx:
        for fx in fxs:
            if be is not None:
                be.fixture = fx.name
                if be.mode == "oracle": be.gold[fx.name] = gold_starts(fx)
            out.append(runner(fx, cfg))
    return score(out, time.perf_counter() - t0)


@contextmanager
def _null():
    yield


SETS = [("politax-train", "lighton")] + [("politax-holdout", b) for b in BACKENDS if b != "minicpm1" or os.environ.get("WITH_MINICPM")] + [("dlnet-whole", "lighton"), ("builtin", "lighton")]


def labels():
    out = {}
    for s, b in [("politax-train", "lighton"), ("politax-holdout", "lighton")]:
        fxs, _ = fixtures(s, b)
        out[s] = {"docs": len(fxs), "pages": sum(f.gold.page_count if hasattr(f.gold, "page_count") else
                                                 max(d.page_end for d in f.gold.documents) + 1 for f in fxs),
                  "boundaries": sum(len(gold_starts(f)) for f in fxs), "files": sorted(f.notes or f.pdf.name for f in fxs)}
    overlap = set(out["politax-train"]["files"]) & set(out["politax-holdout"]["files"])
    assert not overlap, overlap
    out["overlap"] = []
    (B2 / "labels.json").write_text(json.dumps(out, indent=1)); return {k: {kk: vv for kk, vv in v.items() if kk != "files"} if isinstance(v, dict) else v for k, v in out.items()}


def dump():
    """DUMP_BACKENDS=a,b limits the run and appends to an existing pages.jsonl (replacing those backends' rows)."""
    only = set(filter(None, os.environ.get("DUMP_BACKENDS", "").split(",")))
    path = B2 / "pages.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if only and path.exists() else []
    rows = [r for r in rows if r["backend"] not in only or r["set"] != "politax-holdout"]
    for s, b in SETS:
        if only and not (s == "politax-holdout" and b in only): continue
        be = StudentBackend("record"); fxs, runner = fixtures(s, b)
        for fx in fxs:
            be.fixture = fx.name; runner(fx, PipelineConfig(decision_backend=be))
        gold = {fx.name: gold_starts(fx) for fx in fxs}
        for fxn, idx, t in be.seen:
            rows.append({"set": s, "backend": b, "fixture": fxn, "page": idx, "y": int(idx in gold[fxn]), "key": key(t), "text": t})
        print(s, b, len(fxs), "fixtures", len(be.seen), "pages", flush=True)
    (B2 / "pages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def ceiling():
    out = {}
    for s in ["politax-train", "politax-holdout"]:
        out[s] = {"heuristic": run(s, "lighton", None),
                  "oracle_alone": run(s, "lighton", StudentBackend("oracle"), alone=True),
                  "oracle+heuristic": run(s, "lighton", StudentBackend("oracle"))}
        print(s, {k: (v["macro_f1"], v["false_splits"], v["missed"]) for k, v in out[s].items()}, flush=True)
    (B2 / "ceiling.json").write_text(json.dumps(out, indent=1)); return out


def _feats(rows, E, ix):
    """Per page: [e_i, e_i - e_{i-1}, cos(e_i, e_{i-1}), cos(e_i, mean(e_{i-1}, e_{i-2}))] within one fixture."""
    import numpy as np
    X = []; d = E.shape[1]
    for i, r in enumerate(rows):
        e = E[ix[r["key"]]] if r["key"] in ix else np.zeros(d)
        p1 = E[ix[rows[i - 1]["key"]]] if i > 0 and rows[i - 1]["key"] in ix else np.zeros(d)
        p2 = E[ix[rows[i - 2]["key"]]] if i > 1 and rows[i - 2]["key"] in ix else p1
        m = (p1 + p2) / 2; mn = np.linalg.norm(m)
        X.append(np.concatenate([e, e - p1, [float(e @ p1), float(e @ m) / mn if mn else 0.0, float(r["page"] == 0)]]))
    return X


def load_pages():
    import numpy as np
    rows = [json.loads(l) for l in (B2 / "pages.jsonl").read_text().splitlines()]
    keys = json.loads((B2 / "emb_keys.json").read_text()); E = np.load(B2 / "emb.npy"); ix = {k: i for i, k in enumerate(keys)}
    by = {}
    for r in rows: by.setdefault((r["set"], r["backend"], r["fixture"]), []).append(r)
    data = {}
    for k, rs in by.items():
        rs.sort(key=lambda r: r["page"]); data[k] = (rs, _feats(rs, E, ix))
    return data


def _fit(X, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(X, y)


GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]


def train():
    import numpy as np
    data = load_pages()
    tr = {k: v for k, v in data.items() if k[0] == "politax-train"}
    fx = sorted({k[2] for k in tr}); folds = [fx[i::5] for i in range(5)]; oof = {}
    for f in folds:
        X = [x for k, (rs, xs) in tr.items() if k[2] not in f for x in xs]
        y = [r["y"] for k, (rs, xs) in tr.items() if k[2] not in f for r in rs]
        clf = _fit(np.array(X), np.array(y))
        for k, (rs, xs) in tr.items():
            if k[2] in f:
                for r, p in zip(rs, clf.predict_proba(np.array(xs))[:, 1]): oof[(k[2], r["page"])] = float(p)
    from sklearn.metrics import roc_auc_score
    ys = [r["y"] for k, (rs, _) in tr.items() for r in rs]; ps = [oof[(k[2], r["page"])] for k, (rs, _) in tr.items() for r in rs]
    base = run("politax-train", "lighton", None); grid = []
    for t in GRID:
        a = run("politax-train", "lighton", StudentBackend("student", oof, t), alone=True)
        c = run("politax-train", "lighton", StudentBackend("student", oof, t))
        grid.append({"t": t, "alone": {k: a[k] for k in ("macro_f1", "false_splits", "missed")},
                     "plus_heur": {k: c[k] for k in ("macro_f1", "false_splits", "missed")}})
        print(grid[-1], flush=True)

    def pick(arm):
        ok = [g for g in grid if g[arm]["false_splits"] <= base["false_splits"]]
        return max(ok, key=lambda g: (g[arm]["macro_f1"], -g[arm]["false_splits"]))["t"] if ok else None
    out = {"split": "politax-train only, 5 folds grouped by fixture", "oof_auroc": round(roc_auc_score(ys, ps), 4),
           "heuristic_train": {k: base[k] for k in ("macro_f1", "false_splits", "missed")},
           "rule": "max train macro F1 with false_splits <= heuristic train", "grid": grid,
           "t_alone": pick("alone"), "t_plus_heur": pick("plus_heur")}
    X = np.array([x for k, (rs, xs) in tr.items() for x in xs]); y = np.array(ys)
    import pickle
    clf = _fit(X, y)
    (B2 / "student.pkl").write_bytes(pickle.dumps(clf)); (B2 / "student.json").write_text(json.dumps(out, indent=1))
    return {k: v for k, v in out.items() if k != "grid"}


def ablate():
    import pickle

    import numpy as np
    st = json.loads((B2 / "student.json").read_text()); clf = pickle.loads((B2 / "student.pkl").read_bytes())
    data = load_pages(); probs = {}
    for k, (rs, xs) in data.items():
        for r, p in zip(rs, clf.predict_proba(np.array(xs))[:, 1]): probs[(*k, r["page"])] = float(p)
    out = []
    for s, b in SETS[1:]:
        if not any(k[0] == s and k[1] == b for k in data): continue
        pb = {(k[2], r["page"]): probs[(*k, r["page"])] for k, (rs, _) in data.items() if k[0] == s and k[1] == b for r in rs}
        arms = {"heuristic": run(s, b, None)}
        if st["t_alone"] is not None: arms["student_alone"] = run(s, b, StudentBackend("student", pb, st["t_alone"]), alone=True)
        if st["t_plus_heur"] is not None: arms["student+heuristic"] = run(s, b, StudentBackend("student", pb, st["t_plus_heur"]))
        # Diagnostic only, never selected: the unconstrained train-best threshold (student.json grid), to show what the
        # false-split rule costs. It is reported beside the frozen arms, not used for any decision.
        dt = max(st["grid"], key=lambda g: g["alone"]["macro_f1"])["t"]
        arms[f"diag_alone_t{dt}"] = run(s, b, StudentBackend("student", pb, dt), alone=True)
        arms[f"diag_plus_heur_t{dt}"] = run(s, b, StudentBackend("student", pb, dt))
        for a, r in arms.items():
            out.append({"set": s, "backend": b, "arm": a, **{k: v for k, v in r.items() if k != "per_fixture"}, "per_fixture": r["per_fixture"]})
            print(s, b, a, r["macro_f1"], r["micro_precision"], r["micro_recall"], r["false_splits"], r["false_ocr_pages"], flush=True)
    (B2 / "ablation.json").write_text(json.dumps(out, indent=1)); return len(out)


if __name__ == "__main__":
    B2.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if sys.argv[1:] else ""
    f = {"labels": labels, "dump": dump, "ceiling": ceiling, "train": train, "ablate": ablate}.get(cmd)
    if f is None: sys.exit(__doc__)
    r = f(); print(json.dumps(r, default=str)[:3000])
