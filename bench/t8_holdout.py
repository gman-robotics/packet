"""T8 (GMA-256): whole-packet holdout boundary model, ship bar = decision 2C.

Holds out whole dlnet-whole packets, trains a small logistic model on the other
packets (joins intact, real pages only: decision 10A, no synthetic copies), and
scores model and heuristic with T5's ``LookaheadSplitter`` on the same joins.
The model only adds a signal where the detector's is weaker, capped at 0.74
(decision 3A: no immediate commit without a page-number reset).

    python bench/t8_holdout.py [--cache DIR] [--out DIR] [--note bench/t8_holdout.md]

Writes the note (train/scored ids, residual-shape count with evidence, heuristic
then model holdout macro/micro, per-packet rows, ``ship``). Optionally exports a
JSON weights file (``--export-weights``). ``Pipeline()`` is unchanged unless a
scorer is configured on ``PipelineConfig``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from packet.boundary_model import (  # noqa: E402
    FEATURE_NAMES as FEATURES,
    Weights,
    combined_confidence,
    export_weights_json,
    featurize_from_facts,
    predict_probability,
    train_logistic,
)

PROPOSE = 0.4  # LookaheadSplitter's propose level: below it a score is no signal
CAP = 0.74  # decision 3A: a model score never reaches the 0.75 immediate commit


def predict(w: Weights, x: list[float]) -> float:
    return predict_probability(w, x)


# ---- split (2C) ----

@dataclass
class Split:
    scored: list[str]
    train: list[str]
    miss: list[str]
    saturated: list[str]


def split_packets(f1: dict[str, float]) -> Split:
    """Residual miss = every packet under 1.0 on the T5 head; scored = misses + the two
    lowest-id packets at 1.0; train = every other packet, whole."""
    names = sorted(f1)
    miss = [n for n in names if f1[n] < 1.0]
    saturated = [n for n in names if f1[n] >= 1.0][:2]
    scored = sorted(miss + saturated)
    train = [n for n in names if n not in scored]
    return Split(scored, train, miss, saturated)


def check_doc_disjoint(split: Split, docs_by_packet: dict[str, list[str]]) -> None:
    scored = {d for p in split.scored for d in docs_by_packet.get(p, [])}
    train = {d for p in split.train for d in docs_by_packet.get(p, [])}
    shared = sorted(scored & train)
    if shared:
        raise ValueError(f"docs in both scored and train packets: {shared}")


def residual_positives(rows: list[dict], train: list[str]) -> list[tuple[str, int]]:
    """Training gold cuts of the residual shape: no page number and no boundary signal."""
    return [(r["packet"], r["page"]) for r in rows
            if r["packet"] in train and r["gold_cut"] and r["page_number"] is None and r["signal"] is None]


def assert_real_only(rows: list[dict]) -> None:
    bad = [(r["packet"], r["page"]) for r in rows if r.get("synthetic")]
    if bad:
        raise ValueError(f"synthetic rows in training (decision 10A forbids them): {bad}")


# ---- ship rule (2C) ----

def _micro_f1(t: tuple[int, int, int]) -> float:
    matched, pred, gold = t
    return 1.0 if pred + gold == 0 else 2 * matched / (pred + gold)


def ship_decision(heur: dict[str, float], model: dict[str, float], *, saturated: list[str], miss: list[str],
                  residual_positives: int, heur_micro: tuple[int, int, int], model_micro: tuple[int, int, int],
                  decision10: str = "A") -> dict:
    scored = sorted(saturated + miss)
    macro_h = sum(heur[p] for p in scored) / len(scored)
    macro_m = sum(model[p] for p in scored) / len(scored)
    micro_h, micro_m = _micro_f1(heur_micro), _micro_f1(model_micro)
    checks = {
        "saturated_unchanged": all(model[p] == heur[p] for p in saturated),
        "every_miss_strictly_up": bool(miss) and all(model[p] > heur[p] for p in miss),
        "macro_strictly_up": macro_m > macro_h,
        "micro_strictly_up": micro_m > micro_h,
        "residual_positives_or_10B": residual_positives > 0 or decision10 == "B",
    }
    return {"ship": all(checks.values()), "checks": checks, "macro": (macro_h, macro_m), "micro": (micro_h, micro_m)}


# ---- data: per-page facts from the real pipeline pieces ----

def page_facts(pdf: Path) -> list[dict]:
    from packet.boundary import BoundaryDetector, _size_jump, _title_position
    from packet.classify import HeuristicClassifier
    from packet.extract import NativeTextExtractor, Router
    from packet.furniture import FurnitureTagger
    from packet.pipeline import _lazy_lines, _lazy_render
    from packet.source import PdfPageSource
    from packet.topic import content_words, detect_language

    det, cl, router, tg = BoundaryDetector(), HeuristicClassifier(), Router([NativeTextExtractor()]), FurnitureTagger()
    out: list[dict] = []
    with PdfPageSource(pdf) as src:
        for ref in src.pages():
            text = src.text_for(ref.index)
            c = cl.classify(ref, text)
            blocks = tg.tag_page(c, router.extract(c, text, image=_lazy_render(src, c.index),
                                                   layout=_lazy_lines(src, c.index)))
            sig = det.observe(c, blocks)
            body = "\n".join(b.text for b in blocks if b.text)
            out.append({
                "page": ref.index, "blocks": len(blocks), "signal": sig,
                "page_number": det.last_page_number, "page_total": det.last_page_total,
                "title": _title_position(blocks), "text": body, "words": content_words(blocks),
                "size": (ref.width, ref.height, ref.rotation), "lang": detect_language(body),
                "_size_jump": _size_jump,
            })
    return out


featurize = featurize_from_facts


def split_boundaries(facts: list[dict], scores: list[float] | None) -> list[int]:
    """Run T5's splitter (Pipeline defaults) over the pages; return predicted boundaries
    (page index after which a document ends), exactly as ``Pipeline._apply`` closes them."""
    from packet.boundary import BoundarySignal
    from packet.lookahead import LookaheadSplitter, PageObs
    from packet.pipeline import PipelineConfig

    cfg = PipelineConfig()
    sp = LookaheadSplitter(window=cfg.lookahead_window, commit=cfg.boundary_commit_confidence,
                           fallback=cfg.boundary_min_confidence)
    decisions = []
    for i, f in enumerate(facts):
        sig = f["signal"]
        if scores is not None and i > 0:
            conf = combined_confidence(sig.confidence if sig else None, scores[i])
            if conf is not None and (sig is None or conf != sig.confidence):
                sig = BoundarySignal(after_page=i - 1, confidence=conf, reason=(sig.reason + "+model") if sig else "model")
        decisions += sp.push(PageObs(i, sig, f["page_number"], f["page_total"]))
    decisions += sp.flush()
    out, has_blocks = [], False
    for d in sorted(decisions, key=lambda d: d.page):
        if d.split and has_blocks:
            out.append(d.page - 1)
            has_blocks = False
        has_blocks = has_blocks or facts[d.page]["blocks"] > 0
    return out


def score(gold: list[int], pred: list[int]) -> tuple[float, tuple[int, int, int]]:
    matched = len(set(gold) & set(pred))
    t = (matched, len(pred), len(gold))
    return _micro_f1(t), t


# ---- the run ----

def run(cache: Path, out: Path, export_weights: Path | None = None) -> dict:
    from packet.eval.metrics import document_boundaries
    from packet.eval.run import run_dlnet_whole
    from packet.eval.wholedoc import build_fixtures, fetch, load_manifest

    pipeline_rows = {r.name: r for r in run_dlnet_whole(cache, root=out)}  # the T5 head, as `packet eval` scores it
    f1 = {n: r.boundary.f1 for n, r in pipeline_rows.items()}
    s = split_packets(f1)
    m = load_manifest()
    fetch(m, cache=cache)
    fixtures = {fx.name: fx for fx in build_fixtures(m, cache, out / "t8")}
    check_doc_disjoint(s, {n: fx.notes.split("+") for n, fx in fixtures.items()})

    facts = {n: page_facts(fx.pdf) for n, fx in fixtures.items()}
    gold = {n: document_boundaries(fx.gold.documents) for n, fx in fixtures.items()}
    rows = [{"packet": n, "page": f["page"], "gold_cut": f["page"] - 1 in gold[n] and f["page"] > 0,
             "page_number": f["page_number"], "signal": f["signal"], "synthetic": False}
            for n in fixtures for f in facts[n]]
    assert_real_only(rows)
    positives = residual_positives(rows, s.train)

    X, y = [], []
    for n in s.train:
        for i in range(1, len(facts[n])):
            X.append(featurize(facts[n], i))
            y.append(int(i - 1 in gold[n]))
    w = train_logistic(X, y)
    if export_weights is not None:
        export_weights_json(export_weights, w, training_packets=s.train)

    per: dict[str, dict] = {}
    for n in s.scored:
        fx = facts[n]
        probs = [0.0] + [predict(w, featurize(fx, i)) for i in range(1, len(fx))]
        hb, mb = split_boundaries(fx, None), split_boundaries(fx, probs)
        if hb != pipeline_rows[n].boundary.predicted:
            raise RuntimeError(f"{n}: replayed heuristic {hb} != packet eval {pipeline_rows[n].boundary.predicted}")
        hf, ht = score(gold[n], hb)
        mf, mt = score(gold[n], mb)
        per[n] = {"gold": gold[n], "heur_pred": hb, "model_pred": mb, "heur_f1": hf, "model_f1": mf,
                  "heur_t": ht, "model_t": mt,
                  "residual_pages": {str(g + 1): round(probs[g + 1], 3) for g in gold[n]},
                  "noncut_max_p": round(max((probs[i] for i in range(1, len(fx)) if i - 1 not in gold[n]), default=0.0), 3),
                  "noncut_ge_055": sum(1 for i in range(1, len(fx)) if i - 1 not in gold[n] and probs[i] >= 0.55)}
    sumt = lambda key: tuple(sum(per[n][key][j] for n in s.scored) for j in range(3))
    decision = ship_decision({n: per[n]["heur_f1"] for n in s.scored}, {n: per[n]["model_f1"] for n in s.scored},
                             saturated=s.saturated, miss=s.miss, residual_positives=len(positives),
                             heur_micro=sumt("heur_t"), model_micro=sumt("model_t"))
    in_sample = {n: score(gold[n], split_boundaries(facts[n], [0.0] + [predict(w, featurize(facts[n], i)) for i in range(1, len(facts[n]))]))[0]
                 for n in s.train}
    return {"split": s.__dict__, "dlnet_macro_t5": sum(f1.values()) / len(f1), "f1_t5": f1,
            "residual_positives": positives, "train_pages": len(X), "train_cuts": sum(y),
            "features": FEATURES, "coef": dict(zip(FEATURES, [round(c, 3) for c in w[2]])),
            "per_packet": per, **decision, "train_in_sample_f1": in_sample}


def write_note(r: dict, path: Path) -> None:
    s = r["split"]
    (mh, mm), (uh, um) = r["macro"], r["micro"]
    L = ["# T8 holdout boundary model (GMA-256)", "",
         ("Generated by `python bench/t8_holdout.py`. Decisions: 2C ship bar, 3A cap 0.74, 10A real data only. "
          "Splitter: T5's `LookaheadSplitter` with `PipelineConfig()` defaults (window 2, commit 0.75, fallback 0.55). "
          "Weights are optional (`--export-weights`); default `Pipeline()` and `packet eval` use the heuristic only."), "",
         f"- T5 head dlnet-whole macro (heuristic, `packet eval` path): {r['dlnet_macro_t5']:.5f}. T2 baseline macro recorded in the plan: 0.9077 (unrounded figure as logged; not 0.908).",
         f"- Residual miss (every packet under F1 1.0 on the T5 head): {', '.join(s['miss']) or 'none'}. dlw-00 is 1.0 after T2+T5, so it is saturated, not a miss.",
         f"- Scored packets: {', '.join(s['scored'])} (saturated: {', '.join(s['saturated'])}).",
         f"- Train packets (whole, joins intact): {', '.join(s['train'])}. Disjoint from scored by packet and by doc (checked).",
         f"- Training pages: {r['train_pages']}, of which gold cuts: {r['train_cuts']}.",
         f"- Training positives of the residual shape (gold cut, page has no page number and no boundary signal): **{len(r['residual_positives'])}**: "
         + ", ".join(f"{p} p{g}" for p, g in r["residual_positives"]) + ". Real pages only; no synthetic copies (10A).", "",
         "## Holdout (heuristic first, then model; same splitter, same joins, no rows dropped)", "",
         "| | macro F1 | micro F1 |", "|---|---|---|",
         f"| heuristic | {mh:.4f} | {uh:.4f} |", f"| model | {mm:.4f} | {um:.4f} |", "",
         "| packet | gold | heuristic pred | heuristic F1 | model pred | model F1 | model p at gold-cut pages | max p, non-cut pages | non-cut pages p >= 0.55 |", "|---|---|---|---|---|---|---|---|---|"]
    for n, p in r["per_packet"].items():
        L.append(f"| {n} | {p['gold']} | {p['heur_pred']} | {p['heur_f1']:.3f} | {p['model_pred']} | {p['model_f1']:.3f} | {p['residual_pages']} | {p['noncut_max_p']} | {p['noncut_ge_055']} |")
    L += ["", "## Ship (2C)", "", *(f"- {k}: {v}" for k, v in r["checks"].items()), "", f"**ship = {str(r['ship']).lower()}**", "",
          "A 0.45 score is not a split of dlw-12 page 25 (it needs >= 0.55 for `fallback_eager`); the model is capped at 0.74, so it can never immediate-commit.", "",
          "Caveat: three scored packets (5 gold cuts) is a small holdout; the per-page margin columns show how close a non-cut page came to a split.", "",
          "## Model", "", "Class-balanced L2 logistic regression, pure Python, deterministic. Features (no page index, file name or heuristic confidence): "
          + ", ".join(r["features"]) + ".", "", "Coefficients (standardized): " + json.dumps(r["coef"]), "",
          "In-sample train F1 (not the ship rule): " + json.dumps({k: round(v, 3) for k, v in r["train_in_sample_f1"].items()}), ""]
    path.write_text("\n".join(L))


def main(argv: list[str] | None = None) -> int:
    from packet.eval.run import DEFAULT_WHOLEDOC_CACHE

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_WHOLEDOC_CACHE)
    ap.add_argument("--out", type=Path, default=Path(".packet-eval"))
    ap.add_argument("--note", type=Path, default=REPO / "bench" / "t8_holdout.md")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--export-weights", type=Path, help="write JSON logistic weights for T9")
    a = ap.parse_args(argv)
    r = run(a.cache, a.out, export_weights=a.export_weights)
    (mh, mm), (uh, um) = r["macro"], r["micro"]
    print(f"heuristic holdout macro {mh:.4f} micro {uh:.4f}")
    print(f"model     holdout macro {mm:.4f} micro {um:.4f}")
    print(f"residual-shape training positives {len(r['residual_positives'])}; ship={str(r['ship']).lower()}")
    write_note(r, a.note)
    if a.json:
        a.json.write_text(json.dumps(r, default=str, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
