"""GMA-445 Exp A2 analysis over jev_score.py outputs (no model; repo .venv).

  jev_a2.py dev                     testset (dev split): labels, AUROC/P/R, threshold tuning -> P1/a2/dev.json
  jev_a2.py drop-testset T_BLOCK    copy lighton-1b-grounding -> runs/lighton-1b-grounding-jevdrop with blocks prob<T removed
  jev_a2.py drop-politax T_BLOCK    candidate data dir P1/politax-lighton-jevdrop: LightOn holdout records with scored blocks prob<T removed
  jev_a2.py holdout T_BLOCK T_PAGE  flagged fractions + lowest-score spot-check on PoliTax holdout (report only; no tuning)
Labels (dev only): block ok = normalized block text is a substring of normalized gold page text (score.py strip/norm);
page ok = page CER <= 0.02 from _scores.json. The PoliTax holdout has no OCR labels: spot-check only."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench/ocr_compare")); sys.path.insert(0, str(Path(__file__).parent))
import score as sc
from jev_score import TAG, regions  # PIL import only; no model load

P1 = Path.home() / ".cache/packet/model-eval/p1"; RUNS = P1 / "runs"; A2 = P1 / "a2"
ENGINES = ["lighton-1b-grounding", "lfm25x", "lfm3-mps", "minicpm-mps"]
PAGE_OK_CER = 0.02


def auroc(pos, neg):
    """P(score of an ok item > score of an error item); ties count half. None when a class is empty."""
    if not pos or not neg: return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def pr_at(rows, t):
    """Error detection: flag = prob < t; positive class = error (ok == False)."""
    tp = sum(1 for p, ok in rows if p < t and not ok); fp = sum(1 for p, ok in rows if p < t and ok)
    fn = sum(1 for p, ok in rows if p >= t and not ok)
    pr = tp / (tp + fp) if tp + fp else None; rc = tp / (tp + fn) if tp + fn else None
    return {"t": t, "flagged": tp + fp, "n": len(rows), "flag_frac": (tp + fp) / len(rows) if rows else None,
            "precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr and rc else 0.0}


def tune(rows):
    """Threshold maximizing error-detection F1 (ties -> fewer flags) on dev rows; grid of observed probs."""
    best = None
    for t in sorted({0.0, *[p + 1e-9 for p, _ in rows]}):
        m = pr_at(rows, t)
        if best is None or (m["f1"], -m["flagged"]) > (best["f1"], -best["flagged"]): best = m
    return best


def dev():
    out = {"engines": {}, "label_rules": {"block": "substring of gold", "page": f"CER<={PAGE_OK_CER}"}}
    page_rows, block_rows = [], []
    for e in ENGINES:
        jv = json.loads((RUNS / e / "_jev.json").read_text())["pages"]
        cer = {r["page"]: r["cer"] for r in json.loads((RUNS / e / "_scores.json").read_text())["pages"] if r.get("cer") is not None}
        pr, br = [], []
        for page, rows in jv.items():
            raw = (RUNS / e / f"{page}.txt").read_text(); blocks = regions(raw); g = sc.gold_for(page)
            gold = sc.norm(sc.strip(g["full_text"])) if g else None
            for r in rows:
                if r["prob"] is None: continue
                if r["kind"] == "page" and page in cer: pr.append((r["prob"], cer[page] <= PAGE_OK_CER, page))
                if r["kind"] == "block" and gold is not None:
                    t = sc.norm(sc.strip(blocks[r["i"]][2]))
                    if t: br.append((r["prob"], t in gold, page))
        page_rows += [(p, ok) for p, ok, _ in pr]; block_rows += [(p, ok) for p, ok, _ in br]
        out["engines"][e] = {"pages": len(pr), "page_err": sum(not ok for _, ok, _ in pr),
                             "page_auroc": auroc([p for p, ok, _ in pr if ok], [p for p, ok, _ in pr if not ok]),
                             "blocks": len(br), "block_err": sum(not ok for _, ok, _ in br),
                             "block_auroc": auroc([p for p, ok, _ in br if ok], [p for p, ok, _ in br if not ok]),
                             "page_prob_median": sorted(p for p, _, _ in pr)[len(pr) // 2] if pr else None,
                             "pages_detail": sorted([(page, round(p, 4), ok) for p, ok, page in pr], key=lambda x: x[1])}
    out["pooled_page"] = {"n": len(page_rows), "err": sum(not ok for _, ok in page_rows),
                          "auroc": auroc([p for p, ok in page_rows if ok], [p for p, ok in page_rows if not ok]), "tuned": tune(page_rows)}
    out["pooled_block"] = {"n": len(block_rows), "err": sum(not ok for _, ok in block_rows),
                           "auroc": auroc([p for p, ok in block_rows if ok], [p for p, ok in block_rows if not ok]), "tuned": tune(block_rows)}
    for e in ENGINES:  # effective error: CER over pages accepted at the tuned page threshold (flagged pages go to review)
        jv = json.loads((RUNS / e / "_jev.json").read_text())["pages"]
        cer = {r["page"]: r["cer"] for r in json.loads((RUNS / e / "_scores.json").read_text())["pages"] if r.get("cer") is not None}
        t = out["pooled_page"]["tuned"]["t"]; prob = {k: next((r["prob"] for r in v if r["kind"] == "page"), None) for k, v in jv.items()}
        acc = [cer[k] for k in cer if prob.get(k) is not None and prob[k] >= t]
        out["engines"][e].update({"cer_all": sum(cer.values()) / len(cer), "cer_accepted": sum(acc) / len(acc) if acc else None,
                                  "accepted": len(acc), "of": len(cer),
                                  "jev_pages_per_min": round(60 * len(jv) / json.loads((RUNS / e / "_jev.json").read_text())["wall_s"], 2)})
    A2.mkdir(exist_ok=True); (A2 / "dev.json").write_text(json.dumps(out, indent=1)); return out


def _drop(raw, probs, t):
    """Remove grounding blocks whose Jev prob < t (unscored blocks kept). probs: {block_index: prob}."""
    ms = list(TAG.finditer(raw)); keep = []
    if not ms: return raw, 0
    keep.append(raw[:ms[0].start()]); dropped = 0
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(raw)
        p = probs.get(i)
        if p is not None and p < t: dropped += 1; continue
        keep.append(raw[m.start():end])
    return "".join(keep), dropped


def drop_testset(t):
    src, dst = RUNS / "lighton-1b-grounding", RUNS / "lighton-1b-grounding-jevdrop"
    jv = json.loads((src / "_jev.json").read_text())["pages"]; dst.mkdir(exist_ok=True); n = 0
    for f in src.glob("*.txt"):
        probs = {r["i"]: r["prob"] for r in jv.get(f.stem, []) if r["kind"] == "block"}
        txt, d = _drop(f.read_text(), probs, t); (dst / f.name).write_text(txt); n += d
    for f in src.glob("*_labels.json"): shutil.copy(f, dst / f.name)
    print(json.dumps({"dst": str(dst), "dropped_blocks": n}))


def drop_politax(t):
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
    import exp_a
    dst = exp_a.candidate_dir(P1 / "politax-lighton-jevdrop"); exp_a.politax_link(exp_a.LIGHTON_CACHE, dst)
    sc_rows = [json.loads(l) for l in (A2 / "politax-lighton.jsonl").read_text().splitlines()]
    probs = {r["key"]: {x["i"]: x["prob"] for x in r["rows"] if x["kind"] == "block"} for r in sc_rows}
    from packet.eval import politax
    from packet.eval.politax_split import holdout_file_names
    m = politax.load_manifest(); hold = holdout_file_names(m); n = recs = 0
    for d in m.docs:
        if d.file_name not in hold: continue
        for f in sorted((exp_a.LIGHTON_CACHE / "ocr" / d.sha256).glob("p*.json")):
            rec = json.loads(f.read_text()); rec["raw"], k = _drop(rec["raw"], probs.get(f"{d.sha256}/{f.name}", {}), t)
            o = dst / "ocr" / d.sha256 / f.name; o.parent.mkdir(parents=True, exist_ok=True); o.write_text(json.dumps(rec)); n += k; recs += 1
    print(json.dumps({"dst": str(dst), "records": recs, "dropped_blocks": n, "scored_pages": len(probs)}))


def holdout(tb, tp):
    out = {}
    for name in ("lighton", "lfm25x", "lfm3"):
        f = A2 / f"politax-{name}.jsonl"
        if not f.exists(): continue
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        blk = [(x["prob"], r["key"], x["label"], x["i"]) for r in rows for x in r["rows"] if x["kind"] == "block" and x["prob"] is not None]
        pg = [(x["prob"], r["key"]) for r in rows for x in r["rows"] if x["kind"] == "page" and x["prob"] is not None]
        secs = sum(r["s"] for r in rows)
        out[name] = {"pages": len(rows), "jev_pages_per_min": round(60 * len(rows) / secs, 2) if secs else None,
                     "blocks_scored": len(blk), "blocks_flagged_frac": sum(p < tb for p, *_ in blk) / len(blk) if blk else None,
                     "pages_scored": len(pg), "pages_flagged_frac": sum(p < tp for p, _ in pg) / len(pg) if pg else None,
                     "pages_with_flagged_block_frac": (len({k for p, k, *_ in blk if p < tb}) / len(rows)) if blk else None,
                     "lowest_blocks": sorted(blk)[:12], "lowest_pages": sorted(pg)[:8]}
    (A2 / "holdout.json").write_text(json.dumps(out, indent=1)); return out


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "dev": print(json.dumps(dev(), indent=1))
    elif cmd == "drop-testset": drop_testset(float(a[0]))
    elif cmd == "drop-politax": drop_politax(float(a[0]))
    elif cmd == "holdout": print(json.dumps(holdout(float(a[0]), float(a[1])), indent=1))
    else: sys.exit(__doc__)
