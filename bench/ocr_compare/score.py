"""GMA-207 scorer. usage: score.py RUN_DIR [RUN_DIR...] -> prints table, writes RUN_DIR/_scores.json
Text metrics vs gold: CER/WER (jiwer) on normalized text with markup stripped (order-sensitive);
line recall (order-insensitive: fraction of gold lines found as substrings after normalization);
table cell recall (gold cells found as a cell in an output HTML/markdown table);
furniture recall (header/footer/page-number strings present) and furniture-labeled (present AND tagged
as header/footer/page_number by the model's own labels - grounding/Paddle only)."""
import json, re, sys, html
from pathlib import Path
import jiwer
ROOT = Path(__file__).resolve().parent.parent
TAG = re.compile(r"!\[([a-z_]+\+?)\]\((\d+),(\d+),(\d+),(\d+)\)")

def strip(t):
    t = TAG.sub(" ", t)
    t = re.sub(r"</t[dh]>", " ", t); t = re.sub(r"<[^>]+>", " ", t); t = html.unescape(t)
    t = re.sub(r"^\s*#+\s*", "", t, flags=re.M); t = t.replace("**", "").replace("|", " ")
    t = re.sub(r"^[\s:-]+$", "", t, flags=re.M)
    return t
def norm(t): return re.sub(r"\s+", " ", t.replace("|", " ")).strip()
def cells(t):
    out = [norm(html.unescape(re.sub(r"<[^>]+>", "", c))) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", t, flags=re.S)]
    for line in t.splitlines():
        if line.strip().startswith("|"): out += [norm(c) for c in line.strip().strip("|").split("|")]
    return [c for c in out if c and not re.fullmatch(r":?-+:?", c)]
def labeled(t, s):
    """label of the grounding block containing string s (LightOn) or paddle json label"""
    for m in re.finditer(r"!\[([a-z_]+)\+?\]\([^)]*\)\s*(.*?)(?=\s*!\[[a-z_]+\+?\]\(|\Z)", t, flags=re.S):
        if norm(s) in norm(strip(m.group(2))): return m.group(1)
    return None

def gold_for(name):
    for d in ("synth", "packet"):
        p = ROOT / "testset" / d / f"{name}.json"
        if p.exists(): return json.loads(p.read_text())
    return None

def score_dir(run):
    run = Path(run); rows = []
    labels_path = run / "_labels.json"; plabels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    for f in sorted(run.glob("*.txt")):
        g = gold_for(f.stem)
        if not g: continue
        raw = f.read_text(); hyp = norm(strip(raw)); ref = norm(g["full_text"])
        if not ref:
            rows.append({"page": f.stem, "blank_page_output_chars": len(hyp)}); continue
        r = {"page": f.stem, "cer": jiwer.cer(ref, hyp) if hyp else 1.0, "wer": jiwer.wer(ref, hyp) if hyp else 1.0}
        lines = [norm(x) for x in g["full_text"].split("\n") if norm(x)]
        r["line_recall"] = sum(1 for x in lines if x in hyp) / len(lines)
        if g.get("table"):
            gc = [norm(c) for row in g["table"] for c in row if c]; hc = set(cells(raw))
            r["table_cell_recall"] = sum(1 for c in gc if c in hc) / len(gc)
        fur = [(k, g[k]) for k in ("header", "footer", "page_number") if g.get(k)]
        if fur:
            r["furniture_recall"] = sum(1 for k, s in fur if norm(s) in hyp) / len(fur)
            if not f.stem.startswith(("md-", "hs-")):
                rows.append(r); continue  # packet fixtures stack furniture at page top; labels not expected
            lab = [(labeled(raw, s) or plabels.get(f.stem, {}).get(s)) for k, s in fur]
            r["furniture_labeled"] = sum(1 for (k, s), l in zip(fur, lab) if l and (l == k or (k == "page_number" and l in ("page_number", "number")) or (k in ("header","footer") and l.startswith(k)))) / len(fur)
            r["pagenum_found"] = norm(g["page_number"]) in hyp if g.get("page_number") else None
        rows.append(r)
    def mean(k):
        v = [x[k] for x in rows if x.get(k) is not None]; return round(sum(v) / len(v), 4) if v else None
    agg = {k: mean(k) for k in ("cer", "wer", "line_recall", "table_cell_recall", "furniture_recall", "furniture_labeled")}
    agg["n_pages"] = len(rows); agg["blank_page_output_chars"] = [x["blank_page_output_chars"] for x in rows if "blank_page_output_chars" in x]
    (run / "_scores.json").write_text(json.dumps({"agg": agg, "pages": rows}, indent=1)); return agg

if __name__ == "__main__":
    for d in sys.argv[1:]: print(d, json.dumps(score_dir(d)))
