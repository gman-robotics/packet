"""GMA-207: map raw model outputs to packet.types.Block (harness-only; nothing added to src/packet),
then run Packet's own BoundaryDetector over the multi-doc pages. usage: to_blocks.py RUN_DIR [RUN_DIR...]"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # repo root; testset/ and src/ live here
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image
from packet.types import BBox, Block, BlockKind as K, PageRef, PageClass, SourceKind
from packet.boundary import BoundaryDetector

LIGHTON = {"text": K.PARAGRAPH, "title": K.HEADING, "list": K.LIST_ITEM, "header": K.HEADER, "footer": K.FOOTER,
           "page_number": K.PAGE_NUMBER, "footnote": K.PARAGRAPH, "caption": K.CAPTION, "formula": K.FORMULA,
           "code": K.CODE, "table": K.TABLE, "image": K.FIGURE, "chart": K.FIGURE, "header_image": K.HEADER,
           "footer_image": K.FOOTER, "aside_text": K.PARAGRAPH}
PADDLE = {"doc_title": K.TITLE, "paragraph_title": K.HEADING, "text": K.PARAGRAPH, "content": K.PARAGRAPH, "abstract": K.PARAGRAPH,
          "header": K.HEADER, "footer": K.FOOTER, "number": K.PAGE_NUMBER, "table": K.TABLE, "image": K.FIGURE, "chart": K.FIGURE,
          "figure_title": K.CAPTION, "chart_title": K.CAPTION, "table_title": K.CAPTION, "formula": K.FORMULA, "display_formula": K.FORMULA,
          "inline_formula": K.FORMULA, "seal": K.STAMP, "algorithm": K.CODE, "footnote": K.PARAGRAPH, "vision_footnote": K.PARAGRAPH,
          "aside_text": K.PARAGRAPH, "reference": K.PARAGRAPH, "reference_content": K.PARAGRAPH, "header_image": K.HEADER, "footer_image": K.FOOTER}
TAG = re.compile(r"!\[([a-z_]+)(\+?)\]\((\d+),(\d+),(\d+),(\d+)\)\s*")
DPI = 150

def page_size_pt(img):
    w, h = Image.open(img).size; return w * 72 / DPI, h * 72 / DPI, w, h

def find_img(name):
    for d in ("synth", "packet", "public"):
        for p in (ROOT / "testset" / d).glob(f"{name}.*"):
            if p.suffix in (".png", ".jpeg", ".jpg"): return p

def blocks_for(run, name, page_idx, kind):
    raw = (run / f"{name}.txt").read_text(); img = find_img(name); wpt, hpt, wpx, hpx = page_size_pt(img)
    out, lost = [], []
    if kind == "paddle":
        for i, b in enumerate(json.loads((run / f"{name}.blocks.json").read_text())):
            k = PADDLE.get(b["label"], K.UNKNOWN)
            if k == K.UNKNOWN: lost.append(b["label"])
            x0, y0, x1, y1 = b["bbox"] or (0, 0, 0, 0)
            out.append(Block(id=f"{name}-b{i}", page=page_idx, kind=k, text=b["text"], reading_order=i, source=SourceKind.VLM,
                             bbox=BBox(page=page_idx, x0=x0*72/DPI, y0=y0*72/DPI, x1=x1*72/DPI, y1=y1*72/DPI), extra={"label": b["label"]}))
        import os
        if os.environ.get("TITLE_RULE") == "first_heading":  # variant: first paragraph_title on a page -> TITLE
            for b in out:
                if b.kind == K.HEADING: b.kind = K.TITLE; break
        return out, lost
    ms = list(TAG.finditer(raw))
    if ms:  # grounding output
        for i, m in enumerate(ms):
            end = ms[i+1].start() if i + 1 < len(ms) else len(raw); text = raw[m.end():end].strip()
            lab, cont = m.group(1), m.group(2) == "+"; k = LIGHTON.get(lab, K.UNKNOWN)
            if k == K.UNKNOWN: lost.append(lab)
            lvl = None
            if lab == "title":
                h = re.match(r"^(#+)\s*", text); lvl = len(h.group(1)) if h else 1; text = re.sub(r"^#+\s*", "", text)
                if lvl == 1: k = K.TITLE  # H1 -> TITLE (doc title); deeper -> HEADING
            x0, y0, x1, y1 = (int(m.group(j)) for j in range(3, 7))
            out.append(Block(id=f"{name}-b{i}", page=page_idx, kind=k, text=text, reading_order=i, source=SourceKind.VLM, heading_level=lvl,
                             bbox=BBox(page=page_idx, x0=x0/1000*wpt, y0=y0/1000*hpt, x1=x1/1000*wpt, y1=y1/1000*hpt),
                             extra={"label": lab, "continuation": cont} | ({"html": text} if k == K.TABLE or lab == "chart" else {})))
        return out, lost
    # plain markdown: no labels, no boxes -> paragraph split + markdown cues only
    parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    for i, p in enumerate(parts):
        k, lvl = K.PARAGRAPH, None
        if p.startswith("#"): k, lvl = K.HEADING, len(re.match(r"#+", p).group(0)); p = p.lstrip("# "); k = K.TITLE if lvl == 1 else k
        elif p.startswith("<table") or p.startswith("|"): k = K.TABLE
        elif re.match(r"^([-*•]|\d+[.)])\s", p): k = K.LIST_ITEM
        out.append(Block(id=f"{name}-b{i}", page=page_idx, kind=k, text=p, reading_order=i, source=SourceKind.VLM, heading_level=lvl))
    return out, lost

def boundaries(run, kind):
    det = BoundaryDetector(); got = []
    names = [f"md-0{i}" for i in range(6)]
    for i, n in enumerate(names):
        if not (run / f"{n}.txt").exists(): return None
        bl, _ = blocks_for(run, n, i, kind)
        sig = det.observe(PageRef(packet_id="md", index=i, page_class=PageClass.SCAN), bl)
        if sig: got.append({"new_doc_at": sig.after_page + 1, "reason": sig.reason, "conf": round(sig.confidence, 2)})
    return got

if __name__ == "__main__":
    for d in sys.argv[1:]:
        run = Path(d); kind = "paddle" if any(run.glob("*.blocks.json")) else "vlm"
        allk, lost, nb, nbox = {}, set(), 0, 0
        for f in sorted(run.glob("*.txt")):
            bl, l = blocks_for(run, f.stem, 0, kind); lost |= set(l); nb += len(bl); nbox += sum(1 for b in bl if b.bbox)
            for b in bl: allk[b.kind.value] = allk.get(b.kind.value, 0) + 1
            (run / f"{f.stem}.packet_blocks.json").write_text(json.dumps([b.model_dump() for b in bl], indent=1))
        bd = boundaries(run, kind)
        pred = [b["new_doc_at"] for b in bd] if bd is not None else None
        res = {"blocks": nb, "with_bbox": nbox, "kinds": allk, "unmapped_labels": sorted(lost), "boundary_signals": bd,
               "boundary_exact_match_gold_[2,5]": pred == [2, 5] if pred is not None else None}
        (run / "_blocks.json").write_text(json.dumps(res, indent=1)); print(d, json.dumps(res))
