"""GMA-207: PaddleOCR-VL-1.6 page-level pipeline (layout + VL), paddleocr doc_parser, native backend on CPU.
usage: run_paddle.py OUTDIR IMG..."""
import json, sys, time, threading
from pathlib import Path
import psutil
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); imgs = sys.argv[2:]
proc = psutil.Process(); peak = [0]
def mon():
    while True:
        try: peak[0] = max(peak[0], proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True)))
        except Exception: pass
        time.sleep(0.2)
threading.Thread(target=mon, daemon=True).start()
t0 = time.time()
from paddleocr import PaddleOCRVL
pipe = PaddleOCRVL(pipeline_version="v1.6", device="cpu")
load_s = time.time() - t0; recs = []; labels = {}
for p in imgs:
    name = Path(p).stem; t = time.time()
    res = list(pipe.predict(p))
    dt = time.time() - t
    d = out / name; d.mkdir(exist_ok=True)
    md_parts = []
    for r in res:
        r.save_to_json(save_path=str(d)); r.save_to_markdown(save_path=str(d))
    js = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
    blocks = []
    for j in js:
        for b in j.get("parsing_res_list", []):
            blocks.append({"label": b.get("block_label"), "bbox": b.get("block_bbox"), "text": b.get("block_content", "")})
    # flat text in pipeline reading order, labels kept for scoring
    (out / f"{name}.txt").write_text("\n\n".join(f"![{b['label']}]({','.join(str(int(x)) for x in (b['bbox'] or [0,0,0,0]))})\n{b['text']}" for b in blocks))
    (out / f"{name}.blocks.json").write_text(json.dumps(blocks, indent=1))
    r = {"page": name, "seconds": round(dt, 2), "n_blocks": len(blocks)}; recs.append(r); print(json.dumps(r), flush=True)
summ = {"model": "PaddleOCR-VL-1.6 pipeline (paddleocr doc_parser v1.6)", "device": "cpu", "load_s": round(load_s, 1), "pages": len(recs),
        "total_gen_s": round(sum(r["seconds"] for r in recs), 1),
        "pages_per_min": round(60 * len(recs) / max(1e-9, sum(r["seconds"] for r in recs)), 2),
        "peak_rss_gb": round(peak[0] / 1e9, 2), "per_page": recs}
(out / "_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps({k: v for k, v in summ.items() if k != "per_page"}))
