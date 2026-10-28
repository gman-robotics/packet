"""GMA-441 Exp A: LiquidAI LFM2.5-VL-1.6B-Extract on MLX (mlx-vlm), card-style extraction prompt.

The model card's prompt is a system message "Extract the following from the image:\n\n<YAML fields>\n\nRespond with
only a JSON object. Do not include any text outside the JSON." plus a user turn holding only the image; greedy decode.
We pass a page-transcription field list (SCHEMA) so the JSON maps onto Packet furniture/body/table blocks.
usage: run_lfm_extract.py MODEL_DIR_OR_REPO OUTDIR IMG...   (MAXTOK env, default 2048)
Writes per page: <name>.json (raw model text), <name>.txt (rendered markdown for score.py / to_blocks.py),
plus _labels.json (furniture string -> field label, read by score.py) and _summary.json (speed, memory)."""
import time; T0 = time.time()  # noqa: I001 - start the clock before heavy imports
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx_vlm
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

REPO, REV = "LiquidAI/LFM2.5-VL-1.6B-Extract", "21073aa65b326c125ceb07d80fa08900bc2eb0d9"
SCHEMA = """header: Running header text printed at the very top of the page, verbatim, or empty
title: The document or section title, verbatim, or empty
body_text: All other text on the page transcribed verbatim in reading order, one line per printed line, or empty if the page is blank
tables: Every table on the page as a markdown table, verbatim, or empty
footer: Running footer text printed at the bottom of the page, verbatim, or empty
page_number: The page number exactly as printed (for example "Page 2 of 3" or "4"), or empty"""
SYSTEM = f"Extract the following from the image:\n\n{SCHEMA}\n\nRespond with only a JSON object. Do not include any text outside the JSON."
BLANK = {"empty", "none", "n/a", "null", ""}
FIELDS = ("header", "title", "body_text", "tables", "footer", "page_number")


def parse(raw: str) -> tuple[dict, bool]:
    """Strict JSON first; duplicate keys keep the first value (the model sometimes repeats a key)."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`"); s = s[s.find("{"):]
    try:
        d = json.loads(s, object_pairs_hook=lambda kv: {k: v for k, v in reversed(kv)})
        return ({k: d.get(k) for k in FIELDS}, True) if isinstance(d, dict) else ({}, False)
    except ValueError:
        return {}, False


def as_text(v) -> str:
    if v is None: return ""
    if isinstance(v, list): return "\n\n".join(as_text(x) for x in v)
    if isinstance(v, dict): return "\n".join(f"{k}: {as_text(x)}" for k, x in v.items())
    s = str(v).strip()
    return "" if s.lower() in BLANK else s  # the SCHEMA says "or empty"; the model answers the literal word


def render(d: dict) -> str:
    """Markdown in page order: header, # title, body, tables, footer, page number (blank fields dropped)."""
    parts = [as_text(d.get("header")), ("# " + as_text(d["title"])) if as_text(d.get("title")) else "",
             as_text(d.get("body_text")), as_text(d.get("tables")), as_text(d.get("footer")), as_text(d.get("page_number"))]
    return "\n\n".join(p for p in parts if p)


def main():
    src, out, imgs = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True); maxtok = int(os.environ.get("MAXTOK", "2048"))
    if not Path(src).exists():
        from huggingface_hub import snapshot_download
        src = snapshot_download(REPO, revision=REV)
    T1 = time.time(); model, proc = load(src); mx.eval(model.parameters()); T2 = time.time()
    conv = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ""}]  # user turn = image only (card)
    prompt = apply_chat_template(proc, model.config, conv, num_images=1)
    recs, labels = [], {}
    for p in imgs:
        name = Path(p).stem; t = time.time()
        r = generate(model, proc, prompt, image=[p], max_tokens=maxtok, temperature=0.0, verbose=False)
        dt = time.time() - t; raw = r.text; d, ok = parse(raw)
        (out / f"{name}.json").write_text(raw); (out / f"{name}.txt").write_text(render(d) if ok else raw)
        labels[name] = {as_text(d.get(k)): k for k in ("header", "footer", "page_number") if ok and as_text(d.get(k))}
        rec = {"page": name, "seconds": round(dt, 2), "gen_tokens": r.generation_tokens, "prompt_tokens": r.prompt_tokens,
               "json_ok": ok, "truncated": r.generation_tokens >= maxtok}
        recs.append(rec); print(json.dumps(rec), flush=True)
    (out / "_labels.json").write_text(json.dumps(labels, indent=1))
    g = sum(x["seconds"] for x in recs)
    s = {"model": REPO, "revision": REV, "weights": src, "prompt": "card system prompt + SCHEMA", "schema": SCHEMA,
         "backend": f"mlx-vlm {mlx_vlm.__version__} / mlx {mx.__version__}", "dtype": "bf16 (checkpoint native)",
         "imports_s": round(T1 - T0, 2), "load_s": round(T2 - T1, 2), "process_to_loaded_s": round(T2 - T0, 2),
         "pages": len(recs), "total_gen_s": round(g, 1), "pages_per_min": round(60 * len(recs) / max(g, 1e-9), 2),
         "json_ok": sum(x["json_ok"] for x in recs), "truncated": sum(x["truncated"] for x in recs),
         "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2), "per_page": recs}
    (out / "_summary.json").write_text(json.dumps(s, indent=1))
    print("SUMMARY", json.dumps({k: v for k, v in s.items() if k not in ("per_page", "schema")}))


if __name__ == "__main__":
    main()
