"""GMA-441 Exp A harness: feed saved OCR outputs through Packet's own parser, BoundaryDetector and eval gate.

Nothing here changes Packet defaults; engines are swapped in harness-side only.
  exp_a.py builtin RUN_DIR             builtin fixtures; scan pages read RUN_DIR/pk-<fixture>-<i>.txt (false-OCR gate on)
  exp_a.py politax-link SRC DST        candidate PoliTax data dir: symlink PDFs + annotations (never writes SRC)
  exp_a.py politax-ocr-lfm DST         OCR holdout scan pages into DST/ocr with LFM2.5-VL-1.6B-Extract (MLX)
  exp_a.py politax-ocr-lfm3 DST        same with LFM3-VL-2B on transformers MPS (sandbox venv only; trust_remote_code)
  exp_a.py politax-untag SRC DST       control: copy SRC holdout OCR records into DST with grounding tags stripped
  exp_a.py politax-eval DATA_DIR       holdout-only --ocr-cache eval (macro over holdout; per-fixture F1; cache stats)
  exp_a.py md-f1 RUN_DIR               boundary P/R/F1 (tolerance 0) on synth md-00..05 vs gold new docs at [2,5]
Raw OCR text is always parsed by packet.ocr_parse.parse_lighton_grounding (markdown fallback when untagged)."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from packet.backends import EscalatingExtractor
from packet.extract import NativeTextExtractor
from packet.ocr_parse import parse_lighton_grounding
from packet.pipeline import Pipeline
from packet.types import PageClass, PageRef

LIGHTON_CACHE = (Path.home() / ".cache/packet/politax").resolve()


def candidate_dir(p: Path) -> Path:
    """Destination for candidate OCR records; refuses the LightOn cache itself."""
    if p.resolve() == LIGHTON_CACHE:
        sys.exit(f"refusing to write the LightOn OCR cache {LIGHTON_CACHE}; use a separate candidate dir")
    return p


OCR_CLASSES = {PageClass.SCAN, PageClass.MIXED, PageClass.SUSPECT_ENCODING}


class RunDirExtractor:
    """Scan-page extractor that returns blocks parsed from a saved run's page text."""

    def __init__(self, run: Path, fixture: str):
        self.run, self.fixture, self.used, self.missing = run, fixture, 0, 0

    def supports(self, page_class):
        return page_class in OCR_CLASSES

    def extract(self, ref: PageRef, text: str, image=None):
        f = self.run / f"pk-{self.fixture}-{ref.index}.txt"
        if not f.exists():
            self.missing += 1; return []
        self.used += 1
        return parse_lighton_grounding(f.read_text(), ref)


def builtin(run: Path) -> dict:
    from packet.eval.fixtures import build_builtin
    from packet.eval.run import evaluate_gate, results_as_dict, run_fixture
    results, used, missing = [], 0, 0
    for fx in build_builtin(Path(os.environ.get("TMPDIR", "/tmp")) / "gma441-builtin"):
        ex = RunDirExtractor(run, fx.name)
        r = run_fixture(fx, Pipeline(extractors=[EscalatingExtractor(NativeTextExtractor(), ex)]))
        results.append(r); used += ex.used; missing += ex.missing
    gate = evaluate_gate(results)
    out = {"run": str(run), "fixtures": results_as_dict(results), "ocr_pages_used": used, "ocr_pages_missing": missing,
           "gate": {"passed": gate.passed, "failures": gate.failures, "macro_f1": gate.macro_f1}}
    (run / "_packet_eval_builtin.json").write_text(json.dumps(out, indent=1, default=str)); return out


def md_f1(run: Path) -> dict:
    blk = json.loads((run / "_blocks.json").read_text())
    if blk.get("boundary_signals") is None:
        sys.exit(f"{run}: md-00..05 page text missing; no boundary score")
    pred = {b["new_doc_at"] for b in blk["boundary_signals"]}; gold = {2, 5}
    tp = len(pred & gold); p = tp / len(pred) if pred else 0.0; r = tp / len(gold)
    out = {"pred": sorted(pred), "gold": sorted(gold), "p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    (run / "_md_f1.json").write_text(json.dumps(out)); return out


def politax_link(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and (f.suffix == ".pdf" or f.name.endswith(".jsonl")) and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f)


def politax_ocr_lfm(dst: Path) -> None:
    import importlib.metadata as md

    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    from packet.eval import politax
    from packet.eval import politax_ocr as po
    from packet.eval.politax_split import POLITAX_HOLDOUT_SUBJECTS
    from packet.source import PdfPageSource
    sys.path.insert(0, str(Path(__file__).parent))
    import run_lfm_extract as lx
    from huggingface_hub import snapshot_download
    maxtok, px = int(os.environ.get("MAXTOK", "4096")), 1540  # politax-ocr defaults (--max-tokens 4096, --render-px 1540)
    m = politax.load_manifest()
    plans = []
    for subj in sorted(POLITAX_HOLDOUT_SUBJECTS):
        plans += po.plan_pages(m.docs, dst, subject=subj)[0]
    engine = po.EngineId("lfm25x-mlx", lx.REPO, lx.REV, "extract-json->markdown", maxtok, px)
    model, proc = load(snapshot_download(lx.REPO, revision=lx.REV))
    prompt = apply_chat_template(proc, model.config, [{"role": "system", "content": lx.SYSTEM}, {"role": "user", "content": ""}], num_images=1)
    sources = {}

    def recognize(plan):
        t0 = time.perf_counter(); src = sources.get(plan.file)
        if src is None:
            for o in sources.values(): o.close()
            sources.clear(); src = sources[plan.file] = PdfPageSource(dst / plan.file)
        img = src.render(plan.index, longest_px=px); rs = time.perf_counter() - t0
        r = generate(model, proc, prompt, image=[img], max_tokens=maxtok, temperature=0.0, verbose=False)
        d, ok = lx.parse(r.text)
        raw = lx.render(d) if ok else r.text
        stop = "length" if r.generation_tokens >= maxtok else "eos"
        return po.PageResult(raw=raw, tokens=r.generation_tokens, prompt_tokens=r.prompt_tokens,
                             prefill_s=r.prompt_tokens / r.prompt_tps if r.prompt_tps else 0.0,
                             decode_s=r.generation_tokens / r.generation_tps if r.generation_tps else 0.0,
                             stop=stop, render_s=rs)
    vers = {"packet": "harness", "mlx_vlm": md.version("mlx-vlm")}
    s = po.run(plans, recognize, dst, engine, versions=vers)
    print(json.dumps({"planned": s.planned, "ocr_done": s.ocr_done, "skipped_cached": s.skipped_cached, "truncated": s.truncated,
                      "wall_s": s.wall_s, "gen_s": s.gen_s, "pages_per_min_wall": round(60 * s.ocr_done / max(s.wall_s, 1e-9), 2),
                      "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2)}))


def politax_ocr_lfm3(dst: Path) -> None:
    """Sandbox-venv only (/Users/Shared/gman-sandbox/lfm3-mps/venv); see run_lfm3_mps.py."""
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from packet.eval import politax
    from packet.eval import politax_ocr as po
    from packet.eval.politax_split import POLITAX_HOLDOUT_SUBJECTS
    from packet.source import PdfPageSource
    sys.path.insert(0, str(Path(__file__).parent))
    import run_lfm3_mps as l3
    l3.require_sandbox()
    from huggingface_hub import snapshot_download
    maxtok, px = int(os.environ.get("MAXTOK", "4096")), 1540
    m = politax.load_manifest(); plans = []
    for subj in sorted(POLITAX_HOLDOUT_SUBJECTS):
        plans += po.plan_pages(m.docs, dst, subject=subj)[0]
    engine = po.EngineId("lfm3-mps", l3.REPO, l3.REV, "markdown-nothink", maxtok, px)
    snap = snapshot_download(l3.REPO, revision=l3.REV, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(snap, local_files_only=True, dtype=torch.bfloat16, trust_remote_code=True).to("mps").eval()
    proc = AutoProcessor.from_pretrained(snap, trust_remote_code=True); sources = {}; peak = [0]

    def recognize(plan):
        t0 = time.perf_counter(); src = sources.get(plan.file)
        if src is None:
            for o in sources.values(): o.close()
            sources.clear(); src = sources[plan.file] = PdfPageSource(dst / plan.file)
        img = src.render(plan.index, longest_px=px).convert("RGB"); rs = time.perf_counter() - t0
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": l3.PROMPT}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt", enable_thinking=False).to("mps")
        with torch.inference_mode():
            ids = model.generate(**inp, do_sample=False, max_new_tokens=maxtok, repetition_penalty=1.1)
        torch.mps.synchronize(); peak[0] = max(peak[0], torch.mps.driver_allocated_memory())
        gen = ids[0, inp["input_ids"].shape[1]:]; n = int(gen.shape[0])
        raw = proc.decode(gen, skip_special_tokens=True).split("</think>")[-1].strip()
        return po.PageResult(raw=raw, tokens=n, prompt_tokens=int(inp["input_ids"].shape[1]),
                             stop="length" if n >= maxtok else "eos", render_s=rs)
    vers = {"packet": "harness", "transformers": transformers.__version__, "torch": torch.__version__}
    s = po.run(plans, recognize, dst, engine, versions=vers)
    print(json.dumps({"planned": s.planned, "ocr_done": s.ocr_done, "skipped_cached": s.skipped_cached, "truncated": s.truncated,
                      "wall_s": s.wall_s, "gen_s": s.gen_s, "pages_per_min_wall": round(60 * s.ocr_done / max(s.wall_s, 1e-9), 2),
                      "mps_driver_peak_gb": round(peak[0] / 1e9, 2)}))


def politax_untag(src: Path, dst: Path) -> int:
    """Label-dependency control: same LightOn text, ``![label](bbox)`` tags removed (markdown fallback path)."""
    import re

    from packet.eval import politax
    from packet.eval.politax_split import holdout_file_names
    tag = re.compile(r"!\[[a-z_]+\+?\]\(\d+,\d+,\d+,\d+\)\s*")
    m = politax.load_manifest(); hold = holdout_file_names(m); n = 0
    for d in m.docs:
        if d.file_name not in hold: continue
        for f in sorted((src / "ocr" / d.sha256).glob("p*.json")):
            rec = json.loads(f.read_text()); rec["raw"] = tag.sub("", rec["raw"])
            out = dst / "ocr" / d.sha256 / f.name; out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec)); n += 1
    return n


def politax_eval(data: Path) -> dict:
    from packet.eval.politax_ocr_eval import run_with_ocr_cache
    from packet.eval.politax_split import holdout_file_names
    from packet.eval.run import prepare_politax
    m, fixtures = prepare_politax(data)
    hold = holdout_file_names(m)
    fx = [f for f in fixtures if (f.notes or f.pdf.name) in hold]
    from packet.eval.run import OCR_CACHE_FLOORS, evaluate_gate
    results, stats = run_with_ocr_cache(m, fx, data)
    lo, macro_floor = OCR_CACHE_FLOORS["politax"]
    gate = evaluate_gate(results, min_f1=lo, min_macro_f1=macro_floor)  # pool = these holdout rows
    rows = [{"name": r.name, "f1": r.boundary.f1, "pages": r.pages, "gold_docs": r.gold_docs, "pred_docs": r.pred_docs,
             "false_ocr_pages": r.false_ocr_pages} for r in results]
    out = {"data_dir": str(data), "holdout_macro_f1": sum(r["f1"] for r in rows) / len(rows), "fixtures": rows,
           "false_ocr_pages_total": sum(len(r["false_ocr_pages"]) for r in rows), "ocr_cache": stats.as_dict(),
           "gate_holdout_only": {"passed": gate.passed, "failures": gate.failures, "macro_f1": gate.macro_f1,
                                 "floors": [lo, macro_floor], "note": "holdout fixtures only; not the full-set CLI gate"}}
    return out


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "builtin": print(json.dumps(builtin(Path(a[0])), default=str))
    elif cmd == "md-f1": print(json.dumps(md_f1(Path(a[0]))))
    elif cmd == "politax-link": politax_link(Path(a[0]).expanduser(), candidate_dir(Path(a[1]).expanduser()))
    elif cmd == "politax-ocr-lfm": politax_ocr_lfm(candidate_dir(Path(a[0]).expanduser()))
    elif cmd == "politax-ocr-lfm3": politax_ocr_lfm3(candidate_dir(Path(a[0]).expanduser()))
    elif cmd == "politax-untag": print(politax_untag(Path(a[0]).expanduser(), candidate_dir(Path(a[1]).expanduser())))
    elif cmd == "politax-eval": print(json.dumps(politax_eval(Path(a[0]).expanduser()), default=str))
    else: sys.exit(__doc__)
