"""GMA-448 Exp B2: OCR the PoliTax holdout with MiniCPM-OCR 1.0 (transformers MPS, sandbox venv) into a NEW cache dir.

Docs run smallest first so a time-capped run still yields whole documents (boundary eval needs every page of a doc).
usage: ocr_minicpm.py DST   (DST must not be the LightOn cache; run `exp_a.py politax-link` into DST first)
env: MAXTOK (default 4096), MAX_PAGES (stop after N pages; whole docs only), FILES (comma list: run only these docs)."""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_eval"))


def main():
    import exp_a
    import run_minicpm_tf as mc
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from packet.eval import politax
    from packet.eval import politax_ocr as po
    from packet.eval.politax_split import POLITAX_HOLDOUT_SUBJECTS
    from packet.source import PdfPageSource
    mc.require_sandbox()
    from huggingface_hub import snapshot_download
    dst = exp_a.candidate_dir(Path(sys.argv[1]).expanduser())
    maxtok, px = int(os.environ.get("MAXTOK", "4096")), 1540
    cap = int(os.environ.get("MAX_PAGES", "0"))
    m = politax.load_manifest(); plans = []
    for subj in sorted(POLITAX_HOLDOUT_SUBJECTS):
        plans += po.plan_pages(m.docs, dst, subject=subj)[0]
    files = set(filter(None, os.environ.get("FILES", "").split(",")))
    by_file: dict = {}
    for p in plans:
        if files and p.file not in files: continue
        by_file.setdefault(p.file, []).append(p)
    ordered = []
    for f in sorted(by_file, key=lambda f: len(by_file[f])):
        if cap and ordered and len(ordered) + len(by_file[f]) > cap:
            continue
        ordered += by_file[f]
    print(json.dumps({"files": {f: len(v) for f, v in by_file.items()}, "planned": len(ordered)}), flush=True)
    engine = po.EngineId("minicpm-ocr1-mps", mc.REPO, mc.REV, "text-recognition", maxtok, px)
    snap = snapshot_download(mc.REPO, revision=mc.REV, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(snap, local_files_only=True, dtype=torch.bfloat16).to("mps").eval()
    proc = AutoProcessor.from_pretrained(snap, local_files_only=True); sources = {}

    def recognize(plan):
        t0 = time.perf_counter(); src = sources.get(plan.file)
        if src is None:
            for o in sources.values(): o.close()
            sources.clear(); src = sources[plan.file] = PdfPageSource(dst / plan.file)
        img = src.render(plan.index, longest_px=px).convert("RGB"); rs = time.perf_counter() - t0
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": mc.PROMPT}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt").to("mps")
        with torch.inference_mode():
            ids = model.generate(**inp, do_sample=False, max_new_tokens=maxtok)
        torch.mps.synchronize()
        gen = ids[0, inp["input_ids"].shape[1]:]; n = int(gen.shape[0])
        raw = proc.decode(gen, skip_special_tokens=True).split("</think>")[-1].strip()
        print(json.dumps({"file": plan.file, "page": plan.index, "s": round(time.perf_counter() - t0, 2), "tok": n}), flush=True)
        return po.PageResult(raw=raw, tokens=n, prompt_tokens=int(inp["input_ids"].shape[1]),
                             stop="length" if n >= maxtok else "eos", render_s=rs)
    vers = {"packet": "harness", "transformers": transformers.__version__, "torch": torch.__version__}
    s = po.run(ordered, recognize, dst, engine, versions=vers)
    print("SUMMARY", json.dumps({"planned": s.planned, "ocr_done": s.ocr_done, "skipped_cached": s.skipped_cached,
                                 "truncated": s.truncated, "wall_s": s.wall_s,
                                 "pages_per_min_wall": round(60 * s.ocr_done / max(s.wall_s, 1e-9), 2)}), flush=True)


if __name__ == "__main__":
    main()
