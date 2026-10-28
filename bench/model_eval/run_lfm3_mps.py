"""GMA-441 Exp A (Tom 10/27 10:30 PM MT "run it anyway"): LiquidAI LFM3-VL-2B on transformers + torch MPS.

trust_remote_code=True (the repo's custom lfm3_vl code at the pinned revision). Run ONLY from the sandbox venv
/Users/Shared/gman-sandbox/lfm3-mps/venv, with HF_HUB_OFFLINE=1 (no network; also blocks hub kernel fetches).
No-think config (enable_thinking=False, the card's "no think" row), greedy decode, bf16.
usage: run_lfm3_mps.py MODEL_DIR_OR_REPO OUTDIR IMG...   (MAXTOK env, default 2048)"""
import time; T0 = time.time()  # noqa: I001 - start the clock before heavy imports
import json
import os
import sys
from pathlib import Path

import psutil
import torch
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO, REV = "LiquidAI/LFM3-VL-2B", "f83d2b0a2296aaa277f394990a9b8b949ddf9691"
PROMPT = ("Transcribe all text on this page in reading order as markdown. Write tables as markdown tables. "
          "Output only the transcription. If the page is blank, output nothing.")


def main():
    src, out, imgs = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True); maxtok = int(os.environ.get("MAXTOK", "2048"))
    if not Path(src).exists():
        from huggingface_hub import snapshot_download
        src = snapshot_download(REPO, revision=REV)
    T1 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, trust_remote_code=True).to("mps").eval()
    proc = AutoProcessor.from_pretrained(src, trust_remote_code=True); torch.mps.synchronize(); T2 = time.time()
    me = psutil.Process(); peak_rss = me.memory_info().rss; peak_drv = torch.mps.driver_allocated_memory()
    recs = []
    for p in imgs:
        name = Path(p).stem
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(p).convert("RGB")}, {"type": "text", "text": PROMPT}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt", enable_thinking=False).to("mps")
        t = time.time()
        with torch.inference_mode():
            ids = model.generate(**inp, do_sample=False, max_new_tokens=maxtok, repetition_penalty=1.1)
        torch.mps.synchronize(); dt = time.time() - t
        gen = ids[0, inp["input_ids"].shape[1]:]; txt = proc.decode(gen, skip_special_tokens=True)
        txt = txt.split("</think>")[-1].strip()
        (out / f"{name}.txt").write_text(txt)
        peak_rss = max(peak_rss, me.memory_info().rss); peak_drv = max(peak_drv, torch.mps.driver_allocated_memory())
        rec = {"page": name, "seconds": round(dt, 2), "gen_tokens": int(gen.shape[0]), "prompt_tokens": int(inp["input_ids"].shape[1]),
               "truncated": int(gen.shape[0]) >= maxtok}
        recs.append(rec); print(json.dumps(rec), flush=True)
    g = sum(x["seconds"] for x in recs)
    s = {"model": REPO, "revision": REV, "weights": src, "prompt": PROMPT, "enable_thinking": False, "decode": "greedy, repetition_penalty 1.1 (card)",
         "backend": f"transformers {transformers.__version__} / torch {torch.__version__} MPS, trust_remote_code", "dtype": "bf16",
         "imports_s": round(T1 - T0, 2), "load_s": round(T2 - T1, 2), "process_to_loaded_s": round(T2 - T0, 2),
         "pages": len(recs), "total_gen_s": round(g, 1), "pages_per_min": round(60 * len(recs) / max(g, 1e-9), 2),
         "truncated": sum(x["truncated"] for x in recs), "peak_rss_gb": round(peak_rss / 1e9, 2),
         "mps_driver_peak_gb": round(peak_drv / 1e9, 2), "per_page": recs}
    (out / "_summary.json").write_text(json.dumps(s, indent=1))
    print("SUMMARY", json.dumps({k: v for k, v in s.items() if k != "per_page"}))


if __name__ == "__main__":
    main()
