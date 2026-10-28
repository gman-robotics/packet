"""GMA-441 Exp A: openbmb/MiniCPM-OCR-1.0 on transformers (native minicpm_ocr, no remote code) + torch MPS, bf16, greedy.

Page-level use of the card's page prompt "Text Recognition:" (no PP-DocLayoutV3 layout stage). Homebrew llama.cpp
0.7.0 cannot load the mmproj ("unknown projector type: minicpm_ocr_1_0") and the --HEAD build fails, so MPS is the
Mac path here. Run from the sandbox venv /Users/Shared/gman-sandbox/lfm3-mps/venv with HF_HUB_OFFLINE=1.
usage: run_minicpm_tf.py MODEL_DIR_OR_REPO OUTDIR IMG...   (MAXTOK env, default 2048)"""
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

REPO, REV = "openbmb/MiniCPM-OCR-1.0", "6796197bb3a8e224b2f64ae462bfbb3f8a15ac4b"
PROMPT = "Text Recognition:"


SANDBOX = "/Users/Shared/gman-sandbox/lfm3-mps/venv"


def require_sandbox():
    """Refuse to run outside the sandbox venv or with hub network on (trust_remote_code safety)."""
    if os.path.realpath(sys.prefix) != os.path.realpath(SANDBOX) or os.environ.get("HF_HUB_OFFLINE") != "1":
        sys.exit(f"run from {SANDBOX} with HF_HUB_OFFLINE=1 (sys.prefix={sys.prefix})")


def main():
    require_sandbox()
    src, out, imgs = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True); maxtok = int(os.environ.get("MAXTOK", "2048"))
    T1 = time.time()  # imports done; snapshot resolve counts toward load
    if not Path(src).exists():
        from huggingface_hub import snapshot_download
        src = snapshot_download(REPO, revision=REV, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(src, local_files_only=True, dtype=torch.bfloat16).to("mps").eval()
    proc = AutoProcessor.from_pretrained(src, local_files_only=True); torch.mps.synchronize(); T2 = time.time()
    me = psutil.Process(); peak_rss = me.memory_info().rss; peak_drv = torch.mps.driver_allocated_memory()
    recs = []
    for p in imgs:
        name = Path(p).stem
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(p).convert("RGB")}, {"type": "text", "text": PROMPT}]}]
        t = time.time()
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt").to("mps")
        with torch.inference_mode():
            ids = model.generate(**inp, do_sample=False, max_new_tokens=maxtok)
        torch.mps.synchronize(); dt = time.time() - t
        gen = ids[0, inp["input_ids"].shape[1]:]; txt = proc.decode(gen, skip_special_tokens=True)
        txt = txt.split("</think>")[-1].strip()
        (out / f"{name}.txt").write_text(txt)
        peak_rss = max(peak_rss, me.memory_info().rss); peak_drv = max(peak_drv, torch.mps.driver_allocated_memory())
        rec = {"page": name, "seconds": round(dt, 2), "gen_tokens": int(gen.shape[0]), "prompt_tokens": int(inp["input_ids"].shape[1]),
               "truncated": int(gen.shape[0]) >= maxtok}
        recs.append(rec); print(json.dumps(rec), flush=True)
    g = sum(x["seconds"] for x in recs)
    s = {"model": REPO, "revision": REV, "weights": src, "prompt": PROMPT, "decode": "greedy",
         "backend": f"transformers {transformers.__version__} / torch {torch.__version__} MPS, native", "dtype": "bf16",
         "imports_s": round(T1 - T0, 2), "load_s": round(T2 - T1, 2), "process_to_loaded_s": round(T2 - T0, 2),
         "pages": len(recs), "total_gen_s": round(g, 1), "pages_per_min": round(60 * len(recs) / max(g, 1e-9), 2),
         "truncated": sum(x["truncated"] for x in recs), "peak_rss_gb": round(peak_rss / 1e9, 2),
         "mps_driver_peak_gb": round(peak_drv / 1e9, 2), "per_page": recs}
    (out / "_summary.json").write_text(json.dumps(s, indent=1))
    print("SUMMARY", json.dumps({k: v for k, v in s.items() if k != "per_page"}))


if __name__ == "__main__":
    main()
