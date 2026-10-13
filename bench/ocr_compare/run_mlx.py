"""GMA-207: MLX (mlx-vlm) runner on the pinned HF snapshots (no third-party conversions; bf16, converted in memory).
usage: run_mlx.py MODEL_KEY PROMPT_MODE OUTDIR IMG..."""
import json, sys, time, os, glob
from pathlib import Path
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
REVS = {"lighton-1b": ("lightonai/LightOnOCR-3-1B", "23d544085c886bbdace647bcfab91203299417e2"),
        "lighton-0.8b": ("lightonai/LightOnOCR-3-0.8B", "36d8636925327c671bd8e479c517a004f7566bd9"),
        "lighton-4b": ("lightonai/LightOnOCR-3-4B", "b71010f095dc3735de7ee6b9969bff8f8f50b39a"),
        "paddle-vl-1.6": ("PaddlePaddle/PaddleOCR-VL-1.6", "c5630abae1d940eafe0697512a0325494b02ab42")}
key, mode, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); imgs = sys.argv[4:]; out.mkdir(parents=True, exist_ok=True)
repo, rev = REVS[key]
snap = f"{os.environ['HF_HOME']}/hub/models--{repo.replace('/', '--')}/snapshots/{rev}"
t0 = time.time(); model, processor = load(snap, trust_remote_code=False); load_s = time.time() - t0
prompt = {"plain": "", "grounding": "grounding", "paddle": "OCR:"}[mode]
recs = []
for p in imgs:
    fp = apply_chat_template(processor, model.config, prompt, num_images=1)
    if key == 'lighton-1b' and mode == 'plain':  # mlx-vlm template adds a stray newline that flips the 1B into grounding output
        fp = fp.replace('<|image_pad|>\n<|im_end|>', '<|image_pad|><|im_end|>')
    t = time.time()
    r = generate(model, processor, fp, image=[p], max_tokens=int(os.environ.get("MAXTOK", 2048)), temperature=0.0, verbose=False)
    dt = time.time() - t
    txt = r.text if hasattr(r, "text") else r
    name = Path(p).stem; (out / f"{name}.txt").write_text(txt)
    rec = {"page": name, "seconds": round(dt, 2), "new_tokens": getattr(r, "generation_tokens", None)}; recs.append(rec); print(json.dumps(rec), flush=True)
summ = {"model": repo, "revision": rev, "mode": mode, "backend": "mlx-vlm", "dtype": "bf16 (checkpoint native)", "load_s": round(load_s, 1), "pages": len(recs),
        "total_gen_s": round(sum(r["seconds"] for r in recs), 1), "pages_per_min": round(60 * len(recs) / max(1e-9, sum(r["seconds"] for r in recs)), 2),
        "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2), "per_page": recs}
(out / "_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps({k: v for k, v in summ.items() if k != "per_page"}))
