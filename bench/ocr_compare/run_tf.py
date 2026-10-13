"""GMA-207: run a transformers OCR model over the test set; record text, latency, peak RSS.
usage: run_tf.py MODEL_KEY PROMPT_MODE OUTDIR IMG...   (PROMPT_MODE: plain|grounding|paddle)"""
import json, sys, time, os, threading
from pathlib import Path
import psutil, torch
from PIL import Image

REVS = {
 "lighton-1b": ("lightonai/LightOnOCR-3-1B", "23d544085c886bbdace647bcfab91203299417e2"),
 "lighton-0.8b": ("lightonai/LightOnOCR-3-0.8B", "36d8636925327c671bd8e479c517a004f7566bd9"),
 "lighton-4b": ("lightonai/LightOnOCR-3-4B", "b71010f095dc3735de7ee6b9969bff8f8f50b39a"),
 "paddle-vl-1.6": ("PaddlePaddle/PaddleOCR-VL-1.6", "c5630abae1d940eafe0697512a0325494b02ab42"),
}
key, mode, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); imgs = sys.argv[4:]
out.mkdir(parents=True, exist_ok=True)
repo, rev = REVS[key]
dev = os.environ.get("DEV", "mps" if torch.backends.mps.is_available() else "cpu")
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[os.environ.get("DT", "bf16")]
proc = psutil.Process(); peak = [0]
def mon():
    while True:
        try: peak[0] = max(peak[0], proc.memory_info().rss)
        except Exception: pass
        time.sleep(0.2)
threading.Thread(target=mon, daemon=True).start()
t0 = time.time()
import transformers as T
from transformers import AutoProcessor
cls = {"lighton-1b": T.LightOnOcrForConditionalGeneration, "lighton-0.8b": T.Qwen3_5ForConditionalGeneration,
       "lighton-4b": T.Qwen3_5ForConditionalGeneration, "paddle-vl-1.6": T.AutoModelForImageTextToText}[key]
if key == "lighton-1b":
    # Card's from_pretrained path loads 0 LM weights under transformers 5.17 (config says model_type=mistral3,
    # checkpoint uses language_model.model.* keys). Workaround: LightOnOcrConfig from the same dict + manual key remap.
    import json as _j, glob as _g
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    snap = snapshot_download(repo, revision=rev)
    cd = _j.load(open(f"{snap}/config.json")); cd.pop("model_type", None); cd.pop("architectures", None)
    cfg = T.LightOnOcrConfig(**cd)
    model = T.LightOnOcrForConditionalGeneration(cfg)
    sd = {(("model.language_model." + k[len("language_model.model."):]) if k.startswith("language_model.model.") else k): v
          for k, v in load_file(f"{snap}/model.safetensors").items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.tie_weights()
    print("REMAP missing:", missing, "unexpected:", unexpected, flush=True)
    model = model.to(dev, dtype=dtype).eval()
else:
    model = cls.from_pretrained(repo, revision=rev, dtype=dtype).to(dev).eval()
processor = AutoProcessor.from_pretrained(repo, revision=rev)
load_s = time.time() - t0
recs = []
for p in imgs:
    img = Image.open(p).convert("RGB")
    content = [{"type": "image", "image": img}]
    if mode == "grounding": content.append({"type": "text", "text": "grounding"})
    if mode == "paddle": content.append({"type": "text", "text": "OCR:"})
    conv = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = {k: (v.to(dev, dtype=dtype) if v.is_floating_point() else v.to(dev)) for k, v in inputs.items()}
    t = time.time()
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=int(os.environ.get("MAXTOK", 2048)), do_sample=False)
    if dev == "mps": torch.mps.synchronize()
    dt = time.time() - t
    gen = ids[0, inputs["input_ids"].shape[1]:]
    txt = processor.decode(gen, skip_special_tokens=True)
    name = Path(p).stem
    (out / f"{name}.txt").write_text(txt)
    r = {"page": name, "seconds": round(dt, 2), "new_tokens": int(gen.shape[0])}
    recs.append(r); print(json.dumps(r), flush=True)
mps_peak = torch.mps.driver_allocated_memory() if dev == "mps" else 0
summ = {"model": repo, "revision": rev, "mode": mode, "device": dev, "dtype": str(dtype), "load_s": round(load_s, 1),
        "pages": len(recs), "total_gen_s": round(sum(r["seconds"] for r in recs), 1),
        "pages_per_min": round(60 * len(recs) / max(1e-9, sum(r["seconds"] for r in recs)), 2),
        "peak_rss_gb": round(peak[0] / 1e9, 2), "mps_driver_alloc_gb_end": round(mps_peak / 1e9, 2), "per_page": recs}
(out / "_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps({k: v for k, v in summ.items() if k != "per_page"}))
