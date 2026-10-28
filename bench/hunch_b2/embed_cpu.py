"""GMA-448: embed B2/pages.jsonl page heads with google/embeddinggemma-2 @26711154 on CPU (sandbox venv, HF_HUB_OFFLINE=1).

Same text path as bench/embed_eval/embed_pages.py (mean pool incl. prompt, L2), "Document" prompt, fp32 on CPU,
max 256 tokens. Writes B2/emb.npy + B2/emb_keys.json (resumes: only new keys are embedded) and B2/latency.json:
single-page (batch 1) wall time per page, p50/p95 over a fixed sample, which is the interactive latency number."""
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embed_eval"))
B2 = Path(os.environ.get("B2", Path.home() / ".cache/packet/model-eval/b2"))
MAX_LEN = 256


def main():
    import numpy as np
    import torch
    from embed_pages import DOC, REPO, REV, SANDBOX
    if os.path.realpath(sys.prefix) != os.path.realpath(SANDBOX) or os.environ.get("HF_HUB_OFFLINE") != "1":
        sys.exit(f"run from {SANDBOX} with HF_HUB_OFFLINE=1")
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    snap = snapshot_download(REPO, revision=REV, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    model = AutoModel.from_pretrained(snap, local_files_only=True, dtype=torch.float32).eval()

    def enc(texts):
        e = tok([DOC + t for t in texts], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.inference_mode():
            h = model(**e).last_hidden_state.float()
        m = e["attention_mask"].unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1).numpy()

    rows = [json.loads(l) for l in (B2 / "pages.jsonl").read_text().splitlines()]
    texts = {r["key"]: r["text"] for r in rows if r["text"]}
    old_keys = json.loads((B2 / "emb_keys.json").read_text()) if (B2 / "emb_keys.json").exists() else []
    old = np.load(B2 / "emb.npy") if old_keys else None
    new = sorted((k for k in texts if k not in set(old_keys)), key=lambda k: len(texts[k]))
    t0 = time.perf_counter(); vs = []
    for i in range(0, len(new), 16):
        vs.append(enc([texts[k] for k in new[i:i + 16]]))
        if i % 640 == 0: print(i, len(new), round(time.perf_counter() - t0, 1), flush=True)
    bulk = time.perf_counter() - t0
    keys = old_keys + new
    arr = np.concatenate(([old] if old is not None else []) + vs) if (vs or old is not None) else np.zeros((0, 1))
    np.save(B2 / "emb.npy", arr); (B2 / "emb_keys.json").write_text(json.dumps(keys))
    if not (B2 / "latency.json").exists() or os.environ.get("LATENCY"):
        sample = random.Random(448).sample(sorted(texts), min(200, len(texts)))
        enc([texts[sample[0]]]); lat = []
        W = np.random.default_rng(0).standard_normal(arr.shape[1] * 2 + 3)  # logistic-regression cost stand-in
        for k in sample:
            t = time.perf_counter(); v = enc([texts[k]])[0]; _ = 1 / (1 + np.exp(-float(np.concatenate([v, v, [0, 0, 0]]) @ W)))
            lat.append(1000 * (time.perf_counter() - t))
        lat.sort()
        s = {"device": "cpu fp32", "threads": torch.get_num_threads(), "max_len": MAX_LEN, "n": len(lat),
             "p50_ms": round(lat[len(lat) // 2], 1), "p95_ms": round(lat[int(len(lat) * 0.95)], 1), "max_ms": round(lat[-1], 1),
             "bulk_pages": len(new), "bulk_ms_per_page": round(1000 * bulk / max(len(new), 1), 1),
             "note": "single page, batch 1, embed + classify; measured while MiniCPM-OCR ran on MPS"}
        (B2 / "latency.json").write_text(json.dumps(s, indent=1)); print(json.dumps(s))
    print("done", len(keys), "keys")


if __name__ == "__main__":
    main()
