"""GMA-443: embed P3/pages.jsonl texts with google/embeddinggemma-2 @26711154 on torch MPS (sandbox venv, HF_HUB_OFFLINE=1).

Text-only path of the card's sentence-transformers stack, without sentence-transformers: modules.json is
Transformer -> mean Pooling (include_prompt) -> Normalize, so this does masked mean over last_hidden_state then L2.
Prompts from config_sentence_transformers.json: "Document" = "title: none | text: ", "SearchQuery" = "task: search result | query: ".
Writes P3/emb_doc.npy, P3/emb_query.npy, P3/emb_keys.json, P3/embed_stats.json. `sanity` checks the card example ordering."""
import json
import os
import sys
import time
from pathlib import Path

REPO, REV = "google/embeddinggemma-2", "26711154b641697c443124e316f3d21587f9ff5d"
SANDBOX = "/Users/Shared/gman-sandbox/lfm3-mps/venv"
DOC, QUERY = "title: none | text: ", "task: search result | query: "
P3 = Path(os.environ.get("P3", Path.home() / ".cache/packet/model-eval/p3"))


class Embedder:
    def __init__(self):
        if os.path.realpath(sys.prefix) != os.path.realpath(SANDBOX) or os.environ.get("HF_HUB_OFFLINE") != "1":
            sys.exit(f"run from {SANDBOX} with HF_HUB_OFFLINE=1")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch; t = time.time()
        snap = snapshot_download(REPO, revision=REV, local_files_only=True)
        cfg = json.loads(Path(snap, "1_Pooling/config.json").read_text())
        assert cfg["pooling_mode"] == "mean" and cfg["include_prompt"], cfg
        self.tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
        self.model = AutoModel.from_pretrained(snap, local_files_only=True, dtype=torch.bfloat16).to("mps").eval()
        torch.mps.synchronize(); self.load_s = time.time() - t

    def encode(self, texts, prefix, bs=8, max_len=2048):
        torch = self.torch; out = []
        for i in range(0, len(texts), bs):
            enc = self.tok([prefix + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("mps")
            with torch.inference_mode():
                h = self.model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu()); del h, enc, m, v
            torch.mps.empty_cache()  # long tax pages fragment the MPS pool (OOM'd the first query pass)
        return torch.cat(out).numpy()


def sanity():
    e = Embedder(); q = e.encode(["What causes the northern lights?"], QUERY)
    d = e.encode(["The northern lights are caused by charged particles from the sun.", "Quarterly sales rose four percent in the Midwest region."], DOC)
    print(json.dumps({"load_s": round(e.load_s, 2), "cos_relevant": float(q[0] @ d[0]), "cos_irrelevant": float(q[0] @ d[1])}))


def main():
    import numpy as np
    rows = [json.loads(l) for l in (P3 / "pages.jsonl").read_text().splitlines()]
    texts = {}
    for r in rows:
        if r["key"]: texts[r["key"]] = r["text"]
    keys = sorted(texts, key=lambda k: len(texts[k])); e = Embedder(); stats = {"load_s": round(e.load_s, 2), "pages": len(keys)}
    old = json.loads((P3 / "embed_stats.json").read_text()) if (P3 / "embed_stats.json").exists() else {}
    for name, pre in (("doc", DOC), ("query", QUERY)):
        if (P3 / f"emb_{name}.npy").exists() and name in old:  # resume: keep a finished pass (same sorted keys)
            stats[name] = old[name]; continue
        t = time.perf_counter(); v = e.encode([texts[k] for k in keys], pre); dt = time.perf_counter() - t
        np.save(P3 / f"emb_{name}.npy", v); stats[name] = {"wall_s": round(dt, 1), "ms_per_page": round(1000 * dt / len(keys), 1)}
        print(name, stats[name], flush=True); (P3 / "embed_stats.json").write_text(json.dumps(stats, indent=1))
    t = time.perf_counter(); e.encode([texts[keys[len(keys) // 2]]], QUERY, bs=1); stats["single_query_ms_median_len"] = round(1000 * (time.perf_counter() - t), 1)
    stats["mps_driver_peak_gb"] = round(e.torch.mps.driver_allocated_memory() / 1e9, 2)
    (P3 / "emb_keys.json").write_text(json.dumps(keys)); (P3 / "embed_stats.json").write_text(json.dumps(stats, indent=1)); print(json.dumps(stats))


if __name__ == "__main__":
    sanity() if sys.argv[1:] == ["sanity"] else main()
