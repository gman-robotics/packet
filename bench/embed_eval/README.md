# Exp B (GMA-443): EmbeddingGemma 2 retrieval and the embed_shift boundary signal

The model is `google/embeddinggemma-2` @ `26711154b641697c443124e316f3d21587f9ff5d` (Apache-2.0 per the card; not gated).
It runs on torch MPS in the sandbox venv with `HF_HUB_OFFLINE=1`. This harness does not change any Packet default:
`PipelineConfig.embedder` and `BoundaryDetector.embedder` are `None` unless it passes one.

```
PY=../packet/.venv/bin/python; SB=/Users/Shared/gman-sandbox/lfm3-mps/venv/bin/python   # P3 = ~/.cache/packet/model-eval/p3
PYTHONPATH=src $PY bench/embed_eval/exp_b.py dump                  # page text per set -> P3/pages.jsonl
HF_HUB_OFFLINE=1 $SB bench/embed_eval/embed_pages.py               # -> P3/emb_{doc,query}.npy, emb_keys.json, embed_stats.json
PYTHONPATH=src $PY bench/embed_eval/exp_b.py retrieval             # -> P3/retrieval.json
PYTHONPATH=src $PY bench/embed_eval/exp_b.py tune                  # politax-train only -> P3/tune.json
PYTHONPATH=src $PY bench/embed_eval/exp_b.py boundary politax-holdout <T>   # reporting only
```

- **Retrieval task.** No query set exists, so each page with text is a query (`SearchQuery` prompt). The corpus is the
  other pages of the same packet (`Document` prompt), and a page is relevant when it belongs to the same gold document.
  Metrics are NDCG@5 (binary relevance), MRR and Recall@10. The lexical baseline is `packet.topic.topic_similarity`
  over content words.
- **Signal.** `embed_shift` fires when the cosine between the current page's Document embedding and the mean of the previous two
  pages' embeddings is at or below T. Its weight is 0.15, which is below PROPOSE, so it never proposes alone. It counts as
  one independent vote, and it shares that vote with topic_shift and language_changed. `CAP_WITHOUT_RESET = 0.74` and look-ahead are unchanged.
- **Tuning.** T is tuned on PoliTax train (trump) only: maximize macro F1 subject to no more false splits than the v0 detector.
  The PoliTax holdout, dlnet-whole and builtin are reporting only. dlnet-whole has no frozen split, so it is never tuned on.
  The false-OCR hard-fail gate stays on (`run_fixture`).
