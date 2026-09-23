# Packet evals

No third-party PDFs live in git. The harness generates tiny built-in packets
and can plan DocLayNet concatenations from a local catalog.

## Layers

| Layer | Source | Reports as | Train? |
|---|---|---|---|
| built-in | generated PDFs (`text-layer`, `multi-doc`, `shared-header`, `scan`) | `packet-builtin` | no |
| develop | DocLayNet train/val pages | layout F1 (internal) | yes (train only) |
| synth packets | concatenate DocLayNet **documents** from val | `packet-dlnet-synth` | no |
| frozen | OmniDocBench, olmOCR-bench | public parse scores | **never** |

## Commands

```bash
packet eval                  # built-in fixtures
packet eval --json
PACKET_EVAL_DIR=/tmp/pkt packet eval
```

DocLayNet extras (optional):

```bash
# JSONL catalog, one row per page:
# {"split":"val","doc_name":"a.pdf","category":"patents","page_no":1,"pdf_path":"/data/..."}
packet eval-plan --catalog /data/doclaynet/catalog.jsonl --slice poly-seq
```

Placeholder PDFs are written so the planner is testable without the 28 GiB extra zip.
Replace placeholders with real concatenated extras before quoting `packet-dlnet-synth`.

## Metrics

See `docs/EVAL.md`. Built-in run scores **boundary P/R/F1** (`tolerance=0`) and the
**false-OCR hard-fail** gate.
