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
packet eval --json            # {"fixtures": [...], "gate": {"passed", "failures", ...}}
packet eval --min-f1 0.9      # per-fixture boundary F1 floor (default 1.0)
packet eval --min-macro-f1 0.95
```

`packet eval` is a gate: it exits 1 (and prints `GATE FAIL` with reasons) if any
fixture's boundary F1 (tolerance 0) is below `--min-f1`, if the macro-average F1 is
below `--min-macro-f1` (defaults to `--min-f1`), or if any page with a text layer
of at least 40 characters emitted OCR/VLM blocks (false-OCR hard-fail). Otherwise it
prints `GATE PASS` and exits 0.

```bash
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
