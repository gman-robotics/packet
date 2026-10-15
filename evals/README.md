# Packet evals

No third-party PDFs live in git. The harness generates tiny built-in packets
and can plan DocLayNet concatenations from a local catalog.

## Layers

| Layer | Source | Reports as | Train? |
|---|---|---|---|
| built-in | generated PDFs (`text-layer`, `multi-doc`, `shared-header`, `scan`) | `packet-builtin` | no |
| develop | DocLayNet train/val pages | layout F1 (internal) | yes (train only) |
| synth packets | concatenate DocLayNet **documents** from val | `packet-dlnet-synth` | no |
| whole-doc packets | whole public PDFs named by DocLayNet, joined | `packet-dlnet-whole` | no |
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
packet eval --dir /tmp/pkt
```

Whole-document packets (`packet-dlnet-whole`, GMA-216):

```bash
packet eval --set dlnet-whole                     # gate: macro F1 >= 0.6, no false-OCR
packet eval --set dlnet-whole --data-dir /data/dlnet-whole --json
```

DocLayNet only ships sampled pages, so page numbers never restart in joined
samples. `evals/dlnet_whole.json` resolves DocLayNet `scientific_articles`
document names (from the DocLayNet-small sample) to the whole arXiv PDF whose
page count equals DocLayNet's `num_pages`, pinned by URL and sha256. The first
run downloads them (35 PDFs, ~20 MB) to `~/.cache/packet/dlnet-whole`; the
packets are then joined with a seeded plan (13 packets, 2-4 docs each).
Nothing third-party is committed. Licenses: DocLayNet labels are
CDLA-Permissive-1.0; each PDF stays under its arXiv license (recorded per doc
in the manifest; all are arXiv nonexclusive-distrib 1.0), which is why the
files are fetched, never vendored. Limitation: one category (scientific
articles); other DocLayNet categories have no stable public source URL yet.

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
