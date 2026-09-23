# Packet

Streaming document language understanding for files of unknown length.

Repo: [gman-robotics/packet](https://github.com/gman-robotics/packet)  
Name lock (2026-09-22): product **Packet**, package **packet-dl**, CLI **packet**.

**Not to be confused with:** "evidence-packet-protocol" is a separate,
unrelated agent-workflow skill name. This repository — product **Packet**,
package `packet-dl` — is a document language-understanding pipeline, not
an agent protocol. If a link brought you here looking for the protocol,
this is the wrong project.

A PDF (or a folder of page images) is a **packet**, not a document. Packet
classifies each page, extracts only what that page needs, detects logical
document boundaries as pages arrive, and assembles a semantic tree.

Domain extractors (legal, medical, financial, scientific) are adapters that
run *after* a `LogicalDocument` is closed. The core pipeline has no domain
vocabulary.

## Tree

```
Packet                         # one input file or scan batch
  LogicalDocument              # one work product inside the packet
    Section                    # heading-driven hierarchy
      Block                    # paragraph, table, figure, list, ...
```

Furniture (running headers, footers, page numbers) lives beside the body,
not inside it.

## Stream

Pages are the unit of I/O. Documents are the unit of meaning.

```
PageClassified
BlockEmitted
BoundaryProposed
DocumentOpened
SectionOpened / SectionClosed
DocumentClosed
PacketClosed
```

The tree is a materialized view over an append-only event log. Later pages
may reparent earlier blocks when heading levels or boundaries become clear.

## Pipeline

```
source        lazy page iterator (no full render)
classifier    text | scan | mixed | suspect_encoding
backends      native extract and/or layout+OCR/VLM → PageIR
boundary      page-number reset, header fingerprint, title-like blocks,
              media-box jump, optional VLM confirm
assembler     reading order → heading grammar → PacketTree
domain        optional adapters on DocumentClosed
```

Backends are swappable. The product is the tree, the router, and the event
contract — not a particular model.

## Status

v0 is the schema, event contract, page source, heuristic classifier, native
text extractor, and a section assembler. Boundary splitting is **on by
default** (`split_on_boundary=True` in `PipelineConfig`) — a default run
already produces multiple `LogicalDocument`s per packet whenever the
built-in signals (page-number reset, header change, media-box jump,
title-like block) cross a confidence threshold. Pass `--no-split` on the
CLI (or set `split_on_boundary=False`) to force single-document behavior
instead.

That splitting is **eager-commit, not look-ahead**: a document closes the
moment its boundary signal clears threshold, with no confirmation from
later pages. See `DESIGN.md` for the accepted fail modes that follow from
that (false splits on mid-document titles, false merges on shared headers,
flush-at-EOF for the tail). Real backend adapters (Docling, pdf-inspector,
PaddleOCR-VL) are stubs today — see `DESIGN.md`'s Escalation policy
section — so scanned (`SCAN`-classified) pages currently come back empty
rather than OCR'd.

## Install

```bash
pip install -e ".[dev]"
packet inspect path/to/file.pdf
packet parse  path/to/file.pdf -o packet.json
packet eval                        # run built-in fixtures (boundary P/R, false-OCR gate)
pytest
```

See [`docs/EVAL.md`](docs/EVAL.md) and [`evals/README.md`](evals/README.md)
for what `packet eval` checks and how to point it at DocLayNet.

## Design rules

1. Never OCR a page with a trustworthy text layer.
2. Every node carries page, bbox when known, source, and confidence.
3. Core types stay domain-agnostic.
4. A slow or missing backend is not a reason to change the tree schema.
