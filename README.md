# Packet

Streaming document language understanding for files of unknown length.

Repo: [gman-robotics/packet](https://github.com/gman-robotics/packet)  
Name lock (2026-09-22): product **Packet**, package **packet-dl**, CLI **packet**.

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
text extractor, and a single-document assembler. Multi-document boundary
detection and backend adapters (Docling, pdf-inspector, PaddleOCR-VL) come
next.

## Install

```bash
pip install -e ".[dev]"
packet inspect path/to/file.pdf
packet parse  path/to/file.pdf -o packet.json
```

## Design rules

1. Never OCR a page with a trustworthy text layer.
2. Every node carries page, bbox when known, source, and confidence.
3. Core types stay domain-agnostic.
4. A slow or missing backend is not a reason to change the tree schema.
