# Packet design

Generic document language understanding. Domain knowledge is an adapter,
not a type in the core.

## Why this exists

Open converters (Docling, Marker, MinerU, Firecrawl pdf-inspector, PaddleOCR-VL)
turn a page into markdown or layout blocks. They assume one file is one
document, and they usually want the whole file before they speak.

Packet treats an input file as a stream of unknown length:

1. Classify each page (text / scan / mixed) without rendering.
2. Extract only what that page needs, into a shared PageIR (`Block`).
3. Propose logical-document boundaries from layout signals, not domain words.
4. Assemble a section tree as pages arrive.
5. Hand a closed `LogicalDocument` to any number of domain adapters.

## Objects

| Object | Meaning |
|---|---|
| Packet | One input artifact (PDF, image batch) |
| LogicalDocument | One work product inside the packet |
| Section | Heading-driven node in that work product |
| Block | Leaf: paragraph, table, figure, list, furniture, ... |
| Event | Append-only log the tree is projected from |

No `Will`, `Invoice`, or `Paper` type lives here. A domain adapter may set
`LogicalDocument.type_hint` and write structured fields into `extras`.

## Events

```
page_classified
block_emitted
boundary_proposed
document_opened
section_opened / section_closed
document_closed
packet_closed
```

Subscribers (indexers, agents, wiki sinks) consume events. They never wait
for page N of N unless they opt in.

## Classification and routing

| Page class | Default backend |
|---|---|
| text | native operators (pdf-inspector or pypdfium2 text) |
| scan | layout + OCR/VLM (PaddleOCR-VL, Marker, ...) |
| mixed | native where text exists, OCR on image regions |
| suspect_encoding | treat as scan |

Never send a trustworthy text layer through a GPU.

## Boundary signals (domain-free)

- Printed page-number reset (`1 of N` after a high N)
- Header / footer fingerprint change
- Media box or rotation jump
- Title-like block on a non-first page of the current document
- Optional VLM confirm on a 3-page window when score is mid-band

A proposed boundary is not committed until look-ahead pages agree or the
stream ends. That keeps a chapter title from splitting a document.

## Assembler

Detect → order → construct.

- Furniture (headers, footers, page numbers) is stripped from the body.
- Title / heading blocks open sections.
- Numbering grammar (`1.`, `1.1`, `Article IV`, `Appendix`) sets level.
- Captions bind to the nearest table or figure, including across a page break.
- The tree can be repaired: reparent when a later heading clarifies rank.

## Domain adapters

```python
class DomainAdapter(Protocol):
    name: str
    def applies(self, doc: LogicalDocument) -> bool: ...
    def enrich(self, doc: LogicalDocument) -> LogicalDocument: ...
```

Adapters run only on `document_closed`. They must not delete sections or
blocks. Add `type_hint` and `extras`. Examples that do *not* belong in core:

- estate packet field extraction
- invoice line items
- paper abstract / citation graph
- medical intake forms

## Backend adapters

Same rule. Docling, pdf-inspector, PaddleOCR-VL, Marker, Firecrawl `/parse`
each implement `Extractor.extract(ref, text) -> list[Block]`. The PacketTree
schema does not change when a backend is swapped.

## Eval (generic)

Always slice by page condition: born-digital / clean scan / dirty scan / handwriting.

1. Packet F1 — logical document page ranges
2. Heading-path accuracy — parentage of sections
3. Reading-order quality — columns, lists
4. Citation integrity — every extras field points at a Block id

Build the gold set from the domains you will actually run, but score the
core metrics without domain labels.

## What v0 implements

- Schema and events
- Lazy PDF page source
- Heuristic classifier
- Native text extractor
- Boundary detector (page numbers, headers, media box, titles)
- Section assembler
- Domain adapter protocol
- CLI: `packet inspect`, `packet parse`

## Next

1. pdf-inspector classifier adapter
2. Docling PageIR adapter
3. Scan backend (PaddleOCR-VL) on routed pages only
4. Look-ahead commit for boundaries
5. Caption / table cross-page binding
6. Event-log persistence and replay
