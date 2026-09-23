# Packet design

Generic document language understanding. Domain knowledge is an adapter,
not a type in the core.

**Naming note:** "Packet" here is this repository — a document
language-understanding pipeline (`packet-dl` on PyPI, `packet` CLI). It is
unrelated to "evidence-packet-protocol," which is an agent-workflow skill
name used elsewhere. If you arrived here looking for an agent
skill/protocol, this is not it.

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

## Escalation policy: Heuristic → pdf-inspector/VLM

This is the policy for deciding, per page, which backend gets to speak.
It has a **shipped v0** half and a **planned** half. They are documented
separately below so the table never has to stand in for behavior the code
does not have.

### Target design (full policy, not all shipped yet)

| Page class | Backend once fully wired |
|---|---|
| text | native operators (pdf-inspector or pypdfium2 text) |
| scan | layout + OCR/VLM (PaddleOCR-VL, Marker, ...) |
| mixed | native where text exists, OCR on image regions |
| suspect_encoding | treat as scan |

Never send a trustworthy text layer through a GPU. That rule holds today
and drives the fail-closed behavior below.

### What v0 actually ships

- **Classifier**: `HeuristicClassifier` (`src/packet/classify.py`) is a
  char-count threshold, nothing more.
  - `chars >= min_chars` (default `min_chars=40`) → `PageClass.TEXT`,
    confidence `0.7`.
  - `0 < chars < min_chars` → `PageClass.MIXED`, confidence `0.45`.
  - `chars == 0` → `PageClass.SCAN`, confidence `0.6`.
  - These three numbers (`0.7` / `0.45` / `0.6`) are the only "confidence
    thresholds" v0 has. There is no calibration, no held-out validation,
    and no per-domain tuning behind them yet.
- **`InspectorClassifier`** exists as a class but is a stub: if
  `pdf-inspector` is importable it silently delegates to
  `HeuristicClassifier().classify(...)` and returns the same char-count
  result. It does not call pdf-inspector for layout/classification. Treat
  it as a placeholder seam, not a working integration.
- **Mid-band VLM confirm window**: **not implemented.** The 3-page VLM
  confirmation window described for boundaries (see below) and any
  VLM-based reclassification of mid-confidence pages are both future
  work. No VLM classifier exists in this codebase today.
- **`--force-inspect` CLI flag**: **not implemented.** There is no way
  today to force a page or file through pdf-inspector or a VLM from the
  CLI (`packet inspect` / `packet parse` only expose `--no-split` and
  `-o/--output`). This flag is planned; see Next.
- **Router and fail-closed behavior**: `Pipeline` builds its `Router`
  with `[NativeTextExtractor()]` by default (`src/packet/pipeline.py`).
  `NativeTextExtractor.supports()` only claims `TEXT`, `MIXED`, and
  `UNKNOWN` pages. A page classified `SCAN` matches no extractor in the
  default router, so `Router.extract()` falls through its loop and
  returns `[]` — **zero blocks, silently, no error, no OCR fallback.**
  This is intentionally fail-closed with respect to the "never OCR a
  trustworthy text layer" rule (a `SCAN` page has no text layer to
  protect), but it also means a v0 default `Pipeline` run over a
  scanned document produces an empty (or near-empty) `LogicalDocument`
  today, not an OCR'd one. `PaddleExtractor` and `DoclingExtractor`
  (`src/packet/backends.py`) exist as seams and both raise
  `NotImplementedError` if called — they are not wired into the default
  `Router`.

In short: today's real routing table is `{TEXT, MIXED, UNKNOWN} →
NativeTextExtractor`, `SCAN → nothing (empty result)`. The class→backend
table above is the target, not the current behavior.

## Boundary policy: v0 is eager-commit (not look-ahead)

Signals (domain-free, `src/packet/boundary.py`, `BoundaryDetector.observe`):

- Printed page-number reset (`1 of N` after a high N) — score `+0.45`
- Header / footer fingerprint change — score `+0.25`
- Media box or rotation jump — score `+0.2`
- Title-like block on a non-first page of the current document — score `+0.25`

Scores add. `observe()` emits a `BoundarySignal` (as the `boundary_proposed`
event) when the combined score is `>= 0.4`; the reported confidence is
`min(score, 0.95)`.

**What v0 actually does with that signal is eager commit, not look-ahead
deferral.** In `Pipeline.stream` (`src/packet/pipeline.py`):

```python
should_split = (
    self.config.split_on_boundary
    and signal is not None
    and signal.confidence >= self.config.boundary_min_confidence  # 0.55
    and current_blocks
)
if should_split:
    doc = self._close_document(...)   # closes and appends the LogicalDocument now
```

The moment a page's signal clears `boundary_min_confidence` (default
`0.55`), the pipeline calls `_close_document` for the *current* page —
there is no waiting for subsequent pages to agree, no re-open, and no
retraction. `boundary_proposed` and `document_closed` happen on the same
iteration of the page loop. An earlier draft of this document described a
look-ahead-confirmed commit; that behavior does not exist in v0 and the
claim has been removed. Look-ahead commit is tracked as future work
(see Next).

### Accepted v0 fail modes (documented, not silently swallowed)

Because commit is eager and irrevocable:

- **False split on a mid-document title-like block.** A chapter title,
  pull-quote, or all-caps run-in heading on page *k* can score `>= 0.4`
  purely from the `title_block` signal (`+0.25`) plus one other cue
  (e.g. a header change from a running head), closing the document one
  page early. There is no repair pass that can re-merge documents once
  split — `Assembler` repair (reparenting headings) only applies inside
  a single already-open document's section tree, never across a boundary
  commit.
- **False merge when signals under-fire.** If a new physical document in
  the stream restarts its own page numbering as `1 of N` but shares a
  near-identical header/footer with the previous document (or the PDF
  producer stripped `N of M` footers entirely), score can stay below
  `0.4` and pages silently accrete into the prior `LogicalDocument`.
  This is the `shared-header` fixture case in `docs/EVAL.md`.
- **Flush-at-EOF only.** Whatever blocks have accumulated in
  `current_blocks` when the page stream ends are closed as one final
  `LogicalDocument`, regardless of whether a boundary signal ever fired
  for them (`if current_blocks or not packet.documents:` at the end of
  `stream()`). There is no minimum-confidence check on this final flush
  — an under-confident tail is still emitted as a document rather than
  dropped or flagged.
- **No undo.** Once `_close_document` runs, the resulting
  `LogicalDocument` is appended to `packet.documents` and the event is
  yielded. A later page can never cause an already-closed document to be
  reopened, split further, or merged with the next one.

`--no-split` (CLI) / `PipelineConfig.split_on_boundary = False` disables
all of the above and forces the whole packet into one `LogicalDocument`,
which is useful as an escape hatch while boundary quality is unproven.

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

Naming these metrics is not the same as defining them well enough for QA to
implement a gate. Full definitions — how a boundary match is scored (P/R),
the false-OCR hard-fail condition, and the named fixture pack
(`multi-doc` / `shared-header` / `scan` / `text-layer`) — live in
[`docs/EVAL.md`](docs/EVAL.md). Build the gold set from the domains you will
actually run, but score the core metrics without domain labels.

## Licensing

Packet's own code is MIT (see `LICENSE`), but the backends it names
(Docling's bundled models, Marker's model weights, PaddleOCR-VL,
Firecrawl's `pdf-inspector`) each carry their own license terms that do
not automatically inherit MIT just because Packet's glue code does. A
soft mention here is not a substitute for a checklist that has to be
walked before any of those backends ship bundled model weights.
The hard checklist lives in
[`docs/LICENSE_CHECKLIST.md`](docs/LICENSE_CHECKLIST.md).

## What v0 implements

- Schema and events
- Lazy PDF page source
- `HeuristicClassifier` (char-count only — see Escalation policy above)
- `InspectorClassifier` seam that currently delegates to
  `HeuristicClassifier` (not a real pdf-inspector integration)
- `NativeTextExtractor`, routed to `TEXT` / `MIXED` / `UNKNOWN` pages only
- Fail-closed (empty-result, not OCR fallback) for `SCAN` pages under the
  default `Router`
- `BoundaryDetector` scoring (page numbers, headers, media box, titles)
  feeding an **eager-commit** split in `Pipeline` (no look-ahead — see
  Boundary policy above)
- Section assembler with intra-document heading repair
- Domain adapter protocol
- CLI: `packet inspect`, `packet parse` (`--no-split`, `-o/--output`); no
  `--force-inspect` yet

## Next

1. Real pdf-inspector classifier adapter (replace the `InspectorClassifier`
   stub with an actual pdf-inspector call)
2. Docling PageIR adapter (wire `DoclingExtractor`, currently
   `NotImplementedError`)
3. Scan backend (PaddleOCR-VL) on routed pages only (wire
   `PaddleExtractor`, currently `NotImplementedError`), so `SCAN` pages
   stop returning empty results
4. Mid-band VLM confirm window (3-page) for both classification and
   boundary confirmation
5. `--force-inspect` CLI flag
6. **Look-ahead commit for boundaries** — defer `_close_document` until
   subsequent pages agree or EOF, replacing today's eager commit. This is
   deferred deliberately: it is a design decision (propose vs. commit)
   that needs an explicit sign-off before it is implemented, not a small
   follow-on patch. See the note at the top of the PR that introduced this
   revision for the open question.
7. Caption / table cross-page binding
8. Event-log persistence and replay
