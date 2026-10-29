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
- **`--force-inspect` CLI flag**: `packet inspect` / `packet parse
  --force-inspect` sends every page, text pages included, through OCR.
- **OCR and the escalate policy** (`src/packet/backends.py`): the CLI
  builds its router from `default_extractors()`, one
  `EscalatingExtractor`:
  - `TEXT` / `UNKNOWN`: native text layer only (never OCR a trustworthy
    text layer).
  - `SCAN` / `SUSPECT_ENCODING`: OCR on the rendered page image
    (`PdfPageSource.render`, longest side 1540 px, rendered lazily only
    when an OCR engine runs).
  - `MIXED`: OCR; if OCR is unavailable or finds nothing, keep the native
    text so the page is never dropped.
  - OCR engine: `LightOnOCR-3-1B` in grounding mode on MLX (default),
    `PaddleOCR-VL-1.6` page pipeline as fallback when LightOn cannot run
    or crashes (per `bench/ocr_compare/REPORT.md`). Weights download
    lazily at first use (LightOn pinned to a revision), never bundled; see
    `docs/LICENSE_CHECKLIST.md`.
  - If neither engine can run, `packet` prints one warning to stderr and
    scanned pages come back empty; `PACKET_OCR=off` disables OCR.
  - The Python `Pipeline()` default is still `[NativeTextExtractor()]`
    (no model loads); pass `extractors=default_extractors()` to get OCR.

- **Furniture tagging** (`src/packet/furniture.py`, `FurnitureTagger`):
  `Pipeline` runs every native page through it. The page's first and last
  lines become their own blocks; a last line that looks like a page number
  (`Page 3 of 9`, `3/9`, `- 3 -`, `3`) is `PAGE_NUMBER`; a top/bottom line
  repeats on another page within 3 pages is `HEADER` (exact text) or
  `FOOTER` (digit-normalized; also the line above a page number). Lines are
  split into their own block only when tagged. Page 0 is re-tagged when its
  document closes, so that shows in `document_closed` and the final
  Packet but not in the earlier `block_emitted` events. Bare numbers count
  as page numbers only when plausible (not a year, not far past the page). OCR blocks keep the
  engine's labels. `Assembler._split_furniture` then moves these into
  `LogicalDocument.furniture`, so section bodies no longer contain running
  headers or page numbers. The boundary detector adds `sequence_completed`
  (+0.2) when a page-number reset follows an "N of N" page, so a shared
  binder header no longer hides a page-number restart.

In short: today's real routing table is `{TEXT, MIXED, UNKNOWN} →
NativeTextExtractor`, `SCAN → nothing (empty result)`. The class→backend
table above is the target, not the current behavior.

## Boundary policy: look-ahead commit

Signals (domain-free, `src/packet/boundary.py`, `BoundaryDetector.observe`):

- Printed page-number reset (`1 of N`, or a tagged bare `1`, after a higher
  number) — score `+0.45` (`+0.2` more when the previous page was `N of N`)
- Header / footer fingerprint change — score `+0.25`
- Media box or rotation jump — score `+0.2`
- Title that opens the page (first non-furniture block) — score `+0.6`
  (`title_top`); a title lower on the page — `+0.25` (`title_block`)
- Two or more independent signals together are lifted to at least `0.6`
  (GMA-216), so weak signals can combine into a split; look-ahead page-number
  evidence can still veto them.

Weights live as named constants at the top of `boundary.py` and are tuned
against `packet eval --set dlnet-whole` (see `evals/README.md`).

The header/footer signal only compares `BlockKind.HEADER` blocks; when a
page has none — true today, since `NativeTextExtractor` never emits
`HEADER` (see What v0 implements below) — it falls back to comparing the
leading block's text instead. That fallback is a coarse page-content diff,
not a real header/footer fingerprint, so it can fire on ordinary body-text
changes between pages of the *same* document, not only on an actual
template change. See `docs/EVAL.md`'s `shared-header` fixture note for a
concrete case where this makes a fixture pass for a different reason than
the one it names.

Scores add. `observe()` emits a `BoundarySignal` (as the `boundary_proposed`
event) when the combined score is `>= 0.4`; the reported confidence is
`min(score, 0.95)`.

**Commit is look-ahead, not eager** (`src/packet/lookahead.py`,
`LookaheadSplitter`, driven by `Pipeline.stream`):

- confidence `>= boundary_commit_confidence` (default `0.75`, never below
  `boundary_min_confidence`): split as soon as every earlier page is decided;
- `0.4 <= confidence < 0.75`: wait for up to `lookahead_window` (default
  `2`) following pages, then decide on page-number evidence:
  - the page itself continues the current document's numbering (same
    total): no split (`continues_sequence`);
  - a following page continues the new numbering (`num(p) + k`, same
    total when printed): split (`lookahead_numbering`);
  - a following page resumes the current document's numbering (the last
    accepted numbered page, not a rejected insert; same total; within it):
    no split, the page was an insert (`old_sequence_continues`);
  - no evidence in the window or end of file: the old eager rule,
    split iff confidence `>= boundary_min_confidence` (`0.55`).
- An **unnumbered** page, with or without a signal, also waits for the
  window: it splits (`implied_page_one`) when the first later numbered page
  in the window puts it at page 1 (`num(q) - k == 1`) and that page does not
  continue the current document's numbering. Many documents leave page 1
  unnumbered, so this is the most common page-number reset in practice.
  Cost: `document_opened` for unnumbered pages now lags by up to
  `lookahead_window` pages even without a signal.
- `lookahead_window=0` reproduces v0 eager commit exactly.

Streaming: `page_classified`, `block_emitted` and `boundary_proposed`
are still emitted as each page is read. `document_opened` /
`document_closed` for a page can lag by up to `lookahead_window` pages
while a mid-confidence boundary is pending, so `block_emitted` events for
the next pages can arrive **before** the `document_opened` of the document
they end up in. Consumers should assign blocks to documents from
`document_closed` (or the final Packet), not from event order. Decisions
are applied strictly in page order; each document is opened exactly once.

### Remaining fail modes (documented, not silently swallowed)

Look-ahead only uses page-number evidence, and a committed split is still
never re-merged, so these remain:

- **False split on a mid-document title-like block.** A chapter title,
  pull-quote, or all-caps run-in heading on page *k* can score `>= 0.4`
  purely from the `title_block` signal (`+0.25`) plus one other cue
  (e.g. a header change from a running head), closing the document one
  page early. Since GMA-216 a TITLE that *opens* a page (`title_top`,
  `+0.6`) is enough on its own unless look-ahead numbering vetoes it, so
  OCR engines that label chapter openers as TITLE are the main exposure
  (native text only labels page 0 as TITLE today). There is no repair pass that can re-merge documents once
  split. `repair_ranks` reparents numbered headings only while that one
  document is built, never across a boundary commit and never inside a
  document already emitted.
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
  `stream()`). There is no minimum-confidence check on this final flush.
  An under-confident tail is still emitted as a document. When that tail
  is the final document of a multi-document packet and it has no
  page-number support, core records `eof_flush` in `packet.flags`
  (see Uncertainty flags). It does not drop the document.
- **No undo.** Once `_close_document` runs, the resulting
  `LogicalDocument` is appended to `packet.documents` and the event is
  yielded. A later page can never cause an already-closed document to be
  reopened, split further, or merged with the next one.

`--no-split` (CLI) / `PipelineConfig.split_on_boundary = False` disables
all of the above and forces the whole packet into one `LogicalDocument`,
which is useful as an escape hatch while boundary quality is unproven.

## Uncertainty flags

Core writes `LogicalDocument.extras["packet.flags"]`, a list of `Flag`
dicts (`src/packet/types.py`). The key is reserved. Domain adapters must
not overwrite it; `apply_domains` restores a value that was already there
and drops one an adapter inserts when the key was absent. The assembler
writes `heading_rank_repaired` into that key when it reparents a section,
and it does not add the key when it has no such flags. `apply_domains`
keeps that list. The pipeline appends its own flags after adapters run,
so assembler flags come first and citation extras are visible when the
later flags are chosen. A flag never changes a split or a section. The
repair moves the section; the flag only records it.

- **`weak_boundary`** — the split that opened this document had no
  page-number evidence. That is decision reason `fallback_eager`, or
  `high_confidence` whose signal reasons do not include
  `page_number_reset`. Confidence is not a weakness signal: two
  independent signals are raised to `COMBINED` = 0.6
  (`src/packet/boundary.py`, about lines 244-245), and without a
  page-number reset confidence is capped at 0.74. A correct
  `lookahead_numbering` split such as `shared-header` (0.65) is not weak.
- **`fallback_decision`** — a `fallback_below` decision on a page that had
  a boundary signal: a near-miss that did not split. `fallback_eager` is
  `weak_boundary` and is not flagged again.
- **`eof_flush`** — only the final document of a multi-document packet,
  and only when no numbered page in it continues a sequence started in
  that document. A sequence start is any earlier numbered page, or the
  unnumbered opening page. Each later numbered page is checked against
  every one of those anchors, so a numbered insert does not hide an
  earlier run (`1 of 3`, `1 of 1`, `2 of 3` has support). A one-document
  packet never gets it. A one-page final document does not get it when
  `page_num` is 1 (`1 of 1`, with or without a `page_number_reset`). A
  `page_number_reset` reason with no page number is not support. Only an
  unnumbered or otherwise unsupported final page gets the flag.
- **`unresolved_citation`** — a citation id in `extras` that is not a
  block id known when this document closes. The pipeline's known ids are
  the blocks of this document and of earlier documents (streaming), so a
  forward reference to a later document is flagged. `citation_id_counts`
  (`src/packet/citations.py`) returns the total cited ids, counting
  repeats, and the unresolved ids with repeats kept.
  `unresolved_citation_ids` is that second list. Eval
  `citation_integrity` uses the same helper over the whole packet, so
  that forward reference counts as resolved there. `citation_keys` still
  filters extras keys, and the ratio is unchanged (`1.0` when nothing
  is cited).
- **`heading_rank_repaired`** — `repair_ranks` changed this section's
  parent while building its document. `page` is the section's
  `page_start`, `ref` is the section id, and `detail` holds `from_parent`
  (id or null), `to_parent`, and `numbering` (the outline token, such as
  `1.1`). One flag per reparent. A level correction that does not change
  the parent does not emit a flag. The repair stays inside the document
  being built. It does not cross a boundary and does not reopen a
  document already emitted.

## Assembler

Detect → order → construct.

- Furniture (headers, footers, page numbers) is stripped from the body.
- Title / heading blocks open sections. The first pass is a heading-level
  stack (`TITLE` is level 0).
- `repair_ranks` then reads numbering on the heading's first line.
  Decimal outlines are `1`, `1.`, `1.1`, and `1.1.2` (depth is the
  component count). A bare integer above 99 with no dot, such as `2024`
  or `100`, is not an outline. `Section 1.2` is that same decimal.
  `Article IV` and `ARTICLE 4` are depth 1 (roman or arabic) and are not
  decimal prefixes.
- Where that grammar and `heading_level` disagree, grammar sets the
  level. Level 0 sections stay as they are. A numbered title can still
  be a decimal prefix parent.
- A numbered section of depth d is reparented under the nearest preceding
  decimal section of depth d-1 whose number is a prefix, inside this
  document only. With no such parent, it stays. Unnumbered sections stay.
  Each reparent emits one `heading_rank_repaired` and extends the new
  parent's `page_end` to cover the child. There is no cross-document
  repair and no streaming reparent of a document already emitted.
- Captions bind to the nearest table or figure, including across a page break.

## Domain adapters

```python
class DomainAdapter(Protocol):
    name: str
    def applies(self, doc: LogicalDocument) -> bool: ...
    def enrich(self, doc: LogicalDocument) -> LogicalDocument: ...
```

Adapters run only on `document_closed`. The assembler may already have
written `packet.flags` (`heading_rank_repaired`). Adapters must not delete
sections or blocks, and they must not overwrite `extras["packet.flags"]`
(`apply_domains` restores an existing value and drops one an adapter
inserts). The pipeline appends its flags after that, so assembler flags
come first. Add `type_hint` and other `extras`. Examples that do
*not* belong in core:

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
- OCR for `SCAN` / `MIXED` pages in the CLI (LightOnOCR-3-1B grounding on
  MLX, PaddleOCR-VL-1.6 fallback) via the escalate policy
- `BoundaryDetector` scoring (page numbers, headers, media box, titles)
  feeding a **look-ahead commit** split in `Pipeline` (see Boundary policy
  above; `lookahead_window=0` = eager)
- Section assembler: heading stack, then rank repair from numbering
  grammar while that document is built (no cross-document repair, no
  reparent of a document already emitted)
- Domain adapter protocol
- CLI: `packet inspect`, `packet parse` (`--no-split`, `-o/--output`,
  `--force-inspect`)

## Next

1. Real pdf-inspector classifier adapter (replace the `InspectorClassifier`
   stub with an actual pdf-inspector call)
2. Docling PageIR adapter (wire `DoclingExtractor`, currently
   `NotImplementedError`)
3. ~~Scan backend on routed pages only~~ (shipped: LightOnOCR-3-1B default, PaddleOCR-VL-1.6 fallback; wire
   `PaddleExtractor`, currently `NotImplementedError`), so `SCAN` pages
   stop returning empty results
4. Mid-band VLM confirm window (3-page) for both classification and
   boundary confirmation
5. ~~`--force-inspect` CLI flag~~ (shipped)
6. ~~**Look-ahead commit for boundaries**~~ (shipped, signed off by Tom
   2026-10-13; page-number evidence only) — defer `_close_document` until
   subsequent pages agree or EOF, replacing today's eager commit. This is
   deferred deliberately: it is a design decision (propose vs. commit)
   that needs an explicit sign-off before it is implemented, not a small
   follow-on patch. See the note at the top of the PR that introduced this
   revision for the open question.
7. Caption / table cross-page binding
8. Event-log persistence and replay
