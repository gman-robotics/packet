# LangExtract lessons for Packet

Plan only. No `src/` change in this PR. Written against `main` `07c5796`. Source: Tom's five-lesson brief (3:01 PM MT 2026-10-28, via Morgan). The tmg-wiki write-up (`raw/` page, `projects/packet.md` section) had not landed when this was written. Re-check this plan against it when it lands.

Method: poteto-mode architecture review. `/architect` sketches the types. `blast-radius` lists what each change can break. `sequence-verifiable-units` orders the work. `prove-it-works` sets a concrete proof for each item.

## Ground truth today

- Pipeline events: `EventKind` in `src/packet/events.py:11-19`. `document_closed` is yielded at `src/packet/pipeline.py:125` and `:141`. Domain adapters run in `_close_document` at `src/packet/pipeline.py:162`.
- Boundary signal: `BoundarySignal(after_page, confidence, reason)` at `src/packet/boundary.py:110-114`. Cap without a page-number reset is `CAP_WITHOUT_RESET = 0.74` at `src/packet/boundary.py:101`.
- Look-ahead decision: `Decision(page, split, reason)` at `src/packet/lookahead.py:37-41`. Reasons are made in `_drain` (`:76-104`) and `_decide_mid` (`:150-167`). **Gap:** `Pipeline._apply` (`src/packet/pipeline.py:129-150`) consumes `decision.split` and drops `decision.reason`. No event and no output field records why a page split or did not split.
- Output schema: `LogicalDocument.extras: dict[str, Any]` at `src/packet/types.py:95`. `Block.extra` at `:74`. No key in `extras` is reserved or typed today.
- Assembler: `_nest` (`src/packet/assemble.py:54-102`) is a plain level stack. **Gap:** DESIGN.md (Assembler section, "What v0 implements") says the tree "can be repaired: reparent when a later heading clarifies rank". No reparent pass exists in `assemble.py`. A wrong `heading_level` is final and silent.
- Citations: `citation_integrity` exists only in eval (`src/packet/eval/metrics.py:77`). The pipeline never reports an unresolved citation.
- Domain adapters: `DomainAdapter` protocol at `src/packet/domain.py:8`. Only `NoOpDomain` (`:20`) ships.
- Eval: `packet eval` (`src/packet/cli.py:78`, `_cmd_eval` at `:471`) and `run_fixture` (`src/packet/eval/run.py:96`) emit numbers only (`boundary_score`, `false_ocr_hard_fail` in `metrics.py:57`, `:117`). No artifact shows a reviewer what went wrong.
- Fail modes: DESIGN.md "Remaining fail modes": false split on a mid-document title, false merge when signals under-fire, flush-at-EOF with no confidence check, no undo.
- Mid-band VLM confirm window: DESIGN.md "Next" item 4. Not built.

## Item 1: decision trace (prerequisite for items 2, 4 and 5)

The visualizer and the review path need data the pipeline throws away today. Do this first.

- **Current state:** `Decision.reason` is dropped at `pipeline.py:129-150`. `boundary_proposed` carries the signal, but not the look-ahead verdict.
- **Shape:**
  ```python
  # src/packet/events.py
  class EventKind(str, Enum):
      ...
      BOUNDARY_DECIDED = "boundary_decided"   # new, additive

  @classmethod
  def boundary_decided(cls, d: Decision, signal: BoundarySignal | None,
                       page_num: int | None, page_total: int | None) -> Event: ...
  ```
  `Pipeline._apply` yields it before acting on the decision. `packet parse --trace out.jsonl` writes the event stream as JSON lines. Off by default.
- **Blast radius:** consumers that switch on `EventKind` exhaustively (grep `EventKind.` in `tests/`). Event order tests in `tests/` that count events. Default `packet parse` JSON is unchanged.
- **Done when:** every page in a parse has exactly one `boundary_decided`, and the reason set equals the reasons in `lookahead.py`.
- **Proof:** a test on the `shared-header` and `multi-doc` built-in fixtures asserts one decision per page and that split decisions match `packet.documents` starts. `packet eval` numbers are byte-identical before and after.
- **Size:** S.

## Item 2: uncertainty as first-class signals in `extras` (lesson 2)

- **Current state:** `extras` is an untyped dict (`types.py:95`). Weak splits, fallback decisions, EOF flush, heading repairs and unresolved citations leave no trace in the tree.
- **Shape:**
  ```python
  # src/packet/types.py
  class Flag(BaseModel):
      kind: Literal["weak_boundary", "fallback_decision", "eof_flush",
                    "heading_rank_repaired", "unresolved_citation"]
      page: int | None = None
      ref: str | None = None          # block or section id
      detail: dict[str, Any] = Field(default_factory=dict)

  # reserved key, written by core only
  LogicalDocument.extras["packet.flags"]: list[Flag]  (serialized as dicts)
  ```
  Rules: `weak_boundary` when the opening decision had confidence below `boundary_commit_confidence` (0.75). `fallback_decision` for `fallback_eager` and `fallback_below`. `eof_flush` when the last document opened with no signal. `unresolved_citation` reuses the matcher behind `citation_integrity`, moved out of `eval/` into a core helper.
- **Blast radius:** JSON output grows one key under `extras`. Domain adapters must not overwrite `packet.flags` (document it in `domain.py` and DESIGN.md). Eval readers that compare whole trees (grep `extras` in `tests/` and `evals/`).
- **Done when:** each flag kind has a fixture that raises it and a fixture that does not. No flag changes a split or a section.
- **Proof:** tests per kind; `packet eval --set dlnet-whole` and the PoliTax gate give the same F1 and false-split count as `main`. Show the flag counts on PoliTax holdout in the PR.
- **Size:** M. Depends on item 1 (decision data) and item 4b for `heading_rank_repaired`.

## Item 3: few-shot, schema-constrained domain adapters (lesson 3)

- **Current state:** only `NoOpDomain`. The `DomainAdapter` contract (DESIGN.md "Domain adapters") already says adapters run on `document_closed`, add `type_hint` and `extras`, and never delete sections or blocks.
- **Shape:**
  ```python
  # src/packet/domain_fewshot.py  (opt-in, not imported by default)
  @dataclass(frozen=True)
  class Example:
      doc: LogicalDocument            # a real Packet tree, trimmed
      output: dict[str, Any]          # gold fields

  class FewShotAdapter:
      name: str
      schema: type[BaseModel]         # output model; validated, never free text
      examples: list[Example]
      backend: DecisionBackend        # reuse docs/plans/decision-backend.md
      def applies(self, doc: LogicalDocument) -> bool: ...
      def enrich(self, doc: LogicalDocument) -> LogicalDocument: ...
  ```
  Every extracted field carries the ids of the blocks it came from (`source_block_ids`). A field whose source ids are not in the doc is dropped and flagged. The adapter never writes `sections`.
- **Blast radius:** new module only. `DecisionBackend` interface (decision-backend plan) gains a structured-output task. No default change.
- **Done when:** a contract test proves the adapter cannot add, remove or reorder sections or blocks (tree hash before and after equal), and every field is grounded in a block id.
- **Proof:** one example adapter on 3 real PoliTax trees (train split only, never holdout) with a stub backend in CI. Grounding rate 100 percent on the stub.
- **Size:** M. Depends on the decision-backend structured-output task. Lowest value for boundary F1, so it is last.

## Item 4: multi-pass look-ahead (lesson 4)

### 4a. Mid-band VLM confirm pass

- **Current state:** `_decide_mid` (`lookahead.py:150-167`) only reads page numbers. With no evidence it falls back to the eager rule at 0.55. DESIGN.md "Next" item 4 plans a 3-page VLM confirm window.
- **Shape:**
  ```python
  class Confirmer(Protocol):
      def confirm(self, before: PageObs, page: PageObs, after: list[PageObs]) -> bool | None: ...
  LookaheadSplitter(..., confirmer: Confirmer | None = None)
  ```
  Called only where `_decide_mid` would return `fallback_eager` or `fallback_below`. `None` keeps today's rule. New reasons: `confirmed_split`, `confirmed_continue`.
- **Blast radius:** `lookahead.py` reasons (item 1 tests); latency per mid-band page; the 0.74 cap stays upstream and is untouched.
- **Done when:** with `confirmer=None`, decisions are identical to `main` on all eval sets. With a stub confirmer, only fallback pages change.
- **Proof:** replay test over the `boundary_decided` trace. PoliTax holdout F1 at or above 0.41, false splits not higher than `main`, before any default change. Wiki results first.
- **Size:** M. Depends on item 1 and a decision backend that passes images.

### 4b. Section-assembly repair pass

- **Current state:** `_nest` has no repair (see Ground truth). DESIGN.md claims repair exists. That is a doc bug to fix in the same ticket.
- **Shape:** `def repair_ranks(sections: list[Section]) -> tuple[list[Section], list[Flag]]` as a second pass in `Assembler.build_document` (`assemble.py:8-28`). Rule: numbering grammar (`1.`, `1.1`, `Article IV`) wins over `heading_level` when they disagree. Every reparent emits `heading_rank_repaired`.
- **Blast radius:** section trees in every output; tests that assert tree shape (grep `sections` in `tests/`). Boundary F1 unaffected (repair is intra-document only).
- **Done when:** a fixture with a mislabeled `1.1` under a wrong level is repaired and flagged; trees with consistent numbering are byte-identical to `main`.
- **Proof:** golden-tree tests plus a count of changed trees on dlnet-whole, each listed in the PR.
- **Size:** S-M.

Skip: cross-document re-merge after commit. DESIGN.md "No undo" is a streaming contract. Reopening a closed document would break `document_closed` consumers. Out of scope.

## Item 5: self-contained HTML visualizer and visual review path (lessons 1 and 5)

One tool serves both lessons. Lesson 5 is the visualizer run in batch next to `packet eval`.

- **Current state:** no visual output. `packet eval` prints numbers. Metric-blind failures (sections crossing documents, furniture leaking into body, mislabeled titles) need a human to see them.
- **Shape:**
  ```python
  # src/packet/viz.py  (stdlib only: json + string template, no JS deps, no network)
  def render_html(packet: Packet, events: list[Event], pdf: Path | None,
                  gold: list[int] | None = None) -> str: ...
  ```
  CLI: `packet viz <trace.jsonl> -o report.html` and `packet eval --viz-dir DIR` (writes one HTML per fixture; off by default). One file per packet, page images inlined as base64 PNG only when `pdf` is given. Views: page strip with block boxes colored by kind (furniture distinct); per-page signal list, score, and decision reason; a stepper over `document_closed` events showing the section tree; flags from item 2; gold vs predicted boundaries when `gold` is passed.
  Review checks (computed, also shown): section `page_start/page_end` outside its document; `HEADER/FOOTER/PAGE_NUMBER` text found in body blocks; title equal to a furniture string.
- **Blast radius:** new module and two opt-in CLI flags. Eval JSON unchanged. Report files can contain document text: write under `.packet-eval/` (already ignored), never commit PoliTax holdout renders.
- **Done when:** the HTML opens offline, with no external requests, and shows each DESIGN.md fail mode on a fixture made for it.
- **Proof:** tests assert the HTML has no `http` URLs and contains one stepper entry per `document_closed`. Screenshot of the `shared-header` false-merge in the PR. The three review checks each have a positive and a negative fixture.
- **Size:** M (renderer) + S (eval hook and checks).

## Sequencing as verifiable units

Each unit merges alone and keeps every eval number identical to `main` unless it says otherwise.

1. **U1 decision trace** (item 1). Proof: eval byte-identical; one decision per page.
2. **U2 flags in extras** (item 2, without `heading_rank_repaired`). Proof: per-kind tests; eval identical.
3. **U3 visualizer** (item 5 renderer). Proof: offline HTML test; screenshot of fail modes.
4. **U4 eval review path** (item 5 hook and checks). Proof: checks fire only on their fixtures.
5. **U5 heading repair + DESIGN.md fix** (item 4b, adds `heading_rank_repaired`). Proof: golden trees; changed-tree list.
6. **U6 mid-band confirm seam** (item 4a, `confirmer=None`). Proof: identical decisions on all sets.
7. **U7 confirm experiment** (item 4a with a real model). Proof: PoliTax holdout ≥ 0.41, no more false splits, wiki results, Tom's OK before any default.
8. **U8 few-shot adapter** (item 3). Proof: tree-hash contract; grounding 100 percent on stub.

U3 can start after U1, in parallel with U2. U5 and U6 depend only on U1.

## Recommended tickets (for Piper)

| # | Title | Scope | Size | Depends on |
|---|---|---|---|---|
| T1 | Packet: emit `boundary_decided` events and `--trace` JSONL | events.py, pipeline.py `_apply`, cli parse flag, tests | S | none |
| T2 | Packet: `packet.flags` in `LogicalDocument.extras` | types.py `Flag`, pipeline flags, core citation helper, DESIGN.md | M | T1 |
| T3 | Packet: self-contained HTML visualizer (`packet viz`) | viz.py, cli, offline tests | M | T1 |
| T4 | Packet: `packet eval --viz-dir` + metric-blind review checks | eval/run.py hook, three checks, fixtures | S | T3 |
| T5 | Packet: heading rank repair pass + DESIGN.md correction | assemble.py `repair_ranks`, flag, golden trees | S-M | T1 (T2 for flag) |
| T6 | Packet: mid-band `Confirmer` seam in look-ahead (off) | lookahead.py, PipelineConfig, replay test | M | T1 |
| T7 | Packet: mid-band VLM confirm experiment | bench run, PoliTax train/dev, holdout once, wiki page | M | T6, decision backend images |
| T8 | Packet: few-shot schema-constrained domain adapter (opt-in) | domain_fewshot.py, contract test, stub backend | M | decision backend structured output |

## Skip, and why

- **Adopting LangExtract as a dependency.** Packet is domain-free and streams. LangExtract is a whole-document extraction library. Take the lessons, not the code.
- **LLM chunked extraction in the core path.** Core stays rule-based plus opt-in decision backends (decision-backend plan). No model in the default path.
- **Cross-document re-merge or undo.** Breaks the `document_closed` streaming contract (DESIGN.md "No undo").
- **JS framework or hosted viewer.** One static HTML file is enough and keeps reports offline.
- **Free-text adapter output.** Schema-constrained only; invented structure is the failure lesson 3 warns about.
- **Raising or removing the 0.74 cap for the confirm pass.** Separate decision for Tom, with wiki results first.

## Rules carried

No default change without wiki results and Tom's OK. Never train or tune on the PoliTax holdout. Keep the 0.74 cap, look-ahead, the false-OCR hard-fail gate and the 0.41 PoliTax floor. Every new hook is opt-in and off by default.
