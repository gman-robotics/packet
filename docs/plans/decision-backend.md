# Decision-model backend (v1)

Plan only. No code until Tom approves. Written against `main` `da7f030`. Refs GMA-257. Challenge r1, r2, and r3 folded in. Tom's answers (12:58 PM MT, via Morgan) folded in.

Packet grows one decision interface. A config file picks the backend. One OpenAI-compatible HTTP adapter covers a local MLX, llama.cpp, or vLLM server and a hosted router. The in-process backend that exists today, `LogisticBoundaryScorer`, stays on the 16-feature schema-1 vector. The in-process Hugging Face runner is DB-2. It loads only when that backend is constructed. Core HTTP stays on the stdlib client. The base URL comes from config. This code does not probe for a GPU.

Perplexity Decider v1.2 is an example of a typed answer with probabilities. The interface names no vendor and ships no default URL.

## Where packet decides

**Boundary score. Uses the interface.** `BoundaryDetector.observe` (`src/packet/boundary.py`) sums heuristic weights: page-number reset 0.45, sequence completed 0.2, header change 0.25, media-box jump 0.2, title at top 0.45, lower title 0.25, topic shift 0.2, language change 0.25. `sequence_completed` and `title_block` are excluded from the independent-signal vote. Two independent signals lift the score to at least 0.6 (`COMBINED`). When topic shift and language change are both present, the topic-shift vote is dropped and `W_TOPIC_SHIFT` (0.2) is subtracted. A heuristic under 0.4 (`PROPOSE`) is no signal unless the model lifts it. Page index 0 returns before the scorer. When `PipelineConfig.scorer` is set (`src/packet/pipeline.py`), `observe` calls `featurize_from_facts` and `combined_confidence` (`src/packet/boundary_model.py`). The model term is ignored under 0.4, capped at 0.74, and does not lower the heuristic. With no `page_number_reset`, `observe` caps again at 0.74, under the splitter commit of 0.75, then applies `min(conf, 0.95)`. The model path cannot reach 0.95. `packet parse --boundary-weights` still loads that JSON and assigns a `BoundaryScorer`. `observe` keeps calling `probability(features)` on that object. A `DecisionBackend` is a separate object: `observe` calls `decide` on it, and the render thunk is an argument of `decide` only. Both paths feed `combined_confidence`. The heuristic weights stay, because the cap and the numbering veto read the reason list. The schema-1 vector is the 16 names in `FEATURE_NAMES`. Logistic scoring passes that vector to `probability`. The http and transformers default payload appends the fields under Default input. Those lines are request text. They are not coefficients.

**Gap rule. Stays rule-based.** `LookaheadSplitter._continues` (`src/packet/lookahead.py`) rejects a number whose page index is farther than `window + 1` from the anchor (`window` default 2). That rejection is what lets `_drain` open a cleared silent page by `implied_page_one` with no model score. `_drain` commits a numbered page under 0.4 as `no_signal` on that page. `_decide_mid` emits `continues_sequence`, `implied_page_one`, `lookahead_numbering`, `old_sequence_continues`, `fallback_eager`, or `fallback_below`. `fallback_eager` splits at confidence >= 0.55. The splitter keeps owning those reasons and keeps taking one confidence float.

**T9 scorer. Logistic is the weights backend. Transformers is DB-2.** `LogisticBoundaryScorer.probability` (`src/packet/boundary_model.py`) is pure-Python logistic regression on those 16 features and the loaded coefficients. Config id: `logistic`. `run_fixture` calls `Pipeline()` (`src/packet/eval/run.py`), so `packet eval` leaves the scorer unset. A missing or mismatched weights file raises `BoundaryWeightsError` (decision 9). This backend reports `structured_output: false`, `logprobs: false`, `images: false`, and returns one probability. `bench/ocr_compare/run_tf.py` is the OCR bench. It imports torch and is not a decision backend. `src/packet` does not import it. Config id `transformers` is the Hugging Face runner in DB-2. It implements the same `DecisionBackend`, the same capability flags, and the same strike rules. Constructing it is the first import of torch or transformers. Device is `cpu` unless the config sets `device`. There is no CUDA auto path. An unknown backend id still exits 2 at load. Neither `http` nor `transformers` is the default scorer.

**OCR choice. Stays rule-based.** `EscalatingExtractor._wants_ocr` (`src/packet/backends.py`) sends `SCAN`, `MIXED`, and `SUSPECT_ENCODING` to OCR. `HeuristicClassifier.classify` (`src/packet/classify.py`) never emits `SUSPECT_ENCODING`: at least 40 characters is TEXT at 0.7, a shorter non-empty sample is MIXED at 0.45, and an empty sample is SCAN at 0.6. `InspectorClassifier.classify` calls that heuristic. Page class chooses escalate-versus-native. It does not choose the engine. `LightOnExtractor.supports` and `PaddleExtractor.supports` accept the same class set. `build_ocr_extractor` picks the engine from `PACKET_OCR` plus `find_spec("mlx_vlm")` and `find_spec("paddleocr")`. `auto` with `PACKET_OCR_MODEL` returns LightOn only. `auto` without that variable uses LightOn-then-Paddle when `mlx_vlm` is present, otherwise Paddle. Unset `PACKET_OCR` is the legacy LightOn-then-Paddle chain, not `auto`. `--force-inspect` OCRs every class, including TEXT. On MIXED, an empty OCR result falls through to the native layer in `EscalatingExtractor.extract`. An empty result for `SCAN` or `SUSPECT_ENCODING` is final in that same method. `FallbackExtractor` returns an empty result from an engine that ran, and tries the next engine only after a raise. `PACKET_OCR` values `on`, `1`, `true`, and `yes` take the same legacy chain as an unset variable. `false_ocr_hard_fail` (`src/packet/eval/metrics.py`) flags `SourceKind.OCR` and `SourceKind.VLM` on a page whose native sample is at least 40 characters. The engines in tree tag `VLM`. `PACKET_OCR=auto` gains no CUDA, vLLM, or transformers branch. This interface does not grow an OCR branch. Block confidence already written by layout and OCR is copied into the default request. It does not choose the engine.

**Furniture. Stays rule-based.** `FurnitureTagger.tag_page` and `finalize` (`src/packet/furniture.py`) tag HEADER, FOOTER, and PAGE_NUMBER from repeats. Page-number extraction reads those tags, and `finalize` rewrites earlier pages after `block_emitted`. The default request copies the tags and the extracted numbers. The tagger still owns the decision.

**Block kind and domain. Stay rule-based.** `layout_blocks` (`src/packet/layout.py`) and `_guess_kind` (`src/packet/extract.py`) label TITLE, HEADING, and PARAGRAPH from font size or a regex. `apply_domains` (`src/packet/domain.py`) runs the adapters it is given. The stock adapter is `NoOpDomain`. Heading-path accuracy is outside the gate. v1 has one call site, `task=boundary`. The protocol fields (`task`, `labels`, `text`, image, schema) are what a later call site would use.

## Interface

`--decision-config` is parse-only. `packet inspect` keeps building `Pipeline()` with no scorer (`src/packet/cli.py`). No config leaves `PipelineConfig.scorer` as `None`. `--boundary-weights` alone still builds `logistic`. Passing `--decision-config` and `--boundary-weights` together exits 2. A `logistic` config carries a `weights` path. `backend: transformers` loads (DB-2). Any other id exits 2 at load time, before a parse loop.

```json
{
  "backend": "http",
  "base_url": "http://127.0.0.1:8080/v1",
  "model": "local-model",
  "timeout_s": 30,
  "max_retries": 2,
  "max_calls": 0,
  "auth": "env",
  "input": "default",
  "send_images": false,
  "capabilities": {"structured_output": false, "logprobs": false, "images": false}
}
```

`max_calls: 0` is the fail-closed default for `http`. Copying the example raises on the first `http` call. Omitting `input` means `default`.

**Wire.** `base_url` already includes `/v1`. The adapter appends `/chat/completions` and does not follow redirects. `auth: env` sends `Authorization: Bearer` plus `PACKET_DECISION_API_KEY`, read at request time. `auth: none` sends no Authorization header. No other vendor header. Body fields are `model`, `temperature: 0`, and one user message. `input: features` is one line per `FEATURE_NAMES` entry, `name=value` with six digits after the decimal, in that order, joined by `\n` and ending in `\n`. `input: page_text` is the block text already on the page fact, truncated at `max_chars` (default 4000). It is not `source.text_for`. `input: default` is the feature lines followed by the additive lines below. `response_format` is sent only when `capabilities.structured_output` is true, and the object is `{"type":"json_schema","json_schema":{"name":"boundary_decision","strict":true,"schema":{"type":"object","additionalProperties":false,"required":["label","probability","scores"],"properties":{"label":{"type":"string","enum":["boundary","continue"]},"probability":{"type":"number"},"scores":{"type":"array","items":{"type":"object","additionalProperties":false,"required":["label","probability"],"properties":{"label":{"type":"string","enum":["boundary","continue"]},"probability":{"type":"number"}}}}}}}}`. The API `logprobs` flag is sent only when `capabilities.logprobs` is true. The adapter reads `choices[0].message.content`. Recorded fixtures are those content strings. The fake server wraps each one in the chat envelope. Label `scores` are parsed from the model JSON. Token logprobs are not turned into label probabilities.

**Default input.** Q2 does not conflict with Q8. The logistic backend keeps the exact 16-feature vector. The weights file stays `schema_version` 1 with `feature_names` equal to `FEATURE_NAMES`. `featurize_from_facts` still returns those 16 floats, and `LogisticBoundaryScorer.probability` still rejects any other length. The expanded input is only the http and transformers request payload. The new lines are additive fields on that payload. They are optional relative to schema 1: `input: features` and `input: page_text` omit them, and the weights file does not gain them. On `input: default` every additive line is present. A missing source value is an empty right-hand side. No schema change is needed. Nothing is marked FLAG FOR TOM.

The formatter lives in `src/packet/decision.py`. It calls functions that are already in `src/packet`. It does not open a socket. Lines are `name=value`, joined by `\n`, ending in `\n`. Newlines inside `text_excerpt` are already folded. Floats use six digits after the decimal.

The default fixture is `logistic_facts.json` page 1 plus one block: kind `title`, text `alpha`, source `native`, confidence 0.8 (the value `layout_blocks` writes). `FurnitureTagger.tag_page` sees one line and returns the block unchanged. `HeuristicClassifier.classify` on `alpha` (5 characters) returns `mixed` at 0.45. `detect_language` returns null. The user message is exactly `has_num=1.000000\nnum_is_one=1.000000\nhas_total=0.000000\nprev_has_num=0.000000\narxiv_stamp=0.000000\ntitle_top=1.000000\ntitle_lower=0.000000\nlog_words=0.693147\nprev_log_words=0.000000\nhead_abstract=0.000000\nhead_intro=0.000000\nhead_affil=0.000000\nprev_refs=0.000000\nmedia_jump=0.000000\nlang_change=0.000000\ntopic_sim=1.000000\ntext_excerpt=alpha\npage_number=1\npage_total=\ntitle_pos=top\nrotation=0\nhas_header=0\nhas_footer=0\nhas_page_number_tag=0\nlang=\npage_class=mixed\nclass_confidence=0.450000\nsource_kinds=native\nmin_block_confidence=0.800000\n`.

| Field | Source | Type | Cap |
| --- | --- | --- | --- |
| `text_excerpt` | `page_fact_row` (`src/packet/boundary_model.py`), the `text` value | string | 400 characters after `\r\n`, `\n`, and `\r` each become one space |
| `page_number` | `_extract_page_number` (`src/packet/boundary.py`), stored by `page_fact_row` | integer or empty | 1..9999, else empty |
| `page_total` | `_extract_page_number`, stored by `page_fact_row` | integer or empty | 1..9999, else empty |
| `title_pos` | `title_position` (`src/packet/boundary_layout.py`) | `top`, `lower`, or empty | those three tokens |
| `rotation` | `page_fact_row` size triple; `PdfPageSource.pages` (`src/packet/source.py`) writes `PageRef.rotation` | integer or empty | 0, 90, 180, or 270, else empty |
| `has_header` | `FurnitureTagger.tag_page` (`src/packet/furniture.py`) | `0` or `1` | one character |
| `has_footer` | `FurnitureTagger.tag_page` | `0` or `1` | one character |
| `has_page_number_tag` | `FurnitureTagger.tag_page` (`BlockKind.PAGE_NUMBER`) | `0` or `1` | one character |
| `lang` | `detect_language` (`src/packet/topic.py`), stored by `page_fact_row` | string or empty | 16 characters |
| `page_class` | `HeuristicClassifier.classify` (`src/packet/classify.py`), `PageRef.page_class` | `text`, `scan`, `mixed`, `suspect_encoding`, or `unknown` | one of those tokens |
| `class_confidence` | `HeuristicClassifier.classify`, `PageRef.class_confidence` | six-decimal float | six digits; the heuristic writes 0.700000, 0.450000, or 0.600000 |
| `source_kinds` | `Block.source` as set by `layout_blocks` (`src/packet/layout.py`), `NativeTextExtractor.extract` (`src/packet/extract.py`), `parse_lighton_grounding`, `_markdown_blocks`, and `parse_paddle_blocks` (`src/packet/ocr_parse.py`) | comma-separated sorted unique tokens | `native`, `ocr`, `vlm`, `merged` only, at most 32 characters |
| `min_block_confidence` | `Block.confidence` from those same writers (`layout_blocks` 0.8, `NativeTextExtractor.extract` 0.8 or 0.5, `parse_lighton_grounding` 0.9, `_markdown_blocks` 0.8, `parse_paddle_blocks` 0.85) | six-decimal float or empty | 0..1 with six digits; empty when the page has no blocks |

**Images.** `pipeline.stream` still passes `_lazy_render` to `router.extract` on every page. OCR keeps that argument. The object passed into `decide` is separate, and it is passed only when `send_images` and `capabilities.images` are both true. Then the user message content is a list of one `{type: text}` part and one `{type: image_url, image_url: {url}}` part, and the url is `data:image/png;base64,` plus the base64 of the PNG. Either flag false withholds that argument, so `decide` does not call it, and the message has no `image_url`. Default `send_images` is false. Images stay opt-in. `input: page_text` stays opt-in at 4000 characters. The default payload includes `text_excerpt` at 400 characters.

**Secrets.** The only secret read is `PACKET_DECISION_API_KEY`. The loader rejects `api_key`, `authorization`, and `token` case-insensitively at any depth. It rejects a `base_url` with userinfo, and a query parameter `api_key`, `token`, or `authorization`. An unset or empty `PACKET_DECISION_API_KEY` with `auth: env` raises `DecisionBackendError` before any socket. The text names the env var. It omits the key, the Authorization value, URL userinfo, page text, the image, and the response body. Exit 2, including a batch abort.

**Capability flags vs strikes.** `no_structured_output`, `no_logprobs`, and `no_images` are run-level facts. They are set when the config capability is false, and when a 200 omits a declared field. They do not append to `decision_degraded_pages` and do not withhold boundary F1. A content body `0.62` is a usable P(`boundary`): probability 0.62, `scores` null, flags set, no strike.

A strike is a page with no usable P(`boundary`): unparseable content, or a call whose retries are exhausted. That page keeps the heuristic, omits `+model`, and appends the 0-based page index. A probability under 0.4 is the existing floor, not a strike. F1 is withheld, and a comparison would say heuristic-mixed, only when `decision_degraded_pages` is non-empty. Two strikes complete. The third raises `DecisionBackendError`.

**Attempts and abort.** The status set is closed. 200 is parsed. 429, 502, 503, `TimeoutError`, and other connection failures are retried. Every other status, including 500, 408, and 504, and every other 4xx and every 3xx, raises `DecisionBackendError` on that response. Those responses are not retried and do not become a heuristic score. The text is the same redaction: the status and the page index. Exit 2, and the batch aborts. One `decide` gets `1 + max_retries` attempts (`max_retries` default 2). Backoff is 0.5s then 1s on a fake clock. The test does not sleep. Exhaustion of the retried statuses is one strike, the same as unparseable content, not a second abort. A 200 that omits a declared field sets the run-level capability flag and is not a strike. A 200 with no usable P(`boundary`) is a strike. The third-strike text says `unparseable` and the page index, and it omits the body.

`_cmd_parse` catches `DecisionBackendError` the way it catches `BoundaryWeightsError`: one `packet:` line and exit 2. That holds for a single file and for a batch. The batch loop does not continue. Strikes and the `max_calls` count reset at the start of each `parse`. `max_calls` counts `decide` calls, not HTTP attempts, so a page that uses every retry still costs one call. `Pipeline` is built once for the argv list, so the reset is per file, not per process. `max_calls` 0 on `http` raises on the first `decide`. `logistic` and `transformers` ignore `max_calls`. A logistic weights path inside `--decision-config` that is missing or rejected raises `BoundaryWeightsError` and exits 2 before the parse loop, the same catch as `--boundary-weights`. That backend has no timeout and no retry. The transformers runner has no HTTP retry. Its unparseable output uses the same strike rules.

**Gate.** `evaluate_gate` stays tolerance-0 boundary F1, the macro floor, and the false-OCR hard fail. `SET_FLOORS` stays builtin `(1.0, None)` and `dlnet-whole` `(0.0, 0.6)`. A `None` macro argument uses the per-fixture floor, so builtin macro is 1.0 as well. `run_fixture` keeps calling `Pipeline()`. Eval does not read `PACKET_DECISION_API_KEY` and does not open a decision socket. When parse was given a config, `--timings` adds `decision_backend` and `decision_degraded_pages` (strike indexes only).

The corpus plan's default-on list stays the bar for a default logistic scorer: disjoint train and holdout, holdout macro and micro both strictly above the heuristic with no packet F1 down, builtin F1 1.0 with the weights loaded, `dlnet-whole` absent from training and its macro at least that plan's logged heuristic 0.98462, empty false-OCR pages, 0.99 still capped at 0.74 without a page-number reset, a probability under 0.4 still adding nothing, and a license that allows shipping the weights. Item 8: the backend id is `logistic`. `http` and `transformers` stay opt-in. The behavioral lock is that an `http` or `transformers` config does not set the default `PipelineConfig.scorer`, and eval does not read the key. A comparison printout of backend id, capabilities, strike count, and macro and micro beside the heuristic is not in this set. It would not write `SET_FLOORS`.

## PoliTax, T4, leakage, comparability, cost

The corpus plan trains `train_logistic` on PoliTax-Split, Trump PDFs only, and holds out Obama, Harris, and Biden as whole files. Schema stays 1. This backend does not replace that trainer, does not fetch the corpus, and does not edit `SET_FLOORS`.

Leakage: logistic sends 16 numbers. The http and transformers default payload also sends `text_excerpt` (400 characters) and the other additive lines. Gate runs leave `input: page_text` and `send_images` off, and question 3 keeps `http` off the PoliTax gate. Logs store status, latency, model id, and the capability flags. They omit the key, the Authorization value, URL userinfo, and the page body. PoliTax pages are tax forms. A 400-character excerpt can still hold an identifier. Full page text stays off unless the config sets `input: page_text`.

Comparability needs the same gold, the same `LookaheadSplitter`, the same 0.4 floor and 0.74 cap, the same input mode, and an empty strike list. Boundary F1 can then sit beside the heuristic. A `default` run, a `features` run, and a `page_text` run are different inputs and are printed apart. `temperature: 0` does not make a hosted router bit-stable: the router can change provider between calls, and this plan refuses a vendor provider pin. An `http` F1 is one recorded run, not a second gate, and not a clean PoliTax generalization test. The PDFs are public, so a hosted model may have seen the holdout. Holdout files stay out of prompt-tuning.

Cost: page 0 does not call `decide`, so the corpus plan's 3,311 PoliTax pages are 3,310 `decide` calls. The additive lines add bytes to each http or transformers request. They do not add calls. Retries are extra HTTP requests and do not add to that count. `max_calls` is the http ceiling and defaults to 0. `logistic` and `transformers` ignore it. This plan states no vendor price and no transformers pages per minute. T4 (Refs GMA-250) is the LightOn OCR measurement on a host Tom names, under its $5 cap. This plan does not start that host and does not add a decision-model download to it. A local decision server is the operator's process. Packet holds the URL. The transformers runner reads `model` and `revision` from config and does not download them in CI.

## Risks

- Applying the HTTP probability as a commit score, past `combined_confidence`, would let 0.99 immediate-commit with no page-number reset. DB-4's cap fixture locks reason `model` and confidence 0.74.
- Counting a capability flag as a strike would abort a healthy plain-text server on page 3. Strikes are only pages with no usable probability.
- A key hidden under another spelling, in userinfo, or in a query string can land in git. The loader rejects those forms.
- A 400-character excerpt on the default http payload can still carry a taxpayer identifier. Images stay off. The PoliTax gate does not run `http`. Logistic sends the 16 numbers only. `input: page_text` stays opt-in.
- An unbounded `http` run is the spend risk. Default `max_calls` 0 fails closed. The per-parse reset keeps one file from consuming the next file's budget.
- A test that imports torch, sleeps the backoff, or dials a public host breaks the portable core. Tests use a fake clock, recorded `message.content` values, and accept only host `127.0.0.1`. The transformers CI test uses a fake runner. A module-level import of torch or transformers would load that stack when `src/packet` loads.
- A CUDA auto path, or a call to `torch.cuda.is_available`, would assume a GPU host. DB-2 keeps device `cpu` unless config sets `device`.

## Out of scope

- Replacing the gap rule, OCR choice, page class, furniture, layout, or domain adapters.
- Editing `SET_FLOORS`, fetching PoliTax, or training weights.
- A CUDA path, a GPU probe, starting vLLM, or any T4 instance change.
- Vendor SDKs, a default base URL, a provider pin, and prompt-tuning.
- ONNX and a `boundary` extra (already dropped, Refs GMA-257).
- Heading-path accuracy, reading order, and calibration inside the gate.
- The comparison printout described above.
- A schema change to the logistic weights file. Schema stays 1 with `FEATURE_NAMES`.
- Importing `bench/ocr_compare/run_tf.py` from `src/packet`, or treating that file as the decision runner.
- Downloading model weights in CI.

## Questions for Tom

Tom answered at 12:58 PM MT, via Morgan: "I accept the defaults except: 2. The 16 features should be expanded. They seem insufficient on their own. 6. A ticket should be created for the huggingface runner and it added sooner than later." Questions 1, 3, 4, 5, 7, and 8 take their defaults. No new question.

1. Answered. Accepted the default. The boundary scorer is the only v1 call site.
2. Answered. Tom rejected the default. The 16 schema-1 features stay the logistic vector. The http and transformers default payload adds the fields under Default input. Images stay opt-in. This does not conflict with question 8. No schema change.
3. Answered. Accepted the default. `http` does not run on the PoliTax gate. Loopback only, as a side report after the heuristic floor exists. It does not write `SET_FLOORS`.
4. Answered. Accepted the default. Aborting on the third strike is the ceiling, exit 2. A plain `0.62` is not a strike.
5. Answered. Accepted the default. `http` and `transformers` do not become the default scorer.
6. Answered. Tom rejected the default. DB-2 is the in-process Hugging Face runner, ordered immediately after DB-1.
7. Answered. Accepted the default. Timeout 30s and 2 retries on 429, 502, 503, timeout, and connection errors. Exhaustion counts as one strike.
8. Answered. Accepted the default. Schema stays 1.

## Tickets

### DB-1. Protocol and logistic backend

Goal: a `DecisionBackend` protocol with `decide`, and a logistic backend that implements it by calling `LogisticBoundaryScorer.probability`. `BoundaryScorer.probability(features)` stays the method `observe` uses for `--boundary-weights`. The factory accepts `logistic` when the config includes `weights`. Unknown ids exit 2 at load. The default scorer stays `None`. The default-message formatter lives here so DB-2 and DB-3 share one payload. Logistic still scores the 16 floats only.

Files: `src/packet/decision.py`, a wrapper beside `load_logistic_scorer`, `tests/test_decision_logistic.py`, `tests/fixtures/decision/`.

Done when: `tests/fixtures/decision/logistic_facts.json` is two `page_fact_row` dicts, not two 16-vectors. Page 0 is `page_number` null, `page_total` null, `title` null, `text` `""`, `words` `{}`, `lang` null, `size` `[612, 792, 0]`. Page 1 is `page_number` 1, `page_total` null, `title` `"top"`, `text` `"alpha"`, `words` `{"alpha": 1}`, `lang` null, `size` `[612, 792, 0]`. `tests/fixtures/decision/logistic_weights.json` is a document `load_logistic_scorer` accepts: `schema_version` 1, `feature_names` equal to `FEATURE_NAMES`, `mean` sixteen zeros, `std` sixteen ones, `weights` sixteen zeros, `bias` 0. `tests/fixtures/decision/logistic_probability.txt` is `0.500000\n`, the six-decimal form of `probability(featurize_from_facts(facts, 1))`. `tests/fixtures/decision/logistic_capabilities.json` has all three capabilities false. `tests/fixtures/decision/config_logistic.json` points `weights` at that file and loads. `tests/fixtures/decision/config_unknown_backend.json` fails at load. `config_transformers.json` is a DB-2 fixture and is not rejected here. The formatter, given page 1 and the one block named under Default input, returns that section's user-message text. `probability` is still called with the 16 floats only. `tests/test_t9_boundary_scorer.py` still passes, including `_ConstantScorer.probability` and `test_model_probability_capped_at_074_without_page_number_reset`. `run_fixture` still calls `Pipeline()`.

Depends on: none.

### DB-2. In-process Hugging Face runner

Goal: a transformers backend with the same `DecisionBackend` interface, capability flags, and strike rules. Tom asked for this ticket sooner than later. It sits immediately after DB-1 because it only needs the protocol and the factory. It does not need the HTTP adapter or the boundary call site.

Files: `src/packet/decision_transformers.py`, the factory branch in `src/packet/decision.py`, `tests/test_decision_transformers.py`, the fixtures below. `bench/ocr_compare/run_tf.py` stays where it is.

Done when:

- Importing `packet`, `packet.decision`, and `packet.decision_transformers` leaves `torch` and `transformers` out of `sys.modules`. The real load function is the one that imports them. The unmarked test never calls that function.
- `tests/fixtures/decision/config_transformers.json` loads. It is `backend` `transformers`, `model` `packet-ci-fake`, `revision` `ci-revision`, no `device`, no `base_url`, and no key field. The loaded device is `cpu`. The load does not open a socket and does not read `PACKET_DECISION_API_KEY`.
- `tests/fixtures/decision/config_transformers_device.json` sets `device` to `cpu` and loads the same way. A config whose `device` is `cuda` stores that string. The runner does not call `torch.cuda.is_available` and does not take a CUDA auto path.
- CI injects a fake behind the same interface. The fake records `model`, `revision`, and `device`. `tests/fixtures/decision/tf_response_plain.txt` is `0.62`. It yields `tests/fixtures/decision/tf_plain_expected.json`: probability 0.62, `scores` null, run-level flags `no_structured_output` and `no_logprobs`, and no strike.
- `tests/fixtures/decision/tf_response_garbage.txt` yields `tests/fixtures/decision/tf_garbage_expected.json`: no probability, one strike, reason `unparseable`.
- `tests/fixtures/decision/tf_third_strike_expected.json`: unparseable pages 1 and 2 are recorded, and the call for page 3 raises `DecisionBackendError`. The exception text says `unparseable` and `3`, and it omits the body.
- The request text for `logistic_facts.json` page 1 is the Default input string. `tests/fixtures/decision/http_request_default.json` holds that message. Images stay off unless `send_images` and `capabilities.images` are both true.
- An optional test marked `needs_models` constructs the real loader with the config `model` and `revision` and the resolved device. CI does not select `needs_models`, so that test is skipped. The unmarked test is unmarked `needs_network`, `needs_gpu`, `needs_models`, and `needs_ocr_extras`.
- `src/packet` does not import `bench/ocr_compare/run_tf.py`.

Depends on: DB-1.

### DB-3. HTTP adapter and fake server

Goal: one stdlib adapter for the wire format above. Config selects it. The key comes from `PACKET_DECISION_API_KEY` at request time. A `127.0.0.1` server wraps recorded `message.content` values in the chat envelope. No redirects.

Files: `src/packet/decision_http.py`, `tests/test_decision_http.py`, the fixtures below.

Done when:

- Captured request `tests/fixtures/decision/http_request_features.json` is `model`, `temperature` 0, and one user message, for `input: features`. The message text is the sixteen lines from `featurize_from_facts` on `logistic_facts.json` at index 1, `name=value` with six decimal digits, and that text is exactly `has_num=1.000000\nnum_is_one=1.000000\nhas_total=0.000000\nprev_has_num=0.000000\narxiv_stamp=0.000000\ntitle_top=1.000000\ntitle_lower=0.000000\nlog_words=0.693147\nprev_log_words=0.000000\nhead_abstract=0.000000\nhead_intro=0.000000\nhead_affil=0.000000\nprev_refs=0.000000\nmedia_jump=0.000000\nlang_change=0.000000\ntopic_sim=1.000000\n`.
- Captured request `tests/fixtures/decision/http_request_default.json` is the same envelope for `input: default`. The user message is exactly the Default input string.
- `tests/fixtures/decision/http_response_structured.json` bytes are exactly `{"label":"boundary","probability":0.62,"scores":[{"label":"boundary","probability":0.62},{"label":"continue","probability":0.38}]}\n`. The fake server uses that string without the trailing newline as `message.content`. It yields `tests/fixtures/decision/http_structured_expected.json`: label `boundary`, probability 0.62, and scores `boundary` 0.62 and `continue` 0.38. Envelope token logprobs are ignored. The request's `response_format` is the object named in Wire.
- Content `tests/fixtures/decision/http_response_plain.txt` (text `0.62`) yields `tests/fixtures/decision/http_plain_expected.json`: probability 0.62, `scores` null, run-level flags `no_structured_output` and `no_logprobs`, and no strike.
- `tests/fixtures/decision/three_plain_expected.json`: three content bodies `0.62`, zero strikes, those flags set, no raise.
- Content `tests/fixtures/decision/http_response_garbage.txt` yields `tests/fixtures/decision/http_garbage_expected.json`: no probability, one strike, reason `unparseable`.
- `tests/fixtures/decision/third_strike_expected.json`: unparseable pages 1 and 2 are recorded, and the call for page 3 raises `DecisionBackendError`. The exception text says `unparseable` and `3`, and it omits the body.
- One call with `max_retries` 2 makes three attempts. A timeout fixture uses a fake clock and does not sleep the 0.5s and 1s backoffs. Exhaustion is one strike. A 400 and a 401 each raise on that response and are not retried.
- A recorded 500 is one attempt and raises `DecisionBackendError`. The exception text and the timings `error` are the status `500` and the page index. The sentinel is absent. The test does not sleep and does not leave `127.0.0.1`. DB-5 is the test that reads stderr and the timings row.
- A 302 whose `Location` is `http://example.invalid/v1/chat/completions` is not requested. It raises `DecisionBackendError`. The sentinel is absent from the exception.
- An unset `PACKET_DECISION_API_KEY` and an empty one, with `auth: env`, each raise `DecisionBackendError` before any socket. The text names the env var. The connect count is 0.
- `tests/fixtures/decision/max_calls_0_expected.json`: `max_calls` 0 raises on the first `decide`. `tests/fixtures/decision/max_calls_1_expected.json`: the first `decide` may use all of its retries and still counts as one call, and the second `decide` raises.
- `tests/fixtures/decision/http_request_page_text.json`: `input: page_text`. The input text is 4001 `a` characters. The user message is the first 4000 `a` characters and nothing else. It does not include the feature lines or the additive lines.
- `tests/fixtures/decision/http_request_image.json`: both image flags are true and `input` is `default`. The text part is the Default input string. The thunk returns `tests/fixtures/decision/one_pixel.png`, the 1×1 PNG whose base64 is `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC`. Content is `[{type: text, text: <that default string>}, {type: image_url, image_url: {url: data:image/png;base64,<that base64>}}]`.
- `tests/fixtures/decision/http_request_images_off.json`: `send_images` is true and `capabilities.images` is false. The message is the Default input string only. It has no `image_url`. The decide thunk is not called.
- The sentinel value of `PACKET_DECISION_API_KEY` is absent from the exception text. `tests/fixtures/decision/config_with_key.json`, a nested `Token`, a `base_url` with userinfo, and a query `api_key` each fail to load.
- The test is unmarked `needs_network`, `needs_gpu`, `needs_models`, and `needs_ocr_extras`. A request host other than `127.0.0.1` fails the test.

Depends on: DB-1.

### DB-4. Boundary call site

Goal: a configured `DecisionBackend` makes `observe` call `decide` for task `boundary` and feed P(`boundary`) through `combined_confidence` and the existing caps. A `BoundaryScorer` still goes through `probability`. The gap rule stays. `router.extract` keeps `_lazy_render`. The object passed into `decide` is withheld unless `send_images` and `capabilities.images` are both true. Eval does not construct the backend.

Files: `src/packet/boundary.py`, `src/packet/pipeline.py`, `tests/test_decision_boundary.py`.

Done when:

- Page 0 is present, so page 1 is scored. Page 1 has no heuristic signal. Fake probability 1.0 and no page-number reset yield confidence 0.74 and reason `model` (`tests/fixtures/decision/cap_expected.json`).
- Page 0 is present. Page 1 has no heuristic signal. Probability 0.2 makes `observe` return no signal (`tests/fixtures/decision/floor_expected.json`).
- Unparseable text keeps the heuristic, omits `+model`, and records page index 1 in `decision_degraded_pages` (`tests/fixtures/decision/degraded_expected.json`). The capability flags are absent from that list.
- Three pages, then flush. No heuristic signal on any page. The fake returns a probability per page. Page 0 is unnumbered: `observe` returns before `decide`, the decision is `(0, false, no_signal)`, and it sets no numbering anchor. Page 1 is unnumbered and is the silent page; its `decide` return is the injected probability. Page 2 is numbered 1 with no total, and its `decide` returns 0.10, which is under 0.4, so its decision is `(2, false, no_signal)`. Page 2 stays under 0.4 because a mid-band return there would open page 1 with an unset number and `_continues` would mark page 2 `continues_sequence`. For page 1, `implied_page_one` sees `1 - 1 != 1` and stays false, and the anchor is still unset, so `old_sequence_continues` stays false. `tests/fixtures/decision/silent_045_expected.json` is `(0, false, no_signal), (1, false, fallback_below), (2, false, no_signal)`. `tests/fixtures/decision/silent_055_expected.json` is `(0, false, no_signal), (1, true, fallback_eager), (2, false, no_signal)`.
- Default `send_images` false: `decide` is not given the thunk. `router.extract` still receives `_lazy_render`.
- `send_images` true and `capabilities.images` false: `decide` is not given the thunk, and the captured request matches `http_request_images_off.json`.

Depends on: DB-1. The test uses a fake in-process backend, so DB-4 can land beside DB-3. It does not wait on DB-2.

### DB-5. Config, timings, and the checklist guard

Goal: `--decision-config` selects the backend on parse only. `DecisionBackendError` exits 2 for one file and for a batch. Strikes and `max_calls` reset per `parse`. Eval and `inspect` stay on the heuristic.

Files: `src/packet/cli.py`, the loader in `src/packet/decision.py`, `tests/test_decision_config.py`, `tests/fixtures/decision/config_http.json`, `tests/fixtures/decision/config_logistic.json`.

Done when: `config_http.json` (`backend: http`, loopback `base_url` ending in `/v1`, `auth: env`, no key field) builds the HTTP adapter. `config_logistic.json` is a valid loader document: `backend` `logistic` and `weights` pointing at `logistic_weights.json`, and it loads. `tests/fixtures/decision/config_logistic_missing.json` names a weights path that is not on disk, prints one `packet:` line, exits 2, and does not parse. Both CLI flags together exit 2. A single-file parse and a two-file batch each print one `packet:` line and exit 2 on `DecisionBackendError`, and the batch does not parse the second file. The timings `error` for that row is the safe status-and-page text. For the recorded 500, that field is `500` and the page index. The sentinel is absent from stderr and from the timings line. After a one-strike file, the next file in a successful argv starts at zero strikes and zero calls. Eval's source still contains `pipeline or Pipeline()`, and the test shows eval does not read `PACKET_DECISION_API_KEY`. `inspect` builds `Pipeline()` with no scorer. An `http` config and a `transformers` config do not set the default `PipelineConfig.scorer`. A fake parse with `--timings` writes `decision_backend` and `decision_degraded_pages`.

Depends on: DB-3 and DB-4.
