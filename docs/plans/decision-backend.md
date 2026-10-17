# Decision-model backend (v1)

Plan only. No code until Tom approves. Written against `main` `da7f030`. Refs GMA-257.

Packet grows one decision interface. A config file picks the backend. One OpenAI-compatible HTTP adapter posts to `POST /chat/completions`, which covers a local MLX, llama.cpp, or vLLM server and a hosted router. The in-process backend that exists today, `LogisticBoundaryScorer`, stays. Core code stays on the stdlib HTTP client. The base URL comes from config. This code does not probe for a GPU.

Perplexity Decider v1.2 is an example of a model that returns a typed answer and probabilities. The interface names no vendor and ships no default URL.

## Where packet decides

**Boundary score. Uses the interface.** `BoundaryDetector.observe` (`src/packet/boundary.py`) sums heuristic weights: page-number reset 0.45, sequence completed 0.2, header change 0.25, media-box jump 0.2, title at top 0.45, lower title 0.25, topic shift 0.2, language change 0.25. Two independent signals lift the score to at least 0.6 (`COMBINED`). A language change and a topic shift count as one vote. A score under 0.4 (`PROPOSE`) is dropped. When `PipelineConfig.scorer` is set (`src/packet/pipeline.py`), `combined_confidence` (`src/packet/boundary_model.py`) folds in `BoundaryScorer.probability`. The model term is ignored under 0.4, capped at 0.74, and never lowers the heuristic. With no `page_number_reset` in the reason list, `observe` caps again at 0.74, under the splitter commit of 0.75. `packet parse --boundary-weights` loads the JSON logistic file and exits 2 on `BoundaryWeightsError` (`src/packet/cli.py`). The new interface replaces that scorer slot. The heuristic weights stay, because the cap and the numbering veto read the reason list, and those weights are the schema-1 features.

**Gap rule. Stays rule-based.** `LookaheadSplitter._continues` (`src/packet/lookahead.py`) rejects a number whose page index is farther than `window + 1` from the anchor (`window` default 2). `_drain` commits a numbered page under 0.4 as `no_signal` on that page. `_decide_mid` emits `continues_sequence`, `lookahead_numbering`, `old_sequence_continues`, or `fallback_eager` (split at confidence >= 0.55). The gap bound is what opens a cleared silent page with no model score. The splitter keeps owning those reasons and keeps taking one confidence float.

**T9 scorer. This is the in-process backend.** `LogisticBoundaryScorer.probability` (`src/packet/boundary_model.py`) is pure-Python logistic regression on the 16 schema-1 features from `featurize_from_facts`. Config id: `logistic`. `run_fixture` calls `Pipeline()` (`src/packet/eval/run.py`), so `packet eval` leaves the scorer unset. A missing or mismatched weights file raises (decision 9). This backend reports `structured_output: false`, `logprobs: false`, `images: false`, and returns one probability. The transformers file that exists, `bench/ocr_compare/run_tf.py`, is the OCR bench loader. It stays that bench. It is not a decision backend. Config id `transformers` is reserved and exits 2 until a later ticket.

**OCR fallback. Stays rule-based.** `EscalatingExtractor._wants_ocr` (`src/packet/backends.py`) sends SCAN, MIXED, and SUSPECT_ENCODING to OCR and leaves TEXT and UNKNOWN on the native layer. `FallbackExtractor.extract` tries the next engine only after a raise; an empty result is final. `build_ocr_extractor` selects `PACKET_OCR`: `auto` uses LightOn-then-Paddle when `mlx_vlm` imports, otherwise Paddle. Engine choice is installation plus page class. `false_ocr_hard_fail` (`src/packet/eval/metrics.py`) hard-fails a VLM block on a long text page. `PACKET_OCR=auto` gains no CUDA, vLLM, or transformers branch.

**Page class. Stays rule-based.** `HeuristicClassifier.classify` (`src/packet/classify.py`) uses a 40-character count: TEXT at 0.7, MIXED at 0.45, SCAN at 0.6. `InspectorClassifier.classify` calls that heuristic. This is the OCR gate and the corpus plan's scan census.

**Furniture. Stays rule-based.** `FurnitureTagger.tag_page` and `finalize` (`src/packet/furniture.py`) tag HEADER, FOOTER, and PAGE_NUMBER from repeats. Page-number extraction reads those tags, and `finalize` rewrites earlier pages after `block_emitted`.

**Block kind and domain. Stay rule-based.** `layout_blocks` (`src/packet/layout.py`) and `_guess_kind` (`src/packet/extract.py`) label TITLE, HEADING, and PARAGRAPH from font size or a regex. `apply_domains` (`src/packet/domain.py`) runs `NoOpDomain`. Heading-path accuracy is outside the gate. A later task can call this interface. v1 has one call site.

## Interface

`--decision-config` points at a JSON file. The factory switches on the string `backend`. Ids are `logistic`, `http`, and `transformers`. Any other id exits 2. No config leaves `PipelineConfig.scorer` as `None`. `--boundary-weights` alone still builds `logistic`, which is today's parse flag.

```json
{
  "backend": "http",
  "base_url": "http://127.0.0.1:8080/v1",
  "model": "local-model",
  "timeout_s": 30,
  "max_retries": 2,
  "max_calls": 0,
  "auth": "env",
  "input": "features",
  "send_images": false,
  "capabilities": {"structured_output": false, "logprobs": false, "images": false}
}
```

`auth: env` reads `PACKET_DECISION_API_KEY` at request time. `auth: none` sends no Authorization header, for a local server. A field named `api_key`, `authorization`, or `token` fails the load. Errors name the env var. The value stays out of the config file, the timings line, and the traceback.

Request fields: `task` (`boundary` in v1), `labels` (`boundary`, `continue`), `text`, optional image bytes, optional JSON schema. Response fields: `probability` is P(`boundary`), optional `scores` (label, probability, logprob), `raw_text`, and `degraded` flags.

The HTTP adapter uses `urllib`. It sends `temperature: 0`. It sends `response_format` json_schema only when `capabilities.structured_output` is true, and `logprobs` only when `capabilities.logprobs` is true. Capabilities live in the config file, so CI does not discover them from a server. A live body that lacks a declared field sets the matching degraded flag. Vendor headers are omitted.

Default `input: features` sends the 16 schema-1 values, one fixed line per `FEATURE_NAMES` entry. `input: page_text` and `send_images: true` are opt-in. Images also require `capabilities.images`. `page_text` is truncated at `max_chars` (default 4000).

**Degradation, by decision point.**

- Boundary, structured output missing. Parse one JSON object that has `probability`, or else the whole trimmed reply as a single float in [0, 1]. Label `continue` with no scores becomes `1 - probability`. When scores exist, P(`boundary`) comes from the `boundary` score. Unparseable text leaves the heuristic, omits `+model` from the reason, and appends the page index to `decision_degraded_pages`.
- Boundary, logprobs missing. The scalar still goes through `combined_confidence` (floor 0.4, cap 0.74, heuristic kept when it is higher). `scores` is null. Boundary F1 keeps its definition. The run record says `logprobs: false`. That run is a calibration comparison only when the flag is true.
- A probability under 0.4 is the existing floor. That page is not degraded.
- Gap rule, OCR fallback, page class, furniture, and block kind do not call the backend. A missing capability leaves them unchanged.
- Timeout default 30s. Retries: `max_retries` default 2, backoff 0.5s then 1s, and only for HTTP 429, 502, 503, and connection errors. A 401, 403, 400, or 404 raises on that response, with no score emitted. A 200 that does not parse degrades the page and is not retried.
- Three degraded pages in one run raise. The CLI exits 2. A run with any degraded page is a heuristic-mixed result, and its F1 is not reported as the model.
- `max_calls` counts `decide` calls. The next call at the cap raises. Default 0 on `http` means the first call raises. `logistic` ignores `max_calls`.
- Logistic load errors still raise. That backend has no timeout and no retry.

**Gate and the default-on checklist.** `evaluate_gate` stays tolerance-0 boundary F1, the macro floor, and the false-OCR hard fail. `SET_FLOORS` stays builtin per-fixture 1.0, and `dlnet-whole` per-fixture 0.0 with macro 0.6. `run_fixture` keeps calling `Pipeline()`. Eval does not read `PACKET_DECISION_API_KEY` and does not open a decision socket. When parse was given a config, `--timings` adds `decision_backend` and `decision_degraded_pages`.

The corpus plan's default-on list stays the bar for a default logistic scorer: disjoint train and holdout, holdout macro and micro both strictly above the heuristic with no packet F1 down, builtin F1 1.0 with the weights loaded, `dlnet-whole` absent from training and its macro at least that plan's logged heuristic 0.98462, empty false-OCR pages, 0.99 still capped at 0.74 without a page-number reset, a probability under 0.4 still adding nothing, and a license that allows shipping the weights. Item 8: the backend id is `logistic`. `http` and `transformers` stay opt-in. A comparison printout shows backend id, capabilities, degraded count, and macro and micro beside the heuristic. It does not write `SET_FLOORS`.

## PoliTax, T4, leakage, comparability, cost

The corpus plan trains `train_logistic` on PoliTax-Split, Trump PDFs only, and holds out Obama, Harris, and Biden as whole files. Schema stays 1. This backend does not replace that trainer and does not fetch the corpus.

Leakage: the HTTP adapter sends `text` to `base_url`. Default `features` sends 16 numbers. Gate runs leave `page_text` and `send_images` off. Logs store status, latency, model id, and degraded flags. They omit the key, the Authorization header, and the page body. PoliTax pages are tax forms, so full text stays off until Tom confirms the files have no taxpayer identifiers.

Comparability needs the same gold, the same `LookaheadSplitter`, the same 0.4 floor and 0.74 cap, the same input mode, and an empty `decision_degraded_pages`. Boundary F1 can then sit beside the heuristic. A `page_text` run is a different input and is printed apart from a `features` run. Holdout files stay out of prompt-tuning.

Cost: the corpus plan counts 3,311 PoliTax pages, so one call per page is 3,311 calls. `max_calls` is the ceiling and defaults to 0 for `http`. This plan states no vendor price. T4 (Refs GMA-250) is the LightOn OCR measurement on a host Tom names, under its $5 cap. This plan does not start that host and does not add a decision-model download to it. A local decision server is the operator's process. Packet holds the URL.

## Risks

- Applying the HTTP probability as a commit score, past `combined_confidence`, would let 0.99 immediate-commit with no page-number reset. DB-3's cap fixture locks 0.74.
- A degraded run can look like a model win. The three-strike raise and the timings fields are the guard.
- A key field in JSON can land in git. The loader rejects those names.
- Full tax-form text sent to a third-party URL leaks identifiers. Default input is the feature lines.
- An unbounded `http` run over 3,311 pages is the spend risk. Default `max_calls` 0 fails closed.
- A test that imports torch or dials a public host breaks the portable core. Tests use recorded bodies and accept only host 127.0.0.1.

## Out of scope

- Replacing the gap rule, OCR choice, page class, furniture, layout, or domain adapters.
- Editing `SET_FLOORS`, fetching PoliTax, or training weights.
- A CUDA path, a GPU probe, starting vLLM, or any T4 instance change.
- Vendor SDKs, a default base URL, and prompt-tuning.
- ONNX and a `boundary` extra (already dropped, Refs GMA-257).
- Heading-path accuracy, reading order, and calibration inside the gate.
- Downloading model weights in CI.

## Questions for Tom

1. Is the boundary scorer the only v1 call site? Default: yes.
2. Is the default input the 16 schema-1 features, with page text and images off? Default: yes.
3. May `http` run on the PoliTax gate? Default: no. Loopback only, as a side report after the heuristic floor exists.
4. Does one bad page degrade, or does the run raise? Default: degrade that page, raise at 3, exit 2.
5. Can `http` or `transformers` become the default scorer? Default: no.
6. Build an in-process transformers decision loader in this set? Default: no. Reserve the id. Leave `bench/ocr_compare/run_tf.py` as the OCR bench. A later loader is explicit and CPU unless config sets `device`.
7. Timeout 30s and 2 retries on 429, 502, 503, and connection errors? Default: yes.
8. Keep schema 1? Default: yes.

## Tickets

### DB-1. Protocol and logistic backend

Goal: a `DecisionBackend` protocol, a capability report, and a logistic backend over `LogisticBoundaryScorer`. The factory accepts `logistic`. `transformers` exits 2 as not in this set. The default scorer stays `None`.

Files: `src/packet/decision.py`, a wrapper beside `load_logistic_scorer`, `tests/test_decision_logistic.py`, `tests/fixtures/decision/`.

Done when: input `tests/fixtures/decision/logistic_facts.json` (two schema-1 page-fact rows) produces `tests/fixtures/decision/logistic_probability.txt` (one float, six decimals) and `tests/fixtures/decision/logistic_capabilities.json` (all three capabilities false). `tests/test_t9_boundary_scorer.py` still passes. `run_fixture` still calls `Pipeline()`.

Depends on: none.

### DB-2. HTTP adapter and fake server

Goal: one stdlib adapter. Config selects it. The key comes from `PACKET_DECISION_API_KEY`. A loopback server replays recorded bodies.

Files: `src/packet/decision_http.py`, `tests/test_decision_http.py`, the fixtures below.

Done when:

- Input `tests/fixtures/decision/http_request_features.json` with recorded `tests/fixtures/decision/http_response_structured.json` yields label `boundary`, the fixture probability, and two label scores.
- Recorded body `tests/fixtures/decision/http_response_plain.txt` (text `0.62`) yields `tests/fixtures/decision/http_plain_expected.json`: probability 0.62, `scores` null, degraded `no_structured_output` and `no_logprobs`.
- Recorded body `tests/fixtures/decision/http_response_garbage.txt` yields `tests/fixtures/decision/http_garbage_expected.json`: no probability, degraded `unparseable`.
- The third timeout in one run raises. A 401 raises on the first response. The captured log has no key material.
- `tests/fixtures/decision/config_with_key.json` fails to load.
- The test is unmarked `needs_network` and `needs_gpu`. A request host other than 127.0.0.1 fails the test.

Depends on: DB-1.

### DB-3. Boundary call site

Goal: when a backend is configured, `observe` calls `decide` for task `boundary` and feeds P(`boundary`) through `combined_confidence` and the existing cap. The gap rule stays. Eval does not construct the backend.

Files: `src/packet/boundary.py`, `src/packet/pipeline.py`, `tests/test_decision_boundary.py`.

Done when:

- Fake probability 1.0 on a page with no page-number reset yields confidence <= 0.74 and a reason containing `model` (`tests/fixtures/decision/cap_expected.json`).
- Probability 0.2 with a heuristic under 0.4 adds nothing (`tests/fixtures/decision/floor_expected.json`).
- Unparseable text keeps the heuristic, omits `+model`, and records page index 1 in `decision_degraded_pages` (`tests/fixtures/decision/degraded_expected.json`).
- Probability 0.45 on an unnumbered page does not split (`tests/fixtures/decision/silent_045_expected.json`). Fallback stays 0.55.

Depends on: DB-1. The test uses a fake in-process backend, so DB-3 can land beside DB-2.

### DB-4. Config, timings, and the checklist guard

Goal: `--decision-config` selects the backend. Parse timings record the id and the degraded pages. Eval and the default constructor stay on the heuristic. A unit test locks checklist item 8.

Files: `src/packet/cli.py`, the loader in `src/packet/decision.py`, `tests/test_decision_config.py`, `tests/fixtures/decision/config_http.json`, `tests/fixtures/decision/default_on.json`.

Done when: `config_http.json` (`backend: http`, loopback `base_url`, `auth: env`, no key field) builds the HTTP adapter and reads the key from the environment. Eval's source still contains `pipeline or Pipeline()`, and the test shows eval does not read `PACKET_DECISION_API_KEY`. `default_on.json` lists the eight items. The test fails if an `http` config sets the default `PipelineConfig.scorer`. A fake parse with `--timings` writes `decision_backend` and `decision_degraded_pages`.

Depends on: DB-2 and DB-3.
