# Decision-model backend (v1)

Plan only. No code until Tom approves. Written against `main` `da7f030`. Refs GMA-257. Challenge r1 and r2 folded in.

Packet grows one decision interface. A config file picks the backend. One OpenAI-compatible HTTP adapter covers a local MLX, llama.cpp, or vLLM server and a hosted router. The in-process backend that exists today, `LogisticBoundaryScorer`, stays. Core code stays on the stdlib HTTP client. The base URL comes from config. This code does not probe for a GPU.

Perplexity Decider v1.2 is an example of a typed answer with probabilities. The interface names no vendor and ships no default URL.

## Where packet decides

**Boundary score. Uses the interface.** `BoundaryDetector.observe` (`src/packet/boundary.py`) sums heuristic weights: page-number reset 0.45, sequence completed 0.2, header change 0.25, media-box jump 0.2, title at top 0.45, lower title 0.25, topic shift 0.2, language change 0.25. `sequence_completed` and `title_block` are excluded from the independent-signal vote. Two independent signals lift the score to at least 0.6 (`COMBINED`). When topic shift and language change are both present, the topic-shift vote is dropped and `W_TOPIC_SHIFT` (0.2) is subtracted. A heuristic under 0.4 (`PROPOSE`) is no signal unless the model lifts it. Page index 0 returns before the scorer. When `PipelineConfig.scorer` is set (`src/packet/pipeline.py`), `observe` calls `featurize_from_facts` and `combined_confidence` (`src/packet/boundary_model.py`). The model term is ignored under 0.4, capped at 0.74, and does not lower the heuristic. With no `page_number_reset`, `observe` caps again at 0.74, under the splitter commit of 0.75, then applies `min(conf, 0.95)`. The model path cannot reach 0.95. `packet parse --boundary-weights` still loads that JSON and assigns a `BoundaryScorer`. `observe` keeps calling `probability(features)` on that object. A `DecisionBackend` is a separate object: `observe` calls `decide` on it, and the render thunk is an argument of `decide` only. Both paths feed `combined_confidence`. The heuristic weights stay, because the cap and the numbering veto read the reason list. The schema-1 vector is the 16 names in `FEATURE_NAMES`.

**Gap rule. Stays rule-based.** `LookaheadSplitter._continues` (`src/packet/lookahead.py`) rejects a number whose page index is farther than `window + 1` from the anchor (`window` default 2). That rejection is what lets `_drain` open a cleared silent page by `implied_page_one` with no model score. `_drain` commits a numbered page under 0.4 as `no_signal` on that page. `_decide_mid` emits `continues_sequence`, `implied_page_one`, `lookahead_numbering`, `old_sequence_continues`, `fallback_eager`, or `fallback_below`. `fallback_eager` splits at confidence >= 0.55. The splitter keeps owning those reasons and keeps taking one confidence float.

**T9 scorer. This is the in-process backend.** `LogisticBoundaryScorer.probability` (`src/packet/boundary_model.py`) is pure-Python logistic regression on those 16 features and the loaded coefficients. Config id: `logistic`. `run_fixture` calls `Pipeline()` (`src/packet/eval/run.py`), so `packet eval` leaves the scorer unset. A missing or mismatched weights file raises `BoundaryWeightsError` (decision 9). This backend reports `structured_output: false`, `logprobs: false`, `images: false`, and returns one probability. `bench/ocr_compare/run_tf.py` is the OCR bench. It imports torch and is not a decision backend. Config id `transformers` is reserved and exits 2 at load time. Question 6 asks Tom whether a later ticket should add that loader.

**OCR choice. Stays rule-based.** `EscalatingExtractor._wants_ocr` (`src/packet/backends.py`) sends `SCAN`, `MIXED`, and `SUSPECT_ENCODING` to OCR. `HeuristicClassifier.classify` (`src/packet/classify.py`) never emits `SUSPECT_ENCODING`: at least 40 characters is TEXT at 0.7, a shorter non-empty sample is MIXED at 0.45, and an empty sample is SCAN at 0.6. `InspectorClassifier.classify` calls that heuristic. Page class chooses escalate-versus-native. It does not choose the engine. `LightOnExtractor.supports` and `PaddleExtractor.supports` accept the same class set. `build_ocr_extractor` picks the engine from `PACKET_OCR` plus `find_spec("mlx_vlm")` and `find_spec("paddleocr")`. `auto` with `PACKET_OCR_MODEL` returns LightOn only. `auto` without that variable uses LightOn-then-Paddle when `mlx_vlm` is present, otherwise Paddle. Unset `PACKET_OCR` is the legacy LightOn-then-Paddle chain, not `auto`. `--force-inspect` OCRs every class, including TEXT. On MIXED, an empty OCR result falls through to the native layer in `EscalatingExtractor.extract`. An empty result for `SCAN` or `SUSPECT_ENCODING` is final in that same method. `FallbackExtractor` returns an empty result from an engine that ran, and tries the next engine only after a raise. `PACKET_OCR` values `on`, `1`, `true`, and `yes` take the same legacy chain as an unset variable. `false_ocr_hard_fail` (`src/packet/eval/metrics.py`) flags `SourceKind.OCR` and `SourceKind.VLM` on a page whose native sample is at least 40 characters. The engines in tree tag `VLM`. `PACKET_OCR=auto` gains no CUDA, vLLM, or transformers branch. This interface does not grow an OCR branch.

**Furniture. Stays rule-based.** `FurnitureTagger.tag_page` and `finalize` (`src/packet/furniture.py`) tag HEADER, FOOTER, and PAGE_NUMBER from repeats. Page-number extraction reads those tags, and `finalize` rewrites earlier pages after `block_emitted`.

**Block kind and domain. Stay rule-based.** `layout_blocks` (`src/packet/layout.py`) and `_guess_kind` (`src/packet/extract.py`) label TITLE, HEADING, and PARAGRAPH from font size or a regex. `apply_domains` (`src/packet/domain.py`) runs the adapters it is given. The stock adapter is `NoOpDomain`. Heading-path accuracy is outside the gate. v1 has one call site, `task=boundary`. The protocol fields (`task`, `labels`, `text`, image, schema) are what a later call site would use.

## Interface

`--decision-config` is parse-only. `packet inspect` keeps building `Pipeline()` with no scorer (`src/packet/cli.py`). No config leaves `PipelineConfig.scorer` as `None`. `--boundary-weights` alone still builds `logistic`. Passing `--decision-config` and `--boundary-weights` together exits 2. A `logistic` config carries a `weights` path. `backend: transformers` and any other id exit 2 at load time, before a parse loop.

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

`max_calls: 0` is the fail-closed default. Copying the example raises on the first `http` call.

**Wire.** `base_url` already includes `/v1`. The adapter appends `/chat/completions` and does not follow redirects. `auth: env` sends `Authorization: Bearer` plus `PACKET_DECISION_API_KEY`, read at request time. `auth: none` sends no Authorization header. No other vendor header. Body fields are `model`, `temperature: 0`, and one user message. `input: features` is one line per `FEATURE_NAMES` entry, `name=value` with six digits after the decimal, in that order, joined by `\n` and ending in `\n`. `input: page_text` is the block text already on the page fact, truncated at `max_chars` (default 4000). It is not `source.text_for`. `response_format` json_schema is sent only when `capabilities.structured_output` is true. The API `logprobs` flag is sent only when `capabilities.logprobs` is true. The adapter reads `choices[0].message.content`. Recorded fixtures are those content strings. The fake server wraps each one in the chat envelope. Label `scores` are parsed from the model JSON. Token logprobs are not turned into label probabilities.

**Images.** `pipeline.stream` still passes `_lazy_render` to `router.extract` on every page. OCR keeps that argument. The object passed into `decide` is separate, and it is passed only when `send_images` and `capabilities.images` are both true. Then the user message content is a list of one `{type: text}` part and one `{type: image_url, image_url: {url}}` part, and the url is `data:image/png;base64,` plus the base64 of the PNG. Either flag false withholds that argument, so `decide` does not call it, and the message has no `image_url`. Default `send_images` is false. Page text and images stay opt-in.

**Secrets.** The only secret read is `PACKET_DECISION_API_KEY`. The loader rejects `api_key`, `authorization`, and `token` case-insensitively at any depth. It rejects a `base_url` with userinfo, and a query parameter `api_key`, `token`, or `authorization`. An unset or empty `PACKET_DECISION_API_KEY` with `auth: env` raises `DecisionBackendError` before any socket. The text names the env var. It omits the key, the Authorization value, URL userinfo, page text, the image, and the response body. Exit 2, including a batch abort.

**Capability flags vs strikes.** `no_structured_output`, `no_logprobs`, and `no_images` are run-level facts. They are set when the config capability is false, and when a 200 omits a declared field. They do not append to `decision_degraded_pages` and do not withhold boundary F1. A content body `0.62` is a usable P(`boundary`): probability 0.62, `scores` null, flags set, no strike.

A strike is a page with no usable P(`boundary`): unparseable content, or a call whose retries are exhausted. That page keeps the heuristic, omits `+model`, and appends the 0-based page index. A probability under 0.4 is the existing floor, not a strike. F1 is withheld, and a comparison would say heuristic-mixed, only when `decision_degraded_pages` is non-empty. Two strikes complete. The third raises `DecisionBackendError`.

**Attempts and abort.** The status set is closed. 200 is parsed. 429, 502, 503, `TimeoutError`, and other connection failures are retried. Every other status, including 500, 408, and 504, and every other 4xx and every 3xx, raises `DecisionBackendError` on that response. Those responses are not retried and do not become a heuristic score. The text is the same redaction: the status and the page index. Exit 2, and the batch aborts. One `decide` gets `1 + max_retries` attempts (`max_retries` default 2). Backoff is 0.5s then 1s on a fake clock. The test does not sleep. Exhaustion of the retried statuses is one strike, the same as unparseable content, not a second abort. A 200 that omits a declared field sets the run-level capability flag and is not a strike. A 200 with no usable P(`boundary`) is a strike. The third-strike text says `unparseable` and the page index, and it omits the body.

`_cmd_parse` catches `DecisionBackendError` the way it catches `BoundaryWeightsError`: one `packet:` line and exit 2. That holds for a single file and for a batch. The batch loop does not continue. Strikes and the `max_calls` count reset at the start of each `parse`. `max_calls` counts `decide` calls, not HTTP attempts, so a page that uses every retry still costs one call. `Pipeline` is built once for the argv list, so the reset is per file, not per process. `max_calls` 0 on `http` raises on the first `decide`. `logistic` ignores `max_calls`. A logistic weights path inside `--decision-config` that is missing or rejected raises `BoundaryWeightsError` and exits 2 before the parse loop, the same catch as `--boundary-weights`. That backend has no timeout and no retry.

**Gate.** `evaluate_gate` stays tolerance-0 boundary F1, the macro floor, and the false-OCR hard fail. `SET_FLOORS` stays builtin `(1.0, None)` and `dlnet-whole` `(0.0, 0.6)`. A `None` macro argument uses the per-fixture floor, so builtin macro is 1.0 as well. `run_fixture` keeps calling `Pipeline()`. Eval does not read `PACKET_DECISION_API_KEY` and does not open a decision socket. When parse was given a config, `--timings` adds `decision_backend` and `decision_degraded_pages` (strike indexes only).

The corpus plan's default-on list stays the bar for a default logistic scorer: disjoint train and holdout, holdout macro and micro both strictly above the heuristic with no packet F1 down, builtin F1 1.0 with the weights loaded, `dlnet-whole` absent from training and its macro at least that plan's logged heuristic 0.98462, empty false-OCR pages, 0.99 still capped at 0.74 without a page-number reset, a probability under 0.4 still adding nothing, and a license that allows shipping the weights. Item 8: the backend id is `logistic`. `http` and `transformers` stay opt-in. The behavioral lock is that an `http` config does not set the default `PipelineConfig.scorer`, and eval does not read the key. A comparison printout of backend id, capabilities, strike count, and macro and micro beside the heuristic is not in this set. It would not write `SET_FLOORS`.

## PoliTax, T4, leakage, comparability, cost

The corpus plan trains `train_logistic` on PoliTax-Split, Trump PDFs only, and holds out Obama, Harris, and Biden as whole files. Schema stays 1. This backend does not replace that trainer, does not fetch the corpus, and does not edit `SET_FLOORS`.

Leakage: default `features` sends 16 numbers. Gate runs leave `page_text` and `send_images` off. Logs store status, latency, model id, and the capability flags. They omit the key, the Authorization value, URL userinfo, and the page body. PoliTax pages are tax forms, so full text stays off until Tom confirms the files have no taxpayer identifiers.

Comparability needs the same gold, the same `LookaheadSplitter`, the same 0.4 floor and 0.74 cap, the same input mode, and an empty strike list. Boundary F1 can then sit beside the heuristic. A `page_text` run is a different input and is printed apart from a `features` run. `temperature: 0` does not make a hosted router bit-stable: the router can change provider between calls, and this plan refuses a vendor provider pin. An `http` F1 is one recorded run, not a second gate, and not a clean PoliTax generalization test. The PDFs are public, so a hosted model may have seen the holdout. Holdout files stay out of prompt-tuning.

Cost: page 0 does not call `decide`, so the corpus plan's 3,311 PoliTax pages are 3,310 `decide` calls. Retries are extra HTTP requests and do not add to that count. `max_calls` is the ceiling and defaults to 0 for `http`. This plan states no vendor price. T4 (Refs GMA-250) is the LightOn OCR measurement on a host Tom names, under its $5 cap. This plan does not start that host and does not add a decision-model download to it. A local decision server is the operator's process. Packet holds the URL.

## Risks

- Applying the HTTP probability as a commit score, past `combined_confidence`, would let 0.99 immediate-commit with no page-number reset. DB-3's cap fixture locks reason `model` and confidence 0.74.
- Counting a capability flag as a strike would abort a healthy plain-text server on page 3. Strikes are only pages with no usable probability.
- A key hidden under another spelling, in userinfo, or in a query string can land in git. The loader rejects those forms.
- Full tax-form text sent to a third-party URL leaks identifiers. Default input is the feature lines.
- An unbounded `http` run is the spend risk. Default `max_calls` 0 fails closed. The per-parse reset keeps one file from consuming the next file's budget.
- A test that imports torch, sleeps the backoff, or dials a public host breaks the portable core. Tests use a fake clock, recorded `message.content` values, and accept only host `127.0.0.1`.

## Out of scope

- Replacing the gap rule, OCR choice, page class, furniture, layout, or domain adapters.
- Editing `SET_FLOORS`, fetching PoliTax, or training weights.
- A CUDA path, a GPU probe, starting vLLM, or any T4 instance change.
- Vendor SDKs, a default base URL, a provider pin, and prompt-tuning.
- ONNX and a `boundary` extra (already dropped, Refs GMA-257).
- Heading-path accuracy, reading order, and calibration inside the gate.
- The comparison printout described above.
- An in-process transformers decision loader, unless Tom answers question 6 with yes. That would be a new ticket. `run_tf.py` stays the OCR bench.
- Downloading model weights in CI.

## Questions for Tom

1. Is the boundary scorer the only v1 call site? Default: yes.
2. Is the default input the 16 schema-1 features, with page text and images off until the config opts in? Default: yes.
3. May `http` run on the PoliTax gate? Default: no. Loopback only, as a side report after the heuristic floor exists. It does not write `SET_FLOORS`.
4. A strike is a page with no usable P(`boundary`). Is aborting on the third strike the right ceiling? Default: yes, exit 2. A plain `0.62` is not a strike.
5. Can `http` or `transformers` become the default scorer? Default: no.
6. Build an in-process transformers decision loader in this set? Default: no. Reserve the id. Leave `bench/ocr_compare/run_tf.py` as the OCR bench. A later loader is explicit, imported lazily, and CPU unless config sets `device`. A yes is a new ticket, not a torch import into `src/packet`.
7. Timeout 30s and 2 retries on 429, 502, 503, timeout, and connection errors, with exhaustion counting as one strike? Default: yes.
8. Keep schema 1? Default: yes.

## Tickets

### DB-1. Protocol and logistic backend

Goal: a `DecisionBackend` protocol with `decide`, and a logistic backend that implements it by calling `LogisticBoundaryScorer.probability`. `BoundaryScorer.probability(features)` stays the method `observe` uses for `--boundary-weights`. The factory accepts `logistic` when the config includes `weights`. `transformers` and any other id exit 2 at load. The default scorer stays `None`.

Files: `src/packet/decision.py`, a wrapper beside `load_logistic_scorer`, `tests/test_decision_logistic.py`, `tests/fixtures/decision/`.

Done when: `tests/fixtures/decision/logistic_facts.json` is two `page_fact_row` dicts, not two 16-vectors. Page 0 is `page_number` null, `page_total` null, `title` null, `text` `""`, `words` `{}`, `lang` null, `size` `[612, 792, 0]`. Page 1 is `page_number` 1, `page_total` null, `title` `"top"`, `text` `"alpha"`, `words` `{"alpha": 1}`, `lang` null, `size` `[612, 792, 0]`. `tests/fixtures/decision/logistic_weights.json` is a document `load_logistic_scorer` accepts: `schema_version` 1, `feature_names` equal to `FEATURE_NAMES`, `mean` sixteen zeros, `std` sixteen ones, `weights` sixteen zeros, `bias` 0. `tests/fixtures/decision/logistic_probability.txt` is `0.500000\n`, the six-decimal form of `probability(featurize_from_facts(facts, 1))`. `tests/fixtures/decision/logistic_capabilities.json` has all three capabilities false. `tests/fixtures/decision/config_logistic.json` points `weights` at that file and loads. `tests/fixtures/decision/config_transformers.json` and `tests/fixtures/decision/config_unknown_backend.json` each fail at load. `tests/test_t9_boundary_scorer.py` still passes, including `_ConstantScorer.probability` and `test_model_probability_capped_at_074_without_page_number_reset`. `run_fixture` still calls `Pipeline()`.

Depends on: none.

### DB-2. HTTP adapter and fake server

Goal: one stdlib adapter for the wire format above. Config selects it. The key comes from `PACKET_DECISION_API_KEY` at request time. A `127.0.0.1` server wraps recorded `message.content` values in the chat envelope. No redirects.

Files: `src/packet/decision_http.py`, `tests/test_decision_http.py`, the fixtures below.

Done when:

- Captured request `tests/fixtures/decision/http_request_features.json` is `model`, `temperature` 0, and one user message. The message text is the sixteen lines from `featurize_from_facts` on `logistic_facts.json` at index 1, `name=value` with six decimal digits, and that text is exactly `has_num=1.000000\nnum_is_one=1.000000\nhas_total=0.000000\nprev_has_num=0.000000\narxiv_stamp=0.000000\ntitle_top=1.000000\ntitle_lower=0.000000\nlog_words=0.693147\nprev_log_words=0.000000\nhead_abstract=0.000000\nhead_intro=0.000000\nhead_affil=0.000000\nprev_refs=0.000000\nmedia_jump=0.000000\nlang_change=0.000000\ntopic_sim=1.000000\n`. Content `tests/fixtures/decision/http_response_structured.json` yields `tests/fixtures/decision/http_structured_expected.json`: label `boundary`, probability 0.62, two label scores from that JSON. Envelope token logprobs are ignored.
- Content `tests/fixtures/decision/http_response_plain.txt` (text `0.62`) yields `tests/fixtures/decision/http_plain_expected.json`: probability 0.62, `scores` null, run-level flags `no_structured_output` and `no_logprobs`, and no strike.
- `tests/fixtures/decision/three_plain_expected.json`: three content bodies `0.62`, zero strikes, those flags set, no raise.
- Content `tests/fixtures/decision/http_response_garbage.txt` yields `tests/fixtures/decision/http_garbage_expected.json`: no probability, one strike, reason `unparseable`.
- `tests/fixtures/decision/third_strike_expected.json`: unparseable pages 1 and 2 are recorded, and the call for page 3 raises `DecisionBackendError`. The exception text says `unparseable` and `3`, and it omits the body.
- One call with `max_retries` 2 makes three attempts. A timeout fixture uses a fake clock and does not sleep the 0.5s and 1s backoffs. Exhaustion is one strike. A 400 and a 401 each raise on that response and are not retried.
- A recorded 500 is one attempt and raises `DecisionBackendError`. The exception text and the timings `error` are the status `500` and the page index. The sentinel is absent. The test does not sleep and does not leave `127.0.0.1`. DB-4 is the test that reads stderr and the timings row.
- A 302 whose `Location` is `http://example.invalid/v1/chat/completions` is not requested. It raises `DecisionBackendError`. The sentinel is absent from the exception.
- An unset `PACKET_DECISION_API_KEY` and an empty one, with `auth: env`, each raise `DecisionBackendError` before any socket. The text names the env var. The connect count is 0.
- `tests/fixtures/decision/max_calls_0_expected.json`: `max_calls` 0 raises on the first `decide`. `tests/fixtures/decision/max_calls_1_expected.json`: the first `decide` may use all of its retries and still counts as one call, and the second `decide` raises.
- `tests/fixtures/decision/http_request_page_text.json`: the input text is 4001 `a` characters. The user message is the first 4000 `a` characters and nothing else.
- `tests/fixtures/decision/http_request_image.json`: both image flags are true. The text part is the feature lines above. The thunk returns `tests/fixtures/decision/one_pixel.png`, the 1×1 PNG whose base64 is `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC`. Content is `[{type: text, text: <those lines>}, {type: image_url, image_url: {url: data:image/png;base64,<that base64>}}]`.
- `tests/fixtures/decision/http_request_images_off.json`: `send_images` is true and `capabilities.images` is false. The message is the feature lines only. It has no `image_url`. The decide thunk is not called.
- The sentinel value of `PACKET_DECISION_API_KEY` is absent from the exception text. `tests/fixtures/decision/config_with_key.json`, a nested `Token`, a `base_url` with userinfo, and a query `api_key` each fail to load.
- The test is unmarked `needs_network`, `needs_gpu`, `needs_models`, and `needs_ocr_extras`. A request host other than `127.0.0.1` fails the test.

Depends on: DB-1.

### DB-3. Boundary call site

Goal: a configured `DecisionBackend` makes `observe` call `decide` for task `boundary` and feed P(`boundary`) through `combined_confidence` and the existing caps. A `BoundaryScorer` still goes through `probability`. The gap rule stays. `router.extract` keeps `_lazy_render`. The object passed into `decide` is withheld unless `send_images` and `capabilities.images` are both true. Eval does not construct the backend.

Files: `src/packet/boundary.py`, `src/packet/pipeline.py`, `tests/test_decision_boundary.py`.

Done when:

- The cap page has no heuristic signal. Fake probability 1.0 and no page-number reset yield confidence 0.74 and reason `model` (`tests/fixtures/decision/cap_expected.json`).
- Probability 0.2 with a heuristic under 0.4 adds nothing (`tests/fixtures/decision/floor_expected.json`).
- Unparseable text keeps the heuristic, omits `+model`, and records page index 1 in `decision_degraded_pages` (`tests/fixtures/decision/degraded_expected.json`). The capability flags are absent from that list.
- Two pages, then flush. Page 0 is unnumbered. Page 1 is numbered 1, with no total and no earlier anchor, so `implied_page_one` and `old_sequence_continues` stay false. Injected 0.45 does not split and the reason is `fallback_below` (`tests/fixtures/decision/silent_045_expected.json`). The same pages at 0.55 split and the reason is `fallback_eager` (`tests/fixtures/decision/silent_055_expected.json`).
- Default `send_images` false: `decide` is not given the thunk. `router.extract` still receives `_lazy_render`.
- `send_images` true and `capabilities.images` false: `decide` is not given the thunk, and the captured request matches `http_request_images_off.json`.

Depends on: DB-1. The test uses a fake in-process backend, so DB-3 can land beside DB-2.

### DB-4. Config, timings, and the checklist guard

Goal: `--decision-config` selects the backend on parse only. `DecisionBackendError` exits 2 for one file and for a batch. Strikes and `max_calls` reset per `parse`. Eval and `inspect` stay on the heuristic.

Files: `src/packet/cli.py`, the loader in `src/packet/decision.py`, `tests/test_decision_config.py`, `tests/fixtures/decision/config_http.json`, `tests/fixtures/decision/config_logistic.json`.

Done when: `config_http.json` (`backend: http`, loopback `base_url` ending in `/v1`, `auth: env`, no key field) builds the HTTP adapter. `config_logistic.json` is a valid loader document: `backend` `logistic` and `weights` pointing at `logistic_weights.json`, and it loads. `tests/fixtures/decision/config_logistic_missing.json` names a weights path that is not on disk, prints one `packet:` line, exits 2, and does not parse. Both CLI flags together exit 2. A single-file parse and a two-file batch each print one `packet:` line and exit 2 on `DecisionBackendError`, and the batch does not parse the second file. The timings `error` for that row is the safe status-and-page text. For the recorded 500, that field is `500` and the page index. The sentinel is absent from stderr and from the timings line. After a one-strike file, the next file in a successful argv starts at zero strikes and zero calls. Eval's source still contains `pipeline or Pipeline()`, and the test shows eval does not read `PACKET_DECISION_API_KEY`. `inspect` builds `Pipeline()` with no scorer. An `http` config does not set the default `PipelineConfig.scorer`. A fake parse with `--timings` writes `decision_backend` and `decision_degraded_pages`.

Depends on: DB-2 and DB-3.
