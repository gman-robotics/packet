# Native boundary loop profile (T3)

**Machine:** `cursor` (Linux 6.12.94+, x86_64, glibc 2.39 — Cursor cloud agent VM)

**Corpus:** one native `dlnet-whole` pass (13 joined packets, 263 pages). No OCR:
`Pipeline()` defaults to `NativeTextExtractor` only; the command also sets `PACKET_OCR=off`.

## Command

```bash
cd <repo-root>
export PYTHONPATH=src
export PACKET_OCR=off
python3 bench/boundary_profile.py
```

Optional: `--cache ~/.cache/packet/dlnet-whole` (default), `--build-dir .packet-eval/dlnet-whole`.

The script mirrors `Pipeline.stream` per page: **extract** is classify + furniture tagging
(including `Router` / native text extract); **observe** is `BoundaryDetector.observe`;
**lookahead** is `LookaheadSplitter.push` and `flush`. Document assembly and eval scoring
are excluded.

## Measured seconds (wall clock, `time.perf_counter`)

| Run | Pages | Extract (s) | Observe (s) | Lookahead (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 263 | 6.134 | 0.231 | 0.002 |
| 2 | 263 | 5.884 | 0.222 | 0.001 |

Run 2 followed builtin `packet eval` (`GATE PASS`) after run 1. **Two runs are not a spread**
(first pass included cold join/cache work; second pass reused local PDFs). Report both; do not
ratio them against trunk (trunk has no split timing).

On this VM, observe plus lookahead is ~0.23 s for 263 pages (~1 ms/page combined). Extract
dominates (~6 s). That split is evidence for profiling only; there is no plan pass/fail speed
threshold on this ticket.

## Rust

Rust / PyO3 scheduling stays **blocked** (plan decision 7). This note does **not** add a Rust
ticket and does **not** authorize Rust. Tom still needs to write a failing latency target before
any hot-loop port; `--timings` on `packet parse` is not this probe (it folds OCR load into
page time).
