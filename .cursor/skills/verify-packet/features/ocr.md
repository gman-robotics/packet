# OCR paths (LightOnOCR on MLX, force-inspect, OCR off)

`packet inspect` and `packet parse` send scan and mixed pages to OCR: LightOnOCR-3-1B on MLX first, then PaddleOCR-VL if LightOn is missing or returns nothing. `--force-inspect` sends every page to OCR, text pages included. `PACKET_OCR=off` turns OCR off, so scan pages come out with no text. `packet eval` never runs OCR.

## Sub-features

- `ocr-on-scan`: on an image-only PDF, OCR reads the text and the boundary detector splits the packet from that text.
- `ocr-force-inspect`: on a born-digital PDF, `--force-inspect` replaces native text with OCR blocks.
- `ocr-off`: with `PACKET_OCR=off` the same image-only PDF gives one document and no blocks.

## How to get to it (user POV)

- `packet parse scan.pdf -o out.json` with OCR installed (the default).
- `packet inspect born-digital.pdf --force-inspect` to see OCR output on every page.
- `PACKET_OCR=off packet parse scan.pdf -o out.json` to skip OCR.

## Driving it with control-packet

Preconditions:

- `doctor` reports `ok: true` and `ocr_lighton: ready`. `ready` needs Apple silicon, `mlx-vlm` in the env, and the pinned LightOn weights already in the Hugging Face cache. Doctor never downloads. `unknown` means the probe itself failed (for example `packet` not importable). If it says `missing-mlx`, `missing-weights` or `unknown`, report `ocr-on-scan` and `ocr-force-inspect` as blocked, not verified.
- Build the scan input: `S="$(.cursor/skills/verify-packet/control-packet make-scan)"`. This renders the `multi-doc` fixture to images at 150 dpi, so there is no text layer and every page classifies `scan`. It needs Pillow and pypdfium2 in the packet env. (The builtin `scan` fixture is blank pages, so it cannot prove OCR read anything.)

Steps:

- **OCR on.** `PACKET_VERIFY_CMD_LABEL=ocr-on .cursor/skills/verify-packet/control-packet cli -- parse "$S" -o "$PACKET_VERIFY_ARTIFACTS/ocr-on.json"`. Exit `0`. Stdout says `2 document(s), 4 pages`. In the JSON, the documents span pages 0–1 and 2–3, the titles start with `Northwind Invoice` and `Contoso Lease`, and every block has `source: "vlm"` and `extra.engine: "lighton"`. `ocr-on.stderr` must not contain `OCR unavailable`: when no engine loads, packet still exits `0` and only says so on stderr.
- **OCR off.** `PACKET_OCR=off PACKET_VERIFY_CMD_LABEL=ocr-off .cursor/skills/verify-packet/control-packet cli -- parse "$S" -o "$PACKET_VERIFY_ARTIFACTS/ocr-off.json"`. Exit `0`. One document spanning pages 0–3, `title: null`, no blocks. Compare with OCR on to see that the split came from OCR text.
- **Force inspect.** `PACKET_VERIFY_CMD_LABEL=ocr-force .cursor/skills/verify-packet/control-packet cli -- inspect "$(.cursor/skills/verify-packet/control-packet fixture multi-doc)" --force-inspect`. Exit `0`. Every `block_emitted` id ends in `-oN` (an OCR block), for example `multi-doc-p0-o0`. Without the flag the same file gives native ids such as `multi-doc-p0-b0`.
- **Proof.** Append `feature=ocr input=$S run_id=$PACKET_VERIFY_RUN_ID` to `proof.txt`. Keep both JSON files and the `ocr-*` captures.

## Gotchas

- Each OCR page takes about 2–10 s on an M-series Mac, and the first call also loads the model. The three steps together take about 25 s.
- `inspect` cuts each event payload to about 200 characters, so `"source"` is often cut off. Use the block id suffix (`-oN` for OCR, `-bN` for native) or the parse JSON.
- Native `parse` of `multi-doc` can include a few `vlm/lighton` blocks too: short pages classify `mixed` and the escalating extractor OCRs them. That is expected.
- Stderr carries Hugging Face `Fetching N files` progress lines. They are not errors.
