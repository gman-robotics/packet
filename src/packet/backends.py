"""OCR backends and the escalate policy.

Default OCR engine: LightOnOCR-3-1B, grounding mode, on MLX. Fallback: the
PaddleOCR-VL-1.6 page pipeline (layout + VL). Both take a rendered page image,
load lazily on first use, and download weights at runtime into the user's
Hugging Face / PaddleX cache. Nothing is bundled (docs/LICENSE_CHECKLIST.md).
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from collections.abc import Callable
from typing import Any

from packet.extract import Extractor, NativeTextExtractor, call_extractor
from packet.ocr_parse import parse_lighton_grounding, parse_paddle_blocks
from packet.types import Block, PageClass, PageRef

ImageThunk = Callable[[], Any]

log = logging.getLogger(__name__)

_OCR_CLASSES = {PageClass.SCAN, PageClass.MIXED, PageClass.SUSPECT_ENCODING}


class OcrUnavailable(RuntimeError):
    """The OCR engine cannot run here (not installed, no weights, unsupported platform)."""


class DoclingExtractor:
    def supports(self, page_class: PageClass) -> bool:
        return page_class in {PageClass.TEXT, PageClass.MIXED, PageClass.UNKNOWN}

    def extract(self, ref: PageRef, text: str) -> list[Block]:
        raise NotImplementedError("wire DoclingDocument → Block")


def _need_image(image: ImageThunk | None):
    if image is None:
        raise OcrUnavailable("OCR needs a rendered page image")
    return image() if callable(image) else image


class _LazyEngine:
    """Load once on first use; a failed load is cached so it is not retried per page."""

    _loaded: Any = None
    _load_error: OcrUnavailable | None = None

    def _load(self):
        if self._loaded is None:
            if self._load_error is not None:
                raise self._load_error
            try:
                self._loaded = self._load_impl()
            except OcrUnavailable as exc:
                self._load_error = exc
                raise
        return self._loaded

    def _load_impl(self):  # pragma: no cover - overridden
        raise NotImplementedError


class LightOnExtractor(_LazyEngine):
    """LightOnOCR-3-1B, grounding mode, on MLX (mlx-vlm). Default OCR engine."""

    needs_image = True
    model = "lightonai/LightOnOCR-3-1B"
    revision = "23d544085c886bbdace647bcfab91203299417e2"
    mode = "grounding"
    backend = "mlx"

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self._loaded = None
        self._load_error = None

    def supports(self, page_class: PageClass) -> bool:
        return page_class in _OCR_CLASSES

    def _load_impl(self):
        try:
            from huggingface_hub import snapshot_download
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:  # mlx only exists on Apple silicon
            raise OcrUnavailable(f"LightOnOCR needs mlx-vlm: {exc}") from exc
        try:
            path = snapshot_download(self.model, revision=self.revision)
            model, processor = load(path, trust_remote_code=False)
        except Exception as exc:
            raise OcrUnavailable(f"LightOnOCR weights unavailable: {exc}") from exc
        prompt = apply_chat_template(processor, model.config, self.mode, num_images=1)
        return (model, processor, (generate, prompt))

    def extract(self, ref: PageRef, text: str, image: ImageThunk | None = None) -> list[Block]:
        img = _need_image(image)
        model, processor, (generate, prompt) = self._load()
        r = generate(model, processor, prompt, image=[img], max_tokens=self.max_tokens,
                     temperature=0.0, verbose=False)
        raw = r.text if hasattr(r, "text") else str(r)
        return parse_lighton_grounding(raw, ref)


class PaddleExtractor(_LazyEngine):
    """PaddleOCR-VL-1.6 page pipeline (PP-DocLayoutV3 layout + VL recognizer). Fallback engine."""

    needs_image = True
    pipeline_version = "v1.6"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._loaded = None
        self._load_error = None

    def supports(self, page_class: PageClass) -> bool:
        return page_class in _OCR_CLASSES

    def _load_impl(self):
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise OcrUnavailable(f"PaddleOCR-VL needs paddleocr: {exc}") from exc
        try:
            # Orientation/unwarping stay off so bboxes refer to the image we pass in.
            return PaddleOCRVL(pipeline_version=self.pipeline_version, device=self.device,
                               use_doc_orientation_classify=False, use_doc_unwarping=False)
        except Exception as exc:
            raise OcrUnavailable(f"PaddleOCR-VL failed to load: {exc}") from exc

    def extract(self, ref: PageRef, text: str, image: ImageThunk | None = None) -> list[Block]:
        img = _need_image(image)
        pipe = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.png")
            img.save(path)
            results = list(pipe.predict(path))
        raw: list[dict[str, Any]] = []
        for res in results:
            j = res.json if hasattr(res, "json") else res
            j = j.get("res", j) if isinstance(j, dict) else {}
            for b in j.get("parsing_res_list", []):
                raw.append({"label": b.get("block_label"), "bbox": b.get("block_bbox"),
                            "text": b.get("block_content", "")})
        width_px = img.size[0]
        dpi = width_px * 72.0 / float(ref.width or 612.0)
        return parse_paddle_blocks(raw, ref, dpi=dpi)


class FallbackExtractor:
    """Try ``primary``; if it cannot run or crashes, use ``fallback``.

    Engines that do not support the page class are skipped, unless none does
    (``--force-inspect`` on a TEXT page), in which case both are tried. An empty result
    from an engine that ran is final (a blank page stays blank and does not load
    the second engine). A failed engine is never silent: its error is logged as a
    warning and kept in ``last_errors`` for the page, even when the fallback then
    succeeds. If no engine produced a result, raise ``OcrUnavailable`` with every
    reason, so a scanned page never comes back empty silently."""

    needs_image = True

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.last_used = None
        self.last_errors: list[str] = []

    def supports(self, page_class: PageClass) -> bool:
        return self.primary.supports(page_class) or self.fallback.supports(page_class)

    def extract(self, ref: PageRef, text: str, image: ImageThunk | None = None) -> list[Block]:
        self.last_used = None
        self.last_errors = []
        engines = [e for e in (self.primary, self.fallback) if e.supports(ref.page_class)]
        # No engine claims this class (e.g. a TEXT page under --force-inspect, which the
        # escalate policy already chose to OCR): try them all rather than refuse.
        for engine in engines or (self.primary, self.fallback):
            try:
                blocks = call_extractor(engine, ref, text, image)
            except Exception as exc:  # noqa: BLE001 - recorded and logged, then the next engine
                reason = f"{type(engine).__name__}: {type(exc).__name__}: {exc}"
                self.last_errors.append(reason)
                log.warning("OCR engine failed on page %s: %s", ref.index, reason)
                continue
            self.last_used = engine
            return blocks
        raise OcrUnavailable("; ".join(self.last_errors))


class EscalatingExtractor:
    """Per-page escalate policy (DESIGN.md "Escalation policy").

    - TEXT / UNKNOWN: native text layer only; a trustworthy text layer never goes to a GPU.
    - SCAN / SUSPECT_ENCODING: OCR.
    - MIXED: OCR (the thin text layer is usually a page number or stamp); if OCR
      cannot run, keep the native text so the page is not dropped.
    - ``force_inspect``: OCR every page, whatever its class.
    If OCR is unavailable the reason is kept in ``unavailable`` and printed once.
    """

    needs_image = True
    needs_layout = True

    def __init__(self, native: Extractor | None = None, ocr=None, force_inspect: bool = False):
        self.native = native or NativeTextExtractor()
        self.ocr = ocr
        self.force_inspect = force_inspect
        self.unavailable: list[str] = []

    def supports(self, page_class: PageClass) -> bool:
        return True

    def _wants_ocr(self, page_class: PageClass) -> bool:
        return self.ocr is not None and (self.force_inspect or page_class in _OCR_CLASSES)

    def extract(self, ref: PageRef, text: str, image: ImageThunk | None = None,
                layout=None) -> list[Block]:
        if self._wants_ocr(ref.page_class):
            try:
                blocks = call_extractor(self.ocr, ref, text, image)
                if blocks or ref.page_class in {PageClass.SCAN, PageClass.SUSPECT_ENCODING}:
                    return blocks
            except OcrUnavailable as exc:
                if not self.unavailable:
                    print(f"packet: OCR unavailable; scanned pages will be empty and mixed "
                          f"pages keep only their text layer: {exc}", file=sys.stderr)
                self.unavailable.append(str(exc))
        if self.native.supports(ref.page_class) or self.force_inspect:
            return call_extractor(self.native, ref, text, layout=layout)
        return []


def default_ocr_extractor() -> FallbackExtractor:
    """Default OCR: LightOnOCR-3-1B grounding on MLX, PaddleOCR-VL-1.6 pipeline as fallback."""
    return FallbackExtractor(LightOnExtractor(), PaddleExtractor())


def default_extractors(force_inspect: bool = False) -> list[Extractor]:
    """Packet's default routing. ``PACKET_OCR=off`` disables OCR (native text only)."""
    ocr = None if os.environ.get("PACKET_OCR", "").lower() in {"off", "0", "none"} else (
        default_ocr_extractor()
    )
    return [EscalatingExtractor(NativeTextExtractor(), ocr, force_inspect=force_inspect)]
