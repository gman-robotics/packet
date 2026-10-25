"""OCR backends and the escalate policy.

Default OCR engine: LightOnOCR-3-1B, grounding mode, on MLX. Fallback: the
PaddleOCR-VL-1.6 page pipeline (layout + VL). Both take a rendered page image,
load lazily on first use, and download weights at runtime into the user's
Hugging Face / PaddleX cache. Nothing is bundled (docs/LICENSE_CHECKLIST.md).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import tempfile
import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

from packet.extract import Extractor, NativeTextExtractor, call_extractor
from packet.ocr_parse import parse_lighton_grounding, parse_paddle_blocks
from packet.types import Block, PageClass, PageRef

ImageThunk = Callable[[], Any]

StopReason = Literal["eos", "max_tokens", "loop"]


@dataclass(frozen=True)
class LoopRule:
    window_chars: int = 1500
    max_zlib_ratio: float = 0.02
    check_every_tokens: int = 64


@dataclass(frozen=True)
class Recognition:
    raw: str
    tokens: int
    prompt_tokens: int
    prefill_s: float
    decode_s: float
    stop: StopReason


def _empty_generation(text: str = "") -> SimpleNamespace:
    return SimpleNamespace(text=text, generation_tokens=0, prompt_tokens=0,
                           prompt_tps=0.0, generation_tps=0.0, finish_reason=None)


def _last_generation(out: Any) -> Any:
    if isinstance(out, str):
        return _empty_generation(out)
    if hasattr(out, "text"):
        return out
    last = None
    parts: list[str] = []
    for last in out:
        parts.append(last if isinstance(last, str) else getattr(last, "text", "") or "")
    if last is None:
        return _empty_generation()
    raw = "".join(parts)
    if getattr(last, "text", None) == raw:
        return last
    return SimpleNamespace(
        text=raw,
        generation_tokens=getattr(last, "generation_tokens", 0),
        prompt_tokens=getattr(last, "prompt_tokens", 0),
        prompt_tps=getattr(last, "prompt_tps", 0.0),
        generation_tps=getattr(last, "generation_tps", 0.0),
        finish_reason=getattr(last, "finish_reason", None),
    )


def _loop_fired(text: str, rule: LoopRule) -> bool:
    """True when the last window compresses strictly under ``rule.max_zlib_ratio``."""
    if rule.window_chars <= 0 or len(text) < rule.window_chars:
        return False
    raw = text[-rule.window_chars:].encode("utf-8")
    return len(zlib.compress(raw, 9)) / len(raw) < rule.max_zlib_ratio


def _loop_check_due(checked_at: int, tokens: int, every: int) -> bool:
    if tokens <= checked_at:
        return False
    if every <= 0:
        return True
    if tokens - checked_at >= every:
        return True
    return tokens // every > checked_at // every


def _iter_stream(out: Any):
    if isinstance(out, str):
        yield out
        return
    if hasattr(out, "__next__"):
        yield from out
        return
    if hasattr(out, "text"):
        yield out
        return
    try:
        yield from out
    except TypeError:
        yield out


def _consume_until_loop(out: Any, rule: LoopRule) -> tuple[Any, bool]:
    """Drain ``out``. Stop early when the loop rule fires.

    Token cadence prefers each chunk's ``generation_tokens`` (cumulative). Chunks
    without that count as one yield each. A full drain that never fires matches
    ``_last_generation`` when those counts are present.
    """
    parts: list[str] = []
    last = None
    tokens = 0
    saw_gen = False
    checked_at = 0
    fired = False
    for chunk in _iter_stream(out):
        last = chunk
        if isinstance(chunk, str):
            parts.append(chunk)
            gen = None
        else:
            parts.append(getattr(chunk, "text", "") or "")
            gen = getattr(chunk, "generation_tokens", None)
        if isinstance(gen, int):
            tokens = gen
            saw_gen = True
        elif not saw_gen:
            tokens += 1
        if not _loop_check_due(checked_at, tokens, rule.check_every_tokens):
            continue
        checked_at = tokens
        if _loop_fired("".join(parts), rule):
            fired = True
            break
    if last is None:
        return _empty_generation(), False
    raw = "".join(parts)
    if not fired and saw_gen and getattr(last, "text", None) == raw:
        return last, False
    if isinstance(last, str):
        prompt_tokens, prompt_tps, generation_tps, finish = 0, 0.0, 0.0, None
    else:
        prompt_tokens = getattr(last, "prompt_tokens", 0)
        prompt_tps = getattr(last, "prompt_tps", 0.0)
        generation_tps = getattr(last, "generation_tps", 0.0)
        finish = getattr(last, "finish_reason", None)
    return SimpleNamespace(
        text=raw,
        generation_tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        finish_reason=finish,
    ), fired


def _clean_output(processor: Any, last: Any) -> Any:
    """Apply ``processor.clean_output`` to the joined text, as upstream ``generate()`` does.

    mlx-vlm 0.7.10 defines the hook only for some processors (glm4v, aya_vision, ...).
    LightOnOCR's processor has none today, so this is a no-op there. Token counts and
    ``finish_reason`` are kept; only ``text`` changes.
    """
    clean = getattr(processor, "clean_output", None)
    if not callable(clean) or not hasattr(last, "text"):
        return last
    text = clean(last.text)
    if text == last.text:
        return last
    return SimpleNamespace(
        text=text,
        generation_tokens=getattr(last, "generation_tokens", 0),
        prompt_tokens=getattr(last, "prompt_tokens", 0),
        prompt_tps=getattr(last, "prompt_tps", 0.0),
        generation_tps=getattr(last, "generation_tps", 0.0),
        finish_reason=getattr(last, "finish_reason", None),
    )


def _recognition(max_tokens: int, last: Any, looped: bool) -> Recognition:
    raw = last.text if hasattr(last, "text") else str(last)
    tokens = getattr(last, "generation_tokens", 0)
    tokens = tokens if isinstance(tokens, int) else 0
    prompt_tokens = getattr(last, "prompt_tokens", 0)
    prompt_tokens = prompt_tokens if isinstance(prompt_tokens, int) else 0
    prompt_tps = float(getattr(last, "prompt_tps", 0.0) or 0.0)
    generation_tps = float(getattr(last, "generation_tps", 0.0) or 0.0)
    finish = getattr(last, "finish_reason", None)
    if looped:
        stop: StopReason = "loop"
    else:
        stop = "max_tokens" if finish == "length" or tokens >= max_tokens else "eos"
    return Recognition(
        raw=raw,
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prefill_s=prompt_tokens / prompt_tps if prompt_tps > 0 else 0.0,
        decode_s=tokens / generation_tps if generation_tps > 0 else 0.0,
        stop=stop,
    )


log = logging.getLogger(__name__)

_OCR_CLASSES = {PageClass.SCAN, PageClass.MIXED, PageClass.SUSPECT_ENCODING}


class OcrUnavailable(RuntimeError):
    """The OCR engine cannot run here (not installed, no weights, unsupported platform)."""


class OcrConfigError(ValueError):
    """Invalid OCR settings (unknown engine name, explicit engine not available here)."""

    def __init__(self, message: str, *, pip_extra: str | None = None) -> None:
        super().__init__(message)
        self.pip_extra = pip_extra


@dataclass(frozen=True)
class OCRConfig:
    """Opt-in OCR factory settings. ``Pipeline()`` does not use this; CLI parse/inspect does.

    ``model`` is the Hugging Face repo id for LightOn only (`PACKET_OCR=lighton` or `auto` when
    mlx-vlm is available). It is passed to ``snapshot_download`` with the pinned ``revision``.
    It is rejected for ``paddle`` and for ``auto`` when mlx-vlm is missing.
    """

    engine: str = "auto"
    model: str | None = None


class _LegacyOcrChain:
    """Sentinel: use the historical LightOn-then-Paddle chain (not ``engine=auto``)."""


_LEGACY_OCR_CHAIN = _LegacyOcrChain()


def _mlx_vlm_available() -> bool:
    return importlib.util.find_spec("mlx_vlm") is not None


def _paddle_available() -> bool:
    return importlib.util.find_spec("paddleocr") is not None


class DoclingExtractor:
    def supports(self, page_class: PageClass) -> bool:
        return page_class in {PageClass.TEXT, PageClass.MIXED, PageClass.UNKNOWN}

    def extract(self, ref: PageRef, text: str) -> list[Block]:
        raise NotImplementedError("wire DoclingDocument → Block")


def _need_image(image: ImageThunk | None):
    if image is None:
        raise OcrUnavailable("OCR needs a rendered page image")
    return image() if callable(image) else image


# One process-level cache of loaded OCR engines (GMA-218): every extractor instance,
# every page and every file in a run shares a model once it is loaded. A failed load is
# cached too, so it is not retried per page or per file.
_ENGINE_CACHE: dict[tuple, Any] = {}
_ENGINE_LOCK = threading.Lock()


def clear_engine_cache() -> None:
    """Forget every loaded (or failed) engine; the next use loads again."""
    with _ENGINE_LOCK:
        _ENGINE_CACHE.clear()


class _LazyEngine:
    """Load on first use through the process-level engine cache."""

    _loaded: Any = None
    _load_error: OcrUnavailable | None = None

    def _cache_key(self) -> tuple:
        return (type(self).__name__,)

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        key = self._cache_key()
        with _ENGINE_LOCK:
            if key not in _ENGINE_CACHE:
                try:
                    _ENGINE_CACHE[key] = self._load_impl()
                except OcrUnavailable as exc:
                    _ENGINE_CACHE[key] = exc
            cached = _ENGINE_CACHE[key]
        if isinstance(cached, OcrUnavailable):
            self._load_error = cached
            raise cached
        self._loaded = cached
        return cached

    def _load_impl(self):  # pragma: no cover - overridden
        raise NotImplementedError


class LightOnExtractor(_LazyEngine):
    """LightOnOCR-3-1B, grounding mode, on MLX (mlx-vlm). Default OCR engine."""

    needs_image = True
    model = "lightonai/LightOnOCR-3-1B"
    revision = "23d544085c886bbdace647bcfab91203299417e2"
    mode = "grounding"
    backend = "mlx"

    def __init__(self, max_tokens: int = 4096, model: str | None = None, explicit_model: bool = False):
        self.max_tokens = max_tokens
        self._explicit_model = explicit_model or model is not None
        if model is not None:
            self.model = model
        self._loaded = None
        self._load_error = None

    def supports(self, page_class: PageClass) -> bool:
        return page_class in _OCR_CLASSES

    def _cache_key(self) -> tuple:
        return (type(self).__name__, self.model, self.revision, self.mode)

    def _load_impl(self):
        try:
            from huggingface_hub import snapshot_download
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:  # mlx only exists on Apple silicon
            raise OcrUnavailable(f"LightOnOCR needs mlx-vlm: {exc}") from exc
        try:
            from mlx_vlm import stream_generate
        except ImportError:
            stream_generate = generate
        try:
            path = snapshot_download(self.model, revision=self.revision)
            model, processor = load(path, trust_remote_code=False)
            prompt = apply_chat_template(processor, model.config, self.mode, num_images=1)
        except Exception as exc:
            if self._explicit_model:
                raise OcrConfigError(
                    f"PACKET_OCR_MODEL={self.model!r}: failed to load ({exc})",
                ) from exc
            raise OcrUnavailable(f"LightOnOCR failed to load: {exc}") from exc
        return (model, processor, (stream_generate, prompt))

    def recognize(self, img, *, loop_rule: LoopRule | None = None) -> Recognition:
        """Raw grounding text, token counts, stage times, and stop reason for one page image.

        ``loop_rule=None`` drains the stream. A rule stops early with ``stop="loop"``
        when the zlib window fires. ``truncated`` is not decided here.
        """
        model, processor, (stream_generate, prompt) = self._load()
        out = stream_generate(model, processor, prompt, image=[img], max_tokens=self.max_tokens,
                              temperature=0.0, verbose=False)
        if loop_rule is None:
            last, looped = _last_generation(out), False
        else:
            last, looped = _consume_until_loop(out, loop_rule)
        return _recognition(self.max_tokens, _clean_output(processor, last), looped)

    def extract(self, ref: PageRef, text: str, image: ImageThunk | None = None) -> list[Block]:
        img = _need_image(image)
        rec = self.recognize(img)
        return parse_lighton_grounding(rec.raw, ref)


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

    def _cache_key(self) -> tuple:
        return (type(self).__name__, self.pipeline_version, self.device)

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
        # Wall time of every OCR call (render + model), for ``packet parse --timings``.
        self.ocr_timings: list[dict] = []

    def supports(self, page_class: PageClass) -> bool:
        return True

    def _wants_ocr(self, page_class: PageClass) -> bool:
        return self.ocr is not None and (self.force_inspect or page_class in _OCR_CLASSES)

    def extract(self, ref: PageRef, text: str, image: ImageThunk | None = None,
                layout=None) -> list[Block]:
        if self._wants_ocr(ref.page_class):
            start = time.perf_counter()
            try:
                blocks = call_extractor(self.ocr, ref, text, image)
                self.ocr_timings.append({"page": ref.index, "seconds": time.perf_counter() - start})
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


def _default_ocr_chain() -> FallbackExtractor:
    """LightOnOCR-3-1B grounding on MLX, PaddleOCR-VL-1.6 pipeline as fallback."""
    return FallbackExtractor(LightOnExtractor(), PaddleExtractor())


def build_ocr_extractor(config: OCRConfig) -> Extractor:
    """Resolve ``config`` to a concrete OCR engine. CUDA ``auto`` waits on T4 (not wired here)."""
    engine = config.engine.strip().lower()
    if config.model and engine == "paddle":
        raise OcrConfigError(
            "PACKET_OCR_MODEL is only supported with PACKET_OCR=lighton or auto (mlx-vlm)"
        )
    if engine == "auto":
        if config.model:
            if not _mlx_vlm_available():
                raise OcrConfigError(
                    "PACKET_OCR=auto with PACKET_OCR_MODEL requires mlx-vlm",
                    pip_extra="ocr-mlx",
                )
            return LightOnExtractor(model=config.model, explicit_model=True)
        if _mlx_vlm_available():
            return FallbackExtractor(LightOnExtractor(), PaddleExtractor())
        if not _paddle_available():
            raise OcrConfigError(
                "PACKET_OCR=auto: no mlx-vlm and paddleocr is not installed",
                pip_extra="ocr-paddle",
            )
        return PaddleExtractor()
    if engine == "lighton":
        if not _mlx_vlm_available():
            raise OcrConfigError(
                "PACKET_OCR=lighton: mlx-vlm is not installed",
                pip_extra="ocr-mlx",
            )
        return LightOnExtractor(
            model=config.model,
            explicit_model=config.model is not None,
        )
    if engine == "paddle":
        if not _paddle_available():
            raise OcrConfigError(
                "PACKET_OCR=paddle: paddleocr is not installed",
                pip_extra="ocr-paddle",
            )
        return PaddleExtractor()
    raise OcrConfigError(f"unknown OCR engine {config.engine!r}")


def default_ocr_extractor(config: OCRConfig | None = None) -> Extractor:
    """Default OCR routing. No ``config`` keeps the historical LightOn-then-Paddle chain."""
    if config is None:
        return _default_ocr_chain()
    return build_ocr_extractor(config)


def _ocr_config_from_env() -> OCRConfig | None | _LegacyOcrChain:
    """``None`` disables OCR. ``_LEGACY_OCR_CHAIN`` keeps the historical default chain."""
    raw = os.environ.get("PACKET_OCR", "").strip()
    model = os.environ.get("PACKET_OCR_MODEL", "").strip() or None
    if not raw:
        if model:
            raise OcrConfigError("PACKET_OCR_MODEL needs PACKET_OCR=lighton or auto")
        return _LEGACY_OCR_CHAIN
    lowered = raw.lower()
    if lowered in {"off", "0", "none"}:
        return None
    if lowered in {"on", "1", "true", "yes"}:
        if model:
            raise OcrConfigError("PACKET_OCR_MODEL needs PACKET_OCR=lighton or auto")
        return _LEGACY_OCR_CHAIN
    if lowered == "auto":
        return OCRConfig(engine="auto", model=model)
    if lowered in {"lighton", "mlx"}:
        return OCRConfig(engine="lighton", model=model)
    if lowered == "paddle":
        if model:
            raise OcrConfigError(
                "PACKET_OCR_MODEL is only supported with PACKET_OCR=lighton or auto (mlx-vlm)"
            )
        return OCRConfig(engine="paddle", model=None)
    raise OcrConfigError(f"unknown PACKET_OCR value {raw!r}")


def default_extractors(force_inspect: bool = False) -> list[Extractor]:
    """Packet's default routing. ``PACKET_OCR=off`` disables OCR (native text only)."""
    cfg = _ocr_config_from_env()
    if cfg is None:
        ocr = None
    elif cfg is _LEGACY_OCR_CHAIN:
        ocr = default_ocr_extractor()
    else:
        ocr = default_ocr_extractor(cfg)
    return [EscalatingExtractor(NativeTextExtractor(), ocr, force_inspect=force_inspect)]
