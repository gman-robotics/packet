"""Seams for extractors that are not in the default extra set."""

from __future__ import annotations

from packet.types import Block, PageClass, PageRef


class DoclingExtractor:
    def supports(self, page_class: PageClass) -> bool:
        return page_class in {PageClass.TEXT, PageClass.MIXED, PageClass.UNKNOWN}

    def extract(self, ref: PageRef, text: str) -> list[Block]:
        raise NotImplementedError("wire DoclingDocument → Block")


class PaddleExtractor:
    def supports(self, page_class: PageClass) -> bool:
        return page_class in {PageClass.SCAN, PageClass.MIXED, PageClass.SUSPECT_ENCODING}

    def extract(self, ref: PageRef, text: str) -> list[Block]:
        raise NotImplementedError("wire PaddleOCR-VL-1.6 full pipeline output → Block")


_OCR_CLASSES = {PageClass.SCAN, PageClass.MIXED, PageClass.SUSPECT_ENCODING}


class LightOnExtractor:
    """LightOnOCR-3-1B, grounding mode, on MLX. Default OCR engine (GMA-207)."""

    model = "lightonai/LightOnOCR-3-1B"
    revision = "23d544085c886bbdace647bcfab91203299417e2"
    mode = "grounding"
    backend = "mlx"

    def supports(self, page_class: PageClass) -> bool:
        return page_class in _OCR_CLASSES

    def extract(self, ref: PageRef, text: str) -> list[Block]:
        raise NotImplementedError("wire LightOnOCR-3-1B grounding (MLX) output → Block")


class FallbackExtractor:
    """Try ``primary``; if it raises (unavailable or failed), use ``fallback``."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.last_used = None

    def supports(self, page_class: PageClass) -> bool:
        return self.primary.supports(page_class) or self.fallback.supports(page_class)

    def extract(self, ref: PageRef, text: str) -> list[Block]:
        if self.primary.supports(ref.page_class):
            try:
                blocks = self.primary.extract(ref, text)
                self.last_used = self.primary
                return blocks
            except Exception:
                pass
        self.last_used = self.fallback
        return self.fallback.extract(ref, text)


def default_ocr_extractor() -> FallbackExtractor:
    """Default OCR: LightOnOCR-3-1B grounding on MLX, PaddleOCR-VL-1.6 pipeline as fallback."""
    return FallbackExtractor(LightOnExtractor(), PaddleExtractor())
