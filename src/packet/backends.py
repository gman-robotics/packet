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
        raise NotImplementedError("wire PaddleOCR-VL page output → Block")
