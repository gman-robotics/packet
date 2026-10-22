"""Per-page fact rows for boundary feature extraction (shared by trainers)."""

from __future__ import annotations

from pathlib import Path

from packet.boundary import BoundaryDetector
from packet.classify import HeuristicClassifier
from packet.extract import NativeTextExtractor, Router
from packet.furniture import FurnitureTagger
from packet.pipeline import _lazy_lines, _lazy_render
from packet.source import PdfPageSource


def page_facts(pdf: Path) -> list[dict]:
    """Native-text pipeline facts for each page (matches ``bench/t8_holdout.page_facts``)."""
    det = BoundaryDetector()
    cl = HeuristicClassifier()
    router = Router([NativeTextExtractor()])
    tg = FurnitureTagger()
    out: list[dict] = []
    with PdfPageSource(pdf) as src:
        for ref in src.pages():
            text = src.text_for(ref.index)
            c = cl.classify(ref, text)
            blocks = tg.tag_page(
                c,
                router.extract(c, text, image=_lazy_render(src, c.index), layout=_lazy_lines(src, c.index)),
            )
            sig = det.observe(c, blocks)
            out.append(
                {
                    **det._page_facts[-1],
                    "page": ref.index,
                    "blocks": len(blocks),
                    "signal": sig,
                }
            )
    return out
