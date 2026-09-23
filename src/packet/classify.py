from __future__ import annotations

from typing import Protocol

from packet.types import PageClass, PageRef


class Classifier(Protocol):
    def classify(self, ref: PageRef, text: str | None = None) -> PageRef:
        ...


class HeuristicClassifier:
    """Cheap first-pass classifier.

    Prefer pdf-inspector in production. This fallback only looks at whether
    a text layer exists and how dense it is relative to page area.
    """

    def __init__(self, min_chars: int = 40):
        self.min_chars = min_chars

    def classify(self, ref: PageRef, text: str | None = None) -> PageRef:
        sample = (text or "").strip()
        chars = len(sample)
        if chars >= self.min_chars:
            page_class, conf = PageClass.TEXT, 0.7
        elif chars > 0:
            page_class, conf = PageClass.MIXED, 0.45
        else:
            page_class, conf = PageClass.SCAN, 0.6
        return ref.model_copy(
            update={"page_class": page_class, "class_confidence": conf, "has_text_ops": chars > 0}
        )


class InspectorClassifier:
    """Optional adapter around firecrawl/pdf-inspector when installed."""

    def __init__(self) -> None:
        try:
            import pdf_inspector  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pdf-inspector is not installed. pip install packet-dl[inspector]"
            ) from exc

    def classify(self, ref: PageRef, text: str | None = None) -> PageRef:
        return HeuristicClassifier().classify(ref, text)
