from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import pypdfium2 as pdfium

from packet.types import PacketOrigin, PageRef


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PdfPageSource:
    """Lazy page iterator. Opens the file once; does not render pages."""

    def __init__(self, path: str | Path, packet_id: str | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.packet_id = packet_id or self.path.stem
        self._doc = pdfium.PdfDocument(str(self.path))

    @property
    def page_count(self) -> int:
        return len(self._doc)

    def origin(self) -> PacketOrigin:
        return PacketOrigin(
            path=str(self.path),
            sha256=file_sha256(self.path),
            media_type="application/pdf",
            page_count=self.page_count,
        )

    def pages(self) -> Iterator[PageRef]:
        for index, page in enumerate(self._doc):
            width, height = page.get_size()
            text = page.get_textpage().get_text_bounded() if page.get_textpage() else ""
            has_text = bool(text and text.strip())
            yield PageRef(
                packet_id=self.packet_id,
                index=index,
                width=float(width),
                height=float(height),
                rotation=int(page.get_rotation()),
                has_text_ops=has_text,
            )
            page.close()

    def text_for(self, index: int) -> str:
        page = self._doc[index]
        try:
            tp = page.get_textpage()
            return tp.get_text_bounded() if tp else ""
        finally:
            page.close()

    def lines_for(self, index: int):
        """Positioned text lines with font size and weight (``packet.layout.TextLine``)."""
        from packet.layout import read_lines

        page = self._doc[index]
        try:
            if page.get_rotation() % 360:
                return []  # char boxes are unrotated user space: use the plain-text path
            left, _bottom, _right, top = page.get_cropbox()
            tp = page.get_textpage()
            return read_lines(tp, top, left) if tp else []
        finally:
            page.close()

    def render(self, index: int, longest_px: int = 1540):
        """Render page ``index`` to a PIL image whose longest side is ``longest_px``
        (LightOnOCR's recommended input size; about 140 DPI for US Letter)."""
        page = self._doc[index]
        try:
            width, height = page.get_size()
            scale = longest_px / max(width, height, 1.0)
            img = page.render(scale=scale).to_pil().convert("RGB")
        finally:
            page.close()
        if max(img.size) != longest_px:  # rounding: snap the longest side exactly
            w, h = img.size
            r = longest_px / max(w, h)
            img = img.resize((max(1, round(w * r)), max(1, round(h * r))))
        return img

    def close(self) -> None:
        self._doc.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
