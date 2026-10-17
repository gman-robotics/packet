"""Layout helpers shared by boundary detection and boundary model features."""

from __future__ import annotations

from packet.types import Block, BlockKind

_FURNITURE = {BlockKind.HEADER, BlockKind.FOOTER, BlockKind.PAGE_NUMBER}


def _head_candidate(b: Block) -> bool:
    x = b.extra or {}
    return bool(x.get("layout") and x.get("band") == "top" and x.get("small"))


def title_position(blocks: list[Block]) -> str | None:
    """``top`` when the first non-furniture block is a TITLE, ``lower`` for any other TITLE."""
    body = [b for b in blocks if b.kind not in _FURNITURE and not _head_candidate(b)]
    if body and body[0].kind == BlockKind.TITLE:
        return "top"
    return "lower" if any(b.kind == BlockKind.TITLE for b in body) else None


def size_jump(
    prev: tuple[float | None, float | None, int],
    curr: tuple[float | None, float | None, int],
) -> bool:
    if None in prev[:2] or None in curr[:2]:
        return False
    if prev[2] != curr[2]:
        return True
    dw = abs((prev[0] or 0) - (curr[0] or 0))
    dh = abs((prev[1] or 0) - (curr[1] or 0))
    return dw > 20 or dh > 20
