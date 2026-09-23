from __future__ import annotations

from pathlib import Path


def write_text_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Write a born-digital multi-page PDF. Each page is a list of text lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    objects: list[bytes] = []

    def add(obj: str | bytes) -> int:
        if isinstance(obj, str):
            obj = obj.encode("latin-1")
        objects.append(obj)
        return len(objects)

    add("<< /Type /Catalog /Pages 2 0 R >>")
    page_ids: list[int] = []
    content_ids: list[int] = []

    add("<< /Type /Pages /Kids [] /Count 0 >>")
    font_id = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    content_bodies: list[bytes] = []
    for lines in pages:
        stream = _page_stream(lines)
        content_bodies.append(stream)

    for stream in content_bodies:
        cid = add(f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream")
        content_ids.append(cid)

    for cid in content_ids:
        pid = add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {cid} 0 R >>"
        )
        page_ids.append(pid)

    kids = " ".join(f"{i} 0 R" for i in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("latin-1"))
        out.extend(body)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects)+1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    path.write_bytes(out)
    return path


def write_blank_scan_pdf(path: Path, page_count: int = 2) -> Path:
    """Pages with empty content streams (no text operators)."""
    return write_text_pdf(path, [[] for _ in range(page_count)])


def _page_stream(lines: list[str]) -> bytes:
    if not lines:
        return b""
    cmds = ["BT", "/F1 12 Tf", "50 740 Td", "16 TL"]
    first = True
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if first:
            cmds.append(f"({safe}) Tj")
            first = False
        else:
            cmds.append(f"T* ({safe}) Tj")
    cmds.append("ET")
    return ("\n".join(cmds) + "\n").encode("latin-1")
