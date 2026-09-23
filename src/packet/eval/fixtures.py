from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from packet.eval.pdfutil import write_blank_scan_pdf, write_text_pdf
from packet.types import LogicalDocument, Packet, PacketOrigin


@dataclass
class NamedFixture:
    name: str
    pdf: Path
    gold: Packet
    condition: str
    notes: str = ""


def builtin_dir(root: Path | None = None) -> Path:
    base = root or Path(".packet-eval")
    path = base / "builtin"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_builtin(root: Path | None = None) -> list[NamedFixture]:
    out = builtin_dir(root)
    return [
        _text_layer(out),
        _multi_doc(out),
        _shared_header(out),
        _scan(out),
    ]


def _packet(name: str, pdf: Path, docs: list[tuple[int, int]], pages: int) -> Packet:
    logical = [
        LogicalDocument(id=f"{name}-d{i}", page_start=a, page_end=b, title=f"doc-{i}")
        for i, (a, b) in enumerate(docs)
    ]
    return Packet(
        id=name,
        origin=PacketOrigin(path=str(pdf), page_count=pages),
        documents=logical,
    )


def _text_layer(out: Path) -> NamedFixture:
    pdf = write_text_pdf(
        out / "text-layer.pdf",
        [
            ["ACME Annual Report", "1. Overview", "Body of page one.", "Page 1 of 3"],
            ["Section continues.", "More body text on page two.", "Page 2 of 3"],
            ["Closing remarks.", "Page 3 of 3"],
        ],
    )
    return NamedFixture(
        "text-layer",
        pdf,
        _packet("text-layer", pdf, [(0, 2)], 3),
        "born-digital",
        "Single document, zero gold boundaries.",
    )


def _multi_doc(out: Path) -> NamedFixture:
    pdf = write_text_pdf(
        out / "multi-doc.pdf",
        [
            ["Northwind Invoice", "Amount due: 12.00", "Page 1 of 2"],
            ["Thank you for your business.", "Page 2 of 2"],
            ["Contoso Lease", "Premises: 1 Main St", "Page 1 of 2"],
            ["Signatures follow.", "Page 2 of 2"],
        ],
    )
    return NamedFixture(
        "multi-doc",
        pdf,
        _packet("multi-doc", pdf, [(0, 1), (2, 3)], 4),
        "born-digital",
        "Two docs, page numbers restart, distinct titles.",
    )


def _shared_header(out: Path) -> NamedFixture:
    pdf = write_text_pdf(
        out / "shared-header.pdf",
        [
            ["Binder Header", "Chapter A body.", "Page 1 of 2"],
            ["Binder Header", "Chapter A continued.", "Page 2 of 2"],
            ["Binder Header", "Chapter B body.", "Page 1 of 2"],
            ["Binder Header", "Chapter B continued.", "Page 2 of 2"],
        ],
    )
    return NamedFixture(
        "shared-header",
        pdf,
        _packet("shared-header", pdf, [(0, 1), (2, 3)], 4),
        "born-digital",
        "Same header fingerprint; boundary must come from page-number reset.",
    )


def _scan(out: Path) -> NamedFixture:
    pdf = write_blank_scan_pdf(out / "scan.pdf", page_count=2)
    return NamedFixture(
        "scan",
        pdf,
        _packet("scan", pdf, [(0, 1)], 2),
        "clean-scan",
        "No text operators. v0 extractor should emit no OCR blocks.",
    )
