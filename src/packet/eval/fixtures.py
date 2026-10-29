from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from packet.eval.pdfutil import write_blank_scan_pdf, write_layout_pdf, write_text_pdf
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
        _unnumbered_cover(out),
        _scan(out),
    ]


def fail_mode_dir(root: Path | None = None) -> Path:
    base = root or Path(".packet-eval")
    path = base / "fail-modes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_fail_modes(root: Path | None = None) -> list[NamedFixture]:
    """Known fail-mode fixtures for tests and ``packet eval --viz-dir``.

    Not part of ``build_builtin`` and not scored by ``packet eval``. Keeping
    them off the default gated builtin list means a known failure cannot change
    builtin rows, the gate, or macro F1. ``--viz-dir`` still writes their HTML.
    ``shared-header`` stays in ``build_builtin`` as the passing contrast.
    """
    return [_stripped_numbers(fail_mode_dir(root))]


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


def _unnumbered_cover(out: Path) -> NamedFixture:
    pdf = write_text_pdf(
        out / "unnumbered-cover.pdf",
        [
            ["Fabrikam Service Agreement", "Prepared for Contoso"],
            ["1. Scope", "Services are described here.", "Page 2 of 3"],
            ["2. Term", "The term is one year.", "Page 3 of 3"],
        ],
    )
    return NamedFixture(
        "unnumbered-cover",
        pdf,
        _packet("unnumbered-cover", pdf, [(0, 2)], 3),
        "born-digital",
        "Unnumbered cover is page 1 (implied_page_one on page 0); one document.",
    )


def _stripped_numbers(out: Path) -> NamedFixture:
    """Two documents, one shared header, no ``N of M`` footers.

    The second document opens on ``title_top`` alone (0.45), below the 0.55
    fallback, so look-ahead false-merges it into the first document. Intentional
    known failure; not a gated builtin row. ``shared-header`` still splits.
    """
    pdf = write_layout_pdf(out / "stripped-numbers.pdf", [
        _stripped_page("Chapter A", "Chapter A covers the same binder topic briefly."),
        _stripped_page(None, "Chapter A continues in the same binder with no new heading."),
        _stripped_page("Chapter B", "Chapter B covers the same binder topic briefly."),
        _stripped_page(None, "Chapter B continues in the same binder with no new heading."),
    ])
    return NamedFixture(
        "stripped-numbers",
        pdf,
        _packet("stripped-numbers", pdf, [(0, 1), (2, 3)], 4),
        "born-digital",
        "Known false merge: same header, no N of M footers, title_top alone below 0.55. Not gated.",
    )


def _stripped_page(title: str | None, body: str) -> list:
    lines: list = [("Binder Header", 8, 20)]
    if title:
        lines.append((title, 22, 90, True))
    lines.append((body, 11, 150))
    return lines


def _scan(out: Path) -> NamedFixture:
    pdf = write_blank_scan_pdf(out / "scan.pdf", page_count=2)
    return NamedFixture(
        "scan",
        pdf,
        _packet("scan", pdf, [(0, 1)], 2),
        "clean-scan",
        "No text operators. v0 extractor should emit no OCR blocks.",
    )
