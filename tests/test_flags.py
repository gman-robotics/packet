"""GMA-457 (LangExtract T2): ``packet.flags`` on each logical document.

Flags describe uncertainty. They do not move a split or a section.
"""
from __future__ import annotations

from unittest.mock import patch

from packet.boundary import BoundarySignal
from packet.citations import citation_id_counts, unresolved_citation_ids
from packet.domain import apply_domains
from packet.eval.fixtures import build_builtin
from packet.eval.metrics import citation_integrity
from packet.eval.pdfutil import write_text_pdf
from packet.events import EventKind
from packet.lookahead import Decision, PageObs
from packet.pipeline import Pipeline, PipelineConfig, flags_for_document
from packet.types import (
    PACKET_FLAGS_KEY,
    Block,
    BlockKind,
    Flag,
    LogicalDocument,
    Packet,
    PacketOrigin,
    Section,
)

_EN = (
    "The tenant shall pay rent monthly to the landlord. The premises include the "
    "apartment, parking space and storage unit. The landlord maintains the building, "
    "repairs plumbing and heating, and the tenant keeps the apartment clean. Rent "
    "increases require written notice. The lease term runs twelve months and renews "
    "unless the tenant gives notice. Security deposit returns after inspection of "
    "the premises by the landlord. Pets require landlord consent."
)
_DE = (
    "Der Mieter zahlt die Miete monatlich an den Vermieter. Die Wohnung und der "
    "Stellplatz sind in der Miete enthalten und der Vermieter ist für die Reparaturen "
    "zuständig, wenn das Gebäude Schäden hat. Die Kaution wird nach der Übergabe der "
    "Wohnung zurückgezahlt, sofern keine Mängel vorliegen und alle Schlüssel da sind."
)


def _kinds(doc: LogicalDocument) -> list[str]:
    flags = doc.extras[PACKET_FLAGS_KEY]
    assert isinstance(flags, list)
    assert all(isinstance(flag, dict) for flag in flags)
    return [flag["kind"] for flag in flags]


def _parse(pdf, config: PipelineConfig | None = None):
    events = []
    packet = Pipeline(config=config).parse(pdf, on_event=events.append)
    decided = [event.payload for event in events if event.kind == EventKind.BOUNDARY_DECIDED]
    return packet, decided


def _ranges(packet: Packet) -> list[tuple[int, int]]:
    return [(doc.page_start, doc.page_end) for doc in packet.documents]


def _section_tree(packet: Packet) -> list:
    def walk(section: Section) -> dict:
        return {
            "id": section.id,
            "heading": section.heading,
            "children": [walk(child) for child in section.children],
        }

    return [[walk(section) for section in doc.sections] for doc in packet.documents]


def _assert_sections_match_without_flags(
    pdf, packet: Packet, config: PipelineConfig | None = None,
) -> None:
    """Section ids and headings match a parse that computes no flags."""
    with patch("packet.pipeline.flags_for_document", return_value=[]):
        plain, _decided = _parse(pdf, config)
    assert _section_tree(packet) == _section_tree(plain)


def _lines(header: str, body: str, page_label: str | None) -> list[str]:
    words = body.split()
    lines = [header]
    chunk: list[str] = []
    for word in words:
        chunk.append(word)
        if len(" ".join(chunk)) > 90:
            lines.append(" ".join(chunk))
            chunk = []
    if chunk:
        lines.append(" ".join(chunk))
    if page_label:
        lines.append(page_label)
    return lines


def _doc(*ids: str, extras: dict | None = None) -> LogicalDocument:
    blocks = [
        Block(id=block_id, page=0, kind=BlockKind.PARAGRAPH, text="body") for block_id in ids
    ]
    return LogicalDocument(
        id="d",
        page_start=0,
        page_end=0,
        sections=[Section(id="s", page_start=0, page_end=0, blocks=blocks)],
        extras=dict(extras or {}),
    )


def _packet(*docs: LogicalDocument) -> Packet:
    return Packet(
        id="p",
        origin=PacketOrigin(path="p.pdf", page_count=1),
        documents=list(docs),
    )


def _obs(index: int, conf: float | None = None, reason: str = "t",
         num: int | None = None, total: int | None = None) -> PageObs:
    signal = None if conf is None else BoundarySignal(index - 1, conf, reason)
    return PageObs(index, signal, num, total)


def _dec(page: int, split: bool, reason: str, conf: float = 0.0) -> Decision:
    return Decision(page, split, reason, conf)


def _flag_kinds(pages, *, opening=None, eof_eligible=False, doc=None, known=None,
                citation_keys=None) -> list[str]:
    flags = flags_for_document(
        pages,
        opening=opening,
        eof_eligible=eof_eligible,
        doc=doc if doc is not None else _doc(),
        known_ids=set() if known is None else known,
        citation_keys=citation_keys,
    )
    return [flag.kind for flag in flags]


# --- weak_boundary ---------------------------------------------------------


def test_weak_boundary_raised_on_fallback_eager(tmp_path):
    pdf = write_text_pdf(tmp_path / "weak.pdf", [
        ["Alpha report begins with the results.", "Page 1 of 2"],
        ["Alpha report finishes those results.", "Page 2 of 2"],
        ["Beta memo stands alone.", "Page 1 of 1"],
    ])
    packet, decided = _parse(pdf)
    assert _ranges(packet) == [(0, 1), (2, 2)], decided
    _assert_sections_match_without_flags(pdf, packet)
    assert decided[2]["reason"] == "fallback_eager"
    assert _kinds(packet.documents[0]) == []
    assert _kinds(packet.documents[1]) == ["weak_boundary"]
    flag = packet.documents[1].extras[PACKET_FLAGS_KEY][0]
    assert flag["page"] == 2
    assert flag["detail"]["reason"] == "fallback_eager"
    assert "page_number_reset" in flag["detail"]["signal"]


def test_weak_boundary_not_raised_on_lookahead_numbering(tmp_path):
    fx = next(item for item in build_builtin(tmp_path) if item.name == "shared-header")
    packet, decided = _parse(fx.pdf)
    assert _ranges(packet) == [(0, 1), (2, 3)]
    _assert_sections_match_without_flags(fx.pdf, packet)
    assert decided[2]["reason"] == "lookahead_numbering"
    assert decided[2]["confidence"] == 0.65
    assert all("weak_boundary" not in _kinds(doc) for doc in packet.documents)
    assert all(flag.kind != "heading_rank_repaired" for doc in packet.documents
               for flag in _models(doc))


def test_high_confidence_without_reset_is_weak_and_with_reset_is_not():
    weak_obs = _obs(3, 0.9, "title_top+header_changed")
    weak = _dec(3, True, "high_confidence", 0.9)
    assert _flag_kinds([(weak, weak_obs)], opening=(weak, weak_obs)) == ["weak_boundary"]

    strong_obs = _obs(3, 0.9, "page_number_reset+title_top", num=1, total=2)
    strong = _dec(3, True, "high_confidence", 0.9)
    assert _flag_kinds([(strong, strong_obs)], opening=(strong, strong_obs)) == []

    numbering = _dec(2, True, "lookahead_numbering", 0.65)
    numbering_obs = _obs(2, 0.65, "page_number_reset+sequence_completed", num=1, total=2)
    assert _flag_kinds(
        [(numbering, numbering_obs)], opening=(numbering, numbering_obs),
    ) == []
    # Page 0 can decide split=True without closing a previous document.
    assert _flag_kinds([(weak, weak_obs)], opening=None) == []


def _models(doc: LogicalDocument) -> list[Flag]:
    return [Flag.model_validate(raw) for raw in doc.extras[PACKET_FLAGS_KEY]]


# --- fallback_decision -----------------------------------------------------


def test_fallback_decision_raised_when_a_signalled_page_does_not_split(tmp_path):
    pdf = write_text_pdf(tmp_path / "near.pdf", [
        ["Quarterly report body starts here.", "Page 1 of 4"],
        ["More discussion of results follows.", "Page 2 of 4"],
        ["A later page restarts numbering.", "Page 1 of 2"],
    ])
    packet, decided = _parse(pdf)
    assert _ranges(packet) == [(0, 2)], decided
    _assert_sections_match_without_flags(pdf, packet)
    assert decided[2]["reason"] == "fallback_below"
    assert decided[2]["signal"] is not None
    assert _kinds(packet.documents[0]) == ["fallback_decision"]
    flag = packet.documents[0].extras[PACKET_FLAGS_KEY][0]
    assert flag["page"] == 2
    assert flag["detail"]["reason"] == "fallback_below"
    assert flag["detail"]["confidence"] == decided[2]["confidence"]


def test_fallback_decision_not_raised_without_a_near_miss(tmp_path):
    pdf = write_text_pdf(tmp_path / "clean.pdf", [
        ["Invoice body.", "Page 1 of 2"],
        ["Invoice end.", "Page 2 of 2"],
        ["Lease body.", "Page 1 of 3"],
        ["Lease more.", "Page 2 of 3"],
        ["Lease end.", "Page 3 of 3"],
    ])
    packet, decided = _parse(pdf)
    assert _ranges(packet) == [(0, 1), (2, 4)]
    _assert_sections_match_without_flags(pdf, packet)
    assert all(row["reason"] != "fallback_below" for row in decided)
    assert all("fallback_decision" not in _kinds(doc) for doc in packet.documents)


def test_fallback_eager_is_not_also_a_fallback_decision():
    decision = _dec(1, True, "fallback_eager", 0.6)
    obs = _obs(1, 0.6, "title_top+header_changed")
    assert _flag_kinds([(decision, obs)], opening=(decision, obs)) == ["weak_boundary"]
    silent = _dec(1, False, "fallback_below", 0.45)
    assert _flag_kinds([(silent, _obs(1))]) == []


# --- eof_flush -------------------------------------------------------------


def test_eof_flush_raised_on_unnumbered_final_page(tmp_path):
    pdf = write_text_pdf(tmp_path / "flush.pdf", [
        _lines("ACME Quarterly Header", _EN, "Page 1 of 2"),
        _lines("ACME Quarterly Header", _EN, "Page 2 of 2"),
        _lines("Neues Dokument Kopf", _DE + " " + _DE, None),
    ])
    packet, decided = _parse(pdf)
    assert _ranges(packet) == [(0, 1), (2, 2)], decided
    _assert_sections_match_without_flags(pdf, packet)
    assert decided[2]["reason"] == "fallback_eager"
    assert _kinds(packet.documents[1]) == ["weak_boundary", "eof_flush"]
    flush = packet.documents[1].extras[PACKET_FLAGS_KEY][1]
    assert flush["kind"] == "eof_flush" and flush["page"] == 2


def test_eof_flush_absent_on_one_document_packet(tmp_path):
    pdf = write_text_pdf(tmp_path / "one.pdf", [
        ["Only document body.", "Page 1 of 1"],
    ])
    packet, _decided = _parse(pdf)
    assert len(packet.documents) == 1
    assert _kinds(packet.documents[0]) == []

    near = write_text_pdf(tmp_path / "near-one.pdf", [
        ["Quarterly report body starts here.", "Page 1 of 4"],
        ["More discussion of results follows.", "Page 2 of 4"],
        ["A later page restarts numbering.", "Page 1 of 2"],
    ])
    packet, _decided = _parse(near)
    assert len(packet.documents) == 1
    assert "eof_flush" not in _kinds(packet.documents[0])


def test_eof_flush_absent_on_one_page_1_of_1_final_doc(tmp_path):
    pdf = write_text_pdf(tmp_path / "reset.pdf", [
        ["Alpha report begins with the results.", "Page 1 of 2"],
        ["Alpha report finishes those results.", "Page 2 of 2"],
        ["Beta memo stands alone.", "Page 1 of 1"],
    ])
    packet, decided = _parse(pdf)
    assert _ranges(packet) == [(0, 1), (2, 2)], decided
    _assert_sections_match_without_flags(pdf, packet)
    assert "eof_flush" not in _kinds(packet.documents[1])


def test_eof_flush_page_number_support_rules():
    def pages(*specs):
        return [(_dec(index, False, "no_signal"), _obs(index, num=num, total=total))
                for index, num, total in specs]

    one_of_one = pages((4, 1, 1))
    assert "eof_flush" not in _flag_kinds(one_of_one, eof_eligible=True)
    assert "eof_flush" in _flag_kinds(pages((4, None, None)), eof_eligible=True)
    assert "eof_flush" in _flag_kinds(pages((4, 5, 9)), eof_eligible=True)

    bare_reset = [(_dec(4, True, "high_confidence", 0.9),
                   _obs(4, 0.9, "page_number_reset", num=None))]
    assert "eof_flush" in _flag_kinds(bare_reset, eof_eligible=True)
    reset_to_one = [(_dec(4, True, "high_confidence", 0.9),
                     _obs(4, 0.9, "page_number_reset", num=1))]
    assert "eof_flush" not in _flag_kinds(reset_to_one, eof_eligible=True)

    continued = pages((5, 1, 3), (6, 2, 3), (7, 3, 3))
    assert "eof_flush" not in _flag_kinds(continued, eof_eligible=True)
    assert "eof_flush" in _flag_kinds(pages((5, None, None), (6, None, None)), eof_eligible=True)
    assert "eof_flush" not in _flag_kinds(pages((5, None, None), (6, 2, 3)), eof_eligible=True)
    assert "eof_flush" in _flag_kinds(pages((5, 1, 2), (6, None, None)), eof_eligible=True)
    restarted = pages((5, 9, 9), (6, 1, 2), (7, 2, 2))
    assert "eof_flush" not in _flag_kinds(restarted, eof_eligible=True)
    # 2 of 3 continues the earlier 1 of 3, not the inserted 1 of 1.
    inserted = pages((5, 1, 3), (6, 1, 1), (7, 2, 3))
    assert "eof_flush" not in _flag_kinds(inserted, eof_eligible=True)
    # A one-document packet never carries eof_flush, even with no page numbers.
    assert "eof_flush" not in _flag_kinds(pages((0, None, None), (1, None, None)), eof_eligible=False)


# --- unresolved_citation ---------------------------------------------------


def test_unresolved_citation_raised_and_repeats_kept(tmp_path):
    seen: list[LogicalDocument] = []

    class Cite:
        name = "cite"

        def applies(self, doc: LogicalDocument) -> bool:
            return True

        def enrich(self, doc: LogicalDocument) -> LogicalDocument:
            seen.append(doc)
            if len(seen) == 1:
                return doc
            real = seen[0].sections[0].blocks[0].id
            doc.extras["sources"] = [real, "missing-block", "missing-block"]
            doc.extras["blurb"] = {"not": "a citation"}
            return doc

    pdf = write_text_pdf(tmp_path / "cite.pdf", [
        ["Alpha report begins with the results.", "Page 1 of 2"],
        ["Alpha report finishes those results.", "Page 2 of 2"],
        ["Beta memo stands alone.", "Page 1 of 1"],
    ])
    config = PipelineConfig(domain_adapters=[Cite()])
    packet = Pipeline(config=config).parse(pdf)
    assert _ranges(packet) == [(0, 1), (2, 2)]
    _assert_sections_match_without_flags(pdf, packet, config)
    assert "unresolved_citation" not in _kinds(packet.documents[0])
    refs = [flag["ref"] for flag in packet.documents[1].extras[PACKET_FLAGS_KEY]
            if flag["kind"] == "unresolved_citation"]
    assert refs == ["missing-block", "missing-block"]
    assert "weak_boundary" in _kinds(packet.documents[1])


def test_unresolved_citation_not_raised_when_ids_resolve(tmp_path):
    class Cite:
        name = "cite"

        def applies(self, doc: LogicalDocument) -> bool:
            return doc.page_start == 0

        def enrich(self, doc: LogicalDocument) -> LogicalDocument:
            doc.extras["sources"] = [doc.sections[0].blocks[0].id]
            doc.extras["also"] = doc.furniture[0].id
            return doc

    pdf = write_text_pdf(tmp_path / "ok.pdf", [
        ["Only document body.", "Page 1 of 1"],
    ])
    packet = Pipeline(config=PipelineConfig(domain_adapters=[Cite()])).parse(pdf)
    assert _kinds(packet.documents[0]) == []
    assert citation_integrity(packet) == 1.0


def test_adapter_cannot_overwrite_packet_flags(tmp_path):
    class Clobber:
        name = "clobber"

        def applies(self, doc: LogicalDocument) -> bool:
            return True

        def enrich(self, doc: LogicalDocument) -> LogicalDocument:
            doc.extras[PACKET_FLAGS_KEY] = [{"kind": "eof_flush", "page": 99, "ref": None, "detail": {}}]
            # A string extras value is a citation id. This note must not be one.
            doc.extras["note"] = {"status": "kept"}
            return doc

    pdf = write_text_pdf(tmp_path / "clobber.pdf", [
        ["Alpha report begins with the results.", "Page 1 of 2"],
        ["Alpha report finishes those results.", "Page 2 of 2"],
        ["Beta memo stands alone.", "Page 1 of 1"],
    ])
    packet = Pipeline(config=PipelineConfig(domain_adapters=[Clobber()])).parse(pdf)
    assert packet.documents[1].extras["note"] == {"status": "kept"}
    assert _kinds(packet.documents[1]) == ["weak_boundary"]
    assert all(flag["page"] != 99 for flag in packet.documents[1].extras[PACKET_FLAGS_KEY])


def test_apply_domains_restores_existing_flags_and_drops_inserted_ones():
    original = [{"kind": "heading_rank_repaired", "page": 1, "ref": "s", "detail": {}}]
    doc = _doc(extras={PACKET_FLAGS_KEY: original, "keep": 1})

    class Wipe:
        name = "wipe"

        def applies(self, doc: LogicalDocument) -> bool:
            return True

        def enrich(self, doc: LogicalDocument) -> LogicalDocument:
            doc.extras[PACKET_FLAGS_KEY].clear()
            doc.extras[PACKET_FLAGS_KEY].append({"kind": "nope"})
            doc.extras["keep"] = 2
            return doc

    restored = apply_domains(doc, [Wipe()])
    assert original == [{"kind": "heading_rank_repaired", "page": 1, "ref": "s", "detail": {}}]
    assert restored.extras[PACKET_FLAGS_KEY] == original
    assert restored.extras["keep"] == 2

    fresh = _doc()

    class Insert:
        name = "insert"

        def applies(self, doc: LogicalDocument) -> bool:
            return True

        def enrich(self, doc: LogicalDocument) -> LogicalDocument:
            doc.extras[PACKET_FLAGS_KEY] = [{"kind": "eof_flush"}]
            doc.extras["cite"] = ["x"]
            return doc

    cleaned = apply_domains(fresh, [Insert()])
    assert PACKET_FLAGS_KEY not in cleaned.extras
    assert cleaned.extras["cite"] == ["x"]


def test_citation_id_counts_match_packet_wide_integrity():
    child = Block(id="child", page=0, kind=BlockKind.PARAGRAPH, text="c")
    nested = _doc("a")
    nested.sections[0].children = [Section(id="sub", page_start=0, page_end=0, blocks=[child])]
    nested.furniture = [Block(id="furn", page=0, kind=BlockKind.FOOTER, text="f")]
    other = _doc("other-block")
    nested.extras = {
        "sources": ["a", "missing", "missing", "other-block"],
        "one": "furn",
        "skip-me": {"id": "missing"},
        "mixed": ["missing", 1],
        "empty": [],
        PACKET_FLAGS_KEY: [{"kind": "eof_flush", "page": 0, "ref": None, "detail": {}}],
    }
    packet = _packet(nested, other)

    assert citation_id_counts(nested, {"a", "furn", "child", "other-block"}) == (
        5, ["missing", "missing"],
    )
    assert unresolved_citation_ids(nested, {"a", "furn", "child", "other-block"}) == [
        "missing", "missing",
    ]
    assert citation_id_counts(nested, {"a", "furn", "child", "other-block"}, {"sources"}) == (
        4, ["missing", "missing"],
    )
    assert citation_id_counts(_doc(), set()) == (0, [])

    assert citation_integrity(packet) == _legacy_citation_integrity(packet)
    assert citation_integrity(packet, {"sources"}) == _legacy_citation_integrity(packet, {"sources"})
    assert citation_integrity(packet, {"one"}) == 1.0
    assert citation_integrity(_packet(_doc())) == 1.0

    no_flags = nested.model_copy(deep=True)
    del no_flags.extras[PACKET_FLAGS_KEY]
    assert citation_integrity(_packet(no_flags, other)) == citation_integrity(packet)


def _legacy_citation_integrity(packet: Packet, citation_keys: set[str] | None = None) -> float:
    """Pre-T2 ``citation_integrity``, kept here so the rewrite stays identical."""
    keys = citation_keys or set()
    ids: set[str] = set()
    for doc in packet.documents:
        for section in doc.sections:
            _legacy_collect(section, ids)
        ids.update(block.id for block in doc.furniture)
    total = 0
    ok = 0
    for doc in packet.documents:
        for extra_key, value in doc.extras.items():
            if keys and extra_key not in keys:
                continue
            cited = _legacy_as_ids(value)
            if not cited:
                continue
            for cid in cited:
                total += 1
                if cid in ids:
                    ok += 1
    if total == 0:
        return 1.0
    return ok / total


def _legacy_collect(section: Section, ids: set[str]) -> None:
    ids.update(block.id for block in section.blocks)
    for child in section.children:
        _legacy_collect(child, ids)


def _legacy_as_ids(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return []
