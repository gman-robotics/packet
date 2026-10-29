from __future__ import annotations

import re
from dataclasses import dataclass

from packet.types import PACKET_FLAGS_KEY, Block, BlockKind, Flag, LogicalDocument, Packet, Section


class Assembler:
    """Turn a flat block stream into a section tree."""

    def build_document(
        self,
        doc_id: str,
        blocks: list[Block],
        title: str | None = None,
    ) -> LogicalDocument:
        body, furniture = _split_furniture(blocks)
        if title is None:
            title = _guess_title(body)
        pages = [b.page for b in blocks] or [0]
        root_sections = _nest(body)
        root_sections, rank_flags = repair_ranks(root_sections)
        document = LogicalDocument(
            id=doc_id,
            title=title,
            page_start=min(pages),
            page_end=max(pages),
            sections=root_sections,
            furniture=furniture,
        )
        if rank_flags:
            document.extras[PACKET_FLAGS_KEY] = [flag.model_dump() for flag in rank_flags]
        return document

    def attach(self, packet: Packet, doc: LogicalDocument) -> Packet:
        packet.documents.append(doc)
        return packet


def _split_furniture(blocks: list[Block]) -> tuple[list[Block], list[Block]]:
    furniture_kinds = {
        BlockKind.HEADER,
        BlockKind.FOOTER,
        BlockKind.PAGE_NUMBER,
    }
    body = [b for b in blocks if b.kind not in furniture_kinds]
    furniture = [b for b in blocks if b.kind in furniture_kinds]
    return body, furniture


def _guess_title(blocks: list[Block]) -> str | None:
    for block in blocks[:8]:
        if block.kind in {BlockKind.TITLE, BlockKind.HEADING} and block.text:
            return block.text.split("\n", 1)[0][:200]
    if blocks and blocks[0].text:
        return blocks[0].text.split("\n", 1)[0][:200]
    return None


def _nest(blocks: list[Block]) -> list[Section]:
    if not blocks:
        return []

    roots: list[Section] = []
    stack: list[Section] = []
    anon_count = 0

    def open_section(block: Block, level: int) -> Section:
        pages = block.page
        return Section(
            id=f"{block.id}-sec",
            heading=block.text.split("\n", 1)[0][:200],
            level=level,
            page_start=pages,
            page_end=pages,
            blocks=[block],
        )

    for block in blocks:
        if block.kind in {BlockKind.HEADING, BlockKind.TITLE}:
            level = 0 if block.kind == BlockKind.TITLE else (block.heading_level or 1)
            section = open_section(block, level)
            while stack and stack[-1].level >= level:
                closed = stack.pop()
                if stack:
                    stack[-1].page_end = max(stack[-1].page_end, closed.page_end)
            if stack:
                stack[-1].children.append(section)
            else:
                roots.append(section)
            stack.append(section)
            continue

        if not stack:
            anon_count += 1
            section = Section(
                id=f"anon-{anon_count}",
                heading=None,
                level=1,
                page_start=block.page,
                page_end=block.page,
            )
            roots.append(section)
            stack.append(section)
        stack[-1].blocks.append(block)
        stack[-1].page_end = max(stack[-1].page_end, block.page)

    return roots


# A first component above this is a year or amount ("2024 Budget", "100", "100.00").
_BARE_INTEGER_MAX = 99
_DECIMAL = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)(\.)?(?:\s+|$)")
_SECTION = re.compile(r"^section\s+(\d{1,3}(?:\.\d{1,3})*)(\.)?(?:\s+|$)", re.IGNORECASE)
_ARTICLE = re.compile(r"^(article\s+)([IVXLCDM]+|\d{1,3})\.?(?:\s+|$)", re.IGNORECASE)
_ROMAN = re.compile(r"^M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$")
_ROMAN_VALUE = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


@dataclass(frozen=True)
class _Numbering:
    depth: int
    components: tuple[str, ...] | None
    token: str


def repair_ranks(sections: list[Section]) -> tuple[list[Section], list[Flag]]:
    """Reparent numbered sections when numbering grammar disagrees with heading level.

    Grammar is read from the first line of ``section.heading``. The number token
    must start that line and be followed by whitespace or the end of the line.

    Decimal outlines are ``1``, ``1.``, ``1.1``, and ``1.1.2``. Depth is the
    component count. Each component is 1 to 3 digits and keeps its written
    digits, so ``1.01`` is not ``1.1``. A first component greater than 99 is
    not an outline, so ``2024 Budget``, ``100``, and ``100.00`` are left
    alone. ``Section 1.2`` is the same decimal outline; the keyword is not part
    of the number. The flag token is the components joined with dots (``1.1``).

    ``Article IV`` and ``ARTICLE 4`` are depth 1, roman or arabic, value 1..999.
    They are not decimal prefixes: ``4.1`` does not become a child of
    ``Article IV``. A level-0 section (a title) is never reparented and its
    level is not changed. A numbered title can still be a decimal prefix parent.

    A numbered section of depth ``d`` becomes a child of the nearest preceding
    decimal section of depth ``d - 1`` whose components are a prefix, inserted
    among that parent's children in document order. If no such parent exists,
    the section stays where ``_nest`` put it. Unnumbered sections are not
    moved. When grammar and ``heading_level`` disagree, ``level`` is set from
    depth. A level change without a new parent does not emit a flag.

    Each section whose parent changes emits one ``heading_rank_repaired`` flag.
    Old ancestors' ``page_end`` is recomputed from the blocks and children that
    remain. New ancestors' ``page_end`` covers the moved subtree. A tree with
    no numbered headings is returned unchanged, with no flags.
    """
    flat = _flatten(sections)
    numbered: list[tuple[Section, Section | None, _Numbering]] = []
    for section, parent in flat:
        numbering = _parse_numbering(section.heading)
        if numbering is not None:
            numbered.append((section, parent, numbering))
    if not numbered:
        return sections, []

    moves: list[tuple[Section, Section | None, Section, _Numbering]] = []
    level_sets: list[tuple[Section, int]] = []
    seen: list[tuple[Section, _Numbering]] = []
    for section, parent, numbering in numbered:
        if section.level != 0 and section.level != numbering.depth:
            level_sets.append((section, numbering.depth))
        desired = None if section.level == 0 else _prefix_parent(seen, numbering)
        if desired is not None and desired is not parent:
            moves.append((section, parent, desired, numbering))
        seen.append((section, numbering))
    if not moves and not level_sets:
        return sections, []

    for section, depth in level_sets:
        section.level = depth

    parent_of = {id(section): parent for section, parent in flat}
    order = {id(section): index for index, (section, _parent) in enumerate(flat)}
    flags: list[Flag] = []
    for section, old_parent, new_parent, numbering in moves:
        _detach(sections, old_parent, section)
        _recompute_page_ends(old_parent, parent_of)
        _insert_in_order(new_parent, section, order)
        parent_of[id(section)] = new_parent
        _cover_parents(section, parent_of)
        flags.append(Flag(
            kind="heading_rank_repaired",
            page=section.page_start,
            ref=section.id,
            detail={
                "from_parent": None if old_parent is None else old_parent.id,
                "to_parent": new_parent.id,
                "numbering": numbering.token,
            },
        ))
    return sections, flags


def _parse_numbering(heading: str | None) -> _Numbering | None:
    if not heading:
        return None
    line = heading.split("\n", 1)[0]
    article = _ARTICLE.match(line)
    if article:
        raw = article.group(2)
        if raw.isdigit():
            value = int(raw)
            if not 1 <= value <= 999:
                return None
        elif _roman_to_int(raw) is None:
            return None
        word = article.group(1).rstrip()
        return _Numbering(depth=1, components=None, token=f"{word} {raw}")
    section = _SECTION.match(line)
    if section is not None:
        return _decimal_numbering(section.group(1))
    decimal = _DECIMAL.match(line)
    if decimal is not None:
        return _decimal_numbering(decimal.group(1))
    return None


def _decimal_numbering(body: str) -> _Numbering | None:
    parts = body.split(".")
    if int(parts[0]) > _BARE_INTEGER_MAX:
        return None
    components = tuple(parts)
    token = ".".join(components)
    return _Numbering(depth=len(components), components=components, token=token)


def _roman_to_int(token: str) -> int | None:
    text = token.upper()
    if not text or _ROMAN.fullmatch(text) is None:
        return None
    total = 0
    prev = 0
    for char in reversed(text):
        value = _ROMAN_VALUE[char]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    if not 1 <= total <= 999:
        return None
    return total


def _prefix_parent(earlier: list[tuple[Section, _Numbering]], child: _Numbering) -> Section | None:
    if child.components is None or child.depth < 2:
        return None
    want = child.depth - 1
    prefix = child.components[:want]
    for section, numbering in reversed(earlier):
        if (
            numbering.components is not None
            and numbering.depth == want
            and numbering.components == prefix
        ):
            return section
    return None


def _flatten(
    sections: list[Section], parent: Section | None = None,
) -> list[tuple[Section, Section | None]]:
    ordered: list[tuple[Section, Section | None]] = []
    for section in sections:
        ordered.append((section, parent))
        ordered.extend(_flatten(section.children, section))
    return ordered


def _detach(roots: list[Section], parent: Section | None, section: Section) -> None:
    container = roots if parent is None else parent.children
    for index, item in enumerate(container):
        if item is section:
            del container[index]
            return
    raise ValueError(f"section {section.id} was not attached to its parent")


def _subtree_page_end(section: Section) -> int:
    end = section.page_end
    for child in section.children:
        end = max(end, _subtree_page_end(child))
    return end


def _recompute_page_ends(node: Section | None, parent_of: dict[int, Section | None]) -> None:
    while node is not None:
        end = node.page_start
        for block in node.blocks:
            end = max(end, block.page)
        for child in node.children:
            end = max(end, _subtree_page_end(child))
        node.page_end = end
        node = parent_of.get(id(node))


def _insert_in_order(parent: Section, section: Section, order: dict[int, int]) -> None:
    rank = order[id(section)]
    children = parent.children
    index = len(children)
    for cursor, child in enumerate(children):
        if order[id(child)] > rank:
            index = cursor
            break
    children.insert(index, section)


def _cover_parents(section: Section, parent_of: dict[int, Section | None]) -> None:
    cover = _subtree_page_end(section)
    node = parent_of.get(id(section))
    while node is not None:
        node.page_end = max(node.page_end, cover)
        node = parent_of.get(id(node))
