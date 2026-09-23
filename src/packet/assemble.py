from __future__ import annotations

from packet.types import Block, BlockKind, LogicalDocument, Packet, Section


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
        return LogicalDocument(
            id=doc_id,
            title=title,
            page_start=min(pages),
            page_end=max(pages),
            sections=root_sections,
            furniture=furniture,
        )

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
