from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from packet.boundary_layout import size_jump, title_position
from packet.topic import MIN_WORDS, content_words, detect_language, same_language, topic_similarity
from packet.types import Block, BlockKind, PageRef

if TYPE_CHECKING:
    from packet.boundary_model import BoundaryScorer
    from packet.decision import DecisionBackend

# Body and furniture lines count only when the entire line is a page label (not e.g. 1/2 in
# spectroscopic notation embedded in prose).
_PAGE_PREFIX = r"page\s*"
_PAGE_FRAC_TAIL = r"(\d+)\s*(?:of|/)\s*(\d+)\s*$"
_WHOLE_PAGE_NUM = re.compile(
    rf"^(?:{_PAGE_PREFIX})?{_PAGE_FRAC_TAIL}",
    re.IGNORECASE,
)
_WHOLE_PAGE_PREFIXED = re.compile(
    rf"^{_PAGE_PREFIX}{_PAGE_FRAC_TAIL}",
    re.IGNORECASE,
)
_WHOLE_P_PREFIX_PAGE = re.compile(
    rf"^p\.?\s*{_PAGE_FRAC_TAIL}",
    re.IGNORECASE,
)


_BARE_PN = re.compile(r"^\s*(?:page\s*)?(\d{1,4})\s*$", re.IGNORECASE)


def _page_fraction_whole_line(stripped: str) -> tuple[int, int] | None:
    match = _WHOLE_PAGE_NUM.match(stripped) or _WHOLE_P_PREFIX_PAGE.match(stripped)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _native_prefixed_page_label_fraction(stripped: str) -> tuple[int, int] | None:
    """Native page or ``p.`` prefix arms only (not bare ``N of M`` / ``N/M``)."""
    match = _WHOLE_P_PREFIX_PAGE.match(stripped) or _WHOLE_PAGE_PREFIXED.match(stripped)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def fraction_on_whole_line(text: str) -> tuple[int, int] | None:
    """First non-empty line's native whole-line page fraction, if any."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return _page_fraction_whole_line(stripped)
    return None


def keyword_page_label_fraction(text: str) -> tuple[int, int] | None:
    """Whole-line page fraction with a required ``page`` or ``p.`` prefix (OCR split gate)."""
    stripped = text.strip()
    if not stripped:
        return None
    return _native_prefixed_page_label_fraction(stripped)


def match_trailing_page_label(line: str) -> tuple[str, str] | None:
    """Split ``line`` when a trailing suffix passes ``keyword_page_label_fraction``."""
    s = line.rstrip()
    for start in range(len(s) - 1, 0, -1):
        if not s[start - 1].isspace():
            continue
        suffix = s[start:].strip()
        body = s[:start].rstrip()
        if body and keyword_page_label_fraction(suffix) is not None:
            return body, suffix
    return None


# Signal weights. A signal at or above PROPOSE is proposed alone; two or more
# independent weak signals together are lifted to COMBINED, which clears the
# pipeline's eager floor (0.55) but stays below the immediate-commit level, so
# look-ahead page-number evidence can still veto them.
W_PAGE_RESET = 0.45
W_SEQUENCE_COMPLETED = 0.2
W_HEADER_CHANGED = 0.25
W_MEDIA_BOX = 0.2
W_TITLE_TOP = 0.45  # a title that opens the page (below any running header): proposes alone
W_TITLE = 0.25  # a title lower on the page
# Weak content evidence (GMA-218): never proposes alone, but counts as an independent
# signal next to a title, header change or page-number reset.
W_TOPIC_SHIFT = 0.2  # near-zero vocabulary overlap with the previous two pages
W_LANGUAGE = 0.25  # the page's language differs from the previous page's
TOPIC_SHIFT_MAX = 0.01  # cosine of content words at or below this is a topic shift
PROPOSE = 0.4
COMBINED = 0.6
# Without a page-number reset, no combination may reach the splitter's immediate
# commit level (0.75): look-ahead numbering must always get a chance to veto.
CAP_WITHOUT_RESET = 0.74
# Optional embedding evidence (GMA-443, off unless an embedder is passed): low cosine between
# this page's embedding and the mean of the previous two. Weak like topic_shift: never proposes
# alone, and content evidence (topic, language, embedding) casts one independent vote at most.
W_EMBED_SHIFT = 0.15
EMBED_SHIFT_MAX = 0.5


@dataclass
class BoundarySignal:
    after_page: int
    confidence: float
    reason: str


@dataclass
class BoundaryDetector:
    """Generic, domain-free boundary hints."""

    lookback: int = 3
    # embedder(ref, blocks) -> unit-norm vector or None (GMA-443). None keeps v0 behaviour.
    embedder: Callable[[PageRef, list[Block]], Any] | None = None
    embed_shift_max: float = EMBED_SHIFT_MAX
    w_embed_shift: float = W_EMBED_SHIFT
    last_page_number: int | None = None
    last_page_total: int | None = None
    _page_nums: list[tuple[int, int | None]] = field(default_factory=list)
    _page_totals: list[int | None] = field(default_factory=list)
    _tagged_header: list[bool] = field(default_factory=list)
    _headers: list[str] = field(default_factory=list)
    _header_styles: list[tuple | None] = field(default_factory=list)
    _sizes: list[tuple[float | None, float | None, int]] = field(default_factory=list)
    _words: list = field(default_factory=list)
    _langs: list[str | None] = field(default_factory=list)
    _page_facts: list[dict] = field(default_factory=list)
    _embeds: list = field(default_factory=list)
    last_embed_cosine: float | None = None

    def observe(
        self,
        ref: PageRef,
        blocks: list[Block],
        scorer: BoundaryScorer | None = None,
        decision_backend: DecisionBackend | None = None,
        render: Callable[[], Any] | None = None,
    ) -> BoundarySignal | None:
        page_num, page_total = _extract_page_number(blocks)
        self.last_page_number = page_num
        self.last_page_total = page_total
        self._page_totals.append(page_total)
        # header_changed only means something when the previous page carried a tagged
        # running HEADER: it fires when that header stops. Ordinary pages always have
        # different first lines, so comparing those would fire on every page pair.
        header = _leading_fingerprint(blocks)
        self._tagged_header.append(bool(_fingerprint(blocks, BlockKind.HEADER)))
        self._page_nums.append((ref.index, page_num))
        self._headers.append(header)
        self._header_styles.append(_header_style(blocks))
        self._sizes.append((ref.width, ref.height, ref.rotation))
        words = content_words(blocks)
        self._words.append(words)
        self._embeds.append(self.embedder(ref, blocks) if self.embedder is not None else None)
        self._langs.append(detect_language(" ".join(
            b.text for b in blocks if b.kind not in _FURNITURE and b.text)))
        from packet.boundary_model import page_fact_row

        self._page_facts.append(
            page_fact_row(ref, blocks, page_number=page_num, page_total=page_total)
        )

        if ref.index == 0:
            return None

        reasons: list[str] = []
        score = 0.0

        prev_num = self._page_nums[-2][1]
        if page_num == 1 and prev_num is not None and prev_num > 1:
            reasons.append("page_number_reset")
            score += W_PAGE_RESET
            prev_total = self._page_totals[-2]
            if prev_total is not None and prev_num == prev_total:
                # previous page was "N of N": its sequence is complete
                reasons.append("sequence_completed")
                score += W_SEQUENCE_COMPLETED

        prev_header = self._headers[-2]
        prev_style, style = self._header_styles[-2], self._header_styles[-1]
        same_template = prev_style is not None and prev_style == style
        # A layout page with no running-head line at all after one that had it (a
        # chapter opener drops its head) is an absence, not a new template.
        # Only a small-type running head counts: a body-size repeated first line that
        # stops (an inserted exhibit) is still a header change.
        # Only a chapter-opener shape (title at the top, same page geometry) counts: any
        # other page that loses the head is still a header change.
        dropped = (prev_style is not None and prev_style[2] and style is None and _is_layout(blocks)
                   and title_position(blocks) == "top"
                   and not size_jump(self._sizes[-2], self._sizes[-1]))
        # With layout styles on both pages, a running head whose text changes but whose
        # template (font size, position) does not is a chapter head, not a new document.
        if (self._tagged_header[-2] and header and prev_header and header != prev_header
                and not same_template and not dropped):
            reasons.append("header_changed")
            score += W_HEADER_CHANGED

        prev_size = self._sizes[-2]
        curr_size = self._sizes[-1]
        if size_jump(prev_size, curr_size):
            reasons.append("media_box_changed")
            score += W_MEDIA_BOX

        title = title_position(blocks)
        if title == "top":
            reasons.append("title_top")
            score += W_TITLE_TOP
        elif title == "lower":
            reasons.append("title_block")
            score += W_TITLE

        if self._topic_shift():
            reasons.append("topic_shift")
            score += W_TOPIC_SHIFT
        prev_lang, lang = self._langs[-2], self._langs[-1]
        if not same_language(prev_lang, lang):
            reasons.append("language_changed")
            score += W_LANGUAGE
        self.last_embed_cosine = self._embed_cosine()
        if self.last_embed_cosine is not None and self.last_embed_cosine <= self.embed_shift_max:
            reasons.append("embed_shift")
            score += self.w_embed_shift

        # "sequence_completed" refines the reset, it is not independent evidence, and a
        # title lower on the page is too common inside documents (OCR labels table and
        # section captions as titles) to count toward a combination.
        independent = [r for r in reasons if r not in {"sequence_completed", "title_block"}]
        # A language change also zeroes the vocabulary overlap: one event, one vote
        # and one weight.
        if "topic_shift" in independent and "language_changed" in independent:
            independent.remove("topic_shift")
            score -= W_TOPIC_SHIFT
        # The embedding sees the same content change: it adds weight but no extra vote.
        if "embed_shift" in independent and ({"topic_shift", "language_changed"} & set(independent)):
            independent.remove("embed_shift")
        if len(independent) >= 2:
            score = max(score, COMBINED)
        heur_conf = score if score >= PROPOSE else None
        conf = heur_conf
        model_used = False
        if decision_backend is not None:
            from packet.boundary_model import combined_confidence, model_raised_confidence
            from packet.decision import DecisionRequest, decision_backend_passes_images

            decide_render = render if decision_backend_passes_images(decision_backend) else None
            result = decision_backend.decide(
                DecisionRequest(
                    task="boundary",
                    page_index=ref.index,
                    facts=list(self._page_facts),
                    blocks=blocks,
                    ref=ref,
                ),
                render=decide_render,
            )
            if result.probability is not None:
                model_p = result.probability
                model_used = model_raised_confidence(heur_conf, model_p)
                conf = combined_confidence(heur_conf, model_p)
        elif scorer is not None:
            from packet.boundary_model import (
                combined_confidence,
                featurize_from_facts,
                model_raised_confidence,
            )

            features = featurize_from_facts(self._page_facts, ref.index)
            model_p = scorer.probability(features)
            model_used = model_raised_confidence(heur_conf, model_p)
            conf = combined_confidence(heur_conf, model_p)
        if conf is None:
            return None
        if "page_number_reset" not in reasons:
            conf = min(conf, CAP_WITHOUT_RESET)
        reason = "+".join(reasons)
        if model_used:
            reason = f"{reason}+model" if reason else "model"
        return BoundarySignal(
            after_page=ref.index - 1,
            confidence=min(conf, 0.95),
            reason=reason or "boundary",
        )


    def _embed_cosine(self) -> float | None:
        """Cosine of this page's embedding with the mean of the previous two (None if missing)."""
        cur = self._embeds[-1]
        prev = [e for e in self._embeds[-3:-1] if e is not None]
        if cur is None or not prev:
            return None
        mean = [sum(v) / len(prev) for v in zip(*prev)]
        norm = sum(v * v for v in mean) ** 0.5 * sum(v * v for v in cur) ** 0.5
        return sum(a * b for a, b in zip(cur, mean)) / norm if norm else None

    def _topic_shift(self) -> bool:
        cur, prev = self._words[-1], self._words[-2]
        if sum(cur.values()) < MIN_WORDS or sum(prev.values()) < MIN_WORDS:
            return False
        # Only prose compares: tables, figures and forms carry no recognisable language
        # and their vocabulary says little about the topic.
        if not (self._langs[-1] and self._langs[-2]):
            return False
        before = prev + (self._words[-3] if len(self._words) >= 3 else type(prev)())
        return topic_similarity(before, cur) <= TOPIC_SHIFT_MAX


def _is_layout(blocks: list[Block]) -> bool:
    return any(b.extra.get("layout") for b in blocks)


def _header_style(blocks: list[Block]) -> tuple | None:
    """(font size, rounded top) of the page's running-head line: a layout HEADER, or a
    small top-band line not tagged yet (its repeat may lie on a page not seen yet)."""
    for b in blocks:
        x = b.extra or {}
        if x.get("layout") and (b.kind == BlockKind.HEADER
                                or (x.get("band") == "top" and x.get("small"))):
            return (x.get("size"), round(float(x.get("top") or 0) / 4), bool(x.get("small")))
    return None


def _fraction_on_whole_line(text: str) -> tuple[int, int] | None:
    return fraction_on_whole_line(text)


def _extract_page_number(blocks: list[Block]) -> tuple[int | None, int | None]:
    # A tagged PAGE_NUMBER wins: header/footer text often holds "N of M" or "N/M"
    # (a volume label, a date) that must not shadow the printed page number.
    for block in blocks:
        if block.kind == BlockKind.PAGE_NUMBER:
            frac = _fraction_on_whole_line(block.text)
            if frac:
                return frac[0], frac[1]
            bare = _BARE_PN.match(block.text)
            if bare:
                return int(bare.group(1)), None
    for block in blocks:
        is_furniture = block.kind in {BlockKind.PAGE_NUMBER, BlockKind.FOOTER, BlockKind.HEADER}
        if not is_furniture and (block.kind is not BlockKind.PARAGRAPH or len(block.text) > 40):
            continue
        frac = _fraction_on_whole_line(block.text)
        if frac:
            return frac[0], frac[1]
    for block in blocks[-2:]:
        frac = _fraction_on_whole_line(block.text)
        if frac:
            return frac[0], frac[1]
    return None, None


_FURNITURE = {BlockKind.HEADER, BlockKind.FOOTER, BlockKind.PAGE_NUMBER}


def _fingerprint(blocks: list[Block], kind: BlockKind) -> str:
    texts = [b.text.strip().lower() for b in blocks if b.kind == kind and b.text.strip()]
    return " | ".join(texts[:2])


def _leading_fingerprint(blocks: list[Block]) -> str:
    if not blocks:
        return ""
    first = blocks[0].text.strip().lower().split("\n", 1)[0]
    return first[:80]


