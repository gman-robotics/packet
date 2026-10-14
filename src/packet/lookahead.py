"""Look-ahead boundary commit (DESIGN.md "Boundary policy").

High-confidence boundaries split as soon as every earlier page is decided.
Mid-confidence ones wait for up to ``window`` following pages and are decided by
page-number evidence, compared against the last *accepted* numbered page of the
current document (inserted pages that were rejected do not count):

- the page itself continues the current numbering (same total, next number):
  not a boundary (``continues_sequence``);
- a following page continues the *new* numbering (``num(p) + k``, same total when
  both are printed): split (``lookahead_numbering``);
- a following page resumes the *old* numbering (next number(s) of the current
  sequence, same total, within it): not a boundary, the page was an insert
  (``old_sequence_continues``);
- otherwise (no evidence, or end of file): the eager rule, split iff
  confidence >= ``fallback``.

``window=0`` is exactly v0 eager commit: split iff confidence >= ``fallback``.
"""

from __future__ import annotations

from dataclasses import dataclass

from packet.boundary import BoundarySignal


@dataclass
class PageObs:
    index: int
    signal: BoundarySignal | None = None
    page_num: int | None = None
    page_total: int | None = None


@dataclass
class Decision:
    page: int
    split: bool  # a new document starts at ``page``
    reason: str


@dataclass
class _Seq:
    index: int
    num: int | None
    total: int | None


def _same_total(a: int | None, b: int | None) -> bool:
    return a is None or b is None or a == b


class LookaheadSplitter:
    def __init__(self, window: int = 2, commit: float = 0.75, propose: float = 0.4,
                 fallback: float = 0.55):
        self.window = max(0, window)
        self.fallback = fallback
        self.commit = max(commit, fallback)  # never split more eagerly than v0
        self.propose = propose
        self._obs: list[PageObs] = []  # undecided pages + lookahead only
        self._seq: _Seq | None = None  # numbering anchor of the current document

    def push(self, obs: PageObs) -> list[Decision]:
        self._obs.append(obs)
        return self._drain(final=False)

    def flush(self) -> list[Decision]:
        return self._drain(final=True)

    # -- internals --
    @staticmethod
    def _conf(o: PageObs) -> float:
        return o.signal.confidence if o.signal is not None else 0.0

    def _drain(self, final: bool) -> list[Decision]:
        out: list[Decision] = []
        while self._obs:
            o = self._obs[0]
            conf = self._conf(o)
            if self.window == 0:
                d = (Decision(o.index, True, "eager") if conf >= self.fallback
                     else Decision(o.index, False, "below_threshold"))
            elif conf < self.propose:
                d = Decision(o.index, False, "no_signal")
            elif conf >= self.commit:
                d = Decision(o.index, True, "high_confidence")
            elif final or len(self._obs) - 1 >= self.window:
                d = self._decide_mid(o)
            else:
                break  # wait for more pages
            self._record(o, d)
            out.append(d)
            self._obs.pop(0)
        return out

    def _record(self, o: PageObs, d: Decision) -> None:
        if d.split:
            # An unnumbered split still opens a document. Keep the anchor so the
            # next printed run is that document's sequence, not a new boundary.
            self._seq = _Seq(o.index, o.page_num, o.page_total)
        elif d.reason == "old_sequence_continues":
            pass  # an insert: keep comparing against the real sequence
        elif o.page_num is not None:
            self._seq = _Seq(o.index, o.page_num, o.page_total)

    def _continues(self, num: int | None, total: int | None, index: int) -> bool:
        """``num`` continues the current sequence at page ``index`` (inserts in
        between may or may not have been counted)."""
        s = self._seq
        if s is None or num is None or not _same_total(s.total, total):
            return False
        if s.total is not None and num > s.total:
            return False
        gap = index - s.index
        if s.num is None:
            # Opened on an unnumbered page. The first printed number belongs to
            # this document when it fits the pages since the split (the opening
            # page may or may not have been counted).
            return gap >= 1 and 1 <= num <= gap + 1
        return s.num < num <= s.num + gap

    def _decide_mid(self, o: PageObs) -> Decision:
        if self._continues(o.page_num, o.page_total, o.index):
            return Decision(o.index, False, "continues_sequence")
        for k in range(1, self.window + 1):
            if k >= len(self._obs):
                break
            q = self._obs[k]
            if q.page_num is None:
                continue
            if (o.page_num is not None and q.page_num == o.page_num + k
                    and _same_total(o.page_total, q.page_total)):
                return Decision(o.index, True, "lookahead_numbering")
            if self._continues(q.page_num, q.page_total, q.index):
                return Decision(o.index, False, "old_sequence_continues")
        split = self._conf(o) >= self.fallback
        return Decision(o.index, split, "fallback_eager" if split else "fallback_below")
