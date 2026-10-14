"""Look-ahead boundary commit (DESIGN.md "Boundary policy").

High-confidence boundaries commit at once. Mid-confidence ones wait for up to
``window`` following pages and are decided by page-number evidence:

- a following page continues the *new* numbering (``num(p) + k``): confirm;
- a following page continues the *old* numbering (``num(p-1) + k`` or ``+ k + 1``): reject,
  the page was an insert (exhibit, title sheet) inside the current document;
- a following page carries a stronger boundary signal: reject in its favour;
- no evidence in the window (or end of file): fall back to the eager threshold.

``window=0`` reproduces v0 eager commit exactly.
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


class LookaheadSplitter:
    def __init__(self, window: int = 2, commit: float = 0.75, propose: float = 0.4,
                 fallback: float = 0.55):
        self.window = max(0, window)
        self.commit = commit
        self.propose = propose
        self.fallback = fallback
        self._obs: list[PageObs] = []
        self._next = 0

    def push(self, obs: PageObs) -> list[Decision]:
        self._obs.append(obs)
        return self._drain(final=False)

    def flush(self) -> list[Decision]:
        return self._drain(final=True)

    # -- internals --
    def _conf(self, o: PageObs) -> float:
        return o.signal.confidence if o.signal is not None else 0.0

    def _drain(self, final: bool) -> list[Decision]:
        out: list[Decision] = []
        while self._next < len(self._obs):
            i = self._next
            o = self._obs[i]
            conf = self._conf(o)
            if conf < self.propose:
                d = Decision(o.index, False, "no_signal")
            elif conf >= self.commit:
                d = Decision(o.index, True, "high_confidence")
            elif self.window == 0 or final or len(self._obs) - 1 >= i + self.window:
                d = self._decide_mid(i)
            else:
                break  # wait for more pages
            out.append(d)
            self._next += 1
        return out

    def _decide_mid(self, i: int) -> Decision:
        o = self._obs[i]
        conf = self._conf(o)
        prev = self._obs[i - 1].page_num if i > 0 else None
        for k in range(1, self.window + 1):
            if i + k >= len(self._obs):
                break
            q = self._obs[i + k]
            if self._conf(q) >= self.propose and self._conf(q) > conf:
                return Decision(o.index, False, "superseded")
            if q.page_num is not None:
                if o.page_num is not None and q.page_num == o.page_num + k:
                    return Decision(o.index, True, "lookahead_numbering")
                # old numbering resumes, counting the inserted page or not
                if prev is not None and q.page_num in (prev + k, prev + k + 1):
                    return Decision(o.index, False, "old_sequence_continues")
        split = conf >= self.fallback
        return Decision(o.index, split, "fallback_eager" if split else "fallback_below")
