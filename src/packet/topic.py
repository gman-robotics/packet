"""Topic and language-shift evidence between pages (GMA-218).

Lightweight and dependency-free: a page's *content words* (letters only, four or more
characters, no stopwords, furniture excluded; CJK text as character bigrams) and a
conservative language guess from stopword hits and script. ``BoundaryDetector`` uses
them as weak signals: a near-zero vocabulary overlap with the previous pages
(``topic_shift``) or a change of language (``language_changed``) never splits alone,
but adds to a title, header or page-number signal.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from packet.types import Block, BlockKind

_FURNITURE = {BlockKind.HEADER, BlockKind.FOOTER, BlockKind.PAGE_NUMBER}
_WORD = re.compile(r"[^\W\d_]{4,}")
_TOKEN = re.compile(r"[^\W\d_]+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+")

_STOP = {
    "en": "the of and to in is that for it as with was on be by are this from or an at not "
          "which have has were their they will would there these been shall may",
    "de": "der die und den das ist nicht von mit sich des auf für ein eine dem zu im auch "
          "sind wird werden oder bei nach wie aus",
    "fr": "le la les des et est une un du pour dans que qui sur pas par au avec sont ce "
          "il elle ou aux cette",
    "es": "el la los las de y en que es por un una para con del se al lo como su más o "
          "son está",
    "it": "il la di che e un una per con del della non sono le gli dei nel alla è si da "
          "come questo",
    "nl": "de het een en van is dat op te in voor niet met zijn er aan ook als bij door "
          "wordt worden",
    "pt": "o a os as de e do da em um uma para com não que se no na por mais dos das é",
}
_STOPSETS = {lang: set(words.split()) for lang, words in _STOP.items()}
# Content-word filter: long stopwords of every language plus words common to any document.
_IGNORE = {w for s in _STOPSETS.values() for w in s if len(w) >= 4} | set(
    ["that", "this", "with", "from", "have", "were", "been", "their", "they", "will", "would", "there", "these", "which", "other", "into", "more", "such", "than", "when", "also", "some", "only", "about", "over", "under", "after", "before", "between", "through", "each", "most", "being", "both", "same", "where", "while", "because", "could", "should", "does", "upon", "within", "without", "total", "year", "years", "page", "pages", "table", "figure", "note", "notes", "section", "shall", "article", "chapter", "part"]
)

MIN_WORDS = 40  # fewer content words than this: too little text to compare topics
MIN_LANG_WORDS = 20


def content_words(blocks: list[Block]) -> Counter:
    """Content-word counts of a page's non-furniture text."""
    text = " ".join(b.text for b in blocks if b.kind not in _FURNITURE and b.text)
    words = Counter(w for w in (m.lower() for m in _WORD.findall(text)) if w not in _IGNORE)
    for run in _CJK.findall(text):
        words.update(run[i:i + 2] for i in range(len(run) - 1))
    return words


def topic_similarity(a: Counter, b: Counter) -> float:
    """Cosine similarity of two content-word counts (0 when either is empty)."""
    if not a or not b:
        return 0.0
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def detect_language(text: str) -> str | None:
    """A conservative language guess, or None when the text is short or ambiguous."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 30:
        return None
    count = Counter(_script(c) for c in letters)
    script, n = count.most_common(1)[0]
    if n < 0.6 * len(letters):
        return None
    if script == "kana" or (script == "han" and count["kana"] >= 0.05 * len(letters)):
        return "ja"
    if script != "latin":
        return script  # han, hangul, cyrillic, greek, arabic, hebrew...
    words = [w.lower() for w in _TOKEN.findall(text)]
    if len(words) < MIN_LANG_WORDS:
        return None
    hits = sorted(((sum(w in s for w in words), lang) for lang, s in _STOPSETS.items()), reverse=True)
    (best, lang), (second, _) = hits[0], hits[1]
    if best < 0.15 * len(words) or best < 2 * second:
        return None
    return lang


def _script(c: str) -> str:
    o = ord(c)
    if o < 0x250:
        return "latin"
    if 0x3040 <= o <= 0x30FF:
        return "kana"
    if 0x3400 <= o <= 0x9FFF:
        return "han"
    if 0xAC00 <= o <= 0xD7AF:
        return "hangul"
    if 0x400 <= o <= 0x4FF:
        return "cyrillic"
    if 0x370 <= o <= 0x3FF:
        return "greek"
    if 0x600 <= o <= 0x6FF:
        return "arabic"
    if 0x590 <= o <= 0x5FF:
        return "hebrew"
    return "other"
