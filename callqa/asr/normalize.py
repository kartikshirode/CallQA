"""Pure WER text normalizer.

No GPU, no token, no network. This module decides whether real WER is honest,
so it carries the heaviest test coverage in Week 2. It lowercases text, drops
HarperValley non-speech markup, strips punctuation, and collapses whitespace.
Applied identically to reference and hypothesis, so any wording choice it makes
is fair as long as both sides go through it.
"""
from __future__ import annotations

import re

# HarperValley (and Whisper) non-speech markup arrives wrapped in square or
# angle brackets: [noise], [laughter], <unk>, <sil>. Strip any run inside
# [...] or <...>, brackets included, BEFORE punctuation stripping so the marker
# text never leaks through as bare words. Non-greedy so adjacent markers on one
# line stay separate.
MARKER_RE = re.compile(r"\[[^\]]*\]|<[^>]*>")

# Apostrophes are dropped with no space, so a contraction stays one token:
# "don't" becomes "dont", "that's" becomes "thats". This matters because human
# transcripts and Whisper often disagree on apostrophe style ("thats" vs
# "that's"), and splitting the word in two would score that disagreement as a
# real error. Covers the straight quote plus the curly and backtick variants.
_APOS_RE = re.compile(r"['’ʼ`]")

# Everything else that is not a letter, digit, or space becomes a space, so
# words on either side of punctuation stay separate.
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, drop bracketed markers, strip punctuation, collapse spaces.

    Order matters: markers go first (so their contents cannot survive as words),
    then apostrophes join contractions, then punctuation, then whitespace. None
    or an empty string returns "".
    """
    if not text:
        return ""
    text = text.lower()
    text = MARKER_RE.sub(" ", text)
    text = _APOS_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()
