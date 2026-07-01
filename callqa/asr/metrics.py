"""Pure WER and CER metrics on top of jiwer.

No GPU, no token, no network. Scores a single call or pools a whole corpus.
The normalizer runs on both sides by default so HarperValley markup and casing
do not inflate error. Everything here is deterministic and testable offline.
"""
from __future__ import annotations

import jiwer
from pydantic import BaseModel, Field

from callqa.asr.normalize import normalize_text


class AsrScore(BaseModel):
    """Corpus-level ASR score. wer and cer are pooled, not averaged per call."""

    wer: float = Field(ge=0.0)
    cer: float = Field(ge=0.0)
    n_calls: int = Field(ge=0)


def _guard(ref: str, hyp: str) -> float | None:
    """Empty-reference guard.

    jiwer divides by reference length, so an empty reference would blow up.
    Return 0.0 when both sides are empty (nothing said, nothing wrong), else
    1.0 (the model invented words against silence). None means "no guard, score
    normally".
    """
    if ref == "":
        return 0.0 if hyp == "" else 1.0
    return None


def word_error_rate(reference: str, hypothesis: str, *, normalize: bool = True) -> float:
    """WER of one hypothesis against one reference.

    With normalize=True both sides pass through normalize_text first, so a
    difference that is only casing or punctuation scores 0.0.
    """
    ref = normalize_text(reference) if normalize else reference
    hyp = normalize_text(hypothesis) if normalize else hypothesis
    guarded = _guard(ref, hyp)
    if guarded is not None:
        return guarded
    return float(jiwer.wer(ref, hyp))


def character_error_rate(reference: str, hypothesis: str, *, normalize: bool = True) -> float:
    """CER of one hypothesis against one reference. Same guard and normalize
    behavior as word_error_rate."""
    ref = normalize_text(reference) if normalize else reference
    hyp = normalize_text(hypothesis) if normalize else hypothesis
    guarded = _guard(ref, hyp)
    if guarded is not None:
        return guarded
    return float(jiwer.cer(ref, hyp))


def aggregate_score(
    pairs: list[tuple[str, str]], *, normalize: bool = True
) -> AsrScore:
    """Pool many (reference, hypothesis) pairs into one corpus WER and CER.

    jiwer takes lists and pools edits across all calls, so a long call weighs
    more than a short one. That is the honest corpus number, not the mean of
    per-call rates. Empty-reference rows are dropped from both lists so they do
    not skew the pooled denominator; an all-empty corpus scores 0.0.
    """
    refs: list[str] = []
    hyps: list[str] = []
    n = 0
    for reference, hypothesis in pairs:
        n += 1
        ref = normalize_text(reference) if normalize else reference
        hyp = normalize_text(hypothesis) if normalize else hypothesis
        if ref == "":
            continue
        refs.append(ref)
        hyps.append(hyp)

    if not refs:
        return AsrScore(wer=0.0, cer=0.0, n_calls=n)

    wer = float(jiwer.wer(refs, hyps))
    cer = float(jiwer.cer(refs, hyps))
    return AsrScore(wer=wer, cer=cer, n_calls=n)
