"""Pure diarization metrics.

No GPU, no token, no network. These take plain speaker segments (either
SpeakerSegment models or dicts with speaker/start/end) and compute the
diarization error rate, the speaker count, and the pyannote Annotation used
under the hood. Keeping this module pure is what lets the metric tests run
without the pyannote pipeline or the HF token.
"""
from __future__ import annotations

from typing import Iterable, Union

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from callqa.registry.schema import SpeakerSegment

# A segment can arrive as the pydantic model or a plain dict.
SegmentLike = Union[SpeakerSegment, dict]


def _fields(seg: SegmentLike) -> tuple[str, float, float]:
    """Pull speaker, start and end out of a model or a dict."""
    if isinstance(seg, SpeakerSegment):
        return seg.speaker, seg.start, seg.end
    return seg["speaker"], float(seg["start"]), float(seg["end"])


def to_annotation(segments: Iterable[SegmentLike]) -> Annotation:
    """Build a pyannote Annotation from speaker segments."""
    ann = Annotation()
    for seg in segments:
        speaker, start, end = _fields(seg)
        ann[Segment(start, end)] = speaker
    return ann


def speaker_count(segments: Iterable[SegmentLike]) -> int:
    """Number of distinct speaker labels in the segments."""
    return len({_fields(seg)[0] for seg in segments})


def diarization_error_rate(
    reference: Iterable[SegmentLike],
    hypothesis: Iterable[SegmentLike],
) -> float:
    """DER of a hypothesis against a reference, both as segment lists."""
    metric = DiarizationErrorRate()
    ref = to_annotation(reference)
    hyp = to_annotation(hypothesis)
    return float(metric(ref, hyp))
