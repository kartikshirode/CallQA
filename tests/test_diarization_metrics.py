"""Tests for the pure diarization metrics.

These never touch a GPU, the network, or the HF token. They build small
hand-made segment lists and check the DER math, the annotation conversion and
the speaker count. The pyannote pipeline is not loaded here at all.
"""
from pyannote.core import Annotation

from callqa.diarization.metrics import (
    diarization_error_rate,
    speaker_count,
    to_annotation,
)
from callqa.registry.schema import SpeakerSegment


def _segs():
    """Two speakers, four turns, no overlap."""
    return [
        SpeakerSegment(speaker="agent", start=0.0, end=2.0),
        SpeakerSegment(speaker="customer", start=2.0, end=4.0),
        SpeakerSegment(speaker="agent", start=4.0, end=6.0),
        SpeakerSegment(speaker="customer", start=6.0, end=8.0),
    ]


def test_to_annotation_returns_annotation():
    ann = to_annotation(_segs())
    assert isinstance(ann, Annotation)


def test_to_annotation_keeps_labels_and_times():
    ann = to_annotation(_segs())
    assert set(ann.labels()) == {"agent", "customer"}
    # Total speech for agent is two 2-second turns.
    assert abs(ann.label_duration("agent") - 4.0) < 1e-6


def test_to_annotation_accepts_dicts():
    raw = [
        {"speaker": "a", "start": 0.0, "end": 1.0},
        {"speaker": "b", "start": 1.0, "end": 2.0},
    ]
    ann = to_annotation(raw)
    assert set(ann.labels()) == {"a", "b"}


def test_speaker_count_distinct_labels():
    assert speaker_count(_segs()) == 2


def test_speaker_count_single_speaker():
    segs = [
        SpeakerSegment(speaker="agent", start=0.0, end=1.0),
        SpeakerSegment(speaker="agent", start=1.0, end=2.0),
    ]
    assert speaker_count(segs) == 1


def test_speaker_count_empty():
    assert speaker_count([]) == 0


def test_perfect_match_is_zero():
    ref = _segs()
    hyp = _segs()
    assert diarization_error_rate(ref, hyp) == 0.0


def test_merging_two_speakers_into_one_is_high():
    ref = _segs()
    # Hypothesis collapses both roles onto a single label across the whole call.
    hyp = [SpeakerSegment(speaker="spk", start=0.0, end=8.0)]
    der = diarization_error_rate(ref, hyp)
    # Half the call is now confused, so DER should be clearly high.
    assert der >= 0.4


def test_label_names_do_not_matter():
    # DER maps labels optimally, so renaming both speakers keeps a perfect match.
    ref = _segs()
    hyp = [
        SpeakerSegment(speaker="spk0", start=0.0, end=2.0),
        SpeakerSegment(speaker="spk1", start=2.0, end=4.0),
        SpeakerSegment(speaker="spk0", start=4.0, end=6.0),
        SpeakerSegment(speaker="spk1", start=6.0, end=8.0),
    ]
    assert diarization_error_rate(ref, hyp) == 0.0
