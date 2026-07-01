"""Week 1 dataset - the real anchor (HarperValley) and diarization metrics.

The real tier owns WER and DER. These checks stay offline: in-memory dicts for
the HarperValley loader, hand-made segment lists for the DER math. No network,
no GPU, no HF token.
"""
import numpy as np
from pyannote.core import Annotation

from callqa.datasets.harpervalley import (
    gold_from_transcript,
    mix_channels,
    sample_call_ids,
)
from callqa.diarization.metrics import (
    diarization_error_rate,
    speaker_count,
    to_annotation,
)
from callqa.registry.schema import SpeakerSegment


def _segments():
    # Deliberately out of time order so the parser has to sort.
    return [
        {"speaker_role": "caller", "start_ms": 2000, "duration_ms": 1000,
         "human_transcript": "I have a question"},
        {"speaker_role": "agent", "start_ms": 0, "duration_ms": 1500,
         "human_transcript": "Hello how can I help"},
        {"speaker_role": "agent", "start_ms": 3500, "duration_ms": 500,
         "human_transcript": "Sure thing"},
    ]


class TestHarperValley:
    def test_mix_length_is_longer_input(self):
        out = mix_channels(np.zeros(5, dtype=np.float32), np.zeros(9, dtype=np.float32))
        assert len(out) == 9
        out2 = mix_channels(np.array([0.2, 0.2, 0.2], dtype=np.float32),
                            np.array([0.1, 0.1], dtype=np.float32))
        assert len(out2) == 3

    def test_mix_is_bounded_mono_float32(self):
        # Two loud channels would sum past 1.0; the peak limiter must reel it in.
        out = mix_channels(np.full(100, 0.9, dtype=np.float32),
                           np.full(100, 0.9, dtype=np.float32))
        assert out.ndim == 1
        assert out.dtype == np.float32
        assert float(np.max(np.abs(out))) <= 1.0

    def test_gold_transcript_joined_in_time_order(self):
        transcript, _ = gold_from_transcript(_segments())
        assert transcript == "Hello how can I help I have a question Sure thing"

    def test_gold_segments_have_seconds_and_roles(self):
        _, segments = gold_from_transcript(_segments())
        assert segments[0].speaker == "agent"
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
        assert segments[1].speaker == "caller"
        assert segments[1].start == 2.0
        assert segments[1].end == 3.0

    def test_gold_skips_malformed_timing_segments(self):
        # The parser promises to skip malformed segments rather than crash a
        # whole fetch. A non-numeric timing and a negative duration are both
        # malformed and must be dropped, not raise, same as a missing key.
        segs = [
            {"speaker_role": "agent", "start_ms": 0, "duration_ms": 1000,
             "human_transcript": "good one"},
            {"speaker_role": "agent", "start_ms": "1000", "duration_ms": 500,
             "human_transcript": "string timing"},
            {"speaker_role": "caller", "start_ms": 2000, "duration_ms": -500,
             "human_transcript": "negative duration"},
        ]
        transcript, segments = gold_from_transcript(segs)
        assert transcript == "good one"
        assert len(segments) == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 1.0

    def test_sample_is_seed_repeatable_with_count(self):
        ids = [f"sid{i}" for i in range(50)]
        a = sample_call_ids(ids, 10, seed=123)
        assert a == sample_call_ids(ids, 10, seed=123)
        assert len(a) == 10

    def test_sample_different_seed_differs(self):
        ids = [f"sid{i}" for i in range(50)]
        assert sample_call_ids(ids, 10, seed=1) != sample_call_ids(ids, 10, seed=2)


def _two_speakers():
    """Two speakers, four turns, no overlap."""
    return [
        SpeakerSegment(speaker="agent", start=0.0, end=2.0),
        SpeakerSegment(speaker="customer", start=2.0, end=4.0),
        SpeakerSegment(speaker="agent", start=4.0, end=6.0),
        SpeakerSegment(speaker="customer", start=6.0, end=8.0),
    ]


class TestDiarizationMetrics:
    def test_to_annotation_keeps_labels_and_times(self):
        ann = to_annotation(_two_speakers())
        assert isinstance(ann, Annotation)
        assert set(ann.labels()) == {"agent", "customer"}
        # Total speech for agent is two 2-second turns.
        assert abs(ann.label_duration("agent") - 4.0) < 1e-6

    def test_to_annotation_accepts_dicts(self):
        ann = to_annotation([
            {"speaker": "a", "start": 0.0, "end": 1.0},
            {"speaker": "b", "start": 1.0, "end": 2.0},
        ])
        assert set(ann.labels()) == {"a", "b"}

    def test_speaker_count(self):
        assert speaker_count(_two_speakers()) == 2
        assert speaker_count([
            SpeakerSegment(speaker="agent", start=0.0, end=1.0),
            SpeakerSegment(speaker="agent", start=1.0, end=2.0),
        ]) == 1
        assert speaker_count([]) == 0

    def test_perfect_match_is_zero(self):
        assert diarization_error_rate(_two_speakers(), _two_speakers()) == 0.0

    def test_merging_two_speakers_into_one_is_high(self):
        # Hypothesis collapses both roles onto one label across the whole call.
        hyp = [SpeakerSegment(speaker="spk", start=0.0, end=8.0)]
        assert diarization_error_rate(_two_speakers(), hyp) >= 0.4

    def test_label_names_do_not_matter(self):
        # DER maps labels optimally, so renaming both speakers stays perfect.
        hyp = [
            SpeakerSegment(speaker="spk0", start=0.0, end=2.0),
            SpeakerSegment(speaker="spk1", start=2.0, end=4.0),
            SpeakerSegment(speaker="spk0", start=4.0, end=6.0),
            SpeakerSegment(speaker="spk1", start=6.0, end=8.0),
        ]
        assert diarization_error_rate(_two_speakers(), hyp) == 0.0
