"""Tests for the dataset consistency checks.

Each check is fed one clean record and one deliberately broken record so we know
the checker catches the intended problem and stays quiet otherwise. Records are
built in memory, nothing here touches disk.
"""
from callqa.datasets.verify import (
    check_bounds,
    check_speaker_segments,
    check_synthetic_events,
    check_transcript,
    verify_record,
)
from callqa.registry.schema import (
    CallRecord,
    EventLabel,
    SpeakerSegment,
)


def _synthetic_record(**overrides):
    """A small clean synthetic call. Override any field via kwargs."""
    base = dict(
        call_id="syn-billing-99",
        audio_path="data/synthetic/audio/telephone/syn-billing-99.wav",
        duration_seconds=20.0,
        domain="billing",
        reference_transcript="Thanks for calling. How can I help you today?",
        speaker_segments=[
            SpeakerSegment(speaker="agent", start=0.0, end=5.0),
            SpeakerSegment(speaker="customer", start=5.0, end=10.0),
            SpeakerSegment(speaker="agent", start=10.0, end=15.0),
            SpeakerSegment(speaker="customer", start=15.0, end=20.0),
        ],
        event_labels=[
            EventLabel(
                event_type="compliance",
                start=0.0,
                end=5.0,
                metadata={"subtype": "recording_disclosure", "polarity": "positive"},
            ),
            EventLabel(
                event_type="compliance",
                start=5.0,
                end=10.0,
                metadata={"subtype": "identity_verification", "polarity": "positive"},
            ),
            EventLabel(event_type="silence", start=10.0, end=12.0, metadata=None),
            EventLabel(event_type="escalation", start=15.0, end=18.0, metadata=None),
        ],
        source="synthetic-kokoro-v1",
        privacy_notes="synthetic",
    )
    base.update(overrides)
    return CallRecord(**base)


# --- check_bounds ---------------------------------------------------------


def test_check_bounds_clean():
    assert check_bounds(_synthetic_record()) == []


def test_check_bounds_event_past_duration():
    rec = _synthetic_record(
        event_labels=[
            EventLabel(event_type="silence", start=18.0, end=25.0, metadata=None),
        ]
    )
    problems = check_bounds(rec)
    assert len(problems) == 1
    assert "25.0" in problems[0] or "out of bounds" in problems[0].lower()


def test_check_bounds_segment_negative_not_possible_so_use_event_before_zero():
    # start/end ge 0 is enforced by the schema, so we test the duration ceiling
    # on a speaker segment instead.
    rec = _synthetic_record(
        speaker_segments=[
            SpeakerSegment(speaker="agent", start=0.0, end=30.0),
        ]
    )
    problems = check_bounds(rec)
    assert any("30.0" in p or "bounds" in p.lower() for p in problems)


# --- check_transcript -----------------------------------------------------


def test_check_transcript_clean():
    assert check_transcript(_synthetic_record()) == []


def test_check_transcript_empty():
    rec = _synthetic_record(reference_transcript="   ")
    problems = check_transcript(rec)
    assert len(problems) == 1
    assert "transcript" in problems[0].lower()


def test_check_transcript_none():
    rec = _synthetic_record(reference_transcript=None)
    assert len(check_transcript(rec)) == 1


# --- check_speaker_segments ----------------------------------------------


def test_check_speaker_segments_clean():
    assert check_speaker_segments(_synthetic_record()) == []


def test_check_speaker_segments_overlap_allowed():
    # Interruption overlap: second segment starts before the first ends but
    # still after the first starts. Ordering by start is intact, so this is OK.
    rec = _synthetic_record(
        speaker_segments=[
            SpeakerSegment(speaker="agent", start=0.0, end=6.0),
            SpeakerSegment(speaker="customer", start=5.0, end=10.0),
        ]
    )
    assert check_speaker_segments(rec) == []


def test_check_speaker_segments_out_of_order():
    rec = _synthetic_record(
        speaker_segments=[
            SpeakerSegment(speaker="agent", start=10.0, end=15.0),
            SpeakerSegment(speaker="customer", start=5.0, end=9.0),
        ]
    )
    problems = check_speaker_segments(rec)
    assert len(problems) == 1
    assert "order" in problems[0].lower()


# --- check_synthetic_events ----------------------------------------------


def test_check_synthetic_events_clean():
    assert check_synthetic_events(_synthetic_record()) == []


def test_check_synthetic_events_negative_compliance_is_fine():
    rec = _synthetic_record(
        event_labels=[
            EventLabel(
                event_type="compliance",
                start=0.0,
                end=5.0,
                metadata={"subtype": "recording_disclosure", "polarity": "negative"},
            ),
        ]
    )
    assert check_synthetic_events(rec) == []


def test_check_synthetic_events_no_compliance_event():
    rec = _synthetic_record(
        event_labels=[
            EventLabel(event_type="silence", start=10.0, end=12.0, metadata=None),
        ]
    )
    problems = check_synthetic_events(rec)
    assert any("compliance" in p.lower() for p in problems)


def test_check_synthetic_events_zero_duration_silence():
    rec = _synthetic_record(
        event_labels=[
            EventLabel(
                event_type="compliance",
                start=0.0,
                end=5.0,
                metadata={"subtype": "recording_disclosure", "polarity": "positive"},
            ),
            EventLabel(event_type="silence", start=12.0, end=12.0, metadata=None),
        ]
    )
    problems = check_synthetic_events(rec)
    assert any("silence" in p.lower() for p in problems)


def test_check_synthetic_events_skips_non_synthetic():
    rec = _synthetic_record(
        source="harpervalley",
        privacy_notes="public, CC-BY-4.0",
        event_labels=None,
    )
    assert check_synthetic_events(rec) == []


# --- verify_record --------------------------------------------------------


def test_verify_record_clean_synthetic():
    assert verify_record(_synthetic_record()) == []


def test_verify_record_collects_multiple_problems():
    rec = _synthetic_record(
        reference_transcript="",
        event_labels=[
            EventLabel(event_type="silence", start=12.0, end=12.0, metadata=None),
        ],
    )
    problems = verify_record(rec)
    # empty transcript, missing compliance, zero-duration silence
    assert len(problems) >= 3


def test_verify_record_real_anchor_clean():
    rec = CallRecord(
        call_id="hv-test",
        audio_path="data/harpervalley/audio/test.wav",
        duration_seconds=30.0,
        domain="support",
        reference_transcript="hello this is the bank how can i help",
        speaker_segments=[
            SpeakerSegment(speaker="agent", start=0.0, end=5.0),
            SpeakerSegment(speaker="caller", start=5.0, end=10.0),
        ],
        event_labels=None,
        source="harpervalley",
        privacy_notes="public, CC-BY-4.0",
    )
    assert verify_record(rec) == []
