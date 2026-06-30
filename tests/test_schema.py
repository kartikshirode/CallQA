"""Tests for the registry pydantic models."""
import pytest
from pydantic import ValidationError

from callqa.registry.schema import (
    CallRecord,
    Domain,
    EventLabel,
    EventType,
    SpeakerSegment,
)


def test_speaker_segment_basic():
    seg = SpeakerSegment(speaker="agent", start=0.0, end=2.5)
    assert seg.speaker == "agent"
    assert seg.start == 0.0
    assert seg.end == 2.5


def test_speaker_segment_rejects_end_before_start():
    with pytest.raises(ValidationError):
        SpeakerSegment(speaker="agent", start=3.0, end=1.0)


def test_speaker_segment_allows_equal_start_end():
    seg = SpeakerSegment(speaker="customer", start=1.0, end=1.0)
    assert seg.start == seg.end


def test_event_label_basic():
    ev = EventLabel(event_type="silence", start=4.0, end=6.0)
    assert ev.event_type == EventType.silence
    assert ev.metadata is None


def test_event_label_compliance_with_metadata():
    ev = EventLabel(
        event_type="compliance",
        start=1.0,
        end=2.0,
        metadata={"subtype": "recording_disclosure", "polarity": "positive"},
    )
    assert ev.metadata["subtype"] == "recording_disclosure"
    assert ev.metadata["polarity"] == "positive"


def test_event_label_rejects_end_before_start():
    with pytest.raises(ValidationError):
        EventLabel(event_type="interruption", start=5.0, end=2.0)


def test_event_label_rejects_unknown_event_type():
    with pytest.raises(ValidationError):
        EventLabel(event_type="laughter", start=0.0, end=1.0)


def test_domain_covers_plan_and_spec_values():
    for value in [
        "billing",
        "refund",
        "cancellation",
        "tech_support",
        "support",
        "meeting-surrogate",
        "synthetic",
    ]:
        assert Domain(value)


def test_call_record_minimal_valid():
    rec = CallRecord(
        call_id="syn-001",
        audio_path="data/synthetic/syn-001.wav",
        duration_seconds=12.5,
        domain="billing",
        source="kokoro-tts",
        privacy_notes="synthetic",
    )
    assert rec.call_id == "syn-001"
    assert rec.domain == Domain.billing
    assert rec.reference_transcript is None
    assert rec.speaker_segments is None


def test_call_record_rejects_negative_duration():
    with pytest.raises(ValidationError):
        CallRecord(
            call_id="bad",
            audio_path="x.wav",
            duration_seconds=-1.0,
            domain="billing",
            source="kokoro-tts",
            privacy_notes="synthetic",
        )


def test_call_record_full_payload():
    rec = CallRecord(
        call_id="syn-002",
        audio_path="data/synthetic/syn-002.wav",
        duration_seconds=30.0,
        domain="refund",
        reference_transcript="Hello, how can I help.",
        speaker_segments=[
            {"speaker": "agent", "start": 0.0, "end": 2.0},
            {"speaker": "customer", "start": 2.0, "end": 5.0},
        ],
        event_labels=[
            {
                "event_type": "compliance",
                "start": 0.0,
                "end": 2.0,
                "metadata": {"subtype": "recording_disclosure", "polarity": "positive"},
            }
        ],
        summary_reference="Customer asked for a refund.",
        source="kokoro-tts",
        privacy_notes="synthetic",
    )
    assert len(rec.speaker_segments) == 2
    assert rec.event_labels[0].event_type == EventType.compliance


def test_call_record_roundtrips_through_json():
    rec = CallRecord(
        call_id="syn-003",
        audio_path="data/synthetic/syn-003.wav",
        duration_seconds=8.0,
        domain="cancellation",
        source="kokoro-tts",
        privacy_notes="synthetic",
    )
    dumped = rec.model_dump_json()
    loaded = CallRecord.model_validate_json(dumped)
    assert loaded == rec
