"""Week 1 dataset - the registry contract.

Covers the pydantic models every call record must satisfy and the JSONL store
that holds them, including the privacy guard. Grouped into two classes so a
failure points straight at the schema or the store.
"""
import pytest
from pydantic import ValidationError

from callqa.registry.schema import (
    CallRecord,
    Domain,
    EventLabel,
    EventType,
    SpeakerSegment,
)
from callqa.registry.store import PrivacyRefusedError, Registry


class TestSchema:
    def test_speaker_segment_valid(self):
        seg = SpeakerSegment(speaker="agent", start=0.0, end=2.5)
        assert (seg.speaker, seg.start, seg.end) == ("agent", 0.0, 2.5)
        # equal start and end is allowed (a zero-length turn)
        assert SpeakerSegment(speaker="customer", start=1.0, end=1.0).start == 1.0

    def test_speaker_segment_rejects_end_before_start(self):
        with pytest.raises(ValidationError):
            SpeakerSegment(speaker="agent", start=3.0, end=1.0)

    def test_event_label_basic_and_metadata(self):
        plain = EventLabel(event_type="silence", start=4.0, end=6.0)
        assert plain.event_type == EventType.silence
        assert plain.metadata is None
        comp = EventLabel(
            event_type="compliance",
            start=1.0,
            end=2.0,
            metadata={"subtype": "recording_disclosure", "polarity": "positive"},
        )
        assert comp.metadata["subtype"] == "recording_disclosure"
        assert comp.metadata["polarity"] == "positive"

    def test_event_label_rejects_end_before_start(self):
        with pytest.raises(ValidationError):
            EventLabel(event_type="interruption", start=5.0, end=2.0)

    def test_event_label_rejects_unknown_event_type(self):
        with pytest.raises(ValidationError):
            EventLabel(event_type="laughter", start=0.0, end=1.0)

    def test_domain_covers_plan_and_spec_values(self):
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

    def test_call_record_minimal_valid(self):
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

    def test_call_record_rejects_negative_duration(self):
        with pytest.raises(ValidationError):
            CallRecord(
                call_id="bad",
                audio_path="x.wav",
                duration_seconds=-1.0,
                domain="billing",
                source="kokoro-tts",
                privacy_notes="synthetic",
            )

    def test_call_record_full_payload(self):
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

    def test_call_record_roundtrips_through_json(self):
        rec = CallRecord(
            call_id="syn-003",
            audio_path="data/synthetic/syn-003.wav",
            duration_seconds=8.0,
            domain="cancellation",
            source="kokoro-tts",
            privacy_notes="synthetic",
        )
        loaded = CallRecord.model_validate_json(rec.model_dump_json())
        assert loaded == rec


def _record(call_id="syn-001", domain="billing", source="kokoro-tts",
            privacy_notes="synthetic"):
    return CallRecord(
        call_id=call_id,
        audio_path=f"data/synthetic/{call_id}.wav",
        duration_seconds=10.0,
        domain=domain,
        source=source,
        privacy_notes=privacy_notes,
    )


class TestStore:
    def test_add_get_and_missing(self, tmp_path):
        reg = Registry(tmp_path / "registry.jsonl")
        rec = _record()
        reg.add(rec)
        assert reg.get("syn-001") == rec
        assert reg.get("nope") is None

    def test_list_all(self, tmp_path):
        reg = Registry(tmp_path / "registry.jsonl")
        reg.add(_record("a"))
        reg.add(_record("b"))
        assert {r.call_id for r in reg.list_all()} == {"a", "b"}

    def test_filter_by_domain_and_source(self, tmp_path):
        reg = Registry(tmp_path / "registry.jsonl")
        reg.add(_record("a", domain="billing", source="kokoro-tts"))
        reg.add(_record("b", domain="refund", source="harpervalley"))
        assert [r.call_id for r in reg.filter(domain="refund")] == ["b"]
        assert [r.call_id for r in reg.filter(source="harpervalley")] == ["b"]

    def test_save_and_reload_from_disk(self, tmp_path):
        path = tmp_path / "registry.jsonl"
        reg = Registry(path)
        reg.add(_record("a"))
        reg.add(_record("b"))
        reloaded = Registry(path)
        assert {r.call_id for r in reloaded.list_all()} == {"a", "b"}

    def test_add_duplicate_id_raises(self, tmp_path):
        reg = Registry(tmp_path / "registry.jsonl")
        reg.add(_record("a"))
        with pytest.raises(ValueError):
            reg.add(_record("a"))

    def test_privacy_guard_refuses_empty_private_unknown(self, tmp_path):
        reg = Registry(tmp_path / "registry.jsonl")
        for bad in (_record(source=""),
                    _record(privacy_notes="private"),
                    _record(privacy_notes="unknown")):
            with pytest.raises(PrivacyRefusedError):
                reg.add(bad)

    def test_refused_record_not_persisted(self, tmp_path):
        path = tmp_path / "registry.jsonl"
        reg = Registry(path)
        with pytest.raises(PrivacyRefusedError):
            reg.add(_record(source=""))
        assert Registry(path).list_all() == []
