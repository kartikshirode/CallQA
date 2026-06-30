"""Tests for the synthetic script schema models."""
import pytest
from pydantic import ValidationError

from callqa.registry.schema import ComplianceSubtype, Domain, EventType, Polarity
from callqa.synth.script_schema import CallScript, EventTag, ScriptTurn


def test_event_tag_plain_event():
    tag = EventTag(event_type="silence")
    assert tag.event_type == EventType.silence
    assert tag.subtype is None
    assert tag.polarity is None


def test_event_tag_compliance_with_subtype_and_polarity():
    tag = EventTag(
        event_type="compliance",
        subtype="recording_disclosure",
        polarity="positive",
    )
    assert tag.subtype == ComplianceSubtype.recording_disclosure
    assert tag.polarity == Polarity.positive


def test_event_tag_rejects_unknown_event_type():
    with pytest.raises(ValidationError):
        EventTag(event_type="laughter")


def test_script_turn_defaults():
    turn = ScriptTurn(speaker="agent", text="Hello there.")
    assert turn.pause_before == 0.0
    assert turn.interrupt_prev is False
    assert turn.event_tags == []


def test_script_turn_rejects_bad_speaker():
    with pytest.raises(ValidationError):
        ScriptTurn(speaker="robot", text="hi")


def test_script_turn_with_tag():
    turn = ScriptTurn(
        speaker="agent",
        text="This call may be recorded.",
        event_tags=[{"event_type": "compliance", "subtype": "recording_disclosure", "polarity": "positive"}],
    )
    assert turn.event_tags[0].event_type == EventType.compliance


def test_call_script_minimal():
    script = CallScript(
        call_id="syn-billing-01",
        domain="billing",
        seed=7,
        turns=[ScriptTurn(speaker="agent", text="Hi.")],
        notes="test",
    )
    assert script.domain == Domain.billing
    assert len(script.turns) == 1


def test_call_script_event_type_summary():
    script = CallScript(
        call_id="syn-billing-02",
        domain="billing",
        seed=7,
        turns=[
            ScriptTurn(speaker="agent", text="a", event_tags=[{"event_type": "escalation"}]),
            ScriptTurn(speaker="customer", text="b", pause_before=3.0, event_tags=[{"event_type": "silence"}]),
        ],
        notes="",
    )
    present = script.present_event_types()
    assert EventType.escalation in present
    assert EventType.silence in present
    assert EventType.interruption not in present


def test_call_script_roundtrips_through_json():
    script = CallScript(
        call_id="syn-refund-01",
        domain="refund",
        seed=1,
        turns=[ScriptTurn(speaker="customer", text="I want my money back.")],
        notes="n",
    )
    dumped = script.model_dump_json()
    loaded = CallScript.model_validate_json(dumped)
    assert loaded == script
