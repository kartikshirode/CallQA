"""Tests for the synthetic script generator."""
from collections import Counter

import pytest

from callqa.registry.schema import ComplianceSubtype, EventType, Polarity
from callqa.synth.generator import (
    SILENCE_PAUSE_THRESHOLD,
    generate_batch,
    generate_call,
)
from callqa.synth.script_schema import CallScript


def test_generate_call_returns_valid_script():
    script = generate_call("billing", "syn-billing-01", seed=42)
    assert isinstance(script, CallScript)
    assert script.call_id == "syn-billing-01"
    assert len(script.turns) >= 6


def test_generate_call_is_deterministic():
    a = generate_call("refund", "syn-refund-01", seed=7)
    b = generate_call("refund", "syn-refund-01", seed=7)
    assert a.model_dump_json() == b.model_dump_json()


def test_generate_call_different_seed_changes_text():
    a = generate_call("billing", "syn-billing-01", seed=1)
    b = generate_call("billing", "syn-billing-01", seed=2)
    assert a.model_dump_json() != b.model_dump_json()


def test_generate_batch_produces_twenty_scripts():
    batch = generate_batch(seed=2026)
    assert len(batch) == 20


def test_generate_batch_four_domains_five_each():
    batch = generate_batch(seed=2026)
    counts = Counter(s.domain.value for s in batch)
    assert counts == {
        "billing": 5,
        "refund": 5,
        "cancellation": 5,
        "tech_support": 5,
    }


def test_generate_batch_is_byte_identical_across_runs():
    first = [s.model_dump_json() for s in generate_batch(seed=2026)]
    second = [s.model_dump_json() for s in generate_batch(seed=2026)]
    assert first == second


def test_every_generated_script_validates():
    for script in generate_batch(seed=2026):
        # Re-validation through JSON proves the model is well formed.
        CallScript.model_validate_json(script.model_dump_json())


def _negative_count_for(batch, event_type):
    """How many calls are negatives (do not carry a positive) for this event."""
    negatives = 0
    for script in batch:
        if event_type not in script.present_event_types():
            negatives += 1
    return negatives


@pytest.mark.parametrize(
    "event_type",
    [EventType.silence, EventType.interruption, EventType.escalation, EventType.compliance],
)
def test_negative_ratio_near_forty_percent(event_type):
    batch = generate_batch(seed=2026)
    neg = _negative_count_for(batch, event_type)
    ratio = neg / len(batch)
    assert 0.30 <= ratio <= 0.50, f"{event_type} negative ratio {ratio}"


def test_silence_pause_produces_silence_tag():
    """Any turn with a large pause_before must carry a silence tag, and any
    silence tag must sit on a turn with a large pause."""
    batch = generate_batch(seed=2026)
    seen_silence = False
    for script in batch:
        for turn in script.turns:
            has_silence_tag = any(t.event_type is EventType.silence for t in turn.event_tags)
            if turn.pause_before >= SILENCE_PAUSE_THRESHOLD:
                assert has_silence_tag
                seen_silence = True
            if has_silence_tag:
                assert turn.pause_before >= SILENCE_PAUSE_THRESHOLD
    assert seen_silence


def test_interruption_flag_produces_interruption_tag():
    batch = generate_batch(seed=2026)
    seen_interrupt = False
    for script in batch:
        for turn in script.turns:
            has_tag = any(t.event_type is EventType.interruption for t in turn.event_tags)
            if turn.interrupt_prev:
                assert has_tag
                seen_interrupt = True
            if has_tag:
                assert turn.interrupt_prev
    assert seen_interrupt


def test_required_compliance_subtypes_appear_positive_and_negative():
    """recording_disclosure and identity_verification must each show up as a
    present positive in some calls and as an omitted negative in others."""
    batch = generate_batch(seed=2026)
    required = [
        ComplianceSubtype.recording_disclosure,
        ComplianceSubtype.identity_verification,
    ]
    for subtype in required:
        positives = 0
        negatives = 0
        for script in batch:
            for turn in script.turns:
                for tag in turn.event_tags:
                    if tag.event_type is EventType.compliance and tag.subtype is subtype:
                        if tag.polarity is Polarity.positive:
                            positives += 1
                        elif tag.polarity is Polarity.negative:
                            negatives += 1
        assert positives > 0, f"{subtype} never positive"
        assert negatives > 0, f"{subtype} never negative"


def test_negative_compliance_call_has_no_positive_for_that_subtype():
    """When a required phrase is omitted, the call should carry a negative tag
    for it and not also a positive tag for the same subtype."""
    batch = generate_batch(seed=2026)
    for script in batch:
        per_subtype_polarities = {}
        for turn in script.turns:
            for tag in turn.event_tags:
                if tag.event_type is EventType.compliance:
                    per_subtype_polarities.setdefault(tag.subtype, set()).add(tag.polarity)
        for subtype, polarities in per_subtype_polarities.items():
            assert not (Polarity.positive in polarities and Polarity.negative in polarities), (
                f"{script.call_id} has both polarities for {subtype}"
            )
