"""Week 1 dataset - the synthetic tier, end to end.

The synthetic core is scripts built from domain banks, then laid on a timeline
into audio plus ground-truth labels. The classes follow that pipeline: script
schema, banks, generator, batch summary, assembler. This is where the event
labels come from, so it carries the most checks.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from pydantic import ValidationError

from callqa.registry.schema import (
    ComplianceSubtype,
    Domain,
    EventType,
    Polarity,
)
from callqa.synth.assembler import (
    DEFAULT_OVERLAP_SECONDS,
    assemble_call,
)
from callqa.synth.banks import BANKS, REQUIRED_SLOTS
from callqa.synth.generator import (
    SILENCE_PAUSE_THRESHOLD,
    batch_event_summary,
    generate_batch,
    generate_call,
)
from callqa.synth.script_schema import CallScript, EventTag, ScriptTurn

SR = 24000


class TestScriptSchema:
    def test_event_tag_plain_and_compliance(self):
        plain = EventTag(event_type="silence")
        assert plain.event_type == EventType.silence
        assert plain.subtype is None and plain.polarity is None
        comp = EventTag(
            event_type="compliance",
            subtype="recording_disclosure",
            polarity="positive",
        )
        assert comp.subtype == ComplianceSubtype.recording_disclosure
        assert comp.polarity == Polarity.positive

    def test_event_tag_rejects_unknown_event_type(self):
        with pytest.raises(ValidationError):
            EventTag(event_type="laughter")

    def test_script_turn_defaults_and_tag(self):
        turn = ScriptTurn(speaker="agent", text="Hello there.")
        assert turn.pause_before == 0.0
        assert turn.interrupt_prev is False
        assert turn.event_tags == []
        tagged = ScriptTurn(
            speaker="agent",
            text="This call may be recorded.",
            event_tags=[{"event_type": "compliance", "subtype": "recording_disclosure", "polarity": "positive"}],
        )
        assert tagged.event_tags[0].event_type == EventType.compliance

    def test_script_turn_rejects_bad_speaker(self):
        with pytest.raises(ValidationError):
            ScriptTurn(speaker="robot", text="hi")

    def test_call_script_minimal_and_event_summary(self):
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
        assert script.domain == Domain.billing
        present = script.present_event_types()
        assert EventType.escalation in present
        assert EventType.silence in present
        assert EventType.interruption not in present

    def test_call_script_roundtrips_through_json(self):
        script = CallScript(
            call_id="syn-refund-01",
            domain="refund",
            seed=1,
            turns=[ScriptTurn(speaker="customer", text="I want my money back.")],
            notes="n",
        )
        assert CallScript.model_validate_json(script.model_dump_json()) == script


class TestBanks:
    def test_four_domains_each_have_every_slot(self):
        assert set(BANKS) == {"billing", "refund", "cancellation", "tech_support"}
        for domain, bank in BANKS.items():
            for slot in REQUIRED_SLOTS:
                assert slot in bank, f"{domain} missing slot {slot}"
                assert len(bank[slot]) >= 2, f"{domain}.{slot} needs alternatives"

    def test_lines_are_clean_non_empty_strings(self):
        for bank in BANKS.values():
            for lines in bank.values():
                for line in lines:
                    assert isinstance(line, str) and line.strip()
                    assert "—" not in line and "–" not in line


class TestGenerator:
    def test_generate_call_returns_valid_script(self):
        script = generate_call("billing", "syn-billing-01", seed=42)
        assert isinstance(script, CallScript)
        assert script.call_id == "syn-billing-01"
        assert len(script.turns) >= 6

    def test_generate_call_determinism(self):
        same = generate_call("refund", "syn-refund-01", seed=7).model_dump_json()
        again = generate_call("refund", "syn-refund-01", seed=7).model_dump_json()
        assert same == again
        other = generate_call("billing", "syn-billing-01", seed=2).model_dump_json()
        assert other != generate_call("billing", "syn-billing-01", seed=1).model_dump_json()

    def test_generate_batch_shape_and_byte_identical(self):
        batch = generate_batch(seed=2026)
        assert len(batch) == 20
        assert Counter(s.domain.value for s in batch) == {
            "billing": 5, "refund": 5, "cancellation": 5, "tech_support": 5,
        }
        first = [s.model_dump_json() for s in batch]
        second = [s.model_dump_json() for s in generate_batch(seed=2026)]
        assert first == second
        for script in batch:
            CallScript.model_validate_json(script.model_dump_json())

    @pytest.mark.parametrize(
        "event_type",
        [EventType.silence, EventType.interruption, EventType.escalation, EventType.compliance],
    )
    def test_negative_ratio_near_forty_percent(self, event_type):
        batch = generate_batch(seed=2026)
        neg = sum(1 for s in batch if event_type not in s.present_event_types())
        ratio = neg / len(batch)
        assert 0.30 <= ratio <= 0.50, f"{event_type} negative ratio {ratio}"

    def test_silence_tag_matches_long_pause(self):
        batch = generate_batch(seed=2026)
        seen = False
        for script in batch:
            for turn in script.turns:
                has_tag = any(t.event_type is EventType.silence for t in turn.event_tags)
                if turn.pause_before >= SILENCE_PAUSE_THRESHOLD:
                    assert has_tag
                    seen = True
                if has_tag:
                    assert turn.pause_before >= SILENCE_PAUSE_THRESHOLD
        assert seen

    def test_interruption_tag_matches_flag(self):
        batch = generate_batch(seed=2026)
        seen = False
        for script in batch:
            for turn in script.turns:
                has_tag = any(t.event_type is EventType.interruption for t in turn.event_tags)
                if turn.interrupt_prev:
                    assert has_tag
                    seen = True
                if has_tag:
                    assert turn.interrupt_prev
        assert seen

    def test_required_compliance_subtypes_appear_both_polarities(self):
        batch = generate_batch(seed=2026)
        for subtype in (ComplianceSubtype.recording_disclosure,
                        ComplianceSubtype.identity_verification):
            positives = negatives = 0
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

    def test_subtype_never_both_polarities_in_one_call(self):
        for script in generate_batch(seed=2026):
            per_subtype = {}
            for turn in script.turns:
                for tag in turn.event_tags:
                    if tag.event_type is EventType.compliance:
                        per_subtype.setdefault(tag.subtype, set()).add(tag.polarity)
            for subtype, polarities in per_subtype.items():
                assert not (Polarity.positive in polarities and Polarity.negative in polarities), (
                    f"{script.call_id} has both polarities for {subtype}"
                )


class TestBatchSummary:
    def test_summary_covers_events_counts_and_ratio(self):
        batch = generate_batch(seed=2026)
        summary = batch_event_summary(batch)
        assert set(summary) == {
            EventType.silence, EventType.interruption,
            EventType.escalation, EventType.compliance,
        }
        for stats in summary.values():
            assert stats["positive"] + stats["negative"] == len(batch)
            assert 0.30 <= stats["negative_ratio"] <= 0.50


def _clip(seconds: float) -> np.ndarray:
    """A stub clip of ones, so overlaps sum to a visible value."""
    return np.ones(int(round(seconds * SR)), dtype=np.float32)


def _script(turns: list[ScriptTurn], call_id: str = "test-01") -> CallScript:
    return CallScript(call_id=call_id, domain=Domain.billing, seed=1, turns=turns)


class TestAssembler:
    def test_clip_count_must_match_turns(self):
        script = _script([
            ScriptTurn(speaker="agent", text="hello"),
            ScriptTurn(speaker="customer", text="hi"),
        ])
        with pytest.raises(ValueError):
            assemble_call(script, [_clip(1.0)], sample_rate=SR)

    def test_long_pause_becomes_silence_label(self):
        script = _script([
            ScriptTurn(speaker="agent", text="hello"),
            ScriptTurn(speaker="customer", text="thinking", pause_before=3.0),
        ])
        result = assemble_call(script, [_clip(1.0), _clip(1.0)], sample_rate=SR)
        silences = [e for e in result.event_labels if e.event_type is EventType.silence]
        assert len(silences) == 1
        assert silences[0].start == pytest.approx(1.0, abs=0.01)
        assert (silences[0].end - silences[0].start) == pytest.approx(3.0, abs=0.01)

    def test_short_pause_is_not_a_silence_label(self):
        script = _script([
            ScriptTurn(speaker="agent", text="hello"),
            ScriptTurn(speaker="customer", text="quick", pause_before=1.0),
        ])
        result = assemble_call(script, [_clip(1.0), _clip(1.0)], sample_rate=SR)
        assert [e for e in result.event_labels if e.event_type is EventType.silence] == []

    def test_interrupt_label_span_and_shorter_audio(self):
        turns = [
            ScriptTurn(speaker="agent", text="long explanation here"),
            ScriptTurn(speaker="customer", text="wait", interrupt_prev=True,
                       event_tags=[EventTag(event_type=EventType.interruption)]),
        ]
        result = assemble_call(_script(turns), [_clip(2.0), _clip(1.0)], sample_rate=SR)
        interrupts = [e for e in result.event_labels if e.event_type is EventType.interruption]
        assert len(interrupts) == 1
        span = interrupts[0].end - interrupts[0].start
        assert span == pytest.approx(DEFAULT_OVERLAP_SECONDS, abs=0.01)
        # Overlap makes the call shorter than naive 2.0 + 1.0 concatenation.
        assert result.duration_seconds == pytest.approx(3.0 - DEFAULT_OVERLAP_SECONDS, abs=0.01)
        assert len(result.waveform) == int(round(result.duration_seconds * SR))
        # The span sits inside the previous turn and starts at the current turn.
        prev, cur = result.speaker_segments[0], result.speaker_segments[1]
        assert cur.start < prev.end
        assert interrupts[0].start == pytest.approx(cur.start, abs=0.01)
        assert interrupts[0].end == pytest.approx(prev.end, abs=0.01)

    def test_segments_monotonic_end_matches_duration_and_transcript(self):
        turns = [
            ScriptTurn(speaker="agent", text="one"),
            ScriptTurn(speaker="customer", text="two", pause_before=3.0),
            ScriptTurn(speaker="agent", text="three"),
        ]
        result = assemble_call(_script(turns), [_clip(1.0), _clip(1.0), _clip(1.0)], sample_rate=SR)
        segs = result.speaker_segments
        assert len(segs) == 3
        for earlier, later in zip(segs, segs[1:]):
            assert later.start >= earlier.start
        assert segs[-1].end == pytest.approx(result.duration_seconds, abs=1e-6)
        assert result.reference_transcript == "one two three"
        assert result.sample_rate == SR

    def test_compliance_and_escalation_tags_inherit_turn_timestamps(self):
        turns = [
            ScriptTurn(speaker="agent", text="first"),
            ScriptTurn(speaker="agent", text="this call may be recorded",
                       event_tags=[EventTag(event_type=EventType.compliance,
                                            subtype=ComplianceSubtype.recording_disclosure,
                                            polarity=Polarity.positive)]),
            ScriptTurn(speaker="customer", text="get me a manager",
                       event_tags=[EventTag(event_type=EventType.escalation)]),
        ]
        result = assemble_call(_script(turns), [_clip(1.0), _clip(2.0), _clip(1.0)], sample_rate=SR)
        comp = next(e for e in result.event_labels if e.event_type is EventType.compliance)
        comp_owner = result.speaker_segments[1]
        assert comp.start == pytest.approx(comp_owner.start, abs=1e-6)
        assert comp.end == pytest.approx(comp_owner.end, abs=1e-6)
        assert comp.metadata["subtype"] == ComplianceSubtype.recording_disclosure.value
        assert comp.metadata["polarity"] == Polarity.positive.value
        esc = next(e for e in result.event_labels if e.event_type is EventType.escalation)
        esc_owner = result.speaker_segments[2]
        assert esc.start == pytest.approx(esc_owner.start, abs=1e-6)
        assert esc.end == pytest.approx(esc_owner.end, abs=1e-6)

    def test_assembly_is_deterministic(self):
        turns = [
            ScriptTurn(speaker="agent", text="a"),
            ScriptTurn(speaker="customer", text="b", pause_before=3.0),
            ScriptTurn(speaker="agent", text="c", interrupt_prev=True,
                       event_tags=[EventTag(event_type=EventType.interruption)]),
        ]
        clips = [_clip(1.0), _clip(1.0), _clip(1.0)]
        r1 = assemble_call(_script(turns), clips, sample_rate=SR)
        r2 = assemble_call(_script(turns), clips, sample_rate=SR)
        assert r1.duration_seconds == r2.duration_seconds
        assert np.array_equal(r1.waveform, r2.waveform)
        assert [(s.start, s.end) for s in r1.speaker_segments] == [
            (s.start, s.end) for s in r2.speaker_segments
        ]
