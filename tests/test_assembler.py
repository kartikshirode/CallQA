"""Tests for the timeline assembler.

The assembler takes already-synthesized per-turn audio arrays plus a CallScript
and lays them on a timeline, producing the mixed waveform and exact labels. No
Kokoro and no GPU are touched here; stub arrays of ones stand in for clips.
"""
from __future__ import annotations

import numpy as np
import pytest

from callqa.registry.schema import (
    ComplianceSubtype,
    Domain,
    EventType,
    Polarity,
)
from callqa.synth.assembler import (
    DEFAULT_OVERLAP_SECONDS,
    AssembledCall,
    assemble_call,
)
from callqa.synth.script_schema import CallScript, EventTag, ScriptTurn

SR = 24000


def clip(seconds: float) -> np.ndarray:
    """A stub clip: ones, so overlaps sum to a visible value."""
    return np.ones(int(round(seconds * SR)), dtype=np.float32)


def make_script(turns: list[ScriptTurn], call_id: str = "test-01") -> CallScript:
    return CallScript(
        call_id=call_id,
        domain=Domain.billing,
        seed=1,
        turns=turns,
    )


def test_clip_count_must_match_turns():
    script = make_script(
        [
            ScriptTurn(speaker="agent", text="hello"),
            ScriptTurn(speaker="customer", text="hi"),
        ]
    )
    with pytest.raises(ValueError):
        assemble_call(script, [clip(1.0)], sample_rate=SR)


def test_pause_before_becomes_silence_label():
    script = make_script(
        [
            ScriptTurn(speaker="agent", text="hello"),
            ScriptTurn(speaker="customer", text="thinking", pause_before=3.0),
        ]
    )
    result = assemble_call(script, [clip(1.0), clip(1.0)], sample_rate=SR)

    silences = [e for e in result.event_labels if e.event_type is EventType.silence]
    assert len(silences) == 1
    sil = silences[0]
    # First clip ends at 1.0, the 3.0s gap spans 1.0 to 4.0.
    assert sil.start == pytest.approx(1.0, abs=0.01)
    assert (sil.end - sil.start) == pytest.approx(3.0, abs=0.01)


def test_short_pause_is_not_a_silence_label():
    script = make_script(
        [
            ScriptTurn(speaker="agent", text="hello"),
            ScriptTurn(speaker="customer", text="quick", pause_before=1.0),
        ]
    )
    result = assemble_call(script, [clip(1.0), clip(1.0)], sample_rate=SR)
    silences = [e for e in result.event_labels if e.event_type is EventType.silence]
    assert silences == []


def test_interrupt_yields_interruption_label_and_shorter_audio():
    turns = [
        ScriptTurn(speaker="agent", text="long explanation here"),
        ScriptTurn(
            speaker="customer",
            text="wait",
            interrupt_prev=True,
            event_tags=[EventTag(event_type=EventType.interruption)],
        ),
    ]
    result = assemble_call(make_script(turns), [clip(2.0), clip(1.0)], sample_rate=SR)

    interrupts = [
        e for e in result.event_labels if e.event_type is EventType.interruption
    ]
    assert len(interrupts) == 1
    span = interrupts[0].end - interrupts[0].start
    assert span == pytest.approx(DEFAULT_OVERLAP_SECONDS, abs=0.01)

    # Naive concatenation would be 2.0 + 1.0 = 3.0s; overlap makes it shorter.
    naive = 3.0
    assert result.duration_seconds == pytest.approx(naive - DEFAULT_OVERLAP_SECONDS, abs=0.01)
    assert len(result.waveform) == int(round(result.duration_seconds * SR))


def test_interruption_span_overlaps_previous_turn():
    turns = [
        ScriptTurn(speaker="agent", text="long"),
        ScriptTurn(
            speaker="customer",
            text="cut in",
            interrupt_prev=True,
            event_tags=[EventTag(event_type=EventType.interruption)],
        ),
    ]
    result = assemble_call(make_script(turns), [clip(2.0), clip(1.0)], sample_rate=SR)
    prev = result.speaker_segments[0]
    cur = result.speaker_segments[1]
    interrupt = next(
        e for e in result.event_labels if e.event_type is EventType.interruption
    )
    # The interruption starts inside the previous turn and at the current start.
    assert cur.start < prev.end
    assert interrupt.start == pytest.approx(cur.start, abs=0.01)
    assert interrupt.end == pytest.approx(prev.end, abs=0.01)


def test_speaker_segments_monotonic_and_end_matches_duration():
    turns = [
        ScriptTurn(speaker="agent", text="a"),
        ScriptTurn(speaker="customer", text="b", pause_before=3.0),
        ScriptTurn(speaker="agent", text="c"),
    ]
    result = assemble_call(
        make_script(turns), [clip(1.0), clip(1.0), clip(1.0)], sample_rate=SR
    )
    segs = result.speaker_segments
    assert len(segs) == 3
    for earlier, later in zip(segs, segs[1:]):
        assert later.start >= earlier.start
        assert earlier.end <= later.start + 1e-6 or later.start >= earlier.start
    assert segs[-1].end == pytest.approx(result.duration_seconds, abs=1e-6)


def test_reference_transcript_is_turns_in_order():
    turns = [
        ScriptTurn(speaker="agent", text="one"),
        ScriptTurn(speaker="customer", text="two"),
        ScriptTurn(speaker="agent", text="three"),
    ]
    result = assemble_call(
        make_script(turns), [clip(1.0), clip(1.0), clip(1.0)], sample_rate=SR
    )
    assert result.reference_transcript == "one two three"


def test_compliance_tag_inherits_turn_timestamps_and_metadata():
    turns = [
        ScriptTurn(speaker="agent", text="first"),
        ScriptTurn(
            speaker="agent",
            text="this call may be recorded",
            event_tags=[
                EventTag(
                    event_type=EventType.compliance,
                    subtype=ComplianceSubtype.recording_disclosure,
                    polarity=Polarity.positive,
                )
            ],
        ),
    ]
    result = assemble_call(make_script(turns), [clip(1.0), clip(2.0)], sample_rate=SR)
    comp = next(
        e for e in result.event_labels if e.event_type is EventType.compliance
    )
    owning = result.speaker_segments[1]
    assert comp.start == pytest.approx(owning.start, abs=1e-6)
    assert comp.end == pytest.approx(owning.end, abs=1e-6)
    assert comp.metadata["subtype"] == ComplianceSubtype.recording_disclosure.value
    assert comp.metadata["polarity"] == Polarity.positive.value


def test_escalation_tag_inherits_turn_timestamps():
    turns = [
        ScriptTurn(speaker="customer", text="calm"),
        ScriptTurn(
            speaker="customer",
            text="get me a manager",
            event_tags=[EventTag(event_type=EventType.escalation)],
        ),
    ]
    result = assemble_call(make_script(turns), [clip(1.0), clip(1.0)], sample_rate=SR)
    esc = next(
        e for e in result.event_labels if e.event_type is EventType.escalation
    )
    owning = result.speaker_segments[1]
    assert esc.start == pytest.approx(owning.start, abs=1e-6)
    assert esc.end == pytest.approx(owning.end, abs=1e-6)


def test_assembly_is_deterministic():
    turns = [
        ScriptTurn(speaker="agent", text="a"),
        ScriptTurn(speaker="customer", text="b", pause_before=3.0),
        ScriptTurn(
            speaker="agent",
            text="c",
            interrupt_prev=True,
            event_tags=[EventTag(event_type=EventType.interruption)],
        ),
    ]
    clips = [clip(1.0), clip(1.0), clip(1.0)]
    r1 = assemble_call(make_script(turns), clips, sample_rate=SR)
    r2 = assemble_call(make_script(turns), clips, sample_rate=SR)
    assert r1.duration_seconds == r2.duration_seconds
    assert np.array_equal(r1.waveform, r2.waveform)
    assert [(s.start, s.end) for s in r1.speaker_segments] == [
        (s.start, s.end) for s in r2.speaker_segments
    ]


def test_returns_assembled_call_type():
    script = make_script([ScriptTurn(speaker="agent", text="hi")])
    result = assemble_call(script, [clip(1.0)], sample_rate=SR)
    assert isinstance(result, AssembledCall)
    assert result.sample_rate == SR
