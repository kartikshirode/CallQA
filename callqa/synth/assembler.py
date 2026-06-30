"""Timeline assembler: turn per-turn clips into one mixed call plus exact labels.

This is the part that makes the labels ground truth. It takes clips that were
already synthesized (so it never touches Kokoro or a GPU) and lays them on a
single timeline. Where the script asked for a gap, we insert silence and, if the
gap is long enough, label it. Where the script asked for an interruption, we
start the next clip early and sum the overlap, then label that span. Compliance
and escalation tags carry the timestamps of the turn they sit on.

Same clips plus same script give byte-identical output, so the dataset is
reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from callqa.registry.schema import EventLabel, EventType, SpeakerSegment
from callqa.synth.script_schema import CallScript

# Mirror the script generator so silence labeling matches the script intent.
SILENCE_PAUSE_THRESHOLD = 2.0

# How far an interrupting turn starts before the previous one ends.
DEFAULT_OVERLAP_SECONDS = 0.6


@dataclass
class AssembledCall:
    """The assembled call: mixed audio plus the labels that describe it."""

    waveform: np.ndarray
    sample_rate: int
    speaker_segments: list[SpeakerSegment]
    event_labels: list[EventLabel]
    duration_seconds: float
    reference_transcript: str


def _samples(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def assemble_call(
    script: CallScript,
    clips: list[np.ndarray],
    sample_rate: int,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> AssembledCall:
    """Place clips on a timeline and emit the waveform plus exact labels.

    clips must be one array per turn, in turn order. Each clip is a 1-D float
    waveform at sample_rate. Raises ValueError on a count mismatch.
    """
    turns = script.turns
    if len(clips) != len(turns):
        raise ValueError(
            f"need one clip per turn: {len(turns)} turns but {len(clips)} clips"
        )

    if not turns:
        return AssembledCall(
            waveform=np.zeros(0, dtype=np.float32),
            sample_rate=sample_rate,
            speaker_segments=[],
            event_labels=[],
            duration_seconds=0.0,
            reference_transcript="",
        )

    # First pass on the sample grid: figure out where each turn lands, in
    # samples, so timestamps come straight from integer offsets (deterministic).
    starts: list[int] = []
    ends: list[int] = []
    silence_spans: list[tuple[int, int]] = []
    interrupt_spans: list[tuple[int, int]] = []

    cursor = 0  # next free sample on the timeline
    prev_end = 0
    for i, (turn, clip) in enumerate(zip(turns, clips)):
        clip = np.asarray(clip, dtype=np.float32).reshape(-1)
        length = clip.shape[0]

        if i > 0 and turn.interrupt_prev:
            # Start early, before the previous turn ended.
            overlap = min(_samples(overlap_seconds, sample_rate), prev_end)
            start = max(prev_end - overlap, 0)
            if start < prev_end:
                interrupt_spans.append((start, prev_end))
        else:
            pause = _samples(turn.pause_before, sample_rate)
            start = cursor + pause
            if turn.pause_before >= SILENCE_PAUSE_THRESHOLD and pause > 0:
                silence_spans.append((cursor, cursor + pause))

        end = start + length
        starts.append(start)
        ends.append(end)
        cursor = end
        prev_end = end

    total = max(ends) if ends else 0

    # Second pass: mix clips onto the buffer, summing any overlap.
    buffer = np.zeros(total, dtype=np.float32)
    for clip, start in zip(clips, starts):
        clip = np.asarray(clip, dtype=np.float32).reshape(-1)
        buffer[start : start + clip.shape[0]] += clip

    def to_seconds(sample: int) -> float:
        return sample / sample_rate

    speaker_segments = [
        SpeakerSegment(
            speaker=turn.speaker,
            start=to_seconds(start),
            end=to_seconds(end),
        )
        for turn, start, end in zip(turns, starts, ends)
    ]

    event_labels: list[EventLabel] = []

    # Silence and interruption spans come from the timeline layout.
    for start, end in silence_spans:
        event_labels.append(
            EventLabel(
                event_type=EventType.silence,
                start=to_seconds(start),
                end=to_seconds(end),
            )
        )
    for start, end in interrupt_spans:
        event_labels.append(
            EventLabel(
                event_type=EventType.interruption,
                start=to_seconds(start),
                end=to_seconds(end),
            )
        )

    # Compliance and escalation tags inherit the owning turn's timestamps.
    for turn, start, end in zip(turns, starts, ends):
        for tag in turn.event_tags:
            if tag.event_type in (EventType.silence, EventType.interruption):
                # These are owned by the timeline, already added above.
                continue
            metadata = None
            if tag.subtype is not None or tag.polarity is not None:
                metadata = {
                    "subtype": tag.subtype.value if tag.subtype else None,
                    "polarity": tag.polarity.value if tag.polarity else None,
                }
            event_labels.append(
                EventLabel(
                    event_type=tag.event_type,
                    start=to_seconds(start),
                    end=to_seconds(end),
                    metadata=metadata,
                )
            )

    reference_transcript = " ".join(turn.text.strip() for turn in turns).strip()

    return AssembledCall(
        waveform=buffer,
        sample_rate=sample_rate,
        speaker_segments=speaker_segments,
        event_labels=event_labels,
        duration_seconds=to_seconds(total),
        reference_transcript=reference_transcript,
    )
