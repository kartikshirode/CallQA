"""Consistency checks for CallRecords.

These are the checks we run before publishing any metric, so we can say the 20
synthetic calls were actually verified rather than just assumed sound. Each
function takes a loaded CallRecord (synthetic calls may also pass their label
sidecar dict, though the registry row already holds the same fields) and returns
a list of problem strings. An empty list means the record is clean.

Everything here is pure: no disk, no network, no models. The wav duration check
lives in the verify script, not here, because it needs the file on disk.
"""
from __future__ import annotations

from typing import Optional

from callqa.registry.schema import CallRecord, EventType, Polarity

# Sources we treat as the synthetic tier. The synthetic event story only applies
# to these.
_SYNTHETIC_SOURCE_PREFIX = "synthetic"

# Compliance subtypes the script template is meant to place on every synthetic
# call. A missing positive here is allowed (it is a labeled miss), but the event
# itself should exist.
_REQUIRED_COMPLIANCE_SUBTYPES = {"recording_disclosure", "identity_verification"}


def _is_synthetic(record: CallRecord) -> bool:
    return record.source.lower().startswith(_SYNTHETIC_SOURCE_PREFIX)


def check_bounds(record: CallRecord) -> list[str]:
    """Every speaker segment and event label must sit inside the call.

    start and end stay within [0, duration_seconds] and end is not before start.
    The schema already guarantees start/end >= 0 and end >= start, so the new
    information here is the duration ceiling.
    """
    problems: list[str] = []
    dur = record.duration_seconds

    for i, seg in enumerate(record.speaker_segments or []):
        if seg.end > dur:
            problems.append(
                f"speaker_segment[{i}] end {seg.end} exceeds duration {dur}"
            )
        if seg.start > dur:
            problems.append(
                f"speaker_segment[{i}] start {seg.start} exceeds duration {dur}"
            )

    for i, ev in enumerate(record.event_labels or []):
        if ev.end > dur:
            problems.append(
                f"event_label[{i}] ({ev.event_type.value}) end {ev.end} "
                f"out of bounds, exceeds duration {dur}"
            )
        if ev.start > dur:
            problems.append(
                f"event_label[{i}] ({ev.event_type.value}) start {ev.start} "
                f"out of bounds, exceeds duration {dur}"
            )

    return problems


def check_transcript(record: CallRecord) -> list[str]:
    """The reference transcript must be present and not just whitespace."""
    text = record.reference_transcript
    if text is None or not text.strip():
        return [f"reference_transcript is empty for {record.call_id}"]
    return []


def check_speaker_segments(record: CallRecord) -> list[str]:
    """Segments must be ordered by start time.

    Overlaps from interruptions are expected, so we do not flag a segment that
    starts before the previous one ends. We only flag a segment that starts
    before the previous segment started, which would mean the list is out of
    order.
    """
    problems: list[str] = []
    segs = record.speaker_segments or []
    prev_start: Optional[float] = None
    for i, seg in enumerate(segs):
        if prev_start is not None and seg.start < prev_start:
            problems.append(
                f"speaker_segment[{i}] start {seg.start} is out of order, "
                f"earlier than previous start {prev_start}"
            )
        prev_start = seg.start
    return problems


def check_synthetic_events(record: CallRecord) -> list[str]:
    """Validate the event story for a synthetic call.

    Non-synthetic records are skipped and return no problems. For synthetic
    calls:
      - at least one compliance event must exist
      - a required compliance subtype carrying negative polarity is a labeled
        miss and is allowed, not a problem
      - escalation and interruption spans must have end >= start (schema already
        enforces this, but we keep an explicit guard so a future loosening of
        the schema cannot slip a bad span past us)
      - silence events must have a positive duration
    """
    if not _is_synthetic(record):
        return []

    problems: list[str] = []
    events = record.event_labels or []

    compliance_events = [e for e in events if e.event_type == EventType.compliance]
    if not compliance_events:
        problems.append(
            f"{record.call_id} has no compliance event; expected at least one"
        )

    for i, ev in enumerate(events):
        if ev.event_type == EventType.silence:
            if ev.end <= ev.start:
                problems.append(
                    f"silence event[{i}] has non-positive duration "
                    f"(start {ev.start}, end {ev.end})"
                )
        elif ev.event_type in (EventType.escalation, EventType.interruption):
            if ev.end < ev.start:
                problems.append(
                    f"{ev.event_type.value} event[{i}] span is reversed "
                    f"(start {ev.start}, end {ev.end})"
                )

    return problems


def verify_record(record: CallRecord, sidecar: Optional[dict] = None) -> list[str]:
    """Run every applicable check and return the combined problem list.

    sidecar is accepted for the synthetic tier so a caller can pass the loaded
    label JSON, but the registry row already carries the same fields, so the
    checks read from the record. The argument is kept for callers that want to
    cross-check the sidecar against the row in future.
    """
    problems: list[str] = []
    problems.extend(check_bounds(record))
    problems.extend(check_transcript(record))
    problems.extend(check_speaker_segments(record))
    problems.extend(check_synthetic_events(record))
    return problems
