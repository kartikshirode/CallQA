"""Pydantic v2 models for the CallQA audio registry.

These mirror the MVP schema in PLAN.md and the dataset design spec. Every call
we benchmark on becomes one CallRecord. Synthetic rows carry full event labels,
speaker turns and a reference summary; real-anchor rows usually carry just a
reference transcript and speaker turns.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Domain(str, Enum):
    """Call domain. Covers the PLAN.md values and the contact-center
    domains named in the dataset spec."""

    billing = "billing"
    refund = "refund"
    cancellation = "cancellation"
    tech_support = "tech_support"
    support = "support"
    meeting_surrogate = "meeting-surrogate"
    synthetic = "synthetic"


class EventType(str, Enum):
    """Operational events the pipeline detects and scores against."""

    silence = "silence"
    interruption = "interruption"
    escalation = "escalation"
    compliance = "compliance"


class ComplianceSubtype(str, Enum):
    """Which compliance phrase an event is about. Used in EventLabel.metadata."""

    recording_disclosure = "recording_disclosure"
    identity_verification = "identity_verification"
    closing = "closing"


class Polarity(str, Enum):
    """Whether a compliance event is a hit (phrase present) or a miss."""

    positive = "positive"
    negative = "negative"


class SpeakerSegment(BaseModel):
    """One speaker turn on the call timeline."""

    speaker: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check_order(self) -> "SpeakerSegment":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class EventLabel(BaseModel):
    """A labeled operational event anchored to a time span.

    metadata is free-form but for compliance events it typically holds
    a 'subtype' (ComplianceSubtype value) and a 'polarity' (positive/negative).
    """

    event_type: EventType
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    metadata: Optional[dict] = None

    @model_validator(mode="after")
    def _check_order(self) -> "EventLabel":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class CallRecord(BaseModel):
    """One benchmarked call. This is the unit the whole pipeline runs on."""

    call_id: str
    audio_path: str
    duration_seconds: float = Field(ge=0.0)
    domain: Domain
    reference_transcript: Optional[str] = None
    speaker_segments: Optional[list[SpeakerSegment]] = None
    event_labels: Optional[list[EventLabel]] = None
    summary_reference: Optional[str] = None
    source: str
    privacy_notes: str
