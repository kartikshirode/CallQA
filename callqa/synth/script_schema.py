"""Pydantic v2 models for synthetic call scripts (Phase 1).

A script is an ordered list of tagged turns. Phase 2 will synthesize each turn
to audio and assemble it on a timeline. The event labels live in the tags here,
so they are exact by construction: a pause becomes a silence label, an overlap
becomes an interruption, and a recording-disclosure line carries a compliance
tag. No audio is produced in this phase.

These reuse the Phase 0 registry enums. We do not redefine EventType,
ComplianceSubtype, Polarity or Domain.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from callqa.registry.schema import (
    ComplianceSubtype,
    Domain,
    EventType,
    Polarity,
)


class EventTag(BaseModel):
    """One event intent attached to a turn.

    For compliance events, subtype says which phrase (recording_disclosure,
    identity_verification, closing) and polarity says whether the phrase is
    present (positive) or deliberately omitted (negative). For the other event
    types subtype and polarity stay None.
    """

    event_type: EventType
    subtype: Optional[ComplianceSubtype] = None
    polarity: Optional[Polarity] = None

    @model_validator(mode="after")
    def _check_compliance_fields(self) -> "EventTag":
        if self.event_type is EventType.compliance:
            if self.subtype is None or self.polarity is None:
                raise ValueError(
                    "compliance tags need both a subtype and a polarity"
                )
        return self


class ScriptTurn(BaseModel):
    """One spoken turn in the script.

    pause_before is the gap in seconds inserted before this turn. A large gap
    becomes a silence label in Phase 2. interrupt_prev marks that this turn
    overlaps the one before it, which becomes an interruption label.
    """

    speaker: Literal["agent", "customer"]
    text: str
    pause_before: float = Field(default=0.0, ge=0.0)
    interrupt_prev: bool = False
    event_tags: list[EventTag] = Field(default_factory=list)


class CallScript(BaseModel):
    """A full call script: ordered turns plus metadata for reproducibility."""

    call_id: str
    domain: Domain
    seed: int
    turns: list[ScriptTurn]
    notes: str = ""

    def present_event_types(self) -> set[EventType]:
        """Return the set of event types this call is a positive case for.

        Silence, interruption and escalation count when a turn carries the tag.
        Compliance is handled differently: a call is a compliance positive only
        when it has no labeled miss. If any required phrase is omitted (a
        negative compliance tag), the call is a compliance negative, even if
        another phrase was present. That matches the negative-class policy,
        where a missing required phrase is the thing the detector must catch.
        """
        present: set[EventType] = set()
        compliance_positive = False
        compliance_miss = False
        for turn in self.turns:
            for tag in turn.event_tags:
                if tag.event_type is EventType.compliance:
                    if tag.polarity is Polarity.positive:
                        compliance_positive = True
                    elif tag.polarity is Polarity.negative:
                        compliance_miss = True
                else:
                    present.add(tag.event_type)
        if compliance_positive and not compliance_miss:
            present.add(EventType.compliance)
        return present
