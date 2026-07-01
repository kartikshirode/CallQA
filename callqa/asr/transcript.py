"""Pure transcript object shared across ASR providers.

No GPU, no token, no network. This is the interface the Whisper adapter and the
cloud stubs target, so it stays provider-neutral: text, timed segments, timing,
and a model tag. It also serializes to and from JSON for the on-disk cache.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TranscriptSegment(BaseModel):
    """One timed chunk of transcript. start and end are seconds from call zero."""

    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str

    @model_validator(mode="after")
    def _check_order(self) -> "TranscriptSegment":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class Transcript(BaseModel):
    """A full transcription result for one call, provider-neutral.

    model is a tag like "faster-whisper/base.en". latency_seconds is wall-clock
    time the provider took; audio_seconds is the length of the audio it ran on.
    """

    call_id: str
    model: str
    text: str
    segments: list[TranscriptSegment]
    latency_seconds: float = Field(ge=0.0)
    audio_seconds: float = Field(ge=0.0)

    @property
    def real_time_factor(self) -> float:
        """Audio length over wall-clock time. Above 1.0 means faster than real
        time. Returns 0.0 when latency is zero so it never divides by zero."""
        if self.latency_seconds > 0:
            return self.audio_seconds / self.latency_seconds
        return 0.0

    def to_json(self) -> str:
        """Serialize to a JSON string for the on-disk transcript cache."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> "Transcript":
        """Rebuild a Transcript from the cached JSON string."""
        return cls.model_validate_json(s)
