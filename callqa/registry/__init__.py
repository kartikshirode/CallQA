"""Audio registry: schema models plus a JSONL-backed store.

The registry tracks every call we benchmark on, where its audio lives, what
labels it carries, and whether the source is safe to process. Anything marked
private or unknown gets refused before it ever reaches the pipeline.
"""

from callqa.registry.schema import (
    CallRecord,
    ComplianceSubtype,
    Domain,
    EventLabel,
    EventType,
    Polarity,
    SpeakerSegment,
)
from callqa.registry.store import PrivacyRefusedError, Registry

__all__ = [
    "CallRecord",
    "ComplianceSubtype",
    "Domain",
    "EventLabel",
    "EventType",
    "Polarity",
    "SpeakerSegment",
    "Registry",
    "PrivacyRefusedError",
]
