"""JSONL-backed store for CallRecords.

The store keeps every record on one line of a JSONL file so the registry stays
diff-friendly and easy to inspect by hand. A privacy guard refuses any record
whose source is empty or whose privacy notes mark it private or unknown, per
the PLAN.md rule: do not process files marked private or unknown source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from callqa.registry.schema import CallRecord

# Privacy notes that are never safe to process.
_REFUSED_PRIVACY_NOTES = {"private", "unknown"}

DEFAULT_REGISTRY_PATH = Path("data/registry/registry.jsonl")


class PrivacyRefusedError(Exception):
    """Raised when a record is rejected by the privacy guard."""


class Registry:
    """Loads, holds and saves CallRecords backed by a JSONL file."""

    def __init__(self, path: Union[str, Path] = DEFAULT_REGISTRY_PATH):
        self.path = Path(path)
        self._records: dict[str, CallRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = CallRecord.model_validate_json(line)
            self._records[rec.call_id] = rec

    def _check_privacy(self, record: CallRecord) -> None:
        if not record.source.strip():
            raise PrivacyRefusedError(
                f"record {record.call_id!r} has an empty source; refusing"
            )
        if record.privacy_notes.strip().lower() in _REFUSED_PRIVACY_NOTES:
            raise PrivacyRefusedError(
                f"record {record.call_id!r} is marked "
                f"{record.privacy_notes!r}; refusing"
            )

    def add(self, record: CallRecord) -> None:
        """Add a record and persist. Refuses private/unknown sources and
        rejects duplicate call ids."""
        self._check_privacy(record)
        if record.call_id in self._records:
            raise ValueError(f"duplicate call_id {record.call_id!r}")
        self._records[record.call_id] = record
        self._save()

    def get(self, call_id: str) -> Optional[CallRecord]:
        return self._records.get(call_id)

    def list_all(self) -> list[CallRecord]:
        return list(self._records.values())

    def filter(
        self,
        *,
        source: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> list[CallRecord]:
        """Return records matching the given source and/or domain."""
        out = self.list_all()
        if source is not None:
            out = [r for r in out if r.source == source]
        if domain is not None:
            out = [r for r in out if r.domain.value == domain]
        return out

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [r.model_dump_json() for r in self._records.values()]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
