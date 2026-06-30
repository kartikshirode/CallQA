"""Register the synthetic telephone tier into the audio registry.

Run it:
    python scripts/register_synthetic.py

For each call it reads the label sidecar, rebuilds the speaker segments and
event labels, points audio_path at the telephone wav (the one the pipeline will
actually transcribe), and adds a CallRecord to the registry at the default
path. It is idempotent: a call already in the registry is skipped, so rerunning
is safe and never crashes on a duplicate.

After the run it prints a summary: total records, counts by source, counts by
domain.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callqa.registry.schema import (  # noqa: E402
    CallRecord,
    EventLabel,
    SpeakerSegment,
)
from callqa.registry.store import Registry  # noqa: E402

LABELS_DIR = Path("data/synthetic/labels")
TELEPHONE_DIR = Path("data/synthetic/audio/telephone")
SOURCE = "synthetic-kokoro-v1"
PRIVACY = "synthetic"


def build_record(call_id: str, data: dict) -> CallRecord:
    """Turn a label sidecar dict into a CallRecord for the telephone audio."""
    segments = [SpeakerSegment(**seg) for seg in data.get("speaker_segments", [])]
    events = [EventLabel(**ev) for ev in data.get("event_labels", [])]
    audio_path = TELEPHONE_DIR / f"{call_id}.wav"
    return CallRecord(
        call_id=call_id,
        audio_path=str(audio_path),
        duration_seconds=data["duration_seconds"],
        domain=data["domain"],
        reference_transcript=data.get("reference_transcript"),
        speaker_segments=segments,
        event_labels=events,
        summary_reference=None,
        source=SOURCE,
        privacy_notes=PRIVACY,
    )


def main() -> None:
    registry = Registry()
    label_files = sorted(LABELS_DIR.glob("*.json"))

    added = 0
    skipped = 0
    for path in label_files:
        call_id = path.stem
        if registry.get(call_id) is not None:
            skipped += 1
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        record = build_record(call_id, data)
        registry.add(record)
        added += 1

    print(f"Added {added}, skipped {skipped} (already present).\n")

    records = registry.list_all()
    by_source = Counter(r.source for r in records)
    by_domain = Counter(r.domain.value for r in records)

    print(f"Registry at {registry.path}")
    print(f"Total records: {len(records)}\n")

    print("By source:")
    for source, n in sorted(by_source.items()):
        print(f"  {source}: {n}")

    print("\nBy domain:")
    for domain, n in sorted(by_domain.items()):
        print(f"  {domain}: {n}")


if __name__ == "__main__":
    main()
