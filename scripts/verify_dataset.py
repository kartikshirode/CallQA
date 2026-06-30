"""Verify every registered call before we trust any metric.

For each record we run the pure consistency checks in callqa.datasets.verify.
Synthetic rows also get their label sidecar loaded so a later cross-check has
it, and if the audio file is on disk we confirm the wav duration matches the
registered duration within 0.1s. Missing wavs are skipped and noted, not failed,
because HarperValley audio may not be fetched on every machine.

Prints a PASS/FAIL line per call, a summary, and the negative-class ratio per
event type across the synthetic tier (the spec aims for roughly 40%). Exits
nonzero if anything fails so it can gate later work.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callqa.datasets.verify import verify_record
from callqa.registry.schema import CallRecord, EventType, Polarity
from callqa.registry.store import Registry

# Tolerance for the wav-duration vs registered-duration check.
DURATION_TOLERANCE_S = 0.1

SYNTHETIC_LABELS_DIR = Path("data/synthetic/labels")


def load_sidecar(record: CallRecord) -> dict | None:
    """Load a synthetic call's label sidecar if it exists."""
    if not record.source.lower().startswith("synthetic"):
        return None
    path = SYNTHETIC_LABELS_DIR / f"{record.call_id}.json"
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def wav_duration(path: Path) -> float | None:
    """Return wav duration in seconds, or None if the file is absent."""
    if not path.exists():
        return None
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def negative_class_ratios(records: list[CallRecord]) -> dict[str, tuple[int, int]]:
    """Count negative vs total labeled cases per event type on synthetic calls.

    Compliance polarity comes straight from each compliance event. For
    escalation, interruption and silence the negative case is a call that has no
    such event, so the unit is the call, not the event: a call without an
    escalation is one escalation negative.
    """
    synth = [r for r in records if r.source.lower().startswith("synthetic")]

    # Compliance negatives are counted per subtype: a missing recording
    # disclosure and a missing identity check are different negatives, so
    # pooling them would hide the real per-phrase balance.
    compliance_sub = defaultdict(lambda: [0, 0])  # subtype -> [neg, total]
    per_type_calls_with = defaultdict(int)

    for r in synth:
        present_types = set()
        for ev in r.event_labels or []:
            if ev.event_type == EventType.compliance:
                subtype = (ev.metadata or {}).get("subtype", "unknown")
                compliance_sub[subtype][1] += 1
                if (ev.metadata or {}).get("polarity") == Polarity.negative.value:
                    compliance_sub[subtype][0] += 1
            present_types.add(ev.event_type)
        for et in (EventType.escalation, EventType.interruption, EventType.silence):
            if et in present_types:
                per_type_calls_with[et] += 1

    n_calls = len(synth)
    ratios: dict[str, tuple[int, int]] = {}
    for subtype, (neg, total) in compliance_sub.items():
        ratios[f"compliance:{subtype}"] = (neg, total)
    for et in (EventType.escalation, EventType.interruption, EventType.silence):
        neg = n_calls - per_type_calls_with[et]
        ratios[et.value] = (neg, n_calls)
    return ratios


def main() -> int:
    registry = Registry()
    records = registry.list_all()

    total = 0
    clean = 0
    failures: list[tuple[str, list[str]]] = []
    skipped_audio = 0

    print("Per-call results")
    print("-" * 60)
    for rec in records:
        total += 1
        sidecar = load_sidecar(rec)
        problems = verify_record(rec, sidecar)

        # Audio duration cross-check when the wav is on disk.
        audio_path = Path(rec.audio_path)
        dur = wav_duration(audio_path)
        if dur is None:
            skipped_audio += 1
        else:
            if abs(dur - rec.duration_seconds) > DURATION_TOLERANCE_S:
                problems.append(
                    f"wav duration {dur:.3f}s differs from registered "
                    f"{rec.duration_seconds:.3f}s by more than {DURATION_TOLERANCE_S}s"
                )

        if problems:
            failures.append((rec.call_id, problems))
            print(f"FAIL  {rec.call_id}")
            for p in problems:
                print(f"        - {p}")
        else:
            clean += 1
            print(f"PASS  {rec.call_id}")

    print("-" * 60)
    print("Summary")
    print(f"  checked:        {total}")
    print(f"  clean:          {clean}")
    print(f"  failed:         {len(failures)}")
    print(f"  audio skipped:  {skipped_audio} (wav not on disk)")

    if failures:
        print("\nProblems found:")
        for call_id, problems in failures:
            for p in problems:
                print(f"  {call_id}: {p}")

    print("\nNegative-class ratio across synthetic tier (target near 40%)")
    print("-" * 60)
    for event_type, (neg, denom) in negative_class_ratios(records).items():
        pct = (neg / denom * 100.0) if denom else 0.0
        print(f"  {event_type:<14} {neg}/{denom}  = {pct:.1f}%")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
