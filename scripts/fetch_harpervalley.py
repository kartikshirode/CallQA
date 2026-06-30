"""Fetch a subset of the HarperValley corpus and register it.

Run it:
    python scripts/fetch_harpervalley.py --limit 40 --seed 1234

It enumerates every sid in the public repo, takes a seeded sample of N, pulls
the four files per call, mixes the agent and caller channels into one mono wav
at 8 kHz, parses the gold transcript and speaker turns, and adds a CallRecord
to the registry. HarperValley has no silence/interruption/compliance gold, so
event_labels and summary_reference stay None. We do not invent labels.

The run is idempotent: a call already in the registry is skipped. It also writes
a subset manifest with the chosen sids and the seed so the exact subset can be
rebuilt later.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf  # noqa: E402

from callqa.datasets import harpervalley  # noqa: E402
from callqa.registry.schema import CallRecord  # noqa: E402
from callqa.registry.store import Registry  # noqa: E402

BASE_DIR = Path("data/harpervalley")
RAW_DIR = BASE_DIR / "raw"
AUDIO_DIR = BASE_DIR / "audio"
MANIFEST_PATH = BASE_DIR / "subset_manifest.json"

ID_PREFIX = "hv-"
DOMAIN = "support"
SOURCE = "harpervalley"
PRIVACY = "public, CC-BY-4.0, Gridspace-Stanford Harper Valley"

DEFAULT_LIMIT = 40
DEFAULT_SEED = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and register HarperValley calls.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = Registry()

    all_ids = harpervalley.list_call_ids()
    chosen = harpervalley.sample_call_ids(all_ids, args.limit, args.seed)
    print(f"Found {len(all_ids)} sids, selected {len(chosen)} with seed {args.seed}.\n")

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps({"seed": args.seed, "sids": chosen}, indent=2),
        encoding="utf-8",
    )

    added = 0
    skipped = 0
    for sid in chosen:
        call_id = ID_PREFIX + sid
        if registry.get(call_id) is not None:
            skipped += 1
            continue

        harpervalley.fetch_call(sid, RAW_DIR)
        mono_path = harpervalley.build_mono_call(sid, RAW_DIR, AUDIO_DIR)

        segments_json = json.loads(
            (RAW_DIR / sid / "transcript.json").read_text(encoding="utf-8")
        )
        reference_transcript, speaker_segments = harpervalley.gold_from_transcript(
            segments_json
        )

        info = sf.info(str(mono_path))
        duration = info.frames / float(info.samplerate)

        record = CallRecord(
            call_id=call_id,
            audio_path=str(mono_path),
            duration_seconds=duration,
            domain=DOMAIN,
            reference_transcript=reference_transcript,
            speaker_segments=speaker_segments,
            event_labels=None,
            summary_reference=None,
            source=SOURCE,
            privacy_notes=PRIVACY,
        )
        registry.add(record)
        added += 1
        print(f"  {call_id}: {duration:.2f}s, {len(speaker_segments)} turns")

    print(f"\nAdded {added}, skipped {skipped} (already present).\n")

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
