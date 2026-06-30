"""Generate the 20 MVP synthetic call scripts and write them to disk.

Run it:
    python scripts/gen_scripts.py [--seed N] [--out DIR]

It writes one pretty JSON file per call to data/synthetic/scripts/, plus a
manifest.json listing every call with its domain and which event types are
positive or negative. It also prints a negative-ratio table so you can confirm
each event sits near 40% negative across the batch.

No audio is produced here. That is Phase 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callqa.registry.schema import EventType  # noqa: E402
from callqa.synth.generator import (  # noqa: E402
    batch_event_summary,
    generate_batch,
)

DEFAULT_SEED = 2026
DEFAULT_OUT = Path("data/synthetic/scripts")


def _event_flags_for_call(script) -> dict[str, str]:
    """Map each event type to 'positive' or 'negative' for one call."""
    present = script.present_event_types()
    flags = {}
    for event_type in (
        EventType.silence,
        EventType.interruption,
        EventType.escalation,
        EventType.compliance,
    ):
        flags[event_type.value] = "positive" if event_type in present else "negative"
    return flags


def write_batch(seed: int, out_dir: Path) -> tuple[dict, dict]:
    """Generate, write per-call JSON and a manifest, return (manifest, summary)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = generate_batch(seed=seed)

    manifest_calls = []
    for script in batch:
        path = out_dir / f"{script.call_id}.json"
        path.write_text(
            script.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_calls.append(
            {
                "call_id": script.call_id,
                "domain": script.domain.value,
                "events": _event_flags_for_call(script),
                "notes": script.notes,
            }
        )

    summary = batch_event_summary(batch)
    manifest = {
        "seed": seed,
        "count": len(batch),
        "calls": manifest_calls,
        "negative_ratio": {
            et.value: round(stats["negative_ratio"], 3)
            for et, stats in summary.items()
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest, summary


def print_summary(summary: dict, count: int) -> None:
    """Print the negative-ratio table to stdout."""
    print()
    print(f"Negative-class ratio across {count} calls (target ~40%)")
    print("-" * 52)
    print(f"{'event':<14}{'positive':>10}{'negative':>10}{'neg %':>10}")
    print("-" * 52)
    for event_type, stats in summary.items():
        pct = stats["negative_ratio"] * 100
        print(
            f"{event_type.value:<14}"
            f"{int(stats['positive']):>10}"
            f"{int(stats['negative']):>10}"
            f"{pct:>9.0f}%"
        )
    print("-" * 52)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic call scripts.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    manifest, summary = write_batch(args.seed, args.out)
    print(f"Wrote {manifest['count']} scripts to {args.out}")
    print(f"Wrote manifest to {args.out / 'manifest.json'}")
    print_summary(summary, manifest["count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
