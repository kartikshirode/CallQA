"""Synthesize and assemble the synthetic calls from their scripts.

Run it:
    python scripts/build_calls.py [--limit N] [--pair NAME] [--scripts DIR]

For each script in data/synthetic/scripts/ it synthesizes every turn with
Kokoro, assembles the timeline, and writes:
  - clean audio to data/synthetic/audio/clean/<call_id>.wav (24000 Hz)
  - a labels sidecar to data/synthetic/labels/<call_id>.json

The wav here is pre-telephone. Phase 3 degrades it to telephone band later. Use
--limit to build only the first few calls for a quick smoke test, and --pair to
pick the voice pair (default cross).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callqa.synth.assembler import assemble_call  # noqa: E402
from callqa.synth.script_schema import CallScript  # noqa: E402
from callqa.synth.tts import SAMPLE_RATE, synthesize_turn  # noqa: E402
from callqa.synth.voices import voice_for_speaker  # noqa: E402

DEFAULT_SCRIPTS = Path("data/synthetic/scripts")
DEFAULT_AUDIO = Path("data/synthetic/audio/clean")
DEFAULT_LABELS = Path("data/synthetic/labels")


def load_scripts(scripts_dir: Path) -> list[CallScript]:
    """Load every call script from a directory, skipping the manifest."""
    scripts: list[CallScript] = []
    for path in sorted(scripts_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        scripts.append(CallScript.model_validate_json(path.read_text(encoding="utf-8")))
    return scripts


def build_one(script: CallScript, pair: str):
    """Synthesize, assemble, and return the assembled call for one script."""
    clips = []
    for turn in script.turns:
        voice = voice_for_speaker(turn.speaker, pair)
        clips.append(synthesize_turn(turn.text, voice))
    return assemble_call(script, clips, sample_rate=SAMPLE_RATE)


def labels_payload(assembled, pair: str, domain: str) -> dict:
    """Build the labels sidecar dict from an assembled call."""
    return {
        "speaker_segments": [seg.model_dump() for seg in assembled.speaker_segments],
        "event_labels": [ev.model_dump(mode="json") for ev in assembled.event_labels],
        "duration_seconds": assembled.duration_seconds,
        "reference_transcript": assembled.reference_transcript,
        "voice_pair": pair,
        "domain": domain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic call audio + labels.")
    parser.add_argument("--limit", type=int, default=None, help="build only N calls")
    parser.add_argument("--pair", default="cross", help="voice pair name")
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--audio-out", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args()

    args.audio_out.mkdir(parents=True, exist_ok=True)
    args.labels_out.mkdir(parents=True, exist_ok=True)

    scripts = load_scripts(args.scripts)
    if args.limit is not None:
        scripts = scripts[: args.limit]

    print(f"Building {len(scripts)} call(s) with voice pair '{args.pair}'.")
    batch_start = time.perf_counter()

    for script in scripts:
        t0 = time.perf_counter()
        assembled = build_one(script, args.pair)

        wav_path = args.audio_out / f"{script.call_id}.wav"
        sf.write(wav_path, assembled.waveform, SAMPLE_RATE)

        labels_path = args.labels_out / f"{script.call_id}.json"
        payload = labels_payload(assembled, args.pair, script.domain.value)
        labels_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        took = time.perf_counter() - t0
        n_events = len(assembled.event_labels)
        print(
            f"  {script.call_id}: {assembled.duration_seconds:6.2f}s, "
            f"{len(assembled.speaker_segments)} turns, {n_events} events, "
            f"built in {took:5.1f}s -> {wav_path.name}"
        )

    total = time.perf_counter() - batch_start
    print(f"Done. {len(scripts)} call(s) in {total:.1f}s.")


if __name__ == "__main__":
    main()
