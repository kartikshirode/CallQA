"""Push every clean synthetic call through telephone-band degradation.

Run it:
    python scripts/degrade_calls.py

For each wav in data/synthetic/audio/clean/ it resamples to 8 kHz, band-limits,
mu-law compands and adds light line noise, then writes the result to
data/synthetic/audio/telephone/<call_id>.wav. The seed is derived from the
call_id so a given call always degrades the same way.

Resampling preserves time, so the output duration should match the label
sidecar's duration_seconds. The script checks that for every call and flags any
gap larger than the tolerance.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import soundfile as sf

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callqa.audio.telephone import TELEPHONE_SR, degrade_to_telephone  # noqa: E402

CLEAN_DIR = Path("data/synthetic/audio/clean")
TELEPHONE_DIR = Path("data/synthetic/audio/telephone")
LABELS_DIR = Path("data/synthetic/labels")
DURATION_TOLERANCE = 0.05


def seed_for(call_id: str) -> int:
    """Stable per-call seed from the id, so reruns are reproducible."""
    return zlib.crc32(call_id.encode("utf-8"))


def label_duration(call_id: str) -> float | None:
    """Read duration_seconds from the call's label sidecar, if present."""
    path = LABELS_DIR / f"{call_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("duration_seconds")


def main() -> None:
    TELEPHONE_DIR.mkdir(parents=True, exist_ok=True)
    clean_wavs = sorted(CLEAN_DIR.glob("*.wav"))

    if not clean_wavs:
        print(f"No clean wavs found in {CLEAN_DIR}.")
        return

    print(f"Degrading {len(clean_wavs)} call(s) to telephone band.\n")
    header = f"{'call_id':<22} {'in (s)':>8} {'out (s)':>8} {'out SR':>7} {'match':>7}"
    print(header)
    print("-" * len(header))

    mismatches = []
    for wav_path in clean_wavs:
        call_id = wav_path.stem
        audio, sr_in = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        in_seconds = len(audio) / sr_in
        degraded = degrade_to_telephone(audio, sr_in, seed=seed_for(call_id))
        out_seconds = len(degraded) / TELEPHONE_SR

        out_path = TELEPHONE_DIR / f"{call_id}.wav"
        sf.write(out_path, degraded, TELEPHONE_SR)

        ref = label_duration(call_id)
        if ref is None:
            match = "no lbl"
        elif abs(out_seconds - ref) <= DURATION_TOLERANCE:
            match = "ok"
        else:
            match = "OFF"
            mismatches.append((call_id, ref, out_seconds))

        print(
            f"{call_id:<22} {in_seconds:>8.2f} {out_seconds:>8.2f} "
            f"{TELEPHONE_SR:>7} {match:>7}"
        )

    print()
    if mismatches:
        print(f"WARNING: {len(mismatches)} call(s) outside {DURATION_TOLERANCE}s tolerance:")
        for call_id, ref, out_seconds in mismatches:
            print(f"  {call_id}: label {ref:.3f}s vs out {out_seconds:.3f}s")
    else:
        print(f"All durations within {DURATION_TOLERANCE}s of their labels.")


if __name__ == "__main__":
    main()
