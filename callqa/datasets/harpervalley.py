"""HarperValley Bank corpus loader.

This is the real WER/DER anchor. The audio is already 8 kHz telephone, so we do
not degrade it; we only mix the two channel wavs into one mono file and pull the
gold transcript and speaker turns out of the JSON.

The corpus lives in a public GitHub repo, no token needed. Per call there are
four files: an agent channel wav, a caller channel wav, a metadata json and a
transcript json. The transcript json is a list of segment dicts.

The pure logic (channel mixing, transcript parsing, seeded sampling) is split
out from the network helpers so it can be unit-tested offline.
"""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from callqa.registry.schema import SpeakerSegment

# Public raw-file base for the HarperValley repo. No auth required.
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/cricketclub/"
    "gridspace-stanford-harper-valley/master/data"
)

# Git trees API lists every path in one request.
TREES_API_URL = (
    "https://api.github.com/repos/cricketclub/"
    "gridspace-stanford-harper-valley/git/trees/master?recursive=1"
)

# Telephone audio is already at this rate.
SAMPLE_RATE = 8000

# Prefix of the transcript paths we read sids from.
_TRANSCRIPT_PREFIX = "data/transcript/"
_TRANSCRIPT_SUFFIX = ".json"


# --- pure logic -------------------------------------------------------------


def _is_number(value) -> bool:
    """True for a real int or float timing, excluding bool (a bool is an int in
    Python, but a True start_ms is not a valid timing)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def mix_channels(agent_wave: np.ndarray, caller_wave: np.ndarray) -> np.ndarray:
    """Mix the agent and caller channels into one mono track.

    The two channel wavs can differ in length, so the shorter one is padded
    with zeros up to the longer. After summing, a peak limiter scales the
    result back into [-1, 1] so nothing clips. Returns float32.
    """
    a = np.asarray(agent_wave, dtype=np.float64).reshape(-1)
    c = np.asarray(caller_wave, dtype=np.float64).reshape(-1)
    n = max(a.shape[0], c.shape[0])
    if a.shape[0] < n:
        a = np.pad(a, (0, n - a.shape[0]))
    if c.shape[0] < n:
        c = np.pad(c, (0, n - c.shape[0]))
    mixed = a + c
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def gold_from_transcript(
    segments: list[dict],
) -> tuple[str, list[SpeakerSegment]]:
    """Build the reference transcript and speaker turns from segment dicts.

    Segments are ordered by start_ms first. The transcript is the human gold
    text joined with spaces in that order. Each speaker turn carries the role
    and start/end in seconds (start_ms and start_ms+duration_ms over 1000).
    """
    # Skip malformed segments rather than let one bad row kill a whole fetch
    # run. A segment needs a role and timing that is a real, non-negative number
    # to become a speaker turn: a string-typed timing would crash the ordering
    # or the addition below, and a negative duration would build a reversed span
    # the SpeakerSegment validator rejects. Either would abort the whole fetch,
    # which is exactly what the skip is meant to prevent.
    usable = [
        s
        for s in segments
        if isinstance(s, dict)
        and _is_number(s.get("start_ms"))
        and _is_number(s.get("duration_ms"))
        and s["start_ms"] >= 0
        and s["duration_ms"] >= 0
        and s.get("speaker_role")
    ]
    ordered = sorted(usable, key=lambda s: s["start_ms"])
    texts = []
    speaker_segments = []
    for seg in ordered:
        text = (seg.get("human_transcript") or "").strip()
        if text:
            texts.append(text)
        start_ms = seg["start_ms"]
        end_ms = start_ms + seg["duration_ms"]
        speaker_segments.append(
            SpeakerSegment(
                speaker=seg["speaker_role"],
                start=start_ms / 1000.0,
                end=end_ms / 1000.0,
            )
        )
    return " ".join(texts), speaker_segments


def sample_call_ids(all_ids: list[str], n: int, seed: int) -> list[str]:
    """Pick n sids deterministically. Same seed gives the same sids.

    The full list is sorted first so the order fed to the sampler is stable
    no matter what order the API returned things in.
    """
    pool = sorted(all_ids)
    n = min(n, len(pool))
    rng = random.Random(seed)
    return rng.sample(pool, n)


# --- network helpers (kept thin) --------------------------------------------


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def list_call_ids() -> list[str]:
    """Fetch the git trees once and return every sid under data/transcript."""
    payload = json.loads(_http_get(TREES_API_URL))
    ids = []
    for entry in payload.get("tree", []):
        path = entry.get("path", "")
        if path.startswith(_TRANSCRIPT_PREFIX) and path.endswith(
            _TRANSCRIPT_SUFFIX
        ):
            sid = path[len(_TRANSCRIPT_PREFIX) : -len(_TRANSCRIPT_SUFFIX)]
            if sid:
                ids.append(sid)
    return ids


def fetch_call(sid: str, raw_dir: Path) -> Path:
    """Download the four files for one sid into raw_dir/<sid>/.

    Files already on disk are left alone, so reruns are cheap. Returns the
    per-sid directory.
    """
    raw_dir = Path(raw_dir)
    dest = raw_dir / sid
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {
        "agent.wav": f"{RAW_BASE_URL}/audio/agent/{sid}.wav",
        "caller.wav": f"{RAW_BASE_URL}/audio/caller/{sid}.wav",
        "metadata.json": f"{RAW_BASE_URL}/metadata/{sid}.json",
        "transcript.json": f"{RAW_BASE_URL}/transcript/{sid}.json",
    }
    for name, url in wanted.items():
        target = dest / name
        if target.exists():
            continue
        target.write_bytes(_http_get(url))
    return dest


def build_mono_call(sid: str, raw_dir: Path, out_dir: Path) -> Path:
    """Load the two channel wavs, mix them, write a mono wav at 8 kHz.

    Thin wrapper over mix_channels and soundfile. Returns the output path.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_wave, _ = sf.read(str(raw_dir / sid / "agent.wav"), dtype="float32")
    caller_wave, _ = sf.read(str(raw_dir / sid / "caller.wav"), dtype="float32")
    mono = mix_channels(agent_wave, caller_wave)
    out_path = out_dir / f"{sid}.wav"
    sf.write(str(out_path), mono, SAMPLE_RATE, subtype="PCM_16")
    return out_path
