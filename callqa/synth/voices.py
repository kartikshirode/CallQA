"""Kokoro voice config for the two speaker roles.

Voices are kept as plain data so they are easy to retune for the diarization
gate. Each named pair maps to one agent voice and one customer voice. The
"cross" pair mixes a male and a female voice, so diarization has an easy task.
The "female" and "male" pairs keep both speakers on the same gender, which is
harder and a better stress test for the later go/no-go check.
"""
from __future__ import annotations

# Pair name -> (agent voice, customer voice).
VOICE_PAIRS: dict[str, dict[str, str]] = {
    "cross": {"agent": "am_michael", "customer": "af_heart"},
    "female": {"agent": "af_heart", "customer": "af_sarah"},
    "male": {"agent": "am_michael", "customer": "am_adam"},
}

DEFAULT_PAIR = "cross"


def voices_for_pair(pair: str = DEFAULT_PAIR) -> dict[str, str]:
    """Return the agent and customer voices for a pair name.

    Raises KeyError with the known names if the pair is not defined.
    """
    if pair not in VOICE_PAIRS:
        known = ", ".join(sorted(VOICE_PAIRS))
        raise KeyError(f"unknown voice pair {pair!r}; known pairs: {known}")
    return dict(VOICE_PAIRS[pair])


def voice_for_speaker(speaker: str, pair: str = DEFAULT_PAIR) -> str:
    """Return the Kokoro voice for one speaker role under a pair."""
    pair_voices = voices_for_pair(pair)
    if speaker not in pair_voices:
        raise KeyError(f"speaker must be agent or customer, got {speaker!r}")
    return pair_voices[speaker]
