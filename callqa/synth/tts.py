"""Kokoro text-to-speech wrapper.

All real Kokoro use lives here so the assembler stays pure and testable. The
pipeline is built once and cached, since rebuilding KPipeline per turn is slow.
Each call to synthesize_turn returns a single mono float32 waveform at 24000 Hz.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 24000

# Lazily built and cached so we only spin up Kokoro once per process.
_pipeline = None


def _get_pipeline():
    """Build the KPipeline on first use and reuse it afterwards."""
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline

        _pipeline = KPipeline(lang_code="a")
    return _pipeline


def _chunk_to_array(audio) -> np.ndarray:
    """Convert one Kokoro audio chunk to a float32 numpy array.

    Kokoro yields torch tensors; older builds may yield arrays. Handle both.
    """
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def synthesize_turn(text: str, voice: str) -> np.ndarray:
    """Synthesize one turn and return its waveform.

    Returns a 1-D float32 array at SAMPLE_RATE. The pipeline streams chunks per
    utterance; we concatenate them into one clip. Empty text yields an empty
    array so callers do not crash on a blank turn.
    """
    text = text.strip()
    if not text:
        return np.zeros(0, dtype=np.float32)

    pipeline = _get_pipeline()
    chunks: list[np.ndarray] = []
    for result in pipeline(text, voice=voice):
        # Kokoro returns tuples; the third item is the audio for the chunk.
        audio = result[2]
        chunk = _chunk_to_array(audio)
        if chunk.size:
            chunks.append(chunk)

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)
