"""faster-whisper transcription adapter.

Heavy module: GPU and faster-whisper live here so the pure ASR core stays
offline. faster_whisper is imported lazily inside load_model, so importing this
module is cheap and needs no GPU or model download. Each model size loads on
cuda once and is cached, same pattern as the pyannote adapter.

Whisper checkpoints are not gated the way the pyannote pipeline is, so there is
no token to read and no torch.load pickle override here. On torch 2.6 the cu124
build must stay in place; if a dep drags torch to CPU, cuda goes False and the
transcribe call fails at load time.
"""
from __future__ import annotations

import time
from pathlib import Path

from callqa.asr.transcript import Transcript, TranscriptSegment

# Cache keyed by model size string, so "tiny.en" / "base.en" / "small.en" each
# build their WhisperModel once and reuse it across calls.
_MODELS: dict[str, object] = {}

# The three sizes the Week 2 sweep runs. Kept here as the documented default set;
# load_model accepts any size faster-whisper knows, this is just the sweep list.
DEFAULT_SIZES = ("tiny.en", "base.en", "small.en")


def load_model(size: str):
    """Build a faster-whisper WhisperModel on cuda once and cache it by size.

    float16 on cuda is the Week 2 config for the 4060. faster_whisper is
    imported here, not at module top, so this module imports with no GPU.
    """
    cached = _MODELS.get(size)
    if cached is not None:
        return cached
    from faster_whisper import WhisperModel

    model = WhisperModel(size, device="cuda", compute_type="float16")
    _MODELS[size] = model
    return model


def transcribe(wav_path, size: str, *, call_id: str = "") -> Transcript:
    """Transcribe one wav with the given model size and return a Transcript.

    Wall-clock latency is measured with perf_counter around the actual decode.
    faster-whisper's model.transcribe returns (segments, info) where segments is
    a generator, so iterating it is what forces the decode; timing wraps that
    loop. audio_seconds comes from info.duration, the decoder's own reported
    audio length. The live decode path here is runtime-checked at the Phase 3 GPU
    run, since this sandbox has no GPU.
    """
    model = load_model(size)
    path = str(Path(wav_path))

    start = time.perf_counter()
    segments_gen, info = model.transcribe(path)
    segments: list[TranscriptSegment] = []
    for seg in segments_gen:  # iterating the generator forces the decode
        segments.append(
            TranscriptSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=str(seg.text).strip(),
            )
        )
    latency_seconds = time.perf_counter() - start

    text = " ".join(s.text for s in segments).strip()
    # info.duration is faster-whisper's reported length of the decoded audio.
    audio_seconds = float(getattr(info, "duration", 0.0) or 0.0)

    return Transcript(
        call_id=call_id,
        model=f"faster-whisper/{size}",
        text=text,
        segments=segments,
        latency_seconds=latency_seconds,
        audio_seconds=audio_seconds,
    )
