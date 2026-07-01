"""ASR provider interface and cloud stubs.

Mostly pure. The Protocol and the two cloud stubs need no GPU, no network, and
no keys to import or construct. The one heavy path is LocalWhisperProvider, and
even that imports whisper_adapter lazily inside transcribe, so building the
provider stays GPU-free; only calling transcribe touches the model.

This is the seam that lets Deepgram and AssemblyAI slot in during Week 6 without
reshaping the caller. Every provider returns a Transcript, so scoring code never
learns which backend produced it.
"""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from callqa.asr.transcript import Transcript


@runtime_checkable
class AsrProvider(Protocol):
    """The shared ASR contract. All providers, local or cloud, satisfy this.

    Signature: transcribe(self, wav_path, call_id, audio_seconds) -> Transcript.

    call_id is stamped onto the returned Transcript so the caller does not have
    to patch it afterward. audio_seconds is the caller's known audio length,
    passed in so cloud providers that do not report their own duration still
    produce an honest real_time_factor; a provider that measures its own length
    (faster-whisper via info.duration) may prefer that and treat this as a
    fallback. Providers that cannot serve a call raise, they do not return None.
    """

    name: str

    def transcribe(
        self, wav_path, call_id: str, audio_seconds: float
    ) -> Transcript: ...


class LocalWhisperProvider:
    """Local faster-whisper baseline. Wraps whisper_adapter for one model size.

    whisper_adapter is imported inside transcribe, not at construction, so
    building this provider needs no GPU. The size is chosen once at construction
    (default base.en, the Week 2 baseline)."""

    def __init__(self, size: str = "base.en") -> None:
        self.size = size
        self.name = f"local-whisper/{size}"

    def transcribe(
        self, wav_path, call_id: str, audio_seconds: float
    ) -> Transcript:
        """Transcribe with local faster-whisper. Runtime-checked at the GPU run.

        The adapter reports its own audio length via info.duration, so the
        passed audio_seconds is only a fallback used when the decoder reports 0.
        """
        from callqa.asr import whisper_adapter

        transcript = whisper_adapter.transcribe(
            wav_path, self.size, call_id=call_id
        )
        if transcript.audio_seconds == 0.0 and audio_seconds > 0.0:
            transcript = transcript.model_copy(
                update={"audio_seconds": float(audio_seconds)}
            )
        return transcript


class DeepgramProvider:
    """Cloud Deepgram stub. Arrives in Week 6, gated behind DEEPGRAM_API_KEY.

    The constructor may read the key from the environment but makes no network
    call and imports no HTTP client. transcribe raises so no dead cloud code
    ships early."""

    name = "deepgram"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")

    def transcribe(
        self, wav_path, call_id: str, audio_seconds: float
    ) -> Transcript:
        raise NotImplementedError(
            "DeepgramProvider arrives in Week 6 and is gated behind "
            "DEEPGRAM_API_KEY; no cloud call is made yet."
        )


class AssemblyAIProvider:
    """Cloud AssemblyAI stub. Arrives in Week 6, gated behind ASSEMBLYAI_API_KEY.

    Same shape as DeepgramProvider: reads the key if present, never touches the
    network, raises on transcribe."""

    name = "assemblyai"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")

    def transcribe(
        self, wav_path, call_id: str, audio_seconds: float
    ) -> Transcript:
        raise NotImplementedError(
            "AssemblyAIProvider arrives in Week 6 and is gated behind "
            "ASSEMBLYAI_API_KEY; no cloud call is made yet."
        )


def available_providers() -> dict[str, type]:
    """Map provider name to class. Local is usable now; the cloud two are stubs
    until Week 6. Kept small on purpose, just enough for the benchmark script to
    look one up by name."""
    return {
        "local-whisper": LocalWhisperProvider,
        "deepgram": DeepgramProvider,
        "assemblyai": AssemblyAIProvider,
    }
