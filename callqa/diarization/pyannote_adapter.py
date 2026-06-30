"""pyannote pipeline adapter.

All GPU, token and network use lives here, so the metrics module stays pure and
its tests need none of that. The module applies the torch.load patch at import
time because the official pyannote checkpoint will not load otherwise on
torch 2.6.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from callqa.registry.schema import SpeakerSegment

# torch 2.6 defaults torch.load to weights_only=True, which rejects the official
# pyannote checkpoint. Patch load to allow the full pickle for this trusted file.
_orig_load = torch.load


def _patched_load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)


torch.load = _patched_load

_PIPELINE = None
_PIPELINE_NAME = "pyannote/speaker-diarization-3.1"


def load_token() -> str:
    """Read HF_TOKEN from the environment or the repo .env. Never logged."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                return value.strip().strip('"').strip("'")
    raise RuntimeError(
        "HF_TOKEN not found in environment or .env; cannot load the pipeline"
    )


def load_pipeline():
    """Build the pyannote pipeline on cuda once and cache it."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    from pyannote.audio import Pipeline

    token = load_token()
    pipe = Pipeline.from_pretrained(_PIPELINE_NAME, use_auth_token=token)
    if pipe is None:
        raise RuntimeError(
            "pyannote returned no pipeline; check that the HF token has "
            f"accepted the gated terms for {_PIPELINE_NAME}"
        )
    pipe.to(torch.device("cuda"))
    _PIPELINE = pipe
    return _PIPELINE


def diarize(wav_path) -> list[SpeakerSegment]:
    """Run diarization on a wav and return SpeakerSegment turns."""
    pipe = load_pipeline()
    diar = pipe(str(Path(wav_path)))
    out: list[SpeakerSegment] = []
    for seg, _, label in diar.itertracks(yield_label=True):
        out.append(
            SpeakerSegment(speaker=str(label), start=float(seg.start), end=float(seg.end))
        )
    return out
