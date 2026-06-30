"""Environment smoke test for CallQA.

Checks which pipeline dependencies are actually installed and prints a table.
Nothing here imports the heavy GPU stack at module load; every probe is wrapped
so a missing package never crashes the script. Run it any time to see what
still needs installing before the synthesis and ASR phases.

Usage:
    python scripts/env_check.py
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class Probe:
    label: str
    module: str
    phase: str  # which phase this unblocks


# Order roughly follows the pipeline: core, then ASR, diarization, TTS, dashboard.
PROBES = [
    Probe("pydantic", "pydantic", "core"),
    Probe("numpy", "numpy", "core"),
    Probe("soundfile", "soundfile", "core"),
    Probe("librosa", "librosa", "core"),
    Probe("pydub", "pydub", "core"),
    Probe("faster-whisper", "faster_whisper", "ASR"),
    Probe("jiwer", "jiwer", "ASR"),
    Probe("torch", "torch", "diarization"),
    Probe("pyannote.audio", "pyannote.audio", "diarization"),
    Probe("kokoro", "kokoro", "TTS"),
    Probe("streamlit", "streamlit", "dashboard"),
]


def _version(mod) -> str:
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if v is not None:
            return str(v)
    return "unknown"


def check_one(probe: Probe) -> tuple[str, str, str]:
    """Return (status, version, extra) for a single probe."""
    try:
        mod = importlib.import_module(probe.module)
    except Exception as exc:  # noqa: BLE001 - any import error means missing
        return ("MISSING", "-", type(exc).__name__)

    version = _version(mod)
    extra = ""
    if probe.module == "torch":
        extra = _torch_extra(mod)
    return ("INSTALLED", version, extra)


def _torch_extra(torch_mod) -> str:
    """CUDA availability and device name, guarded so a broken build is safe."""
    try:
        if torch_mod.cuda.is_available():
            name = torch_mod.cuda.get_device_name(0)
            return f"CUDA on: {name}"
        return "CUDA off (CPU only)"
    except Exception as exc:  # noqa: BLE001
        return f"CUDA check failed: {type(exc).__name__}"


def main() -> int:
    results = [(p, *check_one(p)) for p in PROBES]

    label_w = max(len(p.label) for p in PROBES)
    print()
    print(f"{'PACKAGE':<{label_w}}  {'STATUS':<9}  {'VERSION':<12}  NOTES")
    print("-" * (label_w + 40))
    for probe, status, version, extra in results:
        print(f"{probe.label:<{label_w}}  {status:<9}  {version:<12}  {extra}")

    missing = [probe for probe, status, _, _ in results if status == "MISSING"]
    print()
    if not missing:
        print("All checked packages are installed.")
    else:
        print(f"{len(missing)} package(s) still missing:")
        by_phase: dict[str, list[str]] = {}
        for probe in missing:
            by_phase.setdefault(probe.phase, []).append(probe.label)
        for phase, names in by_phase.items():
            print(f"  {phase}: {', '.join(names)}")
        print()
        print("Install the heavy GPU stack (torch, pyannote.audio, kokoro,")
        print("faster-whisper) before the synthesis and ASR phases. Match the")
        print("torch wheel to your CUDA version; see https://pytorch.org.")

    # Always exit 0; this is a report, not a gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
