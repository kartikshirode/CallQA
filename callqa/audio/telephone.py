"""Telephone-band degradation for synthetic calls.

Clean 24 kHz TTS audio is too easy for Whisper and pyannote, so the numbers
off it mean nothing. This module runs each call through the kind of damage a
real phone line does: drop to 8 kHz, keep only the 300 to 3400 Hz band, squash
it through mu-law like a codec would, then add a little line noise. Resampling
keeps the timeline intact, so the Phase 2 labels still line up.

The pipeline is split into small helpers so each step can be tested on its own.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

TELEPHONE_SR = 8000

# Passband edges for a narrowband phone channel.
_BAND_LOW_HZ = 300.0
_BAND_HIGH_HZ = 3400.0

# mu-law parameter for 8-bit companding.
_MU = 255.0


def resample_to_8k(waveform: np.ndarray, sample_rate_in: int) -> np.ndarray:
    """Resample to 8000 Hz, preserving duration."""
    x = np.asarray(waveform, dtype=np.float64)
    if sample_rate_in == TELEPHONE_SR:
        return x.astype(np.float32)
    # resample_poly works on the ratio of two integers and keeps length in
    # proportion, so a t second clip stays t seconds long.
    g = np.gcd(int(sample_rate_in), TELEPHONE_SR)
    up = TELEPHONE_SR // g
    down = int(sample_rate_in) // g
    out = resample_poly(x, up, down)
    return out.astype(np.float32)


def bandpass(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Keep roughly the 300 to 3400 Hz phone band.

    Uses a low-order Butterworth filter in second-order-section form, which
    stays numerically stable where a plain transfer-function design can blow up.
    The high edge is clamped just under Nyquist so the design never asks for an
    impossible cutoff.
    """
    x = np.asarray(waveform, dtype=np.float64)
    nyq = sample_rate / 2.0
    high = min(_BAND_HIGH_HZ, nyq * 0.99)
    low = min(_BAND_LOW_HZ, high * 0.5)
    sos = butter(4, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def mu_law_companding(waveform: np.ndarray) -> np.ndarray:
    """Encode to 8-bit mu-law and decode back, picking up quantization."""
    x = np.clip(np.asarray(waveform, dtype=np.float64), -1.0, 1.0)
    # Encode: compress, then quantize to 256 levels.
    compressed = np.sign(x) * np.log1p(_MU * np.abs(x)) / np.log1p(_MU)
    codes = np.round((compressed + 1.0) / 2.0 * 255.0)
    codes = np.clip(codes, 0, 255)
    # Decode: back to the compressed domain, then expand.
    q = codes / 255.0 * 2.0 - 1.0
    decoded = np.sign(q) * (1.0 / _MU) * (np.power(1.0 + _MU, np.abs(q)) - 1.0)
    return decoded.astype(np.float32)


def add_line_noise(
    waveform: np.ndarray,
    sample_rate: int,
    seed: int = 0,
    noise_level: float = 0.003,
    hum_level: float = 0.0015,
    hum_hz: float = 60.0,
) -> np.ndarray:
    """Add subtle seeded Gaussian noise plus a faint mains hum.

    Levels are deliberately small. The point is a believable noise floor, not
    wrecking the speech.
    """
    x = np.asarray(waveform, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_level, size=x.shape)
    t = np.arange(x.shape[0]) / sample_rate
    hum = hum_level * np.sin(2 * np.pi * hum_hz * t)
    return (x + noise + hum).astype(np.float32)


def soft_limit(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize into [-1, 1] so nothing clips."""
    x = np.asarray(waveform, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def degrade_to_telephone(
    waveform: np.ndarray,
    sample_rate_in: int,
    seed: int = 0,
) -> np.ndarray:
    """Run the full telephone-band chain and return float32 at 8000 Hz.

    Steps, in order: resample to 8 kHz, band-pass to 300-3400 Hz, mu-law
    compand, add seeded line noise, then peak-limit. Same seed gives identical
    output.
    """
    x = resample_to_8k(waveform, sample_rate_in)
    x = bandpass(x, TELEPHONE_SR)
    x = mu_law_companding(x)
    x = add_line_noise(x, TELEPHONE_SR, seed=seed)
    x = soft_limit(x)
    return x
