"""Tests for telephone-band degradation.

These use plain sine tones, not Kokoro and not the real wavs, so the suite
stays fast and has no external dependencies.
"""
import numpy as np

from callqa.audio.telephone import TELEPHONE_SR, degrade_to_telephone


def sine(freq, seconds, sample_rate, amplitude=0.5):
    """Build a mono sine tone as float32."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def test_constant_is_8000():
    assert TELEPHONE_SR == 8000


def test_output_sample_rate_is_8000():
    src = sine(1000, 1.0, 24000)
    out = degrade_to_telephone(src, 24000)
    # The function returns the waveform; the contract is 8000 Hz output, so the
    # sample count for a 1 second tone should be about 8000.
    assert abs(len(out) - 8000) <= 80


def test_duration_preserved_within_tolerance():
    seconds = 2.0
    src = sine(440, seconds, 24000)
    out = degrade_to_telephone(src, 24000)
    out_seconds = len(out) / TELEPHONE_SR
    assert abs(out_seconds - seconds) <= 0.05


def test_bandpass_attenuates_5khz_relative_to_1khz():
    low = degrade_to_telephone(sine(1000, 1.0, 24000), 24000)
    high = degrade_to_telephone(sine(5000, 1.0, 24000), 24000)
    # A 5 kHz tone sits outside the 300-3400 Hz passband and should come out
    # much quieter than a 1 kHz tone that sits inside it.
    assert rms(high) < 0.5 * rms(low)


def test_same_seed_is_identical():
    src = sine(600, 1.0, 24000)
    a = degrade_to_telephone(src, 24000, seed=7)
    b = degrade_to_telephone(src, 24000, seed=7)
    assert np.array_equal(a, b)


def test_different_seed_differs():
    src = sine(600, 1.0, 24000)
    a = degrade_to_telephone(src, 24000, seed=1)
    b = degrade_to_telephone(src, 24000, seed=2)
    assert not np.array_equal(a, b)


def test_output_is_float32():
    out = degrade_to_telephone(sine(800, 0.5, 24000), 24000)
    assert out.dtype == np.float32


def test_output_within_unit_range():
    # Push a loud input through and confirm the limiter keeps it bounded.
    out = degrade_to_telephone(sine(800, 1.0, 24000, amplitude=0.99), 24000)
    assert float(np.max(np.abs(out))) <= 1.0
