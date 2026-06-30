"""Week 1 dataset - telephone-band degradation.

Plain sine tones stand in for Kokoro and the real wavs, so the suite stays
fast and offline.
"""
import numpy as np

from callqa.audio.telephone import TELEPHONE_SR, degrade_to_telephone


def sine(freq, seconds, sample_rate, amplitude=0.5):
    """Build a mono sine tone as float32."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def test_output_is_8000_hz_and_preserves_duration():
    out = degrade_to_telephone(sine(1000, 1.0, 24000), 24000)
    assert TELEPHONE_SR == 8000
    # A 1 second tone should land near 8000 samples.
    assert abs(len(out) - 8000) <= 80
    longer = degrade_to_telephone(sine(440, 2.0, 24000), 24000)
    assert abs(len(longer) / TELEPHONE_SR - 2.0) <= 0.05


def test_bandpass_attenuates_5khz_relative_to_1khz():
    low = degrade_to_telephone(sine(1000, 1.0, 24000), 24000)
    high = degrade_to_telephone(sine(5000, 1.0, 24000), 24000)
    # 5 kHz sits outside the 300-3400 Hz passband, so it comes out much quieter.
    assert rms(high) < 0.5 * rms(low)


def test_same_seed_is_identical_different_seed_differs():
    src = sine(600, 1.0, 24000)
    assert np.array_equal(
        degrade_to_telephone(src, 24000, seed=7),
        degrade_to_telephone(src, 24000, seed=7),
    )
    assert not np.array_equal(
        degrade_to_telephone(src, 24000, seed=1),
        degrade_to_telephone(src, 24000, seed=2),
    )


def test_output_is_bounded_float32():
    # Push a loud input through and confirm the limiter keeps it in range.
    out = degrade_to_telephone(sine(800, 1.0, 24000, amplitude=0.99), 24000)
    assert out.dtype == np.float32
    assert float(np.max(np.abs(out))) <= 1.0
