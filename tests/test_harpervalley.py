"""Tests for the HarperValley dataset module.

All of this runs on in-memory dicts and tiny numpy arrays. Nothing here touches
the network, so the suite stays fast and works offline.
"""
import numpy as np

from callqa.datasets.harpervalley import (
    gold_from_transcript,
    mix_channels,
    sample_call_ids,
)


def test_mix_pads_shorter_to_longer():
    agent = np.array([0.2, 0.2, 0.2], dtype=np.float32)
    caller = np.array([0.1, 0.1], dtype=np.float32)
    out = mix_channels(agent, caller)
    assert len(out) == 3


def test_mix_is_mono_float32():
    agent = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    caller = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = mix_channels(agent, caller)
    assert out.ndim == 1
    assert out.dtype == np.float32


def test_mix_stays_within_unit_range():
    # Two loud channels would sum past 1.0; the peak limiter must reel it in.
    agent = np.full(100, 0.9, dtype=np.float32)
    caller = np.full(100, 0.9, dtype=np.float32)
    out = mix_channels(agent, caller)
    assert float(np.max(np.abs(out))) <= 1.0


def test_mix_length_equals_longer_input():
    agent = np.zeros(5, dtype=np.float32)
    caller = np.zeros(9, dtype=np.float32)
    out = mix_channels(agent, caller)
    assert len(out) == 9


def _fake_segments():
    # Deliberately out of time order so the parser has to sort.
    return [
        {
            "speaker_role": "caller",
            "start_ms": 2000,
            "duration_ms": 1000,
            "human_transcript": "I have a question",
        },
        {
            "speaker_role": "agent",
            "start_ms": 0,
            "duration_ms": 1500,
            "human_transcript": "Hello how can I help",
        },
        {
            "speaker_role": "agent",
            "start_ms": 3500,
            "duration_ms": 500,
            "human_transcript": "Sure thing",
        },
    ]


def test_gold_transcript_joined_in_time_order():
    transcript, _segments = gold_from_transcript(_fake_segments())
    assert transcript == "Hello how can I help I have a question Sure thing"


def test_gold_segments_have_seconds_and_roles():
    _transcript, segments = gold_from_transcript(_fake_segments())
    # First in time order is the agent greeting at 0 ms for 1500 ms.
    assert segments[0].speaker == "agent"
    assert segments[0].start == 0.0
    assert segments[0].end == 1.5
    # The caller turn at 2000 ms for 1000 ms.
    assert segments[1].speaker == "caller"
    assert segments[1].start == 2.0
    assert segments[1].end == 3.0


def test_sample_same_seed_is_repeatable():
    ids = [f"sid{i}" for i in range(50)]
    a = sample_call_ids(ids, 10, seed=123)
    b = sample_call_ids(ids, 10, seed=123)
    assert a == b


def test_sample_different_seed_differs():
    ids = [f"sid{i}" for i in range(50)]
    a = sample_call_ids(ids, 10, seed=1)
    b = sample_call_ids(ids, 10, seed=2)
    assert a != b


def test_sample_returns_requested_count():
    ids = [f"sid{i}" for i in range(50)]
    out = sample_call_ids(ids, 7, seed=5)
    assert len(out) == 7
