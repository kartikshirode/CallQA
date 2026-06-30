"""Tests for the batch negative-ratio summary helper."""
from callqa.registry.schema import EventType
from callqa.synth.generator import batch_event_summary, generate_batch


def test_summary_lists_all_four_event_types():
    batch = generate_batch(seed=2026)
    summary = batch_event_summary(batch)
    assert set(summary) == {
        EventType.silence,
        EventType.interruption,
        EventType.escalation,
        EventType.compliance,
    }


def test_summary_counts_add_up_to_batch_size():
    batch = generate_batch(seed=2026)
    summary = batch_event_summary(batch)
    for stats in summary.values():
        assert stats["positive"] + stats["negative"] == len(batch)


def test_summary_negative_ratio_near_forty():
    batch = generate_batch(seed=2026)
    summary = batch_event_summary(batch)
    for stats in summary.values():
        assert 0.30 <= stats["negative_ratio"] <= 0.50
