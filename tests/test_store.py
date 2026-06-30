"""Tests for the Registry JSONL store."""
import pytest

from callqa.registry.schema import CallRecord
from callqa.registry.store import PrivacyRefusedError, Registry


def make_record(call_id="syn-001", domain="billing", source="kokoro-tts",
                privacy_notes="synthetic"):
    return CallRecord(
        call_id=call_id,
        audio_path=f"data/synthetic/{call_id}.wav",
        duration_seconds=10.0,
        domain=domain,
        source=source,
        privacy_notes=privacy_notes,
    )


def test_add_and_get(tmp_path):
    reg = Registry(tmp_path / "registry.jsonl")
    rec = make_record()
    reg.add(rec)
    assert reg.get("syn-001") == rec


def test_get_missing_returns_none(tmp_path):
    reg = Registry(tmp_path / "registry.jsonl")
    assert reg.get("nope") is None


def test_list_all(tmp_path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.add(make_record("a"))
    reg.add(make_record("b"))
    assert {r.call_id for r in reg.list_all()} == {"a", "b"}


def test_filter_by_domain(tmp_path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.add(make_record("a", domain="billing"))
    reg.add(make_record("b", domain="refund"))
    out = reg.filter(domain="refund")
    assert [r.call_id for r in out] == ["b"]


def test_filter_by_source(tmp_path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.add(make_record("a", source="kokoro-tts"))
    reg.add(make_record("b", source="harpervalley"))
    out = reg.filter(source="harpervalley")
    assert [r.call_id for r in out] == ["b"]


def test_save_and_reload_from_disk(tmp_path):
    path = tmp_path / "registry.jsonl"
    reg = Registry(path)
    reg.add(make_record("a"))
    reg.add(make_record("b"))

    reloaded = Registry(path)
    assert {r.call_id for r in reloaded.list_all()} == {"a", "b"}


def test_add_duplicate_id_raises(tmp_path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.add(make_record("a"))
    with pytest.raises(ValueError):
        reg.add(make_record("a"))


def test_refuses_empty_source(tmp_path):
    reg = Registry(tmp_path / "registry.jsonl")
    rec = make_record(source="")
    with pytest.raises(PrivacyRefusedError):
        reg.add(rec)


def test_refuses_private_privacy_notes(tmp_path):
    reg = Registry(tmp_path / "registry.jsonl")
    rec = make_record(privacy_notes="private")
    with pytest.raises(PrivacyRefusedError):
        reg.add(rec)


def test_refuses_unknown_privacy_notes(tmp_path):
    reg = Registry(tmp_path / "registry.jsonl")
    rec = make_record(privacy_notes="unknown")
    with pytest.raises(PrivacyRefusedError):
        reg.add(rec)


def test_refused_record_not_persisted(tmp_path):
    path = tmp_path / "registry.jsonl"
    reg = Registry(path)
    with pytest.raises(PrivacyRefusedError):
        reg.add(make_record(source=""))
    reloaded = Registry(path)
    assert reloaded.list_all() == []
