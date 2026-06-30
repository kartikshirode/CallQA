"""Tests for the domain utterance banks."""
from callqa.synth.banks import BANKS, REQUIRED_SLOTS


def test_four_domains_present():
    assert set(BANKS) == {"billing", "refund", "cancellation", "tech_support"}


def test_every_bank_has_every_slot_with_alternatives():
    for domain, bank in BANKS.items():
        for slot in REQUIRED_SLOTS:
            assert slot in bank, f"{domain} missing slot {slot}"
            assert len(bank[slot]) >= 2, f"{domain}.{slot} needs alternatives"


def test_lines_are_non_empty_strings():
    for bank in BANKS.values():
        for lines in bank.values():
            for line in lines:
                assert isinstance(line, str) and line.strip()


def test_no_em_or_en_dash_in_lines():
    for bank in BANKS.values():
        for lines in bank.values():
            for line in lines:
                assert "—" not in line
                assert "–" not in line
