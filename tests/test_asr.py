"""Week 2 stage: ASR normalizer, WER/CER math, and the transcript object.

All offline. No GPU, no token, no network. The normalizer gets the widest
coverage because it decides whether real WER is honest. The metric and
transcript checks pin the math and the cache round-trip.
"""
import pytest
from pydantic import ValidationError

from callqa.asr.metrics import (
    AsrScore,
    aggregate_score,
    character_error_rate,
    word_error_rate,
)
from callqa.asr.normalize import MARKER_RE, normalize_text
from callqa.asr.transcript import Transcript, TranscriptSegment


class TestNormalizer:
    def test_square_bracket_markers_removed(self):
        assert normalize_text("yes [noise] please") == "yes please"

    def test_angle_bracket_markers_removed(self):
        assert normalize_text("hello <unk> world") == "hello world"

    def test_case_folded(self):
        assert normalize_text("HeLLo World") == "hello world"

    def test_punctuation_stripped(self):
        assert normalize_text("hello, world!") == "hello world"

    def test_whitespace_collapsed(self):
        assert normalize_text("  hello   there\tworld \n") == "hello there world"

    def test_none_returns_empty(self):
        assert normalize_text(None) == ""

    def test_empty_returns_empty(self):
        assert normalize_text("") == ""

    def test_marker_regex_is_compiled_and_matches(self):
        # Exposed so the marker behavior is documented and testable on its own.
        assert MARKER_RE.findall("[noise] mid <unk>") == ["[noise]", "<unk>"]

    def test_marker_content_does_not_leak(self):
        # Bracket contents are dropped whole, they never survive as bare words.
        assert "noise" not in normalize_text("[background noise here]").split()

    def test_realistic_harpervalley_line(self):
        # Markers dropped, contraction joined (that's -> thats), rest lowercased.
        line = "um [noise] yes that's right <unk>"
        assert normalize_text(line) == "um yes thats right"

    def test_contraction_apostrophe_joined_not_split(self):
        # Apostrophe drops with no space so a contraction stays one token, and
        # "thats" vs "that's" normalize the same so a style mismatch is not an
        # error. Covers the straight and curly apostrophe.
        assert normalize_text("don't") == "dont"
        assert normalize_text("that's") == "thats"
        assert normalize_text("that’s") == "thats"
        assert normalize_text("thats") == normalize_text("that's")


class TestWerCer:
    def test_perfect_match_is_zero(self):
        assert word_error_rate("the cat sat", "the cat sat") == 0.0

    def test_one_substitution(self):
        # One word wrong out of three.
        assert word_error_rate("the cat sat", "the cat ran") == pytest.approx(1 / 3)

    def test_cer_known_example(self):
        # One char wrong out of four.
        assert character_error_rate("abcd", "abce") == pytest.approx(0.25)

    def test_empty_reference_both_empty_is_zero(self):
        assert word_error_rate("", "") == 0.0
        assert character_error_rate("", "") == 0.0

    def test_empty_reference_with_hypothesis_is_one(self):
        assert word_error_rate("", "hello") == 1.0
        assert character_error_rate("", "hello") == 1.0

    def test_normalize_absorbs_case_and_punctuation(self):
        assert word_error_rate("Hello, World.", "hello world") == 0.0

    def test_normalize_off_keeps_the_difference(self):
        # Without normalize, casing and punctuation count as errors.
        assert word_error_rate("Hello", "hello", normalize=False) > 0.0


class TestAggregate:
    def test_pooled_score_shape_and_counts(self):
        pairs = [
            ("the cat sat", "the cat ran"),
            ("hello there", "hello there"),
        ]
        score = aggregate_score(pairs)
        assert isinstance(score, AsrScore)
        assert score.n_calls == 2
        # One wrong word pooled over five reference words.
        assert score.wer == pytest.approx(0.2)
        assert 0.0 <= score.cer <= 1.0

    def test_all_empty_references_score_zero(self):
        score = aggregate_score([("", ""), ("", "")])
        assert score.n_calls == 2
        assert score.wer == 0.0
        assert score.cer == 0.0


class TestTranscript:
    def _sample(self) -> Transcript:
        return Transcript(
            call_id="c1",
            model="faster-whisper/base.en",
            text="hi there",
            segments=[
                TranscriptSegment(start=0.0, end=1.0, text="hi"),
                TranscriptSegment(start=1.0, end=2.0, text="there"),
            ],
            latency_seconds=2.0,
            audio_seconds=10.0,
        )

    def test_json_round_trip_is_faithful(self):
        t = self._sample()
        assert Transcript.from_json(t.to_json()) == t

    def test_real_time_factor(self):
        assert self._sample().real_time_factor == pytest.approx(5.0)

    def test_real_time_factor_zero_latency(self):
        t = Transcript(
            call_id="c",
            model="m",
            text="",
            segments=[],
            latency_seconds=0.0,
            audio_seconds=5.0,
        )
        assert t.real_time_factor == 0.0

    def test_segment_end_before_start_raises(self):
        with pytest.raises(ValidationError):
            TranscriptSegment(start=2.0, end=1.0, text="x")
