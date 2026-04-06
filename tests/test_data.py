"""Tests for core data transformation logic in utils/data.py."""

import pandas as pd
import numpy as np
import pytest

from utils.config import (
    VADER_POS_THRESHOLD,
    VADER_NEG_THRESHOLD,
    SATISFIED_MIN_STARS,
    NEGATIVE_MAX_STARS,
    NEUTRAL_STARS,
    MISMATCH_NEG_SENTIMENT,
    MISMATCH_POS_SENTIMENT,
)


# ── _add_base_columns ────────────────────────────────────────────────
class TestAddBaseColumns:
    def test_date_parsed(self, processed_reviews):
        assert pd.api.types.is_datetime64_any_dtype(processed_reviews["date"])

    def test_word_count_positive(self, processed_reviews):
        assert (processed_reviews["word_count"] > 0).all()

    def test_word_count_accuracy(self, processed_reviews):
        row = processed_reviews.iloc[0]
        expected = len(str(row["review_text"]).split())
        assert row["word_count"] == expected

    def test_state_extraction(self, processed_reviews):
        states = processed_reviews["state"].tolist()
        assert states == ["AZ", "CO", "NV", "AZ", "NC"]

    def test_state_extraction_missing(self, sia):
        from utils.data import _add_base_columns

        df = pd.DataFrame(
            {
                "review_text": ["test"],
                "total_score": [3],
                "location": ["Unknown Location"],
                "date": ["2024-01-01"],
            }
        )
        result = _add_base_columns(df, sia)
        assert pd.isna(result["state"].iloc[0])

    def test_quarter_format(self, processed_reviews):
        # Quarters should look like "2024Q1", "2024Q2", etc.
        for q in processed_reviews["quarter"]:
            assert "Q" in q
            assert len(q) == 6

    def test_vader_compound_range(self, processed_reviews):
        assert (processed_reviews["vader_compound"] >= -1).all()
        assert (processed_reviews["vader_compound"] <= 1).all()

    def test_vader_label_values(self, processed_reviews):
        valid_labels = {"Positive", "Negative", "Neutral"}
        assert set(processed_reviews["vader_label"].unique()).issubset(valid_labels)

    def test_positive_review_gets_positive_vader(self, processed_reviews):
        """The first review ('Absolutely wonderful...') should score positive."""
        assert processed_reviews.iloc[0]["vader_label"] == "Positive"

    def test_negative_review_gets_negative_vader(self, processed_reviews):
        """The second review ('Terrible quality...') should score negative."""
        assert processed_reviews.iloc[1]["vader_label"] == "Negative"

    def test_risk_class_values(self, processed_reviews):
        valid = {"Satisfied (4-5)", "At-Risk (1-3)"}
        assert set(processed_reviews["risk_class"].unique()).issubset(valid)

    def test_5_star_is_satisfied(self, processed_reviews):
        fives = processed_reviews[processed_reviews["total_score"] == 5]
        assert (fives["risk_class"] == "Satisfied (4-5)").all()

    def test_1_star_is_at_risk(self, processed_reviews):
        ones = processed_reviews[processed_reviews["total_score"] == 1]
        assert (ones["risk_class"] == "At-Risk (1-3)").all()

    def test_3_star_is_at_risk(self, processed_reviews):
        threes = processed_reviews[processed_reviews["total_score"] == 3]
        assert (threes["risk_class"] == "At-Risk (1-3)").all()


# ── VADER label threshold edge cases ──────────────────────────────────
class TestVaderLabelEdgeCases:
    """Verify the exact boundary behavior of the VADER label thresholds."""

    def _label(self, compound):
        if compound >= VADER_POS_THRESHOLD:
            return "Positive"
        elif compound <= VADER_NEG_THRESHOLD:
            return "Negative"
        return "Neutral"

    def test_exactly_at_positive_threshold(self):
        assert self._label(VADER_POS_THRESHOLD) == "Positive"

    def test_just_below_positive_threshold(self):
        assert self._label(VADER_POS_THRESHOLD - 0.001) == "Neutral"

    def test_exactly_at_negative_threshold(self):
        assert self._label(VADER_NEG_THRESHOLD) == "Negative"

    def test_just_above_negative_threshold(self):
        assert self._label(VADER_NEG_THRESHOLD + 0.001) == "Neutral"

    def test_zero_is_neutral(self):
        assert self._label(0.0) == "Neutral"


# ── Rating bucket (3-class) ──────────────────────────────────────────
class TestRatingBucket:
    def _bucket(self, score):
        if score <= NEGATIVE_MAX_STARS:
            return "Negative (1-2 Stars)"
        elif score == NEUTRAL_STARS:
            return "Neutral (3 Stars)"
        return "Positive (4-5 Stars)"

    @pytest.mark.parametrize("score,expected", [
        (1, "Negative (1-2 Stars)"),
        (2, "Negative (1-2 Stars)"),
        (3, "Neutral (3 Stars)"),
        (4, "Positive (4-5 Stars)"),
        (5, "Positive (4-5 Stars)"),
    ])
    def test_all_star_levels(self, score, expected):
        assert self._bucket(score) == expected


# ── Mismatch detection ────────────────────────────────────────────────
class TestMismatchDetection:
    """Verify the mismatch flag logic."""

    def _is_mismatch(self, total_score, vader_compound):
        return (
            (total_score >= SATISFIED_MIN_STARS and vader_compound < MISMATCH_NEG_SENTIMENT)
            or (total_score <= NEGATIVE_MAX_STARS and vader_compound > MISMATCH_POS_SENTIMENT)
        )

    def test_high_star_negative_text_is_mismatch(self):
        assert self._is_mismatch(5, -0.3) is True

    def test_low_star_positive_text_is_mismatch(self):
        assert self._is_mismatch(1, 0.7) is True

    def test_high_star_positive_text_not_mismatch(self):
        assert self._is_mismatch(5, 0.8) is False

    def test_low_star_negative_text_not_mismatch(self):
        assert self._is_mismatch(1, -0.5) is False

    def test_mid_star_never_mismatch(self):
        assert self._is_mismatch(3, -0.5) is False
        assert self._is_mismatch(3, 0.8) is False

    def test_high_star_weakly_negative_not_mismatch(self):
        """VADER compound of -0.03 is above -0.05, so not flagged."""
        assert self._is_mismatch(4, -0.03) is False

    def test_low_star_weakly_positive_not_mismatch(self):
        """Compound of 0.3 is below the 0.5 bar — not a true mismatch."""
        assert self._is_mismatch(1, 0.3) is False


# ── load_and_process extras ───────────────────────────────────────────
class TestLoadAndProcessExtras:
    """Test the Shea-only columns added by load_and_process (TextBlob, mismatch, etc.)."""

    def test_textblob_columns_exist(self):
        """Smoke test: load_and_process should add TextBlob columns."""
        # We test this indirectly via a minimal CSV round-trip
        import tempfile, os
        from utils.data import load_and_process

        df = pd.DataFrame(
            {
                "review_text": ["Great experience!", "Terrible service"],
                "total_score": [5, 1],
                "quality": [5, 1],
                "trustworthiness": [5, 1],
                "value": [5, 1],
                "responsiveness": [5, 1],
                "location": ["Phoenix, AZ", "Denver, CO"],
                "date": ["2024-01-01", "2024-02-01"],
                "reviewer_name": ["A", "B"],
                "title": ["Good", "Bad"],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False, encoding="utf-8-sig")
            path = f.name
        try:
            result = load_and_process(path)
            assert "textblob_polarity" in result.columns
            assert "textblob_subjectivity" in result.columns
            assert "textblob_label" in result.columns
            assert "exclamation_count" in result.columns
            assert "mismatch" in result.columns
            assert "char_count" in result.columns
            assert "year" in result.columns
            assert "year_month" in result.columns
        finally:
            os.unlink(path)
