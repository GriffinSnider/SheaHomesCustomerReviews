"""Tests for utils/config.py — sanity-check that constants are consistent."""

from utils.config import (
    VADER_POS_THRESHOLD,
    VADER_NEG_THRESHOLD,
    TEXTBLOB_POS_THRESHOLD,
    TEXTBLOB_NEG_THRESHOLD,
    MISMATCH_NEG_SENTIMENT,
    MISMATCH_POS_SENTIMENT,
    SATISFIED_MIN_STARS,
    NEGATIVE_MAX_STARS,
    NEUTRAL_STARS,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    LR_MAX_ITER,
    N_ESTIMATORS,
    GB_MAX_DEPTH,
    LDA_N_TOPICS,
    LDA_MAX_FEATURES,
    LDA_MIN_DF,
    LDA_MAX_DF,
    LDA_MAX_ITER,
    MIN_REVIEWS_STATE,
    MIN_REVIEWS_COMMUNITY,
    MIN_EMPLOYEE_MENTIONS,
    DISTINCTIVE_MIN_RATIO,
    DISTINCTIVE_MIN_COUNT,
    PAGES,
    BUILDER_DISPLAY_NAMES,
    PALETTE_5,
)


# sentiment thresholds
class TestSentimentThresholds:
    def test_vader_positive_above_negative(self):
        assert VADER_POS_THRESHOLD > VADER_NEG_THRESHOLD

    def test_vader_thresholds_symmetric(self):
        assert VADER_POS_THRESHOLD == -VADER_NEG_THRESHOLD

    def test_textblob_mirrors_vader(self):
        assert TEXTBLOB_POS_THRESHOLD == VADER_POS_THRESHOLD
        assert TEXTBLOB_NEG_THRESHOLD == VADER_NEG_THRESHOLD

    def test_mismatch_neg_is_negative(self):
        assert MISMATCH_NEG_SENTIMENT < 0

    def test_mismatch_pos_is_positive(self):
        assert MISMATCH_POS_SENTIMENT > 0

    def test_mismatch_pos_stricter_than_vader(self):
        """The positive-text mismatch bar should be higher than the basic VADER
        positive threshold — weak positivity in a 1-star review is often sarcasm."""
        assert MISMATCH_POS_SENTIMENT > VADER_POS_THRESHOLD


# star rating buckets
class TestRatingBuckets:
    def test_bucket_ordering(self):
        assert NEGATIVE_MAX_STARS < NEUTRAL_STARS < SATISFIED_MIN_STARS

    def test_buckets_cover_1_to_5(self):
        for star in [1, 2]:
            assert star <= NEGATIVE_MAX_STARS
        assert NEUTRAL_STARS == 3
        for star in [4, 5]:
            assert star >= SATISFIED_MIN_STARS


# model hyperparameters
class TestModelHyperparams:
    def test_test_size_between_0_and_1(self):
        assert 0 < TEST_SIZE < 1

    def test_positive_estimators(self):
        assert N_ESTIMATORS > 0

    def test_positive_max_depth(self):
        assert GB_MAX_DEPTH > 0

    def test_lr_max_iter_enough(self):
        assert LR_MAX_ITER >= 1000

    def test_tfidf_features_positive(self):
        assert TFIDF_MAX_FEATURES > 0

    def test_ngram_range_tuple(self):
        assert isinstance(TFIDF_NGRAM_RANGE, tuple)
        assert TFIDF_NGRAM_RANGE[0] <= TFIDF_NGRAM_RANGE[1]


# LDA parameters
class TestLDAParams:
    def test_topics_positive(self):
        assert LDA_N_TOPICS > 0

    def test_max_df_above_min_df(self):
        # max_df is a proportion (0-1), min_df is an absolute count here
        assert 0 < LDA_MAX_DF <= 1

    def test_min_df_positive(self):
        assert LDA_MIN_DF >= 1


# Filtering thresholds
class TestFilteringThresholds:
    def test_min_reviews_state_positive(self):
        assert MIN_REVIEWS_STATE > 0

    def test_min_reviews_community_positive(self):
        assert MIN_REVIEWS_COMMUNITY > 0

    def test_state_stricter_than_community(self):
        assert MIN_REVIEWS_STATE >= MIN_REVIEWS_COMMUNITY

    def test_distinctive_ratio_above_one(self):
        assert DISTINCTIVE_MIN_RATIO > 1.0

    def test_distinctive_count_positive(self):
        assert DISTINCTIVE_MIN_COUNT > 0


# App configuration
class TestAppConfig:
    def test_pages_not_empty(self):
        assert len(PAGES) > 0

    def test_pages_unique(self):
        assert len(PAGES) == len(set(PAGES))

    def test_builder_names_has_shea(self):
        assert "shea-homes" in BUILDER_DISPLAY_NAMES

    def test_palette_has_5_colors(self):
        assert len(PALETTE_5) == 5
