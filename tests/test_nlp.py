"""Tests for NLP helper functions: n-grams, distinctive words, topic naming."""

import pandas as pd
import numpy as np
import pytest

from utils.data import compute_ngrams, get_neg_distinctive, get_stop_words


# ── compute_ngrams ────────────────────────────────────────────────────
class TestComputeNgrams:
    def test_basic_bigrams(self):
        texts = pd.Series(["great home experience", "great home quality"])
        result = compute_ngrams(texts, sw_list=[], n=2, top_k=5)
        # "great home" appears in both texts
        words = [w for w, _ in result]
        assert "great home" in words

    def test_stop_words_removed(self):
        texts = pd.Series(["the great home", "the great team"])
        result = compute_ngrams(texts, sw_list=["the"], n=2, top_k=5)
        words = [w for w, _ in result]
        # "the" should be stripped, so no bigrams starting with "the"
        assert all("the" not in w.split() for w in words)

    def test_short_words_filtered(self):
        """Words with len <= 2 should be excluded."""
        texts = pd.Series(["I am so very happy with my new home"])
        result = compute_ngrams(texts, sw_list=[], n=1, top_k=10)
        words = [w for w, _ in result]
        assert "so" not in words  # len 2
        assert "am" not in words  # len 2

    def test_top_k_limits_results(self):
        texts = pd.Series(["word1 word2 word3 word4 word5 word6"] * 10)
        result = compute_ngrams(texts, sw_list=[], n=1, top_k=3)
        assert len(result) <= 3

    def test_returns_tuples_of_word_and_count(self):
        texts = pd.Series(["hello world hello world"])
        result = compute_ngrams(texts, sw_list=[], n=2, top_k=5)
        assert len(result) > 0
        word, count = result[0]
        assert isinstance(word, str)
        assert isinstance(count, int)

    def test_trigrams(self):
        texts = pd.Series(["one two three four five"] * 5)
        result = compute_ngrams(texts, sw_list=[], n=3, top_k=5)
        assert len(result) > 0
        assert len(result[0][0].split()) == 3

    def test_preserves_apostrophes(self):
        texts = pd.Series(["don't worry about it", "don't stop now"])
        result = compute_ngrams(texts, sw_list=[], n=1, top_k=5)
        words = [w for w, _ in result]
        assert "don't" in words

    def test_empty_input(self):
        texts = pd.Series([], dtype=str)
        result = compute_ngrams(texts, sw_list=[], n=2, top_k=5)
        assert result == []


# ── get_neg_distinctive ───────────────────────────────────────────────
class TestGetNegDistinctive:
    def test_finds_overrepresented_word(self):
        # "warranty" appears heavily in negative, rarely in positive
        neg = pd.Series(["warranty issue"] * 20 + ["other problem"] * 10)
        pos = pd.Series(["great experience"] * 50 + ["warranty fine"] * 2)
        result = get_neg_distinctive(neg, pos, sw_list=[])
        words = [w for w, _, _ in result]
        assert "warranty" in words

    def test_respects_min_count(self):
        """Words below DISTINCTIVE_MIN_COUNT should be excluded."""
        from utils.config import DISTINCTIVE_MIN_COUNT

        neg = pd.Series(["rare_word problem"] * (DISTINCTIVE_MIN_COUNT - 1))
        pos = pd.Series(["great experience"] * 50)
        result = get_neg_distinctive(neg, pos, sw_list=[])
        words = [w for w, _, _ in result]
        assert "rare_word" not in words

    def test_respects_min_ratio(self):
        """Words that appear equally in neg and pos should be excluded."""
        neg = pd.Series(["common word"] * 30)
        pos = pd.Series(["common word"] * 30)
        result = get_neg_distinctive(neg, pos, sw_list=[])
        words = [w for w, _, _ in result]
        assert "common" not in words

    def test_returns_sorted_by_ratio(self):
        neg = pd.Series(
            ["warranty issue"] * 20 + ["defect problem"] * 30
        )
        pos = pd.Series(["great experience"] * 50)
        result = get_neg_distinctive(neg, pos, sw_list=[])
        if len(result) >= 2:
            ratios = [r for _, _, r in result]
            assert ratios == sorted(ratios, reverse=True)

    def test_max_20_results(self):
        # Generate many distinctive words
        neg = pd.Series([f"word{i} problem" for i in range(50)] * 20)
        pos = pd.Series(["great experience"] * 100)
        result = get_neg_distinctive(neg, pos, sw_list=[])
        assert len(result) <= 20

    def test_stop_words_excluded(self):
        neg = pd.Series(["the issue problem"] * 20)
        pos = pd.Series(["great experience"] * 50)
        result = get_neg_distinctive(neg, pos, sw_list=["the"])
        words = [w for w, _, _ in result]
        assert "the" not in words


# ── get_stop_words ────────────────────────────────────────────────────
class TestGetStopWords:
    def test_returns_set(self):
        sw = get_stop_words()
        assert isinstance(sw, set)

    def test_contains_nltk_defaults(self):
        sw = get_stop_words()
        assert "the" in sw
        assert "and" in sw
        assert "is" in sw

    def test_contains_custom_words(self):
        sw = get_stop_words()
        assert "shea" in sw
        assert "homes" in sw
        assert "home" in sw
