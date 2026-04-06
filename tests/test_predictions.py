"""Tests for the prediction pipeline and model-loading functions."""

import os
import pytest
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODELS_EXIST = os.path.isdir(MODEL_DIR) and os.path.isfile(
    os.path.join(MODEL_DIR, "metadata.joblib")
)


@pytest.mark.skipif(not MODELS_EXIST, reason="Trained models not found in models/")
class TestPredictReview:
    """Integration tests that require the trained model files."""

    @pytest.fixture(autouse=True)
    def _load_models(self):
        from utils.data import load_prediction_models

        meta_path = os.path.join(MODEL_DIR, "metadata.joblib")
        self.models = load_prediction_models(
            _model_mtime=os.path.getmtime(meta_path)
        )

    def test_positive_review_predicted_satisfied(self):
        from utils.data import predict_review

        result = predict_review(
            "Absolutely wonderful experience! The sales team was incredible, "
            "the build quality was flawless, and we love our new home.",
            self.models,
        )
        assert "Satisfied" in result["binary_label"] or "Positive" in result["three_label"]

    def test_negative_review_predicted_at_risk(self):
        from utils.data import predict_review

        result = predict_review(
            "Terrible quality. Warranty repairs took months and nobody returned "
            "our calls. Cracks in the drywall, paint peeling, appliances broken.",
            self.models,
        )
        assert "At-Risk" in result["binary_label"] or "Negative" in result["three_label"]

    def test_result_has_expected_keys(self):
        from utils.data import predict_review

        result = predict_review("A simple test review.", self.models)
        expected_keys = {
            "vader_compound",
            "vader_label",
            "binary_label",
            "binary_proba",
            "three_label",
            "three_proba",
            "signal_words",
        }
        assert expected_keys == set(result.keys())

    def test_vader_compound_in_range(self):
        from utils.data import predict_review

        result = predict_review("This is a test.", self.models)
        assert -1 <= result["vader_compound"] <= 1

    def test_binary_proba_sums_to_one(self):
        from utils.data import predict_review

        result = predict_review("Great home, loved it!", self.models)
        total = sum(result["binary_proba"].values())
        assert abs(total - 1.0) < 0.01

    def test_three_proba_sums_to_one(self):
        from utils.data import predict_review

        result = predict_review("Great home, loved it!", self.models)
        total = sum(result["three_proba"].values())
        assert abs(total - 1.0) < 0.01

    def test_signal_words_are_list(self):
        from utils.data import predict_review

        result = predict_review("The warranty process was slow and frustrating.", self.models)
        assert isinstance(result["signal_words"], list)

    def test_binary_label_valid(self):
        from utils.data import predict_review

        result = predict_review("Test review.", self.models)
        assert result["binary_label"] in [
            "Satisfied (4-5 Stars)",
            "At-Risk (1-3 Stars)",
        ]

    def test_three_label_valid(self):
        from utils.data import predict_review

        result = predict_review("Test review.", self.models)
        assert result["three_label"] in [
            "Positive (4-5 Stars)",
            "Neutral (3 Stars)",
            "Negative (1-2 Stars)",
        ]

    def test_vader_label_valid(self):
        from utils.data import predict_review

        result = predict_review("Test review.", self.models)
        assert result["vader_label"] in ["Positive", "Negative", "Neutral"]


@pytest.mark.skipif(not MODELS_EXIST, reason="Trained models not found in models/")
class TestBuildHybridFeatures:
    """Test the feature-building helper from train_models.py."""

    def test_output_shape(self):
        import pandas as pd
        from scipy.sparse import issparse
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = pd.Series(["great home", "terrible quality", "okay experience"])
        extras = pd.DataFrame({"vader": [0.5, -0.5, 0.0], "wc": [2, 2, 2], "exc": [0, 0, 0]})

        from train_models import build_hybrid_features

        tfidf = TfidfVectorizer(max_features=100)
        result = build_hybrid_features(texts, extras, tfidf, fit=True)

        assert issparse(result)
        assert result.shape[0] == 3  # 3 rows
        # columns = tfidf features + 3 extras
        assert result.shape[1] == len(tfidf.get_feature_names_out()) + 3

    def test_fit_vs_transform(self):
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from train_models import build_hybrid_features

        texts = pd.Series(["hello world", "foo bar baz"])
        extras = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        tfidf = TfidfVectorizer(max_features=50)

        # Fit mode
        result_fit = build_hybrid_features(texts, extras, tfidf, fit=True)
        # Transform mode (already fitted)
        result_transform = build_hybrid_features(texts, extras, tfidf, fit=False)

        assert result_fit.shape == result_transform.shape
