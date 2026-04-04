"""
Train and save predictive models for the Shea Homes review analysis.
Run this script to generate the models/ directory used by the Streamlit app.

Usage:
    python train_models.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.sparse import hstack, csr_matrix

nltk.download("vader_lexicon", quiet=True)

EXTRA_FEATURES = ["vader_compound", "word_count", "exclamation_count"]
# Keep apostrophes so contractions like "don't" stay as one token
TOKEN_PATTERN = r"(?u)\b\w[\w']+\b"


def build_hybrid_features(X_texts, X_extras, tfidf_model, fit=False):
    """Combine TF-IDF text features with numeric features into one matrix."""
    if fit:
        text_features = tfidf_model.fit_transform(X_texts)
    else:
        text_features = tfidf_model.transform(X_texts)
    numeric_features = csr_matrix(X_extras.values)
    return hstack([text_features, numeric_features])


def main():
    # Load data
    df = pd.read_csv("shea_homes_reviews.csv", encoding="utf-8-sig")
    print(f"Loaded {len(df)} reviews")

    # Feature engineering
    sia = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["review_text"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )
    df["word_count"] = df["review_text"].apply(lambda x: len(str(x).split()))
    df["exclamation_count"] = df["review_text"].apply(lambda x: str(x).count("!"))

    # ── Binary Classification: Satisfied vs At-Risk ──────────────────────
    df["risk_class"] = df["total_score"].apply(
        lambda x: "Satisfied (4-5 Stars)" if x >= 4 else "At-Risk (1-3 Stars)"
    )

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        df[["review_text"] + EXTRA_FEATURES],
        df["risk_class"],
        test_size=0.2,
        random_state=42,
        stratify=df["risk_class"],
    )

    tfidf_b = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english",
        token_pattern=TOKEN_PATTERN,
    )
    X_train_hyb = build_hybrid_features(
        X_train_b["review_text"].astype(str), X_train_b[EXTRA_FEATURES], tfidf_b, fit=True
    )
    X_test_hyb = build_hybrid_features(
        X_test_b["review_text"].astype(str), X_test_b[EXTRA_FEATURES], tfidf_b, fit=False
    )

    binary_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42, max_depth=5
        ),
    }

    print("\n── Binary Classification ──")
    for name, clf in binary_models.items():
        clf.fit(X_train_hyb, y_train_b)
        y_pred = clf.predict(X_test_hyb)
        acc = accuracy_score(y_test_b, y_pred)
        f1 = f1_score(y_test_b, y_pred, average="macro")
        print(f"  {name}: accuracy={acc:.3f}  macro-F1={f1:.3f}")

    # ── Three-Class Classification ───────────────────────────────────────
    def rating_bucket(score):
        if score <= 2:
            return "Negative (1-2 Stars)"
        elif score == 3:
            return "Neutral (3 Stars)"
        else:
            return "Positive (4-5 Stars)"

    df["rating_class"] = df["total_score"].apply(rating_bucket)

    X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
        df[["review_text"] + EXTRA_FEATURES],
        df["rating_class"],
        test_size=0.2,
        random_state=42,
        stratify=df["rating_class"],
    )

    tfidf_3 = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english",
        token_pattern=TOKEN_PATTERN,
    )
    X_train_3h = build_hybrid_features(
        X_train_3["review_text"].astype(str), X_train_3[EXTRA_FEATURES], tfidf_3, fit=True
    )
    X_test_3h = build_hybrid_features(
        X_test_3["review_text"].astype(str), X_test_3[EXTRA_FEATURES], tfidf_3, fit=False
    )

    three_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42, max_depth=5
        ),
    }

    print("\n── Three-Class Classification ──")
    for name, clf in three_models.items():
        clf.fit(X_train_3h, y_train_3)
        y_pred = clf.predict(X_test_3h)
        acc = accuracy_score(y_test_3, y_pred)
        f1 = f1_score(y_test_3, y_pred, average="macro")
        print(f"  {name}: accuracy={acc:.3f}  macro-F1={f1:.3f}")

    # ── Save everything ──────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    # Vectorizers
    joblib.dump(tfidf_b, "models/binary_tfidf.joblib")
    joblib.dump(tfidf_3, "models/three_class_tfidf.joblib")

    # Binary models
    for name, clf in binary_models.items():
        fname = name.lower().replace(" ", "_")
        joblib.dump(clf, f"models/binary_{fname}.joblib")

    # Three-class models
    for name, clf in three_models.items():
        fname = name.lower().replace(" ", "_")
        joblib.dump(clf, f"models/three_{fname}.joblib")

    # Metadata: test indices for reproducible evaluation
    joblib.dump(
        {
            "binary_test_idx": X_test_b.index.tolist(),
            "three_test_idx": X_test_3.index.tolist(),
            "extra_features": EXTRA_FEATURES,
        },
        "models/metadata.joblib",
    )

    print("\nModels saved to models/")


if __name__ == "__main__":
    main()
