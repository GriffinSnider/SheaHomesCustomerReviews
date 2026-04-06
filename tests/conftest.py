"""Shared fixtures for the shea-reviews test suite."""

import pytest
import pandas as pd
import nltk

nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer


@pytest.fixture(scope="session")
def sia():
    """VADER sentiment analyser, shared across the entire test session."""
    return SentimentIntensityAnalyzer()


@pytest.fixture()
def sample_reviews():
    """Minimal DataFrame that looks like a raw builder CSV."""
    return pd.DataFrame(
        {
            "review_text": [
                "Absolutely wonderful experience! The team was incredible and professional.",
                "Terrible quality. Warranty repairs took months. Nobody returned our calls.",
                "The home is okay. Some issues but overall fine.",
                "Great sales team. Amazing floor plan and design upgrades.",
                "Worst experience ever. Construction defects everywhere, paint peeling, cracks in drywall.",
            ],
            "total_score": [5, 1, 3, 5, 1],
            "quality": [5, 1, 3, 4, 1],
            "trustworthiness": [5, 1, 3, 5, 1],
            "value": [5, 1, 3, 4, 1],
            "responsiveness": [5, 1, 3, 5, 1],
            "location": [
                "Phoenix, AZ",
                "Denver, CO",
                "Las Vegas, NV",
                "San Tan Valley, AZ",
                "Raleigh, NC",
            ],
            "date": [
                "2024-01-15",
                "2024-02-20",
                "2024-03-10",
                "2024-04-05",
                "2024-05-12",
            ],
            "reviewer_name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "title": ["Love it", "Hate it", "It's ok", "Fantastic", "Awful"],
        }
    )


@pytest.fixture()
def processed_reviews(sample_reviews, sia):
    """sample_reviews after running through _add_base_columns."""
    from utils.data import _add_base_columns

    return _add_base_columns(sample_reviews.copy(), sia)
