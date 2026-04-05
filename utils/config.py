import streamlit as st

# Sentiment thresholds
# VADER's recommended cutoffs per Hutto & Gilbert (2014).  Compound scores
# above +0.05 are "positive", below -0.05 are "negative", between is "neutral".
# See: https://github.com/cjhutto/vaderSentiment#about-the-scoring
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

# TextBlob uses the same sign convention; we mirror VADER's cutoffs for
# consistency so the two models are directly comparable.
TEXTBLOB_POS_THRESHOLD = 0.05
TEXTBLOB_NEG_THRESHOLD = -0.05

# Mismatch detection: a 4-5 star review with clearly negative language,
# or a 1-2 star review with strongly positive language.  The 0.5 bar for
# "positive text + low stars" is intentionally stricter / weak positivity
# in a 1-star review is often sarcasm, not a true mismatch.
MISMATCH_NEG_SENTIMENT = -0.05
MISMATCH_POS_SENTIMENT = 0.5

# Star-rating buckets
# Industry-standard NPS-style grouping: 4-5 = satisfied, 1-3 = at-risk.
SATISFIED_MIN_STARS = 4       # >= this -> "Satisfied"
NEGATIVE_MAX_STARS = 2        # <= this -> "Negative" in 3-class
NEUTRAL_STARS = 3             # exactly this -> "Neutral" in 3-class

# Model hyperparameters
# Used by train_models.py.  Changing these requires retraining.
RANDOM_STATE = 42             # reproducibility seed used everywhere
TEST_SIZE = 0.2               # 80/20 train-test split (sklearn convention)

# TF-IDF: 5000 features captures the most discriminative unigrams and
# bigrams without over-fitting.  Empirically tuned — 3000 lost recall on
# minority classes, 10000 added noise with no accuracy gain.
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)    # unigrams + bigrams

# Logistic Regression: 2000 max_iter because the default 100 does not
# converge on this dataset's 5000-feature sparse matrix.
LR_MAX_ITER = 2000

# Tree ensembles: 200 trees balances accuracy vs training time.
# Tested 100 → 500; diminishing returns past 200.
N_ESTIMATORS = 200

# Gradient Boosting max_depth=5 prevents over-fitting on the majority
# class while still capturing interaction effects.  Tested 3-8; 5 gave
# the best macro-F1 on the held-out set.
GB_MAX_DEPTH = 5

# LDA topic modeling
# 6 topics chosen via coherence score comparison (4-10 range).  6 produced
# the most interpretable, non-overlapping themes for this corpus.
LDA_N_TOPICS = 6
LDA_MAX_FEATURES = 2000       # vocabulary cap for CountVectorizer
LDA_MIN_DF = 5                # ignore terms appearing in < 5 reviews
LDA_MAX_DF = 0.7              # ignore terms appearing in > 70% of reviews
LDA_MAX_ITER = 25             # online learning converges by ~20 iterations
LDA_NGRAM_RANGE = (1, 2)

# Data filtering thresholds
# Min reviews required before including a state / community in analysis.
# Prevents noisy conclusions from 1-2 review samples.
MIN_REVIEWS_STATE = 10
MIN_REVIEWS_COMMUNITY = 5

# NER employee extraction: only surface names mentioned 5+ times to
# filter out spaCy false positives (common nouns tagged as PERSON).
MIN_EMPLOYEE_MENTIONS = 5
SPACY_BATCH_SIZE = 200        # spaCy nlp.pipe batch size
ENTITY_CONTEXT_WINDOW = 5     # tokens before/after entity for sentiment

# Distinctive-word analysis parameters
DISTINCTIVE_MIN_RATIO = 1.5   # word must appear 1.5x more often in neg vs pos
DISTINCTIVE_MIN_COUNT = 10    # and at least 10 times total

# Brand colors
SHEA_BLUE = "#1a5276"
SHEA_GOLD = "#d4a843"
SHEA_DARK = "#0e2f44"
POS_GREEN = "#27ae60"
NEU_YELLOW = "#f1c40f"
NEG_RED = "#c0392b"
NEUTRAL_GRAY = "#85929e"
PALETTE_5 = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#1a5276"]

BUILDER_DISPLAY_NAMES = {
    "kb-home": "KB Home",
    "lennar": "Lennar",
    "pulte-homes": "Pulte Homes",
    "shea-homes": "Shea Homes",
}

BUILDER_COLORS = {
    "Shea Homes": "#1a5276",
    "KB Home": "#c0392b",
    "Lennar": "#27ae60",
    "Pulte Homes": "#8e44ad",
}

# pages
PAGES = [
    "Overview",
    "Part 1: Summary Statistics",
    "Part 2: Data Evaluation",
    "Part 3: Sentiment Analysis",
    "Part 4: Advanced NLP",
    "Part 5: Predictive Models",
    "Builder Comparison",
    "Conclusion",
    "Live Prediction Tool",
    "Review Explorer",
]

# global CSS
APP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Source+Serif+4:wght@400;600;700&display=swap');
    .stApp { background-color: #f7f8fa; }
    .block-container { padding-top: 3rem !important; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; color: #0e2f44; }
    h1 { font-size: 2.2rem !important; border-bottom: 3px solid #d4a843; padding-bottom: 0.4rem; margin-bottom: 1rem !important; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e5e8ec; border-radius: 10px; padding: 16px 20px 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
    div[data-testid="stMetric"] label { font-family: 'DM Sans', sans-serif !important; font-size: 12px !important; color: #6b7a8d !important; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: 'Source Serif 4', serif !important; font-size: 28px !important; color: #0e2f44 !important; }
    section[data-testid="stSidebar"] { background-color: #0e2f44; }
    section[data-testid="stSidebar"] * { color: #d6eaf8 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(214,234,248,0.15) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid #e5e8ec; }
    .stTabs [data-baseweb="tab"] { font-family: 'DM Sans'; font-weight: 500; padding: 10px 22px; color: #6b7a8d; }
    .stTabs [aria-selected="true"] { color: #1a5276 !important; border-bottom: 3px solid #d4a843 !important; }
    .section-header { background: linear-gradient(135deg, #0e2f44 0%, #1a5276 100%); color: white !important; padding: 1.2rem 1.8rem; border-radius: 10px; margin: 2rem 0 1.2rem 0; box-shadow: 0 2px 8px rgba(14,47,68,0.15); }
    .section-header h2 { color: white !important; margin: 0 !important; font-size: 1.5rem !important; border: none !important; padding: 0 !important; }
    .section-header p { color: #d6eaf8 !important; margin: 0.3rem 0 0 0 !important; font-size: 0.95rem; opacity: 0.9; }
    .explain-box { background: white; padding: 1rem 1.4rem; border-radius: 0 8px 8px 0; margin: 0.8rem 0 1.2rem 0; font-size: 0.92rem; line-height: 1.6; color: #2c3e50; }
    .commentary-box { background: #eef4f9; padding: 1rem 1.4rem; border-radius: 0 8px 8px 0; margin: 1.2rem 0 1.8rem 0; font-size: 0.92rem; line-height: 1.7; color: #1c2e3d; }
    .commentary-box b { color: #0e2f44; }
    .static-output { background: #1a1a2e; color: #e0e0e0; padding: 1.2rem 1.5rem; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.82rem; line-height: 1.5; overflow-x: auto; white-space: pre-wrap; border: 1px solid #2d2d44; margin: 0.5rem 0; }
    .llm-card { background: white; border: 1px solid #e5e8ec; border-radius: 10px; padding: 1.4rem 1.6rem; margin: 1rem 0; box-shadow: 0 1px 6px rgba(0,0,0,0.05); }
    .llm-card h4 { color: #1a5276; margin-top: 0; }
    .finding { background: linear-gradient(135deg, #fef9e7 0%, #fdebd0 100%); border-left: 4px solid #d4a843; padding: 1rem 1.4rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
    hr { border: none; border-top: 2px solid #e5e8ec; margin: 2.5rem 0; }
</style>
"""
