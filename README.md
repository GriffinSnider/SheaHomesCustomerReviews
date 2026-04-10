# Shea Homes Customer Review Analysis

**NLP and Machine Learning Applied to 49,000+ Homebuyer Reviews Across Four Builders**

[**Live Streamlit App**](https://sheahomesreviews.streamlit.app/)

---

## Overview

NewHomeSource.com is one of the largest online marketplaces for newly built homes in the United States. Prospective buyers use it to browse floor plans, compare communities, and read verified reviews from actual homebuyers after closing. For a production homebuilder like Shea Homes, these reviews feed directly into purchase behavior: a buyer evaluating two builders in the same market will often let this feedback serve as a tiebreaker.

This project applies NLP and machine learning to 49,000+ customer reviews across four major production homebuilders — Shea Homes (~2,000 reviews), KB Home (~11,000), Lennar (~20,000), and Pulte Homes (~15,000) — spanning September 2020 through April 2026. Shea Homes is the primary subject of a deep-dive analysis, with the three competitors providing market context through a side-by-side comparison. The goal is to turn unstructured review text into concrete findings about customer satisfaction, recurring themes, and what predicts a negative rating before it shows up in aggregate scores.

## Analysis Pipeline

The analysis runs in five layers:

**Layer 1: Sentiment Scoring**
Dual-method sentiment analysis using VADER (lexicon-based) and TextBlob (pattern-based) to quantify the emotional tone of every review. Produces compound sentiment scores, polarity, and subjectivity measures across the full dataset.

**Layer 2: Topic Extraction**
TF-IDF keyword analysis and Latent Dirichlet Allocation (LDA) topic modeling to surface what customers talk about most, what distinguishes positive reviews from negative ones, and what themes cluster across communities.

**Layer 3: Predictive Classification**
Machine learning classifiers (Logistic Regression, Random Forest, Gradient Boosting) trained to predict satisfaction level from review text alone. Achieves 84.1% accuracy, flagging at-risk reviews before they move the aggregate rating.

**Layer 4: LLM-Powered Review Intelligence**
LLaMA 3.2 (via Ollama) applied for zero-shot and few-shot sentiment classification, benchmarked against the traditional NLP approaches on the same dataset.

**Layer 5: Builder Comparison**
The same NLP pipeline run on KB Home (~11K reviews), Lennar (~20K reviews), and Pulte Homes (~15K reviews) for a direct comparison of ratings, sentiment, rating dimensions, geographic footprint, and word-level differences between builders.

## Key Findings

- 49,000+ reviews analyzed across four builders (~2,000 Shea Homes, ~11,000 KB Home, ~20,000 Lennar, ~15,000 Pulte Homes)
- Average rating across all Shea Homes reviews: 4.21 / 5.0
- Gradient Boosting classifier accuracy: 84.1%
- Approximately 22% of reviews flagged as at-risk based on model predictions
- TF-IDF and topic modeling point to construction quality and post-close warranty service as the primary drivers of negative sentiment
- LLM-based classification shows competitive performance with traditional ML on this task
- Builder comparison shows where Shea Homes' ratings, sentiment, and customer themes diverge from KB Home, Lennar, and Pulte in shared markets

## Tech Stack

| Component | Tools |
|---|---|
| Data Collection | Python, Playwright, BeautifulSoup |
| Data Processing | pandas, NumPy |
| Sentiment Analysis | VADER (NLTK), TextBlob |
| Topic Modeling | scikit-learn (TF-IDF, LDA) |
| ML Classification | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| NER | spaCy (en_core_web_sm) |
| LLM Analysis | Ollama, LLaMA 3.2 |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Testing | pytest |
| Deployment | Streamlit Community Cloud |

## Project Structure

```
├── streamlit_app.py              # Main Streamlit entry point
├── train_models.py               # ML model training script
├── requirements.txt
│
├── views/                        # Streamlit page modules
│   ├── overview.py               # Introduction
│   ├── summary_stats.py          # Part 1: Summary Statistics
│   ├── data_evaluation.py        # Part 2: Data Evaluation
│   ├── sentiment_analysis.py     # Part 3: Sentiment Analysis
│   ├── advanced_nlp.py           # Part 4: Advanced NLP
│   ├── predictive_models.py      # Part 5: Predictive Models
│   ├── builder_comparison.py     # Part 6: Builder Comparison
│   ├── conclusion.py             # Conclusion
│   ├── live_prediction.py        # Tool: Live Prediction
│   └── review_explorer.py        # Tool: Review Explorer
│
├── utils/
│   ├── config.py                 # Colors, thresholds, page definitions, CSS
│   ├── components.py             # Reusable UI components
│   └── data.py                   # Data loading and processing
│
├── notebooks/                    # Jupyter notebooks (analysis development)
│   ├── 01_setup_and_summary_stats.ipynb
│   ├── 02_data_evaluation.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_advanced_nlp.ipynb
│   ├── 05_predictive_models.ipynb
│   ├── 06_llm_analysis.ipynb
│   └── 07_competitor_comparison.ipynb
│
├── scrapers/
│   └── review_scraper.py         # NewHomeSource review scraper
│
├── builder_reviews/              # Review CSVs (one per builder)
│   ├── shea-homes_reviews.csv
│   ├── kb-home_reviews.csv
│   ├── lennar_reviews.csv
│   └── pulte-homes_reviews.csv
│
├── models/                       # Trained model artifacts (.joblib)
│   ├── metadata.joblib
│   ├── binary_tfidf.joblib
│   ├── three_class_tfidf.joblib
│   ├── binary_logistic_regression.joblib
│   ├── binary_random_forest.joblib
│   ├── binary_gradient_boosting.joblib
│   ├── three_logistic_regression.joblib
│   ├── three_random_forest.joblib
│   └── three_gradient_boosting.joblib
│
└── tests/                        # pytest test suite
    ├── conftest.py
    ├── test_config.py
    ├── test_data.py
    ├── test_nlp.py
    └── test_predictions.py
```

## How to Run

**Streamlit App (local):**
```bash
git clone https://github.com/GriffinSnider/SheaHomesCustomerReviews.git
cd SheaHomesCustomerReviews
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Jupyter Notebooks:**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

The notebooks are numbered 01–07 and follow the same sequence as the Streamlit app. Each notebook can be run independently.

Note: The LLM analysis notebook (`06_llm_analysis.ipynb`) requires [Ollama](https://ollama.com/) installed locally with the LLaMA 3.2 model pulled (`ollama pull llama3.2`). All other notebooks and the Streamlit app run with standard Python packages.

## Refreshing the Data

The dataset is a static CSV snapshot. To pull the latest reviews and retrain the models:

```bash
# 1. Re-scrape reviews from NewHomeSource.com
python scrapers/review_scraper.py

# 2. Retrain the ML models on the updated data
python train_models.py

# 3. (Optional) Launch the app locally to verify
streamlit run streamlit_app.py
```

The scraper writes to `builder_reviews/`. The training script reads from the same directory. The notebooks and Streamlit app both load data from there.

## Tests

```bash
pytest tests/
```

## Data Source

All reviews were scraped from builder profiles on [NewHomeSource.com](https://www.newhomesource.com/). The dataset contains verified homebuyer reviews including review text, star ratings, reviewer names, community names, and review dates across four builders: Shea Homes, KB Home, Lennar, and Pulte Homes. Only public display names were collected, no private or personally identifiable information.

## Author

**Griffin Snider**

[LinkedIn](https://www.linkedin.com/in/griffinsnider/) | [GitHub](https://github.com/GriffinSnider)
