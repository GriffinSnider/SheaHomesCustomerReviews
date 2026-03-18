# Shea Homes Customer Review Analysis

**NLP and Machine Learning Applied to 2,039 Homebuyer Reviews**

[**Live Streamlit App**](https://sheahomesreviews.streamlit.app/)

---

## Overview

NewHomeSource.com is one of the largest online marketplaces for newly built homes in the United States. Prospective buyers use it to browse floor plans, compare communities, and read verified reviews from actual homebuyers after closing. For a production homebuilder like Shea Homes, these reviews are a direct input into purchase behavior: a buyer evaluating two builders in the same market will often let this feedback serve as a tiebreaker.

This project applies a layered suite of NLP and machine learning techniques to extract business intelligence from 2,039 Shea Homes customer reviews spanning September 2020 through March 2026. The goal is to transform unstructured text data, invisible to traditional reporting tools, into actionable insights about customer satisfaction, recurring themes, and predictive indicators of buyer sentiment.

## Analysis Pipeline

The analysis is structured in four progressive layers, each building on the last:

**Layer 1: Sentiment Scoring**
Dual-method sentiment analysis using VADER (lexicon-based) and TextBlob (pattern-based) to quantify the emotional tone of every review. Provides compound sentiment scores, polarity, and subjectivity measures across the full dataset.

**Layer 2: Topic Extraction**
TF-IDF keyword analysis and Latent Dirichlet Allocation (LDA) topic modeling to find what customers talk about most frequently, what distinguishes positive reviews from negative ones, and what themes emerge across communities.

**Layer 3: Predictive Classification**
Machine learning classifiers (Logistic Regression, Random Forest, Gradient Boosting) trained to predict customer satisfaction level from review text alone. Achieves 84.1% accuracy, enabling identification of at-risk reviews before they surface in aggregate ratings.

**Layer 4: LLM-Powered Review Intelligence**
LLaMA 3.2 (via Ollama) applied for zero-shot and few-shot sentiment classification, providing a comparison between traditional NLP baselines and modern large language model approaches on the same dataset.

## Key Findings

- Average rating across all Shea Homes reviews: **4.21 / 5.0**
- Gradient Boosting classifier accuracy: **84.1%**
- Approximately **22%** of reviews flagged as at-risk based on model predictions
- TF-IDF and topic modeling show construction quality and post-close warranty service as the primary drivers of negative sentiment
- LLM-based classification shows competitive performance with traditional ML on this task

## Tech Stack

| Component | Tools |
|---|---|
| Data Collection | Python, BeautifulSoup, requests |
| Data Processing | pandas, NumPy |
| Sentiment Analysis | VADER (NLTK), TextBlob |
| Topic Modeling | scikit-learn (TF-IDF, LDA) |
| ML Classification | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| LLM Analysis | Ollama, LLaMA 3.2 |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |

## How to Run

**Streamlit App (local):**
```bash
git clone https://github.com/GriffinSnider/SheaHomesCustomerReviews.git
cd SheaHomesCustomerReviews
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Jupyter Notebook:**
```bash
pip install -r requirements.txt
jupyter notebook SheaHomesCustomerReviewAnalysis.ipynb
```

Note: The LLM analysis section (Layer 4) requires [Ollama](https://ollama.com/) installed locally with the LLaMA 3.2 model pulled (`ollama pull llama3.2`). All other sections run with standard Python packages.

## Data Source

All reviews were scraped from the [Shea Homes builder profile on NewHomeSource.com](https://www.newhomesource.com/builder/shea-homes/reviews/612/). The dataset contains verified homebuyer reviews including review text, star ratings, reviewer names, community names, and review dates. No private or personally identifiable information beyond public display names was collected.

## Author

**Griffin Snider**

[LinkedIn](https://www.linkedin.com/in/griffinsnider/) | [GitHub](https://github.com/GriffinSnider)
