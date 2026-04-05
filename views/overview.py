import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.config import SHEA_BLUE, SHEA_GOLD, SHEA_DARK, POS_GREEN, NEG_RED, SATISFIED_MIN_STARS
from utils.components import section_header, explain, commentary, clean_fig, nav_buttons


def render(df, fdf, page):
    st.title("Shea Homes Customer Review Project")
    _states_list = sorted(fdf["state"].dropna().unique())
    _states_str = ", ".join(_states_list[:-1]) + ", and " + _states_list[-1] if len(_states_list) > 1 else ", ".join(_states_list)
    explain(f"This project analyzes {len(fdf):,} customer reviews of Shea Homes collected from NewHomeSource.com. NewHomeSource is a review platform where verified homebuyers rate their builder after closing. Each review includes an overall star rating (1-5) plus four sub-ratings: Quality, Trustworthiness, Value, and Responsiveness. Buyers also write open-ended comments describing their experience. This dataset contains reviews for Shea Homes collected between {fdf['date'].min().strftime('%B %Y')} and {fdf['date'].max().strftime('%B %Y')}, covering {fdf['state'].nunique()} markets across {_states_str}. <br> <br> Reading {len(fdf):,} reviews manually would take weeks. Even then, it would be hard to spot patterns consistently, like which markets are struggling, what topics keep coming up in negative reviews, or which customers might be at risk of leaving bad word-of-mouth. This project automates that work using a multi-method Natural Language Processing (NLP) framework. The pipeline includes: (1) sentiment classification using VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based model tuned for customer review text; (2) topic extraction using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and LDA (Latent Dirichlet Allocation), which surface recurring themes without predefined categories; (3) predictive modeling using machine learning classifiers to flag at-risk reviews.")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Reviews", f"{len(fdf):,}"); c2.metric("Avg Rating", f"{fdf['total_score'].mean():.2f}"); c3.metric("Positive", f"{(fdf['vader_label']=='Positive').mean():.0%}")
    c4.metric("Negative", f"{(fdf['vader_label']=='Negative').mean():.0%}"); c5.metric("At-Risk", f"{(fdf['total_score'] < SATISFIED_MIN_STARS).sum():,}"); c6.metric("Markets", f"{fdf['state'].nunique()}")
    cats = ["total_score","quality","trustworthiness","value","responsiveness"]; clabels = ["Overall","Quality","Trust","Value","Responsiveness"]
    cmeans = [fdf[c].mean() for c in cats]
    fig = go.Figure(go.Bar(x=clabels, y=cmeans, marker_color=[SHEA_GOLD if v==max(cmeans) else (NEG_RED if v==min(cmeans) else SHEA_BLUE) for v in cmeans], text=[f"{v:.2f}" for v in cmeans], textposition="outside", textfont_size=14))
    fig.update_yaxes(range=[0,5.5]); fig.update_layout(title="Shea Homes Average Scores by Category"); clean_fig(fig, 400); st.plotly_chart(fig, use_container_width=True)
    _pct_high = (fdf["total_score"] >= SATISFIED_MIN_STARS).mean()
    _at_risk_n = (fdf["total_score"] < SATISFIED_MIN_STARS).sum()
    _at_risk_pct = (fdf["total_score"] < SATISFIED_MIN_STARS).mean()
    commentary(f"Across {len(fdf):,} customer reviews, Shea Homes maintains an overall rating of {fdf['total_score'].mean():.2f} out of 5, with {_pct_high:.0%} of customers awarding four or five stars. Performance is consistent across categories, including quality, trust, value, and responsiveness, with trust receiving the highest scores while quality rates slightly lower than the other dimensions. <br> <br> Within the dataset, {_at_risk_n:,} reviews ({_at_risk_pct:.0%}) fall into the 1–3 star range, representing customers who reported dissatisfaction with some aspect of their experience. These 'at-risk' reviews are particularly important from an operational perspective and serve as the primary focus for the deeper text and sentiment analysis conducted in later sections.")
    st.markdown(f"**Data Source:** [NewHomeSource](https://www.newhomesource.com/builder/shea-homes/reviews/612/) &nbsp;|&nbsp; **Date Range:** {df['date'].min().strftime('%B %Y')} to {df['date'].max().strftime('%B %Y')} &nbsp;|&nbsp; **Author:** Griffin Snider")

    nav_buttons(page)
