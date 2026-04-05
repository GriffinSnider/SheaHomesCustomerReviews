import streamlit as st
import pandas as pd
import os
import warnings

from streamlit_scroll_to_top import scroll_to_here

from utils.config import APP_CSS, PAGES
from utils.data import load_and_process

from views import (
    overview,
    summary_stats,
    data_evaluation,
    sentiment_analysis,
    advanced_nlp,
    predictive_models,
    builder_comparison,
    conclusion,
    live_prediction,
    review_explorer,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Shea Homes Customer Reviews Project",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)

# Load data
DATA_PATH = "builder_reviews/shea-homes_reviews.csv"
try:
    df = load_and_process(DATA_PATH)
except FileNotFoundError:
    st.error(f"**Could not find `{DATA_PATH}`.** Place builder CSV files in the `builder_reviews/` directory.")
    st.stop()
fdf = df.copy()

# Sidebar
with st.sidebar:
    st.markdown("## Shea Homes")
    st.markdown("**Customer Review Project**")
    st.markdown("---")

    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]

    selected_page = st.radio("Navigate", PAGES, index=PAGES.index(st.session_state.page))

    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.session_state.scroll_top = True
        st.rerun()

    page = st.session_state.page

    st.markdown("---")
    st.caption(f"**49,000+** total reviews")
    latest_date = fdf["date"].max()
    if pd.notna(latest_date):
        st.caption(f"Latest review: **{latest_date.strftime('%b %d, %Y')}**")
    csv_mod = pd.Timestamp(os.path.getmtime(DATA_PATH), unit="s")
    st.caption(f"Data refreshed: **{csv_mod.strftime('%b %d, %Y')}**")
    st.markdown("---")
    st.caption("Built by **Griffin Snider**")

if st.session_state.get("scroll_top", False):
    scroll_to_here(0, key=f"top_{page}")
    st.session_state.scroll_top = False

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# ── Page dispatch ─────────────────────────────────────────────────────
PAGE_MODULES = {
    PAGES[0]: overview,
    PAGES[1]: summary_stats,
    PAGES[2]: data_evaluation,
    PAGES[3]: sentiment_analysis,
    PAGES[4]: advanced_nlp,
    PAGES[5]: predictive_models,
    PAGES[6]: builder_comparison,
    PAGES[7]: conclusion,
    PAGES[8]: live_prediction,
    PAGES[9]: review_explorer,
}

PAGE_MODULES[page].render(df, fdf, page)
