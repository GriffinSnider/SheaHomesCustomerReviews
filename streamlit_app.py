import streamlit as st
import pandas as pd
import os
import warnings

from streamlit_scroll_to_top import scroll_to_here

from utils.config import APP_CSS, PAGES, ANALYSIS_PAGES, TOOL_PAGES
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
    st.markdown(
        "<h2 style='margin:0 0 2px 0 !important;padding:0 !important'>Shea Homes</h2>"
        "<p style='margin:0 0 6px 0 !important;color:#d6eaf8'>Customer Review Project</p>"
        "<hr style='margin:4px 0 8px 0 !important;border-color:rgba(214,234,248,0.15)'>",
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state.page = ANALYSIS_PAGES[0]

    page = st.session_state.page

    # Sidebar nav button styling — targets ALL sidebar buttons.
    # Active = gold background, white text.  Inactive = dark, readable light text.
    st.markdown("""
    <style>
    /* Collapse excess vertical space in sidebar */
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.15rem !important; }

    /* All sidebar buttons: dark background, readable light text, compact */
    section[data-testid="stSidebar"] button {
        background: rgba(214,234,248,0.08) !important;
        border: 1px solid rgba(214,234,248,0.1) !important;
        color: #d6eaf8 !important;
        text-align: left !important;
        padding: 4px 10px !important;
        font-size: 0.85rem !important;
        font-family: 'DM Sans', sans-serif !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] button:hover {
        background: rgba(214,234,248,0.18) !important;
        color: #ffffff !important;
    }
    /* Active page: gold highlight */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background: rgba(212,168,67,0.35) !important;
        border: 1px solid rgba(212,168,67,0.5) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    def nav_section(label, pages):
        st.markdown(
            f"<p style='font-size:0.75rem;text-transform:uppercase;"
            f"letter-spacing:1px;color:#85929e;margin:6px 0 2px 4px "
            f"!important;font-weight:600'>{label}</p>",
            unsafe_allow_html=True,
        )
        for p in pages:
            active = p == page
            if st.button(p, key=f"nav_{p}", use_container_width=True,
                         type="primary" if active else "secondary"):
                if not active:
                    st.session_state.page = p
                    st.session_state.scroll_top = True
                    st.rerun()

    nav_section("Analysis", ANALYSIS_PAGES)
    nav_section("Tools", TOOL_PAGES)

    page = st.session_state.page

    st.markdown("<hr style='margin:12px 0 10px 0 !important;border-color:rgba(214,234,248,0.15)'>", unsafe_allow_html=True)
    st.caption(f"**{len(fdf):,}** Shea reviews | **49,000+** across all builders")
    csv_mod = pd.Timestamp(os.path.getmtime(DATA_PATH), unit="s")
    st.caption(f"Data refreshed: **{csv_mod.strftime('%b %d, %Y')}**")
    st.caption("Built by **Griffin Snider**")

if st.session_state.get("scroll_top", False):
    scroll_to_here(0, key=f"top_{page}")
    st.session_state.scroll_top = False

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# page dispatch
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
