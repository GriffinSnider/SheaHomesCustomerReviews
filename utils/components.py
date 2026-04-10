import streamlit as st
from utils.config import SHEA_DARK, ANALYSIS_PAGES


def section_header(title, subtitle=""):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="section-header"><h2>{title}</h2>{sub}</div>', unsafe_allow_html=True)


def explain(text):
    st.markdown(f'<div class="explain-box">{text}</div>', unsafe_allow_html=True)


def commentary(text):
    st.markdown(f'<div class="commentary-box">{text}</div>', unsafe_allow_html=True)


def static_output(text):
    st.markdown(f'<div class="static-output">{text}</div>', unsafe_allow_html=True)


def finding(text):
    st.markdown(f'<div class="finding">{text}</div>', unsafe_allow_html=True)


def clean_fig(fig, height=400):
    fig.update_layout(
        font_family="DM Sans",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
        title_font_size=15,
        title_font_color=SHEA_DARK,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def nav_buttons(page):
    """Back/Next buttons for analysis pages only. Tool pages get no nav buttons."""
    if page not in ANALYSIS_PAGES:
        return
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if page != ANALYSIS_PAGES[0]:
            if st.button("⇽ Back"):
                _previous_page()
    with col3:
        if page != ANALYSIS_PAGES[-1]:
            if st.button("Next ⇾"):
                _next_page()


def _next_page():
    current_index = ANALYSIS_PAGES.index(st.session_state.page)
    if current_index < len(ANALYSIS_PAGES) - 1:
        st.session_state.page = ANALYSIS_PAGES[current_index + 1]
        st.session_state.scroll_top = True
        st.rerun()


def _previous_page():
    current_index = ANALYSIS_PAGES.index(st.session_state.page)
    if current_index > 0:
        st.session_state.page = ANALYSIS_PAGES[current_index - 1]
        st.session_state.scroll_top = True
        st.rerun()
