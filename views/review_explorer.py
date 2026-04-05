import streamlit as st
import pandas as pd

from utils.config import POS_GREEN, NEG_RED, NEU_YELLOW
from utils.components import section_header, explain


def render(df, fdf, page):
    st.title("Review Explorer")
    explain("Search, filter, and browse individual reviews with sentiment scores and star ratings. Use the prediction tool below to analyze any text, or scroll down to browse existing reviews.")

    section_header("Browse Reviews", "Search, filter, and sort the full dataset")
    search = st.text_input("Search reviews", placeholder="keyword (e.g. 'warranty', 'paint', 'delay')")
    col1,col2,col3 = st.columns(3)
    with col1: sent_f = st.multiselect("Sentiment",["Positive","Neutral","Negative"],default=["Positive","Neutral","Negative"])
    with col2: sort_by = st.selectbox("Sort by",["Most Recent","Lowest Rating","Highest Rating","Most Negative Sentiment","Most Positive Sentiment"])
    with col3: n_show = st.slider("Show",10,100,25)
    ex = fdf[fdf["vader_label"].isin(sent_f)].copy()
    if search: ex = ex[ex["review_text"].str.contains(search, case=False, na=False)]
    sm = {"Most Recent":("date",False),"Lowest Rating":("total_score",True),"Highest Rating":("total_score",False),"Most Negative Sentiment":("vader_compound",True),"Most Positive Sentiment":("vader_compound",False)}
    sc,sa = sm[sort_by]; ex = ex.sort_values(sc,ascending=sa)
    st.caption(f"Showing {min(n_show,len(ex)):,} of {len(ex):,} matching reviews"); st.markdown("---")
    for _, r in ex.head(n_show).iterrows():
        scol = POS_GREEN if r["vader_label"]=="Positive" else (NEG_RED if r["vader_label"]=="Negative" else NEU_YELLOW)
        stars_display = f"{int(r['total_score'])}/5 stars"
        ds = r["date"].strftime("%b %d, %Y") if pd.notna(r["date"]) else "N/A"
        st.markdown(f"**{stars_display}** &nbsp;&nbsp; {r['location']} &nbsp;&nbsp; {ds} &nbsp;&nbsp; <span style='color:{scol};font-weight:600'>{r['vader_label']} ({r['vader_compound']:+.2f})</span>", unsafe_allow_html=True)
        st.markdown(f"> {str(r['review_text'])[:600]}{'...' if len(str(r['review_text']))>600 else ''}"); st.markdown("---")
