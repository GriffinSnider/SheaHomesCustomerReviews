import streamlit as st
import pandas as pd
import numpy as np

from utils.components import section_header, explain, commentary, nav_buttons


def render(df, fdf, page):
    st.title("Part 2: Data Evaluation")
    explain(f"Before conducting analysis, the dataset must first be evaluated for quality, representativeness, and potential limitations. Key considerations include whether {len(fdf):,} reviews constitute an adequate sample size, and whether any biases or gaps in coverage may affect the findings. This section provides a transparent assessment of the dataset's strengths and limitations prior to deriving recommendations.")

    section_header("2.1 Business Question")
    explain("<b>Core business question:</b> What do customer reviews show about the Shea Homes homebuying experience, and which aspects of that experience represent the largest opportunity for improvement?")

    section_header("2.2 Sample Size Assessment")
    explain("This section evaluates whether the dataset is large enough to support reliable conclusions. It summarizes the total number of reviews, the average rating, and how reviews are distributed across states.")
    n=len(fdf); mu=fdf["total_score"].mean(); sd=fdf["total_score"].std(); me=1.96*(sd/np.sqrt(n)); ci_lo=mu-me; ci_hi=mu+me
    c1,c2,c3=st.columns(3); c1.metric("Sample Size",f"{n:,}"); c2.metric("Mean Score",f"{mu:.3f}"); c3.metric("95% CI",f"[{ci_lo:.3f}, {ci_hi:.3f}]")
    st.markdown("**Per-state sample sizes:**")
    st.dataframe(fdf["state"].value_counts().reset_index().rename(columns={"index":"State","state":"State","count":"Reviews"}), use_container_width=True, hide_index=True)
    _top2_states = fdf["state"].value_counts().head(2).index.tolist()
    commentary(f"The dataset contains {n:,} reviews with an average rating of {mu:.2f}. To estimate how precise this average is, we calculate a 95% confidence interval, which represents the range where the true average customer rating is likely to fall. The interval of {ci_lo:.3f} to {ci_hi:.3f} indicates that the estimated average rating is statistically stable. Review counts vary across states. {_top2_states[0]} and {_top2_states[1]} account for the largest share of feedback, while several states have smaller samples. Locations with fewer reviews provide less statistical certainty and should be interpreted cautiously.")

    section_header("2.3 Potential Biases")
    explain("Online review datasets often contain structural biases that can influence how results should be interpreted. This section identifies several common sources of bias within the dataset and explains how they may affect the conclusions drawn from the analysis.")
    _b_state_counts = fdf["state"].value_counts()
    _b_top2 = _b_state_counts.head(2)
    _b_top2_pct = _b_top2.sum() / len(fdf)
    _b_top2_names = " and ".join(_b_top2.index.tolist())
    st.dataframe(pd.DataFrame([
        ["Self-selection bias","Customers with strong opinions more likely to review","May overrepresent extremes"],
        ["Geographic concentration",f"{_b_top2_names} account for ~{_b_top2_pct:.0%} of reviews","State-level conclusions should note sample sizes"],
        ["Temporal skew","Peak volume during post-COVID housing boom","Trends may reflect market conditions"],
        ["Platform bias","TrustBuilder is builder-partnered","Sentiment may be inflated vs independent sites"],
    ], columns=["Bias","Description","Impact"]), use_container_width=True, hide_index=True)
    commentary("The most significant is platform bias. Because the reviews originate from a builder-partnered platform, overall ratings may skew more positive than reviews found on independent consumer sites. As a result, the negative reviews that do appear in the dataset are particularly informative. They represent customers whose dissatisfaction was strong enough to be expressed despite the generally positive environment of the platform.")

    nav_buttons(page)
