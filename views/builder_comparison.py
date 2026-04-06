import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.config import SHEA_BLUE, SHEA_GOLD, POS_GREEN, NEG_RED, NEU_YELLOW, BUILDER_COLORS, SATISFIED_MIN_STARS
from utils.components import section_header, explain, commentary, finding, clean_fig, nav_buttons
from utils.data import load_all_builders


def render(df, fdf, page):
    st.title("Part 6: Builder Comparison")
    explain("Side-by-side analysis of Shea Homes against three major competitors — KB Home, Lennar, and Pulte Homes — using the same NLP pipeline applied to each builder's customer reviews on NewHomeSource.com.")

    all_df = load_all_builders()
    builders = sorted(all_df["builder"].unique())

    # scorecard
    section_header("Overall Ratings")
    cols = st.columns(len(builders))
    for i, b in enumerate(builders):
        bslice = all_df[all_df["builder"] == b]
        with cols[i]:
            avg = bslice["total_score"].mean()
            st.metric(b, f"{avg:.2f} / 5.0", f"{len(bslice):,} reviews")

    # star rating distribution
    section_header("Star Rating Distribution")
    dist = all_df.groupby(["builder", "total_score"]).size().reset_index(name="count")
    totals = all_df.groupby("builder").size().reset_index(name="total")
    dist = dist.merge(totals, on="builder")
    dist["pct"] = dist["count"] / dist["total"] * 100
    fig = px.bar(dist, x="total_score", y="pct", color="builder", barmode="group",
                 color_discrete_map=BUILDER_COLORS,
                 labels={"total_score": "Star Rating", "pct": "% of Reviews", "builder": "Builder"})
    fig.update_xaxes(dtick=1)
    clean_fig(fig, 420); st.plotly_chart(fig, use_container_width=True)

    # sub score comparison
    section_header("Rating Dimensions")
    dims = ["quality", "trustworthiness", "value", "responsiveness"]
    dim_data = []
    for b in builders:
        bslice = all_df[all_df["builder"] == b]
        for d in dims:
            dim_data.append({"Builder": b, "Dimension": d.title(), "Avg Score": bslice[d].mean()})
    dim_df = pd.DataFrame(dim_data)
    fig = px.bar(dim_df, x="Dimension", y="Avg Score", color="Builder", barmode="group",
                 color_discrete_map=BUILDER_COLORS)
    fig.update_yaxes(range=[0, 5])
    clean_fig(fig, 420); st.plotly_chart(fig, use_container_width=True)

    commentary_lines = []
    for d in dims:
        best = dim_df[dim_df["Dimension"] == d.title()].sort_values("Avg Score", ascending=False).iloc[0]
        worst = dim_df[dim_df["Dimension"] == d.title()].sort_values("Avg Score", ascending=True).iloc[0]
        commentary_lines.append(f"<b>{d.title()}</b>: {best['Builder']} leads ({best['Avg Score']:.2f}), {worst['Builder']} trails ({worst['Avg Score']:.2f})")
    commentary(" · ".join(commentary_lines))

    # sentiment comparison
    section_header("Sentiment Analysis")
    sent_data = []
    for b in builders:
        bslice = all_df[all_df["builder"] == b]
        for label in ["Positive", "Neutral", "Negative"]:
            ct = (bslice["vader_label"] == label).sum()
            sent_data.append({"Builder": b, "Sentiment": label, "Count": ct, "Pct": ct / len(bslice) * 100})
    sent_df = pd.DataFrame(sent_data)

    col1, col2 = st.columns(2)
    with col1:
        avg_sent = all_df.groupby("builder")["vader_compound"].mean().reset_index()
        avg_sent.columns = ["Builder", "Avg VADER Compound"]
        avg_sent = avg_sent.sort_values("Avg VADER Compound", ascending=True)
        fig = px.bar(avg_sent, x="Avg VADER Compound", y="Builder", orientation="h",
                     color="Builder", color_discrete_map=BUILDER_COLORS)
        fig.update_layout(title="Avg VADER Compound Score", showlegend=False)
        fig.update_xaxes(range=[0, 1])
        clean_fig(fig, 350); st.plotly_chart(fig, use_container_width=True)
    with col2:
        neg_pct = sent_df[sent_df["Sentiment"] == "Negative"][["Builder", "Pct"]].sort_values("Pct", ascending=True)
        fig = px.bar(neg_pct, x="Pct", y="Builder", orientation="h",
                     color="Builder", color_discrete_map=BUILDER_COLORS)
        fig.update_layout(title="% Negative Reviews", showlegend=False)
        clean_fig(fig, 350); st.plotly_chart(fig, use_container_width=True)

    # sentiment stacked bar
    color_map = {"Positive": POS_GREEN, "Neutral": NEU_YELLOW, "Negative": NEG_RED}
    fig = px.bar(sent_df, x="Builder", y="Pct", color="Sentiment", barmode="stack",
                 color_discrete_map=color_map,
                 labels={"Pct": "% of Reviews"})
    fig.update_layout(title="Sentiment Breakdown by Builder")
    clean_fig(fig, 400); st.plotly_chart(fig, use_container_width=True)

    # review volume over time
    section_header("Review Volume Over Time")
    all_df_t = all_df.dropna(subset=["date"]).copy()
    all_df_t["quarter"] = all_df_t["date"].dt.to_period("Q").astype(str)
    vol = all_df_t.groupby(["quarter", "builder"]).size().reset_index(name="count")
    fig = px.line(vol, x="quarter", y="count", color="builder",
                  color_discrete_map=BUILDER_COLORS,
                  labels={"quarter": "Quarter", "count": "Reviews", "builder": "Builder"})
    fig.update_layout(title="Review Volume by Quarter")
    clean_fig(fig, 400); st.plotly_chart(fig, use_container_width=True)

    # sentiment trend over time
    sent_trend = all_df_t.groupby(["quarter", "builder"])["vader_compound"].mean().reset_index()
    fig = px.line(sent_trend, x="quarter", y="vader_compound", color="builder",
                  color_discrete_map=BUILDER_COLORS,
                  labels={"quarter": "Quarter", "vader_compound": "Avg VADER Compound", "builder": "Builder"})
    fig.update_layout(title="Avg Sentiment Over Time")
    clean_fig(fig, 400); st.plotly_chart(fig, use_container_width=True)

    # geographic footprint
    section_header("Geographic Footprint")
    geo = all_df.dropna(subset=["state"]).groupby(["state", "builder"]).agg(
        reviews=("total_score", "size"), avg_rating=("total_score", "mean")
    ).reset_index()

    fig = px.bar(geo, x="state", y="reviews", color="builder", barmode="group",
                 color_discrete_map=BUILDER_COLORS,
                 labels={"state": "State", "reviews": "Reviews", "builder": "Builder"})
    fig.update_layout(title="Review Count by State")
    clean_fig(fig, 420); st.plotly_chart(fig, use_container_width=True)

    shared_states = geo.groupby("state")["builder"].nunique()
    shared_states = shared_states[shared_states > 1].index.tolist()
    if shared_states:
        shared_geo = geo[geo["state"].isin(shared_states)]
        fig = px.bar(shared_geo, x="state", y="avg_rating", color="builder", barmode="group",
                     color_discrete_map=BUILDER_COLORS,
                     labels={"state": "State", "avg_rating": "Avg Rating", "builder": "Builder"})
        fig.update_layout(title="Avg Rating in Shared Markets")
        fig.update_yaxes(range=[0, 5])
        clean_fig(fig, 420); st.plotly_chart(fig, use_container_width=True)

    # at risk review rate
    section_header("At-Risk Reviews")
    risk = all_df.groupby("builder").apply(lambda g: (g["total_score"] < SATISFIED_MIN_STARS).mean() * 100).reset_index(name="at_risk_pct")
    risk = risk.sort_values("at_risk_pct", ascending=True)
    fig = px.bar(risk, x="at_risk_pct", y="builder", orientation="h",
                 color="builder", color_discrete_map=BUILDER_COLORS,
                 labels={"at_risk_pct": "% At-Risk (1-3 Stars)", "builder": "Builder"})
    fig.update_layout(showlegend=False, title="At-Risk Review Rate")
    clean_fig(fig, 350); st.plotly_chart(fig, use_container_width=True)

    # key takeaway
    best_overall = all_df.groupby("builder")["total_score"].mean().idxmax()
    best_score = all_df.groupby("builder")["total_score"].mean().max()
    lowest_neg = sent_df[sent_df["Sentiment"] == "Negative"].sort_values("Pct").iloc[0]["Builder"]
    finding(
        f"<b>Key finding:</b> {best_overall} has the highest average rating ({best_score:.2f}/5.0) "
        f"and {lowest_neg} has the lowest share of negative reviews among the four builders compared. "
        f"The full Shea Homes deep-dive analysis is available in Parts 1–5 of this project."
    )

    nav_buttons(page)
