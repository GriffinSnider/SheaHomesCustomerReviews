import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.config import (
    SHEA_BLUE, SHEA_GOLD, POS_GREEN, NEG_RED, NEU_YELLOW,
    BUILDER_COLORS, SATISFIED_MIN_STARS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
)
from utils.components import section_header, explain, commentary, finding, clean_fig, nav_buttons
from utils.data import load_all_builders


def render(df, fdf, page):
    st.title("Part 6: Builder Comparison")
    explain(
        "Side-by-side analysis of Shea Homes against three major production homebuilders "
        "— KB Home, Lennar, and Pulte Homes — using the same NLP pipeline applied to each "
        "builder's customer reviews on NewHomeSource.com."
    )

    all_df = load_all_builders()
    builders_sorted = sorted(all_df["builder"].unique())

    # 7.1 overall ratings
    section_header("Overall Ratings")

    summary = all_df.groupby("builder").agg(
        reviews=("total_score", "size"),
        avg_rating=("total_score", "mean"),
        median_rating=("total_score", "median"),
        pct_5_star=("total_score", lambda x: (x == 5).mean() * 100),
        pct_at_risk=("total_score", lambda x: (x < SATISFIED_MIN_STARS).mean() * 100),
    ).sort_values("avg_rating", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        order = summary.index.tolist()
        fig = go.Figure(go.Bar(
            y=order, x=summary["avg_rating"],
            orientation="h",
            marker_color=[BUILDER_COLORS[b] for b in order],
            text=[f"{v:.2f}" for v in summary["avg_rating"]],
            textposition="outside",
        ))
        fig.update_xaxes(range=[0, 5.5])
        fig.update_layout(title="Average Rating by Builder")
        clean_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        max_risk = summary["pct_at_risk"].max()
        fig = go.Figure(go.Bar(
            y=order, x=summary["pct_at_risk"],
            orientation="h",
            marker_color=[BUILDER_COLORS[b] for b in order],
            text=[f"{v:.1f}%" for v in summary["pct_at_risk"]],
            textposition="outside",
        ))
        fig.update_xaxes(range=[0, max_risk * 1.25])
        fig.update_layout(title="At-Risk Review Rate (1–3 Stars)")
        clean_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    #  7.2 star rating distribution
    section_header("Star Rating Distribution")

    fig = make_subplots(rows=1, cols=len(builders_sorted), shared_yaxes=True,
                        subplot_titles=builders_sorted)
    for i, builder in enumerate(builders_sorted, 1):
        bslice = all_df[all_df["builder"] == builder]
        dist = bslice["total_score"].value_counts(normalize=True).sort_index() * 100
        fig.add_trace(go.Bar(
            x=dist.index, y=dist.values,
            marker_color=BUILDER_COLORS[builder],
            showlegend=False,
        ), row=1, col=i)
        fig.update_xaxes(dtick=1, title_text="Stars" if i == 1 else "", row=1, col=i)

    fig.update_yaxes(title_text="% of Reviews", row=1, col=1)
    fig.update_layout(title="Star Rating Distribution by Builder")
    clean_fig(fig, 400)
    st.plotly_chart(fig, use_container_width=True)

    # ── 7.3 Rating Dimensions ────────────────────────────────────────
    section_header("Rating Dimensions")

    dims = ["quality", "trustworthiness", "value", "responsiveness"]
    dim_means = all_df.groupby("builder")[dims].mean()

    fig = go.Figure()
    for builder in builders_sorted:
        vals = dim_means.loc[builder]
        fig.add_trace(go.Bar(
            x=[d.title() for d in dims], y=vals,
            name=builder, marker_color=BUILDER_COLORS[builder],
        ))
    fig.update_yaxes(range=[0, 5])
    fig.update_layout(barmode="group", title="Rating Dimensions by Builder")
    clean_fig(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    commentary_lines = []
    for d in dims:
        col_vals = dim_means[d].sort_values(ascending=False)
        best_b, best_v = col_vals.index[0], col_vals.iloc[0]
        worst_b, worst_v = col_vals.index[-1], col_vals.iloc[-1]
        commentary_lines.append(
            f"<b>{d.title()}</b>: {best_b} leads ({best_v:.2f}), {worst_b} trails ({worst_v:.2f})"
        )
    commentary(" · ".join(commentary_lines))

    # 7.4 Sentiment Comparison
    section_header("Sentiment Analysis")

    sent_summary = all_df.groupby("builder").agg(
        avg_vader=("vader_compound", "mean"),
        pct_positive=("vader_label", lambda x: (x == "Positive").mean() * 100),
        pct_neutral=("vader_label", lambda x: (x == "Neutral").mean() * 100),
        pct_negative=("vader_label", lambda x: (x == "Negative").mean() * 100),
    ).sort_values("avg_vader", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        order_s = sent_summary.index.tolist()
        fig = go.Figure(go.Bar(
            y=order_s, x=sent_summary["avg_vader"],
            orientation="h",
            marker_color=[BUILDER_COLORS[b] for b in order_s],
            text=[f"{v:.3f}" for v in sent_summary["avg_vader"]],
            textposition="outside",
        ))
        fig.update_xaxes(range=[0, 1])
        fig.update_layout(title="Average Sentiment Score", showlegend=False)
        clean_fig(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        for label, color in [("Positive", POS_GREEN), ("Neutral", NEU_YELLOW), ("Negative", NEG_RED)]:
            vals = [sent_summary.loc[b, f"pct_{label.lower()}"] for b in builders_sorted]
            fig.add_trace(go.Bar(
                y=builders_sorted, x=vals,
                name=label, orientation="h",
                marker_color=color,
                text=[f"{v:.1f}%" if v >= 6 else "" for v in vals],
                textposition="inside",
                textfont=dict(color="white", size=11),
            ))
        fig.update_layout(barmode="stack", title="Sentiment Breakdown", xaxis_title="% of Reviews")
        clean_fig(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    # 7.5 trends over time
    section_header("Trends Over Time")

    all_df_t = all_df.dropna(subset=["date"]).copy()
    all_df_t["quarter"] = all_df_t["date"].dt.to_period("Q").astype(str)
    all_df_t = all_df_t[all_df_t["quarter"] <= "2026Q1"]

    quarterly = all_df_t.groupby(["quarter", "builder"]).agg(
        reviews=("total_score", "size"),
        avg_rating=("total_score", "mean"),
        avg_sentiment=("vader_compound", "mean"),
    ).reset_index()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Review Volume by Quarter",
                                        "Average Rating by Quarter",
                                        "Average Sentiment by Quarter"],
                        vertical_spacing=0.08)

    for builder in builders_sorted:
        bq = quarterly[quarterly["builder"] == builder]
        fig.add_trace(go.Scatter(
            x=bq["quarter"], y=bq["reviews"],
            name=builder, legendgroup=builder,
            line=dict(color=BUILDER_COLORS[builder], width=1.5),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bq["quarter"], y=bq["avg_rating"],
            name=builder, legendgroup=builder, showlegend=False,
            line=dict(color=BUILDER_COLORS[builder], width=1.5),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=bq["quarter"], y=bq["avg_sentiment"],
            name=builder, legendgroup=builder, showlegend=False,
            line=dict(color=BUILDER_COLORS[builder], width=1.5),
        ), row=3, col=1)

    fig.update_yaxes(title_text="Reviews", row=1, col=1)
    fig.update_yaxes(title_text="Avg Rating", range=[1, 5], row=2, col=1)
    fig.update_yaxes(title_text="Avg VADER", row=3, col=1)
    fig.update_xaxes(title_text="Quarter", row=3, col=1)
    # thin out x-axis labels
    quarters = sorted(quarterly["quarter"].unique())
    tick_vals = quarters[::4]
    for row in range(1, 4):
        fig.update_xaxes(tickvals=tick_vals, tickangle=45, row=row, col=1)
    fig.update_layout(height=750)
    clean_fig(fig, 750)
    st.plotly_chart(fig, use_container_width=True)

    # 7.6 Geographic footprint
    section_header("Geographic Footprint")

    geo = all_df.dropna(subset=["state"]).groupby(["state", "builder"]).agg(
        reviews=("total_score", "size"),
        avg_rating=("total_score", "mean"),
        avg_sentiment=("vader_compound", "mean"),
    ).reset_index()

    # Only include states where Shea Homes has reviews
    shea_states = sorted(
        geo[geo["builder"] == "Shea Homes"]["state"].unique()
    )
    geo = geo[geo["state"].isin(shea_states)]

    fig = go.Figure()
    for builder in builders_sorted:
        bg = geo[geo["builder"] == builder]
        fig.add_trace(go.Bar(
            x=bg["state"], y=bg["reviews"],
            name=builder, marker_color=BUILDER_COLORS[builder],
        ))
    fig.update_layout(barmode="group", title="Review Count by State (Shea Markets)")
    clean_fig(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    for builder in builders_sorted:
        bg = geo[geo["builder"] == builder]
        fig.add_trace(go.Bar(
            x=bg["state"], y=bg["avg_rating"],
            name=builder, marker_color=BUILDER_COLORS[builder],
        ))
    fig.update_yaxes(range=[0, 5])
    fig.update_layout(barmode="group", title="Average Rating by State (Shea Markets)")
    clean_fig(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    for builder in builders_sorted:
        bg = geo[geo["builder"] == builder]
        fig.add_trace(go.Bar(
            x=bg["state"], y=bg["avg_sentiment"],
            name=builder, marker_color=BUILDER_COLORS[builder],
        ))
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(barmode="group", title="Average Sentiment by State (Shea Markets)")
    clean_fig(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    # 7.7 distinctive language
    section_header("Distinctive Language by Builder")

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, stop_words="english",
        ngram_range=TFIDF_NGRAM_RANGE, token_pattern=r"(?u)\b\w[\w']+\b",
    )
    X = tfidf.fit_transform(all_df["review_text"].astype(str))
    feature_names = tfidf.get_feature_names_out()
    overall_mean = np.asarray(X.mean(axis=0)).flatten()

    # Build name tokens to filter out (e.g. "kb", "home", "lennar", "pulte", "shea", "homes")
    _name_tokens = set()
    for b in builders_sorted:
        _name_tokens.update(b.lower().split())
    cols = st.columns(2)
    for idx, builder in enumerate(builders_sorted):
        mask = (all_df["builder"] == builder).values
        avg_tfidf = np.asarray(X[mask].mean(axis=0)).flatten()
        diff = avg_tfidf - overall_mean
        # Sort all terms by lift, then filter out builder name variants
        ranked_idx = diff.argsort()[::-1]
        words, scores = [], []
        for i in ranked_idx:
            term = feature_names[i]
            term_tokens = set(term.lower().split())
            # Skip if any token in the term is part of a builder name
            if term_tokens & _name_tokens:
                continue
            words.append(term)
            scores.append(diff[i])
            if len(words) == 15:
                break

        fig = go.Figure(go.Bar(
            y=words[::-1], x=scores[::-1],
            orientation="h",
            marker_color=BUILDER_COLORS[builder],
        ))
        fig.update_layout(title=f"{builder} — Distinctive Terms",
                          xaxis_title="TF-IDF Lift vs. Average",
                          showlegend=False)
        clean_fig(fig, 380)
        with cols[idx % 2]:
            st.plotly_chart(fig, use_container_width=True)

    # 7.8 Review Length & Engagement
    section_header("Review Length and Engagement")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for builder in builders_sorted:
            bslice = all_df[all_df["builder"] == builder]
            fig.add_trace(go.Histogram(
                x=bslice["word_count"].clip(upper=300),
                name=builder, opacity=0.5,
                marker_color=BUILDER_COLORS[builder],
                nbinsx=50,
            ))
        fig.update_layout(barmode="overlay", title="Review Length Distribution",
                          xaxis_title="Word Count", yaxis_title="Reviews")
        clean_fig(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        wc_by_rating = all_df.groupby(["builder", "total_score"])["word_count"].mean().reset_index()
        fig = go.Figure()
        for builder in builders_sorted:
            bslice = wc_by_rating[wc_by_rating["builder"] == builder]
            fig.add_trace(go.Scatter(
                x=bslice["total_score"], y=bslice["word_count"],
                mode="lines+markers", name=builder,
                line=dict(color=BUILDER_COLORS[builder], width=2),
            ))
        fig.update_xaxes(dtick=1, title_text="Star Rating")
        fig.update_yaxes(title_text="Avg Word Count")
        fig.update_layout(title="Avg Review Length by Rating")
        clean_fig(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    # 7.9 Summary
    section_header("Summary")

    best_builder = summary.index[0]
    best_avg = summary.loc[best_builder, "avg_rating"]
    lowest_neg_b = sent_summary["pct_negative"].idxmin()

    finding(
        f"<b>Key takeaways:</b><br>"
        f"1. <b>Overall positioning:</b> {best_builder} leads with a {best_avg:.2f} average rating; "
        f"Lennar trails with the highest at-risk rate.<br>"
        f"2. <b>Dimensional strengths:</b> Each builder shows a distinct profile across quality, trust, "
        f"value, and responsiveness — no single builder dominates every dimension.<br>"
        f"3. <b>Sentiment patterns:</b> {lowest_neg_b} has the lowest share of negative reviews, "
        f"suggesting stronger language-level satisfaction even beyond star ratings.<br>"
        f"4. <b>Geographic overlap:</b> {len(shea_states)} Shea markets compared head-to-head "
        f"on ratings and sentiment.<br>"
        f"5. <b>Language signals:</b> TF-IDF distinctive terms reveal what topics and phrases "
        f"are unique to each builder's customer base.<br><br>"
        f"The full Shea Homes deep-dive (sentiment, topic modeling, predictive classification, "
        f"LLM analysis) is covered in Parts 1–5."
    )

    nav_buttons(page)
