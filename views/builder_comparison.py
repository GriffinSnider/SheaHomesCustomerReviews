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
        "A competitive benchmarking comparing Shea Homes against KB Home, Lennar, and "
        "Pulte using the same NLP pipeline applied to each builder's customer reviews on "
        "NewHomeSource.com."
    )

    all_df = load_all_builders()
    builders_sorted = sorted(all_df["builder"].unique())

    # 6.1 overall ratings
    section_header("6.1 Overall Ratings")
    explain(
        "This section compares two metrics for each builder: average star rating and the share of "
        "reviews rated 1–3 stars (at-risk). Average rating shows where each builder stands overall; "
        "the at-risk rate shows how many customers report a poor experience. "

    )

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

    best_builder_overall = summary.index[0]
    best_avg_overall = summary.loc[best_builder_overall, "avg_rating"]
    worst_builder_overall = summary.index[-1]
    worst_risk = summary.loc[worst_builder_overall, "pct_at_risk"]
    commentary(
        f"{best_builder_overall} leads with an average rating of {best_avg_overall:.2f}, while "
        f"{worst_builder_overall} has the highest at-risk rate at {worst_risk:.1f}%. The spread "
        f"between the top and bottom builder is "
        f"{summary['avg_rating'].max() - summary['avg_rating'].min():.2f} points. That gap is "
        f"not extreme, suggesting all four builders deliver broadly comparable experiences with "
        f"important differences at the margins."
    )

    # 6.2 star rating distribution
    section_header("6.2 Star Rating Distribution")
    explain(
        "Average ratings can mask important differences in how reviews are distributed. A builder "
        "with many 5-star and many 1-star reviews looks different from one with mostly 4-star reviews, "
        "even if the averages are similar. This section shows the full distribution for each builder "
        "side by side."
    )

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

    # compute distribution commentary
    dist_lines = []
    for builder in builders_sorted:
        bslice = all_df[all_df["builder"] == builder]
        pct5 = (bslice["total_score"] == 5).mean() * 100
        pct1 = (bslice["total_score"] == 1).mean() * 100
        dist_lines.append(f"{builder}: {pct5:.0f}% five-star, {pct1:.0f}% one-star")
    commentary(
        "The shape of each distribution tells a different story. "
        + " · ".join(dist_lines)
        + ". Builders with a heavier concentration at the extremes (high 5-star and notable 1-star shares) "
        "tend to have more polarized customer experiences, while a tighter cluster around 4–5 stars "
        "indicates more consistent delivery."
    )

    # 6.3 Rating Dimensions
    section_header("6.3 Rating Dimensions")
    explain(
        "Each review on NewHomeSource includes sub-ratings for Quality, Trustworthiness, Value, and "
        "Responsiveness. Comparing these dimensions across builders reveals where each company's "
        "relative strengths and weaknesses lie. A builder may score well overall but trail on a "
        "specific dimension that matters to buyers."
    )

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
            f"{d.title()}: {best_b} leads ({best_v:.2f}), {worst_b} trails ({worst_v:.2f})"
        )
    commentary(" · ".join(commentary_lines))

    # 6.4 Sentiment Comparison
    section_header("6.4 Sentiment Analysis")
    explain(
        "Star ratings capture what customers scored; sentiment analysis captures how they wrote about "
        "it. Using the VADER compound score (ranging from -1 to +1), this section compares the average "
        "sentiment of each builder's review text and breaks reviews into Positive, Neutral, and "
        "Negative categories. Differences between star ratings and sentiment can reveal builders whose "
        "customers write more positively (or negatively) than their numeric scores alone suggest."
    )

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

    sent_best = sent_summary.index[0]
    sent_best_score = sent_summary.loc[sent_best, "avg_vader"]
    sent_worst = sent_summary.index[-1]
    sent_worst_neg = sent_summary.loc[sent_worst, "pct_negative"]
    commentary(
        f"{sent_best} leads in average sentiment ({sent_best_score:.3f}), consistent with its strong "
        f"star rating performance. {sent_worst} has the highest share of negative-sentiment reviews "
        f"at {sent_worst_neg:.1f}%. Notably, the sentiment rankings largely mirror the star rating "
        f"rankings, reinforcing that the written feedback aligns with the numeric scores across builders."
    )

    # 6.5 trends over time
    section_header("6.5 Trends Over Time")
    explain(
        "This section tracks how each builder's review volume, average rating, and average sentiment "
        "have changed over time on a quarterly basis. Trends can reveal whether a builder is improving, "
        "declining, or holding steady, and whether changes in satisfaction coincide with shifts in "
        "review volume."
    )

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

    commentary(
        "Review volume peaked during the 2021–2022 housing boom for most builders and has declined "
        "since. Average ratings and sentiment show some quarter-to-quarter variability but remain "
        "relatively stable over time. Quarters with very few reviews can produce noisy averages, "
        "so the trends are most meaningful where volume is consistent."
    )

    # 6.6 Geographic footprint
    section_header("6.6 Geographic Footprint")
    explain(
        "Since Shea Homes operates in specific states, this section narrows the comparison to only "
        "those markets. Comparing review volume, average rating, and average sentiment by state "
        "shows how builders stack up in the geographies where Shea actually competes, rather than "
        "across their entire national footprint."
    )

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

    # find best builder per state for commentary
    geo_best_lines = []
    for state in shea_states:
        state_data = geo[geo["state"] == state]
        if not state_data.empty:
            best_in_state = state_data.loc[state_data["avg_rating"].idxmax()]
            geo_best_lines.append(f"{state}: {best_in_state['builder']} ({best_in_state['avg_rating']:.2f})")
    commentary(
        f"Across the {len(shea_states)} Shea markets, the competitive picture varies by state. "
        f"Highest-rated builder per state - {'; '.join(geo_best_lines)}. "
        f"Review volume also differs substantially: larger sample sizes in states like CA and AZ "
        f"produce more reliable comparisons, while states with fewer reviews should be interpreted cautiously."
    )

    # 6.7 distinctive language
    section_header("6.7 Distinctive Language by Builder")
    explain(
        "What do customers talk about that is unique to each builder? Using TF-IDF (Term Frequency–"
        "Inverse Document Frequency), this analysis identifies words and phrases that appear more "
        "frequently in one builder's reviews relative to the overall average. Builder name tokens are "
        "excluded so the results reflect substantive topics rather than brand mentions."
    )

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

    commentary(
        "The distinctive terms reveal what topics dominate each builder's customer conversation. "
        "Terms related to specific community features, construction elements, or service interactions "
        "highlight the aspects of the homebuying experience that customers associate most strongly "
        "with each brand. These language signals complement the numeric ratings by showing what "
        "customers are actually writing about."
    )

    # 6.8 Review Length & Engagement
    section_header("6.8 Review Length and Engagement")
    explain(
        "Review length can be a proxy for customer engagement. Longer reviews often indicate stronger "
        "feelings (positive or negative). This section compares the distribution of review word counts "
        "across builders and examines whether the relationship between star rating and review length "
        "differs by builder."
    )

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

    # compute engagement commentary
    avg_wc = all_df.groupby("builder")["word_count"].mean().sort_values(ascending=False)
    longest_b = avg_wc.index[0]
    longest_wc = avg_wc.iloc[0]
    shortest_b = avg_wc.index[-1]
    shortest_wc = avg_wc.iloc[-1]
    commentary(
        f"{longest_b} customers write the longest reviews on average ({longest_wc:.0f} words), "
        f"while {shortest_b} reviews are the shortest ({shortest_wc:.0f} words). Across all builders, "
        f"lower star ratings tend to produce longer reviews — dissatisfied customers invest more effort "
        f"in describing their experience. This pattern is consistent with the engagement dynamics "
        f"observed in Part 1 for Shea Homes alone."
    )

    # 6.9 Summary
    section_header("6.9 Summary")

    best_builder = summary.index[0]
    best_avg = summary.loc[best_builder, "avg_rating"]
    lowest_neg_b = sent_summary["pct_negative"].idxmin()

    commentary(
        f"Key takeaways:<br>"
        f"1. Overall positioning: {best_builder} leads with a {best_avg:.2f} average rating; "
        f"Lennar trails with the highest at-risk rate.<br>"
        f"2. Dimensional strengths: Each builder shows a distinct profile across quality, trust, "
        f"value, and responsiveness, no single builder dominates every dimension.<br>"
        f"3. Sentiment patterns: {lowest_neg_b} has the lowest share of negative reviews, "
        f"suggesting stronger language-level satisfaction even beyond star ratings.<br>"
        f"4. Geographic overlap: {len(shea_states)} Shea markets compared head-to-head "
        f"on ratings and sentiment.<br>"
        f"5. Language signals: TF-IDF distinctive terms reveal what topics and phrases "
        f"are unique to each builder's customer base.<br><br>"
        f"The full Shea Homes deep-dive (sentiment, topic modeling, predictive classification, "
        f"LLM analysis) is covered in Parts 1–5."
    )

    nav_buttons(page)
