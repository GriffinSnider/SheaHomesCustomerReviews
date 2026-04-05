import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import (
    SHEA_BLUE, SHEA_GOLD, POS_GREEN, NEG_RED, NEU_YELLOW, PALETTE_5,
    SATISFIED_MIN_STARS, MIN_REVIEWS_STATE, MIN_REVIEWS_COMMUNITY,
)
from utils.components import section_header, explain, commentary, clean_fig, nav_buttons


def render(df, fdf, page):
    st.title("Part 1: Summary Statistics")
    explain("Before applying modeling or artificial intelligence methods, the analysis begins with summary statistics. This step looks at the basic structure of the dataset, including the total number of reviews, how ratings are distributed across the 1–5 star scale, which states and cities generate the most feedback, and the typical length of customer reviews. Establishing these baseline patterns provides context for the more advanced analyses that follow.")

    # 1.1 dataset overview
    section_header("1.1 Dataset Overview")
    c1,c2,c3,c4 = st.columns(4); c1.metric("Total Reviews",f"{len(fdf):,}"); c2.metric("Total Words",f"{fdf['word_count'].sum():,}"); c3.metric("Avg Length",f"{fdf['word_count'].mean():.1f} words"); c4.metric("Unique Cities",f"{fdf['location'].nunique()}")
    c6,c7,c8 = st.columns(3); c6.metric("Median Length",f"{fdf['word_count'].median():.0f} words"); c7.metric("Shortest",f"{fdf['word_count'].min()} words"); c8.metric("Longest",f"{fdf['word_count'].max()} words")
    commentary(f"The dataset contains {len(fdf):,} reviews, totaling {fdf['word_count'].sum():,} words. Reviews average {fdf['word_count'].mean():.1f} words, but the median is just {fdf['word_count'].median():.0f} words, meaning most are brief while a smaller number contain detailed feedback. The reviews span {fdf['location'].nunique()} cities across {fdf['state'].nunique()} states. Length ranges from {fdf['word_count'].min()} words to {fdf['word_count'].max():,} words, capturing everything from quick ratings to detailed accounts of the homebuying experience.")

    # 1.2 star rating distribution
    section_header("1.2 Star Rating Distribution")
    explain("This section shows how customers rated their experience on a 1 to 5 star scale. The left chart counts how many reviews fell at each star level. The right chart shows the average score across five different rating categories that customers fill out: Overall, Quality, Trustworthiness, Value, and Responsiveness.")
    col1, col2 = st.columns(2)
    with col1:
        sc = fdf["total_score"].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=sc.index, y=sc.values, marker_color=PALETTE_5, text=[f"{v:,}\n({v/len(fdf):.0%})" for v in sc.values], textposition="outside"))
        fig.update_xaxes(title="Star Rating",dtick=1); fig.update_yaxes(title="Reviews"); fig.update_layout(title="Total Score Distribution"); clean_fig(fig,420); st.plotly_chart(fig, use_container_width=True)
    with col2:
        cmeans = [fdf[c].mean() for c in ["total_score","quality","trustworthiness","value","responsiveness"]]
        clabels = ["Overall","Quality","Trust","Value","Responsiveness"]
        fig = go.Figure(go.Bar(y=clabels[::-1], x=cmeans[::-1], orientation="h", marker_color=SHEA_BLUE, text=[f"{v:.2f}" for v in cmeans[::-1]], textposition="outside"))
        fig.update_xaxes(range=[0,5.5]); fig.update_layout(title="Average Score by Category"); clean_fig(fig,420); st.plotly_chart(fig, use_container_width=True)
    _star5_pct = (fdf["total_score"] == 5).mean()
    _star1_pct = (fdf["total_score"] == 1).mean()
    _trust_avg = fdf["trustworthiness"].mean()
    _qual_avg = fdf["quality"].mean()
    _val_avg = fdf["value"].mean()
    commentary(f"The distribution is heavily skewed with 5-star reviews making up {_star5_pct:.0%}, while 1-star reviews account for only {_star1_pct:.0%}. All five categories average above 4.0, with Trust scoring highest ({_trust_avg:.2f}) and Quality and Value slightly lower ({min(_qual_avg, _val_avg):.2f}). Value perceptions, specifically how customers weigh cost against what was delivered, may represent an area worth watching.")

    # 1.3 review volume over time
    section_header("1.3 Review Volume Over Time")
    explain("This chart tracks review activity and customer satisfaction over time. The blue bars show the number of reviews submitted each month, representing review volume. The gold line shows the three-month rolling average star rating, which smooths short-term fluctuations to highlight broader trends in customer satisfaction.")
    monthly = fdf.groupby("year_month").agg(count=("total_score","count"),avg=("total_score","mean")).reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"]); monthly = monthly.sort_values("year_month"); monthly["rolling"] = monthly["avg"].rolling(3,min_periods=1).mean()
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=monthly["year_month"],y=monthly["count"],name="Monthly Reviews",marker_color=SHEA_BLUE,opacity=0.6), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["year_month"],y=monthly["rolling"],name="3-Mo Avg Score",line=dict(color=SHEA_GOLD,width=3)), secondary_y=True)
    fig.update_yaxes(title_text="Count",secondary_y=False); fig.update_yaxes(title_text="Avg Score",range=[1,5.5],secondary_y=True)
    fig.add_hline(y=4.0,line_dash="dash",line_color="gray",opacity=0.3,secondary_y=True); fig.update_layout(title="Review Volume & Score Over Time"); clean_fig(fig,420); st.plotly_chart(fig, use_container_width=True)
    commentary("Review volume was highest during 2021–2022, when homebuilding activity was higher following the post-pandemic housing surge. Monthly review counts decline after this period as market activity slowed. Despite fluctuations in volume, the three-month average rating remains consistently above 4.0, showing stable customer satisfaction over time. A small dip appears in late 2025, which may warrant closer examination at the market or community level.")

    # 1.4 geographic breakdown
    section_header("1.4 Geographic Breakdown")
    explain("These charts examine the geographic distribution of customer feedback. The left chart shows the number of reviews submitted from each state, showing where the largest share of customer feedback originates. The right chart shows the average star rating by state, calculated only for states with at least 10 reviews.")
    col1, col2 = st.columns(2)
    with col1:
        stc = fdf["state"].value_counts().reset_index(); stc.columns=["state","count"]
        fig = px.bar(stc,y="state",x="count",orientation="h",color_discrete_sequence=[SHEA_BLUE],text="count"); fig.update_traces(textposition="inside"); fig.update_layout(title="Reviews by State"); clean_fig(fig,max(350,len(stc)*45)); st.plotly_chart(fig, use_container_width=True)
    with col2:
        sts = fdf.groupby("state")["total_score"].agg(["mean","count"]).reset_index(); sts = sts[sts["count"]>=MIN_REVIEWS_STATE].sort_values("mean")
        sts["color"] = sts["mean"].apply(lambda v: POS_GREEN if v>=SATISFIED_MIN_STARS else (NEU_YELLOW if v>=3 else NEG_RED))
        fig = go.Figure(go.Bar(y=sts["state"],x=sts["mean"],orientation="h",marker_color=sts["color"],text=[f"{v:.2f}" for v in sts["mean"]],textposition="inside"))
        fig.add_vline(x=4.0,line_dash="dash",line_color="gray",opacity=0.4); fig.update_xaxes(range=[0,5.5]); fig.update_layout(title="Avg Score by State (10+ reviews)"); clean_fig(fig,max(350,len(sts)*45)); st.plotly_chart(fig, use_container_width=True)
    _state_counts = fdf["state"].value_counts()
    _top1_st, _top1_n = _state_counts.index[0], _state_counts.iloc[0]
    _top2_st, _top2_n = _state_counts.index[1], _state_counts.iloc[1]
    _top2_pct = (_top1_n + _top2_n) / len(fdf)
    _state_names = {"CA": "California", "AZ": "Arizona", "CO": "Colorado", "NV": "Nevada", "TX": "Texas", "NC": "North Carolina", "WA": "Washington", "VA": "Virginia", "ID": "Idaho", "SC": "South Carolina", "FL": "Florida", "OR": "Oregon"}
    commentary(f"Most reviews originate from {_state_names.get(_top1_st, _top1_st)} ({_top1_n:,}) and {_state_names.get(_top2_st, _top2_st)} ({_top2_n:,}), which together account for roughly {_top2_pct:.0%} of the dataset. These states represent the largest share of Shea Homes customer feedback on NewHomeSource. Average ratings across most states remain above 4.0, indicating generally strong satisfaction across markets. A few states show slightly lower averages, which may reflect differences in local operations, project timelines, or customer expectations. States with very small sample sizes are excluded from the average score comparison to avoid misleading results.")

    # 1.5 top cities
    section_header("1.5 Top Cities")
    explain("This chart identifies the cities that generate the largest volume of customer reviews. Each bar represents the number of reviews submitted from a specific city.")
    col1, = st.columns(1)
    with col1:
        tc = fdf["location"].value_counts().head(15).reset_index(); tc.columns=["city","count"]
        fig = go.Figure(go.Bar(y=tc["city"][::-1],x=tc["count"][::-1],orientation="h",marker_color=SHEA_BLUE,text=tc["count"][::-1],textposition="outside"))
        fig.update_layout(title="Top 15 Cities Review Counts"); fig.update_xaxes(range=[0,tc["count"].max()*1.25]); clean_fig(fig,500); st.plotly_chart(fig, use_container_width=True)
    commentary("Customer feedback is concentrated in a small number of communities. Cities such as Wickenburg, AZ; Las Vegas, NV; Rio Verde, AZ; and San Tan Valley, AZ contribute the highest number of reviews. These locations represent the areas where the dataset contains the most direct customer experience information.")

    # 1.6 rating correlations
    section_header("1.6 Rating Correlations")
    explain("This heatmap shows the correlation between the five rating categories: Overall, Quality, Trust, Value, and Responsiveness. Correlation measures how closely two variables move together. Values closer to 1.0 indicate a strong relationship, meaning customers tend to rate those categories similarly.")
    scols = ["total_score","quality","trustworthiness","value","responsiveness"]; slabs = ["Overall","Quality","Trust","Value","Responsiveness"]
    fig = px.imshow(fdf[scols].corr().values, x=slabs, y=slabs, color_continuous_scale="YlOrRd", zmin=0.5, zmax=1, text_auto=".2f")
    fig.update_layout(title="Rating Category Correlations"); clean_fig(fig,450); st.plotly_chart(fig, use_container_width=True)
    commentary("All categories are highly correlated, showing that customers who rate one dimension poorly tend to rate everything poorly. This suggests the overall experience is somewhat holistic: a bad construction experience drags down trust, value, and responsiveness perceptions too.")

    # 1.7 geographic deep dive
    section_header("1.7 Geographic Deep Dive", "Choropleth map and per-dimension ratings by state")
    explain("The earlier geographic charts showed review volume and average overall ratings. This section maps satisfaction across the country and breaks scores down by rating dimension (Quality, Trustworthiness, Value, Responsiveness) to reveal which markets underperform on specific operational areas. States with fewer than 10 reviews are excluded.")

    geo = fdf.groupby("state").agg(
        n=("total_score","count"),
        overall=("total_score","mean"),
        quality=("quality","mean"),
        trust=("trustworthiness","mean"),
        value=("value","mean"),
        resp=("responsiveness","mean"),
    ).reset_index()
    geo = geo[geo["n"] >= MIN_REVIEWS_STATE].copy()
    geo["hover"] = geo.apply(lambda r: (
        f"{r['state']} ({r['n']:.0f} reviews)<br>"
        f"Overall: {r['overall']:.2f}<br>"
        f"Quality: {r['quality']:.2f}<br>"
        f"Trust: {r['trust']:.2f}<br>"
        f"Value: {r['value']:.2f}<br>"
        f"Responsiveness: {r['resp']:.2f}"
    ), axis=1)

    fig = go.Figure(go.Choropleth(
        locations=geo["state"],
        locationmode="USA-states",
        z=geo["overall"],
        zmin=3.5, zmax=4.6,
        colorscale=[[0, NEG_RED], [0.5, NEU_YELLOW], [1, POS_GREEN]],
        colorbar_title="Avg Rating",
        text=geo["hover"],
        hoverinfo="text",
    ))
    fig.update_layout(
        geo=dict(scope="usa", bgcolor="white", lakecolor="white"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_layout(title="Average Overall Rating by State"); clean_fig(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    dims = ["quality", "trust", "value", "resp"]
    dim_labels = ["Quality", "Trustworthiness", "Value", "Responsiveness"]
    dim_colors = [SHEA_BLUE, SHEA_GOLD, POS_GREEN, NEG_RED]
    geo_sorted = geo.sort_values("overall")

    fig = go.Figure()
    for dim, label, color in zip(dims, dim_labels, dim_colors):
        fig.add_trace(go.Bar(
            name=label,
            y=geo_sorted["state"],
            x=geo_sorted[dim],
            orientation="h",
            marker_color=color,
            text=[f"{v:.2f}" for v in geo_sorted[dim]],
            textposition="outside",
        ))
    fig.add_vline(x=4.0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.update_xaxes(range=[3.0, 5.0])
    fig.update_layout(barmode="group", title="Rating Dimensions by State (10+ reviews)")
    clean_fig(fig, max(420, len(geo_sorted) * 55))
    st.plotly_chart(fig, use_container_width=True)

    geo_dims = geo_sorted.copy()
    dim_map = {"quality": "Quality", "trust": "Trustworthiness", "value": "Value", "resp": "Responsiveness"}
    geo_dims["weakest_dim"] = geo_dims[dims].idxmin(axis=1).map(dim_map)
    geo_dims["weakest_score"] = geo_dims[dims].min(axis=1)
    weak_display = geo_dims[["state", "n", "overall", "weakest_dim", "weakest_score"]].copy()
    weak_display.columns = ["State", "Reviews", "Overall Avg", "Weakest Dimension", "Weakest Score"]
    weak_display["Overall Avg"] = weak_display["Overall Avg"].map("{:.2f}".format)
    weak_display["Weakest Score"] = weak_display["Weakest Score"].map("{:.2f}".format)
    st.dataframe(weak_display.sort_values("Weakest Score"), use_container_width=True, hide_index=True)

    _resp_best = geo_sorted.iloc[-1]
    _resp_worst = geo_sorted.iloc[0]
    _weakest_dim_mode = geo_dims["weakest_dim"].mode().iloc[0] if not geo_dims.empty else "Value"
    commentary(f"The map and dimension breakdown reveal that satisfaction is not uniform across markets. Responsiveness varies the most between states: {_resp_best['state']} leads at {_resp_best['resp']:.2f} while {_resp_worst['state']} trails at {_resp_worst['resp']:.2f}, a gap of {_resp_best['resp'] - _resp_worst['resp']:.2f} points. {_weakest_dim_mode} is the weakest dimension in the majority of markets, suggesting it may be a company-wide opportunity rather than a regional issue. These patterns can help prioritize where operational improvements would have the most impact.")

    # 1.8 community leaderboard
    section_header("1.8 Community Leaderboard", "Ranking Shea communities by satisfaction and at-risk rate")
    explain("This leaderboard ranks every Shea Homes community (city) by average rating and percentage of at-risk reviews (1–3 stars). Communities with fewer than 5 reviews are excluded to avoid small-sample noise. Use the toggles to sort by different metrics and surface geographic outliers.")

    MIN_REVIEWS_LB = MIN_REVIEWS_COMMUNITY
    comm = fdf.groupby("location").agg(
        reviews=("total_score", "size"),
        avg_rating=("total_score", "mean"),
        avg_sentiment=("vader_compound", "mean"),
        pct_at_risk=("total_score", lambda x: (x < SATISFIED_MIN_STARS).mean() * 100),
        pct_5_star=("total_score", lambda x: (x == 5).mean() * 100),
        avg_quality=("quality", "mean"),
        avg_value=("value", "mean"),
        avg_resp=("responsiveness", "mean"),
    ).reset_index()
    comm = comm[comm["reviews"] >= MIN_REVIEWS_LB].copy()
    comm.columns = ["Community", "Reviews", "Avg Rating", "Avg Sentiment", "% At-Risk", "% 5-Star", "Avg Quality", "Avg Value", "Avg Responsiveness"]

    lb_col1, lb_col2 = st.columns(2)
    with lb_col1:
        lb_sort = st.selectbox("Sort by", ["Avg Rating", "% At-Risk", "Avg Sentiment", "% 5-Star", "Reviews"], index=0)
    with lb_col2:
        lb_order = st.radio("Order", ["Best first", "Worst first"], horizontal=True)

    ascending = lb_order == "Worst first" if lb_sort != "% At-Risk" else lb_order == "Best first"
    comm_sorted = comm.sort_values(lb_sort, ascending=ascending)

    comm_display = comm_sorted.copy()
    for col in ["Avg Rating", "Avg Sentiment", "Avg Quality", "Avg Value", "Avg Responsiveness"]:
        comm_display[col] = comm_display[col].map("{:.2f}".format)
    for col in ["% At-Risk", "% 5-Star"]:
        comm_display[col] = comm_display[col].map("{:.1f}%".format)

    st.dataframe(comm_display, use_container_width=True, hide_index=True, height=460)

    if len(comm_sorted) >= 10:
        top5 = comm_sorted.head(5)
        bot5 = comm_sorted.tail(5)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top5["Community"], x=top5["Avg Rating"].astype(float) if top5["Avg Rating"].dtype == object else top5["Avg Rating"],
                orientation="h", marker_color=POS_GREEN, text=[f"{v:.2f}" for v in comm_sorted.head(5)["Avg Rating"]], textposition="outside"
            ))
            fig.update_xaxes(range=[0, 5.3])
            fig.update_layout(title=f"Top 5 Communities by {lb_sort}")
            clean_fig(fig, 300); st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=bot5["Community"], x=bot5["Avg Rating"].astype(float) if bot5["Avg Rating"].dtype == object else bot5["Avg Rating"],
                orientation="h", marker_color=NEG_RED, text=[f"{v:.2f}" for v in comm_sorted.tail(5)["Avg Rating"]], textposition="outside"
            ))
            fig.update_xaxes(range=[0, 5.3])
            fig.update_layout(title=f"Bottom 5 Communities by {lb_sort}")
            clean_fig(fig, 300); st.plotly_chart(fig, use_container_width=True)

    _lb_best = comm_sorted.iloc[0]
    _lb_worst = comm_sorted.iloc[-1]
    _high_risk = comm[comm["% At-Risk"] > 30]
    _risk_note = f" {len(_high_risk)} communities have an at-risk rate above 30%." if len(_high_risk) > 0 else " No community has an at-risk rate above 30%."
    commentary(
        f"Among the {len(comm)} communities with {MIN_REVIEWS_LB}+ reviews, "
        f"<b>{_lb_best['Community']}</b> leads with an avg rating of {float(_lb_best['Avg Rating']):.2f} "
        f"({int(_lb_best['Reviews'])} reviews), while <b>{_lb_worst['Community']}</b> trails at "
        f"{float(_lb_worst['Avg Rating']):.2f} ({int(_lb_worst['Reviews'])} reviews).{_risk_note} "
        f"Regional managers can use this ranking to identify which communities may need operational attention and which represent best-practice models."
    )

    nav_buttons(page)
