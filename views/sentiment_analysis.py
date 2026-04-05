import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import (
    SHEA_BLUE, SHEA_GOLD, POS_GREEN, NEG_RED, NEU_YELLOW,
    SATISFIED_MIN_STARS, NEGATIVE_MAX_STARS, MIN_REVIEWS_STATE,
    VADER_NEG_THRESHOLD, MISMATCH_NEG_SENTIMENT, MISMATCH_POS_SENTIMENT,
)
from utils.components import section_header, explain, commentary, clean_fig, nav_buttons
from utils.data import get_stop_words, compute_ngrams, get_neg_distinctive


def render(df, fdf, page):
    st.title("Part 3: Preliminary Sentiment Analysis")
    explain("Sentiment analysis is a natural language processing technique that allows an algorithm to evaluate written feedback and estimate the emotional tone of the text. Instead of manually reading thousands of reviews, algorithms analyze the words used in each comment to determine whether the overall message is positive, negative, or neutral. Words associated with positive experiences (such as 'great' or 'helpful') increase the sentiment score, while words associated with problems or frustration decrease it. <br> <br> Each review is converted into a numerical sentiment score ranging from −1 to +1, where −1 represents very negative language and +1 represents very positive language. To increase reliability, this analysis uses two widely used sentiment models. VADER (Valence Aware Dictionary and sEntiment Reasoner) is designed specifically for social media and review text and accounts for emphasis such as capitalization, punctuation, and emotional wording. TextBlob is a general-purpose language model that evaluates sentiment based on the balance of positive and negative terms within the text. Using two independent methods allows the analysis to compare results and confirm that the patterns observed in customer sentiment are consistent.")

    # 3.1 overall sentiment breakdown
    section_header("3.1 Overall Sentiment Breakdown")
    explain("These charts summarize how each sentiment model classified the reviews. The pie charts show the percentage of reviews labeled positive, neutral, or negative by each tool. The chart on the right compares the average sentiment score with the star rating to verify that the models behave as expected. If the models are working correctly, reviews with higher star ratings should also show more positive sentiment.")
    c1,c2,c3 = st.columns(3)
    with c1:
        vc = fdf["vader_label"].value_counts()
        fig = px.pie(names=vc.index,values=vc.values,color=vc.index,color_discrete_map={"Positive":POS_GREEN,"Neutral":NEU_YELLOW,"Negative":NEG_RED},hole=0.4)
        fig.update_traces(textinfo="percent",textfont_size=13); fig.update_layout(title="VADER"); clean_fig(fig,340); st.plotly_chart(fig, use_container_width=True)
    with c2:
        tc = fdf["textblob_label"].value_counts()
        fig = px.pie(names=tc.index,values=tc.values,color=tc.index,color_discrete_map={"Positive":POS_GREEN,"Neutral":NEU_YELLOW,"Negative":NEG_RED},hole=0.4)
        fig.update_traces(textinfo="percent",textfont_size=13); fig.update_layout(title="TextBlob"); clean_fig(fig,340); st.plotly_chart(fig, use_container_width=True)
    with c3:
        ss = fdf.groupby("total_score")[["vader_compound","textblob_polarity"]].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ss["total_score"],y=ss["vader_compound"],mode="lines+markers",name="VADER",line=dict(color=SHEA_BLUE,width=3),marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=ss["total_score"],y=ss["textblob_polarity"],mode="lines+markers",name="TextBlob",line=dict(color=SHEA_GOLD,width=3,dash="dash"),marker=dict(size=10,symbol="square")))
        fig.add_hline(y=0,line_color="gray",opacity=0.3); fig.update_xaxes(title="Star Rating",dtick=1); fig.update_layout(title="Sentiment vs Stars"); clean_fig(fig,340); st.plotly_chart(fig, use_container_width=True)
    _vader_pos_pct = (fdf["vader_label"] == "Positive").mean()
    _tb_pos_pct = (fdf["textblob_label"] == "Positive").mean()
    commentary(f"Both models produce similar results, showing that the majority of reviews contain positive language. VADER classifies about {_vader_pos_pct:.0%} of reviews as positive, while TextBlob identifies roughly {_tb_pos_pct:.0%} as positive. VADER detects more negative reviews than TextBlob, reflecting its stronger sensitivity to negative wording in review-style text. <br> <br>The sentiment vs. star rating chart provides a validation check. Sentiment scores increase steadily from 1-star to 5-star reviews, confirming that both models are interpreting the language in a way that aligns with the rating customers assigned.")

    # 3.2 sentiment trends over time
    section_header("3.2 Sentiment Trends Over Time")
    explain("This section examines how customer sentiment has changed over time. Reviews are grouped by quarter to identify broader trends. The top chart shows the average sentiment score per quarter, while the bottom chart shows the percentage of reviews classified as positive and negative during each period.")
    qtr = fdf.groupby("quarter").agg(avg_v=("vader_compound","mean"),pct_pos=("vader_label",lambda x:(x=="Positive").mean()),pct_neg=("vader_label",lambda x:(x=="Negative").mean()),cnt=("total_score","count")).reset_index()
    qtr["qt"]=pd.to_datetime(qtr["quarter"]); qtr=qtr.sort_values("qt")
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.08,subplot_titles=["Avg VADER Sentiment","Positive vs Negative Share"])
    fig.add_trace(go.Scatter(x=qtr["qt"],y=qtr["avg_v"],mode="lines+markers",line=dict(color=SHEA_BLUE,width=2),fill="tozeroy",fillcolor="rgba(26,82,118,0.08)",name="Avg VADER"),row=1,col=1)
    fig.add_trace(go.Bar(x=qtr["qt"],y=qtr["pct_pos"]*100,name="% Positive",marker_color=POS_GREEN,opacity=0.7),row=2,col=1)
    fig.add_trace(go.Bar(x=qtr["qt"],y=-qtr["pct_neg"]*100,name="% Negative",marker_color=NEG_RED,opacity=0.7),row=2,col=1)
    clean_fig(fig,550); fig.update_yaxes(range=[0,1],row=1,col=1); fig.update_layout(title="Sentiment Trends Over Time"); st.plotly_chart(fig, use_container_width=True)
    commentary("Customer sentiment remains consistently positive throughout the period, with average sentiment scores generally staying between 0.4 and 0.7 on the VADER scale. The share of positive reviews typically ranges between 70% and 85%, while negative reviews remain a much smaller portion of the dataset. Overall, the charts show no major long-term decline or improvement in sentiment, suggesting that customer satisfaction has remained relatively stable over time.")

    # 3.3 sentiment by state
    section_header("3.3 Sentiment by State")
    explain("This chart compares customer sentiment across states using the VADER sentiment score. Each bar represents the average sentiment of review text within a state. Higher scores indicate that customers in that market tend to describe their experience using more positive language.")
    ss = fdf.groupby("state").agg(avg_v=("vader_compound","mean"),avg_s=("total_score","mean"),cnt=("total_score","count")).reset_index()
    ss = ss[ss["cnt"]>=MIN_REVIEWS_STATE].sort_values("avg_v"); ss["color"]=ss["avg_v"].apply(lambda v: POS_GREEN if v>=0.5 else (NEU_YELLOW if v>=0.3 else NEG_RED))
    fig = go.Figure(go.Bar(y=ss["state"],x=ss["avg_v"],orientation="h",marker_color=ss["color"],text=[f"{v:.3f} (avg {s:.1f}, n={c})" for v,s,c in zip(ss["avg_v"],ss["avg_s"],ss["cnt"])],textposition="outside"))
    fig.update_xaxes(range=[0,ss["avg_v"].max()+0.25]); fig.update_layout(title="Sentiment by State"); clean_fig(fig,max(380,len(ss)*50)); st.plotly_chart(fig, use_container_width=True)
    commentary("Most states show consistently positive sentiment, with scores clustering between 0.45 and 0.55. Colorado and Texas rank among the highest, showing positive review language in those markets. <br> <br> North Carolina and Idaho appear lower in comparison, suggesting that customer experiences in those markets may warrant closer examination. California and Arizona, the two markets with the largest number of reviews, remain positive, showing stable customer sentiment in Shea Homes' largest operating regions.")

    # 3.4 negative vs positive word frequency
    section_header("3.4 Negative vs Positive Word Frequency")
    explain("This analysis examines the most frequently used words in positive and negative reviews. The charts compare language used in 1–2 star reviews with language used in 4–5 star reviews. By analyzing which terms appear most often in each group, we can identify the themes that customers associate with positive experiences and the issues that appear most often in negative feedback.")
    sw = get_stop_words(); sw_list = list(sw)
    neg_t = fdf[fdf["total_score"]<=NEGATIVE_MAX_STARS]["review_text"]; pos_t = fdf[fdf["total_score"]>=SATISFIED_MIN_STARS]["review_text"]
    if len(neg_t)>5 and len(pos_t)>5:
        col1,col2 = st.columns(2)
        for col, texts, label, color in [(col1,neg_t,f"Negative (1-2 star, n={len(neg_t)})",NEG_RED),(col2,pos_t,f"Positive (4-5 star, n={len(pos_t)})",POS_GREEN)]:
            with col:
                bi = compute_ngrams(texts, sw_list, 1, 20)
                if bi:
                    w, c = zip(*bi[::-1])
                    fig = go.Figure(go.Bar(
                        y=list(w), x=list(c), orientation="h", marker_color=color,
                        text=list(c), textposition="outside"
                    ))
                    max_c = max(c)
                    fig.update_layout(title=label, margin=dict(l=40, r=80, t=40, b=40))
                    fig.update_xaxes(range=[0, max_c * 1.10])
                    clean_fig(fig, 500); st.plotly_chart(fig, use_container_width=True)
    commentary("Negative reviews frequently reference words such as issues, quality, warranty, time, and construction, indicating that dissatisfaction is often related to build quality or delays in resolving problems after purchase. In contrast, positive reviews emphasize words such as experience, team, process, and service, suggesting that customers frequently highlight interactions with staff and the overall buying process when describing a positive experience. This contrast suggests that customer-facing interactions are a key strength, while product quality and issue resolution appear more often in negative feedback.")

    # 3.5 distinctive negative review words
    section_header("3.5 Distinctive Negative Review Words")
    explain("This table identifies words that appear disproportionately often in negative reviews compared with positive reviews. Instead of simply counting the most common words, the analysis measures how much more frequently a word appears in complaints than in positive feedback. Words with high overrepresentation scores are strongly associated with dissatisfied customer experiences.")
    dist = []
    if len(neg_t)>5 and len(pos_t)>5:
        dist = get_neg_distinctive(neg_t, pos_t, sw_list)
        if dist: st.dataframe(pd.DataFrame(dist, columns=["Word","Count","Overrepresentation"]).assign(**{"Overrepresentation": lambda d: d["Overrepresentation"].map("{:.1f}x".format)}), use_container_width=True, hide_index=True)
    if dist and len(dist) >= 3:
        _top_words_str = ", ".join([f"{w} ({r:.1f}x)" for w, c, r in dist[:5]])
        commentary(f"Words like {_top_words_str} are disproportionately more common in negative reviews, pointing to specific construction and service pain points. These are not abstract complaints; they are about specific, fixable things that suggest delays and quality issues are a major theme.")
    else:
        commentary("The distinctive word analysis highlights specific terms that appear far more frequently in negative reviews compared to positive ones, pointing to concrete construction and service pain points.")

    # 3.6 score vs. sentiment mismatch
    section_header("3.6 Score vs. Sentiment Mismatch")
    explain("This section identifies reviews where the star rating and the language of the review do not align. In most cases, higher star ratings are associated with positive wording, while lower ratings contain more negative language. When these signals disagree, it can show more complexity in the customer experience that star ratings alone may not capture. <br> <br> One important category is high-star reviews with negative sentiment in the text. These reviews often show that a customer was generally satisfied but still experienced specific problems worth noting. Identifying these cases helps surface issues that may otherwise be overlooked when focusing only on low star ratings.")
    high_neg = fdf[(fdf["total_score"]>=SATISFIED_MIN_STARS)&(fdf["vader_compound"]<MISMATCH_NEG_SENTIMENT)]; low_pos = fdf[(fdf["total_score"]<=NEGATIVE_MAX_STARS)&(fdf["vader_compound"]>MISMATCH_POS_SENTIMENT)]
    c1,c2,c3 = st.columns(3); c1.metric("Total Mismatches",f"{fdf['mismatch'].sum()} ({fdf['mismatch'].mean():.1%})"); c2.metric("High Stars + Neg Text",f"{len(high_neg)}"); c3.metric("Low Stars + Pos Text",f"{len(low_pos)}")
    if len(high_neg)>0:
        st.markdown("**Hidden Complaints (samples):**")
        for _, r in high_neg.sort_values("vader_compound").head(3).iterrows():
            st.markdown(f"> **{r['total_score']} stars** | VADER: {r['vader_compound']:.3f} | {r['location']}  \n> _{str(r['review_text'])[:300]}..._")
    _mismatch_n = fdf["mismatch"].sum()
    _mismatch_pct = fdf["mismatch"].mean()
    commentary(f"A total of {_mismatch_n} reviews ({_mismatch_pct:.1%} of the dataset) show a mismatch between the rating and the sentiment expressed in the text. The majority of these cases, {len(high_neg)} reviews, are high-star ratings paired with negative language, indicating customers who reported specific issues despite assigning an overall positive score. These reviews are particularly valuable from an operational perspective. Because the customer still left a favorable rating, the relationship is largely intact, yet the written feedback highlights clear opportunities for improvement, such as subcontractor performance, installation quality, or post-closing service.")

    # 3.7 sample reviews by sentiment
    section_header("3.7 Sample Reviews by Sentiment")
    explain("To show how the sentiment models interpret review text, this section gives real examples from the dataset across different sentiment categories. These examples help translate the numerical sentiment scores into the type of language customers actually use when describing their experiences.")
    for label, sa in [("Positive",False),("Negative",True),("Neutral",None)]:
        sub = fdf[fdf["vader_label"]==label]
        if len(sub)==0: continue
        sub = sub.sort_values("vader_compound",ascending=sa).head(2) if sa is not None else sub.head(2)
        st.markdown(f"**Most {label} Reviews:**")
        for _, r in sub.iterrows():
            st.markdown(f"> **{r['total_score']} stars** | {r['location']} | {r['date'].strftime('%b %Y') if pd.notna(r['date']) else 'N/A'} | VADER: {r['vader_compound']:.3f}  \n> _{str(r['review_text'])[:350]}..._")
    commentary("Positive reviews typically contain enthusiastic language and references to specific employees, teams, or smooth buying experiences. Negative reviews, in contrast, tend to focus on specific construction defects, installation problems, or delays in resolving issues. Neutral reviews generally contain short or factual statements with little emotional language, which results in sentiment scores near zero. Reviewing these examples provides context for how the sentiment algorithms classify text and what each sentiment category represents in practice.")

    nav_buttons(page)
