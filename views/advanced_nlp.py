import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go

from utils.config import SHEA_BLUE, SHEA_GOLD, POS_GREEN, NEG_RED, NEU_YELLOW, NEGATIVE_MAX_STARS, MIN_EMPLOYEE_MENTIONS
from utils.components import section_header, explain, commentary, finding, clean_fig, nav_buttons
from utils.data import compute_topics, compute_aspects, compute_employees, get_stop_words, compute_ngrams


def render(df, fdf, page):
    st.title("Part 4: Advanced Natural Language Processing Analysis")
    explain("This section applies advanced natural language processing (NLP) techniques to find insights from customer reviews. Rather than only measuring whether reviews are positive or negative, these methods analyze the text to identify recurring themes, frequently discussed topics, and patterns in how customers describe their experiences. <br> <br> The analysis also looks for which aspects of the homebuying experience customers mention most often, such as sales interactions, construction quality, or warranty service. By analyzing groups of words and phrases together, these techniques show more detailed patterns in customer feedback and help find the specific areas of the business that drive satisfaction or dissatisfaction.")

    # 4.1 topic discovery
    section_header("4.1 Topic Discovery (Latent Dirichlet Allocation (LDA))")
    explain("This section uses a technique called topic modeling to automatically identify the main subjects customers discuss in their reviews. The specific method used, Latent Dirichlet Allocation (LDA), analyzes patterns of words that frequently appear together across the dataset. Reviews that contain similar groups of words are grouped into a shared topic, allowing the model to discover common themes without being manually labeled in advance. <br> <br> By applying this approach to all reviews, the model identifies the major themes in customer feedback, such as construction quality, the buying process, sales interactions, and post-purchase issues. This provides a structured overview of what customers talk about most often.")
    tnames, tconf, tkws, tname_map = compute_topics(df["review_text"]); df = df.copy(); df["topic_name"]=tnames; fdf_t=df.copy()
    col1,col2 = st.columns(2)
    with col1:
        tcounts = fdf_t["topic_name"].value_counts().reset_index(); tcounts.columns=["topic","count"]
        fig = px.bar(tcounts,y="topic",x="count",orientation="h",color_discrete_sequence=[SHEA_BLUE],text=[f"{c} ({c/len(fdf_t):.0%})" for c in tcounts["count"]])
        fig.update_traces(textposition="outside"); fig.update_xaxes(range=[0,tcounts["count"].max()*1.25]); fig.update_layout(title="Review Volume by Topic"); clean_fig(fig,420); st.plotly_chart(fig, use_container_width=True)
    with col2:
        ts = fdf_t.groupby("topic_name")["total_score"].mean().sort_values().reset_index(); ts.columns=["topic","avg"]
        ts["color"]=ts["avg"].apply(lambda v: POS_GREEN if v>=4 else (NEU_YELLOW if v>=3 else NEG_RED))
        fig = go.Figure(go.Bar(y=ts["topic"],x=ts["avg"],orientation="h",marker_color=ts["color"],text=[f"{v:.2f}" for v in ts["avg"]],textposition="outside"))
        fig.add_vline(x=4.0,line_dash="dash",line_color="gray",opacity=0.4); fig.update_xaxes(range=[0,5.5]); fig.update_layout(title="Satisfaction by Topic"); clean_fig(fig,420); st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Discovered Topic Keywords:**")
    for num, kws in enumerate(tkws, 1):
        st.markdown(f"- **{tname_map.get(num, f'Topic {num}')}:** {', '.join(kws[:8])}")
    commentary("The algorithm discovered 6 distinct topics without any guidance. Construction quality appears most frequently, representing the largest share of customer discussion. Topics related to the buying process and sales experience also appear often and tend to receive the highest satisfaction scores. In contrast, reviews grouped under issues and problems show noticeably lower satisfaction levels. This pattern suggests that while the sales and purchasing experience is generally well received, dissatisfaction is more commonly associated with construction defects or post-delivery service issues.")

    # 4.2 aspect-based sentiment
    section_header("4.2 Aspect-Based Sentiment")
    explain("Aspect-based sentiment analysis examines how customers feel about specific parts of the homebuying experience. Instead of assigning a single sentiment score to an entire review, this method identifies mentions of key business areas (sales, construction quality, warranty service, communication, design, and pricing) and measures sentiment for each one separately. <br> <br> This approach is useful because customers often discuss multiple aspects within the same review. For example, a customer may write one sentence praising the sales team and another complaining about drywall cracks. Aspect-based sentiment analysis evaluates each of these mentions separately, scoring the sales interaction as positive and the construction issue as negative rather than averaging the entire review into a single score.")
    asp = compute_aspects(fdf["review_text"]); adf = pd.DataFrame(asp).T.sort_values("avg_sentiment")
    col1,col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(y=adf.index,x=adf["mentions"],orientation="h",marker_color=SHEA_BLUE,text=[f"{m:,.0f} ({p:.0%})" for m,p in zip(adf["mentions"],adf["pct"])],textposition="outside"))
        fig.update_xaxes(range=[0,adf["mentions"].max()*1.3]); fig.update_layout(title="What Customers Talk About"); clean_fig(fig,400); st.plotly_chart(fig, use_container_width=True)
    with col2:
        colors = [POS_GREEN if v>=0.3 else (NEU_YELLOW if v>=0.1 else NEG_RED) for v in adf["avg_sentiment"]]
        fig = go.Figure(go.Bar(y=adf.index,x=adf["avg_sentiment"],orientation="h",marker_color=colors,text=[f"{v:+.3f}" for v in adf["avg_sentiment"]],textposition="inside"))
        fig.add_vline(x=0,line_color="black",line_width=1); fig.update_layout(title="Sentiment by Aspect"); clean_fig(fig,400); st.plotly_chart(fig, use_container_width=True)
    w=adf["avg_sentiment"].idxmin(); s=adf["avg_sentiment"].idxmax()
    finding(f"<b>Strongest aspect:</b> {s} ({adf.loc[s,'avg_sentiment']:+.3f}) &nbsp;|&nbsp; <b>Weakest aspect:</b> {w} ({adf.loc[w,'avg_sentiment']:+.3f})")
    commentary("Construction quality is the most frequently discussed topic, appearing in more than half of all reviews. However, it also shows lower sentiment scores compared with other aspects, indicating that many customer concerns are tied to the physical build or post-construction issues. In contrast, sales interactions and communication receive the strongest sentiment scores, suggesting that customers generally view the front-end buying experience positively. Warranty and post-move service shows the weakest sentiment, highlighting it as an area where improvements could have a meaningful impact on overall customer satisfaction.")

    # 4.3 employee recognition mining
    section_header("4.3 Employee Recognition Mining")
    explain(f"Customers often mention Shea team members by name in their reviews. This section scans through all {len(fdf):,} reviews, looks for names, and then reads the sentences around each name to figure out whether the customer was saying something positive or negative about that person. This is useful for identifying top performers who consistently get praised, or for spotting cases where a specific team member keeps coming up in negative contexts.")
    emp = compute_employees(fdf["review_text"].tolist(),fdf["total_score"].tolist(),fdf["location"].tolist(),fdf["state"].tolist())
    if not emp.empty:
        ep = emp.head(20).sort_values("avg_sentiment")
        fig = go.Figure(go.Bar(y=ep.index,x=ep["avg_sentiment"],orientation="h",marker_color=[POS_GREEN if v>=0.3 else (NEU_YELLOW if v>=0 else NEG_RED) for v in ep["avg_sentiment"]],text=[f"{s:+.2f} ({m:.0f} mentions)" for s,m in zip(ep["avg_sentiment"],ep["mentions"])],textposition="outside"))
        fig.update_xaxes(range=[0, max(ep["avg_sentiment"]) * 1.14]); fig.add_vline(x=0,line_color="black",line_width=1); fig.update_layout(title="Employee Sentiment (5+ mentions)"); clean_fig(fig,max(600,len(ep)*32)); st.plotly_chart(fig, use_container_width=True)
        d=emp.head(25).copy(); d["avg_sentiment"]=d["avg_sentiment"].map("{:+.3f}".format); d["avg_stars"]=d["avg_stars"].map("{:.1f}".format); d.columns=["Mentions","Avg Sentiment","Avg Stars","Primary Location"]
        st.dataframe(d, use_container_width=True); st.caption(f"Note: Heuristic name extraction. May include false positives. {MIN_EMPLOYEE_MENTIONS}+ mention threshold reduces noise.")
    if not emp.empty:
        _top3_emp = emp.head(3)
        _emp_mentions = [f"{name} in {row['top_location']}" for name, row in _top3_emp.iterrows()]
        _emp_str = ", ".join(_emp_mentions[:-1]) + ", and " + _emp_mentions[-1] if len(_emp_mentions) > 2 else " and ".join(_emp_mentions)
        commentary(f"{_emp_str} are the most frequently mentioned employees, and all of them carry positive sentiment scores. That means when customers mention these people, they are saying good things. This kind of data could feed directly into recognition programs or performance reviews. It could also flag cases where a specific person keeps showing up in negative reviews, which would be an early signal for coaching or support.")
    else:
        commentary("Employee name extraction did not find enough mentions meeting the 5+ threshold. This may indicate the dataset has fewer explicit name references than expected.")

    # 4.4 common phrases (n-grams)
    section_header("4.4 Common Phrases (N-grams)", "Two and three word phrases in negative vs positive reviews")
    explain("Earlier analysis looked at individual words, but meaning often comes from combinations of words. For example, the phrase 'not responsive' carries the opposite meaning of the single word 'responsive,' even though the word itself is positive. <br> <br> This section analyzes common two-word and three-word phrases that appear frequently in positive and negative reviews. Looking at phrases instead of individual words provides a clearer picture of what customers are actually describing in their experiences.")
    sw_list = list(get_stop_words()); neg_t = fdf[fdf["total_score"]<=NEGATIVE_MAX_STARS]["review_text"]; pos_t = fdf[fdf["total_score"]==5]["review_text"]
    if len(neg_t)>=5 and len(pos_t)>=5:
        col1,col2 = st.columns(2)
        for col,texts,label,color in [(col1,neg_t,"Negative (1-2 star)",NEG_RED),(col2,pos_t,"Positive (5 star)",POS_GREEN)]:
            with col:
                for nv,nl in [(2,"Bigrams"),(3,"Trigrams")]:
                    ngr = compute_ngrams(texts,sw_list,nv,12)
                    if ngr:
                        w,c = zip(*ngr[::-1]); fig = go.Figure(go.Bar(y=list(w),x=list(c),orientation="h",marker_color=color,text=list(c),textposition="outside"))
                        fig.update_layout(title=f"{label}: {nl}"); clean_fig(fig,380); st.plotly_chart(fig, use_container_width=True)
    commentary("The negative phrases highlight specific operational pain points. Terms related to customer service, build quality, and warranty support appear frequently in lower-rated reviews, indicating that dissatisfaction often centers around post-purchase issues or construction defects. <br> <br> In contrast, the most common positive phrases focus on the sales team, the buying experience, and the overall building process. Phrases like 'great experience,' 'customer service,' and 'start to finish' appear frequently in 5-star reviews, suggesting that customers are especially satisfied when the entire homebuying journey feels smooth and well managed from beginning to end.")

    nav_buttons(page)
