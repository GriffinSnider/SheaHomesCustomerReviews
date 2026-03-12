import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import streamlit.components.v1 as components
from streamlit_scroll_to_top import scroll_to_here
import re
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Shea Homes Customer Reviews Project", page_icon="", layout="wide", initial_sidebar_state="expanded")

SHEA_BLUE = "#1a5276"
SHEA_GOLD = "#d4a843"
SHEA_DARK = "#0e2f44"
POS_GREEN = "#27ae60"
NEU_YELLOW = "#f1c40f"
NEG_RED = "#c0392b"
NEUTRAL_GRAY = "#85929e"
PALETTE_5 = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#1a5276"]

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Source+Serif+4:wght@400;600;700&display=swap');
    .stApp { background-color: #f7f8fa; }
    .block-container { padding-top: 1rem !important; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; color: #0e2f44; }
    h1 { font-size: 2.2rem !important; border-bottom: 3px solid #d4a843; padding-bottom: 0.4rem; margin-bottom: 1rem !important; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e5e8ec; border-radius: 10px; padding: 16px 20px 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
    div[data-testid="stMetric"] label { font-family: 'DM Sans', sans-serif !important; font-size: 12px !important; color: #6b7a8d !important; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: 'Source Serif 4', serif !important; font-size: 28px !important; color: #0e2f44 !important; }
    section[data-testid="stSidebar"] { background-color: #0e2f44; }
    section[data-testid="stSidebar"] * { color: #d6eaf8 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(214,234,248,0.15) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid #e5e8ec; }
    .stTabs [data-baseweb="tab"] { font-family: 'DM Sans'; font-weight: 500; padding: 10px 22px; color: #6b7a8d; }
    .stTabs [aria-selected="true"] { color: #1a5276 !important; border-bottom: 3px solid #d4a843 !important; }
    .section-header { background: linear-gradient(135deg, #0e2f44 0%, #1a5276 100%); color: white !important; padding: 1.2rem 1.8rem; border-radius: 10px; margin: 2rem 0 1.2rem 0; box-shadow: 0 2px 8px rgba(14,47,68,0.15); }
    .section-header h2 { color: white !important; margin: 0 !important; font-size: 1.5rem !important; border: none !important; padding: 0 !important; }
    .section-header p { color: #d6eaf8 !important; margin: 0.3rem 0 0 0 !important; font-size: 0.95rem; opacity: 0.9; }
    .explain-box { background: white; border-left: 4px solid #d4a843; padding: 1rem 1.4rem; border-radius: 0 8px 8px 0; margin: 0.8rem 0 1.2rem 0; font-size: 0.92rem; line-height: 1.6; color: #2c3e50; }
    .commentary-box { background: #eef4f9; border-left: 4px solid #1a5276; padding: 1rem 1.4rem; border-radius: 0 8px 8px 0; margin: 1.2rem 0 1.8rem 0; font-size: 0.92rem; line-height: 1.7; color: #1c2e3d; }
    .commentary-box b { color: #0e2f44; }
    .static-output { background: #1a1a2e; color: #e0e0e0; padding: 1.2rem 1.5rem; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.82rem; line-height: 1.5; overflow-x: auto; white-space: pre-wrap; border: 1px solid #2d2d44; margin: 0.5rem 0; }
    .llm-card { background: white; border: 1px solid #e5e8ec; border-radius: 10px; padding: 1.4rem 1.6rem; margin: 1rem 0; box-shadow: 0 1px 6px rgba(0,0,0,0.05); }
    .llm-card h4 { color: #1a5276; margin-top: 0; }
    .finding { background: linear-gradient(135deg, #fef9e7 0%, #fdebd0 100%); border-left: 4px solid #d4a843; padding: 1rem 1.4rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
    hr { border: none; border-top: 2px solid #e5e8ec; margin: 2.5rem 0; }
</style>
""", unsafe_allow_html=True)

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
    fig.update_layout(font_family="DM Sans", plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=20,r=20,t=50,b=20), height=height, title_font_size=15, title_font_color=SHEA_DARK)
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0"); fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig

# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading 2,039 reviews...")
def load_and_process(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["word_count"] = df["review_text"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["review_text"].apply(lambda x: len(str(x)))
    df["state"] = df["location"].str.extract(r",\s*([A-Z]{2})$")
    df["year"] = df["date"].dt.year
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    import nltk; nltk.download("vader_lexicon", quiet=True); nltk.download("stopwords", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer; from textblob import TextBlob
    sia = SentimentIntensityAnalyzer()
    scores = df["review_text"].apply(lambda x: sia.polarity_scores(str(x)))
    df["vader_compound"] = scores.apply(lambda x: x["compound"])
    df["vader_pos"] = scores.apply(lambda x: x["pos"])
    df["vader_neg"] = scores.apply(lambda x: x["neg"])
    df["vader_neu"] = scores.apply(lambda x: x["neu"])
    df["vader_label"] = df["vader_compound"].apply(lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral"))
    df["textblob_polarity"] = df["review_text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["textblob_subjectivity"] = df["review_text"].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    df["textblob_label"] = df["textblob_polarity"].apply(lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral"))
    df["risk_class"] = df["total_score"].apply(lambda x: "Satisfied (4-5)" if x >= 4 else "At-Risk (1-3)")
    df["mismatch"] = ((df["total_score"]>=4)&(df["vader_compound"]<-0.05)) | ((df["total_score"]<=2)&(df["vader_compound"]>0.5))
    return df

@st.cache_data
def get_stop_words():
    import nltk; nltk.download("stopwords", quiet=True); from nltk.corpus import stopwords
    sw = set(stopwords.words("english"))
    sw.update(["home","shea","homes","new","would","one","us","also","get","got","even","like","really","much","could","said","told","went","going","still","back","made","make","well","since","every"])
    return sw

@st.cache_data
def compute_topics(texts, n_topics=6):
    from sklearn.feature_extraction.text import CountVectorizer; from sklearn.decomposition import LatentDirichletAllocation
    sw = get_stop_words()
    vec = CountVectorizer(max_features=2000, stop_words=list(sw), min_df=5, max_df=0.7, ngram_range=(1,2))
    dtm = vec.fit_transform(texts.astype(str)); fnames = vec.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=25, learning_method="online"); lda.fit(dtm)
    tkws = [[fnames[i] for i in t.argsort()[-10:][::-1]] for t in lda.components_]
    hints = {"sales":"Sales Experience","warranty":"Warranty & Post-Purchase","quality":"Construction Quality","community":"Community & Lifestyle","process":"Buying Process","team":"Team & Communication","design":"Design & Features","price":"Value & Pricing","construction":"Construction Quality","move":"Move-In Experience","closing":"Closing Process","issues":"Issues & Problems","response":"Responsiveness"}
    tnames = {}
    for num, kws in enumerate(tkws, 1):
        tnames[num] = f"Topic {num}"
        for kw, lab in hints.items():
            if any(kw in k for k in kws[:6]): tnames[num] = lab; break
    dists = lda.transform(dtm)
    return [tnames[d+1] for d in dists.argmax(axis=1)], dists.max(axis=1).tolist(), tkws, tnames

@st.cache_data
def compute_aspects(texts):
    import nltk; nltk.download("vader_lexicon", quiet=True); from nltk.sentiment.vader import SentimentIntensityAnalyzer; sia = SentimentIntensityAnalyzer()
    ASPECTS = {"Sales & Buying Process":["sales","buying","purchase","contract","closing","escrow","realtor","agent","sales rep","sales team","deposit","financing","mortgage","loan","interest rate"],"Construction Quality":["quality","construction","build","built","craftsmanship","materials","drywall","paint","flooring","foundation","plumbing","electrical","roof","windows","doors","cabinets","concrete","cracks","defects","defect","workmanship"],"Communication":["communication","responsive","response","respond","answered","call","email","phone","update","informed","transparent","follow up","follow-up","reached out","timely","ignored","never called","no response"],"Warranty & Post-Move":["warranty","repair","fix","fixed","issue","issues","problem","problems","maintenance","service request","punch list","walk-through","walkthrough","inspection"],"Design & Floor Plan":["design","floor plan","layout","floorplan","model","upgrade","upgrades","options","features","kitchen","bathroom","bedroom","backyard","garage","space","spacious","open concept","modern","finishes"],"Value & Pricing":["value","price","pricing","cost","expensive","affordable","worth","money","overpriced","budget","hoa","fees","investment","deal","incentive"]}
    results = {}
    for asp, kws in ASPECTS.items():
        n = 0; sents = []
        for text in texts:
            tl = str(text).lower(); matched = [s.strip() for s in re.split(r"[.!?]+", tl) if any(k in s for k in kws)]
            if matched: n += 1; sents.append(sia.polarity_scores(". ".join(matched))["compound"])
        results[asp] = {"mentions":n, "pct":n/len(texts) if len(texts) else 0, "avg_sentiment":np.mean(sents) if sents else 0, "pct_negative":np.mean([s<-0.05 for s in sents]) if sents else 0}
    return results

@st.cache_data
def compute_employees(texts, scores, locations, states):
    import nltk; nltk.download("vader_lexicon", quiet=True); from nltk.sentiment.vader import SentimentIntensityAnalyzer; sia = SentimentIntensityAnalyzer()
    NOT_NAMES = {"The","We","Our","They","He","She","It","My","Very","This","That","From","Would","Every","And","But","For","Not","All","Are","Was","Has","Had","Been","Will","Can","Just","Great","Good","Best","Much","Only","After","With","When","What","How","Which","Their","There","Here","Also","Even","More","Most","Some","Any","Other","Each","Into","Over","About","Through","During","Before","Below","Between","Both","Never","Excellent","Amazing","Outstanding","Quality","Home","Homes","Shea","Everything","Beautiful","Whole","Highly","Professional","Building","Really","First","New","Love","Thank","Loved","Entire","Process","Experience","Always","Team","Customer","Service","Construction","Warranty","Sales","Buying","House","Community","Floor","Plan","Recommend","Well","Working","Year","Time","Day","Way","Lot","Overall","Trilogy","Made","Make","Keep","Know","Come","Take","Give","Look","Help","Work","Need","Feel","Call","Move","Try","Start","Still","Covid","HOA","Design","Manager","Super","Then","Now","Once","Since","While","These","Many","Poor","Field","Thanks","Mortgage","Builder","One","You","However","Encanterra","Mesa","Phoenix","Denver","Vegas","Livermore","Tracy","Being","Pleased","Awesome","Wonderful","Especially","Told","Going","Done","Took","Did","Though","Absolutely","Definitely","Nothing","Something","Another","Little","Positive","Negative","Impressed","Terrible","Horrible","Worst","Fabulous","Perfect","Quick","Responsive","Happy","Smooth","Several","Few","Its","Sheas","Ive","Weve","Communication","Were","Everyone","Ranch","Please","Although","Center","Having","Shae","Despite","Dont","Unfortunately","Your","His","Lots","Wickenburg","Ill","Follow","Dunes","Thats","Multiple","Project","Post","Lake"}
    recs = []
    for text, sc, loc, st_ in zip(texts, scores, locations, states):
        words = str(text).split()
        for i, w in enumerate(words):
            c = re.sub(r"[^A-Za-z]", "", w)
            if c and c[0].isupper() and len(c)>=3 and c not in NOT_NAMES and c.isalpha() and not c.isupper():
                ctx = " ".join(words[max(0,i-5):min(len(words),i+6)])
                recs.append({"name":c,"sentiment":sia.polarity_scores(ctx)["compound"],"total_score":sc,"location":loc,"state":st_})
    if not recs: return pd.DataFrame()
    edf = pd.DataFrame(recs)
    s = edf.groupby("name").agg(mentions=("sentiment","count"),avg_sentiment=("sentiment","mean"),avg_stars=("total_score","mean"),top_location=("location",lambda x: x.mode().iloc[0] if len(x)>0 else "Unknown")).sort_values("mentions",ascending=False)
    return s[s["mentions"]>=5]

@st.cache_data
def compute_ngrams(texts, sw_list, n=2, top_k=15):
    sw = set(sw_list)
    all_text = " ".join(texts.astype(str)).lower()
    words = [w for w in re.findall(r"[a-z']+", all_text) if w not in sw and len(w)>2]
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    return Counter(ngrams).most_common(top_k)

@st.cache_data
def get_neg_distinctive(neg_texts, pos_texts, sw_list):
    sw = set(sw_list)
    def clean(texts):
        return [w for w in re.findall(r"[a-z']+", " ".join(texts.astype(str)).lower()) if w not in sw and len(w)>2]
    neg_w = Counter(clean(neg_texts)); pos_w = Counter(clean(pos_texts))
    neg_rate = {w: c/len(neg_texts) for w,c in neg_w.most_common(200)}
    pos_rate = {w: c/len(pos_texts) for w,c in pos_w.most_common(200)}
    result = []
    for word, rate in sorted(neg_rate.items(), key=lambda x: -x[1]):
        pr = pos_rate.get(word, 0.001); ratio = rate / pr
        if ratio > 1.5 and neg_w[word] >= 10: result.append((word, neg_w[word], ratio))
    return sorted(result, key=lambda x: -x[2])[:20]

# ---------------------------------------------------------------------------
DATA_PATH = "shea_homes_reviews.csv"
try:
    df = load_and_process(DATA_PATH)
except FileNotFoundError:
    st.error(f"**Could not find `{DATA_PATH}`.** Place `shea_homes_reviews.csv` in the same directory as `app.py`.");
    st.stop()

fdf = df.copy()

PAGES = ["Overview", "Part 1: Summary Statistics", "Part 2: Data Evaluation", "Part 3: Sentiment Analysis",
         "Part 4: Advanced NLP", "Part 5: Predictive Models", "Part 6: LLM Analysis", "Review Explorer"]


def next_page():
    current_index = PAGES.index(st.session_state.page)
    if current_index < len(PAGES) - 1:
        st.session_state.page = PAGES[current_index + 1]
        st.session_state.scroll_top = True
        st.rerun()

def previous_page():
    current_index = PAGES.index(st.session_state.page)
    if current_index > 0:
        st.session_state.page = PAGES[current_index - 1]
        st.session_state.scroll_top = True
        st.rerun()


def scroll_to_top():
    components.html(
        """
        <script>
            window.parent.scrollTo({top: 0, behavior: 'smooth'});
        </script>
        """,
        height=0,
    )

with st.sidebar:
    st.markdown("## Shea Homes");
    st.markdown("**Customer Review Project**");
    st.markdown("---")

    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]

    selected_page = st.radio("Navigate", PAGES, index=PAGES.index(st.session_state.page))

    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.session_state.scroll_top = True
        st.rerun()

    page = st.session_state.page

    st.markdown("---")
    st.caption(f"**{len(fdf):,}** total reviews");
    st.markdown("---")
    st.caption("Built by **Griffin Snider**");

if st.session_state.get("scroll_top", False):
    scroll_to_here(0, key=f"top_{page}")
    st.session_state.scroll_top = False


# ===================================================================
if page == PAGES[0]:
    st.title("Shea Homes Customer Review Project")
    explain("This project takes 2,039 customer reviews from NewHomeSource and runs them through a series of AI and data analysis tools. Instead of someone reading through every review one by one, the computer reads all of them in seconds and pulls out the patterns. It figures out which reviews are happy, which are unhappy, what topics keep coming up, which markets are doing well, and where the biggest opportunities for improvement are. <br> <br> This report uses natural language processing tools to read and analyze Shea Homes customer reviews. It: (1) calculates key statistics, (2) understands the emotional tone of each review, (3) finds patterns across time, geography, and rating categories, and (4) surfaces the specific topics customers care about most.")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Reviews", f"{len(fdf):,}"); c2.metric("Avg Rating", f"{fdf['total_score'].mean():.2f}"); c3.metric("Positive", f"{(fdf['vader_label']=='Positive').mean():.0%}")
    c4.metric("Negative", f"{(fdf['vader_label']=='Negative').mean():.0%}"); c5.metric("At-Risk", f"{(fdf['total_score']<=3).sum():,}"); c6.metric("Markets", f"{fdf['state'].nunique()}")
    st.markdown("---")
    cats = ["total_score","quality","trustworthiness","value","responsiveness"]; clabels = ["Overall","Quality","Trust","Value","Responsiveness"]
    cmeans = [fdf[c].mean() for c in cats]
    fig = go.Figure(go.Bar(x=clabels, y=cmeans, marker_color=[SHEA_GOLD if v==max(cmeans) else (NEG_RED if v==min(cmeans) else SHEA_BLUE) for v in cmeans], text=[f"{v:.2f}" for v in cmeans], textposition="outside", textfont_size=14))
    fig.update_yaxes(range=[0,5.5]); fig.update_layout(title="Average Scores by Category"); clean_fig(fig, 400); st.plotly_chart(fig, use_container_width=True)
    commentary("Shea Homes averages a 4.21 out of 5 across 2,039 reviews, with 78% of customers rating 4 or 5 stars. Responsiveness consistently scores highest, while Value trails slightly behind the other categories. The 449 at-risk reviews (1-3 stars) represent 22% of the dataset and are the primary focus of the deeper analysis in later sections.")
    st.markdown(f"**Data Source:** [NewHomeSource](https://www.newhomesource.com/builder/shea-homes/reviews/612/) &nbsp;|&nbsp; **Date Range:** {df['date'].min().strftime('%B %Y')} to {df['date'].max().strftime('%B %Y')} &nbsp;|&nbsp; **Author:** Griffin Snider")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Previous"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()

# ===================================================================
elif page == PAGES[1]:
    st.title("Part 1: Summary Statistics")
    explain("Before doing any AI or analysis, we will start by just looking at the basic numbers. How many reviews are there? How are the star ratings distributed? Which states and cities have the most feedback? How long are the reviews?")
    section_header("1.1 Dataset Overview", "High-level metrics on the full review")
    c1,c2,c3,c4 = st.columns(4); c1.metric("Total Reviews",f"{len(fdf):,}"); c2.metric("Total Words",f"{fdf['word_count'].sum():,}"); c3.metric("Avg Length",f"{fdf['word_count'].mean():.1f} words"); c4.metric("Unique Cities",f"{fdf['location'].nunique()}")
    c5,c6,c7,c8 = st.columns(4); c5.metric("Unique Reviewers",f"{fdf['reviewer_name'].nunique():,}"); c6.metric("Median Length",f"{fdf['word_count'].median():.0f} words"); c7.metric("Shortest",f"{fdf['word_count'].min()} words"); c8.metric("Longest",f"{fdf['word_count'].max()} words")
    commentary("The dataset contains 117,979 total words across 2,039 reviews, averaging about 58 words per review but with a median of only 34, meaning the distribution is right-skewed: most reviews are short, but a handful are very detailed. The longest review at 1,092 words represents a customer who had a lot to say. Reviews span 74 unique cities across 11 states, giving us broad geographic coverage of Shea's markets.")

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
    commentary("The distribution is heavily left-skewed: 5-star reviews dominate at 53%, while 1-star reviews account for only 7%. All five rating categories land above 4.0, but Value (4.09) lags behind the others. This gap suggests customers feel the product is good but that pricing or perceived worth relative to cost is a softer spot. Responsiveness (4.23) leads, indicating the team's communication is a clear strength.")

    section_header("1.3 Review Volume Over Time")
    explain("This chart tracks two things at once. The blue bars show how many reviews were submitted each month. The gold line shows the rolling average star rating over time. Together, they tell us whether customer satisfaction has been getting better, getting worse, or staying consistent.")
    monthly = fdf.groupby("year_month").agg(count=("total_score","count"),avg=("total_score","mean")).reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"]); monthly = monthly.sort_values("year_month"); monthly["rolling"] = monthly["avg"].rolling(3,min_periods=1).mean()
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=monthly["year_month"],y=monthly["count"],name="Monthly Reviews",marker_color=SHEA_BLUE,opacity=0.6), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["year_month"],y=monthly["rolling"],name="3-Mo Avg Score",line=dict(color=SHEA_GOLD,width=3)), secondary_y=True)
    fig.update_yaxes(title_text="Count",secondary_y=False); fig.update_yaxes(title_text="Avg Score",range=[1,5.5],secondary_y=True)
    fig.add_hline(y=4.0,line_dash="dash",line_color="gray",opacity=0.3,secondary_y=True); fig.update_layout(title="Review Volume & Score Over Time"); clean_fig(fig,420); st.plotly_chart(fig, use_container_width=True)
    commentary("Review volume peaked during 2021-2022, coinciding with the post-COVID housing boom when Shea was delivering a high number of homes. Volume has tapered since then as the market cooled. The 3-month rolling average score has remained relatively stable above 4.0 throughout, with some dips in late 2022 and early 2023 that are worth investigating at the market level. The consistency of the score over time suggests systemic strengths.")

    section_header("1.4 Geographic Breakdown")
    explain("These charts show where the reviews come from. The left side counts how many reviews each state contributed. The right side shows the average star rating for each state, but only for states that had at least 10 reviews so the numbers are meaningful.")
    col1, col2 = st.columns(2)
    with col1:
        stc = fdf["state"].value_counts().reset_index(); stc.columns=["state","count"]
        fig = px.bar(stc,y="state",x="count",orientation="h",color_discrete_sequence=[SHEA_BLUE],text="count"); fig.update_traces(textposition="inside"); fig.update_layout(title="Reviews by State"); clean_fig(fig,max(350,len(stc)*45)); st.plotly_chart(fig, use_container_width=True)
    with col2:
        sts = fdf.groupby("state")["total_score"].agg(["mean","count"]).reset_index(); sts = sts[sts["count"]>=10].sort_values("mean")
        sts["color"] = sts["mean"].apply(lambda v: POS_GREEN if v>=4 else (NEU_YELLOW if v>=3 else NEG_RED))
        fig = go.Figure(go.Bar(y=sts["state"],x=sts["mean"],orientation="h",marker_color=sts["color"],text=[f"{v:.2f}" for v in sts["mean"]],textposition="inside"))
        fig.add_vline(x=4.0,line_dash="dash",line_color="gray",opacity=0.4); fig.update_xaxes(range=[0,5.5]); fig.update_layout(title="Avg Score by State (10+ reviews)"); clean_fig(fig,max(350,len(sts)*45)); st.plotly_chart(fig, use_container_width=True)
    commentary("California (684 reviews) and Arizona (532) together make up about 60% of all feedback. On the satisfaction side, most states clear the 4.0 threshold. North Carolina and Colorado trend lower, which may warrant closer examination of specific communities or construction teams in those regions. South Carolina only has 8 reviews and is excluded from the scored chart due to insufficient sample size.")

    section_header("1.5 Top Cities & Review Length")
    explain("The left chart shows which specific cities generated the most reviews. The right chart shows the distribution of review lengths, measured in words. This matters because longer reviews tend to contain more specific and useful information for analysis.")
    col1, col2 = st.columns(2)
    with col1:
        tc = fdf["location"].value_counts().head(15).reset_index(); tc.columns=["city","count"]
        fig = go.Figure(go.Bar(y=tc["city"][::-1],x=tc["count"][::-1],orientation="h",marker_color=SHEA_BLUE,text=tc["count"][::-1],textposition="outside"))
        fig.update_layout(title="Top 15 Cities"); fig.update_xaxes(range=[0,tc["count"].max()*1.25]); clean_fig(fig,500); st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(fdf,x="word_count",nbins=40,color_discrete_sequence=[SHEA_BLUE],labels={"word_count":"Words per Review"})
        fig.add_vline(x=fdf["word_count"].median(),line_dash="dash",line_color=NEG_RED,annotation_text=f"Median: {fdf['word_count'].median():.0f}")
        fig.update_layout(title="Review Length Distribution"); clean_fig(fig,500); st.plotly_chart(fig, use_container_width=True)
    commentary("Rio Verde, AZ and Indio, CA lead in review volume. The review length histogram shows most customers write concise feedback (under 50 words), but a long tail of detailed reviews exists. These longer reviews tend to be the most analytically valuable, as they contain specific complaints or praise that keyword and sentiment analysis can pick up.")

    section_header("1.6 Rating Correlations")
    explain("This heatmap shows how the five rating dimensions relate. Higher correlation means these categories move together.")
    scols = ["total_score","quality","trustworthiness","value","responsiveness"]; slabs = ["Overall","Quality","Trust","Value","Responsiveness"]
    fig = px.imshow(fdf[scols].corr().values, x=slabs, y=slabs, color_continuous_scale="YlOrRd", zmin=0.5, zmax=1, text_auto=".2f")
    fig.update_layout(title="Rating Category Correlations"); clean_fig(fig,450); st.plotly_chart(fig, use_container_width=True)
    commentary("All categories are highly correlated, showing that customers who rate one dimension poorly tend to rate everything poorly. This suggests the overall experience is somewhat holistic: a bad construction experience drags down trust, value, and responsiveness perceptions too. The strongest correlation is between Quality and Trustworthiness (0.93), which makes intuitive sense since build quality is a direct signal of whether a builder can be trusted.")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Previous"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()
# ===================================================================
elif page == PAGES[2]:
    st.title("Part 2: Data Evaluation")
    explain("Before doing analysis, it is important to check whether the data is good enough to support the conclusions we want to draw. Is 2,039 reviews enough? Are there any biases or blind spots we should know about? This section is about being honest about the strengths and limitations of the dataset before we use it to make recommendations.")
    section_header("2.1 Business Question")
    explain("<b>Core business question:</b> What are customers really saying about Shea Homes, and where are the opportunities to improve?")

    section_header("2.2 Sample Size Assessment")
    n=len(fdf); mu=fdf["total_score"].mean(); sd=fdf["total_score"].std(); me=1.96*(sd/np.sqrt(n)); ci_lo=mu-me; ci_hi=mu+me
    c1,c2,c3=st.columns(3); c1.metric("Sample Size",f"{n:,}"); c2.metric("Mean Score",f"{mu:.3f}"); c3.metric("95% CI",f"[{ci_lo:.3f}, {ci_hi:.3f}]")
    st.markdown("**Per-state sample sizes:**")
    st.dataframe(fdf["state"].value_counts().reset_index().rename(columns={"index":"State","state":"State","count":"Reviews"}), use_container_width=True, hide_index=True)
    commentary("With 2,039 reviews and a margin of error of plus or minus 0.048 stars, we can be very confident in the overall average. It is a tight range. However, smaller markets like Idaho with 34 reviews and South Carolina with 8 should be interpreted cautiously. State level comparisons are most trustworthy for California, Arizona, Colorado, Texas, and Nevada where sample sizes exceed 100.")

    section_header("2.3 Potential Biases")
    st.dataframe(pd.DataFrame([
        ["Self-selection bias","Customers with strong opinions more likely to review","May overrepresent extremes"],
        ["Geographic concentration","CA and AZ account for ~60% of reviews","State-level conclusions should note sample sizes"],
        ["Temporal skew","2021-2022 peak volume (post-COVID boom)","Trends may reflect market conditions"],
        ["Platform bias","TrustBuilder is builder-partnered","Sentiment may be inflated vs independent sites"],
    ], columns=["Bias","Description","Impact"]), use_container_width=True, hide_index=True)
    commentary("These biases don't invalidate the analysis, but they frame how to interpret it. The most important one is platform bias: because these reviews come from a builder-partnered site (TrustBuilder), the overall sentiment likely skews more positive than what you would see on Yelp or the BBB. This means the negative reviews we do find are especially significant, as they represent dissatisfaction strong enough to surface despite the platform's positive tilt.")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Previous"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()
# ===================================================================
elif page == PAGES[3]:
    st.title("Part 3: Preliminary Sentiment Analysis")
    explain("Sentiment analysis lets a computer read written feedback and estimate the emotion behind it. Each review is scanned word-by-word using dictionaries of terms known to carry positive or negative meaning (for example great, love, slow, frustrating). These signals are combined into a final score between –1 (very negative) and +1 (very positive) that represents the overall tone of the review. We used two tools to calculate this: VADER, which is designed for reviews and social media and adjusts for emphasis like capitalization or exclamation points, and TextBlob, a general language model that calculates sentiment based on the balance of positive and negative words. Using both methods helps confirm that the patterns we observe in customer sentiment are consistent and reliable.")

    section_header("3.1 Overall Sentiment Breakdown")
    explain("The two pie charts below show what percentage of reviews each tool classified as positive, negative, or neutral. The line chart on the right is a sanity check. It plots average sentiment score against star rating to make sure the tools are actually working correctly. If they are, we should see sentiment go up as star ratings go up.")
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
    commentary("Both methods agree that the vast majority of reviews are positive (VADER: 78%, TextBlob: 80%), which aligns with the 78% of reviews being 4-5 stars. VADER identifies more negative reviews (16%) than TextBlob (8%), because VADER is better at catching negative language patterns in review text. The Sentiment vs Stars chart confirms both tools properly track star ratings, with sentiment scores climbing linearly from 1-star to 5-star, validating that these algorithms are reading the text correctly.")

    section_header("3.2 Sentiment Trends Over Time")
    explain("This tracks customer sentiment over time, broken into quarters. The top chart shows the average sentiment score each quarter. The bottom chart shows what percentage of reviews were positive versus negative in each quarter. Together they tell us whether customer happiness has been changing over time.")
    qtr = fdf.groupby("quarter").agg(avg_v=("vader_compound","mean"),pct_pos=("vader_label",lambda x:(x=="Positive").mean()),pct_neg=("vader_label",lambda x:(x=="Negative").mean()),cnt=("total_score","count")).reset_index()
    qtr["qt"]=pd.to_datetime(qtr["quarter"]); qtr=qtr.sort_values("qt")
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.08,subplot_titles=["Avg VADER Sentiment","Positive vs Negative Share"])
    fig.add_trace(go.Scatter(x=qtr["qt"],y=qtr["avg_v"],mode="lines+markers",line=dict(color=SHEA_BLUE,width=2),fill="tozeroy",fillcolor="rgba(26,82,118,0.08)",name="Avg VADER"),row=1,col=1)
    fig.add_trace(go.Bar(x=qtr["qt"],y=qtr["pct_pos"]*100,name="% Positive",marker_color=POS_GREEN,opacity=0.7),row=2,col=1)
    fig.add_trace(go.Bar(x=qtr["qt"],y=-qtr["pct_neg"]*100,name="% Negative",marker_color=NEG_RED,opacity=0.7),row=2,col=1)
    clean_fig(fig,550); fig.update_yaxes(range=[0,1],row=1,col=1); fig.update_layout(title="Sentiment Trends Over Time"); st.plotly_chart(fig, use_container_width=True)
    commentary("Sentiment has remained broadly stable over the five-year period, hovering between 0.4 and 0.7 on the VADER compound scale. The bottom chart shows the negative share rarely exceeds 25% in any quarter, and positive share consistently dominates at 70-85%. No dramatic deterioration or improvement trend is visible, which suggests Shea's customer experience quality is relatively consistent across time rather than volatile.")

    section_header("3.3 Sentiment by State")
    explain("This chart ranks each state by how positive or negative the review text reads, using the VADER sentiment score. A higher score means customers in that state tend to write more positively. This tells us which markets are performing best from the customer's perspective and which ones might need attention.")
    ss = fdf.groupby("state").agg(avg_v=("vader_compound","mean"),avg_s=("total_score","mean"),cnt=("total_score","count")).reset_index()
    ss = ss[ss["cnt"]>=10].sort_values("avg_v"); ss["color"]=ss["avg_v"].apply(lambda v: POS_GREEN if v>=0.5 else (NEU_YELLOW if v>=0.3 else NEG_RED))
    fig = go.Figure(go.Bar(y=ss["state"],x=ss["avg_v"],orientation="h",marker_color=ss["color"],text=[f"{v:.3f} (avg {s:.1f}, n={c})" for v,s,c in zip(ss["avg_v"],ss["avg_s"],ss["cnt"])],textposition="outside"))
    fig.update_xaxes(range=[0,ss["avg_v"].max()+0.25]); fig.update_layout(title="Sentiment by State"); clean_fig(fig,max(380,len(ss)*50)); st.plotly_chart(fig, use_container_width=True)
    commentary("North Carolina and Idaho stand out as the lowest-sentiment state, which paired with its lower star average warrants a closer look at specific communities or construction teams operating there. Meanwhile, Colorado and Texas lead in sentiment. California and Arizona, the two largest markets, both perform solidly above 0.45.")

    section_header("3.4 Negative vs Positive Word Frequency")
    explain("This section compares the most common words used in 1 to 2 star reviews against the most common words used in 4 to 5 star reviews. By looking at what language unhappy customers use versus happy customers, we can start to see the specific themes driving satisfaction and dissatisfaction")
    sw = get_stop_words(); sw_list = list(sw)
    neg_t = fdf[fdf["total_score"]<=2]["review_text"]; pos_t = fdf[fdf["total_score"]>=4]["review_text"]
    if len(neg_t)>5 and len(pos_t)>5:
        col1,col2 = st.columns(2)
        for col, texts, label, color in [(col1,neg_t,f"Negative (1-2 star, n={len(neg_t)})",NEG_RED),(col2,pos_t,f"Positive (4-5 star, n={len(pos_t)})",POS_GREEN)]:
            with col:
                bi = compute_ngrams(texts, sw_list, 1, 20)
                if bi:
                    w, c = zip(*bi[::-1])

                    fig = go.Figure(go.Bar(
                        y=list(w),
                        x=list(c),
                        orientation="h",
                        marker_color=color,
                        text=list(c),
                        textposition="outside"
                    ))

                    max_c = max(c)

                    fig.update_layout(
                        title=label,
                        margin=dict(l=40, r=80, t=40, b=40)
                    )

                    fig.update_xaxes(range=[0, max_c * 1.10])

                    clean_fig(fig, 500)
                    st.plotly_chart(fig, use_container_width=True)
    commentary("On the negative side, the words that come up over and over are issues, problems, quality, warranty, time, and construction. Unhappy customers are talking about things that went wrong with the house itself and how long it took to get things fixed. On the positive side, the most common words are experience, team, great, process, building, and sales. Happy customers are talking about the people they worked with and how smooth the process was. This is a clear pattern: the people are the strength, and the physical product is where problems show up.")

    section_header("3.5 Distinctive Negative Review Words")
    explain("This table goes a step further than the word charts above. Instead of just counting common words, it compares how often a word shows up in negative reviews versus positive reviews. A word with a high overrepresentation number means it appears far more often in complaints than in praise. This helps pinpoint the specific issues that are driving dissatisfaction.")
    if len(neg_t)>5 and len(pos_t)>5:
        dist = get_neg_distinctive(neg_t, pos_t, sw_list)
        if dist: st.dataframe(pd.DataFrame(dist, columns=["Word","Count","Overrepresentation"]).assign(**{"Overrepresentation": lambda d: d["Overrepresentation"].map("{:.1f}x".format)}), use_container_width=True, hide_index=True)
    commentary("Words like poor (192.7x more common in negative reviews), installed (140.6x), cabinets (109.4x), and flooring (83.3x) point to specific construction pain points. The word waiting (109.4x) and months suggest delays are a major theme. These are not abstract complaints; they are about specific, fixable things: cabinets installed wrong, flooring issues, and long wait times for repairs.")

    section_header("3.6 Score vs. Sentiment Mismatch")
    explain("Reviews where star rating and text sentiment disagree. Hidden complaints (4-5 stars but negative text) gave a good rating but still wrote about problems. These are people who are generally satisfied but have a specific issue they want heard. They are not angry enough to leave a bad rating, but they are telling you exactly what to fix. This section finds those reviews.")
    high_neg = fdf[(fdf["total_score"]>=4)&(fdf["vader_compound"]<-0.05)]; low_pos = fdf[(fdf["total_score"]<=2)&(fdf["vader_compound"]>0.5)]
    c1,c2,c3 = st.columns(3); c1.metric("Total Mismatches",f"{fdf['mismatch'].sum()} ({fdf['mismatch'].mean():.1%})"); c2.metric("High Stars + Neg Text",f"{len(high_neg)}"); c3.metric("Low Stars + Pos Text",f"{len(low_pos)}")
    if len(high_neg)>0:
        st.markdown("**Hidden Complaints (samples):**")
        for _, r in high_neg.sort_values("vader_compound").head(3).iterrows():
            st.markdown(f"> **{r['total_score']} stars** | VADER: {r['vader_compound']:.3f} | {r['location']}  \n> _{str(r['review_text'])[:300]}..._")
    commentary("110 customers gave 4-5 stars but wrote text that VADER scored as negative. These are the hidden complaints: customers who were happy enough overall but had specific grievances. These are arguably the most actionable reviews in the entire dataset because they represent salvageable relationships. A customer who gives 5 stars but complains about landscaping subcontractors is telling you exactly what to fix without being angry enough to leave a bad rating.")

    section_header("3.7 Sample Reviews by Sentiment")
    explain("To make the sentiment scores tangible, here are real reviews from each category. This gives a feel for what the algorithm is actually reading and how it translates to a score.")
    for label, sa in [("Positive",False),("Negative",True),("Neutral",None)]:
        sub = fdf[fdf["vader_label"]==label]
        if len(sub)==0: continue
        sub = sub.sort_values("vader_compound",ascending=sa).head(2) if sa is not None else sub.head(2)
        st.markdown(f"**Most {label} Reviews:**")
        for _, r in sub.iterrows():
            st.markdown(f"> **{r['total_score']} stars** | {r['location']} | {r['date'].strftime('%b %Y') if pd.notna(r['date']) else 'N/A'} | VADER: {r['vader_compound']:.3f}  \n> _{str(r['review_text'])[:350]}..._")
    commentary("These samples show what the sentiment engine is actually picking up. The most positive reviews are enthusiastic and mention specific people by name. The most negative reviews describe specific construction defects and frustrations with response times. The neutral reviews tend to be very short or factual without strong emotional language. Reading a few of these gives a good feel for what each sentiment category actually sounds like in practice.")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Previous"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()
# ===================================================================
elif page == PAGES[4]:
    st.title("Part 4: Advanced Natural Language Processing Analysis")
    explain("This section uses more advanced natural language processing techniques to go beyond simple positive and negative labels. It automatically discovers what topics customers talk about, measures how they feel about specific business areas like sales or warranty, finds which employees are mentioned most often, and looks at multi-word phrases that reveal more nuance than individual words alone.")

    section_header("4.1 Topic Discovery (Latent Dirichlet Allocation (LDA))")
    explain("Topic modeling is a way to have the computer sort all 2,039 reviews into groups based on what each one is talking about. We did not tell the computer what the topics should be. It read through every review on its own and figured out that certain words tend to show up together. For example, reviews that mention warranty, repair, and fix tend to cluster together, so the computer creates a Warranty topic. This gives us a bird's eye view of what customers are actually talking about.")
    tnames, tconf, tkws, tname_map = compute_topics(df["review_text"]); df["topic_name"]=tnames; fdf_t=df.copy()
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
    commentary("The algorithm discovered 6 distinct topics without any guidance. Topics related to warranty and construction issues have the lowest satisfaction scores (often below 4.0), while topics around the sales experience and community lifestyle score highest. This aligns with what the word frequency analysis found: the people and the process are strengths, while the physical build and post-close service are where dissatisfaction concentrates.")

    section_header("4.2 Aspect-Based Sentiment")
    explain("This takes a different approach from topic modeling. Instead of looking at the whole review at once, it breaks each review into specific business topics: sales, construction quality, warranty, design, communication, and pricing. It then measures the sentiment for each topic separately within the same review. So if a customer writes one sentence praising the sales team and another sentence complaining about drywall cracks, this tool scores the sales mention as positive and the construction mention as negative, instead of averaging them together.")
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
    commentary(f"Construction Quality is mentioned in 54% of all reviews, making it the most discussed aspect, but it also has a relatively lower sentiment. Warranty & Post-Move is the weakest aspect at {adf.loc[w,'avg_sentiment']:+.3f}, with 25% of warranty-related mentions carrying negative sentiment. Sales & Buying Process and Communication both score well, reinforcing the pattern that Shea's people are its biggest asset.")

    section_header("4.3 Employee Recognition Mining")
    explain("Customers often mention Shea team members by name in their reviews. This section scans through all 2,039 reviews, looks for names, and then reads the sentences around each name to figure out whether the customer was saying something positive or negative about that person. This is useful for identifying top performers who consistently get praised, or for spotting cases where a specific team member keeps coming up in negative contexts.")
    emp = compute_employees(fdf["review_text"].tolist(),fdf["total_score"].tolist(),fdf["location"].tolist(),fdf["state"].tolist())
    if not emp.empty:
        ep = emp.head(20).sort_values("avg_sentiment")
        fig = go.Figure(go.Bar(y=ep.index,x=ep["avg_sentiment"],orientation="h",marker_color=[POS_GREEN if v>=0.3 else (NEU_YELLOW if v>=0 else NEG_RED) for v in ep["avg_sentiment"]],text=[f"{s:+.2f} ({m:.0f} mentions)" for s,m in zip(ep["avg_sentiment"],ep["mentions"])],textposition="outside"))
        fig.update_xaxes(range=[0, max(ep["avg_sentiment"]) * 1.14]); fig.add_vline(x=0,line_color="black",line_width=1); fig.update_layout(title="Employee Sentiment (5+ mentions)"); clean_fig(fig,max(600,len(ep)*32)); st.plotly_chart(fig, use_container_width=True)
        d=emp.head(25).copy(); d["avg_sentiment"]=d["avg_sentiment"].map("{:+.3f}".format); d["avg_stars"]=d["avg_stars"].map("{:.1f}".format); d.columns=["Mentions","Avg Sentiment","Avg Stars","Primary Location"]
        st.dataframe(d, use_container_width=True); st.caption("Note: Heuristic name extraction. May include false positives. 5+ mention threshold reduces noise.")
    commentary("Mike in Las Vegas, Ryan in Indio, and Josh in Manvel are the most frequently mentioned employees, and all of them carry positive sentiment scores. That means when customers mention these people, they are saying good things. This kind of data could feed directly into recognition programs or performance reviews. It could also flag cases where a specific person keeps showing up in negative reviews, which would be an early signal for coaching or support.")

    section_header("4.4 Common Phrases (N-grams)", "Two and three word phrases in negative vs positive reviews")
    explain("The word charts earlier looked at one word at a time. But sometimes meaning comes from combinations of words. The phrase not responsive means the opposite of responsive, even though the word responsive by itself sounds positive. This section looks at two-word and three-word phrases that show up frequently in negative versus positive reviews, which gives a more accurate picture of what customers are actually saying.")
    sw_list = list(get_stop_words()); neg_t = fdf[fdf["total_score"]<=2]["review_text"]; pos_t = fdf[fdf["total_score"]==5]["review_text"]
    if len(neg_t)>=5 and len(pos_t)>=5:
        col1,col2 = st.columns(2)
        for col,texts,label,color in [(col1,neg_t,"Negative (1-2 star)",NEG_RED),(col2,pos_t,"Positive (5 star)",POS_GREEN)]:
            with col:
                for nv,nl in [(2,"Bigrams"),(3,"Trigrams")]:
                    ngr = compute_ngrams(texts,sw_list,nv,12)
                    if ngr:
                        w,c = zip(*ngr[::-1]); fig = go.Figure(go.Bar(y=list(w),x=list(c),orientation="h",marker_color=color,text=list(c),textposition="outside"))
                        fig.update_layout(title=f"{label}: {nl}"); clean_fig(fig,380); st.plotly_chart(fig, use_container_width=True)
    commentary("The negative phrases reveal specific pain points:  warranty manager and build quality come up repeatedly. On the positive side, sales team, great experience, building process, and start finish dominate. The phrase start finish showing up frequently in 5-star reviews is a strong signal. Customers love it when the whole journey feels seamless from beginning to end.")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Previous"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()
# ===================================================================
elif page == PAGES[5]:
    st.title("Part 5: Predictive Models")
    explain("<b>The Big Question:</b> Can ML predict star ratings from text alone?<br><br><b>Why it matters:</b> Everything up to this point has been about understanding reviews that already have star ratings attached. This section asks a different question: if we only had the text of a review and no star rating, could the computer figure out whether the customer is happy or unhappy just from what they wrote? This matters because a lot of customer feedback, like open-ended survey responses, social media comments, or emails, does not come with a star rating. If we can teach the computer to read text and predict satisfaction, we can apply it to all kinds of feedback that does not have a number attached. We tested three different prediction models to see which one does the best job.<br><br><b>Approach:</b> Hybrid models combining TF-IDF text features (5,000 word/phrase weights) with numeric signals (VADER score, word count, exclamation count). Balanced class weights handle the 78/22% class imbalance. Evaluated with Macro F1 so models cannot cheat by predicting the majority class.")

    section_header("5.1 At-Risk Customer Detection (Binary)", "Satisfied (4-5 star) vs At-Risk (1-3 star)")
    explain("This is the simplest and most practical version of the prediction. We split all reviews into two groups: satisfied customers who gave 4 or 5 stars, and at-risk customers who gave 1, 2, or 3 stars. Then we trained three different models on 80 percent of the reviews, where the model could see both the text and the star rating and learn the patterns. After training, we tested each model on the remaining 20 percent of reviews that it had never seen before, giving it only the text and asking it to predict whether the customer was satisfied or at-risk.")

    # Model comparison chart
    models = ["Logistic Regression","Random Forest","Gradient Boosting"]
    acc = [81.4, 82.8, 84.1]; f1 = [0.753, 0.711, 0.757]; recall = [72, 43, 58]; prec = [56, 67, 66]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Accuracy & Macro F1","At-Risk Recall & Precision"])
    fig.update_layout(title="Models")
    fig.add_trace(go.Bar(name="Accuracy %", x=models, y=acc, marker_color=SHEA_BLUE, text=[f"{v}%" for v in acc], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(name="Macro F1", x=models, y=[v*100 for v in f1], marker_color=SHEA_GOLD, text=[f"{v:.3f}" for v in f1], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(name="At-Risk Recall", x=models, y=recall, marker_color=NEG_RED, text=[f"{v}%" for v in recall], textposition="outside"), row=1, col=2)
    fig.add_trace(go.Bar(name="At-Risk Precision", x=models, y=prec, marker_color=NEUTRAL_GRAY, text=[f"{v}%" for v in prec], textposition="outside"), row=1, col=2)
    fig.update_yaxes(range=[0,105]); fig.update_layout(barmode="group"); clean_fig(fig, 420); st.plotly_chart(fig, use_container_width=True)

    static_output(" RESULTS: Gradient Boosting (Best Model)\n============================================================\n                       precision    recall  f1-score   support\n\n  At-Risk (1-3 Stars)    0.66         0.58    0.62       90\nSatisfied (4-5 Stars)  0.88         0.92    0.90       318 \n \n accuracy                                    0.84       408\n            macro avg               0.77      0.75      0.76       408\n         weighted avg            0.83      0.84      0.84       408")
    commentary("Gradient Boosting wins with 84.1% accuracy and the highest Macro F1 (0.757). It catches 58% of at-risk customers from text alone, with 66% precision, meaning when it flags someone as at-risk, it is right 2 out of 3 times. Logistic Regression has better at-risk recall (72%) but more false positives (56% precision). Random Forest is the weakest here because it struggles with the minority class, only catching 43% of at-risk customers despite high overall accuracy. The tradeoff between recall and precision depends on the business use case: if missing an at-risk customer is costly, Logistic Regression's higher recall may be preferable.")

    section_header("5.2 Three-Class Prediction", "Negative (1-2 star) vs Neutral (3 star) vs Positive (4-5 star)")
    explain("This is a harder version of the same task. Instead of just satisfied versus at-risk, we now ask the model to sort reviews into three groups: negative (1 to 2 stars), neutral (3 stars), and positive (4 to 5 stars). This is trickier because 3-star reviews are genuinely ambiguous. The customer is not really happy or really unhappy, and their language reflects that.")

    models3 = ["Logistic Regression","Random Forest","Gradient Boosting"]
    acc3 = [76.2, 77.5, 79.2]; f1_3 = [0.561, 0.319, 0.510]
    neg_r = [58, 3, 39]; neu_r = [37, 2, 15]; pos_r = [85, 99, 94]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Accuracy & Macro F1","Recall by Class"])
    fig.add_trace(go.Bar(name="Accuracy %", x=models3, y=acc3, marker_color=SHEA_BLUE, text=[f"{v}%" for v in acc3], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(name="Macro F1", x=models3, y=[v*100 for v in f1_3], marker_color=SHEA_GOLD, text=[f"{v:.3f}" for v in f1_3], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(name="Neg Recall", x=models3, y=neg_r, marker_color=NEG_RED, text=[f"{v}%" for v in neg_r], textposition="outside"), row=1, col=2)
    fig.add_trace(go.Bar(name="Neu Recall", x=models3, y=neu_r, marker_color=NEU_YELLOW, text=[f"{v}%" for v in neu_r], textposition="outside"), row=1, col=2)
    fig.add_trace(go.Bar(name="Pos Recall", x=models3, y=pos_r, marker_color=POS_GREEN, text=[f"{v}%" for v in pos_r], textposition="outside"), row=1, col=2)
    fig.update_yaxes(range=[0,115]); fig.update_layout(barmode="group"); clean_fig(fig, 420); st.plotly_chart(fig, use_container_width=True)
    commentary("This is a good example of why you cannot just look at one number. Random Forest had the highest overall accuracy at 77.5 percent, but look at the right side of the chart. It only caught 3 percent of negative reviews and 2 percent of neutral ones. It was basically just predicting positive for every single review and getting a high score because most reviews are positive. That is not useful at all. Logistic Regression did a much better job of actually finding the negative and neutral reviews, catching 58 percent and 37 percent respectively. The takeaway is that prediction is possible, but the middle ground of 3-star reviews is hard to pin down because the language customers use in those reviews is genuinely mixed.")

    section_header("5.3 Most Predictive Words by Category", "What language signals each rating level?")
    explain("When the prediction model reads a review and makes its decision, it is paying attention to specific words more than others. This section shows which words had the biggest influence on pushing a review toward each category. Think of it like asking the model: what are you looking at when you decide a review is negative versus positive? The longer the bar, the more that word mattered to the prediction.")

    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown("**Negative (1-2 star)**")
        neg_words = [("don",1.291),("workmanship average",1.272),("beware",1.231),("care",1.225),("repairs",1.163),("shea",1.136),("bad",1.032),("months",1.031),("appliances",0.941),("average",0.929)]
        nw, nc = zip(*neg_words)
        fig = go.Figure(go.Bar(y=list(nw)[::-1], x=list(nc)[::-1], orientation="h", marker_color=NEG_RED, text=[f"+{v:.3f}" for v in list(nc)[::-1]], textposition="inside"))
        fig.update_xaxes(range=[0,1.6]); fig.update_layout(title="Negative Signals"); clean_fig(fig, 380); st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Neutral (3 star)**")
        neu_words = [("hoa",1.356),("unfortunately",1.199),("trust",1.090),("lovely home",1.073),("electrical",0.992),("kind",0.956),("lacking",0.935),("like",0.934),("nice home",0.922),("better",0.889)]
        nuw, nuc = zip(*neu_words)
        fig = go.Figure(go.Bar(y=list(nuw)[::-1], x=list(nuc)[::-1], orientation="h", marker_color=NEU_YELLOW, text=[f"+{v:.3f}" for v in list(nuc)[::-1]], textposition="inside"))
        fig.update_xaxes(range=[0,1.6]); fig.update_layout(title="Neutral Signals"); clean_fig(fig, 380); st.plotly_chart(fig, use_container_width=True)
    with col3:
        st.markdown("**Positive (4-5 star)**")
        pos_words = [("vader",1.539),("excellent",1.074),("experience",1.019),("professional",0.862),("great",0.810),("home",0.714),("helpful",0.683),("team",0.669),("informed",0.639),("awesome",0.631)]
        pw, pc = zip(*pos_words)
        fig = go.Figure(go.Bar(y=list(pw)[::-1], x=list(pc)[::-1], orientation="h", marker_color=POS_GREEN, text=[f"+{v:.3f}" for v in list(pc)[::-1]], textposition="inside"))
        fig.update_xaxes(range=[0,1.8]); fig.update_layout(title="Positive Signals"); clean_fig(fig, 380); st.plotly_chart(fig, use_container_width=True)
    commentary("The predictive words tell a story that aligns with everything else in this analysis. Negative signals are concrete and specific: 'repairs,' 'months,' 'appliances,' 'workmanship average.' Neutral reviews use hedging language: 'unfortunately,' 'lacking,' 'nice home' (faint praise), and 'better' (implying something could be improved). Positive signals are emotional and relational: 'excellent,' 'professional,' 'helpful,' 'team,' 'awesome.' Note that 'vader' appears as the strongest positive predictor because the VADER sentiment score was included as a numeric feature, and it strongly correlates with positive ratings.")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Previous"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()
# ===================================================================
elif page == PAGES[6]:
    st.title("Part 6: Large Language Model Analysis")
    st.markdown("*LLaMA 3.2 via Ollama (pre-computed locally)*")
    explain("Everything before used specialized tools for narrow tasks. A Large Language Model like Meta's LLaMA 3.2 is a general-purpose AI that can read reviews like a human, follow complex instructions, and produce natural language output. We gave it customer reviews and asked it to do things like classify sentiment, identify themes, suggest action items, and even write an executive summary. These results were generated on a local computer and saved here.")

    section_header("6.1 Zero-Shot Sentiment Classification", "No examples given, the LLM classifies from pure understanding")
    explain("In this test, we gave the LLM a review and simply asked: is this positive or negative? We did not show it any examples first. We did not teach it what Shea Homes reviews look like. We just asked it to figure it out on its own based on its general understanding of language. We tested this on a sample of 50 reviews, half positive and half negative, and compared it against the VADER tool from earlier.")
    static_output("ZERO-SHOT RESULTS (50 reviews)\n=======================================================\n  Method                    Accuracy\n  ----------------------------------------\n  LLaMA 3.2 (Zero-Shot)     76.0%\n  VADER                     68.0%\n=======================================================")
    commentary("Even with zero examples, LLaMA 3.2 outperforms VADER by 8 percentage points. This is because the LLM understands context and nuance that a rule-based system like VADER misses. For instance, a review saying 'I would not recommend this builder' uses negation that VADER can struggle with, but the LLM understands naturally.")

    section_header("6.2 Few-Shot At-Risk Detection", "Train the LLM with a few examples first")
    explain("This time, before asking the LLM to classify reviews, we showed it a few examples first. We gave it 2 positive reviews and 2 negative reviews with their correct labels. Then we asked it to classify the same 50 test reviews.")
    fig = go.Figure(go.Bar(x=["VADER\n(Rule-Based)","LLaMA 3.2\nZero-Shot","LLaMA 3.2\nFew-Shot"],y=[68,76,76],marker_color=[NEUTRAL_GRAY,SHEA_BLUE,SHEA_GOLD],text=["68%","76%","76%"],textposition="outside",textfont_size=16))
    fig.update_yaxes(range=[0,100],title="Accuracy (%)"); fig.update_layout(title="Sentiment Classification Accuracy (50 review sample)"); clean_fig(fig,380); st.plotly_chart(fig, use_container_width=True)
    commentary("Few-shot prompting matched zero-shot at 76%, suggesting the LLM's pre-trained knowledge is already sufficient for this task and the examples didn't add much. Both LLM approaches beat VADER's 68%. In a production setting, the LLM would be more expensive to run at scale (6+ seconds per review vs. milliseconds for VADER), so the right choice depends on the volume and required accuracy.")

    section_header("6.3 Deep Review Intelligence", "The LLM reads reviews like a analyst")
    explain("Instead of just saying positive or negative, we asked it to read each review and produce a full analysis: what category does this fall into, what are the key themes, what should the team do about it, and how urgent is it. This is the kind of work that would normally require a person to sit down, read the review carefully, and write up their findings. The LLM does it in seconds.")

    st.markdown('<div class="llm-card"><h4>Review: 1 star | Denver, NC</h4>', unsafe_allow_html=True)
    st.markdown("> _\"Because this review is going on a public website, I am limiting the review to a summary statement. Overall, the buying experience was poor...\"_")
    st.markdown("**LLaMA 3.2 Analysis:**\n- **CATEGORY:** Value & Pricing\n- **SENTIMENT:** Negative\n- **KEY THEMES:** Inadequate customer service; Perception of poor value\n- **ACTION ITEMS:** Market analysis for pricing gaps; Review customer service protocols; Gather broader feedback\n- **URGENCY:** Low")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="llm-card"><h4>Review: 2 stars | Wickenburg, AZ</h4>', unsafe_allow_html=True)
    st.markdown("> _\"We were delayed over and over. The house sat completely still at frame for 5 1/2 months waited for HVAC. It took ONE day to do the HVAC when they finally showed up. 21 month build time...\"_")
    st.markdown("**LLaMA 3.2 Analysis:**\n- **CATEGORY:** Multiple Issues\n- **SENTIMENT:** Negative\n- **KEY THEMES:** Poor Communication (lengthy delay, no updates); Quality Concerns (incomplete items); Lack of Trust\n- **ACTION ITEMS:** Revise communication protocols; Review quality control; Follow-up with disappointed homeowners\n- **URGENCY:** Critical")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="llm-card"><h4>Review: 5 stars | San Tan Valley, AZ</h4>', unsafe_allow_html=True)
    st.markdown("> _\"Meeting Linda and Destiny in the Shea center was a great and helpful start for the entire experience. They were patient, informative, and always available. We loved the design...\"_")
    st.markdown("**LLaMA 3.2 Analysis:**\n- **CATEGORY:** Communication\n- **SENTIMENT:** Positive\n- **KEY THEMES:** Exceptional Sales Experience; Emotional Connection ('dream home'); Positive Brand Perception\n- **ACTION ITEMS:** Train sales team on emotional connection; Performance reviews; Follow-up for feedback\n- **URGENCY:** Low")
    st.markdown('</div>', unsafe_allow_html=True)
    commentary("The LLM correctly identified categories, extracts specific themes, and generates actionable next steps. The urgency ratings are appropriate: the 21-month delayed build is flagged as Critical while the brief negative summary is Low. At scale, this could process hundreds of reviews into a structured database of categorized issues and action items.")

    section_header("6.4 AI-Generated Executive Briefing", "LLaMA reads 20 recent reviews, writes a 2-minute briefing")
    explain("For this final piece, we gave the LLM the 20 most recent customer reviews and asked it to write a two-minute executive briefing. No human edited the output. This is what it produced on its own.")
    st.markdown("*Based on reviews from Feb 2026 to Mar 2026*")
    st.markdown("""**OVERALL PULSE:** Customers are extremely satisfied overall (70% five-star), but with notable concerns on quality and warranty service.

**TOP POSITIVES:**
- **Excellent Customer Service:** Sales team, site supervisors, and reps consistently praised for professionalism, kindness, and responsiveness.
- **Well-Designed Homes:** Customers express satisfaction with design and layout, some saying it exceeded expectations.
- **Positive Communication:** Clear, transparent communication throughout the buying process, from initial contact to move-in.

**TOP CONCERNS:**
- **Quality and Construction:** Concerns about finishes, construction, and materials not meeting expectations.
- **Home Warranty Issues:** Warranty team lacking construction experience; difficulty understanding homeowner complaints.
- **Post-Close Issues:** Water pressure problems, difficulty resolving issues through customer service after close.

**REGIONAL NOTES:**
- **California:** Most positive (~80% five-star). **Arizona:** Also strong (~70% five-star).
- **Colorado & Nevada:** Slightly lower satisfaction (~60% five-star).

**RECOMMENDED ACTIONS:**
1. **Improve Quality Control:** Enhance QC processes; additional training for construction teams.
2. **Strengthen Warranty Team:** Training and support for warranty reps to resolve complex issues.
3. **Proactive Post-Close Resolution:** Dedicated resources and clear process for post-close problems.""")
    commentary("This briefing was generated entirely by the LLM after reading the 20 most recent reviews. It correctly identifies the core pattern that runs through the entire analysis: the people and process are strengths, while construction quality and post-close warranty are the primary opportunities. The regional notes and recommended actions are specific and actionable. This kind of output could be generated automatically every month as new reviews come in, giving leadership a fresh read on customer sentiment without anyone having to manually review hundreds of submissions.")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Previous"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()
# ===================================================================
elif page == PAGES[7]:
    st.title("Review Explorer")
    explain("Search, filter, and browse individual reviews with sentiment scores and star ratings.")
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
