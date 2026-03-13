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
    .block-container { padding-top: 3rem !important; }
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
         "Part 4: Advanced NLP", "Part 5: Predictive Models", "Conclusion", "Review Explorer"]


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

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)


# ===================================================================
if page == PAGES[0]:
    st.title("Shea Homes Customer Review Project")
    explain("This project analyzes 2,039 customer reviews of Shea Homes collected from NewHomeSource.com. NewHomeSource is a review platform where verified homebuyers rate their builder after closing. Each review includes an overall star rating (1-5) plus four sub-ratings: Quality, Trustworthiness, Value, and Responsiveness. Buyers also write open-ended comments describing their experience. This dataset contains reviews for Shea Homes collected between September 2020 and March 2026, covering 11 markets across Arizona, California, Colorado, Nevada, North Carolina, Texas, Washington, and more. <br> <br> Reading 2,039 reviews manually would take weeks. Even then, it would be hard to spot patterns consistently, like which markets are struggling, what topics keep coming up in negative reviews, or which customers might be at risk of leaving bad word-of-mouth. This project automates that work using a multi-method Natural Language Processing (NLP) framework. The pipeline includes: (1) sentiment classification using VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based model tuned for customer review text; (2) topic extraction using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and LDA (Latent Dirichlet Allocation), which surface recurring themes without predefined categories; (3) predictive modeling using machine learning classifiers to flag at-risk reviews; and (4) contextual analysis using a locally-deployed large language model (LLaMA 3.2 via Ollama) that reads individual reviews and generates recommendations.")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Reviews", f"{len(fdf):,}"); c2.metric("Avg Rating", f"{fdf['total_score'].mean():.2f}"); c3.metric("Positive", f"{(fdf['vader_label']=='Positive').mean():.0%}")
    c4.metric("Negative", f"{(fdf['vader_label']=='Negative').mean():.0%}"); c5.metric("At-Risk", f"{(fdf['total_score']<=3).sum():,}"); c6.metric("Markets", f"{fdf['state'].nunique()}")
    cats = ["total_score","quality","trustworthiness","value","responsiveness"]; clabels = ["Overall","Quality","Trust","Value","Responsiveness"]
    cmeans = [fdf[c].mean() for c in cats]
    fig = go.Figure(go.Bar(x=clabels, y=cmeans, marker_color=[SHEA_GOLD if v==max(cmeans) else (NEG_RED if v==min(cmeans) else SHEA_BLUE) for v in cmeans], text=[f"{v:.2f}" for v in cmeans], textposition="outside", textfont_size=14))
    fig.update_yaxes(range=[0,5.5]); fig.update_layout(title="Shea Homes Average Scores by Category"); clean_fig(fig, 400); st.plotly_chart(fig, use_container_width=True)
    commentary("Across 2,039 customer reviews, Shea Homes maintains an overall rating of 4.21 out of 5, with 78% of customers awarding four or five stars. Performance is consistent across categories, including quality, trust, value, and responsiveness, with trust receiving the highest scores while quality rates slightly lower than the other dimensions. <br> <br> Within the dataset, 449 reviews (22%) fall into the 1–3 star range, representing customers who reported dissatisfaction with some aspect of their experience. These “at-risk” reviews are particularly important from an operational perspective and serve as the primary focus for the deeper text and sentiment analysis conducted in later sections.")
    st.markdown(f"**Data Source:** [NewHomeSource](https://www.newhomesource.com/builder/shea-homes/reviews/612/) &nbsp;|&nbsp; **Date Range:** {df['date'].min().strftime('%B %Y')} to {df['date'].max().strftime('%B %Y')} &nbsp;|&nbsp; **Author:** Griffin Snider")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Back"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()

# ===================================================================
elif page == PAGES[1]:
    st.title("Part 1: Summary Statistics")
    explain("Before applying modeling or artificial intelligence methods, the analysis begins with summary statistics. This step looks at the basic structure of the dataset, including the total number of reviews, how ratings are distributed across the 1–5 star scale, which states and cities generate the most feedback, and the typical length of customer reviews. Establishing these baseline patterns provides context for the more advanced analyses that follow.")
    section_header("1.1 Dataset Overview")
    c1,c2,c3,c4 = st.columns(4); c1.metric("Total Reviews",f"{len(fdf):,}"); c2.metric("Total Words",f"{fdf['word_count'].sum():,}"); c3.metric("Avg Length",f"{fdf['word_count'].mean():.1f} words"); c4.metric("Unique Cities",f"{fdf['location'].nunique()}")
    c6,c7,c8 = st.columns(3); c6.metric("Median Length",f"{fdf['word_count'].median():.0f} words"); c7.metric("Shortest",f"{fdf['word_count'].min()} words"); c8.metric("Longest",f"{fdf['word_count'].max()} words")
    commentary("The dataset contains 2,039 reviews from 1,510 unique reviewers, totaling 117,979 words. Reviews average 57.9 words, but the median is just 34 words, meaning most are brief while a smaller number contain detailed feedback. The reviews span 74 cities across 11 states. Length ranges from 2 words to 1,092 words, capturing everything from quick ratings to detailed accounts of the homebuying experience.")

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
    commentary("The distribution is heavily skewed with 5-star reviews making up 57%, while 1-star reviews account for only 4%. All five categories average above 4.0, with Trust scoring highest (4.21) and Quality and Value slightly lower (4.06). Value perceptions, specifically how customers weigh cost against what was delivered, may represent an area worth watching.")

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

    section_header("1.4 Geographic Breakdown")
    explain("These charts examine the geographic distribution of customer feedback. The left chart shows the number of reviews submitted from each state, showing where the largest share of customer feedback originates. The right chart shows the average star rating by state, calculated only for states with at least 10 reviews.")
    col1, col2 = st.columns(2)
    with col1:
        stc = fdf["state"].value_counts().reset_index(); stc.columns=["state","count"]
        fig = px.bar(stc,y="state",x="count",orientation="h",color_discrete_sequence=[SHEA_BLUE],text="count"); fig.update_traces(textposition="inside"); fig.update_layout(title="Reviews by State"); clean_fig(fig,max(350,len(stc)*45)); st.plotly_chart(fig, use_container_width=True)
    with col2:
        sts = fdf.groupby("state")["total_score"].agg(["mean","count"]).reset_index(); sts = sts[sts["count"]>=10].sort_values("mean")
        sts["color"] = sts["mean"].apply(lambda v: POS_GREEN if v>=4 else (NEU_YELLOW if v>=3 else NEG_RED))
        fig = go.Figure(go.Bar(y=sts["state"],x=sts["mean"],orientation="h",marker_color=sts["color"],text=[f"{v:.2f}" for v in sts["mean"]],textposition="inside"))
        fig.add_vline(x=4.0,line_dash="dash",line_color="gray",opacity=0.4); fig.update_xaxes(range=[0,5.5]); fig.update_layout(title="Avg Score by State (10+ reviews)"); clean_fig(fig,max(350,len(sts)*45)); st.plotly_chart(fig, use_container_width=True)
    commentary("Most reviews originate from California (684) and Arizona (532), which together account for roughly 60% of the dataset. These states represent the largest share of Shea Homes customer feedback on NewHomeSource. Average ratings across most states remain above 4.0, indicating generally strong satisfaction across markets. A few states show slightly lower averages, which may reflect differences in local operations, project timelines, or customer expectations. States with very small sample sizes (South Carolina) are excluded from the average score comparison to avoid misleading results.")

    section_header("1.5 Top Cities")
    explain("This chart identifies the cities that generate the largest volume of customer reviews. Each bar represents the number of reviews submitted from a specific city.")
    col1, = st.columns(1)
    with col1:
        tc = fdf["location"].value_counts().head(15).reset_index(); tc.columns=["city","count"]
        fig = go.Figure(go.Bar(y=tc["city"][::-1],x=tc["count"][::-1],orientation="h",marker_color=SHEA_BLUE,text=tc["count"][::-1],textposition="outside"))
        fig.update_layout(title="Top 15 Cities Review Counts"); fig.update_xaxes(range=[0,tc["count"].max()*1.25]); clean_fig(fig,500); st.plotly_chart(fig, use_container_width=True)
    commentary("Customer feedback is concentrated in a small number of communities. Cities such as Wickenburg, AZ; Las Vegas, NV; Rio Verde, AZ; and San Tan Valley, AZ contribute the highest number of reviews. These locations represent the areas where the dataset contains the most direct customer experience information.")

    section_header("1.6 Rating Correlations")
    explain("This heatmap shows the correlation between the five rating categories: Overall, Quality, Trust, Value, and Responsiveness. Correlation measures how closely two variables move together. Values closer to 1.0 indicate a strong relationship, meaning customers tend to rate those categories similarly.")
    scols = ["total_score","quality","trustworthiness","value","responsiveness"]; slabs = ["Overall","Quality","Trust","Value","Responsiveness"]
    fig = px.imshow(fdf[scols].corr().values, x=slabs, y=slabs, color_continuous_scale="YlOrRd", zmin=0.5, zmax=1, text_auto=".2f")
    fig.update_layout(title="Rating Category Correlations"); clean_fig(fig,450); st.plotly_chart(fig, use_container_width=True)
    commentary("All categories are highly correlated, showing that customers who rate one dimension poorly tend to rate everything poorly. This suggests the overall experience is somewhat holistic: a bad construction experience drags down trust, value, and responsiveness perceptions too.")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Back"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()
# ===================================================================
elif page == PAGES[2]:
    st.title("Part 2: Data Evaluation")
    explain("Before conducting analysis, the dataset must first be evaluated for quality, representativeness, and potential limitations. Key considerations include whether 2,039 reviews constitute an adequate sample size, and whether any biases or gaps in coverage may affect the findings. This section provides a transparent assessment of the dataset's strengths and limitations prior to deriving recommendations.")
    section_header("2.1 Business Question")
    explain("<b>Core business question:</b> What do customer reviews show about the Shea Homes homebuying experience, and which aspects of that experience represent the largest opportunity for improvement?")

    section_header("2.2 Sample Size Assessment")
    explain("This section evaluates whether the dataset is large enough to support reliable conclusions. It summarizes the total number of reviews, the average rating, and how reviews are distributed across states.")
    n=len(fdf); mu=fdf["total_score"].mean(); sd=fdf["total_score"].std(); me=1.96*(sd/np.sqrt(n)); ci_lo=mu-me; ci_hi=mu+me
    c1,c2,c3=st.columns(3); c1.metric("Sample Size",f"{n:,}"); c2.metric("Mean Score",f"{mu:.3f}"); c3.metric("95% CI",f"[{ci_lo:.3f}, {ci_hi:.3f}]")
    st.markdown("**Per-state sample sizes:**")
    st.dataframe(fdf["state"].value_counts().reset_index().rename(columns={"index":"State","state":"State","count":"Reviews"}), use_container_width=True, hide_index=True)
    commentary("The dataset contains 2,039 reviews with an average rating of 4.21. To estimate how precise this average is, we calculate a 95% confidence interval, which represents the range where the true average customer rating is likely to fall. The interval of 4.162 to 4.259 indicates that the estimated average rating is statistically stable. Review counts vary across states. Arizona and California account for the largest share of feedback, while several states have smaller samples. Locations with fewer reviews provide less statistical certainty and should be interpreted cautiously.")

    section_header("2.3 Potential Biases")
    explain("Online review datasets often contain structural biases that can influence how results should be interpreted. This section identifies several common sources of bias within the dataset and explains how they may affect the conclusions drawn from the analysis.")
    st.dataframe(pd.DataFrame([
        ["Self-selection bias","Customers with strong opinions more likely to review","May overrepresent extremes"],
        ["Geographic concentration","CA and AZ account for ~60% of reviews","State-level conclusions should note sample sizes"],
        ["Temporal skew","2021-2022 peak volume (post-COVID boom)","Trends may reflect market conditions"],
        ["Platform bias","TrustBuilder is builder-partnered","Sentiment may be inflated vs independent sites"],
    ], columns=["Bias","Description","Impact"]), use_container_width=True, hide_index=True)
    commentary("The most significant is platform bias. Because the reviews originate from a builder-partnered platform, overall ratings may skew more positive than reviews found on independent consumer sites. As a result, the negative reviews that do appear in the dataset are particularly informative. They represent customers whose dissatisfaction was strong enough to be expressed despite the generally positive environment of the platform.")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Back"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()
# ===================================================================
elif page == PAGES[3]:
    st.title("Part 3: Preliminary Sentiment Analysis")
    explain("Sentiment analysis is a natural language processing technique that allows a algorithm to evaluate written feedback and estimate the emotional tone of the text. Instead of manually reading thousands of reviews, algorithms analyze the words used in each comment to determine whether the overall message is positive, negative, or neutral. Words associated with positive experiences (such as “great” or “helpful”) increase the sentiment score, while words associated with problems or frustration decrease it. <br> <br> Each review is converted into a numerical sentiment score ranging from −1 to +1, where −1 represents very negative language and +1 represents very positive language. To increase reliability, this analysis uses two widely used sentiment models. VADER (Valence Aware Dictionary and sEntiment Reasoner) is designed specifically for social media and review text and accounts for emphasis such as capitalization, punctuation, and emotional wording. TextBlob is a general-purpose language model that evaluates sentiment based on the balance of positive and negative terms within the text. Using two independent methods allows the analysis to compare results and confirm that the patterns observed in customer sentiment are consistent.")

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
    commentary("Both models produce similar results, showing that the majority of reviews contain positive language. VADER classifies about 78% of reviews as positive, while TextBlob identifies roughly 80% as positive. VADER detects more negative reviews than TextBlob, reflecting its stronger sensitivity to negative wording in review-style text. <br> <br>The sentiment vs. star rating chart provides a validation check. Sentiment scores increase steadily from 1-star to 5-star reviews, confirming that both models are interpreting the language in a way that aligns with the rating customers assigned.")

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

    section_header("3.3 Sentiment by State")
    explain("This chart compares customer sentiment across states using the VADER sentiment score. Each bar represents the average sentiment of review text within a state. Higher scores indicate that customers in that market tend to describe their experience using more positive language.")
    ss = fdf.groupby("state").agg(avg_v=("vader_compound","mean"),avg_s=("total_score","mean"),cnt=("total_score","count")).reset_index()
    ss = ss[ss["cnt"]>=10].sort_values("avg_v"); ss["color"]=ss["avg_v"].apply(lambda v: POS_GREEN if v>=0.5 else (NEU_YELLOW if v>=0.3 else NEG_RED))
    fig = go.Figure(go.Bar(y=ss["state"],x=ss["avg_v"],orientation="h",marker_color=ss["color"],text=[f"{v:.3f} (avg {s:.1f}, n={c})" for v,s,c in zip(ss["avg_v"],ss["avg_s"],ss["cnt"])],textposition="outside"))
    fig.update_xaxes(range=[0,ss["avg_v"].max()+0.25]); fig.update_layout(title="Sentiment by State"); clean_fig(fig,max(380,len(ss)*50)); st.plotly_chart(fig, use_container_width=True)
    commentary("Most states show consistently positive sentiment, with scores clustering between 0.45 and 0.55. Colorado and Texas rank among the highest, showing positive review language in those markets. <br> <br> North Carolina and Idaho appear lower in comparison, suggesting that customer experiences in those markets may warrant closer examination. California and Arizona, the two markets with the largest number of reviews, remain positive, showing stable customer sentiment in Shea Homes’ largest operating regions.")

    section_header("3.4 Negative vs Positive Word Frequency")
    explain("This analysis examines the most frequently used words in positive and negative reviews. The charts compare language used in 1–2 star reviews with language used in 4–5 star reviews. By analyzing which terms appear most often in each group, we can identify the themes that customers associate with positive experiences and the issues that appear most often in negative feedback.")
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
    commentary("Negative reviews frequently reference words such as issues, quality, warranty, time, and construction, indicating that dissatisfaction is often related to build quality or delays in resolving problems after purchase. In contrast, positive reviews emphasize words such as experience, team, process, and service, suggesting that customers frequently highlight interactions with staff and the overall buying process when describing a positive experience. This contrast suggests that customer-facing interactions are a key strength, while product quality and issue resolution appear more often in negative feedback.")

    section_header("3.5 Distinctive Negative Review Words")
    explain("This table identifies words that appear disproportionately often in negative reviews compared with positive reviews. Instead of simply counting the most common words, the analysis measures how much more frequently a word appears in complaints than in positive feedback. Words with high overrepresentation scores are strongly associated with dissatisfied customer experiences.")
    if len(neg_t)>5 and len(pos_t)>5:
        dist = get_neg_distinctive(neg_t, pos_t, sw_list)
        if dist: st.dataframe(pd.DataFrame(dist, columns=["Word","Count","Overrepresentation"]).assign(**{"Overrepresentation": lambda d: d["Overrepresentation"].map("{:.1f}x".format)}), use_container_width=True, hide_index=True)
    commentary("Words like poor (192.7x more common in negative reviews), installed (140.6x), cabinets (109.4x), and flooring (83.3x) point to specific construction pain points. The word waiting (109.4x) and months suggest delays are a major theme. These are not abstract complaints; they are about specific, fixable things: cabinets installed wrong, flooring issues, and long wait times for repairs.")

    section_header("3.6 Score vs. Sentiment Mismatch")
    explain("This section identifies reviews where the star rating and the language of the review do not align. In most cases, higher star ratings are associated with positive wording, while lower ratings contain more negative language. When these signals disagree, it can show more complexity in the customer experience that star ratings alone may not capture. <br> <br> One important category is high-star reviews with negative sentiment in the text. These reviews often show that a customer was generally satisfied but still experienced specific problems worth noting. Identifying these cases helps surface issues that may otherwise be overlooked when focusing only on low star ratings.")
    high_neg = fdf[(fdf["total_score"]>=4)&(fdf["vader_compound"]<-0.05)]; low_pos = fdf[(fdf["total_score"]<=2)&(fdf["vader_compound"]>0.5)]
    c1,c2,c3 = st.columns(3); c1.metric("Total Mismatches",f"{fdf['mismatch'].sum()} ({fdf['mismatch'].mean():.1%})"); c2.metric("High Stars + Neg Text",f"{len(high_neg)}"); c3.metric("Low Stars + Pos Text",f"{len(low_pos)}")
    if len(high_neg)>0:
        st.markdown("**Hidden Complaints (samples):**")
        for _, r in high_neg.sort_values("vader_compound").head(3).iterrows():
            st.markdown(f"> **{r['total_score']} stars** | VADER: {r['vader_compound']:.3f} | {r['location']}  \n> _{str(r['review_text'])[:300]}..._")
    commentary("A total of 146 reviews (7.2% of the dataset) show a mismatch between the rating and the sentiment expressed in the text. The majority of these cases, 110 reviews, are high-star ratings paired with negative language, indicating customers who reported specific issues despite assigning an overall positive score. These reviews are particularly valuable from an operational perspective. Because the customer still left a favorable rating, the relationship is largely intact, yet the written feedback highlights clear opportunities for improvement, such as subcontractor performance, installation quality, or post-closing service.")

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

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Back"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()
# ===================================================================
elif page == PAGES[4]:
    st.title("Part 4: Advanced Natural Language Processing Analysis")
    explain("This section applies advanced natural language processing (NLP) techniques to find insights from customer reviews. Rather than only measuring whether reviews are positive or negative, these methods analyze the text to identify recurring themes, frequently discussed topics, and patterns in how customers describe their experiences. <br> <br> The analysis also looks for which aspects of the homebuying experience customers mention most often, such as sales interactions, construction quality, or warranty service. By analyzing groups of words and phrases together, these techniques show more detailed patterns in customer feedback and help find the specific areas of the business that drive satisfaction or dissatisfaction.")

    section_header("4.1 Topic Discovery (Latent Dirichlet Allocation (LDA))")
    explain("This section uses a technique called topic modeling to automatically identify the main subjects customers discuss in their reviews. The specific method used, Latent Dirichlet Allocation (LDA), analyzes patterns of words that frequently appear together across the dataset. Reviews that contain similar groups of words are grouped into a shared topic, allowing the model to discover common themes without being manually labeled in advance. <br> <br> By applying this approach to all reviews, the model identifies the major themes in customer feedback, such as construction quality, the buying process, sales interactions, and post-purchase issues. This provides a structured overview of what customers talk about most often.")
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
    commentary("The algorithm discovered 6 distinct topics without any guidance. Construction quality appears most frequently, representing the largest share of customer discussion. Topics related to the buying process and sales experience also appear often and tend to receive the highest satisfaction scores. In contrast, reviews grouped under issues and problems show noticeably lower satisfaction levels. This pattern suggests that while the sales and purchasing experience is generally well received, dissatisfaction is more commonly associated with construction defects or post-delivery service issues.")

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
    explain("Earlier analysis looked at individual words, but meaning often comes from combinations of words. For example, the phrase “not responsive” carries the opposite meaning of the single word “responsive,” even though the word itself is positive. <br> <br> This section analyzes common two-word and three-word phrases that appear frequently in positive and negative reviews. Looking at phrases instead of individual words provides a clearer picture of what customers are actually describing in their experiences.")
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
    commentary("The negative phrases highlight specific operational pain points. Terms related to customer service, build quality, and warranty support appear frequently in lower-rated reviews, indicating that dissatisfaction often centers around post-purchase issues or construction defects. <br> <br> In contrast, the most common positive phrases focus on the sales team, the buying experience, and the overall building process. Phrases like “great experience,” “customer service,” and “start to finish” appear frequently in 5-star reviews, suggesting that customers are especially satisfied when the entire homebuying journey feels smooth and well managed from beginning to end.")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Back"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()
# ===================================================================
elif page == PAGES[5]:
    st.title("Part 5: Predictive Models")
    explain("<b>The Big Question:</b> Can machine learning predict customer satisfaction from the words in a review alone?<br><br><b>Why it matters:</b> All of the earlier analysis looked at reviews that already had star ratings attached. This section explores a different question: if we only had the written text of a review, could a algorithm determine whether the customer was satisfied or dissatisfied? This is important because much customer feedback, such as survey comments, emails, support tickets, or social media posts, often comes without a rating. If we can train a model to read text and estimate satisfaction, that approach could be applied to any source of written feedback, even when no numerical score is provided. To test this, several machine learning models and evaluated how accurately they could predict star ratings based only on the review text.<br><br><b>Approach:</b> I trained three common machine learning models to learn patterns between review text and star rating <br> <br> 1) Logistic Regression: A statistical model that estimates the probability that a review belongs to a category based on its words. <br> 2) Random Forest: An ensemble of many small decision trees that vote on the predicted category. <br> 3) Gradient Boosting: A sequence of decision trees where each new tree focuses on correcting the mistakes of the previous one. <br> <br> Then I trained three different models on 80 percent of the reviews, where the model could see both the text and the star rating and learn the patterns. After training, I tested each model on the remaining 20 percent of reviews that it had never seen before, giving it only the text and asking it to predict the customer's star rating based on text alone.")

    section_header("5.1 At-Risk Customer Detection (Binary)", "Satisfied (4-5 star) vs At-Risk (1-3 star)")
    explain("This analysis simplifies the prediction problem into two groups: satisfied customers who gave four or five stars, and at-risk customers who gave one to three stars. The goal is to determine whether the language in a review alone can identify customers who may be dissatisfied. <br> <br> The left chart shows overall model performance. Accuracy measures the percentage of reviews the model classified correctly. Macro F1 is a balanced metric that evaluates how well the model performs across both groups rather than favoring the larger group of satisfied customers. <br> <br>The right chart focuses specifically on detecting at-risk customers. At-Risk Recall measures how many dissatisfied customers the model successfully identifies. At-Risk Precision measures how often the model is correct when it flags a review as at-risk. ")

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
    commentary("Gradient Boosting is the most accurate with 84.1% accuracy and the highest Macro F1 (0.757). It catches 58% of at-risk customers from text alone, with 66% precision, meaning when it flags someone as at-risk, it is right 2 out of 3 times. Logistic Regression has better at-risk recall (72%) but more false positives (56% precision). Random Forest is the weakest here because it struggles with the minority class, only catching 43% of at-risk customers despite high overall accuracy. The tradeoff between recall and precision depends on the business use case: if missing an at-risk customer is costly, Logistic Regression's higher recall may be preferable.")

    section_header("5.2 Three-Class Prediction", "Negative (1-2 star) vs Neutral (3 star) vs Positive (4-5 star)")
    explain("This task is more challenging than the previous prediction. Instead of simply identifying satisfied versus at-risk customers, the model must now classify reviews into three groups: negative (1–2 stars), neutral (3 stars), and positive (4–5 stars). <br> <br> This is significantly harder because 3-star reviews are inherently ambiguous. Customers in this group are often neither strongly satisfied nor strongly dissatisfied, and their language tends to reflect a mix of positive and negative comments.<br> <br> The left chart again shows overall performance using accuracy and Macro F1. These metrics summarize how well the model performs across all three categories. <br> <br> The right chart shows recall for each rating group. These values indicate how often the model correctly identifies negative, neutral, and positive reviews individually. This chart helps reveal whether a model is truly recognizing each category or simply predicting the most common outcome.")

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
    commentary("These results show why overall accuracy alone is not sufficient for evaluating model performance. Gradient Boosting achieved the highest overall accuracy at 79.2%, but looking at the recall by class reveals important differences between the models. Random Forest performed poorly at identifying negative and neutral reviews, capturing only 3% of negative reviews and 2% of neutral reviews. In practice, it was largely predicting most reviews as positive and benefiting from the fact that positive reviews dominate the dataset. Logistic Regression provided the most balanced performance, correctly identifying 58% of negative reviews and 37% of neutral reviews, making it more useful for detecting dissatisfied customers. The key takeaway is that while predicting customer sentiment from text is feasible, neutral reviews remain the most difficult category to classify, since the language in 3-star reviews often contains a genuine mix of positive and negative signals.")

    section_header("5.3 Most Predictive Words by Category", "What language signals each rating level?")
    explain("When the prediction model evaluates a review, certain words influence its decision more than others. This section highlights the words that had the strongest impact on predicting each rating category. In simple terms, this analysis asks: which words most strongly signal that a review is negative, neutral, or positive? Longer bars indicate that a word had a stronger influence on the model’s prediction.")

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
            if st.button("⇽ Back"):
                previous_page()

    with col3:
        if page != PAGES[-1]:
            if st.button("Next ⇾"):
                next_page()
# ===================================================================
elif page == PAGES[6]:
    st.title("Conclusion")
    st.markdown("---")
    explain(
        "This project analyzed 2,039 verified customer reviews of Shea Homes using a multi-method NLP framework spanning summary statistics, sentiment analysis, topic modeling, predictive machine learning. The goal was to demonstrate how unstructured customer feedback can be transformed into structured, actionable intelligence at scale. Below is a summary of the key findings from each stage of the analysis, followed by strategic recommendations.")

    # --- Key Findings ---
    section_header("Key Findings", "What the data revealed across five stages of analysis")

    finding(
        "<b>1. Overall Customer Satisfaction Is Strong but Not Uniform</b><br>Shea Homes holds an average rating of 4.21 out of 5, with 78% of reviews at 4 or 5 stars. However, 449 reviews (22%) fall in the 1–3 star range, representing a meaningful segment of at-risk customers. All five rating dimensions (Overall, Quality, Trust, Value, Responsiveness) are highly correlated, meaning a poor experience in one area tends to drag down perceptions across the board.")

    finding(
        "<b>2. Sentiment Analysis Confirms and Extends Star Ratings</b><br>VADER sentiment analysis classified 76% of reviews as positive, 16% as negative, and 8% as neutral. Importantly, sentiment scores revealed nuance that star ratings alone cannot capture. Some 4- and 5-star reviews contained negative language about specific issues, while a few low-rated reviews still acknowledged positive aspects of the experience. This means relying solely on star ratings can miss customers who are satisfied overall but frustrated with specific touchpoints.")

    finding(
        "<b>3. Recurring Pain Points Are Specific and Actionable</b><br>Topic modeling and keyword analysis consistently surfaced the same friction areas: warranty responsiveness and post-close follow-through, construction quality concerns (paint, drywall, appliances, workmanship), and communication gaps during the building process. Negative reviews used concrete, specific language ('repairs,' 'months,' 'workmanship,' 'appliances'), while positive reviews used emotional, relational language ('excellent,' 'professional,' 'helpful,' 'team'). The distinction suggests that negative experiences are driven by tangible, fixable process failures rather than abstract dissatisfaction.")

    finding(
        "<b>4. Machine Learning Can Predict Satisfaction from Text Alone</b><br>Binary classification (Satisfied vs. At-Risk) achieved 84.1% accuracy using Gradient Boosting, with the model correctly flagging 58% of at-risk customers from text alone at 66% precision. Three-class prediction (Negative, Neutral, Positive) proved harder at 79.2% accuracy, with neutral reviews being the most difficult to classify due to their mixed-signal language. These models demonstrate that predictive text classification is viable for customer feedback that arrives without a numerical rating, such as emails, survey comments, or social media posts.")

    finding(
        "<b>5. The Most Informative Signals Come from Specific Language</b><br>Across every method, from TF-IDF to logistic regression coefficients to topic models, the most predictive and diagnostic signals came from domain-specific vocabulary. Words like 'warranty,' 'repairs,' 'workmanship,' and 'months' consistently flagged dissatisfaction, while 'professional,' 'team,' 'helpful,' and 'awesome' signaled satisfaction. This confirms that targeted keyword monitoring, even without full ML pipelines, can provide early warning of emerging issues.")

    # --- Strategic Recommendations ---
    section_header("Strategic Recommendations", "How these findings could translate into action")

    commentary(
        "<b>1. Implement a Real-Time Review Monitoring Pipeline</b><br>The NLP methods demonstrated in this project, specifically VADER sentiment scoring and keyword flagging, could be deployed as an automated pipeline that scores each new review as it arrives. Reviews falling below a sentiment threshold or containing high-risk keywords (e.g., 'warranty,' 'months,' 'repairs') could be automatically routed to the appropriate regional team for follow-up. This would reduce the lag between a customer expressing dissatisfaction and the company responding to it.")

    commentary(
        "<b>2. Prioritize Warranty and Post-Close Communication</b><br>The most consistent theme in negative reviews across every analytical method was frustration with warranty responsiveness and post-close follow-through. Customers who rated construction quality poorly often cited delays in getting issues resolved after move-in, not necessarily the defects themselves. Strengthening the warranty response process and setting clear timelines for issue resolution could meaningfully shift perception in this segment.")

    commentary(
        "<b>3. Use Predictive Models to Flag Unrated Feedback</b><br>The machine learning classifiers trained in this project could be applied to customer feedback channels that do not include star ratings, such as warranty service tickets, post-closing survey comments, or email correspondence. By scoring this text-only feedback with the same models, Shea Homes could identify at-risk customers earlier in the lifecycle, before they reach the point of writing a negative public review.")

    commentary(
        "<b>4. Track Sentiment Trends at the Market and Community Level</b><br>Aggregated national ratings can mask localized issues. The geographic analysis showed variation in satisfaction across states and cities. Running sentiment analysis at the community or project level on a recurring basis would allow regional managers to identify emerging issues before they become systemic patterns, and would provide data-driven evidence for resource allocation decisions.")

    # --- Closing ---
    section_header("Final Note")

    explain(
        "This project was built as a portfolio demonstration of applied NLP and machine learning using real-world customer data. The full analytical pipeline, from data collection through web scraping to predictive modeling and LLM-powered review intelligence, was developed independently by Griffin Snider.<br><br>The techniques shown here are not theoretical. Every method used in this project, from VADER sentiment scoring to TF-IDF topic extraction to Gradient Boosting classification, can be deployed on live data in a production environment. The goal was to show what becomes possible when customer feedback is treated not as a static archive, but as a continuous, analyzable data source.<br><br>Thank you for reviewing this project.")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if page != PAGES[0]:
            if st.button("⇽ Back"):
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
