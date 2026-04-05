import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
from collections import Counter
from scipy.sparse import hstack, csr_matrix

from utils.config import (
    BUILDER_DISPLAY_NAMES,
    VADER_POS_THRESHOLD,
    VADER_NEG_THRESHOLD,
    TEXTBLOB_POS_THRESHOLD,
    TEXTBLOB_NEG_THRESHOLD,
    MISMATCH_NEG_SENTIMENT,
    MISMATCH_POS_SENTIMENT,
    SATISFIED_MIN_STARS,
    NEGATIVE_MAX_STARS,
    NEUTRAL_STARS,
    RANDOM_STATE,
    LDA_N_TOPICS,
    LDA_MAX_FEATURES,
    LDA_MIN_DF,
    LDA_MAX_DF,
    LDA_MAX_ITER,
    LDA_NGRAM_RANGE,
    MIN_EMPLOYEE_MENTIONS,
    SPACY_BATCH_SIZE,
    ENTITY_CONTEXT_WINDOW,
    DISTINCTIVE_MIN_RATIO,
    DISTINCTIVE_MIN_COUNT,
)


def _get_sia():
    """Initialise VADER once and return the analyser."""
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def _add_base_columns(df, sia):
    """Shared processing applied to every builder CSV: dates, VADER, labels."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["word_count"] = df["review_text"].apply(lambda x: len(str(x).split()))
    df["state"] = df["location"].str.extract(r",\s*([A-Z]{2})$")
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    scores = df["review_text"].apply(lambda x: sia.polarity_scores(str(x)))
    df["vader_compound"] = scores.apply(lambda x: x["compound"])
    df["vader_pos"] = scores.apply(lambda x: x["pos"])
    df["vader_neg"] = scores.apply(lambda x: x["neg"])
    df["vader_neu"] = scores.apply(lambda x: x["neu"])
    df["vader_label"] = df["vader_compound"].apply(
        lambda x: "Positive" if x >= VADER_POS_THRESHOLD else ("Negative" if x <= VADER_NEG_THRESHOLD else "Neutral")
    )
    df["risk_class"] = df["total_score"].apply(
        lambda x: "Satisfied (4-5)" if x >= SATISFIED_MIN_STARS else "At-Risk (1-3)"
    )
    return df


@st.cache_data(show_spinner="Loading reviews...")
def load_and_process(path):
    sia = _get_sia()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = _add_base_columns(df, sia)
    # Shea-only extras
    df["char_count"] = df["review_text"].apply(lambda x: len(str(x)))
    df["year"] = df["date"].dt.year
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    from textblob import TextBlob
    df["textblob_polarity"] = df["review_text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["textblob_subjectivity"] = df["review_text"].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    df["textblob_label"] = df["textblob_polarity"].apply(
        lambda x: "Positive" if x > TEXTBLOB_POS_THRESHOLD else ("Negative" if x < TEXTBLOB_NEG_THRESHOLD else "Neutral")
    )
    df["exclamation_count"] = df["review_text"].apply(lambda x: str(x).count("!"))
    df["mismatch"] = (
        ((df["total_score"] >= SATISFIED_MIN_STARS) & (df["vader_compound"] < MISMATCH_NEG_SENTIMENT))
        | ((df["total_score"] <= NEGATIVE_MAX_STARS) & (df["vader_compound"] > MISMATCH_POS_SENTIMENT))
    )
    return df


@st.cache_data(show_spinner="Loading all builder reviews...")
def load_all_builders():
    import glob as _glob
    sia = _get_sia()

    frames = []
    review_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "builder_reviews")
    for path in sorted(_glob.glob(os.path.join(review_dir, "*_reviews.csv"))):
        slug = os.path.basename(path).replace("_reviews.csv", "")
        bdf = pd.read_csv(path, encoding="utf-8-sig")
        bdf = _add_base_columns(bdf, sia)
        bdf["builder"] = BUILDER_DISPLAY_NAMES.get(slug, slug)
        frames.append(bdf)
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner="Computing model predictions...")
def compute_model_results(df, _model_mtime=None):
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    meta = joblib.load(os.path.join(model_dir, "metadata.joblib"))
    extra_features = meta["extra_features"]

    def _load(name):
        return joblib.load(os.path.join(model_dir, name))

    def _build_features(X_texts, X_extras, tfidf):
        text_feat = tfidf.transform(X_texts)
        return hstack([text_feat, csr_matrix(X_extras.values)])

    df = df.copy()
    df["risk_class"] = df["total_score"].apply(lambda x: "Satisfied (4-5 Stars)" if x >= SATISFIED_MIN_STARS else "At-Risk (1-3 Stars)")
    def rating_bucket(s):
        if s <= NEGATIVE_MAX_STARS: return "Negative (1-2 Stars)"
        elif s == NEUTRAL_STARS: return "Neutral (3 Stars)"
        else: return "Positive (4-5 Stars)"
    df["rating_class"] = df["total_score"].apply(rating_bucket)

    # Binary models
    tfidf_b = _load("binary_tfidf.joblib")
    test_idx_b = meta["binary_test_idx"]
    X_test_b = df.loc[test_idx_b, ["review_text"] + extra_features]
    y_test_b = df.loc[test_idx_b, "risk_class"]
    X_test_hyb = _build_features(X_test_b["review_text"].astype(str), X_test_b[extra_features], tfidf_b)

    model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    binary = {"acc": [], "f1": [], "recall": [], "prec": [], "best_report": ""}
    best_acc = 0
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    for name in model_names:
        fname = name.lower().replace(" ", "_")
        clf = _load(f"binary_{fname}.joblib")
        y_pred = clf.predict(X_test_hyb)
        acc = accuracy_score(y_test_b, y_pred)
        f1 = f1_score(y_test_b, y_pred, average="macro")
        report = classification_report(y_test_b, y_pred, output_dict=True, zero_division=0)
        ar = report.get("At-Risk (1-3 Stars)", {})
        binary["acc"].append(round(acc * 100, 1))
        binary["f1"].append(round(f1, 3))
        binary["recall"].append(round(ar.get("recall", 0) * 100))
        binary["prec"].append(round(ar.get("precision", 0) * 100))
        if acc > best_acc:
            best_acc = acc
            binary["best_name"] = name
            binary["best_report"] = classification_report(y_test_b, y_pred, zero_division=0)

    # three class models
    tfidf_3 = _load("three_class_tfidf.joblib")
    test_idx_3 = meta["three_test_idx"]
    X_test_3 = df.loc[test_idx_3, ["review_text"] + extra_features]
    y_test_3 = df.loc[test_idx_3, "rating_class"]
    X_test_3h = _build_features(X_test_3["review_text"].astype(str), X_test_3[extra_features], tfidf_3)

    three = {"acc": [], "f1": [], "neg_r": [], "neu_r": [], "pos_r": []}
    for name in model_names:
        fname = name.lower().replace(" ", "_")
        clf = _load(f"three_{fname}.joblib")
        y_pred = clf.predict(X_test_3h)
        acc = accuracy_score(y_test_3, y_pred)
        f1 = f1_score(y_test_3, y_pred, average="macro")
        report = classification_report(y_test_3, y_pred, output_dict=True, zero_division=0)
        three["acc"].append(round(acc * 100, 1))
        three["f1"].append(round(f1, 3))
        three["neg_r"].append(round(report.get("Negative (1-2 Stars)", {}).get("recall", 0) * 100))
        three["neu_r"].append(round(report.get("Neutral (3 Stars)", {}).get("recall", 0) * 100))
        three["pos_r"].append(round(report.get("Positive (4-5 Stars)", {}).get("recall", 0) * 100))

    # feature importance from 3-class LR
    lr_3 = _load("three_logistic_regression.joblib")
    feature_names = np.array(list(tfidf_3.get_feature_names_out()) + extra_features)
    top_words = {}
    for i, label in enumerate(lr_3.classes_):
        top_idx = lr_3.coef_[i].argsort()[-10:][::-1]
        top_words[label] = [(feature_names[j], round(lr_3.coef_[i][j], 3)) for j in top_idx]

    return {"binary": binary, "three": three, "top_words": top_words}


@st.cache_resource(show_spinner=False)
def load_prediction_models(_model_mtime=None):
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    meta = joblib.load(os.path.join(model_dir, "metadata.joblib"))
    tfidf_b = joblib.load(os.path.join(model_dir, "binary_tfidf.joblib"))
    tfidf_3 = joblib.load(os.path.join(model_dir, "three_class_tfidf.joblib"))
    gb_b = joblib.load(os.path.join(model_dir, "binary_gradient_boosting.joblib"))
    gb_3 = joblib.load(os.path.join(model_dir, "three_gradient_boosting.joblib"))
    lr_3 = joblib.load(os.path.join(model_dir, "three_logistic_regression.joblib"))
    return {
        "tfidf_b": tfidf_b, "tfidf_3": tfidf_3,
        "gb_b": gb_b, "gb_3": gb_3, "lr_3": lr_3,
        "extra_features": meta["extra_features"],
    }


def predict_review(text, models):
    import nltk; from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    vader = sia.polarity_scores(str(text))
    compound = vader["compound"]
    word_count = len(str(text).split())
    excl_count = str(text).count("!")
    extra = np.array([[compound, word_count, excl_count]])

    tf_b = models["tfidf_b"].transform([str(text)])
    X_b = hstack([tf_b, csr_matrix(extra)])
    binary_label = models["gb_b"].predict(X_b)[0]
    binary_proba = models["gb_b"].predict_proba(X_b)[0]
    binary_classes = models["gb_b"].classes_

    tf_3 = models["tfidf_3"].transform([str(text)])
    X_3 = hstack([tf_3, csr_matrix(extra)])
    three_label = models["gb_3"].predict(X_3)[0]
    three_proba = models["gb_3"].predict_proba(X_3)[0]
    three_classes = models["gb_3"].classes_

    lr = models["lr_3"]
    feature_names = list(models["tfidf_3"].get_feature_names_out()) + models["extra_features"]
    tfidf_vec = tf_3.toarray()[0]
    class_idx = list(lr.classes_).index(three_label)
    coefs = lr.coef_[class_idx]
    full_vec = np.concatenate([tfidf_vec, extra[0]])
    active_mask = full_vec != 0
    active_indices = np.where(active_mask)[0]
    if len(active_indices) > 0:
        weighted = [(feature_names[j], coefs[j] * full_vec[j]) for j in active_indices if coefs[j] > 0]
        weighted.sort(key=lambda x: x[1], reverse=True)
        signal_words = weighted[:8]
    else:
        signal_words = []

    return {
        "vader_compound": compound,
        "vader_label": "Positive" if compound >= VADER_POS_THRESHOLD else ("Negative" if compound <= VADER_NEG_THRESHOLD else "Neutral"),
        "binary_label": binary_label,
        "binary_proba": dict(zip(binary_classes, binary_proba)),
        "three_label": three_label,
        "three_proba": dict(zip(three_classes, three_proba)),
        "signal_words": signal_words,
    }


@st.cache_data
def get_stop_words():
    import nltk; nltk.download("stopwords", quiet=True); from nltk.corpus import stopwords
    sw = set(stopwords.words("english"))
    sw.update(["home","shea","homes","new","would","one","us","also","get","got","even","like","really","much","could","said","told","went","going","still","back","made","make","well","since","every"])
    return sw


@st.cache_data
def compute_topics(texts, n_topics=LDA_N_TOPICS):
    from sklearn.feature_extraction.text import CountVectorizer; from sklearn.decomposition import LatentDirichletAllocation
    sw = get_stop_words()
    vec = CountVectorizer(max_features=LDA_MAX_FEATURES, stop_words=list(sw), min_df=LDA_MIN_DF, max_df=LDA_MAX_DF, ngram_range=LDA_NGRAM_RANGE, token_pattern=r"(?u)\b\w[\w']+\b")
    dtm = vec.fit_transform(texts.astype(str)); fnames = vec.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=RANDOM_STATE, max_iter=LDA_MAX_ITER, learning_method="online"); lda.fit(dtm)
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
        results[asp] = {"mentions":n, "pct":n/len(texts) if len(texts) else 0, "avg_sentiment":np.mean(sents) if sents else 0, "pct_negative":np.mean([s < VADER_NEG_THRESHOLD for s in sents]) if sents else 0}
    return results


@st.cache_resource(show_spinner=False)
def _load_spacy():
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])


@st.cache_data(show_spinner="Extracting employee mentions...")
def compute_employees(texts, scores, locations, states):
    import nltk; nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer; sia = SentimentIntensityAnalyzer()
    nlp = _load_spacy()
    NOT_PERSON = {"Shea", "Sheas", "Trilogy", "Covid", "Encanterra", "HOA", "Shae"}
    recs = []
    for doc, sc, loc, st_ in zip(nlp.pipe([str(t) for t in texts], batch_size=SPACY_BATCH_SIZE), scores, locations, states):
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            first = ent.text.split()[0]
            first = re.sub(r"[^A-Za-z]", "", first)
            if first in NOT_PERSON:
                continue
            if len(first) < 3 or not first[0].isupper():
                continue
            start = max(0, ent.start - ENTITY_CONTEXT_WINDOW)
            end = min(len(doc), ent.end + ENTITY_CONTEXT_WINDOW)
            ctx = doc[start:end].text
            recs.append({"name": first, "sentiment": sia.polarity_scores(ctx)["compound"],
                         "total_score": sc, "location": loc, "state": st_})
    if not recs:
        return pd.DataFrame()
    edf = pd.DataFrame(recs)
    s = edf.groupby("name").agg(
        mentions=("sentiment", "count"),
        avg_sentiment=("sentiment", "mean"),
        avg_stars=("total_score", "mean"),
        top_location=("location", lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"),
    ).sort_values("mentions", ascending=False)
    return s[s["mentions"] >= MIN_EMPLOYEE_MENTIONS]


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
        if ratio > DISTINCTIVE_MIN_RATIO and neg_w[word] >= DISTINCTIVE_MIN_COUNT: result.append((word, neg_w[word], ratio))
    return sorted(result, key=lambda x: -x[2])[:20]
