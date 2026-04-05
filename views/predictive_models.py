import streamlit as st
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import SHEA_BLUE, SHEA_GOLD, POS_GREEN, NEG_RED, NEU_YELLOW, NEUTRAL_GRAY
from utils.components import section_header, explain, commentary, static_output, clean_fig, nav_buttons
from utils.data import compute_model_results


def render(df, fdf, page):
    st.title("Part 5: Predictive Models")
    explain("<b>The Big Question:</b> Can machine learning predict customer satisfaction from the words in a review alone?<br><br><b>Why it matters:</b> All of the earlier analysis looked at reviews that already had star ratings attached. This section explores a different question: if we only had the written text of a review, could a algorithm determine whether the customer was satisfied or dissatisfied? This is important because much customer feedback, such as survey comments, emails, support tickets, or social media posts, often comes without a rating. If we can train a model to read text and estimate satisfaction, that approach could be applied to any source of written feedback, even when no numerical score is provided. To test this, I used several machine learning models and evaluated how accurately they could predict star ratings based only on the review text.<br><br><b>Approach:</b> I trained three common machine learning models to learn patterns between review text and star ratings. <br> <br> 1) Logistic Regression: A statistical model that estimates the probability that a review belongs to a category based on its words. <br> 2) Random Forest: An ensemble of many small decision trees that vote on the predicted category. <br> 3) Gradient Boosting: A sequence of decision trees where each new tree focuses on correcting the mistakes of the previous one. <br> <br> I trained these three different models on 80 percent of the reviews, where the model could see both the text and the star rating and learn the patterns. After training, I tested each model on the remaining 20 percent of reviews that it had never seen before, giving it only the text and asking it to predict the customer's star rating based on text alone.")

    # 5.1 binary
    section_header("5.1 At-Risk Customer Detection (Binary)", "Satisfied (4-5 star) vs At-Risk (1-3 star)")
    explain("This analysis simplifies the prediction problem into two groups: satisfied customers who gave four or five stars, and at-risk customers who gave one to three stars. The goal is to determine whether the language in a review alone can identify customers who may be dissatisfied. <br> <br> The left chart shows overall model performance. Accuracy (blue) measures the percentage of reviews the model classified correctly. Macro F1 (yellow) is a balanced metric that evaluates how well the model performs across both groups rather than favoring the larger group of satisfied customers. <br> <br>The right chart focuses specifically on detecting at-risk customers. At-Risk Recall (red) measures how many dissatisfied customers the model successfully identifies. At-Risk Precision (gray) measures how often the model is correct when it flags a review as at-risk. ")

    _meta_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "metadata.joblib")
    _model_mtime = os.path.getmtime(_meta_path) if os.path.exists(_meta_path) else 0
    mr = compute_model_results(df, _model_mtime=_model_mtime)
    b = mr["binary"]; t = mr["three"]

    models = ["Logistic Regression","Random Forest","Gradient Boosting"]
    acc = b["acc"]; f1 = b["f1"]; recall = b["recall"]; prec = b["prec"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Accuracy & Macro F1","At-Risk Recall & Precision"])
    fig.update_layout(title="Models")
    fig.add_trace(go.Bar(name="Accuracy %", x=models, y=acc, marker_color=SHEA_BLUE, text=[f"{v}%" for v in acc], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(name="Macro F1", x=models, y=[v*100 for v in f1], marker_color=SHEA_GOLD, text=[f"{v:.3f}" for v in f1], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(name="At-Risk Recall", x=models, y=recall, marker_color=NEG_RED, text=[f"{v}%" for v in recall], textposition="outside"), row=1, col=2)
    fig.add_trace(go.Bar(name="At-Risk Precision", x=models, y=prec, marker_color=NEUTRAL_GRAY, text=[f"{v}%" for v in prec], textposition="outside"), row=1, col=2)
    fig.update_yaxes(range=[0,105]); fig.update_layout(barmode="group"); clean_fig(fig, 420); st.plotly_chart(fig, use_container_width=True)

    best_name = b["best_name"]
    best_idx = models.index(best_name)
    static_output(f" RESULTS: {best_name} (Best Model)\n============================================================\n{b['best_report']}")
    commentary(f"{best_name} is the most accurate with {acc[best_idx]}% accuracy and a Macro F1 of {f1[best_idx]:.3f}. It catches {recall[best_idx]}% of at-risk customers from text alone, with {prec[best_idx]}% precision, meaning when it flags someone as at-risk, it is right roughly {prec[best_idx]}% of the time. The tradeoff between recall and precision depends on the business use case: if missing an at-risk customer is costly, higher recall may be preferable even at the expense of more false positives.")

    # 5.2 three class
    section_header("5.2 Three-Class Prediction", "Negative (1-2 star) vs Neutral (3 star) vs Positive (4-5 star)")
    explain("This task is more challenging than the previous prediction. Instead of simply identifying satisfied versus at-risk customers, the model must now classify reviews into three groups: negative (1–2 stars), neutral (3 stars), and positive (4–5 stars). <br> <br> This is significantly harder because 3-star reviews are inherently ambiguous. Customers in this group are often neither strongly satisfied nor strongly dissatisfied, and their language tends to reflect a mix of positive and negative comments.<br> <br> The left chart again shows overall performance using accuracy (blue) and Macro F1 (yellow). These metrics summarize how well the model performs across all three categories. <br> <br> The right chart shows recall for each rating group. These values indicate how often the model correctly identifies negative, neutral, and positive reviews individually. This chart helps reveal whether a model is truly recognizing each category or simply predicting the most common outcome.")

    models3 = ["Logistic Regression","Random Forest","Gradient Boosting"]
    acc3 = t["acc"]; f1_3 = t["f1"]
    neg_r = t["neg_r"]; neu_r = t["neu_r"]; pos_r = t["pos_r"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Accuracy & Macro F1","Recall by Class"])
    fig.add_trace(go.Bar(name="Accuracy %", x=models3, y=acc3, marker_color=SHEA_BLUE, text=[f"{v}%" for v in acc3], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(name="Macro F1", x=models3, y=[v*100 for v in f1_3], marker_color=SHEA_GOLD, text=[f"{v:.3f}" for v in f1_3], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(name="Neg Recall", x=models3, y=neg_r, marker_color=NEG_RED, text=[f"{v}%" for v in neg_r], textposition="outside"), row=1, col=2)
    fig.add_trace(go.Bar(name="Neu Recall", x=models3, y=neu_r, marker_color=NEU_YELLOW, text=[f"{v}%" for v in neu_r], textposition="outside"), row=1, col=2)
    fig.add_trace(go.Bar(name="Pos Recall", x=models3, y=pos_r, marker_color=POS_GREEN, text=[f"{v}%" for v in pos_r], textposition="outside"), row=1, col=2)
    fig.update_yaxes(range=[0,115]); fig.update_layout(barmode="group"); clean_fig(fig, 420); st.plotly_chart(fig, use_container_width=True)
    commentary("These results show why overall accuracy alone is not sufficient for evaluating model performance. Looking at recall by class reveals important differences between the models. Random Forest tends to struggle with minority classes, while Logistic Regression often provides the most balanced performance across all three categories, making it more useful for detecting dissatisfied customers. The key takeaway is that while predicting customer sentiment from text is feasible, neutral reviews remain the most difficult category to classify, since the language in 3-star reviews often contains a genuine mix of positive and negative signals.")

    # 5.3 most predictive words
    section_header("5.3 Most Predictive Words by Category", "What language signals each rating level?")
    explain("When the prediction model evaluates a review, certain words influence its decision more than others. This section highlights the words that had the strongest impact on predicting each rating category. In simple terms, this analysis asks: which words most strongly signal that a review is negative, neutral, or positive? Longer bars indicate that a word had a stronger influence on the model's prediction.")

    top_words = mr["top_words"]
    word_configs = [
        ("Negative (1-2 Stars)", "Negative (1-2 star)", NEG_RED, "Negative Signals"),
        ("Neutral (3 Stars)", "Neutral (3 star)", NEU_YELLOW, "Neutral Signals"),
        ("Positive (4-5 Stars)", "Positive (4-5 star)", POS_GREEN, "Positive Signals"),
    ]
    col1,col2,col3 = st.columns(3)
    for col, (class_key, label, color, title) in zip([col1, col2, col3], word_configs):
        with col:
            st.markdown(f"**{label}**")
            words = top_words.get(class_key, [])
            if words:
                ww, wc = zip(*words)
                max_coef = max(wc) * 1.15
                fig = go.Figure(go.Bar(y=list(ww)[::-1], x=list(wc)[::-1], orientation="h", marker_color=color, text=[f"+{v:.3f}" for v in list(wc)[::-1]], textposition="inside"))
                fig.update_xaxes(range=[0, max_coef]); fig.update_layout(title=title); clean_fig(fig, 380); st.plotly_chart(fig, use_container_width=True)
    commentary("The predictive words tell a story that aligns with everything else in this analysis. Negative signals tend to be concrete and specific, referencing repairs, construction issues, and wait times. Neutral reviews use hedging language that implies mixed feelings. Positive signals are emotional and relational, emphasizing the people and overall experience. Note that 'vader' may appear as a strong positive predictor because the VADER sentiment score was included as a numeric feature, and it strongly correlates with positive ratings.")

    nav_buttons(page)
