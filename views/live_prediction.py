import streamlit as st
import os
import numpy as np

from utils.config import POS_GREEN, NEG_RED, NEU_YELLOW
from utils.components import section_header, nav_buttons
from utils.data import load_prediction_models, predict_review


def render(df, fdf, page):
    st.title("Live Prediction Tool")
    section_header("Satisfaction Predictor", "Paste any review text to get a real-time ML prediction")

    with st.form("prediction_form"):
        user_text = st.text_area(
            "Enter review text",
            placeholder="e.g. The sales team was great but after we moved in, warranty repairs took months and nobody returned our calls.",
            height=120,
            key="predictor_input",
        )
        submitted = st.form_submit_button("Submit")

    if submitted and user_text.strip():
        _mp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "metadata.joblib")
        pred_models = load_prediction_models(_model_mtime=os.path.getmtime(_mp) if os.path.exists(_mp) else 0)
        result = predict_review(user_text.strip(), pred_models)

        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            b_label = result["binary_label"]
            b_color = POS_GREEN if "Satisfied" in b_label else NEG_RED
            b_conf = result["binary_proba"][b_label] * 100
            st.markdown(
                f"<div style='text-align:center;padding:12px 8px;background:white;border-radius:10px;border:2px solid {b_color}'>"
                f"<span style='font-size:0.85rem;color:#555'>Binary Prediction</span><br>"
                f"<span style='font-size:1.3rem;font-weight:700;color:{b_color}'>{b_label}</span><br>"
                f"<span style='font-size:0.85rem;color:#888'>{b_conf:.0f}% confidence</span></div>",
                unsafe_allow_html=True
            )

        with pc2:
            t_label = result["three_label"]
            t_color = POS_GREEN if "Positive" in t_label else (NEG_RED if "Negative" in t_label else NEU_YELLOW)
            t_conf = result["three_proba"][t_label] * 100
            st.markdown(
                f"<div style='text-align:center;padding:12px 8px;background:white;border-radius:10px;border:2px solid {t_color}'>"
                f"<span style='font-size:0.85rem;color:#555'>3-Class Prediction</span><br>"
                f"<span style='font-size:1.3rem;font-weight:700;color:{t_color}'>{t_label}</span><br>"
                f"<span style='font-size:0.85rem;color:#888'>{t_conf:.0f}% confidence</span></div>",
                unsafe_allow_html=True
            )

        with pc3:
            v_label = result["vader_label"]
            v_color = POS_GREEN if v_label == "Positive" else (NEG_RED if v_label == "Negative" else NEU_YELLOW)
            v_score = result["vader_compound"]
            st.markdown(
                f"<div style='text-align:center;padding:12px 8px;background:white;border-radius:10px;border:2px solid {v_color}'>"
                f"<span style='font-size:0.85rem;color:#555'>VADER Sentiment</span><br>"
                f"<span style='font-size:1.3rem;font-weight:700;color:{v_color}'>{v_label} ({v_score:+.2f})</span><br>"
                f"<span style='font-size:0.85rem;color:#888'>compound score</span></div>",
                unsafe_allow_html=True
            )

        st.markdown("")
        with st.expander("Detailed confidence breakdown", expanded=submitted):
            det1, det2 = st.columns(2)

            with det1:
                st.markdown("**Binary model**")
                for cls in sorted(result["binary_proba"].keys()):
                    pct = result["binary_proba"][cls] * 100
                    c = POS_GREEN if "Satisfied" in cls else NEG_RED
                    st.markdown(
                        f"<span style='color:{c};font-weight:600'>{cls}</span>: {pct:.1f}%",
                        unsafe_allow_html=True
                    )
                    st.progress(result["binary_proba"][cls])

            with det2:
                st.markdown("**3-class model**")
                order = ["Negative (1-2 Stars)", "Neutral (3 Stars)", "Positive (4-5 Stars)"]
                for cls in order:
                    pct = result["three_proba"].get(cls, 0) * 100
                    c = POS_GREEN if "Positive" in cls else (NEG_RED if "Negative" in cls else NEU_YELLOW)
                    st.markdown(
                        f"<span style='color:{c};font-weight:600'>{cls}</span>: {pct:.1f}%",
                        unsafe_allow_html=True
                    )
                    st.progress(result["three_proba"].get(cls, 0))

            if result["signal_words"]:
                st.markdown("**Top signal words driving this prediction:**")
                sw_text = " &nbsp;&nbsp; ".join([f"`{w}` (+{s:.2f})" for w, s in result["signal_words"]])
                st.markdown(sw_text, unsafe_allow_html=True)

    elif submitted and not user_text.strip():
        st.warning("Please enter some review text before submitting.")

    nav_buttons(page)
