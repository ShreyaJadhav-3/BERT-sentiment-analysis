# app.py — Streamlit demo for BERT Sentiment Analysis

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import torch
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from inference import SentimentPipeline


# ─────────────────────────────────────────────────────────────
# HuggingFace model repo
# CHANGE THIS to your actual HF repo name
# Example: "ShreyaJadhav-3/bert-sentiment-analysis"
# ─────────────────────────────────────────────────────────────

HF_MODEL_NAME = "Shreyu5835/bert-imdb-sentiment"
# ─────────────────────────────────────────────────────────────
# Streamlit page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 BERT Sentiment Analyzer")

st.caption(
    "Fine-tuned on IMDb Movie Reviews · "
    "bert-base-uncased · 92%+ accuracy"
)

st.divider()


# ─────────────────────────────────────────────────────────────
# Load model + tokenizer
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading fine-tuned BERT model...")
def load_pipeline():

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME,
        attn_implementation="eager"
    )

    pipe = SentimentPipeline(
        model=model,
        tokenizer=tokenizer
    )

    return pipe


try:
    pipe = load_pipeline()
    model_loaded = True

except Exception as e:

    st.error(
        f"""
Could not load HuggingFace model.

Error:
{e}

Make sure:
1. The model exists on HuggingFace
2. Repo name is correct
3. Model is public
4. Config/tokenizer/model files are uploaded
"""
    )

    model_loaded = False


# ─────────────────────────────────────────────────────────────
# Input section
# ─────────────────────────────────────────────────────────────

st.subheader("Enter a movie review")


examples = [
    "This film was an absolute masterpiece. The acting was superb and the cinematography breathtaking.",

    "Terrible movie. Completely boring and predictable from the very first scene.",

    "It had some good moments but overall felt rushed and poorly written.",
]


choice = st.selectbox(
    "Or choose an example:",
    ["— type your own —"] + examples
)

default_text = "" if choice == "— type your own —" else choice


text = st.text_area(
    "Review text:",
    value=default_text,
    height=150,
    max_chars=512,
    placeholder="Write or paste a movie review here..."
)


analyze_btn = st.button(
    "Analyze Sentiment",
    type="primary",
    disabled=not model_loaded
)


# ─────────────────────────────────────────────────────────────
# Prediction section
# ─────────────────────────────────────────────────────────────

if analyze_btn and text.strip():

    with st.spinner("Running inference..."):

        result = pipe.predict(text.strip())

    label = result["label"]

    pos_score = result["positive_score"]

    neg_score = result["negative_score"]

    top_tok = result["top_attention"]


    # ── Prediction banner ─────────────────────

    emoji = "😊" if label == "POSITIVE" else "😞"

    color = "green" if label == "POSITIVE" else "red"

    st.markdown(
        f"""
<h2 style='color:{color}; text-align:center'>
{emoji} {label}
</h2>
""",
        unsafe_allow_html=True,
    )


    # ── Metrics ───────────────────────────────

    col1, col2 = st.columns(2)

    col1.metric(
        "Positive probability",
        f"{pos_score:.2%}"
    )

    col2.metric(
        "Negative probability",
        f"{neg_score:.2%}"
    )

    st.progress(
        float(pos_score),
        text="Positive confidence"
    )


    # ── Attention chart ───────────────────────

    if top_tok:

        st.subheader("Top attended tokens")

        tokens, scores = zip(*top_tok)

        scores_arr = np.array(scores, dtype=float)

        fig, ax = plt.subplots(figsize=(8, 3))

        cmap = plt.cm.Oranges

        colors = cmap(scores_arr / scores_arr.max())

        bars = ax.barh(
            range(len(tokens)),
            scores_arr,
            color=colors
        )

        ax.set_yticks(range(len(tokens)))

        ax.set_yticklabels(tokens, fontsize=11)

        ax.invert_yaxis()

        ax.set(
            title="[CLS] attention weights",
            xlabel="Attention score"
        )

        ax.bar_label(
            bars,
            fmt="%.4f",
            padding=3,
            fontsize=9
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()


    st.caption(
        "Attention weights show which tokens "
        "BERT focused on during prediction."
    )


elif analyze_btn and not text.strip():

    st.warning(
        "Please enter some review text before clicking Analyze."
    )


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:

    st.header("About")

    st.markdown(
        """
### Model Information

- **Base model:** bert-base-uncased
- **Task:** Binary sentiment classification
- **Dataset:** IMDb Movie Reviews
- **Accuracy:** 92%+
- **Framework:** PyTorch + HuggingFace

---

### Pipeline

1. Input text is tokenized
2. BERT generates contextual embeddings
3. `[CLS]` token is classified
4. Softmax returns probabilities

---

### Features

✅ Attention visualization  
✅ Real-time inference  
✅ HuggingFace deployment  
✅ Streamlit interface  
✅ Transformer-based NLP
"""
    )

    st.divider()

    device_info = (
        "CUDA"
        if torch.cuda.is_available()
        else (
            "MPS"
            if torch.backends.mps.is_available()
            else "CPU"
        )
    )

    st.caption(f"Running on: **{device_info}**")