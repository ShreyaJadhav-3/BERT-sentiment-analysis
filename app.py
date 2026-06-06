# app.py — Streamlit demo for BERT Sentiment Analysis

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

from inference import SentimentPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 BERT Sentiment Analyzer")
st.caption("Fine-tuned on the IMDb Movie Review Dataset · bert-base-uncased · 92%+ accuracy")
st.divider()

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading fine-tuned BERT model …")
def load_pipeline():
    return SentimentPipeline()

try:
    pipe = load_pipeline()
    model_loaded = True
except Exception as e:
    st.error(f"Could not load model: {e}\n\nPlease run `python src/train.py` first.")
    model_loaded = False

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("Enter a movie review")

examples = [
    "This film was an absolute masterpiece. The acting was superb, the cinematography breathtaking.",
    "Terrible movie. Completely boring and predictable from the very first scene.",
    "It had some good moments but overall felt rushed and poorly written.",
]
choice = st.selectbox("Or pick an example:", ["— type your own —"] + examples)
default_text = "" if choice == "— type your own —" else choice

text = st.text_area("Review text:", value=default_text, height=130, max_chars=512,
                     placeholder="Write or paste a movie review here …")

analyze_btn = st.button("Analyze", type="primary", disabled=not model_loaded)

# ── Prediction ────────────────────────────────────────────────────────────────
if analyze_btn and text.strip():
    with st.spinner("Running inference …"):
        result = pipe.predict(text.strip())

    label     = result["label"]
    pos_score = result["positive_score"]
    neg_score = result["negative_score"]
    top_tok   = result["top_attention"]

    # Verdict banner
    emoji = "😊" if label == "POSITIVE" else "😞"
    color = "green" if label == "POSITIVE" else "red"
    st.markdown(
        f"<h2 style='color:{color};text-align:center'>{emoji} {label}</h2>",
        unsafe_allow_html=True,
    )

    # Score bars
    col1, col2 = st.columns(2)
    col1.metric("Positive probability", f"{pos_score:.2%}")
    col2.metric("Negative probability", f"{neg_score:.2%}")

    st.progress(pos_score, text="Positive confidence")

    # Attention bar chart
    if top_tok:
        st.subheader("Top attended tokens")
        tokens, scores = zip(*top_tok)
        scores_arr = np.array(scores, dtype=float)

        fig, ax = plt.subplots(figsize=(8, 3))
        cmap    = plt.cm.Oranges
        colors  = cmap(scores_arr / scores_arr.max())
        bars    = ax.barh(range(len(tokens)), scores_arr, color=colors)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=11)
        ax.invert_yaxis()
        ax.set(title="[CLS] attention weight (last layer, avg heads)",
               xlabel="Attention score")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.caption("Attention weights show which tokens BERT focused on for this prediction.")

elif analyze_btn and not text.strip():
    st.warning("Please enter some text before clicking Analyze.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
**Model:** `bert-base-uncased`  
**Task:** Binary sentiment classification  
**Dataset:** IMDb (25k train / 25k test)  
**Accuracy:** 92%+ on test set  
**Framework:** PyTorch + HuggingFace Transformers

---
**How it works:**
1. Text is tokenized (WordPiece)
2. BERT encoder produces contextual embeddings
3. The `[CLS]` token is passed through a dropout + linear head
4. Softmax gives POSITIVE / NEGATIVE probabilities

---
Run `python src/train.py` to fine-tune,  
then `streamlit run app.py` to launch this demo.
    """)
    st.divider()
    device_info = "CUDA" if torch.cuda.is_available() else ("MPS" if torch.backends.mps.is_available() else "CPU")
    st.caption(f"Running on: **{device_info}**")
