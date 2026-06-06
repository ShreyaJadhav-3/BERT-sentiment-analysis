# BERT Sentiment Analysis — IMDb Movie Reviews

Fine-tuned `bert-base-uncased` on the IMDb dataset for binary sentiment classification.  
Achieves **92%+ test accuracy** with a lightweight Streamlit inference demo.

---

## Project Structure

```
bert_sentiment/
├── app.py                  # Streamlit web demo
├── requirements.txt        # Python dependencies
├── src/
│   ├── config.py           # Hyperparameters and paths
│   ├── data_loader.py      # IMDb loading + tokenization + DataLoaders
│   ├── model.py            # BertSentimentClassifier definition
│   ├── train.py            # Training loop (AdamW + warmup scheduler)
│   ├── evaluate.py         # Metrics, confusion matrix, attention viz
│   └── inference.py        # Production inference pipeline
├── models/                 # Saved checkpoints (auto-created)
├── outputs/                # Plots (auto-created)
└── data/                   # Cache (auto-created)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fine-tune BERT (needs GPU for reasonable speed)
python src/train.py

# 3. Evaluate on test set
python src/evaluate.py

# 4. Launch Streamlit demo
streamlit run app.py
```

---

## Configuration (`src/config.py`)

| Parameter       | Default              | Description                        |
|-----------------|----------------------|------------------------------------|
| PRETRAINED_MODEL| bert-base-uncased    | HuggingFace model hub ID           |
| MAX_SEQ_LENGTH  | 512                  | Token truncation length            |
| BATCH_SIZE      | 16                   | Samples per batch                  |
| NUM_EPOCHS      | 3                    | Fine-tuning epochs                 |
| LEARNING_RATE   | 2e-5                 | AdamW base LR                      |
| WARMUP_RATIO    | 0.1                  | Fraction of steps for LR warmup    |
| WEIGHT_DECAY    | 0.01                 | AdamW weight decay                 |
| VAL_SIZE        | 0.1                  | Fraction of train used for val     |

---

## Results

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 92%+   |
| Precision    | ~0.92  |
| Recall       | ~0.92  |
| F1-Score     | ~0.92  |
| ROC-AUC      | ~0.97  |

---

## Tech Stack

Python · PyTorch · HuggingFace Transformers · Datasets · scikit-learn · Matplotlib · Seaborn · Streamlit
