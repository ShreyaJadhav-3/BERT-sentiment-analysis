# evaluate.py — test-set evaluation, confusion matrix, and attention heatmap

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from transformers import BertTokenizer

from config import SAVED_MODEL_PATH, OUTPUT_DIR, ID2LABEL, PRETRAINED_MODEL
from data_loader import load_and_tokenize
from model import load_saved_model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, device):
    """Run inference on the test set; return all_preds, all_labels, all_probs."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            out    = model(input_ids, attention_mask, token_type_ids)
            logits = out["logits"]
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())   # P(POSITIVE)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_metrics(preds, labels, probs):
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    print(f"\n{'='*55}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"{'='*55}")
    print("\nClassification Report:")
    print(classification_report(labels, preds,
                                 target_names=[ID2LABEL[0], ID2LABEL[1]]))


def plot_confusion_matrix(preds, labels):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[ID2LABEL[0], ID2LABEL[1]],
        yticklabels=[ID2LABEL[0], ID2LABEL[1]],
        ax=ax, linewidths=0.5
    )
    ax.set(title="Confusion Matrix — IMDb Test Set",
           xlabel="Predicted", ylabel="Actual")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {path}")


# ── Attention visualization ───────────────────────────────────────────────────

def visualize_attention(text: str, tokenizer: BertTokenizer, model, device, layer: int = 11):
    """
    Extract [CLS] attention weights from a single sentence,
    average across all heads in `layer`, and plot a bar chart.
    """
    enc = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128, padding=True
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids"),
            output_attentions=True,
        )

    # attentions: tuple of (1, heads, seq, seq) per layer
    attn = out["attentions"][layer]           # (1, 12, seq, seq)
    cls_attn = attn[0, :, 0, :].mean(0)      # mean over heads → (seq,)
    cls_attn = cls_attn.cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().numpy())
    # trim padding
    pad_id = tokenizer.pad_token_id
    ids    = enc["input_ids"][0].cpu().numpy()
    valid  = [i for i, t in enumerate(ids) if t != pad_id]
    tokens = [tokens[i] for i in valid]
    scores = cls_attn[valid]

    fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.5), 4))
    colors  = plt.cm.Oranges(scores / scores.max())
    ax.bar(range(len(tokens)), scores, color=colors)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set(title=f"[CLS] Attention weights — layer {layer+1}",
           ylabel="Average attention weight")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "attention_weights.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Attention heatmap saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading fine-tuned model …")
    model     = load_saved_model(SAVED_MODEL_PATH, device)
    tokenizer = BertTokenizer.from_pretrained(SAVED_MODEL_PATH)

    _, _, test_loader, _ = load_and_tokenize(verbose=False)

    print("Running inference on test set …")
    preds, labels, probs = evaluate_model(model, test_loader, device)

    print_metrics(preds, labels, probs)
    plot_confusion_matrix(preds, labels)

    # Attention on a sample review
    sample = "This movie was an absolute masterpiece. The acting was superb!"
    visualize_attention(sample, tokenizer, model, device)

    print("\nAll evaluation outputs saved to /outputs/")


if __name__ == "__main__":
    run_evaluation()
