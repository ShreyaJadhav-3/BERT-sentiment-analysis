# inference.py — production-style inference pipeline for single texts or batches

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from config import SAVED_MODEL_PATH, MAX_SEQ_LENGTH, ID2LABEL
from model import load_saved_model


class SentimentPipeline:
    """
    Lightweight inference wrapper. Loads the fine-tuned model once and
    exposes a `predict` method for single texts or lists.

    Usage
    -----
    >>> pipe = SentimentPipeline()
    >>> pipe.predict("This film was absolutely incredible!")
    {'label': 'POSITIVE', 'score': 0.9987, 'negative_score': 0.0013}
    """
    
    def __init__(self, model=None, tokenizer=None):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Load from HuggingFace if model/tokenizer provided
        if model is not None and tokenizer is not None:

            self.model = model.to(self.device)

            self.tokenizer = tokenizer

        else:
            # fallback local loading
            self.model = load_saved_model().to(self.device)

            self.tokenizer = BertTokenizer.from_pretrained(
                SAVED_MODEL_PATH
            )

        self.model.eval()

    @torch.no_grad()
    def predict(self, texts, top_tokens: int = 5):
        """
        Parameters
        ----------
        texts : str | list[str]
        top_tokens : int   number of high-attention tokens to return

        Returns
        -------
        dict | list[dict]
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out   = self.model(**enc, output_attentions=True)
        probs = F.softmax(out["logits"], dim=-1).cpu()

        results = []
        for i, text in enumerate(texts):
            neg_score = probs[i, 0].item()
            pos_score = probs[i, 1].item()
            label     = ID2LABEL[int(probs[i].argmax())]

            # Top-attention tokens from last layer, averaged over heads
            if out["attentions"] is not None:
                attn      = out["attentions"][-1][i]  # (heads, seq, seq)
                cls_attn  = attn[:, 0, :].mean(0).cpu().numpy()
                token_ids = enc["input_ids"][i].cpu().numpy()
                tokens    = self.tokenizer.convert_ids_to_tokens(token_ids)
                # exclude [CLS] [SEP] [PAD]
                skip = {self.tokenizer.cls_token, self.tokenizer.sep_token,
                        self.tokenizer.pad_token}
                scored = [
                    (tok, cls_attn[j])
                    for j, tok in enumerate(tokens)
                    if tok not in skip
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                top = [(t, round(float(s), 4)) for t, s in scored[:top_tokens]]
            else:
                top = []

            results.append({
                "text":             text[:120] + ("…" if len(text) > 120 else ""),
                "label":            label,
                "score":            round(max(neg_score, pos_score), 4),
                "positive_score":   round(pos_score, 4),
                "negative_score":   round(neg_score, 4),
                "top_attention":    top,
            })

        return results[0] if single else results


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "This movie was an absolute masterpiece. Brilliant performances!",
        "Terrible film. Completely boring from start to finish.",
        "It was okay, not great but not terrible either.",
    ]

    pipe = SentimentPipeline()
    for r in pipe.predict(samples):
        print(f"\n[{r['label']} — {r['score']:.2%}] {r['text']}")
        print(f"  Top tokens: {r['top_attention']}")
