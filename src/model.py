# model.py — BERT fine-tuning model for binary sentiment classification

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from config import PRETRAINED_MODEL, NUM_LABELS, ID2LABEL, LABEL2ID


class BertSentimentClassifier(BertPreTrainedModel):
    """
    bert-base-uncased + linear classification head.

    Architecture
    ─────────────
    BERT encoder → [CLS] pooled output (768-d)
       → Dropout(0.3)
       → Linear(768 → 2)
       → logits
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert       = BertModel(config, add_pooling_layer=True)
        self.dropout    = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()  # initializes weights and applies final processing

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=False,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
        )

        pooled = self.dropout(outputs.pooler_output)   # (batch, 768)
        logits = self.classifier(pooled)               # (batch, 2)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {
            "loss":       loss,
            "logits":     logits,
            "attentions": outputs.attentions if output_attentions else None,
        }


def build_model(device: torch.device) -> BertSentimentClassifier:
    """Load pretrained BERT weights and attach the classification head."""
    model = BertSentimentClassifier.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model.to(device)


def load_saved_model(path: str, device: torch.device) -> BertSentimentClassifier:
    """Load a fine-tuned model from disk."""
    model = BertSentimentClassifier.from_pretrained(path)
    return model.to(device).eval()
