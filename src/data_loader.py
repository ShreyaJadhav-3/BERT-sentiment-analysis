# data_loader.py — IMDb dataset loading, tokenization, and DataLoader creation

import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from config import (
    PRETRAINED_MODEL, MAX_SEQ_LENGTH, DATASET_NAME,
    TRAIN_SPLIT, TEST_SPLIT, VAL_SIZE, BATCH_SIZE, SEED
)


class IMDbDataset(Dataset):
    """PyTorch Dataset wrapping the HuggingFace IMDb split."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_and_tokenize(verbose: bool = True):
    """
    Downloads the IMDb dataset, tokenizes with BertTokenizer, and returns
    train / validation / test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    tokenizer : BertTokenizer  (kept for inference)
    """
    if verbose:
        print(f"[1/4] Loading '{DATASET_NAME}' dataset …")

    raw = load_dataset(DATASET_NAME)
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

    if verbose:
        print(f"[2/4] Tokenizing (max_length={MAX_SEQ_LENGTH}) …")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    tokenized = raw.map(tokenize, batched=True, batch_size=512)

    # ── Build PyTorch datasets ────────────────────────────────────────────────
    def make_ds(split):
        enc = {
            "input_ids":      tokenized[split]["input_ids"],
            "attention_mask": tokenized[split]["attention_mask"],
            "token_type_ids": tokenized[split]["token_type_ids"],
        }
        return IMDbDataset(enc, tokenized[split]["label"])

    full_train = make_ds(TRAIN_SPLIT)
    test_ds    = make_ds(TEST_SPLIT)

    # ── Train / val split ────────────────────────────────────────────────────
    val_size   = int(len(full_train) * VAL_SIZE)
    train_size = len(full_train) - val_size
    generator  = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    if verbose:
        print(f"[3/4] Split sizes — train: {train_size}, val: {val_size}, test: {len(test_ds)}")

    def make_loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=2, pin_memory=True)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds)
    test_loader  = make_loader(test_ds)

    if verbose:
        print("[4/4] DataLoaders ready.")

    return train_loader, val_loader, test_loader, tokenizer
