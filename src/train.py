# train.py — fine-tuning loop with AdamW, warmup scheduler, and checkpointing

import os
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from config import (
    NUM_EPOCHS, LEARNING_RATE, WARMUP_RATIO,
    WEIGHT_DECAY, SAVED_MODEL_PATH, OUTPUT_DIR, PRETRAINED_MODEL
)
from data_loader import load_and_tokenize
from model import build_model


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("Apple MPS backend")
    else:
        dev = torch.device("cpu")
        print("Running on CPU (slow — consider Google Colab for GPU)")
    return dev


def run_epoch(model, loader, optimizer, scheduler, device, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            out  = model(input_ids, attention_mask, token_type_ids, labels=labels)
            loss = out["loss"]

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            preds       = out["logits"].argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            if training and step % 100 == 0:
                print(f"  step {step}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader), correct / total


def plot_history(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val loss")
    ax1.set(title="Loss per epoch", xlabel="Epoch", ylabel="Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train acc")
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val acc")
    ax2.set(title="Accuracy per epoch", xlabel="Epoch", ylabel="Accuracy")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    device = get_device()

    # Data
    train_loader, val_loader, _, tokenizer = load_and_tokenize()

    # Model
    model = build_model(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — total: {total_params:,}  trainable: {trainable_params:,}")

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\nTraining for {NUM_EPOCHS} epoch(s) …\n{'='*55}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, scheduler, device, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   optimizer, scheduler, device, training=False)

        elapsed = time.time() - t0
        print(f"  train loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"  val   loss={vl_loss:.4f}  acc={vl_acc:.4f}  ({elapsed:.0f}s)")

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        # Save best checkpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            model.save_pretrained(SAVED_MODEL_PATH)
            tokenizer.save_pretrained(SAVED_MODEL_PATH)
            print(f"  ✓ Best model saved (val_acc={vl_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    plot_history(history)
    return history


if __name__ == "__main__":
    train()
