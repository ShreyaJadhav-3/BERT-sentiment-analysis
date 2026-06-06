# config.py — central configuration for the BERT sentiment project

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────────
PRETRAINED_MODEL  = "bert-base-uncased"
NUM_LABELS        = 2
MAX_SEQ_LENGTH    = 512
SAVED_MODEL_PATH  = os.path.join(MODEL_DIR, "bert_imdb_finetuned")

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE        = 16
NUM_EPOCHS        = 3
LEARNING_RATE     = 2e-5
WARMUP_RATIO      = 0.1
WEIGHT_DECAY      = 0.01
SEED              = 42

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET_NAME      = "imdb"
TRAIN_SPLIT       = "train"
TEST_SPLIT        = "test"
VAL_SIZE          = 0.1          # fraction of train used for validation

# ── Labels ─────────────────────────────────────────────────────────────────────
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}
