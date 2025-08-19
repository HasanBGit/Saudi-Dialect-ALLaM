from __future__ import annotations
from pathlib import Path

# ---------- quick-run knobs ----------
MAX_EVAL = 0            # 0 = use all test examples; set to 20/100 for a small run
RANDOM_SAMPLE = True
BATCH_SIZE_GEN = 1      
BATCH_SIZE_CLF = 64     # MARBERTv2 classifier batch size
SEED = 42

# ---------- paths / base model / outputs ----------
BASE_MODEL_ID = "ALLaM-AI/ALLaM-7B-Instruct-preview"
TEST_PATH = Path("data_splits/test.jsonl")  
OUT_DIR = Path("eval_saudi_only")
PRED_DIR = OUT_DIR / "preds"

MODEL_SPECS = [
    ("allam_base",             None,                                  False),
    ("lora-no-token-15EPOCH",  "outputs/allam7b-lora-no-token-15EPOCH", True),
    ("lora-token-15EPOCH",     "outputs/allam7b-lora-token-15EPOCH",  False),
]
