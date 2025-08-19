from __future__ import annotations
from pathlib import Path

MAX_EVAL       = 0        # 0 = full test; otherwise limit (e.g., 50, 100)
RANDOM_SAMPLE  = True
SEED           = 42
BATCH_SIZE_GEN = 1        
BATCH_SIZE_CLF = 64       # MARBERTv2 classifier batch size

# -------- paths --------
TEST_PATH = Path("data_splits/test.jsonl")
OUT_DIR   = Path("eval_external")
PRED_DIR  = OUT_DIR / "preds"

GEN_KW = dict(max_new_tokens=256, do_sample=True, top_p=0.95, top_k=50, temperature=0.6)

EXTERNAL_MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct",  "llama31_8b",    False),
    ("Qwen/Qwen2.5-7B-Instruct",          "qwen2p5_7b",    False),
    ("inceptionai/jais-13b-chat",         "jais_13b",      True),   # int4 
    ("tiiuae/Falcon3-7B-Instruct",        "falcon3_7b",    False),
    ("FreedomIntelligence/AceGPT-v2-8B-Chat", "acegpt_v2_8b", False),
]
