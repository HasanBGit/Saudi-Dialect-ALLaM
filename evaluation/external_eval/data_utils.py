from __future__ import annotations
import json, random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def subset_rows(rows: list, max_eval: int, seed: int, random_sample: bool):
    if max_eval and 0 < max_eval < len(rows):
        if random_sample:
            rng = random.Random(seed)
            idxs = sorted(rng.sample(range(len(rows)), max_eval))
            return [rows[i] for i in idxs]
        return rows[:max_eval]
    return rows

def apply_chat_template_safe(tok, user_text: str) -> str:
    """
    Use tokenizer chat template if present; otherwise fall back to your ICL format.
    """
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        msgs=[{"role":"user","content":user_text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"### Instruction:\n{user_text}\n\n### Response:\n"
