from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig 
except Exception:
    BitsAndBytesConfig = None  

from .data_utils import apply_chat_template_safe

def load_model(model_id: str, use_4bit: bool = False):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: Dict = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if use_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("BitsAndBytesConfig not available. Install bitsandbytes or set use_4bit=False.")
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return tok, model

@torch.no_grad()
def generate_predictions(model_id: str, out_root: Path, test_rows: List[dict],
                         gen_kw: Dict, use_4bit: bool, batch_size_gen: int) -> Path:
    out_dir  = out_root
    pred_dir = out_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(model_id, use_4bit=use_4bit)

    preds = []

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    for batch in tqdm(list(chunks(test_rows, batch_size_gen)), desc=f"Generating: {model_id.split('/')[-1]}"):
        for ex in batch:
            instr  = ex.get("instruction","")
            prompt = apply_chat_template_safe(tok, instr)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            out    = model.generate(**inputs, **gen_kw, eos_token_id=tok.eos_token_id)
            txt    = tok.decode(out[0], skip_special_tokens=True)
            # fallback if no template separator present
            resp   = txt.split("### Response:",1)[-1].strip() if "### Response:" in txt else txt.strip()
            preds.append({
                "prompt": instr,
                "gold": ex.get("response",""),
                "pred": resp,
                "target_dialect": (ex.get("dialect") or ex.get("meta",{}).get("dialect") or "").strip(),
            })

    out_path = pred_dir / f"{model_id.split('/')[-1]}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved -> {out_path}")
    return out_path

def run_many(models: List[Tuple[str, str, bool]],  
             test_rows: List[dict], base_out_dir: Path,
             gen_kw: Dict, batch_size_gen: int) -> List[Path]:
    pred_files: List[Path] = []
    for model_id, subdir, use_4bit in models:
        out_dir = base_out_dir if subdir == "" else base_out_dir.with_name(f"{base_out_dir.name}_{subdir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_files.append(generate_predictions(model_id, out_dir, test_rows, gen_kw, use_4bit, batch_size_gen))
    return pred_files
