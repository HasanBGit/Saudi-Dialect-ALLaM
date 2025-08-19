from __future__ import annotations
import re, json
from pathlib import Path
from typing import List, Tuple
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .config import BASE_MODEL_ID, BATCH_SIZE_GEN

TAG_RE = re.compile(r'^\s*<\s*DIALECT\s*=\s*(HIJAZI|NAJDI)\s*>\s*', re.IGNORECASE)

def build_prompt(instr: str) -> str:
    return f"### Instruction:\n{instr}\n\n### Response:\n"

@torch.no_grad()
def generate_for_model(rows_to_use: List[dict], name: str, adapter_dir: str | None, strip_tag: bool,
                       out_dir: Path, pred_dir: Path) -> Path:
    print(f"\n==> Generating with: {name}")
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"{name}.jsonl"

    tok = AutoTokenizer.from_pretrained(adapter_dir or BASE_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = base
    if adapter_dir:
        try:
            model = PeftModel.from_pretrained(base, adapter_dir)
        except Exception as e:
            print(f"WARNING: failed to load adapter {adapter_dir}; using base. {e}")

    model.eval()
    preds = []
    gen_kwargs = dict(
        max_new_tokens=256, do_sample=True, top_p=0.95, top_k=50, temperature=0.6,
        eos_token_id=tok.eos_token_id,
    )

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    for batch in tqdm(list(chunks(rows_to_use, BATCH_SIZE_GEN)), desc=f"Generating ({name})", leave=False):
        for ex in batch:
            instr = ex.get("instruction", "")
            if strip_tag:
                instr = TAG_RE.sub("", instr).lstrip()
            inputs = tok(build_prompt(instr), return_tensors="pt").to(model.device)
            out = model.generate(**inputs, **gen_kwargs)
            text = tok.decode(out[0], skip_special_tokens=True)
            resp = text.split("### Response:", 1)[-1].strip()
            preds.append({
                "prompt": instr,
                "gold": ex.get("response", ""),
                "pred": resp,
                "target_dialect": (ex.get("dialect") or ex.get("meta", {}).get("dialect") or "").strip(),
            })

    with out_path.open("w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved -> {out_path}")
    return out_path

def run_generation(rows_to_use: List[dict],
                   model_specs: List[Tuple[str, str|None, bool]],
                   out_dir: Path, pred_dir: Path) -> List[Path]:
    pred_files: List[Path] = []
    for name, adapter, strip_tag in tqdm(model_specs, desc="Models", position=0):
        pred_files.append(generate_for_model(rows_to_use, name, adapter, strip_tag, out_dir, pred_dir))
    return pred_files
