"""
Run base and LoRA-adapted ALLaM-7B-Instruct models with/without dialect tags.

Usage (defaults provided):
    python run_allam_lora.py \
        --instruction "تكلم عن اول يوم دراسي لك في المدرسة" \
        --dialect najdi \
        --adapter-token outputs/allam7b-lora-token-15EPOCH/checkpoint-97 \
        --adapter-notoken outputs/allam7b-lora-no-token-15EPOCH/checkpoint-86

"""

import re
import gc
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL      = "ALLaM-AI/ALLaM-7B-Instruct-preview"
ADAPTER_TOKEN   = "outputs/allam7b-lora-token-15EPOCH/checkpoint-97"
ADAPTER_NOTOKEN = "outputs/allam7b-lora-no-token-15EPOCH/checkpoint-86"

DIALECT_DEFAULT     = "najdi"
INSTRUCTION_DEFAULT = "تكلم عن اول يوم دراسي لك في المدرسة"

GEN_KW_DEFAULT = dict(
    max_new_tokens=160,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.05,
)

# -------------------- Tag helpers --------------------
TAG_RE = re.compile(r'^\s*<\s*DIALECT\s*=\s*(HIJAZI|NAJDI)\s*>\s*', re.IGNORECASE)

def add_tag(instruction, dialect=None):
    """Prepend <DIALECT=...> tag if dialect is known and no tag already exists."""
    if not dialect:
        return instruction
    d = str(dialect).strip().lower()
    if d.startswith("hij"):
        tag = "<DIALECT=HIJAZI>"
    elif d.startswith("naj"):
        tag = "<DIALECT=NAJDI>"
    else:
        return instruction
    if TAG_RE.match(instruction or ""):
        return instruction
    return f"{tag}{(instruction or '').lstrip()}"

def strip_tag(instruction):
    """Remove a leading <DIALECT=...> tag if present."""
    return TAG_RE.sub("", instruction or "").lstrip()

# -------------------- Model/Tokenizer loaders --------------------
def _load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_base():
    tok = _load_tokenizer()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map="auto"
    )
    mdl.config.use_cache = True
    mdl.eval()
    return tok, mdl

def load_with_lora(adapter_dir):
    tok = _load_tokenizer()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map="auto"
    )
    base.config.use_cache = True
    mdl = PeftModel.from_pretrained(base, adapter_dir)
    mdl.eval()
    return tok, mdl

def prove_lora(model):
    """Return a small proof that LoRA adapters are attached."""
    is_peft = isinstance(model, PeftModel)
    try:
        adapters = list(model.peft_config.keys())
    except Exception:
        adapters = []
    lora_modules = [n for n, _ in model.named_modules() if "lora_" in n.lower()]
    sample = lora_modules[:8]
    return {
        "is_peft_model": is_peft,
        "active_adapters": adapters,
        "sample_lora_modules": sample,
        "n_lora_modules": len(lora_modules),
    }

def build_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

@torch.no_grad()
def generate(model, tokenizer, instruction, *,
             max_new_tokens=160, temperature=0.7, top_p=0.95, top_k=50, repetition_penalty=1.05):
    prompt = build_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    return txt.split("### Response:\n", 1)[-1].strip()

def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------- Orchestrator --------------------
def run_all(instruction, dialect=None, gen_kw=None,
            adapter_token=ADAPTER_TOKEN, adapter_notoken=ADAPTER_NOTOKEN):
    """
    Runs:
      1) Base model (optionally with a dialect tag).
      2) LoRA adapter trained with token (forces tag).
      3) LoRA adapter trained without token (forces no tag).
    """
    gen_kw = gen_kw or {}
    results = {}

    # 1) Load tokenizer and BASE model once
    tok = _load_tokenizer()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map="auto"
    )
    base.config.use_cache = True
    base.eval()

    # 2) BASE output (can add a tag to gently nudge style if dialect is set)
    instr_base = add_tag(instruction, dialect) if dialect else instruction
    results["base_instruction_eff"] = instr_base
    results["base_output"] = generate(base, tok, instr_base, **gen_kw)

    # 3) TOKEN LoRA (ensure tag is present)
    peft_m = PeftModel.from_pretrained(base, adapter_token, adapter_name="token")
    peft_m.eval()
    instr_token = add_tag(instruction, dialect or "najdi")
    results["token_instruction_eff"] = instr_token
    results["token_proof"] = prove_lora(peft_m)
    results["token_output"] = generate(peft_m, tok, instr_token, **gen_kw)

    # 4) NO-TOKEN LoRA (ensure NO tag is in the prompt)
    peft_m.load_adapter(adapter_notoken, adapter_name="no_token")
    peft_m.set_adapter("no_token")
    instr_notoken = strip_tag(instruction)
    results["no_token_instruction_eff"] = instr_notoken
    results["no_token_proof"] = prove_lora(peft_m)
    results["no_token_output"] = generate(peft_m, tok, instr_notoken, **gen_kw)

    return results

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run ALLaM-7B base and LoRA adapters with/without dialect tags.")
    p.add_argument("--instruction", type=str, default=INSTRUCTION_DEFAULT,
                   help="Instruction/prompt text.")
    p.add_argument("--dialect", type=str, default=DIALECT_DEFAULT,
                   help="Dialect hint (e.g., 'najdi', 'hijazi').")
    p.add_argument("--adapter-token", type=str, default=ADAPTER_TOKEN,
                   help="Path to LoRA adapter trained WITH token.")
    p.add_argument("--adapter-notoken", type=str, default=ADAPTER_NOTOKEN,
                   help="Path to LoRA adapter trained WITHOUT token.")
    # generation args
    p.add_argument("--max-new-tokens", type=int, default=GEN_KW_DEFAULT["max_new_tokens"])
    p.add_argument("--temperature", type=float, default=GEN_KW_DEFAULT["temperature"])
    p.add_argument("--top-p", type=float, default=GEN_KW_DEFAULT["top_p"])
    p.add_argument("--top-k", type=int, default=GEN_KW_DEFAULT["top_k"])
    p.add_argument("--repetition-penalty", type=float, default=GEN_KW_DEFAULT["repetition_penalty"])
    return p.parse_args()

def main():
    args = parse_args()
    gen_kw = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    clear_mem()
    res = run_all(
        instruction=args.instruction,
        dialect=args.dialect,
        gen_kw=gen_kw,
        adapter_token=args.adapter_token,
        adapter_notoken=args.adapter_notoken,
    )

    print("### Effective instructions")
    print("- BASE     :", res['base_instruction_eff'])
    print("- TOKEN    :", res['token_instruction_eff'])
    print("- NO-TOKEN :", res['no_token_instruction_eff'])
    print()

    print("### Outputs")
    print("[BASE]\n", res['base_output'], "\n", sep="")
    print("[TOKEN]\n", res['token_output'], "\n", sep="")
    print("[NO-TOKEN]\n", res['no_token_output'], sep="")

    print("\n### LoRA attach proofs")
    print("TOKEN   :", res['token_proof'])
    print("NO-TOKEN:", res['no_token_proof'])

if __name__ == "__main__":
    main()
