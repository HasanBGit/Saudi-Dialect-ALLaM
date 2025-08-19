from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
BASE_MODEL = "ALLaM-AI/ALLaM-7B-Instruct-preview"
ADAPTER_DIR = "outputs/allam7b-lora-token"

def load_base():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    mdl.config.use_cache = True  # Changed from False to True for inference
    return tok, mdl

def load_with_adapter(adapter_dir):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.config.use_cache = True  # Changed from False to True for inference
    mdl  = PeftModel.from_pretrained(base, adapter_dir)
    return tok, mdl

def generate(model, tokenizer, instruction, max_new_tokens=160, temperature=0.7, top_p=0.95):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=top_p)
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    return txt.split("### Response:\n",1)[-1].strip()