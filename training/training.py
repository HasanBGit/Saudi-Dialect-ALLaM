from __future__ import annotations
import os, sys, time, math
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, set_seed
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def setup_env(seed: int) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def env_report() -> None:
    print("="*60)
    print("ENVIRONMENT CHECK")
    print(f"Python executable: {sys.executable}")
    print(f"Python version   : {sys.version.split()[0]}")
    print(f"Torch version    : {torch.__version__}")
    print(f"Transformers ver : {transformers.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA device detected!")
    try:
        import datasets; print(f"Datasets version : {datasets.__version__}")
    except Exception: print("Datasets not installed?")
    try:
        import peft; print(f"PEFT version     : {peft.__version__}")
    except Exception: print("PEFT not installed?")
    print("="*60)

def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def build_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False  
    return model

class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.t0 = time.time()
        self.tlast = self.t0
    def on_step_end(self, args, state, control, logs=None, **kw):  
        if not state.is_local_process_zero: return
        logs = logs or {}
        now = time.time()
        tl = logs.get("loss")
        if tl is not None:
            print(
                f"[Step {state.global_step}] epoch={getattr(state,'epoch',0):.2f} "
                f"train_loss={tl:.4f} lr={logs.get('learning_rate','N/A')} "
                f"dt={now-self.tlast:.2f}s elapsed={(now-self.t0)/60:.2f}m",
                flush=True,
            )
            self.tlast = now
    def on_evaluate(self, args, state, control, metrics=None, **kw):  
        if not state.is_local_process_zero: return
        metrics = metrics or {}
        now = time.time()
        el = metrics.get("eval_loss")
        if el is not None:
            print(
                f"[EVAL] epoch={getattr(state,'epoch',0):.2f} "
                f"eval_loss={el:.4f} dt={now-self.tlast:.2f}s "
                f"elapsed={(now-self.t0)/60:.2f}m", flush=True
            )
            self.tlast = now

def build_trainer(
    model,
    tokenizer,
    train_text,
    dev_text,
    lora_cfg: LoraConfig,
    sft_cfg: SFTConfig,
    max_seq_len: int,
) -> SFTTrainer:
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_text,
        eval_dataset=dev_text,
        peft_config=lora_cfg,
        args=sft_cfg,
        max_seq_length=max_seq_len,
    )
    try:
        trainer.model.print_trainable_parameters()
    except Exception:
        pass
    trainer.add_callback(LoggingCallback())
    return trainer

def save_and_log(trainer: SFTTrainer, tokenizer, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model()
    tokenizer.save_pretrained(out_dir)
    print("✅ Saved LoRA adapters and tokenizer to:", out_dir)

def final_eval(trainer: SFTTrainer) -> Dict[str, Any]:
    metrics = trainer.evaluate()
    print("Final evaluation metrics:", metrics)
    if "eval_loss" in metrics and metrics["eval_loss"] is not None:
        try:
            ppl = math.exp(float(metrics["eval_loss"]))
            print(f"Perplexity: {ppl:.3f}")
        except Exception:
            pass
    return metrics

def write_run_config(
    out_dir: Path,
    *,
    seed: int,
    base_model: str,
    train_path: Path,
    dev_path: Path,
    sft_cfg: SFTConfig,
    lora_cfg: LoraConfig,
    max_seq_len: int,
) -> None:
    with open(out_dir / "run_config.txt", "w", encoding="utf-8") as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"Train path: {train_path}\n")
        f.write(f"Dev path  : {dev_path}\n")
        f.write(f"Epochs: {sft_cfg.num_train_epochs}\n")
        f.write(f"LR: {sft_cfg.learning_rate}\n")
        f.write(f"Per-device batch: {sft_cfg.per_device_train_batch_size}\n")
        f.write(f"Grad accum: {sft_cfg.gradient_accumulation_steps}\n")
        f.write(f"Max seq len: {max_seq_len}\n")
        f.write(f"BF16: {sft_cfg.bf16}\n")
        f.write(f"LORA r/alpha/drop: {lora_cfg.r}/{lora_cfg.lora_alpha}/{lora_cfg.lora_dropout}\n")
        f.write(f"Transformers: {transformers.__version__}\n")
        f.write(f"Torch: {torch.__version__}\n")
    print("Saved run_config.txt")
