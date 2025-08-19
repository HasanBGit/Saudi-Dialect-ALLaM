from __future__ import annotations
from pathlib import Path
from trl import SFTConfig
from peft import LoraConfig

# -------- Defaults (override from CLI if you like) --------
BASE_MODEL  = "ALLaM-AI/ALLaM-7B-Instruct-preview"
MAX_SEQ_LEN = 2048
SEED        = 42
DEFAULT_TRAIN = Path("data_splits/train.jsonl")
DEFAULT_DEV   = Path("data_splits/dev.jsonl")
DEFAULT_OUT   = Path("outputs/allam7b-lora-token-15EPOCH")

def make_lora_config() -> LoraConfig:
    """LoRA config matching your script."""
    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

def make_sft_config(
    out_dir: Path,
    *,
    epochs: int = 15,
    per_device_batch: int = 2,
    grad_accum: int = 8,
    lr: float = 5e-5,
    warmup_ratio: float = 0.1,
    bf16: bool = True,
    seed: int = SEED,
) -> SFTConfig:
    
    return SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=warmup_ratio,
        bf16=bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        gradient_checkpointing=True,
        report_to="tensorboard",
        save_total_limit=3,
        seed=seed,
        packing=True,
        dataset_text_field="text",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=str(out_dir / "logs"),
        logging_first_step=True,
        remove_unused_columns=False,
    )
