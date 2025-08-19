from __future__ import annotations
import argparse
from pathlib import Path

from . import config as C
from .dataset_utils import load_jsonl, fmt_token, fmt_no_token_counter, keep_only_text
from .training import (
    setup_env, env_report,
    build_tokenizer, build_model, build_trainer,
    save_and_log, final_eval, write_run_config,
)

def parse_args():
    p = argparse.ArgumentParser(description="LoRA SFT training for Saudi dialect prompts.")
    p.add_argument("--train", type=Path, default=C.DEFAULT_TRAIN, help="Train JSONL path.")
    p.add_argument("--dev",   type=Path, default=C.DEFAULT_DEV,   help="Dev JSONL path.")
    p.add_argument("--out",   type=Path, default=C.DEFAULT_OUT,   help="Output directory.")
    p.add_argument("--base-model", type=str, default=C.BASE_MODEL, help="HF model name or path.")
    p.add_argument("--max-seq-len", type=int, default=C.MAX_SEQ_LEN)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--mode", choices=["token", "no-token"], default="token",
                   help="token: keep tag; no-token: strip leading <DIALECT=...> tag.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--no-bf16", action="store_true", help="Disable bf16.")
    return p.parse_args()

def main():
    args = parse_args()
    setup_env(args.seed)
    env_report()

    # Load datasets
    train_ds = load_jsonl(args.train)
    dev_ds   = load_jsonl(args.dev)

    if args.mode == "no-token":
        fmt, counters = fmt_no_token_counter()
        train_text = keep_only_text(train_ds.map(fmt))
        dev_text   = keep_only_text(dev_ds.map(fmt))
        print(f"Leading <DIALECT=…> tags removed: {counters['count']}/{counters['total']} "
              f"({(100.0*counters['count']/max(1,counters['total'])):.2f}%)")
    else:
        train_text = keep_only_text(train_ds.map(fmt_token))
        dev_text   = keep_only_text(dev_ds.map(fmt_token))

    print(f"Train examples: {len(train_text):,} | Dev examples: {len(dev_text):,}")

    tok = build_tokenizer(args.base_model)
    mdl = build_model(args.base_model)

    lora_cfg = C.make_lora_config()
    sft_cfg  = C.make_sft_config(
        args.out,
        epochs=args.epochs,
        per_device_batch=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        bf16=(not args.no_bf16),
        seed=args.seed,
    )

    trainer = build_trainer(
        model=mdl,
        tokenizer=tok,
        train_text=train_text,
        dev_text=dev_text,
        lora_cfg=lora_cfg,
        sft_cfg=sft_cfg,
        max_seq_len=args.max_seq_len,
    )

    # Train → Save → Eval → Snapshot config
    print("Starting training…", flush=True)
    trainer.train()

    save_and_log(trainer, tok, args.out)
    metrics = final_eval(trainer)

    write_run_config(
        out_dir=args.out,
        seed=args.seed,
        base_model=args.base_model,
        train_path=args.train,
        dev_path=args.dev,
        sft_cfg=sft_cfg,
        lora_cfg=lora_cfg,
        max_seq_len=args.max_seq_len,
    )

if __name__ == "__main__":
    main()
