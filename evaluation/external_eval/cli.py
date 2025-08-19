from __future__ import annotations
import argparse
from pathlib import Path

import torch
from huggingface_hub import login

from . import config as C
from .data_utils import read_jsonl, subset_rows, set_seeds
from .generation import run_many
from .evaluation import evaluate_one_pred_file, save_summary_csv

def parse_args():
    p = argparse.ArgumentParser(description="External HF models: generate on Saudi test set + evaluate.")
    p.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (or set HF_TOKEN env).")
    p.add_argument("--models", type=str, nargs="*", default=None,
                   help="Optional override: list of HF model IDs to evaluate.")
    p.add_argument("--use-4bit", action="store_true", help="Force 4-bit for all models.")
    p.add_argument("--max-eval", type=int, default=C.MAX_EVAL)
    p.add_argument("--random-sample", action="store_true", default=C.RANDOM_SAMPLE)
    p.add_argument("--no-random-sample", action="store_false", dest="random_sample")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--test", type=Path, default=C.TEST_PATH)
    p.add_argument("--out-dir", type=Path, default=C.OUT_DIR)
    p.add_argument("--pred-dir", type=Path, default=C.PRED_DIR)  # used as base name; subdirs will be added
    p.add_argument("--batch-size-gen", type=int, default=C.BATCH_SIZE_GEN)
    p.add_argument("--batch-size-clf", type=int, default=C.BATCH_SIZE_CLF)
    # generation hyperparams
    p.add_argument("--max-new-tokens", type=int, default=C.GEN_KW["max_new_tokens"])
    p.add_argument("--temperature", type=float, default=C.GEN_KW["temperature"])
    p.add_argument("--top-p", type=float, default=C.GEN_KW["top_p"])
    p.add_argument("--top-k", type=int, default=C.GEN_KW["top_k"])
    return p.parse_args()

def main():
    args = parse_args()

    # Login if provided
    token = args.hf_token or None
    if token:
        login(token=token)

    # Seeds
    set_seeds(args.seed)

    # Prepare dirs
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load & subset test set
    rows = read_jsonl(args.test)
    print(f"Loaded test set: {len(rows):,}")
    rows_to_use = subset_rows(rows, args.max_eval, args.seed, args.random_sample)
    if len(rows_to_use) != len(rows):
        print(f"Using subset for eval: {len(rows_to_use)} / {len(rows)}")

    # Model list
    if args.models:
        # user-specified models: use a common base out_dir name with suffixes
        models = [(m, m.split("/")[-1], args.use_4bit) for m in args.models]
    else:
        # defaults from config (each with its own subdir & 4-bit flag)
        models = C.EXTERNAL_MODELS
        if args.use_4bit:
            models = [(mid, sub, True) for (mid, sub, _u4) in models]

    # Generation (each model writes to its own OUT_DIR variant)
    gen_kw = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=True,
    )
    pred_files = run_many(models, rows_to_use, base_out_dir=args.out_dir, gen_kw=gen_kw, batch_size_gen=args.batch_size_gen)

    # Evaluation (per prediction file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = []
    for pf in pred_files:
        out_dir = pf.parent.parent  # go from ".../preds/model.jsonl" -> ".../"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.append(evaluate_one_pred_file(pf, device=device, bs_clf=args.batch_size_clf, out_dir=out_dir))

    # Write summary CSV for the last out_dir group (or just base)
    # If you passed multiple models with different subdirs, you’ll get separate CSVs; here we also create one at base.
    save_summary_csv(summary, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
