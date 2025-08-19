from __future__ import annotations
import argparse
from pathlib import Path
import torch

from . import config as C
from .data_utils import read_jsonl, subset_rows, set_seeds
from .generation import run_generation
from .evaluation import evaluate_pred_files, save_summary_csv

def parse_args():
    p = argparse.ArgumentParser(description="ALLaM evaluations: generate predictions and score metrics.")
    p.add_argument("--test", type=Path, default=C.TEST_PATH, help="Test JSONL file.")
    p.add_argument("--out-dir", type=Path, default=C.OUT_DIR, help="Output directory.")
    p.add_argument("--pred-dir", type=Path, default=C.PRED_DIR, help="Predictions subdir.")
    p.add_argument("--base-model", type=str, default=C.BASE_MODEL_ID, help="Base HF model id (info only).")
    p.add_argument("--max-eval", type=int, default=C.MAX_EVAL, help="0 = all; otherwise limit rows.")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--random-sample", action="store_true", default=C.RANDOM_SAMPLE)
    p.add_argument("--no-random-sample", action="store_false", dest="random_sample")
    return p.parse_args()

def main():
    args = parse_args()

    # prep dirs + seed
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)

    # load + subset test data
    test_rows = read_jsonl(args.test)
    print(f"Loaded test set: {len(test_rows):,}")
    rows_to_use = subset_rows(test_rows, args.max_eval, args.seed, args.random_sample)
    if len(rows_to_use) != len(test_rows):
        print(f"Using subset for eval: {len(rows_to_use)} / {len(test_rows)}")

    # generation
    pred_files = run_generation(rows_to_use, C.MODEL_SPECS, args.out_dir, args.pred_dir)

    # evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = evaluate_pred_files(pred_files, device=device, out_dir=args.out_dir)
    save_summary_csv(summary, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
