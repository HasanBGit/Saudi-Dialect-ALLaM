import argparse
from pathlib import Path

from common import (
    DEFAULT_IN, OUT_CLEAN, OUT_BAL, OUT_CSV,
    BALANCE_MODE, SEED
)
from steps import step_clean, step_balance, step_export_csv  # note: no need to import add_tags

def main():
    p = argparse.ArgumentParser(
        description="Saudi Dialect pipeline: CLEAN (remove tags + banned topics) → BALANCE (50/50) → CSV (final)."
    )
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN,
                   help="Input JSONL (instruction/response/dialect, optional meta.topic)")
    p.add_argument("--out-clean",  type=Path, default=OUT_CLEAN,  help="Cleaned output path (intermediate)")
    p.add_argument("--out-bal",    type=Path, default=OUT_BAL,    help="Balanced output path (intermediate)")
    p.add_argument("--csv-out",    type=Path, default=OUT_CSV,    help="Final CSV output path")

    p.add_argument("--balance", choices=["downsample","oversample"], default=BALANCE_MODE, help="Balance mode")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed")

    # in this version, CSV is always produced; tags are never added (no --skip flags necessary)
    args = p.parse_args()

    # 1) CLEAN (remove leading tags + banned topics)
    step_clean(args.inp, args.out_clean)

    # 2) BALANCE 50/50 (Hijazi/Najdi)
    step_balance(args.out_clean, args.out_bal, mode=args.balance, seed=args.seed)

    # 3) CSV (final)
    step_export_csv(args.out_bal, args.csv_out)

if __name__ == "__main__":
    main()
