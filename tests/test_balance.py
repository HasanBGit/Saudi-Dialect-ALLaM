import json
import sys
import tempfile
import unittest
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_DIR))

from steps import step_balance


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class BalanceTests(unittest.TestCase):
    def test_rejects_missing_najdi_for_both_modes(self):
        rows = [
            {"instruction": "h1", "response": "r1", "dialect": "HIJAZI"},
            {"instruction": "h2", "response": "r2", "dialect": "HIJAZI"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.jsonl"
            write_rows(src, rows)
            for mode in ("downsample", "oversample"):
                with self.subTest(mode=mode):
                    with self.assertRaisesRegex(ValueError, "missing dialect rows for Najdi"):
                        step_balance(src, Path(tmp) / f"{mode}.jsonl", mode=mode)

    def test_balances_both_present(self):
        rows = [
            {"instruction": "h1", "response": "r1", "dialect": "HIJAZI"},
            {"instruction": "h2", "response": "r2", "dialect": "HIJAZI"},
            {"instruction": "n1", "response": "r3", "dialect": "NAJDI"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.jsonl"
            out = Path(tmp) / "out.jsonl"
            write_rows(src, rows)
            step_balance(src, out, mode="downsample", seed=42)
            result = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(result), 2)
            self.assertEqual({row["dialect"] for row in result}, {"HIJAZI", "NAJDI"})


if __name__ == "__main__":
    unittest.main()
