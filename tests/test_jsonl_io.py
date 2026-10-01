import tempfile
import unittest
from pathlib import Path

from data.common import read_jsonl


class JsonlReaderTests(unittest.TestCase):
    def test_skips_only_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.jsonl"
            path.write_text('{"x": 1}\n\n  \n{"x": 2}\n', encoding="utf-8")
            self.assertEqual(list(read_jsonl(path)), [{"x": 1}, {"x": 2}])

    def test_malformed_json_reports_path_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('{"x": 1}\n{not-json\n{"x": 2}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                list(read_jsonl(path))
            message = str(ctx.exception)
            self.assertIn(str(path), message)
            self.assertIn(":2:", message)
            self.assertIn("invalid JSON", message)


if __name__ == "__main__":
    unittest.main()
