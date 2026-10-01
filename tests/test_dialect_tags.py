import sys
import types
import unittest

# dataset_utils only needs the Dataset symbol at import time for these formatter tests.
if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.Dataset = object
    sys.modules["datasets"] = datasets_stub

from data.common import DIALECT_TAG_RE, TAG_RE
from training.dataset_utils import fmt_no_token_counter


class DialectTagTests(unittest.TestCase):
    def test_common_parser_accepts_canonical_tags(self):
        self.assertEqual(TAG_RE.sub("", "<DIALECT=HIJAZI> hello"), "hello")
        self.assertEqual(TAG_RE.sub("", "<DIALECT=NAJDI> hello"), "hello")

    def test_common_parser_accepts_documented_short_tags(self):
        self.assertEqual(TAG_RE.sub("", "<HIJAZI> hello"), "hello")
        self.assertEqual(TAG_RE.sub("", "<NAJDI> hello"), "hello")

    def test_no_token_formatter_strips_both_forms(self):
        formatter, counters = fmt_no_token_counter()
        short = formatter({"instruction": "<HIJAZI> salam", "response": "ok"})
        canonical = formatter({"instruction": "<DIALECT=NAJDI> hala", "response": "ok"})
        self.assertIn("### Instruction:\nsalam\n", short["text"])
        self.assertIn("### Instruction:\nhala\n", canonical["text"])
        self.assertEqual(counters, {"count": 2, "total": 2})

    def test_parser_leaves_other_tags_untouched(self):
        self.assertEqual(TAG_RE.sub("", "<MSA> hello"), "<MSA> hello")

    def test_echo_matcher_finds_both_forms_anywhere(self):
        self.assertIsNotNone(DIALECT_TAG_RE.search("reply <HIJAZI> text"))
        self.assertIsNotNone(DIALECT_TAG_RE.search("reply <DIALECT=NAJDI> text"))
        self.assertIsNone(DIALECT_TAG_RE.search("reply <MSA> text"))


if __name__ == "__main__":
    unittest.main()
