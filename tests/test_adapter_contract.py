import unittest

from evaluation.allam_eval.adapter_contract import load_required_adapter


class AdapterContractTests(unittest.TestCase):
    def test_returns_loaded_adapter(self):
        sentinel = object()

        def loader(base, path):
            self.assertEqual(base, "base")
            self.assertEqual(path, "adapter")
            return sentinel

        self.assertIs(load_required_adapter(loader, "base", "adapter", "model-a"), sentinel)

    def test_failure_is_not_downgraded_to_base_model(self):
        def loader(base, path):
            raise OSError("missing adapter")

        with self.assertRaisesRegex(
            RuntimeError,
            r"Failed to load required adapter for model-a: adapter/path",
        ):
            load_required_adapter(loader, object(), "adapter/path", "model-a")


if __name__ == "__main__":
    unittest.main()
