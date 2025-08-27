import importlib
import sys
import types
import unittest


class TestsPackageInitSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force a clean import to observe top-level execution deterministically
        sys.modules.pop("tests", None)
        cls.pkg = importlib.import_module("tests")

    def test_package_imports_cleanly_and_is_package(self):
        # Happy path: package imports and exposes __path__
        self.assertIsInstance(self.pkg, types.ModuleType)
        self.assertTrue(hasattr(self.pkg, "__path__"), "__path__ missing; 'tests' should be a package")

    def test_dunder_all_is_sequence_of_unique_public_names_if_defined(self):
        # Edge case: __all__ may be undefined; then skip as non-applicable
        if not hasattr(self.pkg, "__all__"):
            self.skipTest("__all__ not defined on tests package")
        all_ = self.pkg.__all__
        self.assertIsInstance(all_, (list, tuple), "__all__ must be list or tuple")
        seen = set()
        for name in all_:
            self.assertIsInstance(name, str, "__all__ entries must be strings")
            self.assertNotEqual(name.strip(), "", "__all__ must not contain empty strings")
            self.assertFalse(name.startswith("_"), f"__all__ contains private name: {name}")
            self.assertNotIn(name, seen, f"Duplicate name in __all__: {name}")
            seen.add(name)
            self.assertTrue(hasattr(self.pkg, name), f"Export '{name}' listed in __all__ but not found on module")

    def test_pytest_plugins_is_well_formed_if_defined(self):
        # Some projects expose pytest plugins via tests.pytest_plugins
        if not hasattr(self.pkg, "pytest_plugins"):
            self.skipTest("pytest_plugins not present")
        plugins = self.pkg.pytest_plugins
        if isinstance(plugins, str):
            plugins = [plugins]
        self.assertIsInstance(plugins, (list, tuple), "pytest_plugins must be a string, list, or tuple")
        for p in plugins:
            self.assertIsInstance(p, str, "pytest_plugins entries must be strings")
            self.assertNotEqual(p.strip(), "", "pytest_plugins entries must not be empty")

    def test_version_attribute_semver_like_if_present(self):
        # Validate common version attribute names without enforcing presence
        for attr in ("__version__", "VERSION", "version"):
            if hasattr(self.pkg, attr):
                value = getattr(self.pkg, attr)
                self.assertIsInstance(value, str, f"{attr} must be a string")
                self.assertRegex(
                    value,
                    r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z\.-]+)?$",
                    f"{attr} should look like a semantic version (e.g., 1.2.3 or 1.2.3-alpha)",
                )
                break
        else:
            self.skipTest("No version attribute found on tests package")

    def test_reimport_is_idempotent_for_public_api(self):
        # Re-import and ensure public symbol set and types remain stable
        before_public = {k: type(v) for k, v in vars(self.pkg).items() if not k.startswith("_")}
        sys.modules.pop("tests", None)
        pkg2 = importlib.import_module("tests")
        after_public = {k: type(v) for k, v in vars(pkg2).items() if not k.startswith("_")}
        self.assertEqual(set(before_public.keys()), set(after_public.keys()), "Public names changed across imports")
        for name in before_public:
            self.assertEqual(before_public[name], after_public[name], f"Type changed for public symbol: {name}")

    def test_docstring_is_string_if_present(self):
        doc = getattr(self.pkg, "__doc__", None)
        if doc is None:
            self.skipTest("Package has no docstring")
        else:
            self.assertIsInstance(doc, str, "__doc__ should be a string")


if __name__ == "__main__":
    unittest.main()