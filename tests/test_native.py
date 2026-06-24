import types
import unittest

from agent.native import NativeExtensionError, load_cast_core


class NativeExtensionLoaderTests(unittest.TestCase):
    def test_missing_cast_core_raises_actionable_error(self):
        def missing(_name):
            raise ModuleNotFoundError("missing cast_core", name="cast_core")

        with self.assertRaises(NativeExtensionError) as caught:
            load_cast_core(importer=missing)

        message = str(caught.exception)
        self.assertIn("cast_core", message)
        self.assertIn("bash build.sh", message)
        self.assertIn("Linux/WSL Python", message)

    def test_nested_module_not_found_is_not_rewritten(self):
        def missing_nested(_name):
            raise ModuleNotFoundError("missing dependency", name="some_dependency")

        with self.assertRaises(ModuleNotFoundError) as caught:
            load_cast_core(importer=missing_nested)

        self.assertEqual(caught.exception.name, "some_dependency")

    def test_loader_returns_imported_module(self):
        module = types.ModuleType("cast_core")

        def importer(name):
            self.assertEqual(name, "cast_core")
            return module

        self.assertIs(load_cast_core(importer=importer), module)


if __name__ == "__main__":
    unittest.main()
