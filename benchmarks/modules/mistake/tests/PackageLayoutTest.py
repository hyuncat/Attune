"""Keep the agreed package boundaries explicit as benchmark features grow."""

import ast
from pathlib import Path
import unittest


class PackageLayoutTest(unittest.TestCase):
    def test_root_implementation_files_follow_the_shared_contract(self):
        modules = Path(__file__).resolve().parents[2]
        for package in ("pitch", "mistake"):
            prefix = package.title()
            expected = {
                prefix + suffix + ".py"
                for suffix in ("Benchmarker", "DetectorBase", "Notebook", "Cache")
            }
            actual = {
                p.name
                for p in (modules / package).glob("*.py")
                if p.name != "__init__.py"
            }
            self.assertEqual(actual, expected, package)

    def test_competitors_have_no_standalone_setup_workers_or_functions(self):
        modules = Path(__file__).resolve().parents[2]
        for package in ("pitch", "mistake"):
            for path in (modules / package / "competitors").glob("*.py"):
                self.assertFalse(path.stem.startswith("setup_"), str(path))
                self.assertFalse(
                    path.stem.endswith(("Worker", "Frames", "Reuse")), str(path)
                )
                tree = ast.parse(path.read_text())
                self.assertFalse(
                    any(
                        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        for n in tree.body
                    ),
                    str(path),
                )

    def test_mistake_adapters_share_the_base_contract(self):
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
        from benchmarks.modules.mistake.competitors.PolyTune import PolyTune
        from benchmarks.modules.mistake.competitors.LadderSym import LadderSym
        from benchmarks.modules.mistake.competitors.Nakamura import Nakamura

        for adapter in (PolyTune, LadderSym, Nakamura):
            self.assertIsInstance(adapter(), MistakeDetectorBase)
