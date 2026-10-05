"""Notebook requests must use current defaults and preserve run options."""

from dataclasses import asdict
import tempfile
import unittest
from unittest.mock import patch
from algorithms.Config import Config
from benchmarks.modules.mistake.MistakeNotebook import MistakeNotebook


class NotebookRunTests(unittest.TestCase):

    def request(self, output, stage):
        return dict(
            stage=stage,
            output=output,
            sources=["source.mid"],
            source_metadata={},
            expected_defaults=asdict(Config()),
            seeds=[0, 1],
            rates=[0.0, 0.25],
            methods=["Attune (no refinement)"],
            tolerances=[0.1],
            workers=2,
            polytune_device="cpu",
            neural_workers=None,
            polytune_last=True,
            force_pitch_detection=True,
        )

    def test_reject_defaults_changed_since_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self.request(tmp, "audio")
            request["expected_defaults"]["alignment_gamma_pitch"] += 1
            with patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.run_comparison"
            ) as runner:
                with self.assertRaisesRegex(ValueError, "defaults changed"):
                    MistakeNotebook.run(request)
                runner.assert_not_called()

    def test_stage_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.run_comparison"
            ) as audio:
                MistakeNotebook.run(self.request(tmp, "audio"))
                self.assertEqual(
                    audio.call_args.kwargs["input_kinds"], ("detected", "audio")
                )
                self.assertTrue(audio.call_args.kwargs["force_pitch_detection"])
                self.assertTrue(audio.call_args.kwargs["polytune_last"])
            with patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.run_symbolic_comparison"
            ) as symbolic:
                MistakeNotebook.run(self.request(tmp, "symbolic"))
                self.assertNotIn("force_pitch_detection", symbolic.call_args.kwargs)
                self.assertEqual(symbolic.call_args.kwargs["seeds"], [0, 1])


if __name__ == "__main__":
    unittest.main()
