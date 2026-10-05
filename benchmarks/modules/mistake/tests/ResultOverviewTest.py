"""Reporting pairs actual cached inputs, not implementation-dependent case IDs."""

import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from benchmarks.modules.mistake.MistakeNotebook import MistakeNotebook


class PairingTests(unittest.TestCase):

    def fixture(self, root, run, case_id, midi=b"identical", truth=None):
        folder = root / run / "cases" / case_id
        folder.mkdir(parents=True)
        (folder / "performed.mid").write_bytes(midi)
        (folder / "manifest.json").write_text(
            json.dumps(
                dict(spec={"source_hash": "score"}, midi=str(folder / "performed.mid"))
            )
        )
        (folder / "net_truth.json").write_text(json.dumps({"net_truth": truth or []}))
        return pd.DataFrame(
            [dict(source="score.mid", seed=0, rate=0.25, case_id=case_id)]
        )

    def test_changed_case_ids_identical_inputs_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fixture(root, "audio", "new")
            b = self.fixture(root, "symbolic", "old")
            MistakeNotebook.validate_pairing(a, b, root / "audio", root / "symbolic")
            MistakeNotebook.validate_pairing(a, b, root / "audio")

    def test_changed_midi_or_truth_is_rejected(self):
        for mismatch in ({"midi": b"different"}, {"truth": [{"type": "insertion"}]}):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                a = self.fixture(root, "audio", "new")
                b = self.fixture(root, "symbolic", "old", **mismatch)
                with self.assertRaisesRegex(ValueError, "MIDI or labels differ"):
                    MistakeNotebook.validate_pairing(
                        a, b, root / "audio", root / "symbolic"
                    )

    def test_different_seed_selection_is_rejected(self):
        a = pd.DataFrame([dict(source="score.mid", seed=0, rate=0.25, case_id="new")])
        b = a.copy()
        b["seed"] = 1
        with self.assertRaisesRegex(ValueError, "selections differ"):
            MistakeNotebook.validate_pairing(a, b, Path("."))


if __name__ == "__main__":
    unittest.main()
