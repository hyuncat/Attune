"""Native data isolation, semantic labels, selection and resumability contracts."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pretty_midi
from benchmarks.modules.mistake.datasets.NativeDatasets import (
    pair_piece,
    test_ids,
    digest,
    save_json,
)
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker


def entry(path):
    return dict(type="file", path=path)


class NativeTests(unittest.TestCase):

    def test_official_test_split_only(self):
        split = {
            "midi_filename": {"0": "year/a.midi", "1": "b.mid", "2": "c.mid"},
            "split": {"0": "test", "1": "train", "2": "validation"},
        }
        self.assertEqual(test_ids(split), {"a"})

    def test_stems_pair_by_identity_not_sorted_position(self):
        entries = {
            "score_midi": [entry("score/p/stems_midi/0_double_bass.mid")],
            "score_audio": [entry("score/p/stems_audio/0_Double Bass.wav")],
            "performance": [entry("mistake/p/stems_audio/0_Double Bass.wav")],
        }
        for kind in ("extra", "missed", "correct"):
            entries[kind] = [
                entry(f"{kind}/p/99_unused.mid"),
                entry(f"{kind}/p/0_double_bass.mid"),
            ]
        (row,) = pair_piece("coco", "p", entries)
        self.assertTrue(row["files"]["extra"]["path"].endswith("0_double_bass.mid"))
        entries["missed"] = []
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            pair_piece("coco", "p", entries)

    def test_empty_classes_do_not_relabel_correct_as_extra(self):
        note = dict(kind="correct", onset=1.0, end=2.0, pitch=60)
        rows = MistakeBenchmarker.native_event_metrics([note], [note])
        native = {
            r["metric"]: r["f1"] for r in rows if r["protocol"] == "native_style_macro"
        }
        self.assertEqual(native["correct"], 1.0)
        self.assertEqual(native["extra"], 0.0)
        self.assertEqual(native["three_class_average"], 1 / 3)
        self.assertTrue(all((r["fp"] == 0 for r in rows if "fp" in r)))

    def test_common_gate_and_class_specific_one_to_one_matching(self):
        truth = [dict(kind="extra", onset=1.0, end=2.0, pitch=60)]
        predicted = [dict(kind="extra", onset=1.075, end=2.0, pitch=60)] * 2
        rows = MistakeBenchmarker.native_event_metrics(predicted, truth)
        common = {r["tolerance"]: r for r in rows if r["metric"] == "audio_pitch"}
        self.assertEqual(
            (common[0.05]["tp"], common[0.05]["fp"], common[0.05]["fn"]), (0, 2, 1)
        )
        self.assertEqual(
            (common[0.1]["tp"], common[0.1]["fp"], common[0.1]["fn"]), (1, 1, 0)
        )

    def test_native_midi_reader_preserves_chords(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "piano.mid"
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(0)
            inst.notes = [pretty_midi.Note(90, p, 1.0, 2.0) for p in (60, 64, 67)]
            pm.instruments.append(inst)
            pm.write(str(path))
            self.assertEqual(
                [
                    e["pitch"]
                    for e in MistakeBenchmarker.native_midi_events(path, "missed")
                ],
                [60, 64, 67],
            )
            json.dumps(MistakeBenchmarker.native_midi_events(path, "missed"))

    def test_coco_weights_and_native_contiguous_inference(self):
        models = MistakeBenchmarker.native_models_for("coco", "cpu")
        self.assertEqual(models["PolyTune"].checkpoint.name, "coco.ckpt")
        self.assertEqual(models["LadderSym"].checkpoint.name, "coco_prompted.ckpt")
        self.assertTrue(models["LadderSym"].contiguous_inference)
        with self.assertRaisesRegex(ValueError, "Only native CocoChorales-E"):
            MistakeBenchmarker.native_models_for("unsupported", "cpu")

    def test_resume_no_truth_in_attune_and_changed_assets_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {}
            for key in (
                "score_midi",
                "score_audio",
                "performance",
                "extra",
                "missed",
                "correct",
            ):
                path = root / key
                path.write_text(key)
                files[key] = str(path)
            case = dict(
                case_id="p/s",
                piece="p",
                stem="s",
                files=files,
                hashes={k: digest(v) for k, v in files.items()},
            )
            manifest = root / "manifest.json"
            save_json(manifest, dict(dataset="coco", split="test", cases=[case]))
            with patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_code_identity",
                return_value={"test": "v1"},
            ), patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_attune_events",
                return_value={"events": []},
            ) as infer, patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_midi_events",
                return_value=[],
            ):
                result = MistakeBenchmarker.native_run(
                    manifest, root / "run", methods=("Attune",), workers=1
                )
                self.assertEqual(len(result), 13)
                infer.assert_called_once_with(
                    files["performance"], files["score_midi"], midi_range=(21, 108)
                )
                infer.reset_mock()
                MistakeBenchmarker.native_run(
                    manifest, root / "run", methods=("Attune",), workers=1
                )
                infer.assert_not_called()
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    MistakeBenchmarker.native_run(
                        manifest,
                        root / "run",
                        methods=("Attune",),
                        midi_range=(30, 100),
                        workers=1,
                    )
                Path(files["extra"]).write_text("changed labels")
                with self.assertRaisesRegex(ValueError, "asset changed"):
                    MistakeBenchmarker.native_run(
                        manifest, root / "run", methods=("Attune",), workers=1
                    )

    def test_inference_failure_never_creates_zero_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.write_text("fixture")
            case = dict(
                case_id="p/s",
                piece="p",
                files={"performance": str(source), "score_midi": str(source)},
                hashes={"performance": digest(source), "score_midi": digest(source)},
            )
            manifest = root / "manifest.json"
            save_json(manifest, dict(dataset="coco", split="test", cases=[case]))
            with patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_code_identity",
                return_value={},
            ), patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_attune_events",
                side_effect=RuntimeError("failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    MistakeBenchmarker.native_run(
                        manifest, root / "run", methods=("Attune",), workers=1
                    )
            self.assertFalse(list((root / "run").rglob("result.json")))


if __name__ == "__main__":
    unittest.main()
