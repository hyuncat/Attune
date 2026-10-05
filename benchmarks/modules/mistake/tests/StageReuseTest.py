"""Small cache fixtures only: no benchmark, model or pitch inference."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import asdict
from algorithms.Config import Config
from benchmarks.modules.mistake.MistakeCache import (
    GENERATION_EQUIVALENT,
    LADDER_DEFAULT_WORKER_UPGRADE,
    MistakeCache,
    WORKER_IMPORT_UPGRADES,
    __file__ as cache_file,
)
from benchmarks.modules.mistake.competitors.PolyTune import PolyTune


class StageReuseTests(unittest.TestCase):

    def test_generation_identity_ignores_cache_plumbing_not_injector(self):
        old, new_hash = GENERATION_EQUIVALENT["MistakeCases.py"][:2]
        spec = dict(
            seed=0,
            rate=0.25,
            code={"MistakeCases.py": old, "MistakeInjector.py": "same"},
        )
        new = dict(spec, code=dict(spec["code"], **{"MistakeCases.py": new_hash}))
        self.assertEqual(
            MistakeCache.generation_signature(spec),
            MistakeCache.generation_signature(new),
        )
        new["code"]["MistakeInjector.py"] = "changed"
        self.assertNotEqual(
            MistakeCache.generation_signature(spec),
            MistakeCache.generation_signature(new),
        )

    def test_case_reuses_inputs_and_pitch_but_never_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old"
            case = old / "cases/key"
            case.mkdir(parents=True)
            spec = dict(
                source_hash="abc", seed=1, rate=0.25, code={"MistakeInjector.py": "v1"}
            )
            item = dict(dataset="coco", track_id="s", spec=spec)
            for k in ("audio", "midi", "truth", "pitch_data", "note_data"):
                path = case / k
                path.write_text(k)
                item[k] = str(path)
            Path(item["pitch_data"] + ".mistake-range.json").write_text("{}")
            (case / "manifest.json").write_text(json.dumps(item))
            (old / "run.json").write_text(json.dumps({"code": {}}))
            (old / "production_defaults.json").write_text(json.dumps(asdict(Config())))
            bench = MagicMock()
            bench.MISTAKE_DIR = root / "new/cases/newkey"
            paths = {
                k: bench.MISTAKE_DIR / k
                for k in ("audio", "midi", "truth", "pitch_data", "note_data")
            }
            bench.mistake_db_paths.return_value = paths
            MistakeCache.asset_candidates.cache_clear()
            got = MistakeCache.reuse_case_assets(bench, spec, root / "new")
            self.assertEqual(Path(got["pitch_data"]).read_text(), "pitch_data")
            self.assertFalse(paths["note_data"].exists())
            paths["audio"].write_text("new")
            self.assertEqual(Path(item["audio"]).read_text(), "audio")

    def test_laddersym_requires_identical_prompt_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "old/cases/key"
            case.mkdir(parents=True)
            identity = {"checkpoint": "abc"}
            import benchmarks.modules.mistake.MistakeCache as _api_MistakeCache

            code = Path(cache_file).parent / "competitors"
            metadata = {
                "status": "running",
                "packages": {"LadderSym": identity},
                "code": {
                    "benchmarks/modules/mistake/competitors/"
                    + name: PolyTune.sha256(code / name)
                    for name in ("LadderSym.py", "PolyTune.py")
                },
            }
            import benchmarks.modules.mistake.MistakeCache as _api_MistakeCache

            (root / "old/run.json").write_text(json.dumps(metadata))
            audio = case / "take.wav"
            audio.write_bytes(b"audio")
            score = case / "score_audio/clean_score.wav"
            score.parent.mkdir()
            score.write_bytes(b"score")
            midi = case / "clean_score.mid"
            midi.write_bytes(b"prompt")
            (case / "manifest.json").write_text(json.dumps({"audio": str(audio)}))
            prediction = case / "laddersym/laddersym.json"
            prediction.parent.mkdir()
            prediction.write_text('{"events": []}')
            task = dict(
                directory=str(root / "new/cases/key/laddersym"),
                audio=str(audio),
                score_audio=str(score),
                score_midi=str(midi),
            )
            MistakeCache.candidates.cache_clear()
            self.assertTrue(
                MistakeCache.cached_prediction(task, identity, method="LadderSym")[
                    "inference_cache_hit"
                ]
            )
            other = root / "changed.mid"
            other.write_bytes(b"other")
            task["score_midi"] = str(other)
            self.assertIsNone(
                MistakeCache.cached_prediction(task, identity, method="LadderSym")
            )

    def test_relocated_worker_reuse_accepts_only_audited_hashes(self):
        from benchmarks.modules.mistake.MistakeCache import (
            LADDER_DEFAULT_WORKER_UPGRADE,
            MistakeCache,
            WORKER_IMPORT_UPGRADES,
        )

        folder = Path(__file__).resolve().parent.parent / "competitors"
        for method, (original, relocated) in WORKER_IMPORT_UPGRADES.items():
            self.assertNotEqual(original, relocated)
            self.assertTrue(MistakeCache.compatible_worker(original, relocated, method))
            self.assertTrue(
                MistakeCache.compatible_worker(relocated, relocated, method)
            )
            self.assertFalse(
                MistakeCache.compatible_worker("unknown", relocated, method)
            )
            self.assertFalse(
                MistakeCache.compatible_worker(original, "changed", method)
            )
        relocated = WORKER_IMPORT_UPGRADES["LadderSym"][1]
        self.assertTrue(
            MistakeCache.compatible_worker(
                LADDER_DEFAULT_WORKER_UPGRADE[0], relocated, "LadderSym"
            )
        )
        self.assertFalse(
            MistakeCache.compatible_worker(
                LADDER_DEFAULT_WORKER_UPGRADE[0],
                relocated,
                "LadderSym",
                contiguous=True,
            )
        )

    def test_ladder_worker_upgrade_is_default_mode_only(self):
        from benchmarks.modules.mistake.MistakeCache import (
            LADDER_DEFAULT_WORKER_UPGRADE,
            MistakeCache,
        )

        before, after = LADDER_DEFAULT_WORKER_UPGRADE
        self.assertTrue(MistakeCache.compatible_worker(before, after, "LadderSym"))
        self.assertFalse(
            MistakeCache.compatible_worker(before, after, "LadderSym", contiguous=True)
        )
        self.assertFalse(MistakeCache.compatible_worker(before, after, "PolyTune"))
        self.assertFalse(MistakeCache.compatible_worker(before, "unknown", "LadderSym"))

    def test_native_note_cache_reuses_threshold_changes_not_segmentation_changes(self):
        from benchmarks.modules.mistake.MistakeCache import MistakeCache
        from app_logic.NoteData import Note, NoteData

        with tempfile.TemporaryDirectory() as tmp:
            rec = MagicMock()
            rec.config = Config()

            def detect():
                rec.note_data = NoteData()
                rec.note_data.write_note(Note(1, 0.1, 0.8, [60]))

            rec.detect_notes.side_effect = detect
            with patch("benchmarks.paths.REPO_ROOT", Path(tmp)), patch(
                "benchmarks.modules.mistake.MistakeCache.digest", return_value="abc"
            ):
                first = MistakeCache.cached_native_notes(
                    rec, {"path": "pitch"}, "score"
                )
                self.assertFalse(first["hit"])
                rec.config.pitch_tolerance = 0.5
                second = MistakeCache.cached_native_notes(
                    rec, {"path": "pitch"}, "score"
                )
                self.assertTrue(second["hit"])
                self.assertEqual(rec.detect_notes.call_count, 1)
                self.assertEqual(rec.note_data.data[0.1].end_time, 0.8)
                rec.resize_score.assert_called_once_with(
                    to_span="pitch", include_transitions=False
                )
                rec.config.min_note_length_cap = 0.05
                self.assertFalse(
                    MistakeCache.cached_native_notes(rec, {"path": "pitch"}, "score")[
                        "hit"
                    ]
                )
                self.assertEqual(rec.detect_notes.call_count, 2)

    def test_native_pitch_key_ignores_threshold_and_minimum(self):
        from benchmarks.modules.mistake.MistakeCache import MistakeCache

        rec = MagicMock()
        rec.config = Config()
        rec.pitch_data.data = []
        cache = MagicMock()
        cache.read.return_value = (rec.pitch_data, {})
        with patch(
            "benchmarks.modules.mistake.MistakeCache.digest", return_value="abc"
        ), patch(
            "benchmarks.modules.pitch.PitchCache.PitchCache", return_value=cache
        ) as factory:
            first = MistakeCache.cached_native_pitches(rec, "audio")["path"]
            rec.config.pitch_tolerance = 0.5
            rec.config.min_note_length_cap = 0.05
            self.assertEqual(
                first, MistakeCache.cached_native_pitches(rec, "audio")["path"]
            )
            rec.config.sr = 16000
            self.assertNotEqual(
                first, MistakeCache.cached_native_pitches(rec, "audio")["path"]
            )
            rec.detect_pitches.assert_not_called()


if __name__ == "__main__":
    unittest.main()
