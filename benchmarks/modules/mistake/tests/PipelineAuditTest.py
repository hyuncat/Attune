import json
from pathlib import Path
import tempfile
import unittest
from algorithms.Config import Config
from app_logic.user.ds.PitchData import Pitch, PitchData
from benchmarks.modules.mistake.provenance.PipelineAudit import (
    frame_counts,
    note_matches,
)
from benchmarks.modules.mistake.tests.CompetitorComparisonTest import notes
from benchmarks.modules.mistake.MistakeCache import MistakeCache
from benchmarks.modules.mistake.competitors.PolyTune import PolyTune


class AuditMetricsTest(unittest.TestCase):

    def test_wrong_pitch_counts_as_both_frame_miss_and_false_pitch(self):
        cfg = Config()
        data = PitchData(cfg)
        data.data = [
            Pitch(t, 0.1, 0.0, 0.0, cfg, value=v, candidates=[(v, 1.0)])
            for t, v in [(0.0, 60.0), (0.1, 72.0), (0.2, -1.0), (0.3, 60.0)]
        ]
        result = frame_counts(data, list(notes([(0, 0.25, 60)]).data.values()))
        self.assertEqual(
            (result["ref_frames"], result["est_frames"], result["correct"]), (3, 3, 1)
        )
        self.assertEqual(
            (
                result["octave_errors"],
                result["unvoiced_on_note"],
                result["voiced_in_silence"],
            ),
            (1, 1, 1),
        )

    def test_note_matching_is_one_to_one_and_offset_gate_is_optional(self):
        reference = list(notes([(0, 0.5, 60)]).data.values())
        estimate = list(notes([(0, 1.0, 60), (0.01, 0.51, 60)]).data.values())
        self.assertEqual(len(note_matches(reference, estimate)), 1)
        self.assertEqual(note_matches(reference, estimate, offsets=True), [(1, 0)])


class PolyTuneReuseTest(unittest.TestCase):

    def test_scope_retains_model_and_scoring_but_ignores_other_competitors(self):
        job = dict(
            method="PolyTune",
            packages={"PolyTune": {"weights": "a"}, "TheGlueNote": "old", "numpy": "1"},
            code={
                "algorithms/NoteDetector.py": "old",
                "benchmarks/modules/mistake/competitors/Nakamura.py": "old",
                "benchmarks/modules/mistake/MistakeDetectorBase.py": "score",
            },
        )
        changed = json.loads(json.dumps(job))
        changed["packages"]["TheGlueNote"] = "new"
        changed["code"]["algorithms/NoteDetector.py"] = "new"
        self.assertEqual(MistakeCache.scoped_job(job), MistakeCache.scoped_job(changed))
        changed["packages"]["PolyTune"]["weights"] = "b"
        self.assertNotEqual(
            MistakeCache.scoped_job(job), MistakeCache.scoped_job(changed)
        )

    def test_audio_and_model_must_match_before_reusing_raw_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old"
            case = old / "cases" / "case"
            (case / "polytune").mkdir(parents=True)
            (case / "score_audio").mkdir()
            audio = case / "performance.wav"
            audio.write_bytes(b"performance")
            score = case / "score_audio/clean_score.wav"
            score.write_bytes(b"score")
            (case / "manifest.json").write_text(json.dumps(dict(audio=str(audio))))
            (case / "polytune/polytune.json").write_text(
                json.dumps(
                    dict(events=[dict(kind="extra", onset=0, pitch=60)], cpu_seconds=10)
                )
            )
            code = Path(__file__).resolve().parent.parent / "competitors"
            (old / "run.json").write_text(
                json.dumps(
                    dict(
                        status="complete",
                        packages={"PolyTune": {"weights": "a"}},
                        code={
                            "benchmarks/modules/mistake/competitors/"
                            + n: PolyTune.sha256(code / n)
                            for n in ["PolyTune.py"]
                        },
                    )
                )
            )
            task = dict(
                directory=str(root / "new/cases/case/polytune"),
                audio=str(audio),
                score_audio=str(score),
            )
            MistakeCache.candidates.cache_clear()
            result = MistakeCache.cached_prediction(task, {"weights": "a"})
            self.assertEqual(result["cpu_seconds"], 10)
            self.assertEqual(result["execution_cpu_seconds"], 0)
            self.assertTrue(result["inference_cache_hit"])
            self.assertIsNone(MistakeCache.cached_prediction(task, {"weights": "b"}))
            other = root / "different.wav"
            other.write_bytes(b"different performance")
            self.assertIsNone(
                MistakeCache.cached_prediction(
                    dict(task, audio=str(other)), {"weights": "a"}
                )
            )
            MistakeCache.candidates.cache_clear()


if __name__ == "__main__":
    unittest.main()
