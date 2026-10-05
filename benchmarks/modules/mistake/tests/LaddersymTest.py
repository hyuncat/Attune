"""Audio-model contracts: clean prompt, independent provenance and model dispatch."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pretty_midi
import soundfile as sf
from benchmarks.modules.mistake.MistakeBenchmarker import (
    COMPETITOR_AUDIO_METHODS,
    COMPETITOR_SYMBOLIC_METHODS,
    MistakeBenchmarker,
)
from benchmarks.modules.mistake.MistakeCache import MistakeCache
from benchmarks.modules.mistake.tests.CompetitorComparisonTest import notes


def multi_audio_case(task):
    from benchmarks.modules.mistake.tests.ParallelTest import synthetic_case

    rows, request = synthetic_case(dict(task, neural=True))
    return (
        rows,
        [
            dict(request, common={"method": method}, score_midi="clean.mid")
            for method in ("PolyTune", "LadderSym")
        ],
    )


class LadderSymTests(unittest.TestCase):

    def test_both_audio_tasks_share_audio_and_use_clean_prompt_without_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mid"
            original = [(i * 0.5, i * 0.5 + 0.4, 60 + i) for i in range(8)]
            MistakeBenchmarker.notedata_to_pm(notes(original), program=73).write(
                str(source)
            )

            def synth(self, midi, out_dir, **kwargs):
                path = Path(out_dir) / (Path(midi).stem + ".wav")
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.zeros(5 * 16000), 16000)
                return path

            with patch.object(MistakeBenchmarker, "synth_midi", synth), patch.object(
                MistakeBenchmarker,
                "load_mistake_pitches",
                side_effect=AssertionError("Audio model used Attune pitches"),
            ), patch(
                "app_logic.user.ds.Recording.Recording.detect_notes",
                side_effect=AssertionError("Audio model used Attune notes"),
            ):
                rows, requests = MistakeBenchmarker.evaluate_case(
                    source,
                    1,
                    0.8,
                    Path(tmp) / "run",
                    COMPETITOR_AUDIO_METHODS,
                    (0.1,),
                    input_kinds=("audio",),
                    defer_polytune=True,
                )
                cached_row = {"input": "detected", "method": "Attune (repeat only)"}
                resumed_rows, pending = MistakeBenchmarker.evaluate_case(
                    source,
                    1,
                    0.8,
                    Path(tmp) / "run",
                    ("Attune (repeat only)", "LadderSym"),
                    (0.1,),
                    input_kinds=("detected", "audio"),
                    cached={("detected", "Attune (repeat only)"): [cached_row]},
                    defer_polytune=True,
                )
                self.assertEqual(resumed_rows, [cached_row])
                self.assertEqual(pending["common"]["method"], "LadderSym")
            self.assertEqual(rows, [])
            self.assertEqual(
                [r["common"]["method"] for r in requests],
                list(COMPETITOR_AUDIO_METHODS),
            )
            self.assertEqual(requests[0]["audio"], requests[1]["audio"])
            self.assertEqual(requests[0]["score_audio"], requests[1]["score_audio"])
            self.assertNotEqual(requests[0]["directory"], requests[1]["directory"])
            prompt = pretty_midi.PrettyMIDI(requests[1]["score_midi"])
            np.testing.assert_allclose(
                [(n.start, n.end, n.pitch) for n in prompt.instruments[0].notes],
                original,
                atol=0.003,
            )
            self.assertEqual(prompt.instruments[0].program, 73)
            self.assertGreater(len(requests[1]["truth"]), 0)
            self.assertEqual(Path(requests[1]["score_midi"]).name, "clean_score.mid")

    def test_model_code_and_weight_provenance_do_not_invalidate_other_methods(self):
        job = dict(
            method="PolyTune",
            packages={"PolyTune": {"sha": "a"}, "LadderSym": {"sha": "b"}},
            code={"x/PolyTune.py": "a", "x/LadderSym.py": "b", "shared.py": "c"},
        )
        before = MistakeCache.scoped_job(job)
        job["packages"]["LadderSym"]["sha"] = "changed"
        job["code"]["x/LadderSym.py"] = "changed"
        self.assertEqual(before, MistakeCache.scoped_job(job))
        job["method"] = "LadderSym"
        scoped = MistakeCache.scoped_job(job)
        self.assertNotIn("PolyTune", scoped["packages"])
        self.assertNotIn("x/PolyTune.py", scoped["code"])
        self.assertIn("x/LadderSym.py", scoped["code"])
        self.assertEqual(MistakeCache.units(["LadderSym"]), [("audio", "LadderSym")])
        self.assertNotIn("LadderSym", COMPETITOR_SYMBOLIC_METHODS)

    def test_multiple_audio_requests_checkpoint_in_both_schedules(self):
        from benchmarks.modules.mistake.MistakeBenchmarker import (
            COMPETITOR_AUDIO_METHODS,
            MistakeBenchmarker,
        )

        class Client:

            def __init__(self, config, directory):
                self.method = config

            def predict(self, audio, score, directory, **kwargs):
                if self.method == "LadderSym":
                    assert kwargs == {"score_midi": "clean.mid"}
                else:
                    assert kwargs == {}
                return dict(input="audio", method=self.method)

            def close(self):
                pass

        for staged in (True, False):
            with self.subTest(staged=staged), tempfile.TemporaryDirectory() as tmp:
                ready = Path(tmp) / "ready"
                ready.mkdir()
                tasks = [
                    dict(
                        key=i,
                        ready=str(ready),
                        jobs={
                            ("detected", "test"): ({}, ""),
                            **{
                                ("audio", m): (
                                    {"method": m},
                                    str(Path(tmp) / f"{i}-{m}.json"),
                                )
                                for m in COMPETITOR_AUDIO_METHODS
                            },
                        },
                    )
                    for i in range(2)
                ]
                results = []
                with patch(
                    "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker._case_worker",
                    multi_audio_case,
                ), patch(
                    "benchmarks.modules.mistake.competitors.PolyTune.PolyTune.Client",
                    Client,
                ), patch(
                    "benchmarks.modules.mistake.competitors.LadderSym.LadderSym.Client",
                    Client,
                ), patch.object(
                    MistakeBenchmarker,
                    "polytune_rows",
                    side_effect=lambda task, payload: [payload],
                ):
                    MistakeBenchmarker.run_parallel(
                        tasks,
                        output=tmp,
                        polytune="PolyTune",
                        laddersym="LadderSym",
                        workers=2,
                        neural_workers=1,
                        polytune_last=staged,
                        on_result=lambda key, batch: results.extend(batch),
                    )
                self.assertEqual(len(list(Path(tmp).glob("*.json"))), 4)
                self.assertEqual(sum((r["method"] == "LadderSym" for r in results)), 2)
                self.assertEqual(sum((r["method"] == "PolyTune" for r in results)), 2)

    def test_pool_switches_models_with_one_live_client_and_forwards_prompt(self):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        clients = []

        class FakeClient:

            def __init__(self, config, directory):
                assert all(
                    (c.closed for c in clients)
                ), "Two models resident in one slot"
                self.closed = False
                self.calls = []
                clients.append(self)

            def predict(self, audio, score, directory, **kwargs):
                self.calls.append(kwargs)
                return {}

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmarks.modules.mistake.competitors.PolyTune.PolyTune.Client",
            FakeClient,
        ), patch(
            "benchmarks.modules.mistake.competitors.LadderSym.LadderSym.Client",
            FakeClient,
        ), patch.object(
            MistakeBenchmarker, "polytune_rows", return_value=[{"ok": True}]
        ):
            pool = MistakeBenchmarker.NeuralPool(None, tmp, 1)
            try:
                for i, method in enumerate(["PolyTune", "LadderSym", "LadderSym"]):
                    task = dict(
                        audio="played.wav",
                        score_audio="score.wav",
                        score_midi="clean.mid",
                        directory=tmp,
                        common={"method": method},
                    )
                    pool.submit(task, {}, Path(tmp) / f"{i}.json").result()
            finally:
                pool.close()
        self.assertEqual(len(clients), 2)
        self.assertEqual(clients[0].calls, [{}])
        self.assertEqual(clients[1].calls, [{"score_midi": "clean.mid"}] * 2)
        self.assertTrue(all((c.closed for c in clients)))


if __name__ == "__main__":
    unittest.main()
