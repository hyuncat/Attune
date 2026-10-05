from __future__ import annotations
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
import pandas as pd
from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
from benchmarks.modules.pitch.tests.PitchMetricsTest import counted_rows
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchNotebook import PitchNotebook
from benchmarks.modules.pitch.datasets.PitchDataset import PitchTrack


class PitchNotebookTest(unittest.TestCase):
    METHODS = ("old_a", "changed", "old_b")

    @staticmethod
    def rows(values: dict[str, float]) -> pd.DataFrame:
        return counted_rows(
            pd.DataFrame(
                [
                    {
                        "model": method,
                        "dataset": "dataset",
                        "track_id": "track",
                        "Raw Pitch Accuracy": value,
                    }
                    for method, value in values.items()
                ]
            )
        )

    def notebook(self, root: Path) -> PitchNotebook:
        notebook = PitchNotebook(
            PitchNotebook.NotebookConfig(workers=1, methods=self.METHODS)
        )
        notebook.runs_root = root
        return notebook

    def test_selective_force_replaces_only_requested_methods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "suite" / "rows.csv"
            path.parent.mkdir(parents=True)
            self.rows({"old_a": 0.1, "changed": 0.2, "old_b": 0.3}).to_csv(
                path, index=False
            )
            notebook = self.notebook(root)
            fresh = self.rows({"changed": 0.9})
            with patch.object(
                notebook, "_run_methods", return_value=fresh
            ) as run, patch("benchmarks.modules.pitch.PitchNotebook.display"):
                result = notebook.run_suite(
                    "suite",
                    datasets=("dataset",),
                    max_tracks=None,
                    force=True,
                    methods={"changed"},
                )
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], ["changed"])
            self.assertTrue(run.call_args.kwargs["force_reanalysis"])
            actual = result.rows.set_index("model")["Raw Pitch Accuracy"]
            self.assertEqual(
                actual.to_dict(), {"old_a": 0.1, "changed": 0.9, "old_b": 0.3}
            )
            self.assertEqual(list(result.summary.index), list(self.METHODS))
            saved = pd.read_csv(path).set_index("model")["Raw Pitch Accuracy"]
            self.assertEqual(saved.to_dict(), actual.to_dict())

    def test_selective_force_never_backfills_unrequested_methods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook = self.notebook(Path(directory))
            fresh = self.rows({"changed": 0.9})
            with patch.object(
                notebook, "_run_methods", return_value=fresh
            ) as run, patch("benchmarks.modules.pitch.PitchNotebook.display"):
                result = notebook.run_suite(
                    "suite",
                    datasets=("dataset",),
                    max_tracks=None,
                    force=True,
                    methods={"changed"},
                )
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], ["changed"])
            self.assertEqual(result.rows["model"].tolist(), ["changed"])

    def test_selective_force_drops_retired_method_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "suite" / "rows.csv"
            path.parent.mkdir(parents=True)
            self.rows({"old_a": 0.1, "retired": 0.2}).to_csv(path, index=False)
            notebook = self.notebook(root)
            with patch.object(
                notebook, "_run_methods", return_value=self.rows({"changed": 0.9})
            ), patch("benchmarks.modules.pitch.PitchNotebook.display"):
                result = notebook.run_suite(
                    "suite",
                    datasets=("dataset",),
                    max_tracks=None,
                    force=True,
                    methods={"changed"},
                )
            self.assertEqual(set(result.rows["model"]), {"old_a", "changed"})

    def test_omitted_methods_still_forces_every_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook = self.notebook(Path(directory))
            fresh = self.rows({method: 1.0 for method in self.METHODS})
            with patch.object(
                notebook, "_run_methods", return_value=fresh
            ) as run, patch("benchmarks.modules.pitch.PitchNotebook.display"):
                notebook.run_suite(
                    "suite", datasets=("dataset",), max_tracks=None, force=True
                )
            self.assertEqual(run.call_args.args[0], list(self.METHODS))
            self.assertTrue(run.call_args.kwargs["force_reanalysis"])

    def test_methods_requires_force_and_valid_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook = self.notebook(Path(directory))
            with self.assertRaisesRegex(ValueError, "force=True"):
                notebook.run_suite(
                    "suite", datasets=("dataset",), max_tracks=None, methods={"changed"}
                )
            with self.assertRaisesRegex(ValueError, "unknown or disabled"):
                notebook.run_suite(
                    "suite",
                    datasets=("dataset",),
                    max_tracks=None,
                    force=True,
                    methods={"missing"},
                )
            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                notebook.run_suite(
                    "suite",
                    datasets=("dataset",),
                    max_tracks=None,
                    force=True,
                    methods=set(),
                )

    def test_streaming_batches_tracks_by_method(self) -> None:
        runner = PitchBenchmarker.PitchStreaming(
            PitchBenchmarker.Options(), PitchBenchmarker.StreamingConfig(workers=1)
        )
        examples = [SimpleNamespace(track_id=name) for name in ("a", "b")]
        calls = []
        progress_indices = []

        def run_example(example, methods, rolling):
            calls.append((methods[0], example.track_id))
            return [
                {"dataset": "test", "track_id": example.track_id, "model": methods[0]}
            ]

        with patch.object(
            runner, "_prepare_detectors", return_value={}
        ) as prepare, patch.object(
            runner, "_run_example", side_effect=run_example
        ), patch.object(
            PitchBenchmarker.Progress,
            "update_compact",
            autospec=True,
            side_effect=lambda renderer, total, method, detail, count: progress_indices.append(
                (renderer.completed, total, detail)
            ),
        ):
            runner.run(examples, methods=("attune", "spice"))
        self.assertEqual(
            [
                (i, total)
                for i, total, detail in progress_indices
                if "| finished:" in detail
            ],
            [(2, 4), (3, 4), (4, 4), (4, 4)],
        )
        self.assertEqual(
            calls, [("attune", "a"), ("attune", "b"), ("spice", "a"), ("spice", "b")]
        )
        self.assertEqual(
            [call.args[0] for call in prepare.call_args_list], [["attune"], ["spice"]]
        )

    def test_streaming_resumes_completed_track_after_interrupt(self):
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"test")
            examples = [
                PitchDetectorBase.PitchExample(
                    track_id=name,
                    dataset="test",
                    audio_path=audio,
                    ref_times=np.array([0.0]),
                    ref_freqs=np.array([440.0]),
                    fmin=200.0,
                    fmax=600.0,
                    estimate_cache_dir=root,
                    stage_cache_path=root / "stage",
                )
                for name in ("a", "b")
            ]
            runner = PitchBenchmarker.PitchStreaming(
                PitchBenchmarker.Options(), PitchBenchmarker.StreamingConfig(workers=1)
            )
            first = [
                {
                    "model": "attune",
                    "dataset": "test",
                    "track_id": "a",
                    "_latency_samples": {"output_latency_ms": np.array([47.0])},
                }
            ]
            with patch.object(
                runner, "_prepare_detectors", return_value={}
            ), patch.object(
                runner, "_run_example", side_effect=[first, KeyboardInterrupt]
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run(
                        examples, methods=["attune"], checkpoint_dir=root / "resume"
                    )
            self.assertEqual(len(list((root / "resume").rglob("*.json"))), 1)
            second = [
                {
                    "model": "attune",
                    "dataset": "test",
                    "track_id": "b",
                    "_latency_samples": {"output_latency_ms": np.array([48.0])},
                }
            ]
            with patch.object(
                runner, "_prepare_detectors", return_value={}
            ), patch.object(runner, "_run_example", return_value=second) as run:
                rows = runner.run(
                    examples, methods=["attune"], checkpoint_dir=root / "resume"
                )
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0].track_id, "b")
            self.assertEqual(set(rows.track_id), {"a", "b"})
            with patch.object(
                runner, "_prepare_detectors", return_value={}
            ), patch.object(runner, "_run_example", side_effect=[first, second]) as run:
                runner.run(
                    examples,
                    methods=["attune"],
                    checkpoint_dir=root / "resume",
                    force=True,
                )
            self.assertEqual(run.call_count, 2)

    def test_streaming_runner_executes_only_selected_method(self) -> None:
        runner = PitchBenchmarker.PitchStreaming(PitchBenchmarker.Options())
        example = SimpleNamespace(track_id="track")
        with patch.object(
            runner, "_load_rolling_detectors", return_value={}
        ), patch.object(
            runner,
            "_run_matched_example",
            return_value=[{"model": "attune", "track_id": "track", "dataset": "test"}],
        ) as matched:
            rows = runner.run([example], methods={"attune"})
        matched.assert_called_once_with(example, ["attune"])
        self.assertEqual(rows["model"].tolist(), ["attune"])

    def test_worker_payload_survives_reloaded_nested_dataclasses(self) -> None:
        options = PitchBenchmarker.Options(datasets=("dataset",))
        job = PitchBenchmarker.Job(
            method="attune",
            first_index=1,
            tracks=(
                PitchTrack(
                    track_id="track",
                    dataset="dataset",
                    audio_path=Path("audio.wav"),
                    annot_path=Path("pitch.csv"),
                    metadata={"instrument": "violin"},
                ),
            ),
        )
        options_payload = PitchBenchmarker._worker_options_payload(options)
        job_payload = PitchBenchmarker._worker_job_payload(job)
        with patch.object(PitchBenchmarker, "Options", object):
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(options)
        pickle.dumps((options_payload, job_payload))
        restored = PitchBenchmarker._worker_job_from_payload(job_payload)
        self.assertEqual(restored.method, "attune")
        self.assertEqual(restored.tracks[0].track_id, "track")
        self.assertEqual(restored.tracks[0].audio_path, Path("audio.wav"))

    def test_streaming_runner_labels_frame_restarted_librosa_separately(self) -> None:
        runner = PitchBenchmarker.PitchStreaming(PitchBenchmarker.Options())
        example = SimpleNamespace(track_id="track")
        with patch.object(
            runner, "_load_rolling_detectors", return_value={}
        ), patch.object(
            runner,
            "_run_pyin_framewise",
            return_value={
                "model": "pyin_framewise",
                "track_id": "track",
                "dataset": "test",
            },
        ) as framewise:
            rows = runner.run([example], methods={"pyin_framewise"})
        framewise.assert_called_once_with(example)
        self.assertEqual(rows["model"].tolist(), ["pyin_framewise"])

    def test_attune_is_one_benchmark_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook = PitchNotebook(
                PitchNotebook.NotebookConfig(workers=1, methods=("attune",))
            )
            notebook.runs_root = Path(directory)
            fresh = self.rows({"attune": 0.9})
            with patch.object(
                notebook, "_run_methods", return_value=fresh
            ) as run, patch("benchmarks.modules.pitch.PitchNotebook.display"):
                result = notebook.run_suite(
                    "suite",
                    datasets=("dataset",),
                    max_tracks=None,
                    force=True,
                    methods={"attune"},
                )
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], ["attune"])
            self.assertTrue(run.call_args.kwargs["force_reanalysis"])
            self.assertEqual(result.rows["model"].tolist(), ["attune"])

    def test_stale_attune_rows_refresh_without_force(self) -> None:
        methods = ("pyin", "attune", "praat")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "suite" / "rows.csv"
            path.parent.mkdir(parents=True)
            old = self.rows({"pyin": 0.8, "attune": 0.1, "praat": 0.9})
            old.to_csv(path, index=False)
            notebook = PitchNotebook(
                PitchNotebook.NotebookConfig(workers=1, methods=methods)
            )
            notebook.runs_root = root
            fresh = self.rows({"attune": 0.95}).assign(
                pitch_cache_version=PitchCache.VERSION
            )
            with patch.object(
                notebook, "_run_methods", return_value=fresh
            ) as run, patch("benchmarks.modules.pitch.PitchNotebook.display"):
                result = notebook.run_suite(
                    "suite", datasets=("dataset",), max_tracks=None
                )
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], ["attune"])
            self.assertFalse(run.call_args.kwargs["force_reanalysis"])
            actual = result.rows.set_index("model")["Raw Pitch Accuracy"]
            self.assertEqual(
                actual.to_dict(), {"pyin": 0.8, "attune": 0.95, "praat": 0.9}
            )

    def test_local_pyin_and_attune_run_as_independent_rows(self) -> None:
        methods = ("pyin", "attune", "praat")
        with tempfile.TemporaryDirectory() as directory:
            notebook = PitchNotebook(
                PitchNotebook.NotebookConfig(workers=1, methods=methods)
            )
            notebook.runs_root = Path(directory)
            fresh = self.rows({"pyin": 0.8, "attune": 0.89, "praat": 0.9})
            with patch.object(
                notebook, "_run_methods", return_value=fresh
            ) as run, patch("benchmarks.modules.pitch.PitchNotebook.display"):
                result = notebook.run_suite(
                    "suite",
                    datasets=("dataset",),
                    max_tracks=None,
                    force=True,
                    methods=methods,
                )
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], list(methods))
            self.assertTrue(run.call_args.kwargs["force_reanalysis"])
            self.assertEqual(set(result.rows["model"]), set(methods))

    def test_method_comparison_ranks_overall_accuracy_within_suite(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "model": method,
                    "track_id": track,
                    "Overall Accuracy": accuracy,
                    "Voicing False Alarm": false_alarm,
                    "Voicing Recall": 0.9,
                    "Raw Pitch Accuracy": accuracy,
                    "Raw Chroma Accuracy": accuracy,
                }
                for method, accuracy, false_alarm in (
                    ("pyin", 0.8, 0.1),
                    ("attune", 0.9, 0.05),
                )
                for track in ("one", "two")
            ]
        )
        result = SimpleNamespace(name="suite", rows=counted_rows(rows))
        with patch("benchmarks.modules.pitch.PitchNotebook.display"):
            comparison = PitchNotebook().show_method_comparison(
                result, methods=("pyin", "attune")
            )
        ranked_methods = comparison.reset_index().sort_values("OA Rank")["Method"]
        self.assertEqual(ranked_methods.iloc[0], "attune")

    def test_rpa_decomposition_separates_voicing_and_pitch_losses(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "model": "pyin",
                    "track_id": "track",
                    "Raw Pitch Accuracy": 0.9,
                    "Voicing Recall": 0.95,
                },
                {
                    "model": "attune",
                    "track_id": "track",
                    "Raw Pitch Accuracy": 0.85,
                    "Voicing Recall": 0.9,
                },
            ]
        )
        result = SimpleNamespace(name="suite", rows=counted_rows(rows))
        with patch("benchmarks.modules.pitch.PitchNotebook.display"):
            table = PitchNotebook().show_rpa_decomposition(
                result, methods=("pyin", "attune")
            )
        pyin = table.loc["suite", "pyin"]
        self.assertAlmostEqual(pyin["Missed-voicing share"], 0.05)
        self.assertAlmostEqual(pyin["Wrong-pitch share"], 0.05)
        self.assertAlmostEqual(pyin["RPA / Voicing Recall"], 0.9 / 0.95)

    def test_default_rerun_reuses_completed_offline_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            notebook = self.notebook(Path(directory))
            path = Path(directory) / "suite" / "rows.csv"
            path.parent.mkdir()
            self.rows({"old_a": 0.7, "changed": 0.8, "old_b": 0.9}).to_csv(
                path, index=False
            )
            with patch.object(notebook, "_run_methods") as run, patch(
                "benchmarks.modules.pitch.PitchNotebook.display"
            ):
                for _ in range(2):
                    result = notebook.run_suite(
                        "suite", datasets=("dataset",), max_tracks=None
                    )
            run.assert_not_called()
            self.assertEqual(len(result.rows), 3)

    def test_default_streaming_rerun_reuses_completed_rows(self):
        notebook = self.notebook(Path("/unused"))
        rows = self.rows(
            {method: 0.9 for method in PitchBenchmarker.PitchStreaming.METHODS}
        ).assign(
            frame_size_samples=4096,
            integration_size_samples=2048,
            hop_size_samples=128,
            sample_rate_hz=44100,
            instrument="violin",
        )
        with patch.object(notebook, "_read_run", return_value=rows), patch(
            "benchmarks.modules.pitch.PitchCache.PitchCache.has_latency_samples",
            return_value=True,
        ), patch(
            "benchmarks.modules.pitch.PitchBenchmarker.PitchBenchmarker.PitchStreaming.run"
        ) as run, patch(
            "benchmarks.modules.pitch.PitchNotebook.display"
        ):
            for _ in range(2):
                result = notebook.run_streaming_comparison(
                    instruments=(), tracks_per_instrument=2
                )
        run.assert_not_called()
        self.assertEqual(len(result.rows), len(PitchBenchmarker.PitchStreaming.METHODS))

    def test_streaming_score_refresh_does_not_rerun_current_peers(self):
        with tempfile.TemporaryDirectory() as directory:
            notebook = self.notebook(Path(directory))
            rows = self.rows(
                {method: 0.9 for method in PitchBenchmarker.PitchStreaming.METHODS}
            ).assign(
                frame_size_samples=4096,
                integration_size_samples=2048,
                hop_size_samples=128,
                sample_rate_hz=44100,
                instrument="violin",
            )
            fresh = rows.loc[rows.model.eq("praat")].copy()
            rows.loc[rows.model.eq("praat"), "scoring_version"] = "old"
            with patch.object(notebook, "_read_run", return_value=rows), patch(
                "benchmarks.modules.pitch.PitchCache.PitchCache.has_latency_samples",
                return_value=True,
            ), patch(
                "benchmarks.modules.pitch.PitchBenchmarker.PitchBenchmarker.tracks",
                return_value=[],
            ), patch(
                "benchmarks.modules.pitch.PitchBenchmarker.PitchBenchmarker.PitchStreaming.run",
                return_value=fresh,
            ) as run, patch(
                "benchmarks.modules.pitch.PitchNotebook.display"
            ):
                result = notebook.run_streaming_comparison(
                    instruments=(), tracks_per_instrument=2
                )
            run.assert_called_once()
            self.assertEqual(run.call_args.kwargs["methods"], ["praat"])
            self.assertFalse(run.call_args.kwargs["force"])
            self.assertEqual(
                len(result.rows), len(PitchBenchmarker.PitchStreaming.METHODS)
            )

    def test_missing_latency_samples_retriggers_only_affected_method(self):
        with tempfile.TemporaryDirectory() as directory:
            notebook = self.notebook(Path(directory))
            rows = self.rows(
                {method: 0.9 for method in PitchBenchmarker.PitchStreaming.METHODS}
            ).assign(
                frame_size_samples=4096,
                integration_size_samples=2048,
                hop_size_samples=128,
                sample_rate_hz=44100,
                instrument="violin",
            )
            fresh = rows.loc[rows.model.eq("praat")].copy()
            with patch.object(notebook, "_read_run", return_value=rows), patch(
                "benchmarks.modules.pitch.PitchCache.PitchCache.has_latency_samples",
                side_effect=lambda row: row["model"] != "praat",
            ), patch(
                "benchmarks.modules.pitch.PitchBenchmarker.PitchBenchmarker.tracks",
                return_value=[],
            ), patch(
                "benchmarks.modules.pitch.PitchBenchmarker.PitchBenchmarker.PitchStreaming.run",
                return_value=fresh,
            ) as run, patch(
                "benchmarks.modules.pitch.PitchNotebook.display"
            ):
                result = notebook.run_streaming_comparison(
                    instruments=(), tracks_per_instrument=2
                )
            run.assert_called_once()
            self.assertEqual(run.call_args.kwargs["methods"], ["praat"])
            self.assertFalse(run.call_args.kwargs["force"])
            self.assertEqual(
                len(result.rows), len(PitchBenchmarker.PitchStreaming.METHODS)
            )

    def test_score_migration_reuses_estimates_for_unforced_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            notebook = self.notebook(Path(directory))
            path = Path(directory) / "suite" / "rows.csv"
            path.parent.mkdir()
            old = self.rows({"old_a": 0.5, "changed": 0.6, "old_b": 0.7})
            old.drop(columns="scoring_version").to_csv(path, index=False)

            def run(methods, *args, **kwargs):
                return self.rows({method: 0.9 for method in methods})

            with patch.object(
                notebook, "_run_methods", side_effect=run
            ) as runner, patch("benchmarks.modules.pitch.PitchNotebook.display"):
                result = notebook.run_suite(
                    "suite",
                    datasets=("dataset",),
                    max_tracks=None,
                    force=True,
                    methods=("changed",),
                )
            calls = runner.call_args_list
            self.assertEqual(calls[0].args[0], ["old_a", "old_b"])
            self.assertFalse(calls[0].kwargs["force_reanalysis"])
            self.assertEqual(calls[1].args[0], ["changed"])
            self.assertTrue(calls[1].kwargs["force_reanalysis"])
            self.assertEqual(len(result.rows), 3)

    def test_paired_analysis_exports_both_modes_and_pooled_coverage(self):
        from benchmarks.modules.pitch.tests.PitchMetricsTest import PitchMetricsTest

        rows = PitchMetricsTest.paired_rows()
        with tempfile.TemporaryDirectory() as directory:
            notebook = self.notebook(Path(directory))
            for mode in ("Offline", "Streaming"):
                result = SimpleNamespace(
                    name="suite",
                    path=Path(directory) / "rows.csv",
                    rows=rows.assign(execution_mode=mode),
                )
                with patch("benchmarks.modules.pitch.PitchNotebook.display"):
                    tests = notebook.show_paired_analysis(
                        result, mode=mode, methods=("attune", "peer"), n_resamples=99
                    )
                self.assertEqual(len(tests), 1 if mode == "Offline" else 3)
                folders = list(
                    (Path(directory) / "paired_analysis" / mode.lower()).iterdir()
                )
                self.assertEqual(len(folders), 1)
                coverage = pd.read_csv(folders[0] / "coverage.csv")
                self.assertEqual(coverage.iloc[0]["tracks"], 8)
                self.assertEqual(coverage.iloc[0]["frames"], 800)
                self.assertTrue((folders[0] / "pooled_metrics.csv").is_file())

    def test_notebook_uses_single_local_pyin_method(self) -> None:
        notebook_path = (
            Path(__file__).resolve().parents[3] / "notebooks" / "pitch.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            ("".join(cell.get("source", [])) for cell in notebook["cells"])
        )
        self.assertIn("**`attune`**", source)
        self.assertNotIn("attune_realtime", source)
        self.assertNotIn("attune_adhoc", source)
        self.assertNotIn("attune2_realtime", source)
        self.assertNotIn("attune2_adhoc", source)
        self.assertIn("PRIMARY_METHODS =", source)
        self.assertIn("cache_methods=CACHE_METHODS", source)
        self.assertIn("methods=PRIMARY_METHODS", source)
        self.assertNotIn("BENCH.TEMPORARY_ABLATION_METHODS", source)
        self.assertFalse(hasattr(PitchNotebook, "PRIMARY_COMPARISON_METHODS"))
        self.assertFalse(hasattr(PitchNotebook, "TEMPORARY_ABLATION_METHODS"))
        self.assertNotIn("pyin_smoothed", source)
        self.assertNotIn("pyin_decoupled_voicing", source)
        self.assertNotIn("pyin_praat_inspired_voicing", source)


if __name__ == "__main__":
    unittest.main()
