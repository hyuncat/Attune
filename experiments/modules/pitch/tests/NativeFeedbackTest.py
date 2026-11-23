"""Native geometry, cache isolation, and exact playback-duration timing."""
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.modules.pitch import FeedbackLatency as F
from experiments.modules.pitch.PitchBenchmarker import PitchBenchmarker as B
from experiments.modules.pitch.PitchCache import PitchCache
from experiments.modules.pitch.PitchDetectorBase import PitchDetectorBase
from experiments.modules.pitch.competitors.CrepeStreaming import CrepeStreaming
from experiments.modules.pitch.competitors.Praat import Praat


class NativeFeedbackTest(unittest.TestCase):
    def test_periodic_and_final_hold(self):
        a, w, startup = F.intervals([0, 1, 2], [100]*3, [1]*3, 3.1)
        self.assertAlmostEqual(startup, .1)
        self.assertAlmostEqual(F.percentile(a, w), 1050, places=6)
        a, w, _ = F.intervals([0], [100], [1], 2.1)
        self.assertAlmostEqual(F.percentile(a, w), 2000, places=6)

    def test_backlog_simultaneous_and_after_end(self):
        # Commits at .1, .4, .4, 4. Latest simultaneous output wins.
        a, w, _ = F.intervals([0, .1, .2, .3], [100, 300, 200, 3700], [1]*4, 1)
        np.testing.assert_allclose(a, [.1, .2])
        np.testing.assert_allclose(w, [.3, .6])
        # CDF: at x>.4 only the second interval is still contributing.
        self.assertAlmostEqual(F.percentile(a, w), 755, places=6)
        a, w, _ = F.intervals([0], [1000], [1], 1)
        self.assertEqual(len(w), 0)
        self.assertTrue(np.isnan(F.percentile(a, w)))

    def test_pool_seconds_not_recording_or_frame_counts(self):
        self.assertAlmostEqual(F.percentile([0, 10], [9, 1]), 10500, places=6)
        for times, latency, cpus in [([0, 1], [1], [1]), ([1, 0], [1, 1], [1, 1]),
                                     ([0, 1], [2000, 0], [1, 1]), ([0], [-1], [1]),
                                     ([np.nan], [1], [1])]:
            with self.assertRaises(ValueError):
                F.intervals(times, latency, cpus, 3)

    def test_hashes_unchanged_methods_and_worker_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            audio = root / 'a.wav'; audio.write_bytes(b'a')
            e = PitchDetectorBase.PitchExample(track_id='a', dataset='test', audio_path=audio,
                ref_times=np.array([0.]), ref_freqs=np.array([440.]), fmin=100, fmax=1000,
                estimate_cache_dir=root, stage_cache_path=root/'stage')
            old = B.PitchStreaming(B.Options(), B.StreamingConfig(workers=1))
            new = B.PitchStreaming(B.Options(), B.StreamingConfig(workers=2, native_feedback_policy=True))
            for m in (*old.METHODS, 'crepe_tiny'):
                a = old._checkpoint_path(root, m, e).name
                b = new._checkpoint_path(root, m, e).name
                self.assertEqual(a == b, m not in ('pyin_framewise', 'praat', 'crepe_tiny'))
                # Reconstruct the pre-change dataclass shape to prove byte-identical legacy hashes.
                legacy_config = asdict(old.config); legacy_config.pop('native_feedback_policy')
                with patch('experiments.modules.pitch.PitchCache.asdict', side_effect=lambda x: legacy_config.copy() if x is old.config else asdict(x)):
                    self.assertEqual(a, old._checkpoint_path(root, m, e).name)

    def test_native_praat_exact_single_frame_and_silence(self):
        import parselmouth
        for floor in (20., 40., 65.406, 440., 1000.):
            size, hop = Praat.FrameLocal.geometry(44100, floor, 2000)
            self.assertEqual(size, int(np.ceil(3 * 44100 / floor)))
            for audio in (np.zeros(size), np.sin(2*np.pi*1100*np.arange(size)/44100)):
                times, freqs = Praat.FrameLocal().predict(audio, 44100, floor, 2000)
                reference = parselmouth.Sound(audio, 44100).to_pitch_ac(
                    time_step=.75/floor, pitch_floor=floor, pitch_ceiling=2000)
                self.assertEqual(len(times), 1)
                np.testing.assert_array_equal(freqs, reference.selected_array['frequency'])
                self.assertLessEqual(times[0], size/44100)
                if not np.any(audio):
                    self.assertEqual(freqs[0], 0)

    def test_crepe_rolling_complete_windows_and_future_independence(self):
        runner = B.PitchStreaming(B.Options(), B.StreamingConfig(native_feedback_policy=True))
        detector = CrepeStreaming(rolling_10ms=True)
        size = detector.buffer_size(44100)
        self.assertEqual(size, 2823)
        audio = np.arange(size + 2*441 + 100, dtype=np.float32)
        seen = []
        detector.predict = lambda frame, *args: (seen.append(frame.copy()) or (np.array([size/88200]), np.array([440.])))
        e = SimpleNamespace(audio=lambda sr: (audio, sr), fmin=200, fmax=600)
        with patch.object(runner, '_row', return_value={}), patch.object(PitchDetectorBase, 'measure', side_effect=lambda f: (f(), .001, .001)):
            runner._run_rolling_example('crepe_tiny', detector, e)
            original = seen[0].copy()
            audio[size:] = -100
            runner._run_rolling_example('crepe_tiny', detector, e)
        self.assertEqual(len(seen), 6)
        np.testing.assert_array_equal(seen[0], seen[3])
        np.testing.assert_array_equal(seen[1], np.arange(441, 441+size))

    def test_pyin_quarter_hop_matches_one_frame_reference(self):
        import librosa
        from experiments.modules.pitch.competitors.PYIN import PYIN
        detector = PYIN()
        frame = np.sin(2*np.pi*440*np.arange(4096)/44100).astype(np.float32)
        f0, voiced, _ = librosa.pyin(frame, sr=44100, fmin=100, fmax=1000,
                                    frame_length=4096, hop_length=1024, center=False)
        actual = detector.predict_frame(frame, 44100, 100, 1000, 1024)
        self.assertAlmostEqual(actual, f0[0] if voiced[0] else 0)
        self.assertEqual(detector.DEFAULT_CONFIG.h1, 128)

    def test_selective_execution_never_runs_unchanged_and_resumes(self):
        import pandas as pd
        from experiments.modules.pitch import NativeFeedbackExperiment as E
        from experiments.modules.pitch.PitchNotebook import PitchNotebook
        with tempfile.TemporaryDirectory() as d:
            notebook = PitchNotebook(); notebook.runs_root = Path(d)
            runner = B.PitchStreaming(B.Options())
            examples = [SimpleNamespace(track_id='a')]
            row = dict(model='pyin_framewise', dataset='test', track_id='a')
            directory = Path(d)/'suite'
            def plan(blocked=False):
                return pd.DataFrame([dict(model='pyin_framewise', reusable_predictions=False,
                    reusable_timing_sidecars=False, reporting_only=False, inference_miss=True,
                    blocked=blocked, reason='test')])
            with patch.object(E, 'prepare', return_value=([(directory, runner, examples, {})], plan(True))), patch.object(runner, 'run', side_effect=AssertionError('inference')):
                with self.assertRaisesRegex(RuntimeError, 'Unchanged-method'):
                    E.run(notebook, dry_run=False)
            cached = {(m, 'a'): dict(row, model=m) for m in E.METHODS}
            partial = {key: value for key, value in cached.items() if key[0] != 'pyin_framewise'}
            with patch.object(E, 'prepare', return_value=([(directory, runner, examples, partial)], plan())), patch.object(runner, 'run', return_value=pd.DataFrame([row])) as inference, patch.object(E, 'report', return_value=pd.DataFrame()), patch.object(B, 'pool_scores', return_value=pd.DataFrame()):
                E.run(notebook, dry_run=False)
                inference.assert_called_once()
                self.assertEqual(inference.call_args.kwargs['methods'], ['pyin_framewise'])
                self.assertEqual(inference.call_args.args[0], examples)
            prepared = ([(directory, runner, examples, cached)], plan())
            with patch.object(E, 'prepare', return_value=prepared), patch.object(runner, 'run', side_effect=AssertionError('inference')), patch.object(E, 'report', return_value=pd.DataFrame()), patch.object(B, 'pool_scores', return_value=pd.DataFrame()):
                E.run(notebook, dry_run=False)
                E.run(notebook, dry_run=False)

    def test_report_exports_new_table_and_coverage(self):
        import pandas as pd
        from experiments.modules.pitch import NativeFeedbackExperiment as E
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = []
            for track, duration in [('a', 3.1), ('b', .1)]:
                source = root / (track + '.npz')
                np.savez(source, estimate_times_seconds=[0., 1., 2.],
                         output_latency_ms=[100., 100., 100.], call_cpu_ms=[1., 1., 1.])
                row = dict(model='attune', dataset='test', track_id=track,
                    latency_samples_path=str(source), latency_sample_count=3,
                    audio_seconds=duration, latency_clock=F.CLOCK)
                row.update(B.score_frames(np.array([0., .01]), np.array([440., 440.]),
                                          np.array([0., .01]), np.array([440., 440.])))
                row.update(dict(stream_adapter='test', frame_size_samples=4096, hop_size_samples=128,
                    sample_rate_hz=44100, update_interval_ms=2.9, algorithmic_lookahead_ms=46.4))
                rows.append(row)
            table = E.report([SimpleNamespace(rows=pd.DataFrame(rows))], root)
            self.assertAlmostEqual(table.loc['attune', 'P95 feedback latency (ms)'], 1050, places=6)
            self.assertEqual(table.loc['attune', 'timing_coverage_failures'], 1)
            self.assertTrue((root/'comparison.csv').exists())
            self.assertTrue((root/'manifest.json').exists())

    def test_saved_v9_timing_derived_without_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'v9.npz'
            values = np.array([100., 100., 100.])
            np.savez(p, estimate_times_seconds=[0.,1.,2.], output_latency_ms=values, call_cpu_ms=[1.,1.,1.])
            before = p.read_bytes()
            row = dict(latency_samples_path=str(p), latency_sample_count=3, audio_seconds=3.1, latency_clock=F.CLOCK)
            a,w,_ = F.read(row, Path(d)/'derived')
            self.assertAlmostEqual(F.percentile(a,w),1050,places=6)
            F.read(row, Path(d)/'derived')
            self.assertEqual(before,p.read_bytes())
            np.testing.assert_array_equal(PitchCache.read_latency_samples(row),values)

if __name__ == '__main__':
    unittest.main()
