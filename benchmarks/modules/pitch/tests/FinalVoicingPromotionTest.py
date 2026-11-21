"""Final-gate promotion must not tighten the HMM's tracking regions."""
from dataclasses import replace
import tempfile
import lzma
import pickle
from pathlib import Path
import unittest
import numpy as np
from algorithms.Config import Config
from algorithms.PitchSmoother import PitchSmoother, VoicingSmoother
from app_logic.user.ds.PitchData import Pitch
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.competitors.Attune import Attune


class FinalVoicingPromotionTest(unittest.TestCase):
    def test_final_gate_preserves_tracking_path_and_strict_boundary(self):
        config = Config(fmin=196, fmax=880)
        self.assertEqual(config.pitch_tracking_unv_thresh, .990)
        self.assertEqual(config.posthoc_unv_thresh, .9875)
        old = replace(config, posthoc_unv_thresh=.990)
        probabilities = [.2, .988, .9875, .9874, .2] * 5
        pitches = [Pitch(time=i*config.h1/config.sr, candidates=[(69., 1-u)],
                         value=-1, volume=.1, unvoiced_prob=u, config=config,
                         live_distance=None) for i,u in enumerate(probabilities)]
        new_tracked = PitchSmoother(config=config).smooth(pitches)
        old_tracked = PitchSmoother(config=old).smooth(pitches)
        np.testing.assert_array_equal([p.value for p in new_tracked], [p.value for p in old_tracked])
        new_mask = VoicingSmoother(config=config).decode(new_tracked)
        np.testing.assert_array_equal(new_mask, np.asarray(probabilities) < .9875)
        self.assertTrue(VoicingSmoother(config=old).decode(old_tracked).all())

    def test_all_production_adapters_get_separate_gates(self):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
        from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
        adapter = Attune()
        config = adapter.config_for(196,880)
        self.assertEqual(config.pitch_tracking_unv_thresh,.990)
        self.assertEqual(config.posthoc_unv_thresh,.9875)
        self.assertIn('pitch_tracking_unv_thresh',Config.PITCH_DETECTION_FIELDS)
        from app_logic.JsonHandler import JsonHandler
        self.assertEqual(JsonHandler.PITCH_ANALYSIS_VERSION,7)
        self.assertEqual(PitchCache.VERSION,15)

    def test_reference_pyin_reuses_v14_without_relaxing_attune(self):
        from types import SimpleNamespace
        from benchmarks.modules.pitch.competitors.PYIN import PYIN
        adapter = PYIN()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'track.pitch.pkl.xz'
            example = SimpleNamespace(stage_cache_path=base, fmin=196., fmax=880.)
            cache = adapter.cache(example)
            timing = {'compute_clock': adapter.COMPUTE_CLOCK, 'pitch_cache_version': 14}
            with lzma.open(cache.path, 'wb') as handle:
                pickle.dump({'version': 14,
                             'stages': {stage: {'pitches': []} for stage in ('raw', 'smoothed')},
                             'metadata': {stage: timing for stage in ('raw', 'smoothed')}}, handle)
            dataset = SimpleNamespace(pitch_cache_path=lambda track: base)
            self.assertTrue(adapter.has_cache(dataset, None))
            from unittest.mock import patch
            with patch.object(adapter, 'detect_stages', side_effect=AssertionError('pYIN must reuse predictions')):
                estimate = adapter.estimate(example)
            self.assertTrue(estimate.from_cache)
            self.assertIsNotNone(cache.read(PitchCache.SMOOTHED, adapter.config_for(196, 880)))
            self.assertFalse(PitchCache(cache.path).has(PitchCache.SMOOTHED))
            self.assertFalse(PitchCache(base).has(PitchCache.SMOOTHED))

    def test_v14_cannot_silently_supply_old_final_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pitch.pkl.xz'
            with lzma.open(path,'wb') as f:
                pickle.dump({'version':14,'stages':{'raw':{'pitches':[]},'smoothed':{'pitches':[]}}},f)
            for stage in (PitchCache.RAW,PitchCache.SMOOTHED):
                self.assertIsNone(PitchCache(path).read(stage,Config()))

if __name__=='__main__':
    unittest.main()
