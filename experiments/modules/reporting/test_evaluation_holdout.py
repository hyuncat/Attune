"""Regression checks for the shared pitch/note evaluation exclusions."""
import unittest
import tempfile
from pathlib import Path

import pandas as pd

from experiments.modules.reporting.evaluation_holdout import Holdout


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.holdout = Holdout()

    def test_exact_tuning_recordings_are_excluded_in_both_paths(self):
        for item in self.holdout.items:
            if item['dataset'] not in self.holdout.track_ids:
                continue
            track = dict(dataset=item['dataset'], track_id=item['source_track_id'],
                         group=item['group'])
            with self.subTest(track=track):
                self.assertEqual(self.holdout.tracks([track]), [])
                self.assertTrue(self.holdout.filter(pd.DataFrame([track])).empty)

    def test_other_stems_of_tuning_pieces_remain(self):
        tracks = [
            dict(dataset='bach10-original', group='02-AchLiebenChristen',
                 track_id='02-AchLiebenChristen-violin'),
            dict(dataset='urmp', group='04_Allegro_fl_fl',
                 track_id='04_Allegro_fl_fl/AuSep_2_fl_04_Allegro'),
        ]
        self.assertEqual(self.holdout.tracks(tracks), tracks)
        self.assertEqual(len(self.holdout.filter(pd.DataFrame(tracks))), 2)

    def test_coco_remains_unchanged(self):
        tracks = [dict(dataset=i['dataset'], track_id=i.get('source_track_id', i['track_id']),
                       group=i['group']) for i in self.holdout.items
                  if i['dataset'] == 'coco']
        self.assertTrue(tracks)
        self.assertEqual(self.holdout.tracks(tracks), tracks)
        self.assertEqual(len(self.holdout.filter(pd.DataFrame(tracks))), len(tracks))

    def test_streaming_reuses_only_matching_fingerprints_across_selections(self):
        from experiments.modules.reporting.evaluation_holdout import seed_streaming_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'evaluation_holdout'
            suffix = Path('pitch/native_feedback_urmp/production_defaults/checkpoints/attune')
            old = root / 'old-selection' / suffix / 'matching.json'
            old.parent.mkdir(parents=True)
            old.write_text('[]')
            stale = old.with_name('different-fingerprint.json')
            stale.write_text('[]')
            target = root / 'new-selection' / suffix / 'matching.json'
            seed_streaming_checkpoint(target)
            self.assertEqual(target.read_text(), old.read_text())
            missing = target.with_name('missing-fingerprint.json')
            seed_streaming_checkpoint(missing)
            self.assertFalse(missing.exists())
            self.assertFalse(target.with_name(stale.name).exists())

    def test_frozen_manifest_counts(self):
        self.assertEqual({d: len(ids) for d, ids in self.holdout.track_ids.items()},
                         {'bach10-original': 4, 'urmp': 13})


if __name__ == '__main__':
    unittest.main()
