"""Check original Bach10 annotation units, timing, and isolated-stem selection."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from benchmarks.modules.pitch.datasets.Bach10 import Bach10


class Bach10Test(unittest.TestCase):
    def test_original_stems_and_reference_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            piece = root / "01-example"
            piece.mkdir()
            midi = np.array([[69.5, 0, 70], [60, 61, 0], [72, 0, 73], [48, 49, 50]])
            savemat(piece / "01-example-GTF0s.mat", {"GTF0s": midi})
            for suffix in Bach10.INSTRUMENTS:
                (piece / f"01-example-{suffix}.wav").touch()
            (piece / "01-example.wav").touch()  # ensemble must not be selected
            dataset = Bach10(root=root)
            tracks = dataset.tracks()
            self.assertEqual(len(tracks), 4)
            self.assertEqual({t.dataset for t in tracks}, {"bach10-original"})
            for row, track in enumerate(tracks):
                times, freqs = dataset.reference(track)
                np.testing.assert_allclose(times, [0.023, 0.033, 0.043])
                voiced = midi[row] > 0
                np.testing.assert_allclose(
                    freqs[voiced], 440 * 2 ** ((midi[row, voiced] - 69) / 12)
                )
                np.testing.assert_array_equal(freqs[~voiced], 0)
                self.assertEqual(dataset.cache_dir(track), root)
            sax = Bach10(root=root, instruments=("saxophone",)).tracks()
            self.assertEqual(len(sax), 1)
            self.assertEqual(sax[0].metadata["f0_row"], 2)

    def test_missing_originals_do_not_fall_back_to_synthesis(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "not used as a fallback"):
                Bach10(root=directory).tracks()


if __name__ == "__main__":
    unittest.main()
