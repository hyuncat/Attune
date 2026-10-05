"""Run All must bypass both result checkpoints and pitch caches."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.tests.CompetitorComparisonTest import notes
from benchmarks.modules.pitch.competitors.Attune import Attune
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
from app_logic.user.ds.Recording import Recording

class FreshPitchRunTests(unittest.TestCase):

    def test_force_reruns_extraction_once_per_case_despite_completed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.mid'
            MistakeBenchmarker.notedata_to_pm(notes([(i, i + 0.5, 60 + i) for i in range(4)])).write(str(source))

            def synth(self, midi, out_dir, **kwargs):
                path = Path(out_dir) / (Path(midi).stem + '.wav')
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.zeros(4 * 44100), 44100)
                return path

            def detect(rec):
                rec.note_data = NoteDetectorBase.clone_note_data(rec.score_data.note_data)
            with patch.object(MistakeBenchmarker, 'synth_midi', synth), patch.object(Attune, 'load_or_detect_pitches', return_value={'pitch_compute_time': 0.0}) as pitches, patch.object(Recording, 'detect_notes', detect):
                kwargs = dict(methods=('Attune (no refinement)', 'Attune (repeat only)'), rates=(0.0,), tolerances=(0.1,), input_kinds=('detected',), parallel=False)
                first = MistakeBenchmarker.run_comparison([source], Path(tmp) / 'results', force_pitch_detection=True, **kwargs)
                second = MistakeBenchmarker.run_comparison([source], Path(tmp) / 'results', force_pitch_detection=True, **kwargs)
                self.assertEqual(pitches.call_count, 2)
                self.assertTrue(all((c.kwargs['use_cache'] is False for c in pitches.call_args_list)))
                self.assertEqual(len(first), len(second))
                with patch.object(MistakeBenchmarker, 'evaluate_case', side_effect=AssertionError('Should resume')):
                    MistakeBenchmarker.run_comparison([source], Path(tmp) / 'results', force_pitch_detection=False, **kwargs)

    def test_worker_forwards_force_option(self):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
        with tempfile.TemporaryDirectory() as tmp, patch.object(MistakeBenchmarker, 'evaluate_case', return_value='done') as evaluate:
            result = MistakeBenchmarker._case_worker(dict(log=str(Path(tmp) / 'worker.log'), args=(), cached={}, source_info={}, force_pitch_detection=True))
            self.assertEqual(result, 'done')
            self.assertTrue(evaluate.call_args.kwargs['force_pitch_detection'])
if __name__ == '__main__':
    unittest.main()
