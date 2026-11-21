"""Practice must read the frame being overwritten while its clock is stalled."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from algorithms.Config import Config
from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch, PitchData
from practice import PracticeTab


class PracticeFrameAddressingTest(unittest.TestCase):
    def test_render_coalesces_progress_and_refreshes_stalled_pitch(self):
        tab = SimpleNamespace(
            is_recording=True, practice_time=1.25,
            _last_render=0., _RENDER_INTERVAL=1 / 30,
            score_data=Mock(), score_viewer=Mock(), guitar_hero=Mock(),
            note_panel=Mock(), _score_viewer_time=lambda t: t,
        )
        with patch('practice.time.monotonic', side_effect=[10., 10.017, 10.034]):
            PracticeTab._render_practice(tab)
            tab.guitar_hero.move_plot.assert_called_with(1.25, refresh_items=True)
            tab.practice_time = 1.267
            PracticeTab._render_practice(tab)
            tab.guitar_hero.move_plot.assert_called_with(1.267, refresh_items=False)
            # Stalled progress must still repaint a changed pitch in that slot.
            PracticeTab._render_practice(tab)
            tab.guitar_hero.move_plot.assert_called_with(1.267, refresh_items=True)
        self.assertEqual(tab.score_viewer.set_playback_time.call_count, 2)
        tab.is_recording = False
        PracticeTab._render_practice(tab)
        self.assertEqual(tab.guitar_hero.move_plot.call_count, 3)

    def test_pitch_events_gate_without_rendering(self):
        tab = SimpleNamespace(
            is_recording=True, practice_time=0.,
            recording=SimpleNamespace(pitch_detector=SimpleNamespace(block=False)),
            _pitch_matches=Mock(side_effect=[True, True, False]),
            _move_views=Mock(),
        )
        for t in (1., 1.003, 1.006):
            PracticeTab.pitch_detected(tab, t)
        self.assertEqual(tab.practice_time, 1.006)
        self.assertTrue(tab.recording.pitch_detector.block)
        tab._move_views.assert_not_called()

    def test_frame_roundtrip(self):
        for sr in (44100, 48000):
            for origin in (0.0, -0.5, 2.345):
                config = Config()
                config.sr = sr
                data = PitchData(config)
                data.t_origin = origin
                for i in range(2000):
                    t = origin + i * config.h1 / sr
                    pitch = Pitch(t, 1., 0., 0., config, [(69., 1.)])
                    data.write(pitch, t)
                    self.assertIs(data.read_pitch(t), pitch, (sr, origin, i))
                self.assertIsNone(data.read_pitch(origin - 0.001))

    def test_stalled_wrong_pitch_can_recover(self):
        config = Config()
        data = PitchData(config)
        notes = NoteData()
        notes.write_note(Note(0, 0., 10., [69]))
        messages = []
        tab = SimpleNamespace(
            score_data=SimpleNamespace(note_datas={0: notes}, active_instrument=0),
            recording=SimpleNamespace(config=config, pitch_data=data),
            status_bar=SimpleNamespace(update_status=messages.append),
        )
        # Choose a frame whose floating-point address floors to its predecessor.
        i = next(i for i in range(1, 2000)
                 if data.time_to_index(i * config.h1 / config.sr) != i)
        t = i * config.h1 / config.sr
        previous_t = (i - 1) * config.h1 / config.sr
        data.write(Pitch(previous_t, 1., 0., 0., config, [(67., 1.)]), previous_t)
        for _ in range(1000):
            data.write(Pitch(t, 1., 0., 0., config, [(67., 1.)]), t)
        self.assertFalse(PracticeTab._pitch_matches(tab, t))
        self.assertEqual(messages[-1], 'Off! Detected note: G4, Target note: A4')
        data.write(Pitch(t, 1., 0., 0., config, [(69., 1.)]), t)
        self.assertTrue(PracticeTab._pitch_matches(tab, t))
        self.assertEqual(messages[-1], 'On! Detected note: A4, Target note: A4')


if __name__ == '__main__':
    unittest.main()
