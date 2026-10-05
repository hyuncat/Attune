"""Run with unittest in an environment with the API dependencies and httpx.

Uses generated inputs, never demo recordings or benchmark result directories.
"""
import tempfile
import unittest
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app_logic.JsonHandler import JsonHandler
from app_logic.midi.ScoreData import ScoreData
from app_logic.user.ds.Recording import Recording
from web.api.analyze_api import app


class PipelineParityTest(unittest.TestCase):
    def test_uploaded_audio_matches_desktop_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            score = Path(directory) / "score.mid"
            audio = Path(directory) / "audio.wav"
            midi = mido.MidiFile()
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.Message("program_change", program=40))
            for pitch in (60, 60, 64):
                track.append(mido.Message("note_on", note=pitch, velocity=90))
                track.append(mido.Message("note_off", note=pitch, time=480))
            midi.save(score)

            sr = 22050
            samples = np.zeros(int(1.8 * sr))
            for start, end, pitch in ((.15, .60, 60), (.65, 1.10, 60), (1.15, 1.60, 64)):
                i, j = int(start * sr), int(end * sr)
                t = np.arange(j - i) / sr
                frequency = 440 * 2 ** ((pitch - 69) / 12)
                envelope = np.minimum(1, t / .01) * np.minimum(1, (t[-1] - t) / .01)
                samples[i:j] = .2 * np.sin(2 * np.pi * frequency * t) * envelope
            sf.write(audio, samples, sr)

            with TestClient(app) as client:
                self.assertEqual(client.get("/health").json(), {"status": "ok"})
                upload = {"score": ("score.mid", score.read_bytes(), "audio/midi")}
                response = client.post("/notedata", files=upload)
                self.assertEqual(response.status_code, 200, response.text)
                score_payload = response.json()
                self.assertTrue(score_payload["measure_onsets_og"])
                self.assertTrue(score_payload["musicxml_b64"])

                response = client.post("/analyze", files={
                    **upload, "audio": ("audio.wav", audio.read_bytes(), "audio/wav"),
                })
                self.assertEqual(response.status_code, 200, response.text)
                actual = response.json()
                self.assertTrue(actual["note_data"])
                self.assertTrue(actual["alignment"]["pairs"])

                # Same fresh-take sequence used by desktop import and Perform.
                sd = ScoreData()
                sd.load(str(score))
                rec = Recording(score_data=sd)
                rec.load_audio(str(audio), score_filepath=str(score), load_cache=False)
                rec.detect_pitches()
                rec.smooth_pitches()
                rec.reset_analysis()
                rec.detect_notes()
                rec.align_score_and_refine()
                rec.update_alignment_distances()
                rec.trim_end()
                expected = JsonHandler(rec).to_cache_payload(score_filepath=str(score))
                for key in ("config", "note_analysis", "note_data", "pitch_data"):
                    self.assertEqual(actual[key], expected[key], key)
                self.assertEqual(actual["alignment"]["pairs"], expected["alignment"]["pairs"])
                self.assertEqual(actual["vibrato"], JsonHandler._vibrato_to_payload(rec.vibrato_data))

                channel = str(score_payload["active_instrument"])
                response = client.post("/realign", json={
                    "user_notes": actual["note_data"],
                    "score_notes": score_payload["note_data"][channel],
                    "pitch_tolerance": .5,
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(response.json()["pairs"])
                response = client.post("/notedata", files={
                    "score": ("invalid.txt", b"bad", "text/plain"),
                })
                self.assertEqual(response.status_code, 400)
