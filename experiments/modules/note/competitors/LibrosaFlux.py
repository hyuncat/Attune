"""Audio-only spectral-flux onset baseline with a shared pYIN note adapter."""
from __future__ import annotations

import numpy as np
from experiments.modules.note.NoteDetectorBase import NoteDetectorBase


class LibrosaFlux(NoteDetectorBase):
    superflux = False

    @classmethod
    def onset_times(cls, y, sr):
        import librosa

        hop = int(sr / 200)
        if cls.superflux:
            spectrum = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=1024, hop_length=hop,
                n_mels=138, fmin=27.5, fmax=min(16000., sr / 2),
            )
            envelope = librosa.onset.onset_strength(
                S=librosa.power_to_db(spectrum, ref=np.max), sr=sr,
                hop_length=hop, n_fft=1024, lag=2, max_size=3,
            )
        else:
            envelope = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=hop, n_fft=2048, lag=1, max_size=1,
            )
        return librosa.onset.onset_detect(
            onset_envelope=envelope, sr=sr, hop_length=hop,
            units="time", backtrack=False,
        )

    @staticmethod
    def decode(onsets, f0, voiced, frame_dt, duration, min_duration=0.03):
        """Split voiced runs at acoustic onsets; never infer pitch-change cuts.

        Voiced-run starts cover soft attacks missed by peak picking. Offsets end
        at the next onset or voicing loss. Pitch is median log-frequency within
        each segment. This adapter is identical for both onset detectors.
        """
        valid = np.asarray(voiced, bool) & np.isfinite(f0) & (np.asarray(f0) > 0)
        edges = np.diff(np.r_[False, valid, False].astype(int))
        intervals, frequencies = [], []
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            lo, hi = start * frame_dt, min(end * frame_dt, duration)
            cuts = np.r_[lo, np.unique(onsets[(onsets > lo) & (onsets < hi)]), hi]
            for left, right in zip(cuts[:-1], cuts[1:]):
                if right - left < min_duration:
                    continue
                a, b = max(start, int(np.ceil(left / frame_dt))), min(end, int(np.ceil(right / frame_dt)))
                if b <= a:
                    continue
                intervals.append((left, right))
                frequencies.append(float(2 ** np.median(np.log2(f0[a:b]))))
        return np.asarray(intervals, float).reshape(-1, 2), np.asarray(frequencies, float)

    @classmethod
    def predict_audio(cls, y, sr):
        import librosa

        started = cls.cpu_seconds()
        pitch_sr, pitch_hop = 22050, 220
        pitch_audio = librosa.resample(y, orig_sr=sr, target_sr=pitch_sr)
        f0, voiced, _ = librosa.pyin(
            pitch_audio, sr=pitch_sr, fmin=32.7032, fmax=2093.005,
            frame_length=4096, hop_length=pitch_hop, fill_na=np.nan,
        )
        frontend = cls.cpu_seconds() - started
        started = cls.cpu_seconds()
        onsets = cls.onset_times(y, sr)
        iv, hz = cls.decode(onsets, f0, voiced, pitch_hop / pitch_sr, len(y) / sr)
        return iv, hz, dict(frontend_cpu_seconds=frontend,
                           segmentation_cpu_seconds=cls.cpu_seconds() - started)

    @classmethod
    def predict_task(cls, task, config):
        import librosa

        started = cls.cpu_seconds()
        y, sr = librosa.load(task['audio'], sr=44100, mono=True)
        loading = cls.cpu_seconds() - started
        iv, hz, timings = cls.predict_audio(y, sr)
        timings['frontend_cpu_seconds'] += loading
        return iv, hz, timings

    def detect(self, **_unused):
        import librosa
        from app_logic.NoteData import Note, NoteData

        audio = self.recording.audio_data
        y = np.asarray(audio.read_all(), dtype=float)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = librosa.resample(y, orig_sr=audio.sr, target_sr=44100)
        iv, hz, _ = self.predict_audio(y, 44100)
        notes = NoteData()
        origin = float(getattr(audio, 't_origin', 0.))
        for i, ((start, end), frequency) in enumerate(zip(iv, hz)):
            notes.write_note(Note(i=i, start_time=start + origin, end_time=end + origin,
                                  midi_num=[float(librosa.hz_to_midi(frequency))]))
        return notes
