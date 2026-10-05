"""Praat's raw-autocorrelation pitch tracker, via parselmouth."""

from __future__ import annotations
from app_logic.user.ds.PitchData import Pitch
from algorithms.Config import Config
from tqdm import tqdm
from scipy.signal import iirfilter, sosfilt
from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
from typing import TYPE_CHECKING
import time
import threading
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase


class Praat(PitchDetectorBase):

    @staticmethod
    def autocorrelation_pitch(audio, sr, time_step, floor, ceiling, *, parselmouth):
        """Shared raw Praat call; callers own preprocessing and output adaptation."""
        sound = parselmouth.Sound(np.ascontiguousarray(audio, dtype=np.float64), sr)
        return sound.to_pitch_ac(
            time_step=time_step, pitch_floor=floor, pitch_ceiling=ceiling
        )

    "Classic DSP baseline: ``Sound.to_pitch_ac`` at the track's own range."
    name = "praat"
    description = "Praat autocorrelation (praat-parselmouth)"
    install_hint = "pip install praat-parselmouth"
    input_sr = None

    def ensure_available(self) -> None:
        try:
            import parselmouth
        except Exception as exc:
            raise self.unavailable(exc) from exc

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        import parselmouth

        floor = max(float(fmin), 40.0)
        ceiling = max(float(fmax), floor + 1.0)
        pitch = Praat.autocorrelation_pitch(
            audio, sr, self.step_seconds, floor, ceiling, parselmouth=parselmouth
        )
        freqs = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
        return (np.asarray(pitch.xs(), dtype=np.float64), freqs)

    class Streaming(QObject):
        """Per-buffer Praat baseline with the app's PitchDetector interface.

        Retains the threaded and batch adapters for app-compatible comparisons.
        For the raw whole-track benchmark baseline, use Praat instead.

        Praat's autocorrelation tracker is already a temporally-smoothed f0 tracker,
        so this detector marks itself as not requiring the pYIN HMM smoother. The
        returned Pitch objects still live on Config.h1 / Config.sr so the existing
        PitchData, NoteDetector, GuitarHero, and trimming paths can read them without
        special cases.
        """

        pitch_detected = pyqtSignal(float)
        status_changed = pyqtSignal(str)
        detection_finished = pyqtSignal()
        requires_smoothing = False

        def __init__(
            self,
            recording: Recording | None = None,
            config: Config | None = None,
            parent: QObject | None = None,
        ):
            super().__init__(parent)
            if not recording and (not config):
                raise ValueError(
                    "Must provide either a recording or a config to initialize the PraatStreaming."
                )
            self.recording = recording
            self.config = config if config else recording.config
            self.pda_thread: threading.Thread | None = None
            self.offline_thread: threading.Thread | None = None
            self.stop_event = threading.Event()
            self.block = False
            self.load_config(self.config)

        def load_config(self, config: Config):
            self.config = config
            self.SR = config.sr
            self.FRAME_SIZE = config.w1
            self.HOP_SIZE = config.h1
            self.STEP_SECONDS = self.HOP_SIZE / self.SR
            padded_fmin = config.midi_to_freq(config.freq_to_midi(config.fmin) - 0.5)
            padded_fmax = config.midi_to_freq(config.freq_to_midi(config.fmax) + 0.5)
            self.PITCH_FLOOR = max(float(padded_fmin), 40.0)
            self.PITCH_CEILING = max(float(padded_fmax), self.PITCH_FLOOR + 1.0)

        def run(self, start_time: float | None = None):
            self.stop()
            self.stop_event.clear()
            self.recording.a2p_queue.init_start_time(start_time)
            self.pda_thread = threading.Thread(target=self._run, daemon=True)
            self.pda_thread.start()

        def _run(self) -> None:
            while not self.stop_event.is_set():
                try:
                    x, t = self.recording.a2p_queue.pop(
                        self.FRAME_SIZE, self.HOP_SIZE, stall=self.block
                    )
                    if x is None:
                        self.stop_event.wait(0.002)
                        continue
                    pitch = self.detect_pitch(x, t)
                    self.recording.write_pitch_data([pitch], t)
                    self.pitch_detected.emit(pitch.time)
                except Exception as exc:
                    print(f"[PraatStreaming] frame skipped due to error: {exc}")
                    continue

        def stop(self):
            if self.pda_thread and self.pda_thread.is_alive():
                self.stop_event.set()
                self.pda_thread.join()

        def detect_pitches_async(self):
            if self.offline_thread and self.offline_thread.is_alive():
                return
            self.offline_thread = threading.Thread(
                target=self._detect_pitches_offline, daemon=True
            )
            self.offline_thread.start()

        def _detect_pitches_offline(self):
            try:
                self.recording.detect_pitches(on_phase=self.status_changed.emit)
            except Exception as exc:
                print(f"[PraatStreaming] offline detection failed: {exc}")
            finally:
                self.detection_finished.emit()

        def detect_pitch(self, x: np.ndarray, start_time: float | None = None) -> Pitch:
            audio = self._mono_audio(x)
            volume = self._frame_volume(audio)
            if volume == 0.0:
                return self._unvoiced_pitch(start_time, volume=0.0)
            praat_audio = self._preprocess_praat_audio(audio)
            if not np.any(praat_audio):
                return self._unvoiced_pitch(start_time, volume=volume)
            _times, freqs, strengths = self._praat_track(
                praat_audio, time_step=self.STEP_SECONDS
            )
            valid = np.flatnonzero(np.isfinite(freqs) & (freqs > 0))
            if valid.size == 0:
                return self._unvoiced_pitch(start_time, volume=volume)
            valid_strengths = strengths[valid]
            if np.any(valid_strengths > 0):
                best = valid[int(np.argmax(valid_strengths))]
            else:
                best = valid[len(valid) // 2]
            return self._voiced_pitch(
                start_time,
                float(freqs[best]),
                volume,
                float(strengths[best]),
                include_distance=True,
            )

        def detect_pitches(
            self,
            x: np.ndarray,
            show_progress: bool = False,
            progress_desc: str = "Detecting pitches with Praat",
            verbose: bool = False,
        ) -> list[Pitch]:
            audio = self._mono_audio(x)
            n_frames = self._num_app_frames(len(audio))
            if n_frames <= 0:
                return []
            start = time.perf_counter()
            if verbose:
                print(f"[PraatStreaming] detecting {n_frames} frame(s)", flush=True)
            praat_audio = self._preprocess_praat_audio(audio)
            praat_times, freqs, strengths = self._praat_track(
                praat_audio, time_step=self.STEP_SECONDS
            )
            frame_times = np.arange(n_frames, dtype=float) * self.STEP_SECONDS
            frame_freqs = self._nearest_on_grid(
                praat_times, freqs, frame_times, fill=0.0
            )
            frame_strengths = self._nearest_on_grid(
                praat_times, strengths, frame_times, fill=0.0
            )
            volumes = self._frame_volumes(audio, n_frames)
            frames = range(n_frames)
            if show_progress:
                frames = tqdm(
                    frames,
                    total=n_frames,
                    desc=progress_desc,
                    leave=False,
                    mininterval=0.25,
                )
            pitches: list[Pitch] = []
            for i in frames:
                freq = float(frame_freqs[i])
                if np.isfinite(freq) and freq > 0:
                    pitch = self._voiced_pitch(
                        float(frame_times[i]),
                        freq,
                        float(volumes[i]),
                        float(frame_strengths[i]),
                        include_distance=False,
                    )
                else:
                    pitch = self._unvoiced_pitch(
                        float(frame_times[i]), volume=float(volumes[i])
                    )
                pitch.live_distance = None
                pitches.append(pitch)
            if verbose:
                print(
                    f"[PraatStreaming] done: {len(pitches)} pitch frame(s) in {time.perf_counter() - start:.2f}s",
                    flush=True,
                )
            return pitches

        @staticmethod
        def _parselmouth():
            try:
                import parselmouth
            except ImportError as exc:
                raise ImportError(
                    "PraatStreaming requires praat-parselmouth. Install it with `pip install praat-parselmouth` or `pip install -r requirements.txt`."
                ) from exc
            return parselmouth

        def _praat_track(
            self, audio: np.ndarray, time_step: float
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if len(audio) == 0 or not np.any(audio):
                empty = np.empty(0, dtype=float)
                return (empty, empty, empty)
            parselmouth = self._parselmouth()
            pitch = Praat.autocorrelation_pitch(
                audio,
                self.SR,
                max(float(time_step), 1e-06),
                self.PITCH_FLOOR,
                self.PITCH_CEILING,
                parselmouth=parselmouth,
            )
            selected = pitch.selected_array
            freqs = np.asarray(selected["frequency"], dtype=float)
            try:
                strengths = np.asarray(selected["strength"], dtype=float)
            except Exception:
                strengths = np.where(freqs > 0, 1.0, 0.0)
            strengths = np.clip(np.nan_to_num(strengths, nan=0.0), 0.0, 1.0)
            times = np.asarray(pitch.xs(), dtype=float)
            freqs = np.nan_to_num(freqs, nan=0.0, posinf=0.0, neginf=0.0)
            freqs = np.where(freqs > 0, freqs, 0.0)
            return (times, freqs, strengths)

        def _voiced_pitch(
            self,
            t: float | None,
            freq: float,
            volume: float,
            strength: float,
            include_distance: bool,
        ) -> Pitch:
            midi = self.config.freq_to_midi(freq)
            probability = float(np.clip(strength, 0.0, 1.0))
            if probability == 0.0:
                probability = 1.0
            candidates = [(midi, probability)]
            unvoiced_prob = 1.0 - probability
            distance = None
            if include_distance and self.recording is not None:
                score_data = getattr(self.recording, "score_data", None)
                note = score_data.current_note() if score_data is not None else None
                distance = note.midi_num[0] - midi if note and candidates else None
            return Pitch(
                time=t,
                candidates=candidates,
                volume=volume,
                unvoiced_prob=unvoiced_prob,
                live_distance=distance,
                config=self.config,
            )

        def _unvoiced_pitch(self, t: float | None, volume: float) -> Pitch:
            return Pitch(
                time=t,
                candidates=[],
                volume=volume,
                unvoiced_prob=1.0,
                live_distance=None,
                config=self.config,
            )

        @staticmethod
        def _mono_audio(x: np.ndarray) -> np.ndarray:
            audio = np.asarray(x, dtype=np.float64)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        @staticmethod
        def _frame_volume(frame: np.ndarray) -> float:
            if len(frame) == 0:
                return 0.0
            centered = frame - np.mean(frame)
            return float(np.sqrt(np.mean(centered**2)))

        def _preprocess_praat_audio(self, audio: np.ndarray) -> np.ndarray:
            """Match PitchDetector preprocessing for Praat input only.

            Volume is computed separately from the raw frame; this prepares the audio
            passed into Praat by centering, peak-normalizing, then bandpassing with
            the same fmin/fmax padding used by PitchDetector.preprocess_audio().
            """
            if len(audio) == 0:
                return np.array([], dtype=np.float64)
            x = np.asarray(audio, dtype=np.float64)
            x = x - np.mean(x)
            peak = np.max(np.abs(x))
            if peak == 0:
                return np.zeros_like(x)
            x = x / peak
            return self.bandpass_filter(
                x, fmin=self.config.fmin * 0.8, fmax=self.config.fmax * 1.2
            )

        def bandpass_filter(
            self, x: np.ndarray, fmin: float = 50, fmax: float = 4000
        ) -> np.ndarray:
            sos = iirfilter(
                N=2,
                Wn=[fmin, fmax],
                btype="bandpass",
                ftype="butter",
                output="sos",
                fs=self.SR,
            )
            return sosfilt(sos, x)

        def _num_app_frames(self, n_samples: int) -> int:
            if n_samples < self.FRAME_SIZE:
                return 0
            return 1 + (n_samples - self.FRAME_SIZE) // self.HOP_SIZE

        def _frame_volumes(self, audio: np.ndarray, n_frames: int) -> np.ndarray:
            if n_frames <= 0:
                return np.empty(0, dtype=float)
            starts = np.arange(n_frames, dtype=np.int64) * self.HOP_SIZE
            ends = starts + self.FRAME_SIZE
            prefix = np.concatenate(([0.0], np.cumsum(audio, dtype=np.float64)))
            prefix_sq = np.concatenate(
                ([0.0], np.cumsum(audio * audio, dtype=np.float64))
            )
            sums = prefix[ends] - prefix[starts]
            sums_sq = prefix_sq[ends] - prefix_sq[starts]
            mean = sums / self.FRAME_SIZE
            variance = sums_sq / self.FRAME_SIZE - mean * mean
            return np.sqrt(np.maximum(variance, 0.0))

        @staticmethod
        def _nearest_on_grid(
            source_times: np.ndarray,
            values: np.ndarray,
            target_times: np.ndarray,
            fill: float,
        ) -> np.ndarray:
            if len(source_times) == 0 or len(values) == 0:
                return np.full_like(target_times, fill, dtype=float)
            right = np.searchsorted(source_times, target_times, side="left")
            right = np.clip(right, 0, len(source_times) - 1)
            left = np.clip(right - 1, 0, len(source_times) - 1)
            choose_right = np.abs(source_times[right] - target_times) < np.abs(
                target_times - source_times[left]
            )
            nearest = np.where(choose_right, right, left)
            return values[nearest].astype(float, copy=False)
