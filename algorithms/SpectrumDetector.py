from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from algorithms.Config import Config

if TYPE_CHECKING:
    from app_logic.user.ds.Recording import Recording
    from app_logic.user.ds.TimbreData import TimbreData


class SpectrumDetector:
    """Hann-spectrum analysis plus its dedicated frame worker.

    Like PitchDetector, this class owns both the per-frame DSP and the threaded
    streaming lifecycle. A config-only instance can project cached spectra for
    display; a recording-bound instance additionally accepts raw frames from
    PitchDetector's shared scheduler and computes them on its own worker.
    """

    FLOOR_DB = -120.0
    REPRESENTATION = "hann_fft_bins_v1"
    _STOP = object()

    def __init__(
        self,
        recording: Recording | None = None,
        config: Config | None = None,
    ):
        if recording is None and config is None:
            raise ValueError(
                "SpectrumDetector requires either a recording or a config."
            )
        self.recording = recording
        self.config = config if config is not None else recording.config
        self.load_config(self.config)

        self.thread: threading.Thread | None = None
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._drain_on_stop = False
        self._accepting = False
        self._target: TimbreData | None = None
        self._state_lock = threading.Lock()
        self.error: Exception | None = None

    def load_config(self, config: Config):
        if hasattr(self, "thread"):
            self.stop(drain=False)
        self.config = config
        self.sr = int(config.sr)
        self.n_fft = int(config.w1)
        self.midi_min = int(config.cqt_midi_min)
        self.midi_max = int(config.cqt_midi_max)
        self.fft_frequencies = np.fft.rfftfreq(
            self.n_fft, d=1.0 / self.sr)
        self.bin_start, self.bin_stop = self.fft_bin_bounds(config)
        self.frequencies = self.fft_frequencies[self.bin_start:self.bin_stop]
        self.display_midis = np.arange(
            self.midi_min, self.midi_max + 1, dtype=np.float64)
        self.window = np.hanning(self.n_fft).astype(np.float64)
        self.scale = 2.0 / max(float(self.window.sum()), np.finfo(float).eps)
        self.display_filterbank = self._build_display_filterbank()

    # ------------------------------------------------------------------ #
    # Threaded app API
    # ------------------------------------------------------------------ #
    def start(self, target: TimbreData | None = None):
        """Start a fresh worker for live or offline frame submissions."""
        if self.recording is None and target is None:
            raise ValueError(
                "A config-only SpectrumDetector requires an explicit target."
            )
        self.stop(drain=False)
        with self._state_lock:
            self._queue = queue.SimpleQueue()
            self._stop_event = threading.Event()
            self._drain_on_stop = False
            self._target = (
                target if target is not None else self.recording.timbre_data
            )
            self._accepting = True
            self.error = None
            work_queue = self._queue
            stop_event = self._stop_event
            worker_target = self._target
            self.thread = threading.Thread(
                target=self._run,
                args=(work_queue, stop_event, worker_target),
                daemon=True,
                name="AttuneSpectrumDetector",
            )
            self.thread.start()

    def submit(self, raw_frame: np.ndarray, start_time: float):
        """Queue one raw frame at the exact pYIN frame-start grid point."""
        with self._state_lock:
            if not self._accepting or self._target is None:
                return
            target = self._target
            work_queue = self._queue

        frame = np.asarray(raw_frame, dtype=np.float32).reshape(-1)
        if len(frame) < self.n_fft:
            return
        frame = np.array(frame[:self.n_fft], copy=True)
        frame_index = int(round(
            (float(start_time) - target.t_origin)
            * (self.sr / self.config.h1)
        ))
        if frame_index < 0:
            return
        work_queue.put((frame_index, frame))

    def stop(self, drain: bool = True):
        """Stop accepting frames and optionally finish every queued spectrum."""
        with self._state_lock:
            thread = self.thread
            if thread is None:
                self._accepting = False
                return
            self._accepting = False
            self._drain_on_stop = bool(drain)
            self._stop_event.set()
            self._queue.put(self._STOP)
        if thread.is_alive() and thread is not threading.current_thread():
            thread.join()
        with self._state_lock:
            if self.thread is thread:
                self.thread = None
                self._target = None
                self._drain_on_stop = False

    def is_running(self) -> bool:
        thread = self.thread
        return thread is not None and thread.is_alive()

    def detect_frames_async(
        self,
        audio: np.ndarray,
        target: TimbreData,
        n_frames: int | None = None,
        on_done=None,
    ) -> bool:
        """Backfill a whole-audio spectrum grid on this detector's thread."""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        available = max(
            0,
            1 + (len(samples) - self.n_fft) // int(self.config.h1),
        )
        count = available if n_frames is None else min(available, int(n_frames))
        if count <= 0:
            if on_done is not None:
                on_done()
            return False

        self.stop(drain=False)
        with self._state_lock:
            self._queue = queue.SimpleQueue()
            self._stop_event = threading.Event()
            self._drain_on_stop = False
            self._target = target
            self._accepting = False
            self.error = None
            stop_event = self._stop_event
            self.thread = threading.Thread(
                target=self._detect_frames,
                args=(samples, count, stop_event, target, on_done),
                daemon=True,
                name="AttuneSpectrumDetector",
            )
            self.thread.start()
        return True

    def _run(
        self,
        work_queue: queue.SimpleQueue,
        stop_event: threading.Event,
        target: TimbreData,
    ):
        while True:
            item = work_queue.get()
            if item is self._STOP:
                return
            if stop_event.is_set() and not self._drain_on_stop:
                return
            frame_index, frame = item
            try:
                target.write(frame_index, self.power_db(frame))
            except Exception as exc:
                self.error = exc
                print(f"[SpectrumDetector] frame skipped due to error: {exc}")

    def _detect_frames(
        self,
        audio: np.ndarray,
        n_frames: int,
        stop_event: threading.Event,
        target: TimbreData,
        on_done,
    ):
        """Whole-audio worker used to populate missing spectrum caches."""
        try:
            frames = np.lib.stride_tricks.sliding_window_view(
                audio,
                self.n_fft,
            )[::int(self.config.h1)]
            for frame_index, frame in enumerate(frames[:n_frames]):
                if stop_event.is_set() and not self._drain_on_stop:
                    break
                target.write(frame_index, self.power_db(frame))
        except Exception as exc:
            self.error = exc
            print(f"[SpectrumDetector] backfill failed: {exc}")
        finally:
            if on_done is not None:
                on_done()

    # ------------------------------------------------------------------ #
    # Spectrum analysis
    # ------------------------------------------------------------------ #
    @classmethod
    def bin_count(cls, config: Config) -> int:
        start, stop = cls.fft_bin_bounds(config)
        return stop - start

    @classmethod
    def fft_bin_bounds(cls, config: Config) -> tuple[int, int]:
        """Stable half-open FFT range covering the display plus one semitone."""
        n_fft = int(config.w1)
        sr = int(config.sr)
        bin_hz = sr / n_fft
        midi_min = int(config.cqt_midi_min) - 1
        midi_max = int(config.cqt_midi_max) + 1
        fmin = 440.0 * 2.0 ** ((midi_min - 69) / 12.0)
        fmax = 440.0 * 2.0 ** ((midi_max - 69) / 12.0)
        start = max(1, int(np.floor(fmin / bin_hz)))
        stop = min(n_fft // 2 + 1, int(np.ceil(fmax / bin_hz)) + 1)
        return start, max(start, stop)

    @property
    def display_bin_count(self) -> int:
        return len(self.display_midis)

    def _build_display_filterbank(self) -> csr_matrix:
        """Triangular semitone bands used only for the compact live image."""
        fft_bin_hz = self.sr / self.n_fft
        rows, cols, values = [], [], []
        for row, midi in enumerate(self.display_midis):
            center = self.config.midi_to_freq(midi)
            lower_neighbor = self.config.midi_to_freq(midi - 1.0)
            upper_neighbor = self.config.midi_to_freq(midi + 1.0)
            half_width = max(
                center - 0.5 * (lower_neighbor + center),
                0.5 * (center + upper_neighbor) - center,
                1.5 * fft_bin_hz,
            )
            weights = np.maximum(
                1.0 - np.abs(self.frequencies - center) / half_width,
                0.0,
            )
            nz = np.flatnonzero(weights > 0.0)
            if nz.size == 0:
                nz = np.asarray([
                    int(np.argmin(np.abs(self.frequencies - center)))
                ])
                weights[nz] = 1.0
            normalized = weights[nz] / weights[nz].sum()
            rows.extend([row] * len(nz))
            cols.extend(nz.tolist())
            values.extend(normalized.tolist())
        return csr_matrix(
            (values, (rows, cols)),
            shape=(self.display_bin_count, len(self.frequencies)),
            dtype=np.float64,
        )

    def power_db(self, x: np.ndarray) -> np.ndarray:
        """One raw frame -> native cropped FFT-bin power in clipped dBFS."""
        frame = np.asarray(x, dtype=np.float64).reshape(-1)
        if len(frame) < self.n_fft:
            frame = np.pad(frame, (0, self.n_fft - len(frame)))
        elif len(frame) > self.n_fft:
            frame = frame[:self.n_fft]
        spectrum = np.abs(np.fft.rfft(frame * self.window)) * self.scale
        power = (spectrum * spectrum)[self.bin_start:self.bin_stop]
        db = 10.0 * np.log10(np.maximum(power, 10.0 ** (self.FLOOR_DB / 10.0)))
        return np.clip(db, self.FLOOR_DB, 0.0).astype(np.float32)

    def semitone_power_db(self, raw_db: np.ndarray) -> np.ndarray:
        """Project stored FFT-bin dB values onto compact semitone bands."""
        raw = np.asarray(raw_db, dtype=np.float64)
        if raw.shape[0] != self.bin_count(self.config):
            raise ValueError(
                f"expected {self.bin_count(self.config)} FFT bins, "
                f"got {raw.shape[0]}"
            )
        power = np.power(10.0, raw / 10.0)
        display_power = np.asarray(self.display_filterbank @ power)
        display_db = 10.0 * np.log10(np.maximum(
            display_power,
            10.0 ** (self.FLOOR_DB / 10.0),
        ))
        return np.clip(display_db, self.FLOOR_DB, 0.0).astype(np.float32)
