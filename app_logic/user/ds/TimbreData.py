import threading
from math import ceil, floor

import numpy as np

from algorithms.SpectrumDetector import SpectrumDetector
from algorithms.Config import Config


class TimbreData:
    """Frame-dense FFT spectrum owned by one Recording.

    Column i corresponds to pitch frame i and therefore to app-time
    t_origin + (i*h1 + w1/2)/sr. Missing/uncomputed columns read at the
    display floor. Stored rows retain every native FFT bin across the configured
    frequency range; the widget requests a compact semitone projection. Unlike
    VibratoData, this raw-audio-derived stream is cached.
    """

    GROW = 1024

    def __init__(self, config: Config):
        self.config = config
        self.stride = int(config.spectrum_stride)
        self.t_origin = 0.0
        self.midi_min = int(config.cqt_midi_min)
        self.midi_max = int(config.cqt_midi_max)
        self.representation = SpectrumDetector.REPRESENTATION
        self.floor_db = SpectrumDetector.FLOOR_DB
        self.spectrum = SpectrumDetector(config=config)
        self.lock = threading.Lock()
        self.values = np.full(
            (self.n_bins, self.GROW), self.floor_db, dtype=np.float32)
        self.written = np.zeros(self.GROW, dtype=bool)
        self.computed_until = 0

    @property
    def n_bins(self) -> int:
        return self.spectrum.bin_stop - self.spectrum.bin_start

    def update_config(self, config: Config):
        """Refresh musical projection metadata, preserving compatible FFT data.

        A tuning change only rebuilds the semitone display filters. Sample-rate,
        frame-size, stride, or FFT-bin-bound changes invalidate stored columns
        because they change the source time/frequency grid.
        """
        replacement = SpectrumDetector(config=config)
        new_stride = int(config.spectrum_stride)
        compatible = (
            self.stride == new_stride
            and self.spectrum.sr == replacement.sr
            and self.spectrum.n_fft == replacement.n_fft
            and self.spectrum.bin_start == replacement.bin_start
            and self.spectrum.bin_stop == replacement.bin_stop
        )
        self.config = config
        self.stride = new_stride
        self.midi_min = int(config.cqt_midi_min)
        self.midi_max = int(config.cqt_midi_max)
        self.spectrum = replacement
        if compatible:
            return
        with self.lock:
            self.values = np.full(
                (self.n_bins, self.GROW), self.floor_db, dtype=np.float32)
            self.written = np.zeros(self.GROW, dtype=bool)
            self.computed_until = 0

    def midi_bounds(self) -> tuple[float, float]:
        """Outer cell edges for the integer-semitone display rows."""
        return self.midi_min - 0.5, self.midi_max + 0.5

    def grid_dt(self) -> float:
        return self.stride * self.config.h1 / self.config.sr

    def index_time(self, i):
        cfg = self.config
        return self.t_origin + (i * self.stride * cfg.h1 + 0.5 * cfg.w1) / cfg.sr

    def grid_pos(self, t: float) -> float:
        cfg = self.config
        return ((t - self.t_origin) * cfg.sr - 0.5 * cfg.w1) / cfg.h1 / self.stride

    def index_range(self, t0: float, t1: float) -> tuple[int, int]:
        i0 = max(0, ceil(self.grid_pos(t0)))
        i1 = min(self.computed_until, floor(self.grid_pos(t1)) + 1)
        return i0, max(i0, i1)

    def _grow_to(self, i: int):
        if i < self.values.shape[1]:
            return
        grow = max(self.GROW, i + 1 - self.values.shape[1])
        self.values = np.pad(
            self.values, ((0, 0), (0, grow)),
            constant_values=self.floor_db,
        )
        self.written = np.pad(self.written, (0, grow), constant_values=False)

    def write(self, i: int, column: np.ndarray):
        col = np.asarray(column, dtype=np.float32).reshape(-1)
        if len(col) != self.n_bins:
            raise ValueError(f"expected {self.n_bins} timbre bins, got {len(col)}")
        with self.lock:
            self._grow_to(i)
            self.values[:, i] = np.clip(col, self.floor_db, 0.0)
            self.written[i] = True
            self.computed_until = max(self.computed_until, i + 1)

    def matrix(self, t0: float, t1: float):
        """Full-resolution (times, FFT bins x columns) for [t0, t1]."""
        i0, i1 = self.index_range(t0, t1)
        with self.lock:
            matrix = self.values[:, i0:i1].astype(float, copy=True)
            written = self.written[i0:i1].copy()
        if matrix.size and not written.all():
            matrix[:, ~written] = self.floor_db
        times = self.index_time(np.arange(i0, i1, dtype=float))
        return times, matrix

    def display_matrix(self, t0: float, t1: float):
        """(times, integer-semitone bands x columns) for the live widget."""
        times, raw = self.matrix(t0, t1)
        if raw.shape[1] == 0:
            return times, np.empty((self.spectrum.display_bin_count, 0))
        return times, self.spectrum.semitone_power_db(raw)

    def range_db(self) -> tuple[float, float]:
        """Robust range over the displayed semitone bands (fallback [-80, 0])."""
        with self.lock:
            mask = self.written[:self.computed_until]
            if not mask.any():
                return -80.0, 0.0
            vals = self.values[:, :self.computed_until][:, mask]
        display = self.spectrum.semitone_power_db(vals)
        low = float(np.percentile(display, 5))
        high = float(np.max(display))
        if high <= low:
            low = max(self.floor_db, high - 20.0)
        return low, high

    def is_empty(self) -> bool:
        with self.lock:
            return not self.written[:self.computed_until].any()

    def trim_to(self, t: float):
        keep = max(0, floor(self.grid_pos(t)) + 1)
        with self.lock:
            self.computed_until = min(self.computed_until, keep)

    def load_quantized(self, quantized: np.ndarray, n_cols: int):
        """Restore bins x columns uint8 half-dB offsets from a sidecar."""
        n_cols = max(0, int(n_cols))
        q = np.asarray(quantized, dtype=np.uint8)
        if q.size != self.n_bins * n_cols:
            raise ValueError("timbre cache dimensions do not match its blob")
        vals = q.reshape(self.n_bins, n_cols).astype(np.float32) * 0.5 + self.floor_db
        with self.lock:
            capacity = max(self.GROW, n_cols)
            self.values = np.full(
                (self.n_bins, capacity), self.floor_db, dtype=np.float32)
            self.written = np.zeros(capacity, dtype=bool)
            self.values[:, :n_cols] = vals
            self.written[:n_cols] = True
            self.computed_until = n_cols
