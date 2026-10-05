import threading
from math import ceil, floor

import numpy as np

from algorithms.Config import Config


class VibratoData:
    """Time-indexed vibrato-characteristic track: speed (Hz), peak-to-peak
    width (cents), and fit quality per point of a uniform pitch-frame grid
    (index i <-> pitch frame i*stride <-> that frame's center time). A computed
    0 Hz / 0 cents sample means no measurable oscillation (including unvoiced
    pitch); NaN is reserved for unwritten/not-yet-computed time. Filled by
    a vibrato detector; never persisted — it derives purely from the pitch
    track, so a cache load just recomputes it.

    ``centers`` is an optional MIDI-pitch track for the estimated
    vibrato-less note contour. Detector 2 obtains it from the same joint fit as
    the sinusoid; older detectors leave it NaN. It is kept on the same grid so
    the note panel can show systematic pitch movement without recomputing
    analysis.

    Detector 1 source-maps overlapping fixed-window estimates. Detector 2
    instead stores one smooth whole-note fit offline, or interpolated causal
    fit anchors live. In either case, speed and width remain frame-dense.

    The note association is arithmetic on the uniform grid
    (note_index_range), not a stored map, so it cannot go stale when
    analysis rebuilds the note objects."""

    GROW = 1024

    def __init__(self, config: Config):
        self.config = config
        self.stride = max(1, int(config.vib_stride))
        self.t_origin = 0.0
        self.lock = threading.Lock()
        self.rates = np.full(self.GROW, np.nan, dtype=np.float32)
        self.widths = np.full(self.GROW, np.nan, dtype=np.float32)
        self.qualities = np.full(self.GROW, np.nan, dtype=np.float32)
        self.centers = np.full(self.GROW, np.nan, dtype=np.float32)
        self.computed_until = 0  # grid high-water mark (exclusive)
        # Cached by VibratoDetector after the first real pitch frame appears;
        # avoids rescanning a long leading clip gap on every live callback.
        self.source_first_index: int | None = None

    # --- the uniform grid (pitch-frame centers, mirroring Pitch.time) ---
    def grid_dt(self) -> float:
        return self.stride * self.config.h1 / self.config.sr

    def index_time(self, i):
        """App-time of grid index i (vectorizes over numpy arrays)."""
        cfg = self.config
        return self.t_origin + (i * self.stride * cfg.h1 + 0.5 * cfg.w1) / cfg.sr

    def grid_pos(self, t: float) -> float:
        """Fractional grid position of app-time t (inverse of index_time)."""
        cfg = self.config
        return ((t - self.t_origin) * cfg.sr - 0.5 * cfg.w1) / cfg.h1 / self.stride

    def _index_range_to(self, t0: float, t1: float,
                        high_water: int) -> tuple[int, int]:
        i0 = max(0, ceil(self.grid_pos(t0)))
        i1 = min(high_water, floor(self.grid_pos(t1)) + 1)
        return i0, max(i0, i1)

    def index_range(self, t0: float, t1: float) -> tuple[int, int]:
        """Half-open computed grid range whose center times lie in [t0, t1]."""
        return self._index_range_to(t0, t1, self.computed_until)

    def note_index_range(self, note) -> tuple[int, int]:
        """The note -> vibrato-samples association: half-open grid indices
        covering the note's [start, end]."""
        return self.index_range(note.start_time, note.end_time)

    # --- writing (vibrato detector) ---
    def _ensure_capacity_unlocked(self, i: int):
        if i >= len(self.rates):
            grow = max(self.GROW, i + 1 - len(self.rates))
            pad = np.full(grow, np.nan, dtype=np.float32)
            self.rates = np.concatenate([self.rates, pad])
            self.widths = np.concatenate([self.widths, pad.copy()])
            self.qualities = np.concatenate([self.qualities, pad.copy()])
            self.centers = np.concatenate([self.centers, pad.copy()])

    def write(self, i: int, rate: float, width: float, quality: float,
              center: float = np.nan):
        with self.lock:
            self._ensure_capacity_unlocked(i)
            self.rates[i] = rate
            self.widths[i] = width
            self.qualities[i] = quality
            self.centers[i] = center
            self.computed_until = max(self.computed_until, i + 1)

    # --- queries ---
    @staticmethod
    def _median3(a: np.ndarray) -> np.ndarray:
        """3-point median where both neighbors exist: one bad analysis window
        can't flick the curve (isolated nonzero islands drop, single-sample
        dropouts heal); any two agreeing neighbors pass through unchanged."""
        if len(a) < 3:
            return a
        prev, cur, nxt = a[:-2], a[1:-1], a[2:]
        ok = np.isfinite(prev) & np.isfinite(cur) & np.isfinite(nxt)
        out = a.copy()
        out[1:-1] = np.where(ok, np.median(np.vstack([prev, cur, nxt]), axis=0), cur)
        return out

    def curve(self, t0: float, t1: float):
        """(times, speeds, peak-to-peak widths) over [t0, t1].

        Values are NaN where they have not been computed.
        Read-side 3-point median (one extra sample pulled past each end so
        edge values smooth identically) — the stored grid stays raw."""
        i0, i1 = self.index_range(t0, t1)
        j0, j1 = max(0, i0 - 1), min(self.computed_until, i1 + 1)
        with self.lock:
            rates = self.rates[j0:j1].astype(float, copy=True)
            widths = self.widths[j0:j1].astype(float, copy=True)
        rates = self._median3(rates)[i0 - j0:i1 - j0]
        widths = self._median3(widths)[i0 - j0:i1 - j0]
        times = self.index_time(np.arange(i0, i1, dtype=float))
        return times, rates, widths

    def center_curve(self, t0: float, t1: float):
        """(times, MIDI centers) for the detector's vibrato-less contour."""
        i0, i1 = self.index_range(t0, t1)
        with self.lock:
            centers = self.centers[i0:i1].astype(float, copy=True)
        times = self.index_time(np.arange(i0, i1, dtype=float))
        return times, centers

    def global_characteristic_range(
            self, metric: str,
    ) -> tuple[float, float] | None:
        """Recording-wide min/max for a displayed vibrato characteristic.

        Uses the same median-smoothed values as :meth:`curve` and only samples
        with a positive detected rate. The stored 0 Hz / 0 cents sentinel means
        "no measurable vibrato", so including it would make every recording's
        slow/narrow endpoint zero rather than the least/most subtle vibrato the
        performer actually produced.
        """
        if metric not in {"rate", "width"}:
            raise ValueError(f"Unknown vibrato metric: {metric}")
        with self.lock:
            rates = self.rates[:self.computed_until].astype(float, copy=True)
            widths = self.widths[:self.computed_until].astype(float, copy=True)
        rates = self._median3(rates)
        widths = self._median3(widths)
        values = rates if metric == "rate" else widths
        detected = (
            np.isfinite(rates)
            & np.isfinite(widths)
            & (rates > 0.0)
        )
        if not detected.any():
            return None
        return float(np.min(values[detected])), float(np.max(values[detected]))

    def at(self, t: float) -> tuple[float, float] | tuple[None, None]:
        """(speed, width) at the nearest grid point (either may be NaN)."""
        i = int(round(self.grid_pos(t)))
        if not (0 <= i < self.computed_until):
            return None, None
        with self.lock:
            return float(self.rates[i]), float(self.widths[i])

    def note_summary(self, note) -> tuple[float, float] | tuple[None, None]:
        """Per-note median (speed_hz, peak_to_peak_width_cents), or None.

        Detector 2 has already converted a fit below its whole-note rate or
        width floor to the 0/0 sentinel used for no measurable vibrato.
        """
        i0, i1 = self.note_index_range(note)
        with self.lock:
            rates = self.rates[i0:i1].astype(float, copy=True)
            widths = self.widths[i0:i1].astype(float, copy=True)
        mask = np.isfinite(rates) & np.isfinite(widths) & (rates > 0.0)
        if not mask.any():
            return None, None
        rate = float(np.median(rates[mask]))
        width = float(np.median(widths[mask]))
        return rate, width

    def trim_to(self, t: float):
        """Drop samples past app-time t (mirrors the take's trim_end)."""
        keep = max(0, floor(self.grid_pos(t)) + 1)
        with self.lock:
            self.computed_until = min(self.computed_until, keep)
