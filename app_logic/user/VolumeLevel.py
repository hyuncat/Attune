"""Voiced-frame dB statistics and symmetric mean-relative volume colors."""
import math
import numpy as np


class VolumeColorScale:
    """Linear dB colors on a symmetric range around mean voiced-frame dB.

    The largest absolute deviation sets both endpoints. Constant takes map
    to teal; missing voiced data has no color position.
    """

    LIVE_HALF_RANGE_DB = 12.0

    def __init__(self, levels):
        levels = np.asarray(levels, dtype=float)
        levels = levels[np.isfinite(levels)]
        self.reference_db = float(np.mean(levels)) if len(levels) else None
        self.half_range_db = (float(np.max(np.abs(levels - self.reference_db)))
                              if len(levels) else 0.0)

    @staticmethod
    def normalized_db(db, reference_db, half_range_db):
        if reference_db is None or db is None or not math.isfinite(db):
            return None
        if half_range_db <= 1e-10:
            return 0.5
        return max(0.0, min(1.0, 0.5 + (db - reference_db) / (2 * half_range_db)))

    def fraction_db(self, db):
        return self.normalized_db(db, self.reference_db, self.half_range_db)

    def fraction(self, volume):
        if volume is None or not math.isfinite(volume) or volume <= 0:
            return None
        return self.fraction_db(20.0 * math.log10(volume))


class RelativeLevelStats:
    """Incremental mean voiced-frame dB, shared by panels and volume colors.

    Only written frames contribute. Reset after in-place analysis edits;
    replacing or trimming the pitch track is detected automatically.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = None
        self.frames_seen = 0
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.levels = []
        self.live_fractions = {}
        self._color_scale = None
        self._tail_signature = None

    @staticmethod
    def _signature(pitch):
        return (getattr(pitch, "volume", None), getattr(pitch, "value", None),
                getattr(pitch, "unvoiced_prob", None))

    def sync(self, pitch_data):
        end = pitch_data.frames_available()
        tail_changed = (self.data is pitch_data.data and self.frames_seen > 0
                        and end >= self.frames_seen
                        and self._signature(pitch_data.data[self.frames_seen - 1])
                        != self._tail_signature)
        if self.data is not pitch_data.data or end < self.frames_seen or tail_changed:
            self.reset()
            self.data = pitch_data.data
        for pitch in pitch_data.data[self.frames_seen:end]:
            if not pitch_data.is_voiced_pitch(pitch):
                continue
            volume = float(pitch.volume)
            if not math.isfinite(volume) or volume <= 0:
                continue
            db = 20.0 * math.log10(volume)
            self.total += db
            self.count += 1
            self.minimum = min(self.minimum, db)
            self.maximum = max(self.maximum, db)
            self.levels.append(db)
            # Assign chronologically, even when frames arrive in a batch or
            # volume mode is selected late. Later frames never recolor these.
            self.live_fractions[id(pitch)] = VolumeColorScale.normalized_db(
                db, self.reference_db, VolumeColorScale.LIVE_HALF_RANGE_DB)
            self._color_scale = None
        self.frames_seen = end
        self._tail_signature = self._signature(pitch_data.data[end - 1]) if end else None

    @property
    def reference_db(self):
        return self.total / self.count if self.count else None

    @property
    def color_scale(self):
        if self._color_scale is None:
            self._color_scale = VolumeColorScale(self.levels)
        return self._color_scale
