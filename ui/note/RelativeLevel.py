import math


class RelativeLevelStats:
    """Incremental mean voiced-frame dB level shared by Volume and Timbre.

    Only written frames contribute. Call reset after in-place analysis edits;
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

    def sync(self, pitch_data):
        end = pitch_data.frames_available()
        if self.data is not pitch_data.data or end < self.frames_seen:
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
        self.frames_seen = end

    @property
    def reference_db(self):
        return self.total / self.count if self.count else None
