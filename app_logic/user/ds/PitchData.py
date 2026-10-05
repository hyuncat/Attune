import threading
from math import floor, ceil
import numpy as np

from algorithms.Config import Config

class Pitch:
    def __init__(self, time: float, volume: float, unvoiced_prob: float,
                 live_distance: float, config: Config,
                 candidates: list[tuple[float, float]] | None=None,
                 value: float | None=None):
        """
        The quintessential pitch object for the app. Corresponds to a given [time] in the
        PitchData and stores all possible pitch [candidates] = [(midi_num, prob), ...]
        sorted from most --> least probable as well as the volume and a reference to the
        settings (config) with which it was computed
        """
        # --- essential --- #
        candidates = [
            (float(midi), float(prob))
            for midi, prob in (candidates or [])
        ]
        self.config = config  # tuning / fmin/fmax / note naming
        self.time = time      # center time of the frame when pitch was computed
        # ``None`` means "select the leading candidate".  ``-1`` is an actual
        # unvoiced value and must survive cache round-trips even when candidate
        # evidence is retained for the offline smoother.
        self.value = (
            candidates[0][0]
            if value is None and candidates
            else (-1 if value is None else value)
        )
        self.volume = volume  # centered integration-window RMS

        # ---> pre-smoother characteristics
        self.unvoiced_prob = unvoiced_prob # how messy the signal was (no clean periodicity)
        self.candidate_pitches = candidates # [(midi_num, prob), ...]; descending probability
        # --- used in note detection --- #
        self.is_transition = None

        # --- alignment-based --- #
        self.live_distance = live_distance  # distance to target note, filled in live
        self.aligned_distance = None # post-analysis distance

    def get_note_name(self) -> str:
        return self.config.get_note_name(self.value)

    def set_transition(self, value: bool):
        self.is_transition = value

    @property
    def candidates(self) -> list[tuple[float, float]]:
        """Backward-compatible alias for pre-rename caches/code."""
        return self.__dict__.get(
            "candidate_pitches",
            self.__dict__.get("candidates", []),
        )

    @candidates.setter
    def candidates(self, value: list[tuple[float, float]] | None):
        self.__dict__["candidate_pitches"] = [
            (float(midi), float(prob))
            for midi, prob in (value or [])
        ]

    @property
    def distance(self):
        return self.__dict__.get("live_distance", self.__dict__.get("distance"))

    @distance.setter
    def distance(self, value):
        self.__dict__["live_distance"] = value

    @property
    def align_distance(self):
        return self.__dict__.get(
            "aligned_distance",
            self.__dict__.get("align_distance"),
        )

    @align_distance.setter
    def align_distance(self, value):
        self.__dict__["aligned_distance"] = value

    def ensure_compatible(self, config: Config | None = None) -> "Pitch":
        """Patch legacy pickled Pitch instances in-place.

        Older caches used `candidates`, `distance`, and `align_distance`; the
        current app uses `candidate_pitches`, `value`, `live_distance`, and
        `aligned_distance`.
        """
        if config is not None:
            self.config = config
        elif not hasattr(self, "config") or self.config is None:
            self.config = Config()

        if not hasattr(self, "candidate_pitches"):
            self.candidate_pitches = [
                (float(midi), float(prob))
                for midi, prob in (self.__dict__.get("candidates", []) or [])
            ]
        else:
            self.candidate_pitches = [
                (float(midi), float(prob))
                for midi, prob in (self.candidate_pitches or [])
            ]

        if not hasattr(self, "value"):
            self.value = self.candidate_pitches[0][0] if self.candidate_pitches else -1
        if not hasattr(self, "live_distance"):
            self.live_distance = self.__dict__.get("distance")
        if not hasattr(self, "aligned_distance"):
            self.aligned_distance = self.__dict__.get("align_distance")
        if not hasattr(self, "is_transition"):
            self.is_transition = None
        if not hasattr(self, "volume"):
            self.volume = 0.0
        if not hasattr(self, "unvoiced_prob"):
            self.unvoiced_prob = 1.0
        # v3/v4 caches carried ordinary post-hoc candidates beside prominent
        # live candidates. Promotion removed prominence, so legacy objects
        # collapse to that ordinary list and then discard the duplicate fields.
        legacy_posthoc = self.__dict__.pop("posthoc_candidate_pitches", None)
        if legacy_posthoc is not None:
            self.candidate_pitches = [
                (float(midi), float(prob))
                for midi, prob in (legacy_posthoc or [])
            ]
        legacy_posthoc_unvoiced = self.__dict__.pop(
            "posthoc_unvoiced_prob",
            None,
        )
        if legacy_posthoc_unvoiced is not None:
            self.unvoiced_prob = float(legacy_posthoc_unvoiced)
        if not hasattr(self, "time"):
            self.time = 0.0
        return self

class PitchData:
    def __init__(self, config: Config):
        """
        an audio data-like pitch data: an array of pitches
	       - indexable using the (SR / hop size)
        """
        # reference to the global config (for sr + hop size used in pitch detection)
        self.config = config

        # App-time (sec) that frame 0 represents (mirrors AudioData.t_origin). A
        # Perform take's one-beat runway records from a NEGATIVE app-time; t_origin
        # keeps the frame array 0-indexed while time<->index stays in app-time.
        self.t_origin = 0.0

        # the essential time to index lambda
        self.time_to_index = lambda sec: floor((sec - self.t_origin)*(self.config.sr / self.config.h1))
        
        DEFAULT_LENGTH = 60 # (sec)
        self.data: list[Pitch] = [None] * ceil(self.time_to_index(DEFAULT_LENGTH))
        self.lock = threading.Lock()
        # high-water mark: one past the last frame write() has landed. The
        # offline/cache paths assign `data` wholesale instead — see
        # frames_available() for the fallback they rely on.
        self.end_index = 0

        self.UNVOICED_THRESHOLD = config.unv_thresh # threshold above which a pitch is considered unvoiced

    def resize(self, resize_factor=2):
        """increase the capacity of the current pitch array"""
        with self.lock:
            new_data = [None] * (len(self.data) * resize_factor)
            self.data.extend(new_data)

    def load(self, pitches: list[Pitch]):
        """load in an entire pitch array"""
        self.data = [
            p.ensure_compatible(self.config) if p is not None else None
            for p in pitches
        ]
        self.end_index = len(self.data)

    def write(
        self,
        pitches: list[Pitch] | Pitch,
        start_time: float | None = None,
    ):
        """write the pitches to the data at the given time index"""
        if isinstance(pitches, Pitch):
            pitches = [pitches]
        pitches = [
            p.ensure_compatible(self.config) if p is not None else None
            for p in pitches
        ]
        if start_time is None:
            start_time = pitches[0].time

        # get indices into data array
        # Live writes are addressed by an exact frame-start grid point. Round
        # that point rather than flooring a nearly-integral float, which can
        # otherwise map two adjacent live frames onto the same storage slot.
        i = int(round(
            (start_time - self.t_origin)
            * (self.config.sr / self.config.h1)
        ))
        j = i+len(pitches)

        if j > len(self.data)*0.8: # if close enough to end
            self.resize()

        with self.lock:
            self.data[i:j] = pitches
            self.end_index = max(self.end_index, j)

    def frames_available(self) -> int:
        """One past the last frame that holds data. write() tracks it live;
        data assigned wholesale (offline detection, cache load) is an
        exact-length list with a real tail, so fall back to the length then."""
        n = self.end_index
        if self.data and self.data[-1] is not None:
            n = max(n, len(self.data))
        return min(n, len(self.data))

    def read(self, start_time: float=0, end_time: float=0, i: int=None, j: int=None, clean=False, include_transitions: bool=True) -> list[Pitch]:
        """returns the array of pitches corresponding to start_time <--> end_time"""
        if not i and not j:
            i = max(0, self.time_to_index(start_time))
            j = min(self.time_to_index(end_time), len(self.data)-1)

        if clean:
            return [p for p in self.data[i:j] if self.is_voiced_pitch(p, include_transitions=include_transitions)]

        return self.data[i:j]
    
    def read_pitch(self, start_time: float=0) -> Pitch:
        """returns the closest pitch to the start_time"""
        i = self.time_to_index(start_time)
        if i < 0 or i >= len(self.data):
            return None
        return self.data[i]

    def is_voiced_pitch(self, pitch: Pitch, include_transitions: bool=True) -> bool:
        if pitch is not None:
            pitch.ensure_compatible(self.config) # patches legacy pitch caches in-place
        return (
            pitch is not None
            and pitch.value != -1
            and pitch.unvoiced_prob < self.config.unv_thresh
            and (include_transitions or not getattr(pitch, "is_transition", False))
        )

    def _frame_time(self, i: int) -> float:
        """App-time of frame ``i`` on the active live or centered grid."""
        return (
            self.t_origin
            + i * self.config.h1 / self.config.sr
            + self._frame_time_offset()
        )

    def _frame_time_offset(self) -> float:
        """Infer whether this track is causal-live or centered post-hoc.

        Live frames are stored at their capture-window start but carry a
        ``w1/2`` center timestamp. Whole-file pYIN uses centered padding, so its
        frame-zero timestamp is exactly ``t_origin``. Keeping the distinction
        on the data itself lets curve readers serve both without changing the
        public indexing API.
        """
        step = self.config.h1 / self.config.sr
        for index, pitch in enumerate(self.data):
            if pitch is not None:
                return float(pitch.time) - self.t_origin - index * step
        return 0.5 * self.config.w1 / self.config.sr

    def _curve_index_range(self, t0: float, t1: float) -> tuple[int, int]:
        """Half-open frame-index range whose CENTER times fall in [t0, t1].

        `time_to_index` addresses streaming writes by their frame-start time;
        completed-audio frames instead use pYIN's centered grid. Infer and undo
        the appropriate offset from the stored frame timestamps.
        """
        cfg = self.config
        offset = self._frame_time_offset()
        pos0 = (t0 - self.t_origin - offset) * cfg.sr / cfg.h1
        pos1 = (t1 - self.t_origin - offset) * cfg.sr / cfg.h1
        i0 = max(0, int(np.ceil(pos0)))
        i1 = min(len(self.data), max(i0, int(np.floor(pos1)) + 1))
        return i0, i1

    def pitch_curve(self, t0: float, t1: float, include_transitions: bool=True) -> tuple[np.ndarray, np.ndarray]:
        """(times, midis) over the frames in [t0, t1) — the pitch contour,
        with NaN where a frame is missing/unvoiced so plots drawn with
        connect='finite' break across the gaps."""
        i0, j = self._curve_index_range(t0, t1)
        times = np.empty(j - i0)
        midis = np.full(j - i0, np.nan)
        for k in range(j - i0):
            p = self.data[i0 + k]
            times[k] = p.time if p is not None else self._frame_time(i0 + k)
            if p is not None and self.is_voiced_pitch(p, include_transitions=include_transitions):
                midis[k] = p.value
        return times, midis

    def volume_curve(self, t0: float, t1: float,
                     floor_db: float = -120.0) -> tuple[np.ndarray, np.ndarray]:
        """(times, dBFS) over every WRITTEN frame in [t0, t1).

        Pitch voicing is irrelevant to loudness: an unpitched/noisy frame still
        has meaningful RMS volume. Digital silence is drawn at `floor_db` so a
        continuous recorded stream stays continuous; only an unwritten `None`
        frame remains NaN and breaks the curve.
        """
        i0, j = self._curve_index_range(t0, t1)
        times = np.empty(j - i0)
        dbs = np.full(j - i0, np.nan)
        for k in range(j - i0):
            p = self.data[i0 + k]
            times[k] = p.time if p is not None else self._frame_time(i0 + k)
            if p is None:
                continue
            vol = float(getattr(p, "volume", 0.0) or 0.0)
            dbs[k] = (max(float(floor_db), 20.0 * np.log10(vol))
                      if vol > 0 else float(floor_db))
        return times, dbs

    def volume_range_db(self) -> tuple[float | None, float | None]:
        """(min_dBFS, max_dBFS) over the voiced frames' volumes, or (None, None)
        when nothing voiced carries volume. Review-mode volume coloring maps
        this range onto the quiet->loud ramp."""
        frames = self.read(i=0, j=len(self.data), clean=True)
        vols = [float(p.volume) for p in frames if getattr(p, "volume", 0.0) > 0]
        if not vols:
            return (None, None)
        return (20.0 * float(np.log10(min(vols))), 20.0 * float(np.log10(max(vols))))

    def mean_volume(self, start_time: float, end_time: float) -> float:
        """Mean voiced-frame volume in the window (0.0 when nothing voiced)."""
        frames = self.read(start_time=start_time, end_time=end_time, clean=True)
        vols = [float(p.volume) for p in frames if getattr(p, "volume", 0.0) > 0]
        return float(np.mean(vols)) if vols else 0.0

    def get_voiced_range(self, include_transitions: bool=True) -> tuple[float, float]:
        """Return the app-time range covered by voiced pitch frames."""
        first = None
        last = None
        for pitch in self.data:
            if self.is_voiced_pitch(
                pitch,
                include_transitions=include_transitions,
            ):
                first = pitch
                break
        for pitch in reversed(self.data):
            if self.is_voiced_pitch(
                pitch,
                include_transitions=include_transitions,
            ):
                last = pitch
                break
        if first is None or last is None:
            return None
        frame_dt = self.config.h1 / self.config.sr
        return first.time, last.time + frame_dt
