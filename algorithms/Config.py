import numpy as np
from dataclasses import dataclass
from typing import ClassVar

# --- pYIN voicing / volume-gate defaults (shared with the benchmark harness) ---
PRAAT_DEFAULT_VOICING_THRESHOLD = 0.45
PYIN_PRAAT_MIRROR_UNV_THRESH = 1.0 - PRAAT_DEFAULT_VOICING_THRESHOLD
# Promoted on the 26-track, two-per-instrument URMP development ablation.
# Live and completed-audio gates deliberately have separate volume references:
# the former uses a causal running peak, the latter a recording-wide p95 RMS.
PYIN_DEFAULT_UNV_THRESH = 0.985
PYIN_DEFAULT_MIN_VOLUME = 0.03
# Completed-audio recovery: retain >1% voiced confidence. Validated with the
# real-take and paired mistake replay in docs/duration-confidence-recovery-2026-09-27.md.
PYIN_POSTHOC_UNV_THRESH = 0.990
PYIN_POSTHOC_MIN_VOLUME = 0.04
PYIN_GLOBAL_VOLUME_PERCENTILE = 95.0
PYIN_RANGE_PADDING_SEMITONES = 8.0


def guarded_pyin_frequency(frequency_hz: float, *, lower: bool) -> float:
    """Return the legacy guard retained by synthetic vibrato fixtures.

    The promoted pYIN+ detector deliberately uses its exact configured range,
    matching the defensive pitch ablation. Vibrato dataset generation still
    records this older window-sizing policy in historical experiment metadata.
    """
    frequency_hz = float(frequency_hz)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be a positive finite frequency")
    semitone_offset = (
        -PYIN_RANGE_PADDING_SEMITONES
        if lower else PYIN_RANGE_PADDING_SEMITONES
    )
    return frequency_hz * 2.0 ** (semitone_offset / 12.0)


@dataclass
class Config:
    DEFAULT_MIN_NOTE_LENGTH: ClassVar[float] = 0.03
    # These change the detected pitch track itself. They are production-owned,
    # so a sidecar made with an older detector default must be re-detected.
    PITCH_DETECTION_FIELDS: ClassVar[tuple[str, ...]] = (
        "unv_thresh",
        "min_volume",
        "posthoc_unv_thresh",
        "posthoc_min_volume",
    )
    # These settings change note boundaries while reusing the cached pitch
    # track.  They are code-owned (there is currently no per-take UI for them),
    # so a changed default must invalidate cached notes/alignment instead of
    # being silently replaced by an older sidecar value.
    NOTE_SEGMENTATION_FIELDS: ClassVar[tuple[str, ...]] = (
        "pitch_thresh",
        "min_note_length_factor",
        "min_note_length_cap",
        "mistake_correction_min_length",
        "min_silence_duration_ms",
    )
    # These are likewise production-owned rather than per-take preferences.
    # Cached sidecars must not silently restore weights from an older cost model.
    ALIGNMENT_FIELDS: ClassVar[tuple[str, ...]] = (
        "ins_cost",
        "del_cost",
        "alignment_alpha_onset",
        "alignment_alpha_duration",
        "alignment_gamma_pitch",
        "alignment_gamma_time",
        "repeat_split_timing_slack",
    )
    # Vibrato is derived afresh from the cached pitch/note track on every load,
    # and none of these controls is currently a per-take preference.  Keeping
    # them code-owned prevents an old sidecar from silently undoing a changed
    # detector default (for example vib_min_cycles).
    VIBRATO_FIELDS: ClassVar[tuple[str, ...]] = (
        "vib_win_sec",
        "vib_order",
        "vib_min_cycles",
        "vib_min_quality",
        "vib_max_gap_sec",
        "vib2_live_sec",
        "vib2_live_analysis_hz",
        "vib2_live_rate_scan_bins",
        "vib2_curve_sec",
        "vib2_fit_rate_min_hz",
        "vib2_fit_rate_max_hz",
        "vib2_seed_candidate_amplitude_min_semitones",
        "vib2_seed_candidate_amplitude_max_semitones",
        "vib2_seed_amplitude_min_semitones",
        "vib2_seed_amplitude_max_semitones",
        "vib2_fit_amplitude_max_semitones",
        "vib2_rate_scan_bins",
        "vib2_max_rms_ratio",
        "vib2_min_rate_hz",
        "vib2_max_rate_hz",
        "vib2_min_width_cents",
        "vib2_max_nfev",
        "vib2_phase_smoothness",
        "vib2_center_smoothness",
        "vib2_width_smoothness",
        "vib2_hold_edge_values",
        "vib2_onset_taper_seconds",
        "vib2_onset_taper_max_fraction",
        "vib2_onset_taper_floor",
    )

    # note-name spellings indexed by pitch class (midi % 12)
    # get_note_name() picks one; the transpose autocomplete offers both
    SHARP_NOTE_NAMES: ClassVar[list] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    FLAT_NOTE_NAMES: ClassVar[list] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    verbose: bool = False

    # Keep expected notes away from pYIN's search/bin endpoints and allow
    # nearby wrong notes, including in low-register parts.
    PITCH_RANGE_MARGIN_SEMITONES: ClassVar[int] = 4

    @classmethod
    def padded_midi_range(cls, low: float, high: float) -> tuple[float, float]:
        return (low - cls.PITCH_RANGE_MARGIN_SEMITONES,
                high + cls.PITCH_RANGE_MARGIN_SEMITONES)

    # --- PITCH DETECTION PARAMETERS ---
    sr: int = 44100     # sample rate
    w1: int = 1024 * 4  # librosa-compatible pYIN analysis-frame size
    h1: int = 128       # hop size
    fmin: float = 196.0 # Hz
    fmax: float = 3000.0 # Hz
    tuning: float = 440.0  # Hz
    unv_thresh: float = PYIN_DEFAULT_UNV_THRESH  # larger unvoiced probability -> unvoiced

    # Live gain-relative RMS floor against the causal running peak.
    min_volume: float = PYIN_DEFAULT_MIN_VOLUME

    # Completed-audio voicing controller. It uses ordinary pYIN periodicity
    # and a recording-wide PYIN_GLOBAL_VOLUME_PERCENTILE RMS reference.
    posthoc_unv_thresh: float = PYIN_POSTHOC_UNV_THRESH
    posthoc_min_volume: float = PYIN_POSTHOC_MIN_VOLUME

    # --- NOTE DETECTION PARAMETERS ---
    # Smallest idealized pitch step worth a KernelCPD boundary. It determines
    # beta = 0.5 * min_segment_frames * pitch_thresh**2.
    pitch_thresh: float = 0.60
    # Score-derived shortest-note duration. Recording.update_min_note_length()
    # refreshes this after every score-to-take fit and before correction.
    min_note_length: float = 0.03  # seconds
    # Initial detection admits short erroneous notes even in a slow score.
    # min(200 ms, half the shortest tempo-adjusted score note): the larger cap
    # resists expressive fragments in slow passages while fast scores retain
    # their shorter minimum. Zero cap disables the ceiling for comparisons.
    min_note_length_factor: float = 0.50
    min_note_length_cap: float = 0.200  # seconds; 0 means uncapped
    # Note refinement caps the detector's score-relative minimum at this
    # correction-only absolute threshold.
    mistake_correction_min_length: float = 0.15  # seconds
    # Width of the decoded-silence majority window. At sr=44100 and h1=128,
    # 40 ms maps to 13 frames, requiring a majority of seven unvoiced frames.
    min_silence_duration_ms: float = 40.0

    # --- STRING EDIT PARAMETERS ---
    ins_cost: float = 5
    del_cost: float = 5
    # Time-aware string-edit weights:
    #   C_time = alpha_onset*|onset error| + alpha_duration*|duration error|
    #   C_pair = gamma_pitch*|pitch error| + gamma_time*C_time
    #   C_ins  = ins_cost + gamma_time*user duration
    #   C_del  = del_cost + gamma_time*score duration
    # alpha_onset + alpha_duration must equal 1. The gamma weights convert the
    # raw semitone and second-valued terms into a common edit-cost scale. Gap
    # operations use the full unmatched duration because no paired onset exists
    # with which to blend it.
    # Pitch weight 4 promoted from mistake2: tied-best event F1 with only one
    # cost changed (82.83% vs 80.39%; six fewer FP, unchanged TP/FN). Confirmed
    # with the current note defaults and repeat correction on the same corpus.
    # Other costs retain the runner-v2 seed-0 defaults. Robust score-time fitting
    # handles the global onset placement before string editing; duration then
    # supplies the local temporal evidence without letting expressive onset
    # shifts manufacture insertion/deletion cascades.
    alignment_alpha_onset: float = 0.0
    alignment_alpha_duration: float = 1.0
    alignment_gamma_pitch: float = 4.0
    # Legacy collapsed-repeat option retained for old benchmark configurations.
    # Current recovery uses deletion-neutral fees instead of this timing gate.
    repeat_split_timing_slack: float = 0.100
    alignment_gamma_time: float = 1.0
    pitch_tolerance: float = 1   # semitones
    timing_tolerance: float = 0.25  # sec

    # --- VIBRATO ---
    # Detector 1's centered LS-Prony window.
    vib_win_sec: float = 0.4      # sliding analysis window (McLeod ch. 9)
    # Frame-dense is an invariant, not a persisted per-take option: old caches
    # may contain the former value 4 and must not silently restore decimation.
    vib_stride: ClassVar[int] = 1
    vib_order: int = 2            # linear-prediction order (2 = one real sinusoid)
    # Detector 1 only: minimum fitted/observed cycles for its windowed model.
    vib_min_cycles: float = 1.0
    # Detector 1's minimum explained-energy quality.
    vib_min_quality: float = 0.3  # below this, report continuous 0 Hz / 0 cents
    # Detector 1 and Detector 2's provisional live fallback bridge unvoiced
    # dropouts up to this duration before note boundaries exist. Detector 2's
    # finalized note-aware fit has no separate gap threshold: note
    # detection/correction exclusively owns its hard boundaries.
    vib_max_gap_sec: float = 0.06

    # Detector 2 fits one smooth, time-varying sinusoidal model per completed
    # note. Live analysis uses McLeod's 0.4-second analysis duration.
    vib2_live_sec: float = 0.4
    # The pitch grid is ~344 Hz, but speed/width envelopes evolve much more
    # slowly than the carrier oscillation. Evaluate causal anchors at this rate
    # and interpolate them back onto every pitch-frame grid point. The worker
    # always skips stale pending anchors and fits the newest available history.
    vib2_live_analysis_hz: float = 20.0
    # Live-only fallback seed resolution. Accepted causal fits warm-start the
    # next anchor; this global constant-rate grid is used for initial
    # acquisition, periodic reacquisition, and recovery after a weak fit. The
    # nonlinear refinement still uses the original pitch samples. Offline
    # retains vib2_rate_scan_bins below.
    vib2_live_rate_scan_bins: int = 32
    # Interior spacing of the cubic B-spline bases for center and width. Speed
    # uses three positive Bezier controls over the whole note/live history.
    # This is curve flexibility, not a sliding analysis window. The earlier
    # 0.4 s spacing could not follow intonation drift inside a note, so the
    # drift landed in the residual the sinusoid is fitted to and cost extent
    # accuracy; 0.1 s raised changing-profile extent F1 from 0.443 to 0.501
    # and cut straight-note false alarms, at roughly half the offline
    # throughput. Live analysis is unaffected: the causal basis spans the whole
    # trailing history with no interior knots.
    vib2_curve_sec: float = 0.1
    # Research-supported musical reach envelope used by both the constant-rate
    # seed scan and refined controls. Typical vibrato is concentrated around
    # 4--8 Hz, while 3--10 Hz retains deliberately slow and fast reach cases
    # without admitting the 20--30 Hz tracking-noise failures observed in the
    # unconstrained model.
    vib2_fit_rate_min_hz: float = 3.0
    vib2_fit_rate_max_hz: float = 10.0
    # Candidate admission is deliberately wider than the initialized model:
    # pYIN smoothing and the simultaneously fitted center attenuate the cheap
    # constant-sine coefficient on real audio. Once admitted, the amplitude
    # seed is clamped to the benchmark's supported +/-0.15--1.00-semitone
    # one-sided range. This is not a forced minimum in the final linear solve:
    # straight notes must remain free to fit zero amplitude. The refined-fit
    # maximum has a small reach margin and is applied to representative
    # amplitude rather than harmless local spline-edge overshoot.
    vib2_seed_candidate_amplitude_min_semitones: float = 0.05
    vib2_seed_candidate_amplitude_max_semitones: float = 1.25
    vib2_seed_amplitude_min_semitones: float = 0.15
    vib2_seed_amplitude_max_semitones: float = 1.0
    vib2_fit_amplitude_max_semitones: float = 1.25
    vib2_rate_scan_bins: int = 100
    # Accept Detector 2 only when its final RMS residual is less than this
    # fraction of the RMS variation left after removing the pitch center.
    vib2_max_rms_ratio: float = 0.75
    # Reject a whole-note fit when its median characteristic is below either
    # floor. Width is Attune's full peak-to-peak measure, so 10 cents is a
    # +/-0.05-semitone one-sided amplitude. These defaults were selected on
    # the preliminary average and native stress CocoChorales suites.
    vib2_min_rate_hz: float = 3.0
    vib2_max_rate_hz: float = 10.0
    vib2_min_width_cents: float = 10.0
    vib2_max_nfev: int = 24
    vib2_phase_smoothness: float = 20.0
    vib2_center_smoothness: float = 200.0
    vib2_width_smoothness: float = 1.0
    # Offline fits always taper quality across the weakly supported first and
    # last two cycles.  This switch controls only whether rate/width values are
    # also held at the nearest interior value.  It stays enabled in production
    # while the fitted-edge policy is evaluated as a benchmark ablation.
    vib2_hold_edge_values: bool = True
    # Optional completed-note onset confidence taper. It is disabled in
    # production until the benchmark accepts it. When enabled, its span is
    # capped to this fraction of the note and every finite pitch retains at
    # least ``floor`` confidence; this never hard-drops a voiced onset frame.
    vib2_onset_taper_seconds: float = 0.0
    vib2_onset_taper_max_fraction: float = 0.25
    vib2_onset_taper_floor: float = 0.25

    # --- TIMBRE (Hann-windowed FFT on a uniform musical display axis) ---
    # Frame-dense is an invariant shared with pYIN, not a persisted per-take
    # option. Legacy cqt_* range names remain for sidecar compatibility.
    spectrum_stride: ClassVar[int] = 1
    cqt_midi_min: int = 36        # C2
    cqt_midi_max: int = 108       # C8 (inclusive)

    # --- note-name helper ---
    @staticmethod
    def get_note_name(midi_num: float | None, prefer_flats: bool = False) -> str:
        """Convert a MIDI number to a letter name like C4, F#3 (or Bb3 with
        prefer_flats). Note naming is tuning-independent, so this is a static
        method that both Pitch and Note route through. Rests/unvoiced (None or a
        negative midi_num) render as an em dash."""
        if midi_num is None or midi_num < 0:
            return "—"
        n = int(round(midi_num))
        names = Config.FLAT_NOTE_NAMES if prefer_flats else Config.SHARP_NOTE_NAMES
        return f"{names[n % 12]}{n // 12 - 1}"

    def get_min_note_length(self, type: str="sec"):
        """Return the minimum note length in seconds or pitch frames (h1/sr grid)."""
        if type == "sec":
            return self.min_note_length
        elif type == "frames":
            fr = self.sr / self.h1
            return max(1, int(np.ceil(self.min_note_length * fr)))
        else:
            raise ValueError(f"Invalid type {type} for get_min_note_length()")

    def set_min_note_length(self, sec: float):
        """Set the central shortest-note estimate in seconds."""
        if sec is None or sec <= 0:
            sec = self.DEFAULT_MIN_NOTE_LENGTH
        self.min_note_length = float(sec)

    def set_min_note_length_from_notedata(self, note_data) -> float:
        """Compatibility helper for older benchmark/notebook callers."""
        if note_data is None:
            self.set_min_note_length(self.DEFAULT_MIN_NOTE_LENGTH)
            return self.min_note_length
        try:
            sec = note_data.get_min_note_length(
                default=self.DEFAULT_MIN_NOTE_LENGTH,
                clean=True,
            )
        except AttributeError:
            sec = self.DEFAULT_MIN_NOTE_LENGTH
        self.set_min_note_length(sec)
        return self.min_note_length

    def note_segmentation_config(self) -> dict[str, float | int]:
        """The code-owned settings that determine detected note boundaries."""
        return {
            name: getattr(self, name)
            for name in self.NOTE_SEGMENTATION_FIELDS
        }

    def note_segmentation_signature(self) -> tuple:
        """Stable comparison key used to invalidate stale note analysis."""
        return tuple(
            (name, getattr(self, name))
            for name in self.NOTE_SEGMENTATION_FIELDS
        )

    def min_note_seconds(self, factor: float = 1.0) -> float:
        return max(0.0, float(self.min_note_length) * float(factor))

    def min_note_pitch_frames(self, factor: float = 1.0) -> int:
        frame_rate = self.sr / self.h1
        return max(1, int(np.ceil(self.min_note_seconds(factor) * frame_rate)))

    def note_detection_min_seconds(self) -> float:
        """Initial segment/run minimum; correction has its own duration rule."""
        seconds = self.min_note_seconds(self.min_note_length_factor)
        return min(seconds, self.min_note_length_cap) if self.min_note_length_cap > 0 else seconds

    def note_detection_min_frames(self) -> int:
        return max(1, int(np.ceil(self.note_detection_min_seconds() * self.sr / self.h1)))

    # --- pitch conversion methods ---
    def freq_to_midi(self, freq: float) -> float:
        """
        Convert a frequency to a MIDI note number.
        """
        if freq <= 0:
            # print("bad freq")
            return(-1)
        return 69 + 12 * np.log2(freq / self.tuning)

    def midi_to_freq(self, midi_num: float) -> float:
        """
        Convert a MIDI note number to frequency.
        """
        return self.tuning * (2 ** ((midi_num - 69) / 12))


    def __repr__(self):
        return (f"Config\n---\n   sr={self.sr}, w1={self.w1}, h1={self.h1}, fmin={self.fmin}, fmax={self.fmax}, tuning={self.tuning}, unv_thresh={self.unv_thresh}, min_volume={self.min_volume}, posthoc_unv_thresh={self.posthoc_unv_thresh}, posthoc_min_volume={self.posthoc_min_volume},\n"
                f"   pitch_thresh={self.pitch_thresh}, min_note_length={self.min_note_length:.3f}, "
                f"min_note_length_factor={self.min_note_length_factor}, "
                f"min_note_length_cap={self.min_note_length_cap}, "
                f"mistake_correction_min_length={self.mistake_correction_min_length}, "
                f"min_silence_duration_ms={self.min_silence_duration_ms},\n"
                f"   ins_cost={self.ins_cost}, del_cost={self.del_cost}, "
                f"alignment_alpha_onset={self.alignment_alpha_onset}, alignment_alpha_duration={self.alignment_alpha_duration}, "
                f"alignment_gamma_pitch={self.alignment_gamma_pitch}, alignment_gamma_time={self.alignment_gamma_time}, "
                f"pitch_tolerance={self.pitch_tolerance}, timing_tolerance={self.timing_tolerance}")
