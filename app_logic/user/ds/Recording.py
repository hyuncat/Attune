import numpy as np
import threading
import time
from pathlib import Path
from typing import Literal

from app_logic.user.ds.AudioData import AudioData
from app_logic.user.ds.PitchData import PitchData, Pitch
from app_logic.user.ds.VibratoData import VibratoData
from app_logic.user.ds.TimbreData import TimbreData
from app_logic.midi.ScoreData import ScoreData
from app_logic.Alignment import Alignment
from app_logic.NoteData import NoteData
from app_logic.user.ds.Buffer import Buffer
from app_logic.JsonHandler import JsonHandler
from algorithms.Config import Config

ResizeSpan = Literal["pitch", "note", "onset", "raw"]

class Recording:
    TRAILING_AUDIO_PAD_SEC = 0.2

    def __init__(self, score_data: ScoreData=None, config: Config=None):
        """the user data, associated with a singular recording of a score.
        each recording has its own audio data, pitch data, note data, and alignment
        as well as its own set of algorithms and parameters for processing that data"""
        self.score_data = score_data if score_data is not None else ScoreData()
        # inherit the score's current active instrument so new recordings
        # always target whichever channel was selected when they were created
        self.active_instrument = self.score_data.active_instrument
        self._note_segmentation_signature = None
        self.analysis_notice = ""
        if config is None:
            config = self._default_config_from_score()
        self.update_config(config)

        # algorithms!!
        from algorithms.PitchDetector import PitchDetector
        from algorithms.PitchSmoother import VoicingSmoother, PitchSmoother
        from algorithms.NoteDetector import NoteDetector, TransitionDetector
        from algorithms.MistakeDetector import MistakeDetector
        from algorithms.RepeatSplitter import RepeatSplitter
        from algorithms.VibratoDetector import VibratoDetector
        from algorithms.SpectrumDetector import SpectrumDetector

        self.pitch_detector = PitchDetector(recording=self)
        self.pitch_smoother = PitchSmoother(recording=self)
        self.voicing_smoother = VoicingSmoother(recording=self)
        self.note_detector = NoteDetector(recording=self)
        self.transition_detector = TransitionDetector(recording=self)
        self.mistake_detector = MistakeDetector(recording=self)
        self.repeat_splitter = RepeatSplitter(recording=self)
        self.vibrato_detector = VibratoDetector(recording=self)

        # essential data variables
        self.audio_data = AudioData(config=self.config)
        self.pitch_data = PitchData(config=self.config)
        self.pitches_smoothed = False
        self.vibrato_data = VibratoData(config=self.config)
        self.timbre_data = TimbreData(config=self.config)
        self.spectrum_detector = SpectrumDetector(recording=self)
        self.note_data = NoteData()
        self.alignment: Alignment = Alignment(config=self.config) # filled in later
        self.overridden_mistake_indices = set()

        # Persistence metadata. Folder/library entries point at files until their
        # parent score is active; live takes become dirty as soon as audio is
        # written and are clean again after save_audio().
        self.audio_filepath: Path | None = None
        self.unsaved_changes = False
        self.loaded_from_cache = False

        # queue data structures for real time pitch detection
        self.a2p_queue = Buffer(self.config.sr) #audio-to-pitches

    def _default_config_from_score(self) -> Config:
        """Build the no-sidecar defaults shown by the left Settings panel."""
        config = Config()
        midi_range = self.score_data.note_midi_range(self.active_instrument)
        if midi_range is not None:
            low_midi, high_midi = config.padded_midi_range(*midi_range)
            config.fmin = config.midi_to_freq(low_midi)
            config.fmax = config.midi_to_freq(high_midi)
        return config

    def update_config(self, config: Config=None):
        """initialize the config, either with a provided one or a default one"""
        if config is None:
            self.config = Config()
        else:
            self.config = config

        signature = self.config.note_segmentation_signature()
        previous_signature = self._note_segmentation_signature
        segmentation_changed = (
            previous_signature is not None
            and previous_signature != signature
        )
        self._note_segmentation_signature = signature

        if hasattr(self, 'pitch_detector'):
            self.pitch_detector.load_config(self.config)
        if hasattr(self, 'pitch_smoother'):
            self.pitch_smoother.update_config(self.config)
        if hasattr(self, 'voicing_smoother'):
            self.voicing_smoother.update_config(self.config)
        if hasattr(self, 'note_detector'):
            self.note_detector.update_config(self.config)
        if hasattr(self, 'mistake_detector'):
            self.mistake_detector.update_config(self.config)
        if hasattr(self, 'repeat_splitter'):
            self.repeat_splitter.update_config(self.config)
        if hasattr(self, 'vibrato_detector'):
            self.vibrato_detector.update_config(self.config)
        if hasattr(self, 'spectrum_detector'):
            self.spectrum_detector.load_config(self.config)
        if hasattr(self, 'timbre_data'):
            self.timbre_data.update_config(self.config)
        if segmentation_changed and hasattr(self, 'note_data'):
            changed = [
                name for (name, old), (_, new) in zip(previous_signature, signature)
                if old != new
            ]
            self.reset_analysis()
            self.analysis_notice = (
                "Note analysis cleared after segmentation setting change: "
                + ", ".join(changed)
                + ". Click Analyze to recompute."
            )
    # def on_pitches_detected(self, pitches):
    #     self.pitch_data.data = pitches

    def load_audio(
        self,
        audio_filepath: str,
        score_filepath: str | Path | None = None,
        recording_name: str | None = None,
        load_cache: bool = True,
    ):
        """load in a pre-recorded audio file from a filepath into audio_data.
        Pitch detection is kicked off separately by the caller (app.py runs the
        cleanup + detection together so the views reset as one)."""
        path = Path(audio_filepath)
        self.audio_data.load_data(str(path))
        # The pitch pipeline stamps frame times as i*h1/config.sr and converts YIN
        # lags to frequency with config.sr too. If that rate disagrees with the
        # audio file's REAL rate, BOTH drift: pitch times stretch (the track slides
        # "later and later" over its length) and detected pitches come out mistuned
        # (e.g. a 48kHz file read at 44.1k lands ~1.5 semitones flat). The audio
        # file is the source of truth, so pin config.sr to it — and re-init the
        # detectors + pitch grid — before detection runs.
        self._sync_sr_to_audio()
        self.audio_filepath = path
        self.unsaved_changes = False
        self.loaded_from_cache = False
        self.pitch_data = PitchData(config=self.config)
        self.pitches_smoothed = False
        self.vibrato_data = VibratoData(config=self.config)
        self.timbre_data = TimbreData(config=self.config)
        self.reset_analysis()
        if load_cache:
            self.load_cache(score_filepath=score_filepath, recording_name=recording_name)

    def _sync_sr_to_audio(self):
        """Pin config.sr to the loaded audio's real sample rate and propagate it to
        the detectors/pitch grid, so pitch times + frequencies are computed on the
        audio's own sample grid rather than the 44.1k default. No-op when they
        already match. (A stale cache saved at a different sr overrides this again
        on load — see JsonHandler; such caches must be re-detected to be correct.)"""
        audio_sr = int(getattr(self.audio_data, "sr", 0) or 0)
        if audio_sr and audio_sr != self.config.sr:
            self.config.sr = audio_sr
            self.update_config(self.config)
        if audio_sr and hasattr(self, "a2p_queue"):
            self.a2p_queue.sr = audio_sr

    def save_audio(self, audio_filepath: str | Path):
        """Persist this recording's current audio buffer to disk."""
        self.trim_end(mark_unsaved=False)
        path = Path(audio_filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_data.save_data(str(path))
        self.audio_filepath = path
        self.unsaved_changes = False

    def needs_save(self) -> bool:
        """True when this recording contains live edits/takes not written to disk."""
        return self.audio_data.end_index > 0 and self.unsaved_changes

    def has_pitch_data(self) -> bool:
        return any(p is not None for p in self.pitch_data.data)

    def save_cache(
        self, score_filepath: str | Path=None, recording_name: str=None,
    ) -> bool:
        # A cache saved after loading a pre-timbre sidecar should be upgraded in
        # place even when the user never opened the Timbre panel.
        if self.timbre_data.is_empty() and self.audio_data.end_index >= self.config.w1:
            if self.ensure_timbre():
                thread = self.spectrum_detector.thread
                if thread is not None and thread.is_alive():
                    thread.join()
        return JsonHandler(self).save_cache(
            score_filepath=score_filepath,
            recording_name=recording_name,
        )

    def load_cache(
        self,
        score_filepath: str | Path=None,
        recording_name: str=None,
    ) -> bool:
        loaded = JsonHandler(self).load_cache(
            score_filepath=score_filepath,
            recording_name=recording_name,
        )
        # VibratoData deliberately is not persisted: it derives solely from the
        # pitch track. Rebuilding it against cached notes guarantees it cannot
        # be stale against their current boundaries.
        if loaded:
            self.recompute_vibrato(note_aware=True)
        return loaded

    def cleanup(self):
        """Re-init essential data structures. Called before load_score() in app."""
        self.audio_data = AudioData(config=self.config)
        self.pitch_data = PitchData(config=self.config)
        self.pitches_smoothed = False
        self.vibrato_data = VibratoData(config=self.config)
        self.timbre_data = TimbreData(config=self.config)
        self.reset_analysis()

    def reset_analysis(self):
        """Re-init analysis-derived data structures. Called before re-analyze() in app."""
        self.note_data = NoteData()
        self.alignment = Alignment(config=self.config)
        self.overridden_mistake_indices = set()
        self._pending_override_pairs = set()

    def detect_pitches(self, on_phase=None, verbose: bool = False):
        """run pitch detection, then smoothing, on the current audio data.
        `on_phase(text)`, if given, is called at the start of each stage so a
        caller can surface progress (e.g. a status-bar message)."""
        audio = self.audio_data.read_all()
        self.vibrato_data = VibratoData(config=self.config)
        self.timbre_data = TimbreData(config=self.config)
        self.timbre_data.t_origin = self.audio_data.t_origin
        stop_status = self._phase_status_timer(on_phase, "Detecting pitches")
        try:
            self.pitch_data.data = self.pitch_detector.detect_pitches(
                audio,
                show_progress=verbose,
                verbose=verbose,
            )
            self.pitches_smoothed = False
        finally:
            stop_status()

        # Attune's completed-audio method tracks the ordinary pYIN pitch-bin
        # path, then applies the promoted confidence + global RMS controller.
        # Trackers like Praat can opt out when they already return a final f0
        # contour.
        if getattr(self.pitch_detector, "requires_smoothing", True):
            stop_status = self._phase_status_timer(on_phase, "Tracking pitches")
            try:
                self.pitch_data.data = self.pitch_smoother.smooth(
                    self.pitch_data.data,
                    verbose=verbose,
                )
            finally:
                stop_status()
            stop_status = self._phase_status_timer(on_phase, "Smoothing voicing")
            try:
                self.pitch_data.data = self.voicing_smoother.smooth(
                    self.pitch_data.data,
                    verbose=verbose,
                )
                self.pitches_smoothed = True
            finally:
                stop_status()
        else:
            self.pitches_smoothed = True
        # the offline pass stamps frame times relative to buffer index 0, which
        # represents app-time `t_origin` (NEGATIVE for a Perform runway recorded
        # before the head). Mirror the audio buffer's origin onto the pitch data
        # and lift the frame times onto it so notes/alignment read in app-time.
        origin = self.audio_data.t_origin
        self.pitch_data.t_origin = origin
        if origin:
            for p in self.pitch_data.data:
                if p is not None:
                    p.time += origin

        stop_status = self._phase_status_timer(on_phase, "Detecting vibrato")
        try:
            self.recompute_vibrato(note_aware=False)
        finally:
            stop_status()

    def smooth_pitches(self):
        """Smooth the written live pitch track once before offline analysis."""
        if self.pitches_smoothed:
            return
        if not getattr(self.pitch_detector, "requires_smoothing", True):
            self.pitches_smoothed = True
            return

        n_frames = self.pitch_data.frames_available()
        if n_frames <= 0:
            return
        smoothed = self.pitch_smoother.smooth(
            list(self.pitch_data.data[:n_frames]),
        )
        smoothed = self.voicing_smoother.smooth(smoothed)
        with self.pitch_data.lock:
            self.pitch_data.data[:n_frames] = smoothed
            self.pitch_data.end_index = max(self.pitch_data.end_index, n_frames)
        self.pitches_smoothed = True

    @staticmethod
    def _phase_status_timer(on_phase, label: str):
        """Emit a per-phase elapsed-time status until the current phase finishes."""
        if on_phase is None:
            return lambda: None

        stop_event = threading.Event()
        start = time.perf_counter()

        def message() -> str:
            return f"{label}... {time.perf_counter() - start:.1f}s"

        def tick():
            while not stop_event.wait(0.25):
                on_phase(message())

        on_phase(message())
        thread = threading.Thread(target=tick, daemon=True)
        thread.start()

        def stop():
            stop_event.set()
            thread.join(timeout=0.5)
            on_phase(message())

        return stop

    def update_min_note_length(self) -> float:
        """Sync score-derived segment thresholds to the current score timeline."""
        notes = self.score_data.clipped_note_data(channel=self.active_instrument)
        self.config.set_min_note_length(
            notes.get_min_note_length(default=self.config.get_min_note_length(), clean=True)
        )
        return self.config.min_note_length

    def analyze_notes(self, verbose: bool = False):
        """Run production segmentation, score alignment, and final repeat recovery.

        Pitch detection/smoothing must already be complete. Vibrato is refreshed
        on the final recovered notes, never on score-provided note boundaries.
        """
        self.reset_analysis()
        self.detect_notes()
        self.align_score_and_refine(verbose=verbose)
        self.update_alignment_distances()
        self.recompute_vibrato(note_aware=True)
        return self.note_data

    def detect_notes(self):
        """Run the fixed frame-dense linear-KernelCPD production detector."""
        # Transition flags are legacy/benchmark-derived data persisted with pitch
        # frames. Production segmentation never uses them, and clearing them also
        # keeps downstream vibrato/coloring independent of a stale cached run.
        self.transition_detector.clear_transitions(self.pitch_data.data)
        # resize to the stable pitch span so score-derived note-length heuristics
        # use a tempo close to the take before segmentation runs.
        self.resize_score(to_span="pitch", include_transitions=False)
        self.update_min_note_length()
        # Voicing gaps and change points are the only initial boundary sources.
        self.note_data = self.note_detector.detect_notes(self.pitch_data.data)
        # Replace the provisional whole-track/live vibrato pass. Detected note
        # spans are hard boundaries even when transition-based note segmentation
        # is disabled, so adjacent pitches cannot manufacture edge vibrato.
        self.recompute_vibrato(note_aware=True)

    def recompute_vibrato(self, note_aware: bool = True):
        """Rebuild vibrato from pitch data, optionally bounded by current notes."""
        note_data = self.note_data if note_aware and self.note_data.times else None
        self.vibrato_data = self.vibrato_detector.detect(
            self.pitch_data,
            note_data=note_data,
        )
        return self.vibrato_data


    def detect_mistakes(self, verbose: bool = False):
        # Final gap reconciliation can change pair indices. Carry overrides by
        # stable note IDs across a redraw/re-alignment, including a substitution
        # that temporarily becomes two raw gaps before final reconciliation.
        pending = set(getattr(self, '_pending_override_pairs', set()))
        for i in self.overridden_mistake_indices:
            if 0 <= i < len(self.alignment.pitch_mistakes):
                m = self.alignment.pitch_mistakes[i]
                pending.add(self._alignment_pair_key((m.user_note, m.midi_note)))
        self._pending_override_pairs = pending
        # The MistakeDetector only ever sees the clip's score notes (the full
        # NoteData when unclipped) — see ScoreData.clipped_note_data.
        user_notes = self.note_data
        score_notes = self.score_data.clipped_note_data(
            channel=self.active_instrument
        )
        self.alignment = self.mistake_detector.detect_mistakes(
            user_notes=user_notes,
            score_notes=score_notes,
            verbose=verbose,
        )
        self.overridden_mistake_indices = {
            i for i, m in enumerate(self.alignment.pitch_mistakes)
            if self._alignment_pair_key((m.user_note, m.midi_note)) in pending
        }
        self.alignment.reapply_overrides(self.overridden_mistake_indices)

    @staticmethod
    def _alignment_pair_key(pair):
        return tuple(note.id if note is not None else None for note in pair)

    def reindex_mistakes(self):
        """Finalize compatible gap pairs and refresh mistake/override indices."""
        if self.alignment is None:
            return
        finalized = self.mistake_detector.reconcile_gap_pairs(self.alignment)
        if finalized is not self.alignment:
            self.alignment = finalized
            self.overridden_mistake_indices = {
                i for i, m in enumerate(finalized.pitch_mistakes) if m.overridden
            }
        pending = getattr(self, '_pending_override_pairs', set())
        self.overridden_mistake_indices.update(
            i for i, m in enumerate(self.alignment.pitch_mistakes)
            if self._alignment_pair_key((m.user_note, m.midi_note)) in pending
        )
        self._pending_override_pairs = set()
        self.alignment.reindex_mistakes()
        self.alignment.reapply_overrides(self.overridden_mistake_indices)

    def write_data(self, indata: np.ndarray, start_time: float):
        """write indata to the audio_data at the given start_time
        and append to our queue for pitch processing
        """
        self.audio_data.write_data(indata, start_time)
        self.unsaved_changes = True
        self.a2p_queue.push(indata)

    def write_pitch_data(self, indata: list[Pitch], start_time: float):
        """Write detected pitches to pitch_data at the given start_time."""
        self.pitch_data.write(indata, start_time)
        self.pitches_smoothed = False
        # Vibrato is fit on its own worker (VibratoDetector.run): only wake it so
        # the live pitch loop never blocks on an AM/FM fit. Repeated wakeups
        # coalesce naturally when that lower-priority worker is still busy.
        self.vibrato_detector.notify()

    def get_length(self, raw=True):
        if raw:
            if len(self.note_data.times) > 0:
                return self.note_data.get_length()
            else:
                return self.audio_data.get_length()
        bounds = self.note_data.get_bounds(clean=True)
        if bounds is None:
            return 0.0
        start, end = bounds
        return end - start

    def audio_bounds(self) -> tuple[float, float] | None:
        """App-time bounds for the recording audio currently considered live."""
        if self.audio_data.end_index <= 0:
            return None
        return self.audio_data.get_bounds()

    def audio_end_time(self) -> float:
        """App-time of the recording's logical audio end."""
        return self.audio_data.get_end_time()
    
    def transpose(self, dx: float=None, dy: float=None):
        """Move the WHOLE recorded take by `dx` sec on the app-time line and/or
        `dy` semitones. Audio/pitch frames move via their shared time origin (no
        array copy); notes go through NoteData.transpose; the alignment's
        time index is rebuilt (its pairs reference the same Note objects, so
        only its keys go stale). General primitive; NOTE resize_score does not
        call it — the take is kept fixed and the score is moved onto it instead
        (so audio/pitch stay indexed in their own app-time)."""
        if not dx and not dy:
            return
        if dx:
            self.audio_data.t_origin += dx
            self.pitch_data.t_origin += dx
            self.vibrato_data.t_origin += dx
            self.timbre_data.t_origin += dx
        for p in self.pitch_data.data:
            if p is None:
                continue
            if dx:
                p.time += dx
            if dy:
                if p.value != -1:
                    p.value += dy
                p.candidate_pitches = [(m + dy, prob) for m, prob in p.candidate_pitches]
        self.note_data.transpose(dx=dx, dy=dy)
        self.alignment.refresh()

    def resize_score(
        self,
        to_span: ResizeSpan = "pitch",
        respect_clip: bool = True,
        include_transitions: bool = True,
    ) -> bool:
        """Stretch the score to a recording span and move the score ONTO the take.

        Args:
            to_span:
                "pitch" uses first/last voiced pitch frames. This pre-note pass
                gives score-derived note-length heuristics a tempo close to the
                take before PELT computes its min_size.
                "note" uses the detected voiced note span for the final resize.
                "onset" uses the first/last voiced note ONSETS (start-to-start,
                ignoring the final note's duration), and matches the score
                onset-to-onset too. Useful when the take's last note is held
                longer or shorter than the score's — the onsets still line up.
                "raw" uses the full detected-note span, or full audio span if
                notes have not been detected yet.
            respect_clip:
                If true and a score clip exists, match the clip span to the take
                and anchor the clip's first note onto the take's first voiced
                note. Otherwise match the whole active-instrument score span and
                anchor the score's first note onto the take's. The take never
                moves — it is the app-time truth; only the score is stretched and
                shifted to fit it.
            include_transitions:
                Only used for to_span="pitch". When false, high-slope transition
                frames flagged by detect_transitions() are ignored when finding
                the first/last voiced frame.
        """
        if to_span == "pitch":
            bounds = self.pitch_data.get_voiced_range(
                include_transitions=include_transitions,
            )
        elif to_span == "note":
            bounds = self.note_data.get_bounds(clean=True, use_note_end=True)
        elif to_span == "onset":
            bounds = self.note_data.get_bounds(clean=True, use_note_end=False)
        elif to_span == "raw":
            bounds = self.audio_bounds()
        else:
            raise ValueError(f"unknown resize span: {to_span!r}")

        if not bounds or bounds[1] <= bounds[0]:
            return False

        start, end = bounds
        target_span = end - start
        take_anchor_time = start  # the take's first voiced app-time; the take STAYS here

        sd = self.score_data
        # "onset" fits the score onset-to-onset, so measure the score span the
        # same way (first note start -> last note start); every other mode fits
        # against note ends.
        score_use_note_end = to_span != "onset"
        score_bounds = sd.get_bounds(
            channel=self.active_instrument,
            respect_clip=respect_clip,
            use_note_end=score_use_note_end,
        )
        if score_bounds is None:
            return False
        score_span = score_bounds[1] - score_bounds[0]
        if score_span <= 0:
            return False

        # Stretch the relevant score span to the user's recording span. Keep the
        # bpm FRACTIONAL: rounding to an integer quantizes the scale (~0.8%/bpm at
        # 120 => ~0.5s drift over a 60s piece), which lands the score's endpoint
        # off the take's even when the spans were measured perfectly.
        new_bpm = max(1.0, sd.bpm * score_span / target_span)
        sd.change_tempo(new_bpm)
        # change_tempo rebuilds score notes, so refresh every score-derived
        # segmentation threshold before any subsequent detection/correction.
        self.update_min_note_length()

        score_bounds = sd.get_bounds(
            channel=self.active_instrument,
            respect_clip=respect_clip,
            use_note_end=score_use_note_end,
        )
        if score_bounds is None:
            return False
        # Move the SCORE onto the fixed take: land the (clip) first note ON the
        # take's first voiced note, instead of dragging the take to t=0. The take
        # never moves, so its audio/pitch/note timeline stays the single app-time
        # truth the slider and audio player index against (no negative t_origin,
        # no take-transpose — which is what used to desync the waveform from the
        # cursor).
        sd.transpose(dx=take_anchor_time - score_bounds[0])

        self._update_pitch_distances()
        return True

    def resize_score_to_aligned_onsets(self, respect_clip: bool = True) -> bool:
        """Robustly fit the score timeline from pitch-consistent matched onsets.

        A first/last-onset fit gives either endpoint complete control over the
        tempo.  That is especially damaging when a release or vibrato fragment
        is aligned to the final score note.  Instead, use matches whose played
        pitch agrees with the score, estimate the timeline scale from the median
        slope between every pair of those anchors (a Theil-Sen fit), then use
        the median anchor offset for translation.  A small number of bad edge
        matches can therefore remain local alignment errors rather than moving
        the entire score.
        """
        if self.alignment is None or not self.alignment.pairs:
            return False

        score_notes = (
            self.score_data.clipped_note_data(channel=self.active_instrument)
            if respect_clip
            else self.score_data.note_datas.get(self.active_instrument)
        )
        if score_notes is None or len(score_notes.times) < 2:
            return False

        allowed_ids = {
            note.id for note in score_notes.read(i=0, j=len(score_notes.times))
        }
        matches = [
            (user_note, score_note)
            for user_note, score_note in self.alignment.pairs
            if user_note is not None
            and score_note is not None
            and score_note.id in allowed_ids
        ]
        if len(matches) < 2:
            return False

        matches.sort(key=lambda pair: pair[1].start_time)

        # Substitutions should not decide where the score timeline lands.
        anchors = [
            (user_note, score_note)
            for user_note, score_note in matches
            if abs(user_note.midi_num[0] - score_note.midi_num[0])
            < self.config.pitch_tolerance
        ]
        if len(anchors) < 2:
            return False

        slopes = []
        for i, (left_user, left_score) in enumerate(anchors[:-1]):
            for right_user, right_score in anchors[i + 1:]:
                score_span = right_score.start_time - left_score.start_time
                user_span = right_user.start_time - left_user.start_time
                if score_span > 0 and user_span > 0:
                    slopes.append(user_span / score_span)
        if not slopes:
            return False

        time_scale = float(np.median(slopes))
        if not np.isfinite(time_scale) or time_scale <= 0:
            return False

        sd = self.score_data
        sd.change_tempo(max(1.0, sd.bpm / time_scale))
        # Stabilization can resize the score once per outer correction loop.
        # Each rebuilt timeline must drive the next KernelCPD minimum length.
        self.update_min_note_length()

        # change_tempo rebuilds score Note objects. Resolve every anchor by its
        # stable id and translate by the median residual, not by one endpoint.
        current_notes = sd.note_datas.get(self.active_instrument)
        current_by_id = current_notes.notes_by_id() if current_notes else {}
        offsets = [
            user_note.start_time - current_by_id[score_note.id].start_time
            for user_note, score_note in anchors
            if score_note.id in current_by_id
        ]
        if not offsets:
            return False
        sd.transpose(dx=float(np.median(offsets)))
        self._update_pitch_distances()
        return True

    def refit_score_alignment_once(self, verbose: bool = False) -> bool:
        """Replace the bootstrap tempo with one matched-onset fit and realign.

        Call after initial alignment, before repeat correction. Keep the
        provisional timeline when too few pitch-consistent anchors exist.
        This never changes detected note boundaries or invokes legacy refiners.
        """
        if not self.resize_score_to_aligned_onsets():
            return False
        self.detect_mistakes(verbose=verbose)
        return True

    def align_score_and_refine(self, verbose: bool = False):
        """Align original notes, fit score timing, then recover missing repeats locally."""
        self.resize_score(to_span="onset")
        self.detect_mistakes(verbose=verbose)
        self.refit_score_alignment_once(verbose=verbose)
        self.stabilize_score_alignment(verbose=verbose)
        self.reindex_mistakes()

    def stabilize_score_alignment(self, verbose: bool = False) -> None:
        """Run the selected refinement strategy after the initial score fit.

        The production repeat-only pass preserves the already robust-fitted timeline and checks
        local alternatives using deletion-neutral split fees, preserving missing notes. Legacy checkers
        retain their iterative score-fitting protocol for explicit comparisons.

        For legacy checkers, raw first/last detected onsets can include an insertion or miss a
        deletion, biasing the provisional tempo fit. Once an alignment exists,
        refit from matched note onsets, then re-run the same time-aware edit and
        correction algorithms on that corrected timeline. Repeat only while the
        note boundaries or pairing structure change; a seen-state guard prevents
        oscillation without introducing another tuning parameter.
        """
        if not getattr(self.repeat_splitter, 'refit_score_alignment', True):
            self.repeat_splitter.check_mistakes(verbose=verbose)
            return

        def state() -> tuple:
            note_state = tuple(
                (
                    note.id,
                    note.start_time,
                    note.end_time,
                    tuple(note.midi_num),
                )
                for note in self.note_data.data.values()
            )
            pair_state = tuple(
                (
                    user_note.id if user_note is not None else None,
                    score_note.id if score_note is not None else None,
                )
                for user_note, score_note in self.alignment.pairs
            )
            return note_state, pair_state

        seen = set()
        while True:
            before = state()
            if before in seen:
                return
            seen.add(before)

            # Keep the provisional timeline when there are too few reliable
            # pitch anchors. Repeating the raw endpoint fit here would restore
            # the exact edge sensitivity this stabilization pass avoids.
            self.resize_score_to_aligned_onsets()
            self.detect_mistakes(verbose=verbose)
            self.repeat_splitter.check_mistakes(verbose=verbose)

            after = state()
            if after == before:
                return

    def change_tempo(self, new_bpm: float):
        """Change the tempo of the recording by changing the BPM of the score data, which will automatically update the note timings and pitch distances."""
        self.score_data.change_tempo(new_bpm)
        self.update_min_note_length()
        self._update_pitch_distances()

    def _update_pitch_distances(self):
        """Update the distance to target note for all pitches in the recording, based on the current score data."""
        for note in self.score_data.note_datas[self.active_instrument].data.values():
            if note is None:
                continue
            pitches = self.pitch_data.read(start_time=note.start_time, end_time=note.end_time, clean=True)
            for p in pitches:
                p.live_distance = note.midi_num[0] - p.value

    # default aligned_distance for voiced pitches that no aligned note covers
    # (transitions / slides between notes): 0.0 => green. Don't penalize them.
    TRANSITION_DISTANCE = 0.0

    def update_alignment_distances(self):
        """Recompute every pitch's `aligned_distance` from the current pitch-mistake
        alignment (call after analyze()/detect_mistakes()) with the following coloring:
          - deletion: nothing to color, skipped
          - insertion: all pitches -> inf (red)
          - good/substitution: distance to the *aligned* score note's pitch.
        """
        # reset first so stale post-analysis colors never linger
        for p in self.pitch_data.data:
            if p is not None:
                p.aligned_distance = None

        for pair_index, (user_note, midi_note) in enumerate(self.alignment.pairs):
            if user_note is None:
                continue  # deletion: score note with no user pitches to color
            pitches = self.pitch_data.read(
                start_time=user_note.start_time,
                end_time=user_note.end_time,
                clean=True,
            )
            # transition frames are mid-slide: leave them None (grey), never score
            pitches = [p for p in pitches if not p.is_transition]
            if pair_index in self.alignment.overridden_pair_indices:
                # overridden mistake: the user dismissed it, so force its pitches
                # to distance 0 => green, regardless of the underlying mismatch.
                for p in pitches:
                    p.aligned_distance = 0.0
            elif midi_note is None:
                # insertion: a note that isn't in the score at all -> all red
                for p in pitches:
                    p.aligned_distance = float('inf')
            else:
                target = midi_note.midi_num[0]
                for p in pitches:
                    p.aligned_distance = target - p.value

        # any remaining voiced (drawable) pitch no note covered: color green by
        # default instead of falling back to live coloring -- but skip high-slope
        # transition frames, which stay None (grey) so slides aren't penalized.
        for p in self.pitch_data.data:
            if (p is not None and p.aligned_distance is None and p.value != -1
                    and not p.is_transition):
                p.aligned_distance = self.TRANSITION_DISTANCE
    
    def toggle_mistake_override(self, mistake_index: int):
        # error checking
        if not (0 <= mistake_index < len(self.alignment.pitch_mistakes)):
            return
        # toggle persisted override state for one mistake
        mistake = self.alignment.pitch_mistakes[mistake_index]
        pair_index = mistake.get_pair_index()
        if mistake_index in self.overridden_mistake_indices:
            self.overridden_mistake_indices.remove(mistake_index)
            self.alignment.toggle_overridden_pair_indices(pair_index, False)
            overridden = False
        else:
            self.overridden_mistake_indices.add(mistake_index)
            self.alignment.toggle_overridden_pair_indices(pair_index, True)
            overridden = True

        if 0 <= mistake_index < len(self.alignment.pitch_mistakes):
            self.alignment.pitch_mistakes[mistake_index].set_override(overridden)

        # recolor the affected pitches: overridden notes -> green (distance 0)
        # un-overridden -> back to their real alignment distance
        self.update_alignment_distances()

    def has_analysis(self):
        """Return True if this recording has been analyzed (notes detected => alignment filled in)"""
        return len(self.note_data.times) > 0

    def trim_end(
        self, pad_sec: float = TRAILING_AUDIO_PAD_SEC, mark_unsaved: bool = True,
    ) -> bool:
        """Remove trailing silence after the last voiced pitch.

        Pitch times live in app-time, while AudioData stores samples from its
        own `t_origin`; AudioData.trim_end handles that conversion.
        """
        voiced_range = self.pitch_data.get_voiced_range()
        if voiced_range is None:
            return False

        _, voiced_end = voiced_range
        trim_time = voiced_end + max(0.0, pad_sec)
        audio_changed = self.audio_data.trim_end(trim_time)

        keep_count = 0
        for i, pitch in enumerate(self.pitch_data.data):
            if pitch is None:
                continue
            if pitch.time > trim_time:
                break
            keep_count = i + 1

        pitch_changed = keep_count < len(self.pitch_data.data)
        if pitch_changed:
            with self.pitch_data.lock:
                self.pitch_data.data = self.pitch_data.data[:keep_count]
                self.pitch_data.end_index = min(self.pitch_data.end_index, keep_count)

        self.vibrato_data.trim_to(trim_time)
        self.timbre_data.trim_to(trim_time)

        if audio_changed and mark_unsaved:
            self.unsaved_changes = True
        return audio_changed or pitch_changed

    def ensure_timbre(self, on_done=None) -> bool:
        """Lazily backfill TimbreData from raw audio on a daemon thread.

        Returns True when work was started. Old caches omit the additive
        timbre payload; this path restores it without rerunning pYIN.
        """
        # A newly started live/offline spectrum worker may not have published
        # its first column yet. Do not replace its captured TimbreData target
        # with a competing lazy-backfill object during that brief empty state.
        if self.spectrum_detector.is_running():
            if on_done is not None:
                on_done()
            return False
        if self.audio_data.end_index < self.config.w1:
            if on_done is not None:
                on_done()
            return False
        if not self.timbre_data.is_empty():
            if on_done is not None:
                on_done()
            return False
        audio = self.audio_data.read_all()
        cfg = self.config
        target = TimbreData(config=cfg)
        target.t_origin = self.audio_data.t_origin
        # Publish the target before filling it so TimbreWidget can show columns
        # progressively while an old cache backfills.
        self.timbre_data = target
        available = 1 + (len(audio) - cfg.w1) // cfg.h1
        pitch_frames = self.pitch_data.frames_available()
        n_frames = min(
            available,
            pitch_frames if pitch_frames > 0 else available,
        )
        return self.spectrum_detector.detect_frames_async(
            audio,
            target,
            n_frames=n_frames,
            on_done=on_done,
        )


    # --- JSON LOADING / SAVING WRAPPERS ---
    @staticmethod
    def cache_path_for(audio_filepath: str | Path) -> Path:
        return JsonHandler.cache_path_for(audio_filepath)

    @staticmethod
    def delete_cache_for(audio_filepath: str | Path) -> None:
        JsonHandler.delete_cache_for(audio_filepath)

    def cache_path(self) -> Path | None:
        return self.cache_path_for(self.audio_filepath) if self.audio_filepath else None

    def audio_file_exists(self) -> bool:
        return self.audio_filepath is not None and Path(self.audio_filepath).exists()

    def rename_files(self, new_stem: str) -> Path | None:
        return JsonHandler.rename_recording_files(self, new_stem)
