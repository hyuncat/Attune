# code for performance / analysis mode
from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QMessageBox, QLabel, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiPlayer import MidiPlayer
from app_logic.JsonHandler import JsonHandler
from app_logic.user.ds.Recording import Recording
from app_logic.user.AudioPlayer import AudioPlayer
from app_logic.user.AudioRecorder import AudioRecorder
from app_logic.user.ds.PitchData import PitchData

from ui.Colors import Colors
from ui.score.ScoreViewer import ScoreViewer
from ui.score.ScoreAnnotations import ScoreAnnotations
from ui.guitarhero.GuitarHero import GuitarHero
from ui.info.ClipDialog import ClipDialog
from ui.info.Gradient import VolumeGradient
from ui.info.Legend import Legend
from ui.time.ScoreTimeMap import ScoreTimeMap


class PerformTab(QWidget):

    viewer_ready = pyqtSignal()   # the ScoreViewer's JS API finished loading
    analyzed = pyqtSignal()       # an Analyze pass just resized/aligned the score
    clip_changed = pyqtSignal(object)  # (i0,i1) note-index clip or None; host mirrors it onto the other tab

    def __init__(self, score_data: ScoreData, parent=None):
        super().__init__(parent)
        # important data structures
        self.score_data = score_data
        self.recording: Recording | None = None   # the active take (set by host)

        # audio engines
        self.audio_player = AudioPlayer(None)
        self.audio_recorder = AudioRecorder(self.recording)

        # playback variables
        self.is_playing = False
        self.is_recording = False
        self.user_playback_enabled = True

        # pitch_detectors we've already wired signals for (one per recording), so
        # _wire_detector never double-connects the same detector.
        self._wired_detectors: set = set()
        # score viewer: render only the active instrument's part (default) or the
        # full score. The host (app.py) owns the toggle and pushes it in via
        # set_show_full; this is the panel's render-time cache of it.
        self.viewer_show_full = False
        self.score_color_mode = "pitch"

        # barline-anchored app<->Verovio time correspondence: keeps the score
        # cursor on the MIDI/NoteData timeline (the source of truth) instead of
        # Verovio's independently-drifting timemap. Rebuilt on every re-render.
        self._time_map = ScoreTimeMap()

        # injected via attach_timekeeping()
        self.wall_clock = None
        self.slider = None
        self.status_bar = None
        self.midi_player = None
        self.mistake_widget = None
        self.note_panel = None

        self.init_ui()
        self.init_signals()

    def init_ui(self):
        """Create the panel layout (just the views — the transport is shared)."""
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

        # the score viewer owns its own "Loading..." placeholder until Verovio's
        # JS API is ready (see ScoreViewer).
        self.score_viewer = ScoreViewer()
        self.score_viewer.set_annotation_color_mode(self.score_color_mode)
        self.guitar_hero = GuitarHero(self.recording)
        self.score_panel = self._build_score_panel()

        # score viewer stacked ON TOP of the guitar hero, in a vertical splitter
        # so both are adjustable in height.
        self.center_splitter = QSplitter(Qt.Orientation.Vertical)
        self.center_splitter.addWidget(self.score_panel)
        self.center_splitter.addWidget(self.guitar_hero)
        self.center_splitter.setStretchFactor(0, 1)  # score viewer grows
        self.center_splitter.setStretchFactor(1, 1)  # guitar hero grows
        # pre-load default; every render then auto-fits the pane to the score's
        # full height (_fit_score_viewer_height) until the user drags the handle.
        self.center_splitter.setSizes([180, 520])
        self._score_splitter_user_set = False
        self._last_content_height = 0.0
        self.center_splitter.splitterMoved.connect(self._on_score_splitter_moved)
        self._layout.addWidget(self.center_splitter)
        
        #TODO: move this somewhere where it makes more sense
        gh_margins = self.guitar_hero.layout().contentsMargins()
        self._score_legend_row.setContentsMargins(
            gh_margins.left() + 8, 2, gh_margins.right() + 8, 2)

    def init_signals(self):
        self.score_viewer.load_finished.connect(self.on_score_viewer_loaded)
        self.score_viewer.note_clicked.connect(self.on_note_clicked)
        self.score_viewer.annotation_clicked.connect(self.on_annotation_clicked)
        self.score_viewer.trim_requested.connect(self.on_trim_requested)
        self.score_viewer.content_height_changed.connect(self._fit_score_viewer_height)

    def _build_score_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.score_viewer, stretch=1)
        layout.addLayout(self._build_score_legend_row())
        return panel

    def _build_score_legend_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(8, 2, 8, 2)  # L/R re-matched to GuitarHero's in init_ui
        row.setSpacing(14)
        self._score_legend_row = row

        self._score_legend_items = QHBoxLayout()
        self._score_legend_items.setContentsMargins(0, 0, 0, 0)
        self._score_legend_items.setSpacing(14)
        row.addLayout(self._score_legend_items)
        row.addStretch(1)

        picker = QHBoxLayout()
        picker.setSpacing(6)
        picker.addWidget(QLabel("Colors:"))
        self.score_color_combo = QComboBox()
        self.score_color_combo.addItems(["Pitch", "Timing", "Volume"])
        self.score_color_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.score_color_combo.setStyleSheet(GuitarHero._COMBO_STYLE)
        self.score_color_combo.currentTextChanged.connect(self._on_score_color_mode_changed)
        picker.addWidget(self.score_color_combo)
        row.addLayout(picker)

        self._rebuild_score_legend()
        return row

    def _rebuild_score_legend(self):
        while self._score_legend_items.count():
            item = self._score_legend_items.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

        # swatches mirror what the SCORE paints, so they take its dimmed colors
        # (Colors.SCORE_DIM); the cursor is the one color that stays full.
        current = (Colors.CURRENT_RGB, "current")
        if self.score_color_mode == "volume":
            self._score_legend_items.addWidget(
                Legend.gradient_strip(VolumeGradient(dim=True)))
            self._score_legend_items.addWidget(Legend.swatch(*current))
            return

        if self.score_color_mode == "timing":
            items = [(Colors.mistake_rgb("timing", dim=True), "error"), current]
        else:
            # insertions and deletions share one color and one label: both are a
            # note that shouldn't be where it is, played or missed.
            items = [
                (Colors.mistake_rgb("insertion", dim=True), "wrong note"),
                (Colors.mistake_rgb("substitution", dim=True), "off-pitch"),
                current,
            ]
        for rgb, text in items:
            self._score_legend_items.addWidget(Legend.swatch(rgb, text))

    def _on_score_color_mode_changed(self, text: str):
        label = text.lower()
        if label.startswith("timing"):
            self.score_color_mode = "timing"
        elif label == "volume":
            self.score_color_mode = "volume"
        else:
            self.score_color_mode = "pitch"
        self._rebuild_score_legend()
        self.score_viewer.set_annotation_color_mode(self.score_color_mode)

    # --- SCORE PANE AUTO-FIT ---
    def _on_score_splitter_moved(self, *_):
        """The user dragged the score/guitar-hero handle: their sizing wins from
        now on (auto-fit stops re-asserting the rendered height)."""
        self._score_splitter_user_set = True

    def _fit_score_viewer_height(self, content_height: float):
        """Size the score pane to the freshly rendered system (plus the legend
        row) so the default view shows the whole line without scrolling."""
        self._last_content_height = content_height
        if self._score_splitter_user_set:
            return
        sizes = self.center_splitter.sizes()
        total = sum(sizes)
        if total <= 0:
            return
        MIN_GUITAR_HERO = 160
        needed = int(content_height) + self._score_legend_row.sizeHint().height() + 4
        needed = min(needed, total - MIN_GUITAR_HERO)
        if needed <= 0:
            return
        self.center_splitter.setSizes([needed, total - needed])

    def showEvent(self, event):
        """Re-assert the auto-fit when the tab becomes visible: a load that ran
        while the tab was hidden fitted against stale splitter geometry."""
        super().showEvent(event)
        if self._last_content_height:
            self._fit_score_viewer_height(self._last_content_height)

    def attach_timekeeping(self, wall_clock, slider, status_bar, midi_synth,
                           mistake_widget, note_panel=None):
        """Inject the shared transport (owned by the host) plus the Perform-only
        MistakeWidget and the shared NotePanel. The panel drives the transport
        during playback/recording; the host routes the matching button clicks /
        clock+slider ticks back. The MIDI player is the panel's OWN (sharing
        only the synth + clock) so it plays this tab's score independently of
        the Practice tab's."""
        self.wall_clock = wall_clock
        self.slider = slider
        self.status_bar = status_bar
        self.midi_player = MidiPlayer(midi_synth, wall_clock)
        self.mistake_widget = mistake_widget
        self.note_panel = note_panel
        # keep the shared slider following the plot as it moves
        self.guitar_hero.plot_moved.connect(self.slider.handle_timer_update)
        # mistake list <-> guitar hero highlight/override coupling
        self.mistake_widget.selected.connect(self.on_mistake_selected)
        self.mistake_widget.cleared.connect(self.guitar_hero.clear_highlight)
        self.mistake_widget.override_toggled.connect(self.on_mistake_override_toggled)
        self.mistake_widget.mode_changed.connect(self.guitar_hero.set_mistake_mode)

    # --- HOST-DRIVEN STATE ---
    def set_active_recording(self, rec: Recording):
        """Make `rec` the active take. Wires its pitch detector once, then loads
        its analysis/audio into the views, audio engines and mistake list."""
        self.recording = rec
        if rec.score_data is not self.score_data:
            self.load_score(rec.score_data)
        self._wire_detector(rec)
        self.guitar_hero.load_user(rec)
        self.audio_player.load_audio(rec.audio_data)
        self.audio_recorder.load_recording(rec)
        self._refresh_mistake_widget(rec)
        if rec.analysis_notice and self.status_bar is not None:
            self.status_bar.update_status(rec.analysis_notice)

    def _wire_detector(self, rec: Recording):
        """Each recording owns its own pitch detector; connect each only once."""
        if rec is None or rec.pitch_detector in self._wired_detectors:
            return
        rec.pitch_detector.status_changed.connect(self.status_bar.update_status)
        rec.pitch_detector.detection_finished.connect(self._on_detection_finished)
        self._wired_detectors.add(rec.pitch_detector)

    def set_user_audio_enabled(self, enabled: bool):
        """Mirror the toolbar 'User' checkbox; stop playback now if turning off."""
        self.user_playback_enabled = enabled
        if not enabled:
            self.audio_player.stop()

    def load_score(self, score_data: ScoreData):
        """Point the views + the panel's MIDI player at a freshly loaded score
        (the host re-creates the active recording separately and calls
        set_active_recording)."""
        self.score_data = score_data
        if self.midi_player is not None:
            self.midi_player.load_score(self.score_data)
        self.guitar_hero.load_score(self.score_data)
        self.refresh_score_viewer()

    def cleanup(self):
        """Reset all state - unalive, then cleanup recording and mistake_widget."""
        # just in case!
        self.stop_recording()
        self.stop_playback()
        rec = self.recording
        if rec is not None:
            rec.cleanup()
            self.guitar_hero.load_user(rec)
        if self.mistake_widget is not None:
            self.mistake_widget.clear()
        if self.note_panel is not None:
            self.note_panel.clear()
        self.score_viewer.clear_mistake_annotations()

    def _clear_analysis(self):
        """Clear stale analysis (notes/alignment/mistakes/overrides) and refresh
        the views, but KEEP the recording's audio + pitch data. Used before a
        re-detection: unlike cleanup() (which calls Recording.cleanup() and so
        wipes audio_data back to a 60s zero buffer), this preserves the take we
        just loaded so detection actually runs on the real waveform."""
        self.stop_recording()
        self.stop_playback()
        rec = self.recording
        if rec is not None:
            rec.reset_analysis()
            self.guitar_hero.load_user(rec)
        if self.mistake_widget is not None:
            self.mistake_widget.clear()
        if self.note_panel is not None:
            self.note_panel.clear()
        self.score_viewer.clear_mistake_annotations()

    def set_active_instrument(self, channel: int):
        """Make `channel` the active instrument: wipe analysis-derived data, re-init
        the algorithms from the (unchanged) Config, and re-render the views."""
        if not self._has_recording():
            return
        self.score_data = self.recording.score_data
        self.score_data.active_instrument = channel
        self.recording.active_instrument = channel
        self._clear_analysis()
        self.recording.update_config(self.recording.config)
        self.refresh_score_viewer()

    def set_show_full(self, show_full: bool):
        """Host-driven (app.py owns the toggle): show the full score (True) or just
        the active instrument's part (False), then re-render the viewer."""
        self.viewer_show_full = show_full
        self.refresh_score_viewer()

    def transpose(self, semitones: int):
        """Transpose this tab's score by `semitones` half steps, then re-render
        the piano-roll + sheet-music views (playback reads the shifted MIDI
        live). Pitch-only: timing and the clip are untouched."""
        if self.score_data is None or self.score_data.score is None:
            return
        self.score_data.transpose(dy=semitones)
        self.guitar_hero.update_view_items()
        self.refresh_score_viewer()

    # --- AUDIO / DETECTION ---
    def refresh_audio(self):
        """A new take's raw audio is loaded: make it playable and (re-)run offline
        pitch detection on it in the background."""
        if not self._has_recording():
            return
        self.audio_player.load_audio(self.recording.audio_data)
        if self.recording.has_pitch_data():
            self.guitar_hero.load_user(self.recording)
            self._refresh_mistake_widget(self.recording)
            self._refresh_guitar_hero_now()
            return
        self.detect_pitches()

    def detect_pitches(self):
        """Clear stale analysis, then (re-)run offline pitch detection on the
        active recording's audio in the background. When detection finishes,
        _on_detection_finished loads the fresh pitch data into view."""
        rec = self.recording
        if rec is None or rec.audio_data.end_index <= 0:
            return
        self._clear_analysis()  # clear stale analysis but KEEP the loaded audio
        rec.pitch_data = PitchData(config=rec.config)
        rec.pitches_smoothed = False
        self.guitar_hero.load_user(rec)
        JsonHandler(rec).save_cache()
        self._wire_detector(rec)  # just in case
        rec.pitch_detector.detect_pitches_async()

    def _detection_in_flight(self) -> bool:
        """True while offline pitch detection+smoothing is still running."""
        rec = self.recording
        if rec is None:
            return False
        thread = getattr(rec.pitch_detector, "offline_thread", None)
        return bool(thread and thread.is_alive())

    def _on_detection_finished(self):
        """Offline pitch detection finished (queued onto the main thread): clear
        the status and load the now-ready pitch data into view."""
        sender = self.sender()
        if (self.recording is None
                or (sender is not None and sender is not self.recording.pitch_detector)):
            return
        self.status_bar.update_status("")
        self.guitar_hero.load_user(self.recording)
        self._refresh_guitar_hero_now()
        if self.note_panel is not None:
            self.note_panel.refresh()
        JsonHandler(self.recording).save_cache()

    def _refresh_guitar_hero_now(self):
        """Rebuild visible GuitarHero items and ask Qt to repaint immediately."""
        self.guitar_hero.update_view_items()
        self.guitar_hero.plot.viewport().update()
        self.guitar_hero.update()

    # --- PLAYBACK / RECORDING (called by the host when this tab is active) ---
    def toggle_playback(self) -> bool:
        """Toggle plain playback. Returns the new is_playing state so the host can
        update the shared play button icon."""
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
        return self.is_playing

    def start_playback(self):
        # a clipped take/playback always begins at the clip start (bounds[0])
        self.slider.sync_clip_window(self.score_data)
        t = self.slider.get_time()
        self.is_playing = True
        self.wall_clock.start(t)
        self.midi_player.play(start_time=t)
        if self.user_playback_enabled:
            self.audio_player.play(start_time=t)
        self.status_bar.update_status("Playing...")

    def stop_playback(self):
        if not self.is_playing:
            return
        self.is_playing = False
        self.wall_clock.pause()
        self.midi_player.stop()
        self.audio_player.stop()
        self.status_bar.update_status("")

    def prime_recording(self):
        """Count-in started: warm the mic input device so its first-start
        transient settles before the record point (see AudioRecorder.prime)."""
        self.audio_recorder.prime()

    def unprime_recording(self):
        """Count-in cancelled before it armed: release the primed mic stream."""
        if not self.is_recording:
            self.audio_recorder.stop()

    def start_recording(self, start_time: float | None = None):
        """Start recording; called after app.py's count-in.

        `start_time` is where the count-in left the playhead — one beat BEFORE the
        head (the runway), which can be NEGATIVE when recording from the very start.
        We capture that lead-in anyway: a negative time origin keeps the audio/pitch
        buffers 0-indexed, and the clock floor lets the cursor show the runway. The
        take is realigned to the score's clip start at analysis (Recording.resize)."""
        # a clipped take always begins at the clip start (bounds[0])
        self.slider.sync_clip_window(self.score_data)
        t = self.slider.get_time() if start_time is None else start_time
        # record a one-beat runway even into negative app-time: origin (= the
        # earliest, possibly-negative start) keeps the buffers 0-indexed.
        origin = min(0.0, t)
        self.recording.audio_data.t_origin = origin
        self.recording.pitch_data.t_origin = origin
        self.recording.vibrato_data.t_origin = origin
        self.recording.timbre_data.t_origin = origin
        self.wall_clock.set_floor(origin)
        self.is_recording = True
        cfg = self.recording.config
        # One display-only compensation, derived from the detector window.
        # GuitarHero retains it while move_plot continues emitting the real
        # transport time, so it cannot feed back into the shared slider.
        self.guitar_hero.set_live_playhead_offset(-0.5 * cfg.w1 / cfg.sr)
        self.guitar_hero.set_live(True)
        if self.note_panel is not None:
            self.note_panel.set_live(True)  # trailing-window mode
        self.audio_player.stop()
        # Fit vibrato incrementally on its own worker so it never blocks the
        # pitch->plot loop. Starting with the take avoids a costly historical
        # backfill if the user opens the Vibrato graph midway through.
        self.recording.vibrato_detector.run()
        # Ready the consumer before capture, then start the clock only after
        # PortAudio confirms that the input stream is active. This prevents
        # stream-start time from becoming a permanent plot/playhead offset.
        self.recording.pitch_detector.run(start_time=t)
        self.audio_recorder.run(start_time=t)
        self.wall_clock.start(t)
        self.midi_player.play(start_time=t)  # play whatever audio the user enabled

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.guitar_hero.set_live(False)  # pitch-dot opacity: remap to the take's range
        if self.note_panel is not None:
            self.note_panel.set_live(False)
        self.wall_clock.pause()
        self.wall_clock.set_floor(0.0)  # drop the runway floor for plain playback
        self.audio_recorder.stop()
        self.midi_player.stop()
        # The mic is now closed, so drain every complete analysis frame that it
        # captured before stopping. Previously this discarded the queued tail,
        # leaving the pitch track visibly behind the paused playhead.
        self.recording.pitch_detector.stop(drain=True)
        # Stop the vibrato worker after the drain so its final flush covers the
        # fully-drained pitch track.
        self.recording.vibrato_detector.stop()
        self.guitar_hero.set_live_playhead_offset(0.0)
        self.guitar_hero.update_pitches()
        self.guitar_hero.plot.viewport().update()
        self.status_bar.update_status("")

    # --- VIEW DRIVING (called by the host's shared clock/slider dispatch) ---
    def move_views(self, t: float):
        self.score_data.update_time(t)
        self.score_viewer.set_playback_time(self._score_viewer_time(t))
        self.guitar_hero.move_plot(t)
        if self.note_panel is not None:
            self.note_panel.update_time(t)

    def render_at(self, t: float):
        """Public alias used by the host (e.g. on tab switch) to line this tab's
        views up with a given time."""
        self.move_views(t)

    def on_clock_tick(self, t: float):
        """Shared wall-clock tick: drive the views during playback AND recording.
        Recording drives off the clock (not the slider) so the cursor can show the
        negative pre-head runway — the slider clamps to 0 and would otherwise pin
        the plot there."""
        if not (self.is_playing or self.is_recording):
            return
        self.move_views(t)

    def on_slider_changed(self, t: float):
        """Shared slider moved: move the views only when we're idle (scrubbing).
        During playback/recording the wall clock owns the cursor (see
        on_clock_tick), so ignore the slider's clamped echoes here."""
        if self.is_playing or self.is_recording:
            return
        self.move_views(t)

    # --- ANALYSIS PIPELINE ---
    def analyze(self):
        if not self._has_recording(warn=True) or self._recording_is_empty(warn=True):
            return

        rec = self.recording
        # don't analyze raw/partial pitches: if offline detection+smoothing is
        # still running in the background, the smoothed track (octave errors /
        # noise cleaned up) isn't ready yet. Warn the user to retry once it's
        # finished rather than silently queuing the analysis.
        if self._detection_in_flight():
            QMessageBox.warning(
                self, "Still detecting pitches",
                "Pitch detection is still running. Please wait for it to finish, "
                "then try analyzing again.",
            )
            return
        print("analyzing... ")
        rec.smooth_pitches()
        rec.reset_analysis()  # clear stale notes/alignment/mistakes before recomputing
        rec.detect_notes()

        # Align original notes, then recover missing repeats locally.
        rec.align_score_and_refine()
        rec.update_alignment_distances() # color the user pitches by the final alignment
        rec.trim_end()
        rec.analysis_notice = ""

        # reload every view with the fresh analysis (note/alignment may have been
        # overwritten by the repeat-refinement pass)
        self.guitar_hero.load_alignment(rec.alignment)
        self.guitar_hero.load_user(rec)
        self.guitar_hero.update_view_items()
        self.slider.update_range(score_data=self.score_data, recording=rec)
        # Resize anchored the score onto the take (which never moves). Land the
        # view on the aligned start — the clip start when clipped, else the take's
        # first note — and redraw the dim bands at the new positions.
        start_bounds = self.score_data.get_bounds()
        if start_bounds is not None:
            self.slider.set_time(start_bounds[0])
        self.guitar_hero.update_clip_overlay()
        self._refresh_mistake_widget(rec)
        if self.note_panel is not None:
            self.note_panel.refresh()
        JsonHandler(rec).save_cache()

        # the resize stretched the score to match the take => its BPM/length
        # changed; let the host reflect that in the tempo display.
        self.analyzed.emit()

    def reanalyze_if_analyzed(self):
        """Re-run Analyze only if the take has already been analyzed (used after a
        pitch-tolerance change, which affects the pitch-mistake step)."""
        if self._has_analysis(warn=False):
            self.analyze()

    # --- MISTAKE LIST <-> GUITAR HERO ---
    def _refresh_mistake_widget(self, rec: Recording):
        """Push both mistake lists into the panel: the detected PITCH mistakes
        and the derived TIMING mistakes (early/late/short/long). The dropdown
        picks which is shown. Both lists are built during alignment."""
        self.mistake_widget.load_mistakes(rec.alignment.pitch_mistakes)
        self.mistake_widget.load_timing_mistakes(rec.alignment.timing_mistakes)
        self._refresh_score_mistakes(rec)

    def refresh_mistake_widget(self, rec: Recording | None = None):
        """Public wrapper used by app-level controls that update derived mistakes
        without running a full Analyze pass."""
        rec = rec or self.recording
        if rec is not None:
            self._refresh_mistake_widget(rec)

    def _refresh_score_mistakes(self, rec: Recording | None = None):
        """Send the score viewer the active recording's mistakes keyed by
        drift-free score-note indices. The web layer handles only rendering and
        click popups; it never infers note identity from Verovio seconds."""
        rec = rec or self.recording
        if rec is None or not rec.has_analysis():
            self.score_viewer.clear_mistake_annotations()
            return
        self.score_viewer.set_mistake_annotations(ScoreAnnotations.build(rec))

    def on_mistake_selected(self, idx: int):
        """Triggered after a mistake is clicked in MistakeWidget.
        Calls GuitarHero to highlight the respective note(s).

        Args:
            idx (int): row index into the list currently shown (pitch OR timing).
        """
        if self.recording is None:
            return
        # index the in-view list: timing rows aren't part of alignment.pitch_mistakes
        mistakes = self.mistake_widget.mistakes_in_view()
        if 0 <= idx < len(mistakes):
            self.guitar_hero.highlight_mistake(mistakes[idx])

    def on_mistake_override_toggled(self, idx: int):
        if self.recording is None:
            return
        # idx indexes the in-view list. Timing mistakes aren't part of the pitch
        # alignment (no pitch recolor / pair bookkeeping); just flip their flag
        # and grey the row. Pitch mistakes go through the recording's override
        # machinery (persisted indices + pitch recoloring).
        if self.mistake_widget.is_timing_mode():
            mistakes = self.recording.alignment.timing_mistakes
            if not (0 <= idx < len(mistakes)):
                return
            mistakes[idx].toggle_override()
            self.mistake_widget.refresh_override(idx)
            self.guitar_hero.update_highlight_override(mistakes[idx].is_overridden())
            self._refresh_score_mistakes(self.recording)
            JsonHandler(self.recording).save_cache()
            return
        self.recording.toggle_mistake_override(idx)
        mistake = self.recording.alignment.pitch_mistakes[idx]
        self.mistake_widget.refresh_override(idx)
        self.guitar_hero.update_highlight_override(mistake.is_overridden())
        self.guitar_hero.update_view_items()
        self._refresh_score_mistakes(self.recording)
        JsonHandler(self.recording).save_cache()

    # --- SCORE VIEWER ---
    def on_note_clicked(self, viewer_sec: float):
        """A note was clicked in the sheet music: jump the transport (slider +
        cursor + GuitarHero) to that note. Scrub-only — ignored while playing or
        recording, when the clock owns the cursor."""
        if self.is_playing or self.is_recording:
            return
        self.slider.set_time(self._score_note_start_from_viewer(viewer_sec))
        self.move_views(self.slider.get_time())

    def on_annotation_clicked(self, app_sec: float):
        """A colored score-note annotation or insertion triangle was clicked.
        Unlike ordinary score-note clicks, annotations already carry app-time
        from the drift-free NoteData/alignment mapping."""
        if self.is_playing or self.is_recording:
            return
        self.slider.set_time(float(app_sec))
        self.move_views(self.slider.get_time())

    def _score_note_start_from_viewer(self, viewer_t: float) -> float:
        """Map the clicked Verovio note time to app time (ScoreTimeMap), then
        snap to the nearest rendered score note onset so the shared slider lands
        on the NoteData start time."""
        app_t = self._time_map.app_time(viewer_t, self.score_data)
        starts = self.score_data.note_starts(all_instruments=self.viewer_show_full)
        if not starts:
            return app_t
        return min(starts, key=lambda t: abs(t - app_t))

    def _score_viewer_time(self, t: float) -> float:
        """Wall-clock time -> Verovio cursor time (see ScoreTimeMap.viewer_time)."""
        return self._time_map.viewer_time(t, self.score_data)

    def refresh_score_viewer(self):
        """Re-render the Verovio score viewer.
        Reflects any active_instrument/full-score state changes."""
        if self.score_data is None or self.score_data.score is None:
            return
        channel = None if self.viewer_show_full else self.score_data.active_instrument
        self.score_viewer.load_score(self.score_data, channel=channel)
        self.score_viewer.set_annotation_color_mode(self.score_color_mode)
        # rebuild the barline time map for the freshly laid-out score (async pull
        # of Verovio's measure onsets), then re-assert the clip grey-out so it
        # survives the re-layout (and clears itself when the score isn't clipped).
        self._rebuild_time_map(channel)
        self._refresh_clip_focus()
        self._refresh_score_mistakes()

    def _rebuild_time_map(self, channel):
        """Re-anchor the app<->Verovio time map to the freshly rendered score, so
        the cursor tracks the MIDI/NoteData timeline (see ScoreTimeMap). Pairs the
        score's own measure onsets with Verovio's measure timemap, pulled async."""
        app_onsets = self.score_data.measure_onsets_og(channel)

        def _store(vero_onsets):
            if not vero_onsets or not app_onsets:
                self._time_map.clear()  # fall back to the plain bpm/bpm_og scalar
            else:
                self._time_map.set_anchors(app_onsets, vero_onsets)
            # the clip grey-out is placed through the map; the first
            # _refresh_clip_focus ran before the anchors landed, so re-assert it.
            self._refresh_clip_focus()

        self.score_viewer.get_measure_timemap(_store)

    def on_score_viewer_loaded(self, ok: bool = True):
        """The ScoreViewer's JS API is ready: render whatever score is loaded, and
        let the host know (it loads the demo score on first ready)."""
        self.refresh_score_viewer()
        self.viewer_ready.emit()

    # --- CLIP (measure-range focus; stored on ScoreData as note indices) ---
    def start_clip_selection(self):
        """Clip menu 'Select measures': arm measure picking in the score viewer.
        The user clicks a start + end measure, then right-clicks — the viewer's
        trim_requested lands in on_trim_requested to confirm + apply."""
        self.score_viewer.set_selection_mode(True)
        if self.status_bar is not None:
            self.status_bar.update_status(
                "Select the first and last measures of your desired range, "
                "then right-click to clip.")

    def on_trim_requested(self):
        """Right-click while selecting: pull the picked measures, then confirm
        (async pull; _on_trim_selection shows the popup and applies)."""
        self.score_viewer.get_clip_selection(self._on_trim_selection)

    def _on_trim_selection(self, sel: dict | None):
        """Confirm ("Trim?") and apply the pulled measure selection. `sel` holds
        inclusive measure INDICES (startIdx/endIdx); ScoreData resolves them to
        the exact notes in those measures off its own MIDI timeline, so the clip
        can't drift even where Verovio's rendered timeline runs ahead."""
        if not sel:
            if self.status_bar is not None:
                self.status_bar.update_status("Click a start and an end measure first.")
            return
        if not ClipDialog.ask(self):
            return  # keep selecting
        if self.status_bar is not None:
            self.status_bar.update_status("")
        clip = self.score_data.note_index_range_for_measures(
            sel["startIdx"], sel["endIdx"])
        if clip is None:
            return
        self.score_viewer.set_selection_mode(False)
        self.set_clip(clip, seek=True)
        self.clip_changed.emit(clip)  # mirror onto the other tab (global clip)

    def reset_clip(self):
        """Clip menu 'Reset': drop the clip (mirrored onto the other tab) and
        disarm any in-progress measure selection."""
        self.score_viewer.set_selection_mode(False)
        self.set_clip(None)
        self.clip_changed.emit(None)

    def set_clip(self, clip, seek: bool = False):
        """Apply `clip` ((i0, i1) note indices, or None) to THIS tab's score and
        refresh its slider window / grey-out / views. Used both to clip/reset here
        and by the host to mirror the clip onto the inactive tab."""
        if clip is None:
            self.score_data.clear_clip()
        else:
            self.score_data.set_clip(*clip)
        self.score_viewer.clear_clip_selection()
        self.slider.update_range(score_data=self.score_data, recording=self.recording)
        if seek and self.score_data.is_clipped():
            self.slider.set_time(self.score_data.get_bounds()[0])  # jump the cursor to the clip start
        self._refresh_clip_focus()
        self.move_views(self.slider.get_time())
        if self.recording is not None:
            JsonHandler(self.recording).save_cache()

    def sync_clip(self, clip):
        """Mirror a clip made in the OTHER tab onto this score (the clip is global).
        Updates this tab's grey-out + guitar-hero but NOT the shared slider — the
        active tab and the tab-switch handler own the slider window."""
        if clip is None:
            self.score_data.clear_clip()
        else:
            self.score_data.set_clip(*clip)
        self._refresh_clip_focus()
        self.guitar_hero.update_view_items()
        if self.recording is not None:
            JsonHandler(self.recording).save_cache()

    def _refresh_clip_focus(self):
        """(Re)assert (or clear) the score-viewer grey-out from the clip. Keyed on
        the clip's measure indices (derived from its notes) so it greys exactly
        the clipped measures regardless of Verovio's timeline drift."""
        mr = self.score_data.clip_measure_range()
        if mr is not None:
            self.score_viewer.set_clip_range(mr[0], mr[1])
        else:
            self.score_viewer.clear_clip_range()

    # --- ERROR HANDLING ---
    # the following warn the user of erroneous inputs on warn=True
    def _has_recording(self, warn=False) -> bool:
        """True if there's an active recording, else optionally warn."""
        if self.recording is None:
            if warn:
                QMessageBox.warning(self, "No recording selected", "Please select a recording first.")
            return False
        return True

    def _recording_is_empty(self, warn=False) -> bool:
        """True if the active recording has no audio (optionally warn)."""
        if not self._has_recording(warn=False):
            return False
        if self.recording.audio_data.end_index <= 0:
            if warn:
                QMessageBox.warning(self, "No audio available", "Please record or upload audio first.")
            return True
        return False

    def _has_analysis(self, warn=False) -> bool:
        """True if the active recording has been analyzed (optionally warn)."""
        if not self._has_recording():
            return False
        if not self.recording.has_analysis():
            if warn:
                QMessageBox.warning(self, "No analysis available", "Please analyze the recording first.")
            return False
        return True
