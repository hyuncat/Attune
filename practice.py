# code for practice mode
from __future__ import annotations
import time

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiPlayer import MidiPlayer
from app_logic.user.ds.Recording import Recording
from app_logic.user.AudioRecorder import AudioRecorder

from ui.score.ScoreViewer import ScoreViewer
from ui.guitarhero.GuitarHero import GuitarHero
from ui.info.ClipDialog import ClipDialog
from ui.time.ScoreTimeMap import ScoreTimeMap


class PracticeTab(QWidget):

    clip_changed = pyqtSignal(object)  # (i0,i1) note-index clip or None; host mirrors it onto the other tab

    def __init__(self, parent=None):
        super().__init__(parent)
        # Practice keeps its OWN independent score (loaded in load_score) so that
        # Perform's analyze / resize / tempo changes never mutate the MIDI shown
        # here. The app keeps the two in sync only for shared edits it pushes in
        # explicitly (instrument selection, score upload).
        self.score_data = ScoreData()
        self.recording = Recording(score_data=self.score_data)
        self.recording.config.pitch_tolerance = 0.3
        self.recording.update_config(self.recording.config)

        self.is_playing = False
        self.is_recording = False

        # While RECORDING, practice mode is driven by the PitchDetector's emitted
        # pitch times (not the wall clock): the audio->pitch buffer only advances
        # its read time when the user's pitch matches the score, so the last
        # emitted time IS the playhead. Render on a separate cadence so batches
        # of detector signals do not also dictate repaint timing.
        self.practice_time = 0.0
        self._RENDER_INTERVAL = 1.0 / 30.0
        self._last_render = 0.0
        self._render_timer = QTimer(self)
        self._render_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._render_timer.setInterval(17)
        self._render_timer.timeout.connect(self._render_practice)

        # score viewer: render only the active instrument's part (default) or the
        # full score. The host (app.py) owns the toggle and pushes it in via
        # set_show_full; this is the panel's render-time cache of it.
        self.viewer_show_full = False

        # barline-anchored app<->Verovio time correspondence: keeps the score
        # cursor on the MIDI/NoteData timeline (the source of truth) instead of
        # Verovio's independently-drifting timemap. Rebuilt on every re-render.
        self._time_map = ScoreTimeMap()

        self.audio_recorder = AudioRecorder(self.recording)

        # shared transport, injected by the host via attach_timekeeping()
        self.wall_clock = None
        self.slider = None
        self.status_bar = None
        self.midi_player = None
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

        self.guitar_hero = GuitarHero(self.recording)
        self.guitar_hero.load_score(self.score_data)

        # score viewer stacked ON TOP of the guitar hero, in a vertical splitter
        # so both are adjustable in height (mirrors the main app's center column).
        self.center_splitter = QSplitter(Qt.Orientation.Vertical)
        self.center_splitter.addWidget(self.score_viewer)
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

    def init_signals(self):
        self.score_viewer.load_finished.connect(self.refresh_score_viewer)
        self.score_viewer.note_clicked.connect(self.on_note_clicked)
        self.score_viewer.trim_requested.connect(self.on_trim_requested)
        self.score_viewer.content_height_changed.connect(self._fit_score_viewer_height)
        self.recording.pitch_detector.pitch_detected.connect(self.pitch_detected)

    # --- SCORE PANE AUTO-FIT (mirrors PerformTab, minus the legend row) ---
    def _on_score_splitter_moved(self, *_):
        """The user dragged the score/guitar-hero handle: their sizing wins from
        now on (auto-fit stops re-asserting the rendered height)."""
        self._score_splitter_user_set = True

    def _fit_score_viewer_height(self, content_height: float):
        """Size the score pane to the freshly rendered system so the default
        view shows the whole line without scrolling."""
        self._last_content_height = content_height
        if self._score_splitter_user_set:
            return
        sizes = self.center_splitter.sizes()
        total = sum(sizes)
        if total <= 0:
            return
        MIN_GUITAR_HERO = 160
        needed = min(int(content_height) + 4, total - MIN_GUITAR_HERO)
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
                           note_panel=None):
        """Inject the shared transport components (owned by the host app) plus
        the shared NotePanel. The panel drives these directly during practice
        playback/recording; the host routes the matching button clicks /
        clock+slider ticks back to us.

        The MIDI player is the panel's OWN (sharing only the synth + clock) so it
        plays *this tab's* independent score — Perform's resize never leaks in."""
        self.wall_clock = wall_clock
        self.slider = slider
        self.status_bar = status_bar
        self.note_panel = note_panel
        self.midi_player = MidiPlayer(midi_synth, wall_clock)
        # the plot is the master view while recording (driven by emitted pitch
        # times, not the clock): keep the shared slider following it.
        self.guitar_hero.plot_moved.connect(self.slider.handle_timer_update)

    def load_score(self, filepath):
        """Load a score from `filepath` into practice mode. Practice parses its
        OWN ScoreData copy (rather than sharing the app's) so Perform's resize /
        tempo edits never mutate the MIDI shown here; the app keeps the active
        instrument in sync via set_active_instrument. The slider range is re-synced
        by the host."""
        self.score_data = ScoreData()
        self.score_data.load(filepath)
        # match the app's default: first real (non-metronome) instrument channel
        self.score_data.active_instrument = next(
            (ch for ch in self.score_data.instruments
             if ch != self.score_data.metronome_channel),
            0,
        )
        self.recording = Recording(score_data=self.score_data)
        self.recording.config.pitch_tolerance = 0.3
        self.recording.update_config(self.recording.config)
        if self.midi_player is not None:
            self.midi_player.load_score(self.score_data)
        self.guitar_hero.load_score(self.score_data)
        self.guitar_hero.load_user(self.recording)
        self.audio_recorder.load_recording(self.recording)
        self.recording.pitch_detector.pitch_detected.connect(self.pitch_detected)
        # render the score into the viewer (no-op if its JS API isn't ready yet;
        # on_score_viewer_loaded re-renders once it is).
        self.refresh_score_viewer()

    # --- SETTINGS-RELATED ---
    def set_active_instrument(self, channel: int):
        """Mirror an instrument change made in the main app. ScoreData is shared,
        so its `active_instrument` is already set by the time we're called; we
        just point our own Recording at the same channel and re-render the views
        (the GuitarHero/ScoreViewer both key off score_data.active_instrument)."""
        self.score_data.active_instrument = channel
        self.recording.active_instrument = channel
        self.recording.update_config(self.recording.config)
        self.refresh_score_viewer()
        self.guitar_hero.update_view_items()

    def set_freq_range(self, fmin: float, fmax: float):
        """Update config with new fmin/fmax. Triggered when user updates range and
        this tab is open"""
        config = self.recording.config
        config.fmin = fmin
        config.fmax = fmax
        self.recording.update_config(config)

    def set_tuning(self, tuning: float):
        """Mirror a tuning change: set our Recording's Config A4 reference (Hz)."""
        self.recording.config.tuning = tuning
        self.recording.update_config(self.recording.config)

    def set_pitch_tolerance(self, tolerance: float):
        """Apply a pitch-tolerance change: set our Recording's Config value, which
        is the semitone slack `_pitch_matches` uses to decide whether a live
        pitch is close enough to let the playhead advance."""
        self.recording.config.pitch_tolerance = tolerance
        self.recording.update_config(self.recording.config)
        # the guitarHero's green band / red ramp track this recording's tolerance
        self.guitar_hero.load_user(self.recording)

    def set_show_full(self, show_full: bool):
        """Host-driven (app.py owns the toggle): show the full score (True) or just
        the active instrument's part (False), then re-render the viewer. Mirrors
        the Perform tab so the full-score view stays consistent across both."""
        self.viewer_show_full = show_full
        self.refresh_score_viewer()

    def transpose(self, semitones: int):
        """Mirror a transpose made in the main app: shift this tab's (independent)
        score by `semitones` half steps and re-render its piano-roll + sheet
        music. Pitch-only, so it stays in sync with the Perform tab's score."""
        if self.score_data is None or self.score_data.score is None:
            return
        self.score_data.transpose(dy=semitones)
        self.guitar_hero.update_view_items()
        self.refresh_score_viewer()

    # --- PLAYBACK / RECORDING (called by the host when this tab is active) ---
    def toggle_playback(self) -> bool:
        """Toggle plain (wall-clock-driven) playback. Returns the new is_playing
        state so the host can update the shared play button icon."""
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
        return self.is_playing

    def start_playback(self):
        self.slider.sync_clip_window(self.score_data)  # clipped -> begin at b0
        t = self.slider.get_time()
        self.is_playing = True
        self.wall_clock.start(t)
        self.midi_player.play(start_time=t)
        self.status_bar.update_status("Practicing...")

    def stop_playback(self):
        if not self.is_playing:
            return
        self.is_playing = False
        self.wall_clock.pause()
        self.midi_player.stop()
        self.status_bar.update_status("")

    def prime_recording(self):
        """Count-in started: warm the mic input device so its first-start
        transient settles before capture (see AudioRecorder.prime)."""
        self.audio_recorder.prime()

    def unprime_recording(self):
        """Count-in cancelled before it armed: release the primed mic stream."""
        if not self.is_recording:
            self.audio_recorder.stop()

    def start_recording(self, start_time: float | None = None):
        """Begin a practice run.
        Advances forward only when user's pitch matches the score.

        `start_time` is the count-in handoff (the head); when None we fall back to
        the cursor / clip start. Practice is pitch-driven, so this is just where
        the playhead is seeded — it then only advances on a correct pitch."""
        self.midi_player.stop()           # stop things we don't want
        self.wall_clock.pause()           # the clock must NOT advance the slider
        self.recording.pitch_detector.block = False
        # seed the pitch-driven playhead at the current slider position (a clipped
        # take begins at the clip start, bounds[0])
        self.slider.sync_clip_window(self.score_data)
        t = self.slider.get_time() if start_time is None else max(0.0, start_time)
        self.practice_time = t
        self._last_render = 0.0
        self.is_recording = True
        # Dots, cursor, and matching all use the gated score-clock address.
        self.guitar_hero.set_live_playhead_offset(0.0)
        self.guitar_hero.set_live(True)  # pitch-dot opacity: fixed absolute dB window
        if self.note_panel is not None:
            self.note_panel.set_live(True)  # trailing-window mode
        self.recording.pitch_detector.run(start_time=t, practice_timing=True)
        # Keep this incremental and off the pitch thread. Starting with the
        # take prevents a historical backfill spike if Vibrato is opened later.
        self.recording.vibrato_detector.run()
        self.audio_recorder.run(start_time=t)
        self._render_timer.start()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self._render_timer.stop()
        self.guitar_hero.set_live(False)  # pitch-dot opacity: remap to the take's range
        if self.note_panel is not None:
            self.note_panel.set_live(False)
        self.wall_clock.pause()
        self.audio_recorder.stop()
        self.recording.pitch_detector.stop()
        self.recording.vibrato_detector.stop()
        self.guitar_hero.set_live_playhead_offset(0.0)

    # --- VIEW DRIVING (called by the host's shared clock/slider dispatch) ---
    def _move_views(self, t: float):
        """Move this tab's score cursor + guitar-hero playhead to time `t`."""
        self.score_data.update_time(t)
        self.score_viewer.set_playback_time(self._score_viewer_time(t))
        self.guitar_hero.move_plot(t)
        if self.note_panel is not None:
            self.note_panel.update_time(t)

    def render_at(self, t: float):
        """Public alias used by the host (e.g. on tab switch) to line this tab's
        views up with a given time."""
        self._move_views(t)

    def on_clock_tick(self, t: float):
        """Called whenever the WallClock ticks forward.
        Only triggers move during playback; recording driven by PitchDetector"""
        if not self.is_playing:
            return
        self._move_views(t)

    def on_slider_changed(self, t: float):
        """Shared slider moved. Only acts when the user is scrubbing (neither
        playing nor recording) — during recording the slider is moved
        programmatically to follow the pitch playhead, so we must ignore those
        echoes here to avoid double-rendering / feedback."""
        if self.is_playing or self.is_recording:
            return
        self._move_views(t)

    def on_slider_end(self):
        """Shared slider reached its end: stop whatever this tab was doing."""
        self.stop_recording()
        self.stop_playback()

    def pitch_detected(self, t: float):
        """Master driver while recording. `t` is the time of the just-emitted
        pitch frame; because the audio->pitch buffer only advances its read time
        on a correct pitch (we block it otherwise), `t` only moves forward when
        the user matches the score. So we use `t` directly as the playhead:
        (1) decide whether this frame matched and block/unblock the buffer for the
        NEXT frame, then (2) publish `t` for the render timer.
        """
        if not self.is_recording:
            return
        # block the buffer (freeze the next emitted time) until the user lands the
        # right note; unblock to let `t` advance. This is the whole mechanism by
        # which the playhead only moves forward on a correct pitch.
        self.recording.pitch_detector.block = not self._pitch_matches(t)
        # `t` is the time of the last emitted pitch -> drive the whole UI from it
        self.practice_time = t

    def _render_practice(self):
        """Coalesce pitch events and pan at ~60 Hz without predicting progress.

        Rebuild dots/notes and update the web score at ~30 Hz; intermediate
        ticks only move the existing scene and cursor. Refresh even when stalled
        because the current pitch slot can change without the time advancing.
        """
        if not self.is_recording:
            return
        t = self.practice_time
        now = time.monotonic()
        refresh = now - self._last_render >= self._RENDER_INTERVAL
        self.score_data.update_time(t)
        if refresh:
            self._last_render = now
            self.score_viewer.set_playback_time(self._score_viewer_time(t))
            if self.note_panel is not None:
                self.note_panel.update_time(t)
        self.guitar_hero.move_plot(t, refresh_items=refresh)

    def _pitch_matches(self, t: float) -> bool:
        """Whether the detected pitch at time `t` should let the playhead advance.

        Advance (True) when there's no note to hold for — a gap, before the first
        note / after the last, or a rest (midi -1) — otherwise the playhead would
        deadlock on a spot with no note. For a real note, advance only when a
        clean, finite pitch lands within the Config pitch_tolerance of the target;
        silence, an unvoiced/too-noisy frame, a NaN/inf candidate, or a wrong
        pitch all hold. (The tolerance follows the side-panel Tolerance control.)
        """
        note_data = self.score_data.note_datas.get(self.score_data.active_instrument)
        # The onset-only lookup drops the final note after its onset and holds
        # the preceding note through rests. Gate on the actual sounding span.
        target = note_data.note_containing(t) if note_data else None
        m = target.midi_num[0] if target is not None else None

        if m is None or m == -1:
            return True

        p = self.recording.pitch_data.read_pitch(t)
        if (
            not self.recording.pitch_data.is_voiced_pitch(p)
            or not p.candidate_pitches
        ):
            self.status_bar.update_status(f"Waiting for note: {target.get_note_name()}…")
            return False

        u = p.value
        # NaN/inf guard: abs(nan - m) <= tol is False anyway, but be explicit so a
        # garbage candidate can never be read as "on pitch".
        if not np.isfinite(u):
            self.status_bar.update_status(f"Waiting for note: {target.get_note_name()}…")
            return False

        tolerance = self.recording.config.pitch_tolerance
        on_pitch = abs(u - m) <= tolerance
        state = "On" if on_pitch else "Off"
        played_name = self.recording.config.get_note_name(u)
        self.status_bar.update_status(
            f"{state}! Detected note: {played_name}, Target note: {target.get_note_name()}"
        )
        return on_pitch

    # --- SCORE VIEWER ---
    def on_note_clicked(self, viewer_sec: float):
        """A note was clicked in the sheet music: jump the transport (slider +
        cursor + GuitarHero) to that note. Scrub-only — ignored while playing or
        recording (practice recording is pitch-driven, the click must not move
        the playhead). Mirrors PerformTab.on_note_clicked."""
        if self.is_playing or self.is_recording:
            return
        self.slider.set_time(self._score_note_start_from_viewer(viewer_sec))
        self._move_views(self.slider.get_time())

    def _score_note_start_from_viewer(self, viewer_t: float) -> float:
        """Map the clicked Verovio note time to app time (ScoreTimeMap), then
        snap to the nearest rendered score note onset. Mirrors
        PerformTab._score_note_start_from_viewer."""
        app_t = self._time_map.app_time(viewer_t, self.score_data)
        starts = self.score_data.note_starts(all_instruments=self.viewer_show_full)
        if not starts:
            return app_t
        return min(starts, key=lambda t: abs(t - app_t))

    def _score_viewer_time(self, t: float) -> float:
        """Wall-clock time -> Verovio cursor time (see ScoreTimeMap.viewer_time)."""
        return self._time_map.viewer_time(t, self.score_data)

    def refresh_score_viewer(self, *_):
        """Re-render the score viewer to match the active instrument's part.
        Rk: Accepts ignored signal args so load_finished(bool) can connect directly.
        """
        if self.score_data is None or self.score_data.score is None:
            return
        channel = None if self.viewer_show_full else self.score_data.active_instrument
        self.score_viewer.load_score(self.score_data, channel=channel)
        # rebuild the barline time map for the freshly laid-out score (async pull
        # of Verovio's measure onsets), then re-assert the clip grey-out so it
        # survives the re-layout (and clears itself when the score isn't clipped).
        self._rebuild_time_map(channel)
        self._refresh_clip_focus()

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
        """Practice score viewer finished loading its JS API; render whatever
        score is currently loaded (no-op if none yet)."""
        self.refresh_score_viewer()

    def cleanup(self):
        """Clean-up any moving parts + clear audio and pitch data. 
        Called before load_score."""
        self.stop_recording()
        self.stop_playback()
        self.practice_time = 0.0
        self._last_render = 0.0
        self.recording.cleanup() # get rid of any stale pitch/audio data

    # --- CLIP (measure-range focus; stored on ScoreData as note indices) ---
    def start_clip_selection(self):
        """Clip menu 'Select measures': arm measure picking in the score viewer.
        Mirrors PerformTab.start_clip_selection."""
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
        inclusive measure INDICES; ScoreData resolves them to notes off its own
        MIDI timeline (drift-free). Mirrors PerformTab._on_trim_selection."""
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
            self.slider.set_time(self.score_data.get_bounds()[0])
        self._refresh_clip_focus()
        self._move_views(self.slider.get_time())

    def sync_clip(self, clip):
        """Mirror a clip made in the OTHER tab onto this score (the clip is global).
        Updates this tab's grey-out + guitar-hero but NOT the shared slider."""
        if clip is None:
            self.score_data.clear_clip()
        else:
            self.score_data.set_clip(*clip)
        self._refresh_clip_focus()
        self.guitar_hero.update_view_items()

    def _refresh_clip_focus(self):
        """(Re)assert (or clear) the score-viewer grey-out from the clip, keyed on
        the clip's measure indices (derived from its notes) so it greys exactly
        the clipped measures regardless of Verovio drift."""
        mr = self.score_data.clip_measure_range()
        if mr is not None:
            self.score_viewer.set_clip_range(mr[0], mr[1])
        else:
            self.score_viewer.clear_clip_range()
