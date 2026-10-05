import os
# force QtWebEngine's chromium GPU process to use software rasterization
# eg, run on the CPU. set before importing
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-gpu --disable-gpu-compositing --disable-software-rasterizer=false",
)

from pathlib import Path
import sys
import threading
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSplitter, QMessageBox, QTabWidget, QFileDialog
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QKeySequence
import qdarktheme

from ui.time.Slider import Slider
from ui.time.WallClock import WallClock
from ui.time.CountdownTimer import CountdownTimer

from ui.info.Toolbar import Toolbar
from ui.info.StatusBar import StatusBar
from ui.info.RecordingTree import RecordingTree
from ui.info.SettingsWidget import SettingsWidget
from ui.info.MistakeWidget import MistakeWidget
from ui.info.ToleranceWidget import ToleranceWidget
from ui.note.NotePanel import NotePanel

# app logic imports
from app_logic.user.ds.Recording import Recording
from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiSynth import MidiSynth
from app_logic.JsonHandler import JsonHandler

# the two center "mode" tabs, each in its own file
from perform import PerformTab
from practice import PracticeTab

class Attune(QMainWindow):
    """Each Attune instance is associated with a single score and multiple
    recordings associated to that score, each with its own analysis and settings. 
    This top level class controls shared ScoreData, manages signals + timekeeping
    + settings. 
    
    Owns two center tabs:
        - PerformTab: time-invariant performance mode, contains mistake analysis
        - PracticeTab: live pitch-matching, blocks until you play in tune
    """
    def __init__(self):
        super().__init__()
        self.score_data = ScoreData()
        self.base_score_data = self.score_data
        self.recordings: dict[str, Recording] = {}  # name -> Recording
        self.active_recording: Recording | None = None
        self.active_recording_name: str | None = None
        self.current_score_path: Path | None = None
        self._suppress_tab_change = False
        self._current_tab_index = 0
        # Folder imports can contain many recordings; keep offline pitch work
        # bounded so a score switch does not spawn one worker per file.
        self._pitch_detection_semaphore = threading.BoundedSemaphore(2)

        self.DEMO_FOLDER_PATH = str(Path(__file__).resolve().parent / "resources" / "demo")

        # shared transport engines. The synth + wall clock are shared by both
        # tabs; each tab builds its OWN MidiPlayer on them (in attach_timekeeping)
        # so it plays its own independent score.
        # 30 Hz matches CountdownTimer.FPS: recording/playback scroll at the same
        # rate as the count-in, and live pitch dots land within a tick (~33 ms)
        # of detection instead of sitting invisible for up to 100 ms at 10 Hz.
        self.wall_clock = WallClock(hz=30)
        self.SOUNDFONT = "resources/MuseScore_General.sf3"
        self.midi_synth = MidiSynth(self.SOUNDFONT)
        # the metronome count-in is shared by both tabs
        self.is_counting_in = False
        # the head (cursor pos) a count-in is scrolling toward; used to snap the
        # plot back if the user un-arms recording mid-count-in.
        self._countin_head = 0.0
        # full-score vs active-part view toggle, shared across both tabs: app.py
        # owns it (single source of truth) and pushes it into each tab via
        # set_show_full, like the other side-panel settings.
        self.viewer_show_full = False

        # IMPORTANT COMPONENTS
        # left column
        self.recordings_tree = RecordingTree(self.recordings)
        self.settings_widget = SettingsWidget()
        # center tabs
        self.perform_tab = PerformTab(self.score_data)
        self.practice_tab = PracticeTab()  # owns its own independent score
        # right column
        self.mistake_widget = MistakeWidget()
        self.tolerance_widget = ToleranceWidget()
        self.note_panel = NotePanel()

        self.init_ui()
        self.init_signals()

    def init_ui(self):
        """
        Initialize all UI components for the window.
            - main window (title + geom)
            - splitter
                - recordings file tree + instrument panel (left)
                - center tabs: Perform | Practice
                - mistake widget + tolerance panel (right)
            - transport row (play / record / time / slider / analyze)
            - status bar (+ countdown timer)
            - toolbar
            - dialogs (settings, clipper)
        """
        self.setWindowTitle("Attune")
        self.setGeometry(100, 100, 1300, 800)

        # --- CENTRAL LAYOUT WIDGET ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self._layout = QVBoxLayout(self.central_widget)

        # --- splitter: RECORDINGS TREE | [PERFORM / PRACTICE] | MISTAKES ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # ------> LEFT COLUMN
        # recordings tree <-splitter-> settings panel (instrument/range/tuning/
        # transpose, internally scrollable).
        self.left_column = QSplitter(Qt.Orientation.Vertical)
        self.left_column.addWidget(self.recordings_tree)
        self.left_column.addWidget(self.settings_widget)
        self.left_column.setStretchFactor(0, 1)  # tree takes the slack
        self.left_column.setStretchFactor(1, 0)  # settings panel fixed-ish
        self.left_column.setMinimumWidth(180)
        self.left_column.setMaximumWidth(320)

        # ------> CENTER COLUMN
        # perform tab - record and analyze mistakes
        # practice tab - blocking mode for off-pitches
        self.center_tabs = QTabWidget()
        self.center_tabs.addTab(self.perform_tab, "Perform")
        self.center_tabs.addTab(self.practice_tab, "Practice")

        # ------> RIGHT COLUMN
        # mistake widget + tolerance slider on top; the per-note inspector
        # (NotePanel) below, behind a splitter so the two share the column
        # height adjustably.
        self.right_top = QWidget()
        self.right_top_layout = QVBoxLayout(self.right_top)
        self.right_top_layout.setContentsMargins(0, 0, 0, 0)
        self.right_top_layout.addWidget(self.mistake_widget)
        self.right_top_layout.addWidget(self.tolerance_widget)
        self.right_top_layout.setStretch(0, 1)  # yes MistakeWidget stretch!
        self.right_top_layout.setStretch(1, 0)  # no ToleranceWidget stretch
        self.tolerance_widget.setFixedHeight(self.tolerance_widget.sizeHint().height())
        self.right_column = QSplitter(Qt.Orientation.Vertical)
        self.right_column.addWidget(self.right_top)
        self.right_column.addWidget(self.note_panel)
        self.right_column.setStretchFactor(0, 1)  # mistakes take the slack
        self.right_column.setStretchFactor(1, 0)  # note panel keeps its size
        self.right_column.setSizes([460, 300])
        # keep the column as narrow as the fixed-width MistakeWidget
        self.right_column.setMinimumWidth(self.mistake_widget.minimumWidth())
        self.right_column.setMaximumWidth(self.mistake_widget.maximumWidth())

        # add the widgets
        self.splitter.addWidget(self.left_column)
        self.splitter.addWidget(self.center_tabs)
        self.splitter.addWidget(self.right_column)
        # set behavior controls
        self.splitter.setStretchFactor(0, 0)  # recordings tree is fixed-ish
        self.splitter.setStretchFactor(1, 1)  # center column grows
        self.splitter.setStretchFactor(2, 0)  # mistake widget is fixed-ish
        # default split: side panels start compact (still user-resizable).
        self.splitter.setSizes([240, 764, 296])

        self._layout.addWidget(self.splitter)

        # --- INIT TRANSPORT ROW ---
        self.init_slider_layout()
        # --- UTILITIES ---
        self.status_bar = StatusBar(name="untitled_recording")
        self.setStatusBar(self.status_bar)
        self.countdown_timer = CountdownTimer(self.status_bar, midi_synth=self.midi_synth)
        self.toolbar = Toolbar(score_data=self.score_data)
        self.addToolBar(self.toolbar)
        self.init_shortcuts()

        # hand both tabs the shared transport now that it all exists. The Perform
        # tab also drives the (Perform-only) MistakeWidget.
        self.perform_tab.attach_timekeeping(
            wall_clock=self.wall_clock,
            slider=self.slider,
            status_bar=self.status_bar,
            midi_synth=self.midi_synth,
            mistake_widget=self.mistake_widget,
            note_panel=self.note_panel,
        )
        self.practice_tab.attach_timekeeping(
            wall_clock=self.wall_clock,
            slider=self.slider,
            status_bar=self.status_bar,
            midi_synth=self.midi_synth,
            note_panel=self.note_panel,
        )
        # --- DIALOGS ---
        self.show()  # run the show :)

    def init_shortcuts(self):
        """Window-level keyboard shortcuts, dispatched to the active tab like the
        transport buttons they stand in for."""
        self._add_shortcut(QKeySequence.StandardKey.Save, self.save_active_recording)
        # Space arms/stops recording from anywhere in the window. A text field
        # being edited (renaming a take, the tuning box) takes the key back for
        # itself — Qt offers it the shortcut first — so typing a space still works.
        self._add_shortcut(QKeySequence(Qt.Key.Key_Space), self.toggle_recording)

    def _add_shortcut(self, key, slot):
        action = QAction(self)
        action.setShortcut(key)
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        action.triggered.connect(slot)
        self.addAction(action)

    def init_slider_layout(self):
        """Initialize the shared transport row: play/pause, record, time label,
        slider, and the (Perform-only) Analyze button."""
        self.transport_widget = QWidget()
        self.slider_layout = QHBoxLayout(self.transport_widget)
        self.slider_layout.setContentsMargins(0, 0, 0, 0)

        # get the play/pause button icons
        app_directory = os.path.dirname(__file__)
        play_filepath = os.path.join(app_directory, 'resources', 'icons', 'play.png')
        pause_filepath = os.path.join(app_directory, 'resources', 'icons', 'pause.png')
        record_filepath = os.path.join(app_directory, 'resources', 'icons', 'record.png')

        self.play_icon = QIcon(play_filepath)
        self.pause_icon = QIcon(pause_filepath)
        self.record_icon = QIcon(record_filepath)

        # play button
        self.play_button = QPushButton()
        self.play_button.setIcon(self.play_icon)
        self.play_button.setFixedSize(QSize(26, 26))
        self.play_button.clicked.connect(self.toggle_playback)
        self.slider_layout.addWidget(self.play_button)

        # record button
        self.record_button = QPushButton()
        self.record_button.setIcon(self.record_icon)
        self.record_button.setFixedSize(QSize(26, 26))
        self.record_button.clicked.connect(self.toggle_recording)
        self.slider_layout.addWidget(self.record_button)

        # time label (current/total)
        self.time_label = QLabel("00:00.0 / 00:00.0")
        self.time_label.setMinimumWidth(100)
        self.slider_layout.addWidget(self.time_label)

        # the slider
        self.slider = Slider(self.wall_clock)
        self.slider_layout.addWidget(self.slider)

        # analyze button (Perform-only; hidden on the Practice tab)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.perform_tab.analyze)
        self.slider_layout.addWidget(self.analyze_button)

        self._layout.addWidget(self.transport_widget)

    def init_signals(self):
        """Connect all signals and slots for UI / app logic."""
        # --- TOOLBAR SIGNALS ---
        # score/audio uploaded
        self.toolbar.score_uploaded.connect(self.load_score)
        self.toolbar.audio_uploaded.connect(self.load_audio)
        self.toolbar.folder_uploaded.connect(self.load_folder)
        self.toolbar.save_recording_requested.connect(self.save_active_recording)
        # clip stuff
        self.toolbar.select_measures_requested.connect(self.on_select_measures)
        self.toolbar.clip_reset.connect(self.on_clip_reset)
        # the clip is GLOBAL: a clip set in one tab mirrors onto the other (note
        # indices are tab-independent, so the same clip means the same measures).
        self.perform_tab.clip_changed.connect(self.practice_tab.sync_clip)
        self.practice_tab.clip_changed.connect(self.perform_tab.sync_clip)
        # user audio playback
        self.toolbar.user_audio_toggled.connect(self.perform_tab.set_user_audio_enabled)
        # bpm / tempo handling
        self.toolbar.tempo_changed.connect(self.on_tempo_changed)

        # --- CENTRAL TABS ---
        self.center_tabs.currentChanged.connect(self.on_center_tab_changed)
        self.perform_tab.viewer_ready.connect( # triggers demo load
            lambda: self.load_folder(self.DEMO_FOLDER_PATH)
        )
        self.perform_tab.analyzed.connect( # reflect new bpm / length
            lambda: self.toolbar.set_tempo(self.score_data.bpm)
        ) 

        # --- TIMEKEEPING ---
        self.wall_clock.time_changed.connect(self.time_changed)
        self.slider.slider_changed.connect(self.slider_changed)
        self.slider.slider_end.connect(self.slider_end)
        self.countdown_timer.finished.connect(self._on_countdown_finished)
        self.countdown_timer.progress.connect(self._on_countdown_progress)

        # --- SIDE PANELS ---
        # recordings tree
        self.recordings_tree.selected.connect(self.on_recording_selected)
        self.recordings_tree.score_renamed.connect(self.on_score_renamed)
        self.recordings_tree.recording_renamed.connect(self.on_recording_renamed)
        self.recordings_tree.score_file_selected.connect(self.on_score_file_selected)
        # settings panel (instrument / range / tuning / transpose)
        self.settings_widget.instrument_applied.connect(self.on_instrument_applied)
        self.settings_widget.range_applied.connect(self.on_range_applied)
        self.settings_widget.tuning_applied.connect(self.on_tuning_applied)
        self.settings_widget.full_score_toggled.connect(self.on_full_score_toggled)
        self.settings_widget.transpose_applied.connect(self.on_transpose_applied)
        # tolerance panel
        self.tolerance_widget.tolerance_applied.connect(self.on_tolerance_applied)
        self.mistake_widget.mode_changed.connect(self.on_mistake_mode_changed)

    # --- ACTIVE-TAB HELPERS ---
    def _practice_active(self) -> bool:
        """True when the Practice tab is the one currently shown."""
        return self.center_tabs.currentWidget() is self.practice_tab

    def _active_tab(self):
        """The mode panel that the shared transport should drive right now."""
        return self.practice_tab if self._practice_active() else self.perform_tab

    def _has_recording(self, warn=False) -> bool:
        """True if there's an active recording, else optionally warn. Used by the
        side-panel handlers (instrument / range / tuning / tolerance)."""
        if self.active_recording is None:
            if warn:
                QMessageBox.warning(self, "No recording selected", "Please select a recording first.")
            return False
        return True

    # --- LOAD SCORE / AUDIO / FOLDER ---
    def load_score(self, filepath: str):
        """Load one directly selected score file."""
        if not self._confirm_unsaved_before_navigation():
            return
        self._load_score_file(filepath, reset_tree=True)

    def load_folder(self, folderpath: str):
        """Scan a folder into the RecordingTree and load its first score."""
        folder = Path(folderpath)
        if not self.recordings_tree.folder_contains_score(folder):
            QMessageBox.warning(
                self,
                "No scores found",
                "No score folders were found. A score folder needs a same-named "
                "score file, or a valid score file directly inside the folder.",
            )
            return
        if not self._confirm_unsaved_before_navigation():
            return
        score_paths = self.recordings_tree.init_folder(folder)
        self._load_score_file(score_paths[0], reset_tree=False)

    def load_audio(self, filepath: str):
        if not self._has_recording():
            return
        if not self._maybe_save_recording(self.active_recording, self.active_recording_name):
            return
        rec = self.active_recording
        rec.load_audio(filepath, score_filepath=self.current_score_path, recording_name=self.active_recording_name)
        # default the recording's name to the uploaded audio file's name
        new_name = self.recordings_tree.set_recording_name(Path(filepath).stem)
        if new_name:
            self.active_recording_name = new_name
            self.status_bar.update_name(new_name)
            self.recordings_tree.update_recording_file(new_name, filepath)
            self._save_recording_cache(rec, recording_name=new_name)
        self.sync_slider()
        # make it playable + (re-)detect pitches in the background (the Perform
        # tab owns the audio player + detection pipeline)
        self.perform_tab.refresh_audio()

    def _new_score_data_for_path(self, filepath: str | Path) -> ScoreData:
        """Parse a fresh ScoreData for one recording.

        Recordings intentionally do not share ScoreData: tempo, clip,
        transposition and active-instrument state are per-recording.
        """
        score_data = ScoreData()
        score_data.load(filepath)
        title = self.recordings_tree.score_title(filepath)
        if title:
            score_data.set_title(title)
        return score_data

    def _activate_perform_score(self, score_data: ScoreData, reload_tab: bool = True):
        """Point app-level Perform controls at `score_data`."""
        self.score_data = score_data
        self.toolbar.score_data = score_data
        self.toolbar.populate_instrument_menu()
        self.toolbar.set_tempo(score_data.bpm)
        self.settings_widget.load_score(score_data)
        self._sync_transpose_input()
        if reload_tab:
            self.perform_tab.load_score(score_data)

    def _load_score_file(
        self,
        filepath: str | Path,
        reset_tree: bool,
        select_recording_path: str | Path | None = None,
    ):
        """Internal score swap. Does not prompt; callers handle unsaved state."""
        self.unalive()
        self.perform_tab.cleanup()
        self.practice_tab.cleanup()

        filepath = Path(filepath)
        self.current_score_path = Path(self._path_key(filepath))
        self.active_recording = None
        self.active_recording_name = None
        self.recordings.clear()
        self.note_panel.set_recording(None)

        self.base_score_data = self._new_score_data_for_path(filepath)
        self.score_data = self.base_score_data
        tree_title = self.recordings_tree.score_title(filepath)
        if tree_title:
            self.score_data.set_title(tree_title)

        self.slider.update_range(score_data=self.score_data)

        if reset_tree:
            self.recordings_tree.init_score(filepath=filepath, score_data=self.score_data)
        else:
            self.recordings_tree.set_active_score(filepath, score_data=self.score_data)

        self._activate_perform_score(self.score_data)
        self.practice_tab.load_score(filepath)

        recording_entries = self.recordings_tree.recording_entries_for_score(filepath)
        if recording_entries:
            self._hydrate_recordings(recording_entries)
            target_name = None
            if select_recording_path is not None:
                target_name = self.recordings_tree.recording_name_for_path(filepath, select_recording_path)
            target_name = target_name or recording_entries[0][0]
            self.recordings_tree.select_recording_name(target_name, score_path=filepath)
            self._detect_imported_recordings()
        else:
            self.recordings_tree._add_recording(name="untitled_recording")
        self._reset_transport_position()

    def _hydrate_recordings(self, recording_entries: list[tuple[str, Path]]):
        """Load every recording file for the currently active score."""
        for name, audio_path in recording_entries:
            rec = Recording(score_data=self._new_score_data_for_path(self.current_score_path))
            try:
                rec.load_audio(
                    str(audio_path),
                    score_filepath=self.current_score_path,
                    recording_name=name,
                )
            except Exception as e:
                print(f"Could not load recording '{audio_path}': {e}")
                continue
            self.recordings[name] = rec

    def _detect_imported_recordings(self):
        """Run bounded background pitch detection for loaded file-backed takes."""
        for name, rec in list(self.recordings.items()):
            if rec.audio_data.end_index <= 0:
                continue
            if rec.has_pitch_data():
                continue
            thread = getattr(rec.pitch_detector, "offline_thread", None)
            if thread and thread.is_alive():
                continue
            self._detect_recording_with_limit(rec, recording_name=name, score_path=self.current_score_path)

    def _detect_recording_with_limit(
        self,
        rec: Recording,
        recording_name: str | None = None,
        score_path: str | Path | None = None,
    ):
        score_path = score_path or self.current_score_path

        def worker():
            with self._pitch_detection_semaphore:
                try:
                    rec.detect_pitches(on_phase=rec.pitch_detector.status_changed.emit)
                    self._save_recording_cache(rec, recording_name=recording_name, score_path=score_path)
                except Exception as e:
                    print(f"[PitchDetector] imported recording detection failed: {e}")
                finally:
                    rec.pitch_detector.detection_finished.emit()

        rec.pitch_detector.offline_thread = threading.Thread(target=worker, daemon=True)
        rec.pitch_detector.offline_thread.start()

    def _reset_transport_position(self):
        """Start a freshly loaded score from its own beginning, not the previous
        score's playhead. Analyze anchors the score onto the take (which stays
        put), so the beginning is the score's first-note time — not a hard 0."""
        self.wall_clock.stop()
        sd = self._active_score_data()
        bounds = sd.get_bounds() if sd is not None else None
        self.slider.set_time(bounds[0] if bounds is not None else 0.0)
        self.update_time_label(self.slider.get_time())
        self._active_tab().render_at(self.slider.get_time())

    # --- SAVE / UNSAVED RECORDINGS ---
    def save_active_recording(self) -> bool:
        """Toolbar action: save the active tab's recording."""
        if self._practice_active():
            return self._save_practice_recording()
        if not self._has_recording(warn=True):
            return False
        return bool(self._save_recording(self.active_recording, self.active_recording_name))

    def _show_saved_status(self):
        self.status_bar.update_status("Saved!")

    def _save_recording_cache(
        self,
        rec: Recording,
        recording_name: str | None = None,
        score_path: str | Path | None = None,
    ) -> bool:
        return JsonHandler(rec).save_cache(
            score_filepath=score_path or self.current_score_path,
            recording_name=recording_name,
        )

    def _save_practice_recording(self) -> bool:
        rec = self.practice_tab.recording
        score_title = self.recordings_tree.score_title(self.current_score_path) or self.score_data.title
        default_name = f"{score_title}_practice"
        saved_path = self._save_recording(rec, default_name)
        if not saved_path:
            return False

        # Practice owns an independent ScoreData copy. Add a fresh Perform
        # Recording pointing at the saved file so later selection uses the app's
        # active score object.
        name = self._unique_recording_name(saved_path.stem)
        item = self.recordings_tree.ensure_recording_item(
            name,
            filepath=saved_path,
            score_path=self.current_score_path,
            select=False,
        )
        if item is not None:
            name = item.data(0, self.recordings_tree.NAME_ROLE) or name
        loaded = Recording(score_data=self._new_score_data_for_path(self.current_score_path))
        loaded.load_audio(str(saved_path), score_filepath=self.current_score_path, recording_name=name)
        self.recordings[name] = loaded
        if not loaded.has_pitch_data():
            self._detect_recording_with_limit(loaded, recording_name=name, score_path=self.current_score_path)
        return True

    def _save_recording(self, rec: Recording | None, default_name: str | None):
        if rec is None:
            return False
        if rec.audio_data.end_index <= 0:
            QMessageBox.warning(self, "No audio available", "There is no audio to save.")
            return False

        score_dir = self._current_score_folder()
        if score_dir is None:
            QMessageBox.warning(self, "No score folder", "Load a score before saving a recording.")
            return False

        default_name = self._safe_filename(default_name or "recording")
        rec.trim_end()
        writable_suffixes = {".wav", ".wave", ".flac", ".ogg", ".aif", ".aiff"}
        current_suffix = rec.audio_filepath.suffix.lower() if rec.audio_filepath else ""
        can_overwrite_audio = current_suffix in writable_suffixes
        if rec.audio_file_exists() and (not rec.unsaved_changes or can_overwrite_audio):
            cache_name = (
                self.active_recording_name
                if rec is self.active_recording
                else default_name
            )
            if rec.unsaved_changes:
                try:
                    rec.save_audio(rec.audio_filepath)
                except Exception as e:
                    QMessageBox.warning(self, "Save failed", f"Could not save recording:\n{e}")
                    return False
            if not self._save_recording_cache(rec, recording_name=cache_name):
                QMessageBox.warning(self, "Save failed", "Could not save recording metadata.")
                return False
            rec.unsaved_changes = False
            self._show_saved_status()
            return rec.audio_filepath

        default_suffix = current_suffix if current_suffix in writable_suffixes else ".wav"
        default_path = score_dir / f"{default_name}{default_suffix or '.wav'}"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Recording",
            str(default_path),
            "WAV (*.wav);;FLAC (*.flac);;AIFF (*.aiff *.aif);;OGG (*.ogg);;All Files (*)",
        )
        if not file_path:
            return False

        chosen = Path(file_path)
        if not chosen.suffix:
            chosen = chosen.with_suffix(".wav")
        save_path = score_dir / chosen.name

        try:
            rec.save_audio(save_path)
        except Exception as e:
            QMessageBox.warning(self, "Save failed", f"Could not save recording:\n{e}")
            return False

        cache_name = default_name
        if rec is self.active_recording:
            new_name = self.recordings_tree.set_recording_name(save_path.stem, old_name=self.active_recording_name)
            if new_name:
                self.active_recording_name = new_name
                self.status_bar.update_name(new_name)
                self.recordings_tree.update_recording_file(new_name, save_path)
                cache_name = new_name
        if not self._save_recording_cache(rec, recording_name=cache_name):
            QMessageBox.warning(self, "Save failed", "Could not save recording metadata.")
            return False
        self._show_saved_status()
        return save_path

    def _confirm_unsaved_before_navigation(self) -> bool:
        if not self._maybe_save_recording(self.active_recording, self.active_recording_name):
            return False
        practice_rec = getattr(self.practice_tab, "recording", None)
        if practice_rec is not None and practice_rec.needs_save():
            return self._maybe_save_recording(practice_rec, "practice recording", practice=True)
        return True

    def _maybe_save_recording(
        self,
        rec: Recording | None,
        name: str | None,
        practice: bool = False,
    ) -> bool:
        if rec is None or not rec.needs_save():
            return True
        display_name = name or "recording"
        choice = QMessageBox.warning(
            self,
            "Unsaved recording",
            f"Save changes to {display_name}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Save:
            if practice:
                return self._save_practice_recording()
            return bool(self._save_recording(rec, name))
        if choice == QMessageBox.StandardButton.Discard:
            rec.unsaved_changes = False
            return True
        return False

    def _current_score_folder(self) -> Path | None:
        if self.current_score_path is None:
            return None
        return Path(self.current_score_path).parent

    def _unique_recording_name(self, base: str) -> str:
        base = (base or "recording").strip()
        if base not in self.recordings:
            return base
        n = 2
        while f"{base} ({n})" in self.recordings:
            n += 1
        return f"{base} ({n})"

    @staticmethod
    def _safe_filename(name: str) -> str:
        cleaned = "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()
        return cleaned or "recording"

    @staticmethod
    def _path_key(path: str | Path | None) -> str | None:
        if path is None:
            return None
        return str(Path(path).expanduser().resolve())

    # --- SHARED TRANSPORT DISPATCH ---
    def toggle_playback(self):
        """Play/pause the active tab via the shared play button."""
        playing = self._active_tab().toggle_playback()
        self.play_button.setIcon(self.pause_icon if playing else self.play_icon)

    def toggle_recording(self):
        panel = self._active_tab()
        # Perform needs an active recording to capture into; Practice always has
        # its own, so only guard on the Perform tab.
        if panel is self.perform_tab and not self._has_recording(warn=True):
            return
        if self.is_counting_in:
            # clicking again during the count-in cancels it (un-arm recording) and
            # snaps the plot back to where the user hit record (the head).
            self.countdown_timer.cancel()
            self.is_counting_in = False
            self.record_button.setIcon(self.record_icon)
            self._active_tab().unprime_recording()  # release the warmed mic stream
            self._active_tab().render_at(self._countin_head)
            return
        if not panel.is_recording:
            # scroll a one-measure metronome count-in into the head, then begin
            # recording (_on_countdown_finished). Use the active tab's score so the
            # count-in beats land at that tab's (possibly different) tempo.
            sd = self._active_score_data()
            # snap the cursor to the clip start (if any) so the count-in scrolls in
            # toward THAT head, and remember it for snap-back on cancel.
            self.slider.sync_clip_window(sd)
            head = self.slider.get_time()
            beats = sd.count_in_beats()
            spb = (beats[1][0] - beats[0][0]) if len(beats) >= 2 else 0.5
            # Perform records a beat before the head (runway) — even into negative
            # app-time when recording from the start; the take is realigned to the
            # clip start at analysis. Practice is pitch-driven, so it starts at the head.
            record_at = (head - spb) if panel is self.perform_tab else head
            # Perform's 4th click lands on the record beat — play it only when the
            # metronome is on (the channel is in the score's playing set).
            metronome_on = (sd.metronome_channel is not None
                            and sd.metronome_channel in sd.playing_instruments)
            self._countin_head = head
            self.is_counting_in = True
            self.record_button.setIcon(self.pause_icon)
            # Warm the mic input device now so its first-start transient settles
            # during the count-in, not into the first second of the take.
            panel.prime_recording()
            self.countdown_timer.start(
                beats=beats,
                channel=sd.metronome_channel,
                head_time=head,
                record_time=record_at,
                metronome_on=metronome_on,
            )
        else:
            panel.stop_recording()
            self.record_button.setIcon(self.record_icon)
            self.status_bar.update_status("")

    def _on_countdown_progress(self, t: float):
        """Count-in tick: scroll the active tab's views to plot time `t`."""
        if not self.is_counting_in:
            return
        self._active_tab().render_at(t)

    def _on_countdown_finished(self, record_time: float):
        """Shared count-in finished: start recording in whichever tab is active,
        beginning capture at `record_time` (where the count-in scroll left off)."""
        self.is_counting_in = False
        self.record_button.setIcon(self.pause_icon)
        self._active_tab().start_recording(record_time)

    def update_time_label(self, t: float):
        """Update the shared time label (current/total) from time `t`."""
        current = Slider.format_time(t)
        total = Slider.format_time(self.slider.get_total_time())
        self.time_label.setText(f"{current} / {total}")

    def time_changed(self, t: float):
        """Shared wall-clock tick: refresh the time label, then drive the active
        tab's plots (each panel guards on its own playing state)."""
        self.update_time_label(t)
        self._active_tab().on_clock_tick(t)

    def slider_changed(self, t: float):
        """Shared slider moved: refresh the time label, then drive the active
        tab's plots (each panel guards against moving while it's playing)."""
        self.update_time_label(t)
        self._active_tab().on_slider_changed(t)

    def slider_end(self, t: float):
        """Shared slider hit its end (or the clip end). On the Practice tab, stop
        the take and reset the transport. On the Perform tab, stop playback only
        when a clip is active — its end is the natural stop; an unclipped Perform
        tab still plays to the true end as before."""
        if self._practice_active():
            self.practice_tab.on_slider_end()
            self.play_button.setIcon(self.play_icon)
            self.record_button.setIcon(self.record_icon)
            self.status_bar.update_status("")

        elif self.slider.clip_window is not None and self.perform_tab.is_playing:
            self.perform_tab.stop_playback()
            self.play_button.setIcon(self.play_icon)

    # --- SIDE-PANEL HANDLERS ---
    # ------> RECORDINGS TREE
    def on_recording_selected(self, recording_name: str):
        """
        Triggers when recording is selected in RecordingTree.
        Updates it as the active_recording, pushes its config into side panels,
        and updates Performance Tab with it.
        """
        if recording_name is None:
            self.active_recording = None
            self.active_recording_name = None
            if self.base_score_data is not None:
                self._activate_perform_score(self.base_score_data, reload_tab=True)
            self.status_bar.update_name("untitled_recording")
            self.note_panel.set_recording(None)
            self.sync_slider()
            return
        if recording_name not in self.recordings.keys():
            print(f"No recording named '{recording_name}' was found.")
            return

        if (self.active_recording is not None
                and self.active_recording is not self.recordings[recording_name]):
            old_name = self.active_recording_name
            if not self._maybe_save_recording(self.active_recording, old_name):
                if old_name:
                    self.recordings_tree.select_recording_name(old_name, emit=False)
                return
        
        # set the active recording here!
        self.active_recording = self.recordings[recording_name]
        self.active_recording_name = recording_name
        rec = self.active_recording
        self._activate_perform_score(rec.score_data, reload_tab=True)

        # push config into side panels
        self.settings_widget.set_active_instrument(rec.active_instrument)
        self.settings_widget.set_tuning(rec.config.tuning)
        self._sync_tolerance_widget()
        self.status_bar.update_name(recording_name)
        # also pass to Performance Tab
        self.perform_tab.set_active_recording(rec)
        self.practice_tab.sync_clip(rec.score_data.clip)
        self.note_panel.set_recording(rec)
        self.sync_slider()

    def on_score_file_selected(self, score_path: str, recording_path):
        """A folder-library score or recording pointer was selected."""
        if not score_path:
            return
        current = self._path_key(self.current_score_path)
        requested = self._path_key(score_path)

        if requested == current:
            if recording_path:
                name = self.recordings_tree.recording_name_for_path(score_path, recording_path)
                if name:
                    self.recordings_tree.select_recording_name(name, score_path=score_path)
            return

        old_name = self.active_recording_name
        if not self._confirm_unsaved_before_navigation():
            if old_name:
                self.recordings_tree.select_recording_name(old_name, emit=False)
            else:
                self.recordings_tree.select_score(current, emit=False)
            return
        self._load_score_file(
            score_path,
            reset_tree=False,
            select_recording_path=recording_path,
        )
    
    def on_score_renamed(self, title: str):
        """The Score Title was edited in the RecordingTree (the source of truth).
        Push it to BOTH tabs' (independent) scores and re-render their viewers."""
        if self.base_score_data is not None:
            self.base_score_data.set_title(title)
        for name, rec in self.recordings.items():
            rec.score_data.set_title(title)
            self._save_recording_cache(rec, recording_name=name)
        self.score_data.set_title(title)
        self.practice_tab.score_data.set_title(title)
        self.perform_tab.refresh_score_viewer()
        self.practice_tab.refresh_score_viewer()

    def on_recording_renamed(self, old_name: str, new_name: str):
        rec = self.recordings.get(new_name)
        if rec is None:
            return
        if self.active_recording_name == old_name or self.active_recording is rec:
            self.active_recording_name = new_name
            self.status_bar.update_name(new_name)
        self._save_recording_cache(rec, recording_name=new_name)

    # ------> SELECT INSTRUMENT PANEL
    def on_instrument_applied(self, channel: int):
        """Make the selected `channel` the active instrument for the 
        active recording. Keep both tabs (sharing score_data) in sync."""
        if not self._has_recording():
            return
        # the range defaults follow the newly selected instrument's note range
        self.settings_widget.populate_range_from_score(self.score_data, channel)
        # update the panels
        self.perform_tab.set_active_instrument(channel)
        self.practice_tab.set_active_instrument(channel)
        # the transpose anchor (first note) follows the newly selected instrument
        self._sync_transpose_input()
        self._save_recording_cache(self.active_recording, recording_name=self.active_recording_name)

    def on_transpose_applied(self, target_midi: int):
        """Transpose BOTH tabs' scores so the active instrument's first note lands
        on `target_midi`. Pitch-only, so it's mirrored across tabs (like the clip)
        — each tab keeps its own tempo. Refreshes playback + both viewers, then
        re-syncs the widget to the new first-note pitch."""
        sd = self._active_score_data()
        if sd is None or sd.score is None:
            return
        first = sd.first_note_midi()
        if first is None:
            return
        delta = int(target_midi) - first
        if delta == 0:
            return
        self.perform_tab.transpose(delta)
        self.practice_tab.transpose(delta)
        self._sync_transpose_input()
        # the score's pitches moved, so its note range moved too: slide the
        # detection frequency range along with it (and refresh the Range inputs).
        self._sync_range_after_transpose(delta)
        if self.active_recording is not None and not self._practice_active():
            if self.active_recording.has_analysis():
                self.active_recording.update_alignment_distances()
            self._save_recording_cache(self.active_recording, recording_name=self.active_recording_name)

    def _sync_range_after_transpose(self, delta: int):
        """A transpose of `delta` half steps shifts every note, so the score's
        note range — and the frequency range pitch detection should look in —
        moves with it. Refresh the Range inputs from the transposed score and
        slide each recording's Config fmin/fmax by `delta`. Forward-looking only:
        the existing take's already-detected pitches are left alone, since
        transposing the SCORE doesn't change the recorded audio."""
        # 1. displayed Range defaults track the (now transposed) score
        self.settings_widget.populate_range_from_score(
            self._active_score_data(), self.settings_widget.current_channel()
        )
        # 2. slide the detection range so a future take is detected in the new
        #    range (mirror the explicit Apply path, minus the re-detect)
        if not self._has_recording():
            return
        config = self.active_recording.config
        fmin = config.midi_to_freq(config.freq_to_midi(config.fmin) + delta)
        fmax = config.midi_to_freq(config.freq_to_midi(config.fmax) + delta)
        config.fmin = fmin
        config.fmax = fmax
        self.active_recording.update_config(config)
        self.practice_tab.set_freq_range(fmin, fmax)
        self._save_recording_cache(self.active_recording, recording_name=self.active_recording_name)

    def _sync_transpose_input(self):
        """Reflect the active tab's current first-note pitch in the settings
        panel's transpose input (no emit). Call whenever the score / active
        instrument changes."""
        self.settings_widget.set_first_note(self._active_score_data().first_note_midi())

    def on_full_score_toggled(self, show_full: bool):
        """Toggle BOTH tabs' score viewers between the full score and the active
        part. app.py owns the flag (single source of truth) and pushes it into
        each tab via set_show_full, so the two stay in sync like the other
        side-panel settings."""
        self.viewer_show_full = show_full
        self.perform_tab.set_show_full(show_full)
        self.practice_tab.set_show_full(show_full)

    def on_range_applied(self, low_midi: int, high_midi: int):
        """Set the active recording's Config frequency range, then re-detect."""
        if not self._has_recording():
            return
        rec = self.active_recording
        config = rec.config
        fmin = config.midi_to_freq(low_midi)
        fmax = config.midi_to_freq(high_midi)
        config.fmin = fmin
        config.fmax = fmax
        rec.update_config(config)
        # mirror into the Practice tab's recording config
        self.practice_tab.set_freq_range(fmin, fmax)
        # re-run pitch detection on the existing audio (if any) with the new range
        self.perform_tab.detect_pitches()

    def on_tuning_applied(self, tuning: float):
        """Set the active recording's Config tuning (A4 Hz), then re-detect."""
        if not self._has_recording():
            return
        # update the active recording's appropriate config
        rec = self.active_recording
        rec.config.tuning = tuning
        rec.update_config(rec.config)
        # also update in practice panel
        self.practice_tab.set_tuning(tuning)
        self.perform_tab.detect_pitches()

    # ------> TOLERANCE PANEL
    def _sync_tolerance_widget(self):
        if self.active_recording is None:
            return
        self.tolerance_widget.set_tolerances(
            self.active_recording.config.pitch_tolerance,
            self.active_recording.config.timing_tolerance,
        )

    def on_mistake_mode_changed(self, mode: str):
        """The MistakeWidget tab controls which tolerance the side control edits."""
        self._sync_tolerance_widget()
        self.tolerance_widget.set_mode(mode)

    def on_tolerance_applied(self, mode: str, tolerance: float):
        """Set the active recording's pitch or timing tolerance and refresh the
        relevant mistake view."""
        if not self._has_recording():
            return
        rec = self.active_recording
        if mode == "timing":
            rec.config.timing_tolerance = tolerance
            rec.update_config(rec.config)
            if rec.has_analysis():
                # A tolerance edit changes labels, not the fitted repeat path.
                rec.alignment = rec.mistake_detector.alignment_from_pairs(rec.alignment.pairs)
                rec.reindex_mistakes()
                self.perform_tab.guitar_hero.load_alignment(rec.alignment)
                self.perform_tab.refresh_mistake_widget(rec)
            self._save_recording_cache(rec, recording_name=self.active_recording_name)
            return

        rec.config.pitch_tolerance = tolerance
        rec.update_config(rec.config)
        # mirror into the Practice tab (drives its live pitch match)
        self.practice_tab.set_pitch_tolerance(tolerance)
        self.perform_tab.guitar_hero.user_pitches.set_tolerance(tolerance)
        self.perform_tab.guitar_hero.update_view_items()
        self.perform_tab.reanalyze_if_analyzed()
        self._save_recording_cache(rec, recording_name=self.active_recording_name)

    # --- TOOLBAR THINGS ---
    def _is_live(self) -> bool:
        """True if either tab is playing/recording, or a count-in is running."""
        return (self.perform_tab.is_playing or self.perform_tab.is_recording
                or self.practice_tab.is_playing or self.practice_tab.is_recording
                or self.is_counting_in)
    
    def on_tempo_changed(self, new_bpm: int):
        """Triggers when tempo changes, pushes changes to the active tab's score.
        Rejects when recording/playing back."""
        sd = self._active_score_data()
        if self._is_live():
            self.toolbar.set_tempo(sd.bpm)  # revert UI
            return
        if not self._practice_active() and self.active_recording is not None:
            self.active_recording.change_tempo(new_bpm)
            self._save_recording_cache(self.active_recording, recording_name=self.active_recording_name)
        else:
            sd.change_tempo(new_bpm)
        self._active_tab().guitar_hero.update_view_items()
        self.sync_slider() # re-range the shared slider for the active tab

    def _active_score_data(self) -> ScoreData:
        """The score the active tab is showing — Perform's is the app's; Practice
        keeps its own independent copy."""
        return self.practice_tab.score_data if self._practice_active() else self.score_data

    def _on_analyzed(self):
        """Analyze resized the Perform score: show its new BPM in the toolbar (it
        only runs on the Perform tab, whose score is self.score_data)."""
        self.toolbar.set_tempo(self.score_data.bpm)

    # --- TAB SWITCHING / SLIDER SYNC ---
    def _active_slider_recording(self):
        """The recording whose length should size the shared slider, picked by the
        active tab (Practice keeps its own take separate from Perform's)."""
        if self._practice_active():
            return self.practice_tab.recording
        return self.active_recording

    def sync_slider(self):
        """Re-range the shared slider to the active tab's score/recording length."""
        self.slider.update_range(
            score_data=self._active_score_data(), recording=self._active_slider_recording()
        )

    def unalive(self):
        """Go 'un-live': stop playback/recording/count-in on both tabs"""
        self.perform_tab.stop_playback()
        self.perform_tab.stop_recording()
        self.practice_tab.stop_playback()
        self.practice_tab.stop_recording()
        if self.is_counting_in:
            self.countdown_timer.cancel()
            self.is_counting_in = False

    def on_center_tab_changed(self, index: int):
        """The shared transport drives whichever tab is active. On switch: halt the
        old tab, hide the Perform-only Analyze button on Practice, reset the
        transport buttons, then re-range the shared slider to the new tab and line
        its views up with the current time (the two tabs can have different total
        lengths, so the slider position is clamped/re-rendered to stay consistent)."""
        if self._suppress_tab_change:
            return
        old_practice = self.center_tabs.widget(self._current_tab_index) is self.practice_tab
        practice = self.center_tabs.widget(index) is self.practice_tab
        if old_practice and not practice:
            practice_rec = getattr(self.practice_tab, "recording", None)
            if practice_rec is not None and practice_rec.needs_save():
                if not self._maybe_save_recording(practice_rec, "practice recording", practice=True):
                    self._suppress_tab_change = True
                    self.center_tabs.setCurrentIndex(self._current_tab_index)
                    self._suppress_tab_change = False
                    return
        self._current_tab_index = index
        self.unalive()

        self.analyze_button.setVisible(not practice)
        # reset the shared transport to a stopped state for the new tab
        self.play_button.setIcon(self.play_icon)
        self.record_button.setIcon(self.record_icon)
        self.status_bar.update_status("")
        # the tempo display reflects the active tab's (independent) score tempo
        self.toolbar.set_tempo(self._active_score_data().bpm)
        # ...and the transpose anchor reflects that tab's score's first note
        self._sync_transpose_input()

        # the note panel follows the new tab's recording (Practice keeps its own)
        self.note_panel.set_recording(self._active_slider_recording())

        # preserve the current time, re-range for the new tab, clamp, and render.
        t = self.slider.get_time()
        self.sync_slider()
        t = min(t, self.slider.get_total_time())
        self.slider.handle_timer_update(t)   # move the shared handle to clamped t
        self._active_tab().render_at(t)

    # --- CLIP (measure-range focus) ---
    def on_select_measures(self):
        """Clip menu 'Select measures': arm measure selection in the active tab
        (the user then picks a start + end measure and right-clicks to clip)."""
        self._active_tab().start_clip_selection()

    def on_clip_reset(self):
        """Clip menu 'Reset': restore the active tab's full score."""
        self._active_tab().reset_clip()

    def closeEvent(self, event):
        if self._confirm_unsaved_before_navigation():
            self.unalive()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    # create the pyqt app instance and run it
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    window = Attune()
    window.show()
    sys.exit(app.exec())
