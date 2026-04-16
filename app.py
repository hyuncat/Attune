import os
from pathlib import Path
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QStatusBar, QPushButton, QLabel, QTreeWidget, QTreeWidgetItem, QSplitter,
    QInputDialog, QMenu, QMessageBox, QStackedLayout
)
from PyQt6.QtCore import Qt, QSize, QTimer, QPoint, pyqtSignal, QObject
from PyQt6.QtGui import QIcon
import qdarktheme

from ui.ScoreViewer import ScoreViewer
from ui.GuitarHero import GuitarHero
from ui.time.Slider import Slider
from ui.time.WallClock import WallClock
from ui.time.Clipper import ClipperDialog
from ui.time.CountdownTimer import CountdownTimer

from ui.info.Toolbar import Toolbar
from ui.info.StatusBar import StatusBar
from ui.info.RecordingTree import RecordingTree
from ui.info.Settings import SettingsDialog

# app logic imports
from app_logic.user.ds.Recording import Recording
from app_logic.user.ds.PitchData import PitchConfig
from app_logic.user.AudioPlayer import AudioPlayer
from app_logic.user.AudioRecorder import AudioRecorder
from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiSynth import MidiSynth
from app_logic.midi.MidiPlayer import MidiPlayer
from app_logic.Alignment import Alignment

from algorithms.Config import Config
from practice import PracticeAttune

class Attune(QMainWindow):
    """each attune instance is associated with a single score (midi/musicxml)
    and allows you to create multiple recordings associated to that score
    each with its own analysis and settings"""
    def __init__(self):
        super().__init__()
        self.score_data = ScoreData()
        self.recordings: dict[str, Recording] = {}  # name -> Recording
        self.active_recording: Recording | None = None
        # rk: each recording comes with their own algorithms

        # PLAYBACK stuff
        self.wall_clock = WallClock(hz=10)
        self.metronome = None # TODO later

        self.SOUNDFONT = "resources/MuseScore_General.sf3"
        self.midi_synth = MidiSynth(self.SOUNDFONT)
        self.midi_player = MidiPlayer(self.midi_synth, self.wall_clock)
        self.audio_player = AudioPlayer(None)
        self.audio_recorder = AudioRecorder(self.active_recording)
        # --> playback state variables
        self.is_playing = False
        self.is_recording = False
        self.user_playback_enabled = True

        # instrument control
        self.displayed_instruments: set[int] = set() # programs to display
        self.playing_instruments: set[int] = set() # channels to play

        # initialize other important stuff
        self.init_ui()
        self.init_signals()

    def init_ui(self):
        """
        Initialize all UI components for the window.
            - main window (title + geom)
            - splitter
                - recordings file tree
                - score viewer
            - slider layout
                - play/pause button
                - record button
                - time label
                - slider
                - analyze button
            - status bar
                - countdown timer
            - toolbar
            - dialogs (settings, clipper)
        """
        self.setWindowTitle("Attune")
        self.setGeometry(100, 100, 1300, 800)

        # --- CENTRAL LAYOUT WIDGET ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self._layout = QVBoxLayout(self.central_widget)

        # --- (splitter stuff) MAIN RECORDINGS TREE // SCORE VIEWER ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal) # allows horizontal resizing
        self.recordings_tree = RecordingTree(self.recordings)
        ABSOLUTE_PROJECT_ROOT = Path(__file__).resolve().parent
        
        # score viewer requires a loading screen
        self.score_viewer_container = QWidget()
        stack = QStackedLayout(self.score_viewer_container)
        self.score_viewer = ScoreViewer(project_root=ABSOLUTE_PROJECT_ROOT)
        loading = QLabel("Loading...")
        loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stack.addWidget(loading)
        stack.addWidget(self.score_viewer)
        stack.setCurrentWidget(loading) # show loading screen until viewer is ready
        self.score_viewer.load_finished.connect(lambda ok: stack.setCurrentIndex(1) if ok else 0)

        self.guitar_hero = GuitarHero(self.active_recording)
        # add the widgets
        self.splitter.addWidget(self.recordings_tree)
        self.splitter.addWidget(self.score_viewer_container)
        self.splitter.addWidget(self.guitar_hero)
        # set behavior controls
        self.splitter.setStretchFactor(0, 0)  # left widget is fixed-ish
        self.splitter.setStretchFactor(1, 1)  # right widget grows
        self.splitter.setStretchFactor(2, 1)  # right widget grows too
        
        self._layout.addWidget(self.splitter)

        # --- INIT SLIDER LAYOUT ---
        self.init_slider_layout()

        # --- UTILITIES --- 
        self.status_bar = StatusBar(name="untitled_recording") # with default recording name
        self.setStatusBar(self.status_bar)
        self.countdown_timer = CountdownTimer(self.status_bar, duration=2.0)
        self.toolbar = Toolbar(score_data=self.score_data)
        self.addToolBar(self.toolbar)
        
        # --- DIALOGS ---
        self.settings_dialog = SettingsDialog()
        self.clipper_dialog = ClipperDialog()
        self.practice_attune = PracticeAttune(self.score_data, self.midi_synth) # practice mode window, initialized but not shown yet

        self.show() # run the show :)
        
    def init_slider_layout(self):
        """
        Initialize the layout containing the play/pause, 
        record button and the slider.
        """
        self.slider_layout = QHBoxLayout()

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
        self._layout.addLayout(self.slider_layout)

        # analyze button
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze)
        self.slider_layout.addWidget(self.analyze_button)

    def init_signals(self):
        """Connect all signals and slots for UI / app logic"""
        # toolbar signals
        self.toolbar.score_uploaded.connect(self.load_score)
        self.toolbar.audio_uploaded.connect(self.load_audio)
        self.toolbar.show_settings.connect(self.settings_dialog.show)
        self.toolbar.show_clipper.connect(self.clipper_dialog.show)
        self.toolbar.user_audio_toggled.connect(self.on_user_audio_toggled)
        self.toolbar.practice_toggled.connect(self.on_practice_toggled)
        self.toolbar.tempo_changed.connect(self.on_tempo_changed)

        # timekeeping signals
        self.wall_clock.time_changed.connect(self.time_changed)
        self.slider.slider_changed.connect(self.slider_changed)
        self.slider.slider_end.connect(self.slider_end)
        self.countdown_timer.finished.connect(self._start_recording)

        self.recordings_tree.selected.connect(self.on_recording_selected)
        self.score_viewer.load_finished.connect(self.on_score_viewer_loaded)

        # settings dialog signals

    # --- LOAD SCORE / AUDIO ---
    def load_score(self, filepath: str):
        """Load the score into the app."""
        # reload same score_data so all modules which reference it 
        # get updated data without needing a manual refresh
        filepath = Path(filepath)
        self.score_data.load(filepath)

        # load into important ui components
        self.toolbar.populate_instrument_menu()
        self.slider.update_range(score_data=self.score_data)
        self.recordings_tree.init_score(filepath, self.score_data)
        self.recordings_tree._add_recording(name="untitled_recording") # dummy init
        self.guitar_hero.load_score(self.score_data)
        self.practice_attune.load_score(self.score_data)
        # wait for viewer to be ready
        _ = self.score_viewer.load_score(self.score_data)

        # load into playback engines
        self.midi_player.load_score(self.score_data)

    def load_audio(self, filepath: str):
        if self.active_recording is None:
            QMessageBox.warning(self, "No recording selected", "Please select a recording to load the audio into.")
            return
        self.active_recording.load_audio(filepath)
        self.guitar_hero.load_user(self.active_recording)
        self.slider.update_range(score_data=self.score_data, recording=self.active_recording)
        self.audio_player.load_audio(self.active_recording.audio_data)
        
    # --- PLAYBACK / RECORDING TOGGLES ---
    def toggle_playback(self):
        t = self.slider.get_time()

        if not self.is_playing:
            self.is_playing = True
            self.wall_clock.start(t)
            self.midi_player.play(start_time=t)
            if self.user_playback_enabled:
                self.audio_player.play(start_time=t)
            # update UI
            self.play_button.setIcon(self.pause_icon)

        elif self.is_playing:
            self.is_playing = False
            self.wall_clock.pause()
            self.midi_player.stop()
            self.audio_player.stop()
            # update UI
            self.play_button.setIcon(self.play_icon)

    def toggle_recording(self):
        if self.active_recording is None:
            QMessageBox.warning(self, "No recording selected", "Please select a recording to record into.")
            return
        if not self.is_recording:
            # start the countdown timer, and once finished start the recording
            self.countdown_timer.start()
        else:
            self._stop_recording()

    def _start_recording(self):
        """Called when the countdown timer finishes, to start the 
        recording and playback."""
        # update UI
        self.record_button.setIcon(self.pause_icon)
        # stuff
        t = self.slider.get_time()
        self.is_recording = True
        self.audio_player.stop()
        self.wall_clock.start(t)
        self.audio_recorder.run(start_time=t)
        self.active_recording.pitch_detector.run(start_time=t)
        self.midi_player.play(start_time=t) # play whatever audio the user has enabled

    def _stop_recording(self):
        """Called when user clicks the record button while already recording, 
        to stop the recording and playback."""
        # update UI
        self.record_button.setIcon(self.record_icon)
        # stuff
        self.is_recording = False
        self.wall_clock.pause()
        self.audio_recorder.stop()
        self.midi_player.stop()
        self.active_recording.pitch_detector.stop()

    def analyze(self):
        print("analyzing... ")
        # detect notes
        self.active_recording.detect_notes()
        # update midi length to match recording length
            # update p.distances to reflect new midi note durations
        l = self.active_recording.get_length()
        self.active_recording.resize(new_length = l)
        # string edit
        self.active_recording.detect_mistakes()

        # update scoreplot
        # update guitar hero bounds
        self.guitar_hero.update_view_items()
        self.slider.update_range(score_data=self.score_data, recording=self.active_recording)
        
    # --- SIGNAL-RELATED ACTIONS ---
    def update_time_label(self, t: float):
        """Update the time label based on current time t."""
        def format_time(seconds: float) -> str:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins:02}:{secs:04.1f}"

        current_time_str = format_time(t)
        total_length = self.slider.get_total_time()
        total_time_str = format_time(total_length)
        self.time_label.setText(f"{current_time_str} / {total_time_str}")

    def time_changed(self, t: float):
        """Called when the wall clock time changes. Update the time label and
        move the score viewer and guitar hero plots IF currently playing."""
        self.update_time_label(t)
        if not self.is_playing:
            return
        # else, move the score and guitar hero plots
        self.score_data.update_time(t)
        self.score_viewer.set_playback_time(t)
        self.guitar_hero.move_plot(t)

    def slider_changed(self, t: float):
        """Called when slider is moved, to handle case when we are not in playback
        or recording mode but still want to see our plots move."""
        self.update_time_label(t)
        if self.is_playing:
            return
        # else, move the score and guitar hero plots
        self.score_data.update_time(t)
        self.score_viewer.set_playback_time(t)
        self.guitar_hero.move_plot(t)

    def slider_end(self, t: float):
        pass

    def on_recording_selected(self, recording_name: str):
        """When a recording is selected from the recordings tree, update the active 
        recording and refresh the score viewer and other relevant UI components."""
        if recording_name not in self.recordings.keys():
            print(f"No recording named '{recording_name}' was found.")
            return
        self.active_recording = self.recordings[recording_name]
        print(f"Setting active recording to '{recording_name}'")
        self.status_bar.update_name(recording_name)
        self.guitar_hero.load_user(self.active_recording)
        self.audio_player.load_audio(self.active_recording.audio_data)
        self.audio_recorder.load_recording(self.active_recording)
        self.slider.update_range(score_data=self.score_data, recording=self.active_recording)

    def on_score_viewer_loaded(self):
        """Called after score viewer is done loading JS and ready to receive data."""
        DEMO_SCORE_PATH = Path(__file__).resolve().parent / "resources" / "scores" / "c_major_scale.mxl"
        self.load_score(str(DEMO_SCORE_PATH))

    def on_user_audio_toggled(self, checked: bool):
        """Called when user toggles the user audio playback option in the toolbar."""
        self.user_playback_enabled = checked
        if not checked:
            self.audio_player.pause()

    def on_tempo_changed(self, new_bpm: int):
        """Called when user changes the tempo in the toolbar. Update the score data and 
        midi player accordingly."""
        if self.is_playing: # revert the ui back to old value
            self.toolbar.tempo_spinbox.setValue(self.score_data.bpm)
            return
        self.score_data.change_tempo(new_bpm)
        self.guitar_hero.update_view_items()
        self.score_viewer.load_score(self.score_data) # reload score to update tempo changes
        self.slider.update_range(score_data=self.score_data, recording=self.active_recording)

    def on_practice_toggled(self):
        """Called when user clicks the practice mode button in the toolbar."""
        # open a confirmation popup
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Enter Practice Mode?")
        msg_box.setText("Are you sure you want to enter practice mode? This will open a new window.")
        msg_box.setStandardButtons(QMessageBox.StandardButton.No | QMessageBox.StandardButton.Yes)
        result = msg_box.exec()

        if result == QMessageBox.StandardButton.Yes:
            # open new window with just guitar hero
            # implement later
            print("Entering practice mode...")
            self.practice_attune.show()
            self.practice_attune.raise_() # bring to front
            self.practice_attune.activateWindow() # focus
        else:
            # close popup
            msg_box.close()


if __name__ == "__main__":
    # create the pyqt app instance and run it
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    window = Attune()
    window.show()
    sys.exit(app.exec())