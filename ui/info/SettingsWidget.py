import re

from PyQt6.QtCore import Qt, QSize, QEvent, pyqtSignal, QStringListModel
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QCompleter, QMessageBox, QToolTip,
    QSizePolicy, QScrollArea, QFrame,
)

from algorithms.Config import Config
from app_logic.midi.ScoreData import ScoreData
from resources.program_map import program_to_name
from ui.info.MistakeWidget import _svg_icon


# note-name <-> MIDI helpers. Display routes through Config.get_note_name (sharps
# by default, matching MistakeWidget / Note.get_note_name; octave = midi // 12 - 1,
# so C4 == MIDI 60). Parsing + the transpose autocomplete additionally accept
# flats, so 'A#4' and 'Bb4' both resolve to the same pitch.
_PITCH_CLASS = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
    "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}
_NAME_RE = re.compile(r"^\s*([A-Ga-g])([#bB]?)(-?\d+)\s*$")


def midi_to_name(m: int, prefer_flats: bool = False) -> str:
    """MIDI number -> note name, e.g. 60 -> 'C4' (or 'Db4' with prefer_flats)."""
    return Config.get_note_name(m, prefer_flats=prefer_flats)


def name_to_midi(name: str) -> int | None:
    """Note name -> MIDI number, e.g. 'C4' -> 60, 'Bb3' -> 58. Accepts sharps or
    flats. Returns None if unparseable."""
    match = _NAME_RE.match(name)
    if not match:
        return None
    letter, accidental, octave = match.group(1).upper(), match.group(2), int(match.group(3))
    pc = _PITCH_CLASS.get(letter + accidental.upper())
    if pc is None:
        return None
    return (octave + 1) * 12 + pc


class SettingsWidget(QWidget):
    """
    Left-column panel (below the RecordingTree) holding the per-recording /
    per-score settings, stacked in a vertical scroll area so it can grow without
    capping the column:

      - "Instrument": a dropdown of every instrument detected in the score (one
        per MIDI channel, minus the metronome). Applying it makes that channel
        the active instrument (the analysis target).
      - "Range": low/high note inputs that auto-complete to the discretized note
        bins (MIDI 0..127, named like 'G3' / 'C#5'). Defaults are filled in from
        the active instrument's note range. Applying it sets the Config's
        fmin/fmax (so pitch detection is re-run).
      - "Tuning": set A4's freq in Hz, reruns pitch detection if different
      - "Transpose": a note-name input that transposes the WHOLE score so its
        first note lands on the entered pitch (relative to the first note). Emits
        the target MIDI; app.py shifts the score's MIDI / music21 / NoteData.

    The widget only collects/validates input and emits signals; app.py owns the
    Recording / ScoreData and performs the actual reset + re-analysis.
    """
    instrument_applied = pyqtSignal(int)   # emits the active-instrument channel
    range_applied = pyqtSignal(int, int)   # emits (lowest_midi, highest_midi)
    tuning_applied = pyqtSignal(float)     # emits the A4 reference tuning in Hz
    transpose_applied = pyqtSignal(int)    # emits the target MIDI for the first note
    full_score_toggled = pyqtSignal(bool)  # True = show full score, False = active instrument only

    # the discretized note bins the range / transpose inputs auto-complete to;
    # black keys appear as both spellings (e.g. 'C#4' and 'Db4'), sharp first.
    _NOTE_BIN_NAMES = [
        name
        for m in range(128)
        for name in dict.fromkeys((midi_to_name(m), midi_to_name(m, prefer_flats=True)))
    ]
    _COMBO_STYLE = """
        QComboBox {
            padding-left: 6px;
            padding-right: 22px;
        }
        QComboBox QAbstractItemView::item {
            padding: 2px 8px 2px 6px;
            min-height: 24px;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.score_data: ScoreData | None = None
        self._check_icon = _svg_icon("check.svg")
        self._first_note_midi: int | None = None
        self._TRANSPOSE_HELP = "Transposes the score relative to the pitch of the first note."

        # the whole panel scrolls vertically: a fixed-frame scroll area wraps an
        # inner content widget that all the rows live in (init_ui fills _layout).
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content = QWidget()
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._scroll.setWidget(content)
        outer.addWidget(self._scroll)

        self.init_ui()

        # keep the column narrow; height is left free so the vertical splitter
        # can size it and the inner scroll area handles any overflow.
        self.setMinimumWidth(180)
        self.setMaximumWidth(320)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    # --- UI ---
    def init_ui(self):
        # a single shared model backs the range + transpose inputs' completion
        self._note_bin_model = QStringListModel(self._NOTE_BIN_NAMES, self)

        # --- ACTIVE INSTRUMENT SELECT ---
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Instrument:"))
        self.instrument_combo = QComboBox()
        self.instrument_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.instrument_combo.setStyleSheet(self._COMBO_STYLE)
        self.instrument_combo.view().setTextElideMode(Qt.TextElideMode.ElideNone)
        row.addWidget(self.instrument_combo, 1)

        self.instrument_apply = self._make_apply_button("Apply instrument")
        self.instrument_apply.clicked.connect(self._on_instrument_apply)
        row.addWidget(self.instrument_apply)
        self._layout.addLayout(row)

        # --- RANGE UI ---
        row = QHBoxLayout()
        row.addWidget(QLabel("Range:"))
        self.low_input = self._make_note_input("G3")
        row.addWidget(self.low_input, 1)
        row.addWidget(QLabel("—"))
        self.high_input = self._make_note_input("E7")
        row.addWidget(self.high_input, 1)

        self.range_apply = self._make_apply_button("Apply frequency range")
        self.range_apply.clicked.connect(self._on_range_apply)
        row.addWidget(self.range_apply)
        self._layout.addLayout(row)

        # --- TUNING UI ---
        # the A4 reference pitch (Hz) the pitch detector tunes to; applying it
        # updates the Config and re-runs pitch detection.
        row = QHBoxLayout()
        row.addWidget(QLabel("Tuning:"))
        self.tuning_input = QLineEdit()
        default_tuning = f"{Config().tuning:g}"
        self.tuning_input.setPlaceholderText(default_tuning)
        self.tuning_input.setText(default_tuning)
        row.addWidget(self.tuning_input, 1)
        row.addWidget(QLabel("Hz"))

        self.tuning_apply = self._make_apply_button("Apply tuning")
        self.tuning_apply.clicked.connect(self._on_tuning_apply)
        row.addWidget(self.tuning_apply)
        self._layout.addLayout(row)

        # --- TRANSPOSE UI ---
        # transposes the WHOLE score so its first note lands on the entered pitch.
        row = QHBoxLayout()
        help_icon = _svg_icon("circle-help.svg", px=32)
        self.transpose_help = QLabel()
        self.transpose_help.setPixmap(help_icon.pixmap(QSize(14, 14)))
        self.transpose_help.setToolTip(self._TRANSPOSE_HELP)
        self.transpose_help.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        # hover tooltips can silently fail on macOS, so the icon is also clickable:
        # a click forces the same text into a small popup (mirrors ToleranceWidget).
        self.transpose_help.setCursor(Qt.CursorShape.PointingHandCursor)
        self.transpose_help.installEventFilter(self)
        row.addWidget(self.transpose_help)

        row.addWidget(QLabel("Transpose:"))
        self.transpose_input = self._make_note_input("C4")
        self.transpose_input.returnPressed.connect(self._on_transpose_apply)
        row.addWidget(self.transpose_input, 1)

        self.transpose_apply = self._make_apply_button("Apply transpose")
        self.transpose_apply.clicked.connect(self._on_transpose_apply)
        row.addWidget(self.transpose_apply)
        self._layout.addLayout(row)

        # keep the rows hugging the top; any extra column height is empty space.
        self._layout.addStretch(1)

    def _make_apply_button(self, tooltip: str) -> QPushButton:
        button = QPushButton()
        button.setIcon(self._check_icon)
        button.setIconSize(QSize(16, 16))
        button.setFixedSize(QSize(28, 28))
        button.setToolTip(tooltip)
        return button

    def _make_note_input(self, placeholder: str) -> QLineEdit:
        box = QLineEdit()
        box.setPlaceholderText(placeholder)
        completer = QCompleter(self._note_bin_model, box)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        box.setCompleter(completer)
        return box

    def _fit_instrument_combo_to_contents(self):
        if self.instrument_combo.count() == 0:
            return
        metrics = self.instrument_combo.fontMetrics()
        width = max(
            metrics.horizontalAdvance(self.instrument_combo.itemText(i))
            for i in range(self.instrument_combo.count())
        ) + 46
        self.instrument_combo.setMinimumWidth(width)
        self.instrument_combo.view().setMinimumWidth(width)

    def eventFilter(self, obj, event):
        """Clicking the transpose help icon shows its text in a small popup on top
        of the icon — a reliable fallback for when hover tooltips don't fire."""
        if obj is self.transpose_help and event.type() == QEvent.Type.MouseButtonPress:
            QToolTip.showText(
                self.transpose_help.mapToGlobal(self.transpose_help.rect().topLeft()),
                self._TRANSPOSE_HELP,
                self.transpose_help,
            )
            return True
        return super().eventFilter(obj, event)

    # --- PUBLIC API ---
    def load_score(self, score_data: ScoreData):
        """Populate the instrument dropdown from a freshly loaded score, then
        sync the dropdown + range defaults to the current active instrument."""
        self.score_data = score_data
        self._populate_instruments()
        self.set_active_instrument(score_data.active_instrument)

    def set_active_instrument(self, channel: int):
        """Reflect `channel` in the dropdown and refresh the range defaults to
        that instrument's note range. Does not emit (no Apply)."""
        idx = self.instrument_combo.findData(channel)
        if idx >= 0:
            self.instrument_combo.setCurrentIndex(idx)
        if self.score_data is not None:
            self.populate_range_from_score(self.score_data, self.current_channel())

    def current_channel(self) -> int | None:
        """The channel currently selected in the dropdown, or None if empty."""
        data = self.instrument_combo.currentData()
        return int(data) if data is not None else None

    def populate_range_from_score(self, score_data: ScoreData, channel: int | None):
        """Prefill the low/high note inputs from the score's note range for the
        given instrument channel (defaults populated 'based on the score')."""
        if channel is None:
            return
        midi_range = score_data.note_midi_range(channel)
        if midi_range is None:
            return
        low, high = Config.padded_midi_range(*midi_range)
        self.low_input.setText(midi_to_name(low))
        self.high_input.setText(midi_to_name(high))

    def set_first_note(self, midi: int | None):
        """Reflect the score's current first-note pitch in the transpose input (no
        emit). Disables Apply when there's no note to anchor the transposition to."""
        self._first_note_midi = midi
        if midi is None:
            self.transpose_input.clear()
            self.transpose_input.setEnabled(False)
            self.transpose_apply.setEnabled(False)
            return
        self.transpose_input.setEnabled(True)
        self.transpose_apply.setEnabled(True)
        self.transpose_input.setText(midi_to_name(int(midi)))

    # --- INTERNAL ---
    def _populate_instruments(self):
        self.instrument_combo.blockSignals(True)
        self.instrument_combo.clear()
        if self.score_data is not None:
            for channel, program in self.score_data.instruments.items():
                if channel == self.score_data.metronome_channel:
                    continue  # don't expose the metronome click track
                self.instrument_combo.addItem(program_to_name(program), channel)
        self._fit_instrument_combo_to_contents()
        self.instrument_combo.blockSignals(False)

    def _on_instrument_apply(self):
        channel = self.current_channel()
        if channel is None:
            return
        self.instrument_applied.emit(channel)

    def _on_range_apply(self):
        low = name_to_midi(self.low_input.text())
        high = name_to_midi(self.high_input.text())
        if low is None or high is None:
            QMessageBox.warning(
                self, "Invalid note",
                "Enter valid note names like 'G3' or 'C#5' for both ends of the range.",
            )
            return
        if low > high:
            low, high = high, low  # be forgiving if the inputs are swapped
        self.range_applied.emit(low, high)

    def set_tuning(self, tuning: float):
        """Reflect the active recording's current tuning (Hz) in the input,
        without emitting (no Apply). Mirrors the range defaults sync."""
        self.tuning_input.setText(f"{tuning:g}")

    def _on_tuning_apply(self):
        try:
            tuning = float(self.tuning_input.text().strip())
        except ValueError:
            tuning = None
        if tuning is None or tuning <= 0:
            QMessageBox.warning(
                self, "Invalid tuning",
                "Enter a positive number of Hz for the tuning (e.g. 440).",
            )
            return
        self.tuning_applied.emit(tuning)

    def _on_transpose_apply(self):
        if self._first_note_midi is None:
            return
        target = name_to_midi(self.transpose_input.text())
        if target is None:
            QMessageBox.warning(
                self, "Invalid note",
                "Enter a valid note name like 'C4' or 'F#3' to transpose the "
                "first note to.",
            )
            return
        self.transpose_applied.emit(target)
