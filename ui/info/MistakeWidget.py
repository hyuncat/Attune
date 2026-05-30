from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem, QPushButton,
)
from app_logic.Alignment import Mistake


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_name(midi_num: float) -> str:
    """Convert a MIDI number to a letter name like C4 or F#3."""
    n = int(round(midi_num))
    return f"{_NOTE_NAMES[n % 12]}{n // 12 - 1}"


class MistakeWidget(QWidget):
    """
    Right-side panel listing all analyzed mistakes for the active recording.

    Columns: Index | Pair | Type | Intended | Actual | Override
    """
    selected = pyqtSignal(int)         # emits mistake index on row click
    override_toggled = pyqtSignal(int) # emits mistake index when Override is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)

        self._mistakes: list[Mistake] = []
        self._override_buttons: dict[int, QPushButton] = {}

        self.init_ui()
        self.init_signals()

    def init_ui(self):
        self.header_label = QLabel("Mistakes")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self.header_label)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["#", "Pair", "Type", "Intended", "Actual", ""])
        self.tree.setIndentation(0)
        self.tree.setRootIsDecorated(False)

        self.tree.setColumnWidth(0, 30)
        self.tree.setColumnWidth(1, 36)
        self.tree.setColumnWidth(2, 54)
        self.tree.setColumnWidth(3, 60)
        self.tree.setColumnWidth(4, 60)
        self.tree.setColumnWidth(5, 72)

        self._layout.addWidget(self.tree)

        self.setMinimumWidth(200)
        self.setMaximumWidth(380)

    def init_signals(self):
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)

    # --- PUBLIC API ---

    def load_mistakes(self, mistakes: list[Mistake]):
        """Populate the tree with a new list of mistakes."""
        self._mistakes = mistakes
        self._override_buttons = {}
        self.tree.clear()
        for idx, mistake in enumerate(mistakes):
            item = self._make_item(idx, mistake)
            self.tree.addTopLevelItem(item)
            btn = self._make_override_button(idx, mistake)
            self.tree.setItemWidget(item, 5, btn)
            self._override_buttons[idx] = btn

    def clear(self):
        self._mistakes = []
        self._override_buttons = {}
        self.tree.clear()

    def refresh_override(self, idx: int):
        """Update the override button appearance for a single mistake."""
        btn = self._override_buttons.get(idx)
        if btn is None or idx >= len(self._mistakes):
            return
        overridden = self._mistakes[idx].is_overridden()
        btn.setText("Overridden" if overridden else "Override")
        btn.setStyleSheet("color: #888;" if overridden else "")

    # --- INTERNAL ---

    _TYPE_ABBREV = {"insertion": "INS", "deletion": "DEL", "substitution": "SUB"}

    @staticmethod
    def _note_name(note) -> str:
        if note is None:
            return "—"
        midi = note.midi_num
        val = midi[0] if isinstance(midi, (list, tuple)) and len(midi) > 0 else midi
        return _midi_to_name(val)

    def _make_item(self, idx: int, mistake: Mistake) -> QTreeWidgetItem:
        pair = str(mistake.pair_index) if mistake.pair_index >= 0 else "—"
        intended = self._note_name(mistake.midi_note)
        actual = self._note_name(mistake.user_note)
        type_label = self._TYPE_ABBREV.get(mistake.type, mistake.type)

        item = QTreeWidgetItem([str(idx), pair, type_label, intended, actual, ""])
        item.setData(0, Qt.ItemDataRole.UserRole, idx)
        for col in range(5):
            item.setTextAlignment(col, Qt.AlignmentFlag.AlignCenter)
        return item

    def _make_override_button(self, idx: int, mistake: Mistake) -> QPushButton:
        overridden = mistake.is_overridden()
        btn = QPushButton("Overridden" if overridden else "Override")
        if overridden:
            btn.setStyleSheet("color: #888;")
        btn.clicked.connect(lambda: self.override_toggled.emit(idx))
        return btn

    def _on_selection_changed(self):
        item = self.tree.currentItem()
        if item is None:
            return
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is not None:
            self.selected.emit(idx)
