from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QWidget

from app_logic.user.NoteInfo import NoteInfo
from ui.info.Gradient import VolumeGradient
from ui.info.Legend import Legend


class NotePopupGH(QFrame):
    """Small popup next to the cursor with a clicked GuitarHero user note's
    characteristics (see NoteInfo). Qt.Popup, so any click outside dismisses
    it — and so it takes the keyboard while it's up, which is what lets the
    left/right arrows walk the take note by note (`stepped`) without a global
    shortcut fighting the trees and text fields for those keys.

    One popup serves every note: the host re-fills it via set_info rather than
    reopening it per note. (The ScoreViewer's counterpart is NotePopupSV.)"""

    stepped = pyqtSignal(int)  # -1 / +1: the arrow keys, ie show the prev/next note
    closed = pyqtSignal()      # dismissed; the host drops the note's highlight

    # mistake.type -> display label, mirroring the MistakeWidget timing tab
    _TIMING_LABELS = {"early": "Early", "late": "Late",
                      "long": "Too long", "short": "Too short"}

    _STEP_KEYS = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}

    def __init__(self, parent=None):
        super().__init__(parent, flags=Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("""
            NotePopupGH {
                background-color: rgb(32, 33, 38);
                border: 1px solid rgb(95, 95, 105);
                border-radius: 6px;
            }
            QLabel {
                color: rgb(228, 231, 235);
                font-size: 12px;
                background: transparent;
                border: none;
            }
        """)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 9, 12, 9)
        self._layout.setSpacing(3)

    def set_info(self, chars: NoteInfo):
        """(Re)fill the rows for `chars`. The arrow keys cycle notes through this
        one popup, so the rows are rebuilt in place rather than reopened."""
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget is not None:
                widget.hide()  # deleteLater is deferred; hide now so no stale flash
                widget.deleteLater()

        for row in self._rows(chars):
            self._layout.addWidget(row)
        self.adjustSize()

    def _label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.RichText)
        return label

    def _rows(self, chars: NoteInfo) -> list[QWidget]:
        rows = [self._label(f"<b>Pitch:</b> {chars.note_name} {chars.cents:+.0f}¢")]

        onset = f"<b>Onset:</b> {chars.onset:.2f}s"
        if chars.onset_mistake:
            onset += f" ({self._TIMING_LABELS[chars.onset_mistake]})"
        rows.append(self._label(onset))

        duration = f"<b>Duration:</b> {chars.duration:.2f}s"
        if chars.duration_mistake:
            duration += f" ({self._TIMING_LABELS[chars.duration_mistake]})"
        rows.append(self._label(duration))

        if chars.vibrato_rate_hz is not None:
            rate_text = format(chars.vibrato_rate_hz, ".3g")
            rows.append(self._label(f"<b>Vibrato:</b> f={rate_text}Hz, "
                                    f"W={chars.vibrato_width_cents:.0f}¢ peak-to-peak"))
        else:
            rows.append(self._label("<b>Vibrato:</b> —"))

        rows.extend(self._volume_rows(chars))
        return rows

    def _volume_rows(self, chars: NoteInfo) -> list[QWidget]:
        """Loudness is shown RELATIVE only — the ramp handle sits within the
        take's own [quietest, loudest] range (gain-invariant, and it matches the
        note's color). No absolute dB number: a computer mic is uncalibrated, and
        an absolute dBFS reading contradicts the take-relative color ramp."""
        if chars.volume_frac is None:
            return [self._label("<b>Volume:</b> —")]
        return [
            # self._label("<b>Volume:</b>"),
            Legend.gradient_strip(VolumeGradient(frac=chars.volume_frac), help=False),
        ]

    def popup_at(self, global_pos: QPoint):
        """Show next to the cursor, nudged to stay on screen."""
        self.adjustSize()
        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        x = min(global_pos.x() + 12, geo.right() - self.width())
        y = min(global_pos.y() + 12, geo.bottom() - self.height())
        self.move(max(geo.left(), x), max(geo.top(), y))
        self.show()

    def keyPressEvent(self, event):
        """Left/right walk to the neighboring note. Everything else falls through
        to QWidget, which closes a Qt.Popup on Escape."""
        step = self._STEP_KEYS.get(event.key())
        if step is None:
            super().keyPressEvent(event)
            return
        event.accept()
        self.stepped.emit(step)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
