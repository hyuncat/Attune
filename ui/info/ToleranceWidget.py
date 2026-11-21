from PyQt6.QtCore import Qt, QSize, QEvent, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox,
    QSizePolicy, QToolTip, QSlider
)

from ui.info.MistakeWidget import _svg_icon

class ToleranceWidget(QWidget):
    """
    Small panel beneath the MistakeWidget for tuning whichever tolerance belongs
    to the current mistake tab.

    Pitch mode presents the pitch-mistake tolerance in semitone units. Timing mode
    presents the post-alignment timing tolerance in seconds. In both modes the
    line edit is the source of truth and accepts any non-negative number; the
    slider is a bounded convenience control.
    """
    tolerance_applied = pyqtSignal(str, float)  # mode ("pitch"/"timing"), raw value

    def __init__(self, parent=None):
        super().__init__(parent)

        self._PITCH_HELP_TEXT = "How close to the intended note (in semitones)\nthe user can play to be counted correct.\n1 = Nearest semitone, 2 = Nearest whole step, etc."
        self._TIMING_HELP_TEXT = "How far off +/- the user's note can vary from the score in timing."
        # Pitch is shown/edited in semitones, one-to-one with the raw Config
        # pitch_tolerance: 1 semitone displayed == 1.0 pitch_tolerance == 1 MIDI
        # step. The slider uses tenths of a semitone; the box retains arbitrary
        # non-negative values, including values outside the slider range.
        self._TOL_PER_SEMITONE = 1.0
        self._PITCH_SLIDER_MIN = 3   # 0.3 semitones
        self._PITCH_SLIDER_MAX = 30  # 3.0 semitones
        self._DEFAULT_SEMITONES = 0.5  # -> 0.5 tolerance (matches Config.pitch_tolerance)
        self._TIMING_SLIDER_MIN = 1    # hundredths of a second, i.e. 0.01s
        self._TIMING_SLIDER_MAX = 100  # 1.00s
        self._DEFAULT_TIMING = 0.25

        self._mode = "pitch"
        self._pitch_tolerance = self._semitones_to_tolerance(self._DEFAULT_SEMITONES)
        self._timing_tolerance = self._DEFAULT_TIMING

        self.init_ui()
    
    def init_ui(self):
        self._check_icon = _svg_icon("check.svg")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        help_icon = _svg_icon("circle-help.svg", px=32)
        self.help_label = QLabel()
        self.help_label.setPixmap(help_icon.pixmap(QSize(14, 14)))
        self.help_label.setToolTip(self._help_text())
        self.help_label.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        # hover tooltips can silently fail to appear on macOS (see below), so the
        # icon is also clickable: a click forces the same text into a small popup.
        self.help_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_label.installEventFilter(self)
        layout.addWidget(self.help_label)

        self.label = QLabel("Tolerance:")
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Orientation.Horizontal)

        # square off the round themed handle into a rectangle
        self._ACCENT = "#8ab4f8" # match qdarktheme
        self.slider.setStyleSheet(
            "QSlider::handle:horizontal {"
            f"  background-color: {self._ACCENT};"
            "  width: 12px;"
            "  border-radius: 2px;"
            "}"
        )
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, 1)

        # editable readout — the source of truth on Apply.
        self.value_box = QLineEdit()
        self.value_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_box.setFixedWidth(48)
        self.value_box.editingFinished.connect(self._on_text_edited)
        layout.addWidget(self.value_box)

        self.apply_button = QPushButton()
        self.apply_button.setIcon(self._check_icon)
        self.apply_button.setIconSize(QSize(16, 16))
        self.apply_button.setFixedSize(QSize(28, 28))
        self.apply_button.setToolTip("Apply")
        self.apply_button.clicked.connect(self._on_apply)
        layout.addWidget(self.apply_button)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.setMaximumHeight(self.sizeHint().height())
        self._refresh_controls()

    def eventFilter(self, obj, event):
        """Clicking the help icon shows its text in a small popup, right on top
        of the icon — a reliable fallback for when hover tooltips don't fire."""
        if obj is self.help_label and event.type() == QEvent.Type.MouseButtonPress:
            QToolTip.showText(
                self.help_label.mapToGlobal(self.help_label.rect().topLeft()),
                self._help_text(),
                self.help_label,
            )
            return True
        return super().eventFilter(obj, event)

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:g}"

    def _read_display_value(self):
        """Parse the box (the source of truth); None if blank/invalid/negative."""
        try:
            value = float(self.value_box.text().strip())
        except ValueError:
            return None
        return value if value >= 0 else None

    def set_mode(self, mode: str):
        """Show the pitch or timing tolerance controls without emitting."""
        if mode not in ("pitch", "timing") or mode == self._mode:
            return
        self._mode = mode
        self._refresh_controls()

    def set_pitch_tolerance(self, tolerance: float):
        self._pitch_tolerance = max(0.0, float(tolerance))
        if self._mode == "pitch":
            self._refresh_controls()

    def set_timing_tolerance(self, tolerance: float):
        self._timing_tolerance = max(0.0, float(tolerance))
        if self._mode == "timing":
            self._refresh_controls()

    def set_tolerances(self, pitch_tolerance: float, timing_tolerance: float):
        self._pitch_tolerance = max(0.0, float(pitch_tolerance))
        self._timing_tolerance = max(0.0, float(timing_tolerance))
        self._refresh_controls()

    def _refresh_controls(self):
        self.label.setText("Tolerance (s):" if self._mode == "timing" else "Tolerance:")
        self.help_label.setToolTip(self._help_text())
        self.value_box.setToolTip(
            "Timing tolerance in seconds"
            if self._mode == "timing"
            else "Pitch tolerance in semitones"
        )
        self.slider.blockSignals(True)
        if self._mode == "timing":
            self.slider.setMinimum(self._TIMING_SLIDER_MIN)
            self.slider.setMaximum(self._TIMING_SLIDER_MAX)
            self.slider.setSingleStep(1)
            self.slider.setPageStep(10)
        else:
            self.slider.setMinimum(self._PITCH_SLIDER_MIN)
            self.slider.setMaximum(self._PITCH_SLIDER_MAX)
            self.slider.setSingleStep(1)
            self.slider.setPageStep(1)
        value = self._display_value()
        self.value_box.setText(self._fmt(value))
        self._sync_slider_to(value)
        self.slider.blockSignals(False)

    def _display_value(self) -> float:
        if self._mode == "timing":
            return self._timing_tolerance
        return self._tolerance_to_semitones(self._pitch_tolerance)

    def _sync_slider_to(self, display_value: float):
        """Move the slider to match `display_value`, clamped to its range, without
        looping back into the box (values past the range stay only in the box)."""
        if self._mode == "timing":
            slider_value = round(display_value * 100)
            clamped = min(max(slider_value, self._TIMING_SLIDER_MIN), self._TIMING_SLIDER_MAX)
        else:
            clamped = round(min(max(display_value * 10, self._PITCH_SLIDER_MIN), self._PITCH_SLIDER_MAX))
        self.slider.setValue(int(clamped))

    def _on_slider_changed(self, value: int):
        # user dragged the slider -> it becomes the source of truth.
        display_value = value / 100.0 if self._mode == "timing" else value / 10.0
        self.value_box.setText(self._fmt(display_value))

    def _on_text_edited(self):
        value = self._read_display_value()
        if value is None:
            return
        blocked = self.slider.blockSignals(True)
        self._sync_slider_to(value)
        self.slider.blockSignals(blocked)

    def _on_apply(self):
        value = self._read_display_value()
        if value is None:
            QMessageBox.warning(
                self, "Invalid tolerance",
                "Enter a non-negative number.",
            )
            return
        tolerance = self._display_to_tolerance(value)
        if self._mode == "timing":
            self._timing_tolerance = tolerance
        else:
            self._pitch_tolerance = tolerance
        self.tolerance_applied.emit(self._mode, tolerance)

    # helper helpers
    def _help_text(self) -> str:
        return self._TIMING_HELP_TEXT if self._mode == "timing" else self._PITCH_HELP_TEXT

    def _display_to_tolerance(self, value: float) -> float:
        if self._mode == "timing":
            return value
        return self._semitones_to_tolerance(value)

    def _semitones_to_tolerance(self, semitones: float) -> float:
        return semitones * self._TOL_PER_SEMITONE
    def _tolerance_to_semitones(self, tolerance: float) -> float:
        return tolerance / self._TOL_PER_SEMITONE
