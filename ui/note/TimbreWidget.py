import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer, QRectF, Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel

from ui.Colors import Colors
from ui.guitarhero.MidiBackground import MidiAxis
from ui.info.Gradient import TimbreGradient
from ui.note.NoteCurveWidget import NoteCurveWidget
from ui.note.RelativeLevel import RelativeLevelStats


class TimbreWidget(NoteCurveWidget):
    """Semitone heatmap projected from the stored full FFT-bin spectrum."""

    CONTOUR_ROLE = "timbre"
    HELP = ("Timbre shows Hann-windowed FFT power grouped into semitone "
            "bands. Hotter colors mean more energy at that frequency; stacked "
            "bright bands are the fundamental and harmonics. The full FFT-bin "
            "spectrum is retained with the recording. The black line with a white outline is the "
            "detected pitch contour. Colors show relative level (dB), using the "
            "same average played level as Volume: 0 dB is the take's mean "
            "voiced-frame level. Individual frequency bands are usually below "
            "that overall level. During recording the average updates as you "
            "play, excluding unvoiced frames and silence; review uses the "
            "whole take. The outlined black dot marks the average (0 dB). "
            "The scale appears once a voiced level is available.")
    LIVE_LEVELS = (-60.0, 12.0)  # relative dB; stable color scale while live

    def _axis_items(self):
        return {"left": MidiAxis(orientation="left")}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setLookupTable(Colors.magma_lut())
        self.image.setZValue(0)
        self.plot.addItem(self.image)
        self.contour.setZValue(2)
        self.contour.setPen(pg.mkPen("black", width=2))
        self.contour.setShadowPen(pg.mkPen("white", width=4))
        self.set_y_label("Pitch")
        self._level_stats = RelativeLevelStats()
        self._gradient = TimbreGradient(*self.LIVE_LEVELS)
        self.legend = QWidget()
        legend_layout = QHBoxLayout(self.legend)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(8)
        legend_layout.addWidget(QLabel("Volume:"))
        low, high = self._gradient.ends()
        low.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        high.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        # Give spare width to the ramp, keeping both captions equally close.
        self._gradient.setMinimumWidth(60)
        legend_layout.addWidget(low)
        legend_layout.addWidget(self._gradient, 1)
        legend_layout.addWidget(high)
        self._layout.addWidget(self.legend)
        self.legend.hide()

        self._backfill_timer = QTimer(self)
        self._backfill_timer.setInterval(150)
        self._backfill_timer.timeout.connect(self._poll_backfill)

    def set_recording(self, rec):
        super().set_recording(rec)
        self._ensure_timbre()

    def refresh(self):
        self._level_stats.reset()
        super().refresh()
        self._ensure_timbre()

    def _ensure_timbre(self):
        rec = self.recording
        if rec is None or not rec.timbre_data.is_empty():
            return
        if rec.ensure_timbre():
            self._backfill_timer.start()

    def _poll_backfill(self):
        rec = self.recording
        if rec is None:
            self._backfill_timer.stop()
            return
        self._window = self._current_window()
        self._redraw()
        if not rec.spectrum_detector.is_running():
            self._backfill_timer.stop()

    def _render(self, t0: float, t1: float):
        data = self.recording.timbre_data
        y0, y1 = data.midi_bounds()
        self.set_default_y_range(y0, y1, padding=0.0)
        self._level_stats.sync(self.recording.pitch_data)
        reference_db = self._level_stats.reference_db
        if reference_db is None:
            self._render_blank()
            return
        # Shift both the spectrum and its review bounds by the SAME reference.
        # Live uses fixed relative bounds so gain changes do not dominate color.
        levels = (self.LIVE_LEVELS if self.live else
                  tuple(db - reference_db for db in data.range_db()))
        # Keep the mean visible even when every spectral band is below it.
        levels = (min(levels[0], 0.0), max(levels[1], 0.0))
        if levels[1] <= levels[0]:
            levels = (levels[0], levels[0] + 1.0)
        _times, matrix = data.display_matrix(t0, t1)
        if matrix.shape[1] == 0:
            self._render_blank()
            return
        self._gradient.set_levels(*levels)
        self.legend.show()
        self.image.setImage(matrix - reference_db, autoLevels=False, levels=levels)
        self.image.setRect(QRectF(t0, y0, max(t1 - t0, 1e-9), y1 - y0))

    def _render_blank(self):
        self.legend.hide()
        self.image.setImage(np.empty((0, 0)), autoLevels=False)

    def _contour_transform(self, times, midis):
        return times, midis
