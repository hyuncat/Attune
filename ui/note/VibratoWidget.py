import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QComboBox

from ui.Colors import Colors
from ui.guitarhero.GuitarHero import GuitarHero
from ui.info.Gradient import VibratoGradient
from ui.info.Legend import Legend
from ui.note.NoteCurveWidget import NoteCurveWidget


class VibratoWidget(NoteCurveWidget):
    """Vibrato speed/width over the note under the cursor.

    Width is always the full peak-to-peak pitch excursion. The selected
    metric is the y value; viridis brushes encode the
    other metric. All signal analysis lives in VibratoDetector, so repainting
    only slices VibratoData and updates one ScatterPlotItem.
    """

    CONTOUR_ROLE = "vibrato"
    DEFAULT_RANGES = {
        "Width": (0.0, 200.0),
        "Speed": (0.0, 10.0),
    }
    Y_PADDING = 0.15
    HELP = ("Vibrato over the note under the cursor. Speed is the oscillation "
            "rate in Hz; Width is the full peak-to-peak pitch excursion in "
            "cents. Dot colors use the minimum and maximum "
            "across the whole recording for direct note-to-note comparison. "
            "The grey line is the measured pitch contour.")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Width", "Speed"])
        self.metric_combo.setStyleSheet(GuitarHero._COMBO_STYLE)
        self._metric = self.metric_combo.currentText()
        self._metric_ranges: dict[str, tuple[float, float]] = {}
        self.metric_combo.currentTextChanged.connect(self._on_metric_changed)

        self.brushes = Colors.viridis_brushes()
        self.curve = pg.PlotDataItem(
            x=[], y=[],
            pen=pg.mkPen(*Colors.viridis(0.55), 230, width=4),
            connect="finite",
        )
        self.curve.setZValue(2)
        self.plot.addItem(self.curve)
        self.points = pg.ScatterPlotItem(
            x=[], y=[], pen=pg.mkPen(None), size=5, brush=self.brushes[0])
        self.points.setZValue(3)
        self.plot.addItem(self.points)

        self.legend = None
        self._rebuild_legend()
        self._apply_axis()

    def header_widgets(self):
        return [self.metric_combo]

    def _on_metric_changed(self, text: str):
        current = self.current_y_range()
        if current is not None:
            self._metric_ranges[self._metric] = current
        self._metric = text
        saved = self._metric_ranges.get(text)
        if saved is None:
            self.reset_y_range()
        else:
            self.set_y_range(*saved)
        self._apply_axis()
        self._rebuild_legend()
        self.refresh()

    def _apply_axis(self):
        if self.metric_combo.currentText() == "Width":
            self.set_y_label("cents")
        else:
            self.set_y_label("Hz")

    def _rebuild_legend(self):
        if self.legend is not None:
            self._layout.removeWidget(self.legend)
            self.legend.hide()
            self.legend.deleteLater()
        if self.metric_combo.currentText() == "Width":
            gradient = VibratoGradient("slow", "fast")
        else:
            gradient = VibratoGradient("narrow", "wide")
        self.legend = Legend.gradient_strip(gradient, width=120, help=False)
        self._layout.addWidget(self.legend)

    def _render(self, t0: float, t1: float):
        self._apply_axis()
        metric = self.metric_combo.currentText()
        self.set_default_y_range(
            *self.DEFAULT_RANGES[metric], padding=self.Y_PADDING)
        data = getattr(self.recording, "vibrato_data", None)
        if data is None:
            self._render_blank()
            return
        times, rates, widths = data.curve(t0, t1)
        width_mode = self.metric_combo.currentText() == "Width"
        values = widths if width_mode else rates
        colors = rates if width_mode else widths
        mask = np.isfinite(times) & np.isfinite(values) & np.isfinite(colors)
        if not mask.any():
            self._render_blank()
            return
        visible_colors = colors[mask]
        color_metric = "rate" if width_mode else "width"
        color_range = data.global_characteristic_range(color_metric)
        if color_range is not None and color_range[1] > color_range[0]:
            color_min, color_max = color_range
            fractions = np.clip(
                (visible_colors - color_min) / (color_max - color_min),
                0.0,
                1.0,
            )
        else:
            fractions = np.full(len(visible_colors), 0.5)
        indices = np.rint(fractions * (len(self.brushes) - 1)).astype(int)
        brushes = [self.brushes[i] for i in indices]
        self.curve.setData(x=times[mask], y=values[mask], connect="finite")
        self.points.setData(x=times[mask], y=values[mask], brush=brushes)

    def _render_blank(self):
        self.curve.setData(x=[], y=[])
        self.points.setData(x=[], y=[])
