import pyqtgraph as pg
from PyQt6.QtCore import Qt

from ui.Colors import Colors
from ui.note.NoteCurveWidget import NoteCurveWidget
from ui.note.RelativeLevel import RelativeLevelStats


class VolumeWidget(NoteCurveWidget):
    """Recorded level relative to the take's mean voiced-frame level in dB.

    Live statistics consume only newly written frames (not allocated capacity).
    Review uses the same reference across every note. Refresh invalidates the
    statistics after analysis edits; ordinary transport ticks keep them cached.
    """

    CONTOUR_ROLE = "volume"
    HELP = ("Shows how the recorded level changes during the note, relative to "
            "the average played level in this take. 0 dB is average; positive "
            "values are louder and negative values softer. During recording, "
            "the average updates as you play, excluding unvoiced frames and "
            "silence. The grey line is the pitch contour.")
    Y_PADDING = 0.15
    LIVE_RANGE_DB = (-24.0, 24.0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level_stats = RelativeLevelStats()
        self.reference_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(*Colors.NOTE_AXIS_RGB, 120, style=Qt.PenStyle.DashLine),
            label="Average", labelOpts={"position": 0.98,
                                         "anchors": [(1, 1), (1, 0)],
                                         "color": Colors.NOTE_AXIS_RGB})
        self.plot.addItem(self.reference_line, ignoreBounds=True)
        self.reference_line.hide()
        self.curve = pg.PlotDataItem(
            pen=pg.mkPen(*Colors.NOTE_VOLUME_RGB, 255, width=4),
            connect="finite")
        self.curve.setZValue(2)
        self.plot.addItem(self.curve)
        self.set_y_label("Relative volume (dB)")

    def refresh(self):
        self._level_stats.reset()
        super().refresh()

    def _render(self, t0: float, t1: float):
        self._level_stats.sync(self.recording.pitch_data)
        reference_db = self._level_stats.reference_db
        if reference_db is None:
            # No played level to compare with yet. Do not invent a reference
            # from silence or lock in a review y-range before data arrives.
            self._render_blank()
            return
        if self.live:
            bounds = self.LIVE_RANGE_DB
        else:
            bounds = (min(-6.0, self._level_stats.minimum - reference_db),
                      max(6.0, self._level_stats.maximum - reference_db))
        self.set_default_y_range(*bounds, padding=self.Y_PADDING)
        times, dbs = self.recording.pitch_data.volume_curve(
            t0, t1, floor_db=self._yr[0] + reference_db)
        self.curve.setData(times, dbs - reference_db, connect="finite")
        self.reference_line.show()

    def _render_blank(self):
        self.curve.setData([], [])
        self.reference_line.hide()
