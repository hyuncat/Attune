from PyQt6.QtCore import Qt, QRectF, QEvent
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QLabel, QWidget, QToolTip

from ui.Colors import Colors
from ui.Icons import svg_pixmap


class Gradient(QWidget):
    """Base for the read-only legend ramps: the groove holds the full gradient
    a subclass names via `anchors()`; `frac` adds the slider-style handle marking
    one value on it (None = plain legend strip, no handle). `dim` picks the
    score's knocked-back ramp (Colors.SCORE_DIM) — the GuitarHero surface uses
    the ramp full strength.

    Subclasses supply the ramp, the two end captions (`ends()`) and a default
    `HELP` blurb; `Legend.gradient_strip` lays those out around the groove."""
    GROOVE_H = 6
    HANDLE_W, HANDLE_H = 6, 12
    HELP = ""

    def __init__(self, frac: float | None = None, dim: bool = False,
                 help_text: str | None = None, parent=None):
        super().__init__(parent)
        self.frac = None if frac is None else min(max(frac, 0.0), 1.0)
        self.dim = dim
        self.help_text = self.HELP if help_text is None else help_text
        self.setFixedHeight(14)
        self.setMinimumWidth(140)

    def anchors(self) -> list:
        raise NotImplementedError

    def ends(self) -> tuple[QWidget, QWidget]:
        """(low, high) captions flanking the groove."""
        raise NotImplementedError

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        w, h = self.width(), self.height()
        mid = h / 2.0

        anchors = self.anchors()
        gradient = QLinearGradient(0, 0, w, 0)
        for i, rgb in enumerate(anchors):
            gradient.setColorAt(i / (len(anchors) - 1), QColor(*rgb))
        painter.setBrush(gradient)
        painter.drawRoundedRect(
            QRectF(0, mid - self.GROOVE_H / 2.0, w, self.GROOVE_H), 3, 3)

        if self.frac is not None:
            handle_x = min(max(self.frac * w - self.HANDLE_W / 2.0, 0.0), w - self.HANDLE_W)
            painter.setBrush(QColor(230, 230, 235))
            painter.drawRoundedRect(
                QRectF(handle_x, mid - self.HANDLE_H / 2.0, self.HANDLE_W, self.HANDLE_H), 2, 2)
        painter.end()


class VolumeGradient(Gradient):
    """Viridis volume strip (purple = quiet -> yellow-green = loud), flanked by
    the lucide volume / volume-2 icons."""

    HELP = ("How loud each note was played, relative to the rest of this take: "
            "purple is the quietest, yellow-green the loudest.")

    def anchors(self) -> list:
        return Colors.viridis_anchors(dim=self.dim)

    def ends(self) -> tuple[QWidget, QWidget]:
        quiet, loud = QLabel(), QLabel()
        quiet.setPixmap(svg_pixmap("volume.svg", 16))
        quiet.setToolTip("quiet")
        loud.setPixmap(svg_pixmap("volume-2.svg", 16))
        loud.setToolTip("loud")
        return quiet, loud


class PitchGradient(Gradient):
    """Plasma pitch-error strip (green = on-pitch -> red = way off), the same
    ramp the plot shades detected notes by distance-to-target with."""

    HELP = ("How far each note strayed from the pitch the score asks for. "
            "Green is on-pitch, red is off by a lot.")

    def anchors(self) -> list:
        return Colors.plasma_anchors(dim=self.dim)

    def ends(self) -> tuple[QWidget, QWidget]:
        return QLabel("correct"), QLabel("way off")


class VibratoGradient(Gradient):
    """Viridis legend for the metric coloring VibratoWidget's dots."""

    HELP = ("Dot color shows the other vibrato measurement: slow to fast when "
            "the graph displays width, or narrow to wide when it displays speed. "
            "The scale uses the minimum and maximum across the whole recording.")

    def __init__(self, low: str, high: str, parent=None):
        super().__init__(parent=parent)
        self.low = low
        self.high = high

    def anchors(self) -> list:
        return Colors.viridis_anchors()

    def ends(self) -> tuple[QWidget, QWidget]:
        return QLabel(self.low), QLabel(self.high)


class TimbreGradient(Gradient):
    """Magma spectrum-level legend relative to the mean played level."""

    HELP = ("Color shows energy at each pitch-frequency bin: black/purple is "
            "quiet and orange/yellow is strong spectral energy. Relative level (dB) "
            "uses the same mean voiced-frame reference as the Volume plot.")

    def __init__(self, low_db: float, high_db: float, parent=None):
        super().__init__(parent=parent)
        self._low_label, self._high_label = QLabel(), QLabel()
        self.set_levels(low_db, high_db)

    def anchors(self) -> list:
        return Colors.magma_anchors()

    def ends(self) -> tuple[QWidget, QWidget]:
        return self._low_label, self._high_label

    def set_levels(self, low_db: float, high_db: float):
        self.low_db, self.high_db = low_db, high_db
        for label, value in ((self._low_label, low_db), (self._high_label, high_db)):
            rounded = round(value)
            text = "0 dB" if rounded == 0 else f"{rounded:+d} dB"
            if label.text() != text:
                label.setText(text)
        self.update()

    def _mean_dot_rect(self):
        if not self.low_db <= 0.0 <= self.high_db or self.high_db <= self.low_db:
            return QRectF()
        # The mean is 0 dB, not necessarily the midpoint of the color scale.
        fraction = -self.low_db / (self.high_db - self.low_db)
        radius = 4.0
        x = min(max(fraction * self.width(), radius + 1),
                self.width() - radius - 1)
        return QRectF(x - radius, self.height() / 2 - radius,
                      2 * radius, 2 * radius)

    def event(self, event):
        if event.type() == QEvent.Type.ToolTip:
            dot = self._mean_dot_rect()
            if not dot.isEmpty() and dot.adjusted(-2, -2, 2, 2).contains(event.pos().x(), event.pos().y()):
                QToolTip.showText(event.globalPos(), "Mean volume", self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        dot = self._mean_dot_rect()
        if dot.isEmpty():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("white"), 1.5))
        painter.setBrush(QColor("black"))
        painter.drawEllipse(dot)
        painter.end()
