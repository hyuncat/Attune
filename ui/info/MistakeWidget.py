from pathlib import Path

from PyQt6.QtCore import Qt, QRectF, QSize, QEvent, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QTreeWidget,
    QTreeWidgetItem, QStyledItemDelegate, QStyle, QStyleOptionViewItem,
    QApplication,
)
from app_logic.Alignment import Mistake

_ICON_DIR = Path(__file__).resolve().parents[2] / "resources" / "icons"

# per-row flag (read off column 0) marking an overridden/dismissed mistake, so
# the delegate can darken the whole row
_OVERRIDE_ROLE = Qt.ItemDataRole.UserRole + 1


def _svg_icon(filename: str, px: int = 64) -> QIcon:
    """Rasterize an SVG (already white-colored on disk) into a QIcon, rendered
    at `px` so it stays crisp when the view scales it down. The SVG's aspect
    ratio is preserved and centered within the square pixmap so the tall, narrow
    flat/sharp glyphs aren't stretched."""
    renderer = QSvgRenderer(str(_ICON_DIR / filename))
    pix = QPixmap(px, px)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    size = renderer.defaultSize()
    w, h = size.width(), size.height()
    if w > 0 and h > 0:
        scale = min(px / w, px / h)
        tw, th = w * scale, h * scale
        renderer.render(painter, QRectF((px - tw) / 2.0, (px - th) / 2.0, tw, th))
    else:
        renderer.render(painter)
    painter.end()
    return QIcon(pix)


class _CenteredIconDelegate(QStyledItemDelegate):
    """Draws the cell's decoration icon centered (the default delegate left-
    aligns it). Used for the icon-only Type and Override columns. A single
    delegate handles every row, so we avoid creating N per-row native widgets —
    which the comment on MistakeWidget warns starves the GPU process on macOS."""

    def __init__(self, columns: set[int], override_brush: QBrush,
                 override_fg: QColor, icon_px: int = 18, parent=None):
        super().__init__(parent)
        self._columns = set(columns)
        self._icon_px = icon_px
        self._override_brush = override_brush
        self._override_fg = override_fg

    def set_icon_columns(self, columns: set[int]):
        """Which columns draw a centered icon vs. text. The set changes per
        MistakeWidget mode: the Type column is icon-only for pitch mistakes but
        holds a text label ("Too long" etc.) for timing mistakes."""
        self._columns = set(columns)

    def paint(self, painter, option, index):
        overridden = bool(index.data(_OVERRIDE_ROLE))
        is_icon_col = index.column() in self._columns

        if not overridden:
            # default path: text columns rendered by the base delegate, icon
            # columns drawn with a centered (white) glyph
            if is_icon_col:
                self._paint_icon_cell(painter, option, index, tint=None)
            else:
                super().paint(painter, option, index)
            return

        # --- overridden row: fully custom-drawn so the grey survives selection.
        # The palette can't dim selected text here because qdarktheme sets the
        # selected-text color via the `selection-color` style-sheet property,
        # which QStyleSheetStyle uses instead of the palette. So we draw the
        # selection chrome with no text, then paint the text/icon ourselves grey.
        # First darken the (unselected) background; a selected row's blue fill is
        # drawn over it by the chrome below.
        painter.fillRect(option.rect, self._override_brush)
        if is_icon_col:
            self._paint_icon_cell(painter, option, index, tint=self._override_fg)
        else:
            self._paint_text_cell(painter, option, index, self._override_fg)

    def _paint_icon_cell(self, painter, option, index, tint):
        """Draw the cell chrome (background/selection) with a centered icon. The
        icon is recolored to `tint` when given (overridden rows), else drawn as-is."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.icon = QIcon()
        widget = opt.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(icon, QIcon) and not icon.isNull():
            s = self._icon_px
            r = option.rect
            x = r.x() + (r.width() - s) // 2
            y = r.y() + (r.height() - s) // 2
            if tint is not None:
                self._draw_tinted_icon(painter, icon, x, y, s, tint)
            else:
                icon.paint(painter, x, y, s, s)

    def _paint_text_cell(self, painter, option, index, color: QColor):
        """Draw the cell chrome (background/selection) then the text in `color`,
        bypassing the style's own (selection-aware) text coloring."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = opt.text
        opt.text = ""
        widget = opt.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, opt, widget
        )
        elided = opt.fontMetrics.elidedText(
            text, Qt.TextElideMode.ElideRight, text_rect.width()
        )
        painter.save()
        painter.setPen(color)
        painter.setFont(opt.font)
        painter.drawText(text_rect, int(opt.displayAlignment), elided)
        painter.restore()

    @staticmethod
    def _draw_tinted_icon(painter, icon: QIcon, x: int, y: int, s: int, color: QColor):
        """Draw `icon` recolored to a flat `color`, preserving its alpha mask, so
        the icons on overridden rows match the dimmed text."""
        pm = icon.pixmap(s, s)
        tp = QPainter(pm)
        tp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        tp.fillRect(pm.rect(), color)
        tp.end()
        painter.drawPixmap(x, y, pm)


#converts number of seconds to a well-formatted time to display to the user in the mistake widget
def _format_time(seconds: float) -> str:
    #Convert seconds to decimal notation if under a minute, MM:SS for over a minute.
    if seconds < 60:
        return f"{seconds:.2f}"
    else:
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}:{secs:02d}"


class MistakeWidget(QWidget):
    """
    Right-side panel listing the analyzed mistakes for the active recording, with
    a "Mistakes:" Pitch / Timing dropdown that swaps which kind is shown (both
    share this one tree). Row indices always refer to the list currently *in
    view* — read it back with mistakes_in_view() rather than indexing
    alignment.pitch_mistakes.

    PITCH mode columns:  Index | Time | Type | Intended | Actual | Override
      Type is an icon: plus (insertion), minus (deletion), flat/sharp
      (substitution, depending on whether the user played under/over the target).

    TIMING mode columns: Index | Time | Type | Note | Amount | Override
      Type is a TEXT label ("Too long" / "Too short" / "Early" / "Late"). Amount
      is a signed-seconds deviation (onset offset for early/late, duration
      difference for too long/short). Timing mistakes are built during alignment.

    Both modes share an Override column: a trash-can to dismiss a flagged mistake,
    an undo arrow to undo. The Type/Override icon cells are rendered via item
    decorations + a centered-icon delegate (rather than an embedded widget per
    row). This avoids the macOS-specific GPU pressure of creating N native widgets
    when there are many mistakes — which on long pieces was starving the
    QtWebEngine GPU process and causing segfaults on resize/repaint. (The Type
    column is text, not an icon, in timing mode — the delegate's icon-column set
    is swapped per mode, see _apply_headers.)
    """
    selected = pyqtSignal(int)         # emits row index (into the in-view list) on row click
    override_toggled = pyqtSignal(int) # emits row index (into the in-view list) when Override cell is clicked
    cleared = pyqtSignal()             # emits when the selection is cleared (e.g. click empty space)
    mode_changed = pyqtSignal(str)     # "pitch" or "timing"

    _TYPE_COL = 2
    _OVERRIDE_COL = 5

    # mistake.type values that belong to the Timing tab, mapped to their labels
    _TIMING_LABELS = {"long": "Too long", "short": "Too short",
                      "early": "Early", "late": "Late"}
    _TIMING_TYPES = frozenset(_TIMING_LABELS)

    # per-mode column widths (each sums to the same total so the fixed-width panel
    # stays valid); timing needs a wider Type for the text labels.
    _PITCH_WIDTHS = [34, 50, 38, 72, 64, 36]
    _TIMING_WIDTHS = [34, 50, 70, 42, 62, 36]
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

    # translucent dark tint laid over an overridden (dismissed) row so it reads
    # as "set aside" — pairs with the green highlight the same note gets in
    # GuitarHero. A null brush clears it back to the default row background.
    _OVERRIDE_BG = QBrush(QColor(0, 0, 0, 115))
    # grey the row's text + icons when overridden (qdarktheme's disabled color)
    _OVERRIDE_FG = QColor(105, 113, 119)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)

        # the list currently shown in the tree (pitch or timing); kept in sync by
        # _populate so refresh_override / mistakes_in_view stay mode-correct
        self._mistakes: list[Mistake] = []
        self._pitch_mistakes: list[Mistake] = []
        self._timing_mistakes: list[Mistake] = []
        self._timing_mode = False

        # icons are fixed, so build them once and share across rows
        self._icons = {
            "insertion": _svg_icon("plus.svg"),
            "deletion": _svg_icon("minus.svg"),
            "flat": _svg_icon("flatsign.svg"),
            "sharp": _svg_icon("sharpsign.svg"),
            "trash": _svg_icon("trash-2.svg"),
            "undo": _svg_icon("undo-2.svg"),
            "pencil": _svg_icon("pencil.svg"),
        }

        self.init_ui()
        self.init_signals()

    def init_ui(self):
        # "Mistakes:" + a dropdown to pick Pitch vs Timing, on one line. Combo
        # index 0 == Pitch, 1 == Timing (mirrors self._timing_mode).
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self.header_label = QLabel("Mistakes:")
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Pitch", "Timing"])
        # size the combo (button + popup) to its widest item so neither truncates
        self._mode_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._mode_combo.setMinimumContentsLength(
            max(len(self._mode_combo.itemText(i))
                for i in range(self._mode_combo.count())))
        self._mode_combo.setStyleSheet(self._COMBO_STYLE)
        self._fit_mode_combo_to_contents()
        header_row.addWidget(self.header_label)
        header_row.addWidget(self._mode_combo)
        header_row.addStretch(1)
        self._layout.addLayout(header_row)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["#", "Time", "Type", "Intended", "Actual", ""])
        self.tree.setIndentation(0)
        self.tree.setRootIsDecorated(False)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.tree.setIconSize(QSize(20, 20))
        self._delegate = _CenteredIconDelegate(
            {self._TYPE_COL, self._OVERRIDE_COL},
            self._OVERRIDE_BG,
            self._OVERRIDE_FG,
            parent=self.tree,
        )
        self.tree.setItemDelegate(self._delegate)

        for col, w in enumerate(self._PITCH_WIDTHS):  # pitch is the default mode
            self.tree.setColumnWidth(col, w)

        # prevent the last column from stretching to fill remaining header width
        self.tree.header().setStretchLastSection(False)

        # center all column header labels
        self.tree.header().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        # pencil icon in the override column header
        self.tree.headerItem().setIcon(self._OVERRIDE_COL, self._icons["pencil"])

        self._layout.addWidget(self.tree)

        col_total = sum(self.tree.columnWidth(i) for i in range(self.tree.columnCount()))
        m = self._layout.contentsMargins()
        content_width = col_total + m.left() + m.right() + self.tree.frameWidth() * 2
        self.setMinimumWidth(content_width)
        self.setMaximumWidth(content_width)

    def init_signals(self):
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemClicked.connect(self._on_item_clicked)
        # clicking empty space in the tree should clear the selection (and thus the
        # GuitarHero highlight); QTreeWidget doesn't do this on its own
        self.tree.setMouseTracking(True)
        self.tree.viewport().installEventFilter(self)

    # --- PUBLIC API ---
    def load_mistakes(self, mistakes: list[Mistake]):
        """Set the PITCH mistake list (and show it if Pitch mode is active)."""
        self._pitch_mistakes = mistakes
        if not self._timing_mode:
            self._populate(mistakes)

    def load_timing_mistakes(self, mistakes: list[Mistake]):
        """Set the TIMING mistake list (and show it if Timing mode is active)."""
        self._timing_mistakes = mistakes
        if self._timing_mode:
            self._populate(mistakes)

    def mistakes_in_view(self) -> list[Mistake]:
        """The list the tree is currently showing — what a `selected` index maps
        to. Use this rather than indexing alignment.pitch_mistakes, since Timing-mode
        rows aren't part of that list."""
        return self._mistakes

    def is_timing_mode(self) -> bool:
        """True when the Timing list is shown (so the host overrides the right
        list: timing mistakes vs. alignment.pitch_mistakes)."""
        return self._timing_mode

    def _populate(self, mistakes: list[Mistake]):
        """Render `mistakes` into the tree, replacing whatever was there."""
        self._mistakes = mistakes
        self.tree.clear()
        # Batch the inserts: addTopLevelItems is much cheaper than N calls
        # to addTopLevelItem when the model is being watched.
        items = [self._make_item(idx, m) for idx, m in enumerate(mistakes)]
        self.tree.addTopLevelItems(items)

    def _fit_mode_combo_to_contents(self):
        metrics = self._mode_combo.fontMetrics()
        width = max(
            metrics.horizontalAdvance(self._mode_combo.itemText(i))
            for i in range(self._mode_combo.count())
        ) + 46
        self._mode_combo.setMinimumWidth(width)
        self._mode_combo.view().setMinimumWidth(width)
        self._mode_combo.view().setTextElideMode(Qt.TextElideMode.ElideNone)

    def clear(self):
        self._mistakes = []
        self._pitch_mistakes = []
        self._timing_mistakes = []
        self.tree.clear()

    def refresh_override(self, idx: int):
        """Update the override-cell appearance for a single mistake (in-view list)."""
        if not (0 <= idx < len(self._mistakes)):
            return
        item = self.tree.topLevelItem(idx)
        if item is None:
            return
        self._set_override_cell(item, self._mistakes[idx].is_overridden())

    # --- INTERNAL ---
    @staticmethod
    def _note_name(note) -> str:
        if note is None:
            return "—"
        return note.get_note_name()

    def _type_icon_and_tip(self, mistake: Mistake) -> tuple[QIcon, str]:
        """Pick the Type-column icon + tooltip for a mistake."""
        if mistake.type == "insertion":
            return self._icons["insertion"], "Insertion (extra note played)"
        if mistake.type == "deletion":
            return self._icons["deletion"], "Deletion (note missed)"
        # substitution: flat if the user played under the target pitch, sharp if over
        user = mistake.user_note.midi_num[0] if mistake.user_note else 0
        target = mistake.midi_note.midi_num[0] if mistake.midi_note else 0
        if user < target:
            return self._icons["flat"], "Substitution (played flat)"
        return self._icons["sharp"], "Substitution (played sharp)"

    def _make_item(self, idx: int, mistake: Mistake) -> QTreeWidgetItem:
        if mistake.type in self._TIMING_TYPES:
            return self._make_timing_item(idx, mistake)
        #time is based on the MIDI Note rather than the user note
        time = _format_time(mistake.midi_note.start_time) if mistake.midi_note else "—"
        intended = "—" if mistake.type == "insertion" else self._note_name(mistake.midi_note)
        actual = "—" if mistake.type == "deletion" else self._note_name(mistake.user_note)

        item = QTreeWidgetItem([str(idx), time, "", intended, actual, ""])
        item.setData(0, Qt.ItemDataRole.UserRole, idx)
        for col in range(6):
            item.setTextAlignment(col, Qt.AlignmentFlag.AlignCenter)

        type_icon, type_tip = self._type_icon_and_tip(mistake)
        item.setIcon(self._TYPE_COL, type_icon)
        item.setToolTip(self._TYPE_COL, type_tip)

        self._set_override_cell(item, mistake.is_overridden())
        return item

    def _make_timing_item(self, idx: int, mistake: Mistake) -> QTreeWidgetItem:
        """Build a Timing-mode row: # | Time | Type(text) | Note | Amount | Override.
        A timing mistake is always a matched pair, so both notes share a pitch —
        we show that note name once and the signed-seconds deviation as Amount."""
        time = _format_time(mistake.midi_note.start_time) if mistake.midi_note else "—"
        note = self._note_name(mistake.midi_note)
        label = self._TIMING_LABELS.get(mistake.type, mistake.type)
        item = QTreeWidgetItem([str(idx), time, label, note, mistake.info, ""])
        item.setData(0, Qt.ItemDataRole.UserRole, idx)
        for col in range(6):
            item.setTextAlignment(col, Qt.AlignmentFlag.AlignCenter)
        self._set_override_cell(item, mistake.is_overridden())
        return item

    def _set_override_cell(self, item: QTreeWidgetItem, overridden: bool):
        if overridden:
            item.setIcon(self._OVERRIDE_COL, self._icons["undo"])
            item.setToolTip(self._OVERRIDE_COL, "Undo override")
        else:
            item.setIcon(self._OVERRIDE_COL, self._icons["trash"])
            item.setToolTip(self._OVERRIDE_COL, "Override (dismiss this mistake)")
        # flag every column so the delegate darkens the whole row, and each cell
        # repaints when toggled live (see _CenteredIconDelegate)
        for col in range(self.tree.columnCount()):
            item.setData(col, _OVERRIDE_ROLE, overridden)

    def _on_selection_changed(self):
        # read the selection, not currentItem(): clicking empty space clears the
        # selection but leaves currentItem set, so we'd never detect the deselect.
        items = self.tree.selectedItems()
        if not items:
            self.cleared.emit()
            return
        idx = items[0].data(0, Qt.ItemDataRole.UserRole)
        if idx is not None:
            self.selected.emit(idx)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        if column != self._OVERRIDE_COL:
            return
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is not None:
            self.override_toggled.emit(idx)

    def _on_mode_changed(self, index: int):
        """Switch the tree between the Pitch and Timing lists. Repopulating clears
        the selection (so GuitarHero drops its highlight via `cleared`)."""
        self._timing_mode = (index == 1)
        self._apply_headers()
        self._populate(self._timing_mistakes if self._timing_mode else self._pitch_mistakes)
        self.mode_changed.emit("timing" if self._timing_mode else "pitch")

    def _apply_headers(self):
        """Swap the column headers, header tooltips, column widths, and the
        delegate's icon-column set to match the active mode. Both modes keep the
        Override column (pencil header icon); only timing makes Type a text
        column."""
        if self._timing_mode:
            self.tree.setHeaderLabels(["#", "Time", "Type", "Note", "Amount", ""])
            widths = self._TIMING_WIDTHS
            icon_cols = {self._OVERRIDE_COL}            # Type is text in timing mode
            type_tip = "The type of timing mistake the user made"
            note_tip = "The note the user played"
        else:
            self.tree.setHeaderLabels(["#", "Time", "Type", "Intended", "Actual", ""])
            widths = self._PITCH_WIDTHS
            icon_cols = {self._TYPE_COL, self._OVERRIDE_COL}
            type_tip = note_tip = ""
        # setHeaderLabels REBUILDS the header item, so (re)apply everything that
        # lives on it afterwards: the override pencil icon + its tooltip, the
        # per-mode column tooltips, and centered alignment.
        header = self.tree.headerItem()
        header.setIcon(self._OVERRIDE_COL, self._icons["pencil"])
        header.setToolTip(self._OVERRIDE_COL, "Override the user mistake")
        header.setToolTip(self._TYPE_COL, type_tip)
        header.setToolTip(3, note_tip)  # "Note" / "Intended" column
        self.tree.header().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self._delegate.set_icon_columns(icon_cols)
        for col, w in enumerate(widths):
            self.tree.setColumnWidth(col, w)

    def eventFilter(self, obj, event):
        if obj is self.tree.viewport():
            if event.type() == QEvent.Type.MouseMove:
                index = self.tree.indexAt(event.position().toPoint())
                if index.isValid() and index.column() == self._OVERRIDE_COL:
                    obj.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    obj.unsetCursor()
            elif event.type() == QEvent.Type.Leave:
                obj.unsetCursor()
        # a press on empty tree space (no item under the cursor) clears the
        # selection, which fires _on_selection_changed -> `cleared`
        if (obj is self.tree.viewport()
                and event.type() == QEvent.Type.MouseButtonPress
                and self.tree.itemAt(event.position().toPoint()) is None):
            self.tree.clearSelection()
        return super().eventFilter(obj, event)
