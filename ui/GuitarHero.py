from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtGui import QColor
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
import qdarktheme

from app_logic.midi.ScoreData import ScoreData
from app_logic.user.ds.Recording import Recording
from app_logic.Alignment import Alignment

class MidiBackground(pg.ImageItem):
    """
    A custom ImageItem to display a fixed MIDI background
    Immutable background stripes (0..127 MIDI), built once!
    """
    def __init__(self):
        super().__init__(axisOrder='row-major')
        self.setZValue(-1)  # set as global -1 position
        # color mappings
        self.LETTER_RGB = {
            'A': (230,  60,  60),  # red
            'B': (255, 150,  40),  # orange
            'C': (245, 220,  70),  # yellow
            'D': ( 70, 200,  90),  # green
            'E': ( 70, 140, 240),  # blue
            'F': (100,  90, 210),  # indigo
            'G': (170,  90, 210),  # purple
        }
        # MIDI number to letter mapping
        self.N_MIDI = 128
        self.PC_TO_LETTER = {
            0:'C', 1:'C', 2:'D',
            3:'D', 4:'E', 5:'F',
            6:'F', 7:'G', 8:'G',
            9:'A', 10:'A', 11:'B'
        }
        self.midi_is_sharp = lambda m: (m % 12) in {1, 3, 6, 8, 10}
        self._init_bg()

    def midi_to_rgba(self, m, alpha=50):
        """Convert a MIDI number to an RGBA color tuple."""
        letter = self.PC_TO_LETTER[m % 12]
        r,g,b = self.LETTER_RGB[letter]

        if self.midi_is_sharp(m): # make sharps darker
            r = int(r*0.70)
            g = int(g*0.70)
            b = int(b*0.70)

        return (r, g, b, alpha)
    
    def _init_bg(self):
        """
        Build a fixed 0..127 MIDI RGBA texture (height=128 rows, one per MIDI).
        This never changes, so colors are locked to absolute MIDI.
        """
        # width can be tiny; GPU stretches it. Use width=2 for stability.
        arr = np.zeros((self.N_MIDI, 2, 4), dtype=np.ubyte)

        # generate color array for each MIDI number
        for m in range(self.N_MIDI): 
            r,g,b,a = self.midi_to_rgba(m)
            arr[m, :, 0] = r
            arr[m, :, 1] = g
            arr[m, :, 2] = b
            arr[m, :, 3] = a

        self.setImage(arr[:, :, :], autoLevels=False)

        # pin the image's Y rect to [0,128] forever
        # set X span to default dummy values
        xmin, xmax = -1, 4
        rect = pg.QtCore.QRectF(xmin, 0.325, xmax - xmin, 128.325)
        self.setRect(rect)
        self.update_x(xmin, xmax)

    def update_x(self, xmin: float, xmax: float):
        """
        Update the image's X span; uses setRect() to change 
        only X, keep Y fixed 0..128.

        Args:
            xmin (float): new minimum x-value
            xmax (float): new maximum x-value
        """
        # Keep Y locked to MIDI domain 0..128 (1 unit = 1 MIDI)
        rect = pg.QtCore.QRectF(xmin, 0.325, xmax - xmin, 128.325)
        self.setRect(rect)

class MidiAxis(pg.AxisItem):
    """
    Overloaded pyqtgraph AxisItem to display y-axis as note names
    rather than as raw MIDI numbers. Eg, 60 -> C4.
    """
    NOTE_NAMES = [
        'C', 'C#', 'D', 'D#', 'E', 'F',
        'F#', 'G', 'G#', 'A', 'A#', 'B'
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyle(autoExpandTextSpace=True)

    def tickValues(self, minVal, maxVal, size):
        """
        Return tick levels as [(majorStep, majorValues), (minorStep, minorValues)].
        We choose a semitone-based major step based on zoom span.
        """
        span = float(maxVal - minVal)
        if span <= 0:
            return []

        # target ~8–12 major labels depending on pixel height
        target_labels = max(6, min(12, int(size / 35)))

        # candidate steps in semitones
        candidates = np.array([1, 2, 3, 4, 6, 12, 24, 36, 48], dtype=int)
        # pick the smallest step that yields <= target_labels
        labels_per_span = span / candidates
        try:
            major_step = int(candidates[np.argmax(labels_per_span <= target_labels)])
            if labels_per_span.max() > target_labels and major_step == 0:
                major_step = int(candidates[-1])
        except Exception:
            major_step = 12  # sane default
        if major_step <= 0:
            major_step = 12

        # align majors to the step boundary
        start_major = int(np.floor(minVal / major_step) * major_step)
        end_major   = int(np.ceil (maxVal / major_step) * major_step)
        majors = np.arange(start_major, end_major + 1, major_step, dtype=int)

        # minors at 1 semitone (only when not too dense)
        if major_step >= 6:
            start_minor = int(np.floor(minVal))
            end_minor   = int(np.ceil (maxVal))
            minors = np.arange(start_minor, end_minor + 1, 1, dtype=int)
            # drop those that coincide with majors
            minors = minors[~np.isin(minors, majors)]
            return [(major_step, majors), (1, minors)]
        else:
            return [(major_step, majors)]

    def tickStrings(self, values, scale, spacing):
        """
        Label only the first tick level (majors). Pyqtgraph passes majors first.
        Values for minors will be ignored by this method for that level.
        """
        # values can be floats; they are exactly integers from our tickValues
        return [self.midi_to_name(int(round(v))) for v in values]

    @staticmethod
    def midi_to_name(m: int) -> str:
        """Convert MIDI number to name, e.g. 60 -> C4."""
        pitch = m % 12
        octave = (m // 12) - 1
        return f"{MidiAxis.NOTE_NAMES[pitch]}{octave}"

class GuitarHero(QWidget):
    def __init__(self, recording: Recording=None):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # important objects
        self.recording = recording
        self.score_data = recording.score_data if recording else None
        self.alignment: Alignment = recording.alignment if recording else None

        # --- TIMEKEEPING VARIABLES ---
        # windowing variables
        self.w, self.h = 5, 50 # seconds, midi numbers
        self.x_range, self.y_range = (-1, 4), (40, 90) # initial view ranges
        self.x_margin = 0.5 # 50% margin on the x-axis

        # timeline variables
        self.t = 0 # current time in seconds
        self.timeline_offset = 0.2 # x fraction of screen from left
        self.is_moving = False

        # ---- THE PLOT. ----
        self.y_axis = MidiAxis(orientation='left') # our custom y-axis
        self.plot = pg.PlotWidget(axisItems={'left': self.y_axis})
        self._layout.addWidget(self.plot)

        self.init_background()
        self.init_colors()
        self.init_objects()
        self.init_view()

    # --- INIT STUFF ---
    def init_background(self):
        self.plot.setBackground((20,20,25))
        self.bg = MidiBackground() # our colorful custom MIDI background :-)
        self.plot.addItem(self.bg, ignoreBounds=True) # don't let bg affect autorange

    def init_colors(self):
        """Define all colors used in the plot."""
        self.colors = {
            'midi': pg.mkBrush(255, 255, 255, 200), # white
            'user_note': pg.mkBrush(55, 155, 144, 150),
            'user_pitch': pg.mkBrush(41, 177, 240, 255), 
            'timeline': pg.mkPen(0, 255, 0, 255), # green
            'insertion': pg.mkBrush(0, 200, 0, 200), # green
            'deletion': pg.mkBrush(255, 0, 0, 200), # red
            'substitution': pg.mkBrush(255, 220, 0, 60) # translucent yellow
        }
        # new shit
        self.rest_brush = pg.mkBrush(140, 140, 140)
        self.distance_brushes = []
        self.max_dist = 5.0
        self.distance_step = 0.05  # 100 buckets for 0..5

        num_buckets = int(self.max_dist / self.distance_step) + 1
        for i in range(num_buckets):
            d = i * self.distance_step

            if d <= 0.5:
                hue = 120
            else:
                alpha = (d - 0.5) / (self.max_dist - 0.5)
                alpha = max(0.0, min(alpha, 1.0))
                hue = int(120 * (1.0 - alpha))
        
            color = QColor()
            color.setHsv(hue, 255, 255)
            self.distance_brushes.append(pg.mkBrush(color))

    def init_objects(self):
        """Initialize all foreground plot items, including:
            - MIDI notes
            - user notes
            - user pitches
            - the timeline
        """
        self.NOTE_HEIGHT = 0.5 # height of note rectangles
        # ---- foreground items ----
        self.midi_notes = pg.BarGraphItem(
            x=[], height=self.NOTE_HEIGHT, 
            width=[], y0=0, 
            brush=self.colors['midi'], pen=None
        )
        self.midi_notes.setZValue(1) # above bg

        self.midi_notes_del = pg.BarGraphItem(
            x=[], height=self.NOTE_HEIGHT, y0=0, width=[],
            brush=self.colors['deletion'], pen=None
        )
        self.midi_notes_del.setZValue(1.1)

        self.user_notes = pg.BarGraphItem(
            x=[], height=self.NOTE_HEIGHT, y0=0, width=[],
            brush=self.colors['user_note'], pen=None
        )
        self.user_notes.setZValue(2) # above midi notes

        self.user_notes_ins = pg.BarGraphItem(
             x=[], height=self.NOTE_HEIGHT, y0=0, width=[],
            brush=self.colors['insertion'], pen=None
        )
        self.user_notes_ins.setZValue(2.1)

        self.match_lines = pg.PlotDataItem(
            x=[], y=[],
            pen=pg.mkPen(255, 255, 255, 140, width=1.5, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.match_lines.setZValue(2.2)
        self.plot.addItem(self.match_lines)

        self.user_pitches = pg.ScatterPlotItem(
            x=[], y=[], pen=pg.mkPen(None), brush=self.colors['user_pitch'], size=8
        )
        self.user_pitches.setZValue(3) # above user notes
        self.timeline = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen(self.colors['timeline'])
        )
        self.timeline.setZValue(4) # above everything

        # add foreground after bg
        self.plot.addItem(self.midi_notes)
        self.plot.addItem(self.midi_notes_del)
        self.plot.addItem(self.user_notes)
        self.plot.addItem(self.user_notes_ins)
        self.plot.addItem(self.user_pitches)
        self.plot.addItem(self.timeline)

    def init_view(self):
        """initialize the viewbox settings"""
        vb = self.plot.getViewBox()
        self.plot.enableAutoRange('xy', False)
        vb.setLimits(yMin=0, yMax=128)

        # set initial ranges
        vb.setRange(xRange=self.x_range, yRange=self.y_range, padding=0)
        vb.sigRangeChanged.connect(self.update_zoom)


    # ---------- PAN/ZOOM HANDLING ----------
    def update_zoom(self, viewbox, view_range):
        """updates the zoom of the plot when the viewbox range changes
        also updates the background accordingly
        """
        # ignore update_zoom calls while moving to avoid error accumulation
        if self.is_moving:
            return

        xmin, xmax = self.plot.viewRange()[0]
        self.bg.update_x(xmin, xmax)

        # store the new ranges
        self.x_range = view_range[0]
        self.y_range = view_range[1]
        self.w = self.x_range[1] - self.x_range[0]
        self.h = self.y_range[1] - self.y_range[0]

        self.update_view_items()

    def move_plot(self, t: float):
        """Move the plot to time t (sec).
        Update the window boundaries, the background, and our viewbox range.
        Note that we keep a "is_moving" flag on to avoid accumulating errors 
        in the auto-zoom logic.

        Args:
            t (float): time in seconds to move the plot to
        """
        # print(f"--> Moving plot to {t} sec")
        self.is_moving = True # avoid accumulating errors in zoom

        # update the window boundaries
        self.t = t # update current time
        x_lower = self.t - (self.w*self.timeline_offset)
        x_upper = self.t + (self.w * (1-self.timeline_offset))
        self.x_range = (x_lower, x_upper)

        # update the background and our viewbox range
        self.bg.update_x(x_lower, x_upper)
        self.plot.getViewBox().setRange(xRange=self.x_range, yRange=self.y_range, padding=0)
        self.timeline.setPos(t) # also update the timeline pos

        self.is_moving = False # now we good
        self.update_view_items()


    # ---------- DATA LOADING ----------
    def load_score(self, score_data: ScoreData):
        """Load a MidiData object and display its notes."""
        print("Loading MIDI data into ScorePlot...")
        self.score_data = score_data
        self.recording = None
        self.alignment = None
        self.update_view_items()

    def load_user(self, recording: Recording):
        """Load a Recording object and display its notes and pitches."""
        print("Loading Recording into ScorePlot...")
        self.recording = recording
        self.score_data = recording.score_data
        self.alignment = recording.alignment
        self.update_view_items()

    def load_alignment(self, alignment: Alignment):
        """Plot the alignment results (user notes + mistakes)."""
        print("Plotting alignment...")
        self.alignment = alignment
        self.update_view_items()


    # --- THE ESSENTIAL PLOTTING STUFF (called every time we refresh the view) ---
    def update_view_items(self):
        """Force all view items to update/redraw. Called whenever:
        1. view range changes
        2. view items change
            2.1 midi/user data loaded
            2.2 alignment loaded
            2.3 pitch detected
        """
        PAD = 1
        xmin, xmax = self.plot.viewRange()[0]
        x_range = (xmin-PAD, xmax+PAD)
        # print(f"Updating view items for x_range={x_range}...")
        
        # --- USER PITCHES + NOTES UPDATING ---
        self.update_user_items(x_range)
        self.update_midi_items(x_range)
        self.update_alignment_items(x_range)


    def update_user_items(self, x_range: tuple[float, float]):
        """Update the currently plotted user items (pitches, notes) to fit the given x_range"""
        if self.recording is None:
            print("nothing :(")
            self.user_pitches.setData(x=[], y=[])
            self.user_notes.setOpts(x=[], width=[], y0=[], height=[])
            return
        
        # read the current pitches and notes in the x_range
        user_pitches = self.recording.pitch_data.read(x_range[0], x_range[1], clean=True) if self.recording else []
        user_notes = self.recording.note_data.read(
            start_time=x_range[0], 
            end_time=x_range[1], 
            clean=True
        ) if self.recording else []

        # --- update PITCHES ---
        xs, ys, brushes = [], [], []
        for p in user_pitches:
            for c in p.candidates:
                xs.append(p.time)
                ys.append(c[0]) # pitch value
                brushes.append(self.get_distance_brush(getattr(p, "distance", None)))
                break
                    
        # get_alpha = lambda p: int(50 + 205*(1 - p.candidates[0][1]))
        # alphas = np.asarray([get_alpha(p) for p in user_pitches], dtype=np.float32)
        # brushes = [pg.mkBrush(41, 177, 240, a) for a in alphas]

        self.user_pitches.setData(x=xs, y=ys, brush=brushes)

        if self.recording.note_data is None:
            self.user_notes.setOpts(x=[], width=[], y0=[], height=[])
            return
        
        # --- update NOTES ---
        # get the note parameters for the BarGraphItem
        # starts = np.asarray([n.start_time for n in user_notes], dtype=np.float64)
        # ends = np.asarray([n.end_time for n in user_notes], dtype=np.float64)
        # midis = np.asarray([n.midi_num for n in user_notes], dtype=np.float64)

        starts, ends, midis = [], [], []
        for n in user_notes:
            for m in n.midi_num:
                starts.append(n.start_time)
                ends.append(n.end_time)
                midis.append(m)
                break
        
        starts = np.array(starts, dtype=np.float64)
        ends = np.array(ends, dtype=np.float64)
        midis = np.array(midis, dtype=np.float64)


        x = 0.5 * (starts + ends) # each rect starts at the center
        width = (ends - starts) # width is duration
        y0 = (midis - 0.5*self.NOTE_HEIGHT) # bottom y-pos
        height = np.full_like(midis, self.NOTE_HEIGHT) # constant height

        self.user_notes.setOpts(x=x, width=width, y0=y0, height=height)

    def update_midi_items(self, x_range: tuple[float, float]):
        """Update the currently plotted MIDI items (notes) to fit the given x_range"""
        if self.score_data is None:
            self.midi_notes.setOpts(x=[], width=[], y0=[], height=[])
            return

        # --- MIDI NOTE UPDATING ---
        # read the current midi notes in the x_range
        note_data = self.score_data.note_datas.get(self.score_data.active_instrument, None)
        midi_notes = note_data.read(x_range[0], x_range[1]) if note_data else []

        # get the note parameters for the BarGraphItem
        starts = np.array([n.start_time for n in midi_notes], dtype=np.float32)
        ends = np.array([n.end_time for n in midi_notes], dtype=np.float32)
        midis = np.array([n.midi_num[0] for n in midi_notes], dtype=np.float32)

        # print(f"   plotting {len(midi_notes)} MIDI notes...")

        x = 0.5 * (starts + ends)
        width = (ends - starts)
        y0 = (midis - 0.5*self.NOTE_HEIGHT)
        height = np.full_like(midis, self.NOTE_HEIGHT)
        
        self.midi_notes.setOpts(x=x, width=width, y0=y0, height=height)

    def update_alignment_items(self, x_range: tuple[float, float]):
        """Update the alignment overlay items (insertions, deletions, match lines)
        to fit the given x_range."""
        if self.alignment is None:
            self.user_notes_ins.setOpts(x=[], width=[], y0=[], height=[])
            self.midi_notes_del.setOpts(x=[], width=[], y0=[], height=[])
            self.match_lines.setData(x=[], y=[])
            return
        
        # --- CORRECTIONS OVERLAY ---
        # retrieve all alignment related components for the xrange
        goods, subs, ins, dels = self.alignment.get_alignment(x_range[0], x_range[1])

        # ---> MATCH LINES --->
        matches = goods + subs
        xs, ys = [], []
        for n, m in matches:
            # compute midpoints for USER and MIDI notes
            ux = 0.5 * (n.start_time + n.end_time)
            uy = float(n.midi_num[0])
            mx = 0.5 * (m.start_time + m.end_time)
            my = float(m.midi_num[0])

            # rk: np.nan separates line segments from e/o
            xs.extend([ux, mx, np.nan]) 
            ys.extend([uy, my, np.nan])

        self.match_lines.setData(x=np.asarray(xs, dtype=np.float32),
                                y=np.asarray(ys, dtype=np.float32))
        # print("plotted alignment updates")

        # ---> USER INSERTIONS OVERLAY --->
        if ins:
            starts = np.asarray([n.start_time for n in ins], dtype=np.float64)
            ends   = np.asarray([n.end_time   for n in ins], dtype=np.float64)
            midis  = np.asarray([n.midi_num[0]   for n in ins], dtype=np.float64)

            x = 0.5 * (starts + ends)
            width = (ends - starts)
            y0 = (midis - 0.5*self.NOTE_HEIGHT)
            height = np.full_like(midis, self.NOTE_HEIGHT)

            self.user_notes_ins.setOpts(
                x=x, width=width, y0=y0, height=height
            )
        # ---> MIDI DELETIONS OVERLAY --->
        if dels:
            starts = np.asarray([n.start_time for n in dels], dtype=np.float32)
            ends   = np.asarray([n.end_time   for n in dels], dtype=np.float32)
            midis  = np.asarray([n.midi_num[0]   for n in dels], dtype=np.float32)

            x = 0.5 * (starts + ends)
            width = (ends - starts)
            y0 = (midis - 0.5*self.NOTE_HEIGHT)
            height = np.full_like(midis, self.NOTE_HEIGHT)

            self.midi_notes_del.setOpts(
                x=x, width=width, y0=y0, height=height
            )

    def get_distance_brush(self, d: float | None):
        if d is None:
            return self.rest_brush

        d = abs(float(d))
        d = min(d, self.max_dist)

        idx = int(d / self.distance_step)
        idx = min(idx, len(self.distance_brushes) - 1)
        return self.distance_brushes[idx]



class RunScorePlot:
    def __init__(self, recording: Recording=None, score_data: ScoreData=None, alignment: Alignment=None, app=None):
        import sys
        from PyQt6.QtWidgets import QApplication, QMainWindow

        if app is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = app

        self.main_window = QMainWindow()
        self.app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
        self.main_window.setWindowTitle("Score Plot")
        self.main_window.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout(self.central_widget)
        self.main_window.setCentralWidget(self.central_widget)

        # initialize the visualizer widget adding it to the layout
        self.vis = GuitarHero()
        self.central_layout.addWidget(self.vis)
        self.init_toolbar()

        if score_data is not None:
            self.vis.load_score(score_data)
        if recording is not None:
            self.vis.load_user(recording)
        if alignment is not None:
            self.vis.load_alignment(alignment)

        self.main_window.show()
        self.app.exec()

    def init_toolbar(self):
        from PyQt6.QtWidgets import QToolBar
        from PyQt6.QtCore import Qt

        self.toolbar = QToolBar()
        self.toolbar.setOrientation(Qt.Orientation.Horizontal)
        self.toolbar.addAction("Exit", self.close)
        self.main_window.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

    def close(self):
        self.app.quit()