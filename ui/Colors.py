import math
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import pyqtgraph as pg


class Colors:
    """THE theme: every color any UI surface draws with, in one editable place.

    Nothing else in `ui/` (or the Verovio page) should hold a literal color —
    widgets pull brushes/pens/ramps/role colors from here, and the web layer
    gets `score_theme()` pushed into its CSS custom properties. Edit a palette
    below and every surface follows.

    Two ramps carry meaning:
      - PLASMA = pitch error (green = on-pitch -> red = way off). Its end anchor
        is the score's "wrong note" red, so plot gradient and score noteheads
        can't drift apart.
      - VIRIDIS = volume (purple = quiet -> yellow-green = loud).
    Both are stored FULL strength (what the GuitarHero plot draws); the
    ScoreViewer dims by SCORE_DIM (see score_theme)."""

    # --- plasma: pitch error (matplotlib plasma sampled every 0.25, REVERSED so
    # index 0 is the "good" end). [0]/[1]/[2] are also the score's mistake trio.
    PLASMA_ANCHORS = [
        (72, 195, 0), 
        (255, 243, 10),
        (220, 33, 0), 
    ]

    # --- viridis: volume ---
    VIRIDIS_ANCHORS = [
        (68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)
    ]

    # --- magma: timbre heatmap (matplotlib magma, dark -> hot) ---
    MAGMA_ANCHORS = [
        (0, 0, 4), (51, 15, 106), (120, 28, 129),
        (189, 55, 121), (249, 105, 92), (252, 253, 191),
    ]

    # The score draws on white paper with the cursor color landing ON TOP of
    # annotated noteheads, so every score color is knocked back 10%. The
    # GuitarHero plot has no such constraint and uses them full strength.
    SCORE_DIM = 0.9
    # ...except the hovered note's pitch dots, knocked back 20% to pick the
    # frames belonging to that note out of the surrounding track.
    HOVER_DIM = 0.8

    # --- role colors (legends, score noteheads, alignment overlays) ---
    MISTAKE_RGB = {
        'substitution': (236, 145, 0),  # off-pitch
        'insertion': (220, 33, 0),     # extra note
        'timing': (220, 33, 0),        # early/late/long/short
        'deletion': (220, 33, 0),      # missing note
    }
    CURRENT_RGB = (0, 110, 154)  # the score's playback cursor (never dimmed)
    TRANSITION_RGB = (140, 140, 140)  # unvoiced / transition frames

    # --- note-detail panel (ui/note): timeline / contour / axes ---
    PLOT_BG_RGB = (20, 20, 25)       # the pyqtgraph panels' background
    NOTE_TIMELINE_RGB = (0, 255, 0)  # mirrors the GuitarHero timeline (own knob)
    NOTE_AXIS_RGB = (230, 230, 235)  # near-white axes for legibility
    NOTE_VOLUME_RGB = (94, 201, 98)  # the volume curve — viridis's "loud" green
    # the flat single-color pitch contour, editable per graph: volume/vibrato
    # draw it UNDER their curve on the dark bg; timbre ABOVE the magma heatmap
    NOTE_CONTOUR_RGB = {
        'volume': (95, 100, 110),
        'vibrato': (95, 100, 110),
        'timbre': (170, 170, 175),
    }
    # the GuitarHero pointer box, keyed by what it points AT. Clean takes the
    # score cursor's own blue: pointing at a note nothing flagged means "you are
    # here", the same thing it means on the score, so it must not read as an error.
    HIGHLIGHT_RGB = {
        'clean': CURRENT_RGB,
        'mistake': (255, 80, 80),
        'overridden': (80, 255, 80),
    }

    # --- pitch-distance ramps (bucketed brushes along PLASMA_ANCHORS) ---
    DISTANCE_STEP = 0.05        # semitones per bucket
    ALIGN_MAX_MULT = 4.0        # live/review ramp ends at this multiple of the tolerance

    # --- volume ramp ---
    VOLUME_BUCKETS = 48
    VOL_LIVE_FLOOR_DB = -42.0  # live: this far below full-scale (0 dBFS) => quietest

    # --- MIDI background stripes: one color per letter name, sharps darkened ---
    LETTER_RGB = {
        'A': (230,  60,  60),  # red
        'B': (255, 150,  40),  # orange
        'C': (245, 220,  70),  # yellow
        'D': ( 70, 200,  90),  # green
        'E': ( 70, 140, 240),  # blue
        'F': (100,  90, 210),  # indigo
        'G': (170,  90, 210),  # purple
    }
    PC_TO_LETTER = {
        0: 'C', 1: 'C', 2: 'D',
        3: 'D', 4: 'E', 5: 'F',
        6: 'F', 7: 'G', 8: 'G',
        9: 'A', 10: 'A', 11: 'B',
    }

    # --- primitives ---
    @staticmethod
    def dim(rgb: tuple, factor: float = None) -> tuple[int, int, int]:
        """Knock a color back toward black (default: the score's SCORE_DIM)."""
        factor = Colors.SCORE_DIM if factor is None else factor
        return tuple(round(c * factor) for c in rgb)

    @staticmethod
    def css_rgb(rgb: tuple) -> str:
        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

    @staticmethod
    def ramp(anchors: list, frac: float, dim: bool = False) -> tuple[int, int, int]:
        """Linear interpolation along an anchor list for a 0..1 fraction."""
        frac = max(0.0, min(1.0, frac)) * (len(anchors) - 1)
        i = min(int(frac), len(anchors) - 2)
        t = frac - i
        c0, c1 = anchors[i], anchors[i + 1]
        rgb = tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))
        return Colors.dim(rgb) if dim else rgb

    @staticmethod
    def mistake_rgb(role: str, dim: bool = False) -> tuple[int, int, int]:
        """A mistake role's color ('substitution' | 'insertion' | 'timing' |
        'deletion'); `dim` for the score's knocked-back variant."""
        rgb = Colors.MISTAKE_RGB[role]
        return Colors.dim(rgb) if dim else rgb

    @staticmethod
    def plot_colors() -> dict:
        """The general GuitarHero palette (brushes/pens keyed by role)."""
        return {
            'midi': pg.mkBrush(255, 255, 255, 200),      # score notes (white)
            'midi_dim': pg.mkBrush(120, 120, 120, 150),  # score notes OUTSIDE the clip
            'user_note': pg.mkBrush(55, 155, 144, 150),
            'user_pitch': pg.mkBrush(41, 177, 240, 255),
            'rest': pg.mkBrush(*Colors.TRANSITION_RGB),  # transition / unvoiced grey
            'timeline': pg.mkPen(0, 255, 0, 255),        # green
            # ins/del bars are pitch errors: same roles the score paints
            'insertion': pg.mkBrush(*Colors.mistake_rgb('insertion'), 200),
            'deletion': pg.mkBrush(*Colors.mistake_rgb('deletion'), 200),
            'match_line': pg.mkPen(255, 255, 255, 140, width=1.5,
                                   style=Qt.PenStyle.DashLine),
            'measure': pg.mkPen(230, 230, 240, 20, width=4.0),  # measure starts: thicker
            'beat': pg.mkPen(230, 230, 240, 20, width=1.0),     # beats: thinner
            'clip_dim': pg.mkBrush(0, 0, 0, 110),        # dim bands outside the clip
        }

    @staticmethod
    def hover_brush(brush):
        """A pooled brush knocked back by HOVER_DIM, keeping its alpha. Every
        dot ramp feeds through here, so hovering dims a frame whatever coloring
        (pitch / align / volume / transition) it happens to be drawn with."""
        c = brush.color()
        rgb = Colors.dim((c.red(), c.green(), c.blue()), Colors.HOVER_DIM)
        return pg.mkBrush(QColor(*rgb, c.alpha()))

    @staticmethod
    def highlight_style(state: str = "clean") -> tuple:
        """(brush, pen) for the pointer box, by what it points at: 'clean' (blue
        — nothing flagged it), 'mistake' (red) or 'overridden' (green).
        Deliberately outside the plasma ramp — this marks SELECTION, not an
        error."""
        rgb = Colors.HIGHLIGHT_RGB.get(state, Colors.HIGHLIGHT_RGB['clean'])
        return pg.mkBrush(*rgb, 130), pg.mkPen(*rgb, 255, width=2)

    # --- note-detail panel pens ---
    @staticmethod
    def note_timeline_pen():
        return pg.mkPen(*Colors.NOTE_TIMELINE_RGB, 255)

    @staticmethod
    def note_contour_pen(role: str):
        """The flat pitch-contour pen for one note-panel graph
        ('volume' | 'vibrato' | 'timbre')."""
        return pg.mkPen(*Colors.NOTE_CONTOUR_RGB[role], 255, width=4)

    @staticmethod
    def note_axis_pen():
        return pg.mkPen(*Colors.NOTE_AXIS_RGB)

    @staticmethod
    def note_axis_hex() -> str:
        """The axis color as '#rrggbb' (pyqtgraph label styles want css)."""
        return "#{:02x}{:02x}{:02x}".format(*Colors.NOTE_AXIS_RGB)

    @staticmethod
    def midi_rgba(m: int, alpha: int = 50) -> tuple[int, int, int, int]:
        """Background stripe color for a MIDI number (sharps darkened)."""
        r, g, b = Colors.LETTER_RGB[Colors.PC_TO_LETTER[m % 12]]
        if (m % 12) in {1, 3, 6, 8, 10}:  # sharps
            r, g, b = int(r * 0.70), int(g * 0.70), int(b * 0.70)
        return (r, g, b, alpha)

    # --- the web score's theme ---
    @staticmethod
    def score_theme() -> dict[str, str]:
        """Every color the Verovio page draws with, keyed by role. JS mirrors
        each onto the `--score-<role>` CSS custom property viewer.css reads, and
        fills the insertion markers it injects from the same map. Dimmed by
        SCORE_DIM — except the cursor, which has to read ON TOP of them."""
        theme = {role: Colors.css_rgb(Colors.mistake_rgb(role, dim=True))
                 for role in Colors.MISTAKE_RGB}
        theme['current'] = Colors.css_rgb(Colors.CURRENT_RGB)
        return theme

    # --- volume ---
    @staticmethod
    def viridis_anchors(dim: bool = False) -> list:
        """The volume ramp's anchors, score-dimmed on request."""
        if not dim:
            return list(Colors.VIRIDIS_ANCHORS)
        return [Colors.dim(rgb) for rgb in Colors.VIRIDIS_ANCHORS]

    @staticmethod
    def viridis(frac: float, dim: bool = False) -> tuple[int, int, int]:
        """RGB along the truncated viridis ramp for a 0..1 volume fraction."""
        return Colors.ramp(Colors.VIRIDIS_ANCHORS, frac, dim=dim)

    @staticmethod
    def viridis_brushes(buckets: int = None) -> list:
        """Pooled brushes along the viridis ramp (the volume dots and the
        vibrato panel's secondary-metric dots share this pool)."""
        n = buckets or Colors.VOLUME_BUCKETS
        return [pg.mkBrush(QColor(*Colors.viridis(i / (n - 1)))) for i in range(n)]

    @staticmethod
    def volume_brushes() -> list:
        """Pooled brushes along the viridis ramp, one per volume bucket."""
        return Colors.viridis_brushes()

    # --- timbre ---
    @staticmethod
    def magma_anchors() -> list:
        return list(Colors.MAGMA_ANCHORS)

    @staticmethod
    def magma(frac: float) -> tuple[int, int, int]:
        return Colors.ramp(Colors.MAGMA_ANCHORS, frac)

    @staticmethod
    def magma_lut(n: int = 256) -> np.ndarray:
        n = max(2, int(n))
        return np.asarray(
            [Colors.magma(i / (n - 1)) for i in range(n)], dtype=np.ubyte)

    @staticmethod
    def volume_frac(volume: float, vmin_db: float = None, vmax_db: float = None) -> float:
        """A frame's volume as a 0..1 fraction (quietest..loudest). With a take's
        own [vmin_db, vmax_db] dBFS range, interpolate across it; without one
        (live recording) use the fixed [VOL_LIVE_FLOOR_DB, 0] window so a dot's
        color locks the moment it's drawn."""
        if not volume or volume <= 0:
            return 0.0
        db = 20.0 * math.log10(volume)  # dBFS, ref = 1.0 (full scale)
        if vmin_db is None or vmax_db is None or vmax_db <= vmin_db:
            frac = (db - Colors.VOL_LIVE_FLOOR_DB) / (0.0 - Colors.VOL_LIVE_FLOOR_DB)
        else:
            frac = (db - vmin_db) / (vmax_db - vmin_db)
        return max(0.0, min(1.0, frac))

    # --- pitch-distance ---
    @staticmethod
    def plasma_anchors(dim: bool = False) -> list:
        """The pitch-error ramp's anchors, score-dimmed on request."""
        if not dim:
            return list(Colors.PLASMA_ANCHORS)
        return [Colors.dim(rgb) for rgb in Colors.PLASMA_ANCHORS]

    @staticmethod
    def plasma(frac: float, dim: bool = False) -> tuple[int, int, int]:
        """RGB along the pitch-error ramp for a 0..1 fraction (0 = on-pitch
        yellow, 1 = way-off indigo)."""
        return Colors.ramp(Colors.PLASMA_ANCHORS, frac, dim=dim)

    @staticmethod
    def pitch_brushes(correct_thresh: float, max_dist: float) -> list:
        """Bucketed plasma ramp: solid green within `correct_thresh` semitones,
        then walking the palette to red at `max_dist`."""
        step = Colors.DISTANCE_STEP
        brushes = []
        for i in range(int(max_dist / step) + 1):
            d = i * step
            frac = 0.0 if d <= correct_thresh else \
                (d - correct_thresh) / (max_dist - correct_thresh)
            brushes.append(pg.mkBrush(QColor(*Colors.plasma(frac))))
        return brushes

    @staticmethod
    def align_pitch_brushes(tolerance: float) -> tuple[list, float]:
        """The adaptive live/review ramp: green within the recording's
        pitch-mistake `tolerance`, ramping to red at ALIGN_MAX_MULT *
        tolerance (insertions clamp to the last bucket => solid red). Returns
        (brushes, max_dist) so lookups can clamp consistently."""
        correct_thresh = max(float(tolerance), 0.0)
        # keep at least one bucket of ramp even if tolerance is ~0
        max_dist = max(Colors.ALIGN_MAX_MULT * correct_thresh,
                       correct_thresh + Colors.DISTANCE_STEP)
        return Colors.pitch_brushes(correct_thresh, max_dist), max_dist

    @staticmethod
    def ramp_brush(brushes: list, value: float, max_value: float):
        """Clamped bucket lookup into a pitch-distance ramp (inf clamps to the
        last bucket)."""
        d = min(abs(float(value)), max_value)
        return brushes[min(int(d / Colors.DISTANCE_STEP), len(brushes) - 1)]
