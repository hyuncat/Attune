from PyQt6.QtCore import pyqtSignal, QTimer, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSlider
from math import ceil, floor

from ui.time.WallClock import WallClock

class Slider(QWidget):

    slider_changed = pyqtSignal(float) # emits current time in seconds
    slider_end = pyqtSignal(bool)

    def __init__(self, wall_clock: WallClock):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # slider <==> timer resolution variables
        self.wall_clock = wall_clock
        self.wall_clock.time_changed.connect(self.handle_timer_update)

        # init our slider!!
        self.DEFAULT_LENGTH_SEC = 30
        self.TICKS_PER_SEC = self.wall_clock.hz # 10 ticks per sec
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.midi_length_ticks = int(self.DEFAULT_LENGTH_SEC*self.TICKS_PER_SEC)
        self.midi_length_sec = self.DEFAULT_LENGTH_SEC
        self.slider.setRange(0, self.midi_length_ticks)

        # slider emissions
        # self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.valueChanged.connect(self.slider_moved)

        self._layout.addWidget(self.slider)
        
    # --- SIGNAL RELATED ---
    def slider_moved(self, value: int) -> None:
        """is called whenever the slider moves. emits the slider_changed signal
        corresponding to what time in the plot it now is at.
        
        Args:
            value (int): the current tick value of the slider
        """
        self.current_tick = value
        t = value / self.TICKS_PER_SEC # convert to seconds
        self.slider_changed.emit(t)
        # print(f"Slider moved to {t} sec")
        
        if value >= self.midi_length_ticks: # emit signal when reached end of slider
            self.slider_end.emit(True)

    # --- RANGE HANDLING ---
    def update_range(self, score_data=None, recording=None):
        """Update the slider range based on max(MIDI.length, audio.length)"""
        m0, m1, u1 = 0, 0, 0
        if score_data:
            note_data = score_data.note_datas.get(score_data.active_instrument, None)
            if note_data:
                m0, m1 = note_data.get_bounds()
        if recording and recording.audio_data:
            u1 = recording.audio_data.get_length()

        x0 = m0
        x1 = max(m1, u1+m0)
        self._update_range(x0, x1)

    def _update_range(self, start_time: float, end_time: float):
        """update the slider range to have [sec] amount of space"""
        start_ticks = int(floor(start_time * self.TICKS_PER_SEC))
        end_ticks = int(ceil(end_time * self.TICKS_PER_SEC))
        self.midi_length_ticks = end_ticks
        self.midi_length_sec = end_time
        self.slider.setRange(start_ticks, end_ticks)
    
    # --- TIMER RELATED ---
    def handle_timer_update(self, t: float) -> None:
        """called whenever timer is updated (every 100ms)"""
        tick = int(t * self.wall_clock.hz)
        self.current_tick = tick

        # ensure current tick never exceeds maximum
        if self.current_tick > self.slider.maximum():
            self.current_tick = self.slider.maximum()
            self.slider_end.emit(True)
            # self.wall_clock.stop() # this may not always be good

        self.slider.setValue(self.current_tick)

    # utils
    def get_time(self):
        """get current time of slider in seconds"""
        return self.slider.value() / self.wall_clock.hz
    
    def get_total_time(self):
        """get total time of slider in seconds"""
        return self.slider.maximum() / self.TICKS_PER_SEC
