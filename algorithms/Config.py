import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    # --- PITCH DETECTION PARAMETERS ---
    sr: int = 44100    # sample rate
    w1: int = 1024 * 6  # frame size
    h1: int = 128       # hop size
    fmin: float = 196.0
    fmax: float = 5000.0
    tuning: float = 440.0
    unv_thresh: float = 0.05 # if unvoiced_prob > unv_thresh, consider the frame unvoiced

    # --- NOTE DETECTION PARAMETERS ---
    w2: int = 30 # frame size
    h2: int = 27 # hop size
    pitch_thresh: float = 0.75
    slope_thresh: float = 1.5
    unv_ratio: float = 0.5 # proportion of unvoiced pitches in a window to consider the window unvoiced

    # --- SCORE-GUIDED NOTE DETECTION PARAMETERS ---
    prox_window: float = 1.0 # semitones; max distance from expected MIDI for a candidate to be preferred over top-prob

    # --- STRING EDIT PARAMETERS ---
    ins_cost: float = 1.5
    del_cost: float = 2
    sub_cost: float = 1
    tolerance: float = 1

    # tiger-mom parameter
    tiger_level: int = 1

    # --- loader ---
    def load_config(self, config: dict):
        """load in a config dictionary"""
        self.w1 = config.get("w1", self.w1)
        self.h1 = config.get("h1", self.h1)
        self.fmin = config.get("fmin", self.fmin)
        self.fmax = config.get("fmax", self.fmax)
        self.tuning = config.get("tuning", self.tuning)
        self.unv_thresh = config.get("unv_thresh", self.unv_thresh)

        self.w2 = config.get("w2", self.w2)
        self.h2 = config.get("h2", self.h2)
        self.pitch_thresh = config.get("pitch_thresh", self.pitch_thresh)
        self.slope_thresh = config.get("slope_thresh", self.slope_thresh)
        self.prox_window = config.get("prox_window", self.prox_window)

        self.ins_cost = config.get("ins_cost", self.ins_cost)
        self.del_cost = config.get("del_cost", self.del_cost)
        self.sub_cost = config.get("sub_cost", self.sub_cost)
        self.tolerance = config.get("tolerance", self.tolerance)

        self.tiger_level = config.get("tiger_level", self.tiger_level)

    # --- pitch conversion methods ---
    def freq_to_midi(self, freq: float) -> float:
        """
        Convert a frequency to a MIDI note number.
        """
        if freq <= 0:
            # print("bad freq")
            return(-1)
        return 69 + 12 * np.log2(freq / self.tuning)

    def midi_to_freq(self, midi_num: float) -> float:
        """
        Convert a MIDI note number to frequency.
        """
        return self.tuning * (2 ** ((midi_num - 69) / 12))


    def __repr__(self):
        return (f"Config\n---\n   sr={self.sr}, w1={self.w1}, h1={self.h1}, fmin={self.fmin}, fmax={self.fmax}, tuning={self.tuning}, unv_thresh={self.unv_thresh},\n"
                f"   w2={self.w2}, h2={self.h2}, pitch_thresh={self.pitch_thresh}, slope_thresh={self.slope_thresh:.3f}, unv_ratio={self.unv_ratio},\n"
                f"   ins_cost={self.ins_cost}, del_cost={self.del_cost}, sub_cost={self.sub_cost}, tolerance={self.tolerance}, tiger_level={self.tiger_level}")