import numpy as np

from app_logic.user.ds.AudioData import AudioData
from app_logic.user.ds.PitchData import PitchData, Pitch
from app_logic.midi.ScoreData import ScoreData
from app_logic.Alignment import Alignment
from app_logic.NoteData import NoteData
from app_logic.user.ds.Buffer import Buffer
from algorithms.Config import Config

class Recording:
    def __init__(self, score_data: ScoreData=None, config: Config=None):
        """the user data, associated with a singular recording of a score.
        each recording has its own audio data, pitch data, note data, and alignment
        as well as its own set of algorithms and parameters for processing that data"""
        self.score_data = score_data if score_data is not None else ScoreData()
        # each recording is associated with an instrument (channel #) in the score
        self.active_instrument = 0
        self.update_config(config)

        # algorithms!!
        from algorithms.PitchDetector import PitchDetector
        from algorithms.NoteDetector2 import NoteDetector2
        from algorithms.StringEditor import StringEditor
        self.pitch_detector = PitchDetector(recording=self)
        self.note_detector = NoteDetector2(recording=self)
        self.string_editor = StringEditor(recording=self)

        # essential data variables
        self.audio_data = AudioData(config=self.config)
        self.pitch_data = PitchData(config=self.config)
        self.note_data = NoteData()
        self.alignment: Alignment = Alignment(config=self.config) # filled in later

        # queue data structures for real time pitch + note detection + correction
        self.a2p_queue = Buffer(self.config.sr) #audio-to-pitches
        self.p2n_queue = Buffer(sr=self.config.sr/self.config.h1) #pitches-to-notes
        self.n2c_queue = None #notes-to-corrections

    def update_config(self, config: Config=None):
        """initialize the config, either with a provided one or a default one"""
        if config is None:
            config = {
                'sr': 44100,    # sample rate
                'w1': 1024 * 2,  # frame size
                'h1': 128,       # hop size
                'fmin': 196.0,
                'fmax': 3000.0,
                'tuning': 440.0,
                'unv_thresh': 0.9, # if unvoiced_prob > unv_thresh, consider the frame unvoiced

                # --- NOTE DETECTION PARAMETERS ---
                'w2': 21, # frame size (NOTE: should always be odd)
                'h2': 19, # hop size
                'pitch_thresh': 0.5,
                'slope_thresh': 0.75 / 21,
                'unv_ratio': 0.8, # proportion of unvoiced pitches in a window to consider the window unvoiced

                # --- STRING EDIT PARAMETERS ---
                'ins_cost': 1.5,
                'del_cost': 2,
                'sub_cost': 1,
                'tolerance': 1,
                # tiger-mom parameter
                'tiger_level': 1
            }
            self.config = Config(**config)
        else:
            self.config = config

    # def on_pitches_detected(self, pitches):
    #     self.pitch_data.data = pitches

    def load_audio(self, audio_filepath: str):
        """load in a pre-recorded audio file from a filepath
        also computes pitches on the entire file"""
        self.audio_data.load_data(audio_filepath)
        self.detect_pitches()
        # self.detect_notes()

    def detect_pitches(self):
        """run pitch detection on the current audio data"""
        self.pitch_data.data = self.pitch_detector.detect_pitches(self.audio_data.data)

    def detect_notes(self):
        """run note detection on the current pitch data"""
        self.note_data = self.note_detector.detect_notes(self.pitch_data)

    def detect_mistakes(self):
        user_notes, midi_notes = self.note_data, self.score_data.note_datas[self.active_instrument]
        notes, mistakes = self.string_editor.string_edit(user_string=user_notes, midi_string=midi_notes)
        self.alignment.load_alignment(notes, mistakes)

    def write_data(self, indata: np.ndarray, start_time: float):
        """write indata to the audio_data at the given start_time
        and append to our queue for pitch processing
        """
        self.audio_data.write_data(indata, start_time)
        self.a2p_queue.push(indata)

    def write_pitch_data(self, indata: list[Pitch], start_time: float):
        """write indata to the pitch_data at the given start_time
        and append to our queue for note processing
        """
        self.pitch_data.write(indata, start_time)
        self.p2n_queue.push(indata)

    def get_length(self):
        if len(self.note_data.times) > 0:
            return self.note_data.get_length()
        else:
            return self.audio_data.get_length()
    
    def resize(self, new_length: float):
        """Resize the score_data to a new length by changing the BPM of the score data, 
        updating the note timings and pitch distances as well."""
        factor = new_length / self.score_data.length
        new_bpm = round(self.score_data.bpm * factor)
        self.score_data.change_tempo(new_bpm, _factor=factor)
        self._update_pitch_distances()

    def change_tempo(self, new_bpm: float):
        """Change the tempo of the recording by changing the BPM of the score data, which will automatically update the note timings and pitch distances."""
        self.score_data.change_tempo(new_bpm)
        self._update_pitch_distances()

    def _update_pitch_distances(self):
        """Update the distance to target note for all pitches in the recording, based on the current score data."""
        for note in self.score_data.note_datas[self.active_instrument].data.values():
            if note is None:
                continue
            pitches = self.pitch_data.read(start_time=note.start_time, end_time=note.end_time, clean=True)
            for p in pitches:
                p.distance = note.midi_num[0] - p.candidates[0][0]