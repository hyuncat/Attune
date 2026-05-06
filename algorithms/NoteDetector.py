import numpy as np
from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch
from PyQt6.QtCore import pyqtSignal, QObject
import threading

from app_logic.user.ds.Recording import Recording
from app_logic.user.ds.PitchData import PitchData
from algorithms.Config import Config

class NoteDetector(QObject):
    note_detected = pyqtSignal(float)
    
    def __init__(self, recording: Recording=None, config: Config=None, parent: QObject|None=None):
        """initialize the note detection algorithm parameters"""
        super().__init__(parent)

        # algorithm params
        self.recording = recording
        self.config = recording.config if recording else config
        self.w = self.config.w2
        self.hop = self.config.h2
        self.PITCH_THRESH = self.config.pitch_thresh
        self.SLOPE_THRESH = self.config.slope_thresh
        
        self.UNVOICED_PROP = self.config.unv_ratio # if more than 50% of pitches are unvoiced
        self.UNV_THRESH = self.config.unv_thresh # unvoiced pitches have unv_prob > sens
        
        # threading variables
        self.nda_thread: threading.Thread = None
        self.stop_event = threading.Event()

    def stop(self):
        if self.nda_thread and self.nda_thread.is_alive():
            self.stop_event.set()
            self.nda_thread.join() # pause the main thread until recording thread recognizes the stop event

    def get_slope(self, pitches: list[Pitch]):
        """get slope of all voiced pitches in the window"""
        # select only voiced x and y values
        mask  = np.array([p.unvoiced_prob < self.UNV_THRESH if p else False for p in pitches]) # boolean mask

        all_x = np.linspace(start=0, stop=len(pitches), num=len(pitches))
        x_voiced = all_x[mask]
        y_voiced = np.array([p.candidates[0][0] for p, m in zip(pitches, mask) if m])

        if x_voiced.size == 0:
            return 0.0, 0.0

        # get slope + intercept of only voiced pitches
        A = np.vstack([x_voiced, np.ones_like(x_voiced)]).T
        slope, intercept = np.linalg.lstsq(A, y_voiced, rcond=None)[0]

        return slope, intercept
    
    def is_unvoiced(self, unvoiced_probs: list[float]) -> bool:
        """returns whether the window is voiced or not
        based on whether the proportion of unvoiced pitches
        exceeds the UNVOICED_PROP threshold
        """
        arr = [p > self.UNV_THRESH for p in unvoiced_probs]
        if sum(arr) > self.UNVOICED_PROP*len(arr):
            return True
        return False
    
    def get_median_pitches(self, pitches: list[Pitch]):
        """return median pitches of whatever exists in the candidate
        slots for indices 0:2"""
        N = 3
        medians = [-1] * N

        # Select only voiced frames
        voiced = [p for p in pitches if p and p.unvoiced_prob < self.UNV_THRESH]

        if not voiced:
            return medians

        # Collect candidates in each column
        cols = [[] for _ in range(N)]

        for p in voiced:
            # p.candidates should be a list of (midi, prob)
            for i in range(min(N, len(p.candidates))):
                pitch_val = p.candidates[i][0]
                if pitch_val != -1:
                    cols[i].append(pitch_val)

        # Compute medians
        for i in range(N):
            if cols[i]:
                medians[i] = float(np.median(cols[i]))

        return medians
        
    
    def handle_window(self, pitches: list[Pitch]):
        """
        returns key results about the window used for note processing
            (1) is_flat, (2) is_unv, (3) median_pitch, (4) start_time
        """
        unvoiced_probs = [p.unvoiced_prob if p else 1.0 for p in pitches]
        slope, _ = self.get_slope(pitches) 

        # key results
        is_flat = slope < self.SLOPE_THRESH
        is_unv = self.is_unvoiced(unvoiced_probs)
        med_pitches = self.get_median_pitches(pitches)
        
        # print(f"t({pitches[0].time:.4f}): slope({slope:.2f}), is_flat({is_flat}), is_unv({is_unv}), med_pitch({med_pitches[0]:.2f})")
        
        return is_flat, is_unv, med_pitches
    
    def detect_notes(self, pitch_data: PitchData ) -> NoteData:
        """writes all notes completely offline"""
        nd = NoteData()
        prev_note = None
        prev_time = None
        prev_good_time = None
        note_index = 0

        # iterate through all pitches
        for i in range(0, len(pitch_data.data)-self.w-1, self.hop):
            x = pitch_data.read(i=i, j=i+self.w, clean=False)

            if x[0] is None:
                continue
            
            t = x[0].time
            is_flat, is_unv, med_pitches = self.handle_window(x)

            if prev_note is None:
                if is_unv:
                    prev_note = [-1, -1, -1]
                    prev_time = t
                elif is_flat:
                    prev_note = med_pitches
                    prev_time = t
            else:
                # if different enough...
                if abs(prev_note[0] - med_pitches[0]) > self.PITCH_THRESH:
                    if not is_flat and not is_unv:
                        # it's okay to not be 'flat' if unvoiced
                        continue
                    n = Note(
                        i=note_index,
                        start_time=prev_time, 
                        end_time=t,
                        midi_num=prev_note
                    )
                    nd.write_note(n)
                    # update iteration variables
                    prev_note = [-1, -1, -1] if is_unv else med_pitches
                    prev_time = t
                    note_index += 1
                prev_good_time = t

        # write the last note! :,)
        n = Note(
            i=i,
            start_time=prev_time, 
            end_time=prev_good_time,
            midi_num=prev_note
        )
        nd.write_note(n)
        return nd
    

    def run(self, start_time: float=None):
        self.stop()
        self.stop_event.clear()
        self.recording.p2n_queue.init_start_time(start_time)
        self.nda_thread = threading.Thread(
            target=self._run, daemon=True
        )
        self.nda_thread.start()

    def _run(self) -> None:
        """the note detection algorithm for real time processing.

        an onset-based approach, where a window is an *onset* if
            - it's flat enough and voiced
            - or if it's mostly unvoiced
        
        and if a window is an onset,
        we compare it to the last valid onset
            - if it's different, it's a new note
            - if it's the same, it's not a new note
        """
        prev_note = None
        prev_time = None
        i = 0
        while not self.stop_event.is_set():
            try:
                x, t = self.recording.p2n_queue.pop(self.w, self.hop)
                if x is None or t < 0: # if invalid data read, skip frame
                    continue

                is_flat, is_unv, med_pitch = self.handle_window(x)

                # print(f"this window: is_flat({is_flat}), is_unv({is_unv}), med_pitch({med_pitch}), t({t})")

                # --- finding the first note phase ---
                if prev_note is None:
                    prev_note = -1 if is_unv else med_pitch
                    prev_time = t
                    continue

                # --- the second note and beyond ---
                if abs(prev_note - med_pitch) < self.PITCH_THRESH:
                    continue

                # ignore if the current window is unvoiced or flat
                prev_time = t
                if not is_flat and not is_unv:
                    # but still advance prev_time so we stay contiguous
                    # prev_note = -1 if is_unv else med_pitch
                    continue

                # ---> if we reach here, we have a NEW NOTE!
                # print(f"NEW NOTE! pitch={prev_note}, start={prev_time}, end={t}")
                n = Note(
                    i=i,
                    start_time=prev_time, 
                    end_time=t,
                    midi_num=prev_note
                )
                self.recording.note_data.write_note(n)
                i += 1

                # update iteration variables
                prev_note = -1 if is_unv else med_pitch
                self.note_detected.emit(n.start_time)

            except Exception as e:
                print(f"[NoteDetector] frame skipped due to error: {e}")
                continue
