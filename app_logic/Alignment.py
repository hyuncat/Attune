from bisect import bisect_left, bisect_right
import numpy as np
from app_logic.NoteData import Note
from algorithms.Config import Config

class Mistake:
    def __init__(self, type: str, user_note: Note, midi_note: Note):
        self.type = type
        self.user_note = user_note
        self.midi_note = midi_note
        self.overridden = False
        self.pair_index = -1

    def is_overridden(self) -> bool:
        return bool(self.overridden)
    
    def set_override(self, override_value: bool):
        self.overridden=override_value

    #switch the override status on the given mistake
    def toggle_override(self):
        self.overridden = not self.overridden  

    def set_pair_index(self, pair_index_value: int):
        self.pair_index = pair_index_value
    
    def get_pair_index(self):
        return(self.pair_index)

class Alignment:
    def __init__(self, config: Config, notes: list[tuple[Note, Note]]=None, mistakes: list[Mistake]=None):
        self.config = config
        # these are crucial
        self.pairs: list[tuple[Note, Note]] = notes if notes else []
        self.mistakes: list[Mistake] = mistakes if mistakes else []
        self.overridden_pair_indices = set()

        # our time-indexable {t: (n,m)} dictionary
        # is there any way to make each just store a reference or smth?...
        if notes and mistakes:
            self.init_2(notes)
        else:
            self.pairs_1, self.pairs_2 = {}, {}
            self.times_1, self.times_2 = [], []
        self.THRESH = 1 # same as StringEditor.TOLERANCE

    def load_alignment(self, notes: list[tuple[Note, Note]], mistakes: list[Mistake]):
        """load in the alignment data, and initialize the pairs dictionaries for time indexing"""
        self.pairs = notes
        self.mistakes = mistakes
        self.overridden_pair_indices = set()
        self.init_2(notes)

    def init_2(self, pairs):
        """initialize the two pairs dictionaries for faster time indexing
        in self.get_alignment"""
        self.pairs_1 = {}
        self.pairs_2 = {}
        times_1 = []
        times_2 = []

        for n, m in pairs: # go through and dissect the pairs
            if n is None and m is None:
                continue
            if n is None:
                tmin = m.start_time
                tmax = m.end_time
            elif m is None:
                tmin = n.start_time
                tmax = n.end_time
            else:
                tmin = min(n.start_time, m.start_time)
                tmax = max(n.end_time, m.end_time)
            
            times_1.append(tmin)
            times_2.append(tmax)
            self.pairs_1[tmin] = (n, m)
            self.pairs_2[tmax] = (n, m)

        self.times_1 = sorted(times_1)
        self.times_2 = sorted(times_2)

        # print(f"Initialized with\npairs1\n---\n{self.pairs_1}\npairs2\n---\n{self.pairs_2}")

    def get_alignment(self, t_min: float, t_max: float) -> tuple[list[tuple[Note, Note]], 
                                                                 list[tuple[Note, Note]], 
                                                                 list[Note], list[Note]]:
        """returns all note pairs found within the time boundaries (for guitarhero)
        
        Args:
            t_min (float): minimum time (sec)
            t_max (float): maximum time (sec)
        
        Returns:
            tuple: (goods, subs, ins, dels)
        """
        i = bisect_left(self.times_1, t_min) # yes good
        j = bisect_right(self.times_2, t_max)

        pairs = self.pairs[i:j]
        ins, dels, subs, goods, = [], [], [], []
        for relative_idx, (n, m) in enumerate(pairs):
            absolute_idx = i + relative_idx
            if absolute_idx in self.overridden_pair_indices:
                goods.append((n, m))
                continue
            if n and not m: # insertion
                ins.append(n)
            elif not n and m: # deletion
                dels.append(m)
            elif abs(n.midi_num[0]-m.midi_num[0]) > self.THRESH:
                subs.append((n, m)) # substitution
            else: # good
                goods.append((n, m))

        # print(f"alignment @ {t_min}:\n---\n"
        #     f"goods: {goods},\n"
        #     f"subs: {subs},\n"
        #     f"ins: {ins},\n"
        #     f"dels: {dels}"
        # )

        # the end
        return goods, subs, ins, dels
    
    def reapply_overrides(self, overridden_mistake_indices: set[int]):
        self.reset_overrides()
        for index in overridden_mistake_indices:
            if 0 <= index < len(self.mistakes):
                m = self.mistakes[index]
                m.set_override(True)
                if m.pair_index is not None and 0 <= m.pair_index < len(self.pairs):
                    self.overridden_pair_indices.add(m.pair_index)
    
    #removes all overrides
    def reset_overrides(self):
        self.overridden_pair_indices = set()
        for m in self.mistakes:
            m.set_override(False)

    def toggle_overridden_pair_indices(self, index: int, toggle_to: bool):
        if toggle_to:
            self.overridden_pair_indices.add(index)
        else:
            self.overridden_pair_indices.discard(index)