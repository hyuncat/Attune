from pathlib import Path
import logging
import mido

from app_logic.NoteData import NoteData, Note

logger = logging.getLogger(__name__)

# --- metronome / woodblock click constants ---
# shared between init_metronome (embedded score clicks) and the count-in
WOODBLOCK_PROGRAM = 115
CLICK_DURATION = 0.1  # sec
DOWNBEAT_NOTE = 80
BEAT_NOTE = 40

class MidiData:
    def __init__(self, filepath: str | Path):
        """
        messages_og = the original times
        messages = the times after tempo changes
        (hopefully avoids accumulating errors)
        """
        self.filepath = Path(filepath)

        # --- THE ESSENTIAL STUFF ---
        # store data by {elapsed_time: Message}
        self.messages_og: dict[float, list[mido.Message]] = {}
        self.programs_og: dict[float, list[mido.Message]] = {} # stores "turn on instrument" messages
        self.metas_og: dict[float, list[mido.MetaMessage]] = {}

        self.messages: dict[float, list[mido.Message]] = {}
        self.programs: dict[float, list[mido.Message]] = {} # stores "turn on instrument" messages
        self.metas: dict[float, list[mido.MetaMessage]] = {}

        # metadata
        # self.length = 0 # length of the piece in seconds
        # self.bpm: int = 120 # default bullshit tempo (set to musescore default)
        self.instruments: dict[int, int] = {} # {channel: program_number}
        self.normalized_channel_groups: list[dict] = []

        # metronome click track lives on its own channel; the click note_on/off
        # messages are (re)generated from the beat grid by set_metronome so they
        # always track score-timing changes (tempo / resize) — see set_metronome.
        self.metronome_channel: int | None = None

        self.parse_messages(mido.MidiFile(self.filepath))

    def parse_messages(self, midi_data: mido.MidiFile):
        """Load a MIDI file using mido, parse messages by elapsed time.
        Iterate through all messages, categorize by 
            - meta
            - program change
            - all messages
        then store as dict[elapsed_time, list[Message]].
        Also track all instruments (and what channels they play on). 
        And find the BPM if possible.
        """
        logger.debug("Handling MIDI file...")

        metas, messages, programs = {}, {}, {}
        instruments = {}

        elapsed_time = 0
        for msg in midi_data:
            elapsed_time += msg.time # update time elapsed

            if msg.is_meta:
                if msg.type == "set_tempo":
                    self.bpm = round(mido.tempo2bpm(msg.tempo))
                metas.setdefault(elapsed_time, []).append(msg)
                continue

            # append MESSAGE with elapsed time into messages
            messages.setdefault(elapsed_time, []).append(msg)

            # track program changes
            if msg.type == "program_change":
                programs.setdefault(elapsed_time, []).append(msg)
                instruments[msg.channel] = msg.program

        # ---> error handling: if no channels used, add a fake one (violin lmao)
        if not instruments:
            instruments[0] = 40
            fake_msg = mido.Message('program_change', program=40, channel=0, time=0)
            programs[0] = [fake_msg]

        # update results
        self.messages_og = messages
        self.programs_og = programs
        self.metas_og = metas
        self.messages, self.programs, self.metas = messages, programs, metas

        self.instruments = instruments
        self.length_og = elapsed_time
        self.length = elapsed_time # total length of the piece in seconds

    def normalize_duplicate_program_channels(self, min_channels: int = 4) -> bool:
        """Collapse many channels with the same program into one logical channel.

        Some scraped/converted solo MIDI files shard one instrument across many
        MIDI channels even though every channel has the same program. That makes
        the app show a long list of identical instruments and, worse, each
        channel only contains a fragment of the part. Collapse only when the
        duplicate group is numerous enough that it is almost certainly a channel
        artifact rather than a small ensemble split.
        """
        by_program: dict[int, list[int]] = {}
        for channel, program in self.instruments.items():
            by_program.setdefault(program, []).append(channel)

        channel_map: dict[int, int] = {}
        normalized_groups = []
        for program, channels in by_program.items():
            channels = sorted(channels)
            if len(channels) < min_channels:
                continue
            target = channels[0]
            for channel in channels:
                channel_map[channel] = target
            normalized_groups.append({
                "program": program,
                "target": target,
                "channels": channels,
            })

        if not channel_map:
            return False

        def remap_message(msg):
            channel = getattr(msg, "channel", None)
            if channel not in channel_map:
                return msg
            target = channel_map[channel]
            if msg.type == "program_change" and channel != target:
                return None
            return msg.copy(channel=target)

        def remap_messages(source: dict) -> dict:
            remapped = {}
            for t, msgs in source.items():
                kept = []
                for msg in msgs:
                    new_msg = remap_message(msg)
                    if new_msg is not None:
                        kept.append(new_msg)
                if kept:
                    remapped[t] = kept
            return remapped

        self.messages_og = remap_messages(self.messages_og)
        self.messages = remap_messages(self.messages)
        self.programs_og = remap_messages(self.programs_og)
        self.programs = remap_messages(self.programs)

        collapsed = {}
        for channel, program in self.instruments.items():
            target = channel_map.get(channel, channel)
            if target == channel:
                collapsed[target] = program
        self.instruments = dict(sorted(collapsed.items()))
        self.normalized_channel_groups = normalized_groups
        return True
    
    def make_notedatas(self) -> dict[int, NoteData]:
        """Convert the stored messages into Note objects in NoteData.
        Should be called after loading a MIDI or MusicXML file.

        Returns:
            NoteData for the MIDI file
        """
        note_lists = {channel: dict() for channel in self.instruments.keys()}
        note_onsets = {}
        # notes = {}

        i = 0
        for elapsed_time, msgs in self.messages.items():
            for msg in msgs:
                # skip non-note related stuff
                if msg.type not in {'note_on', 'note_off'}:
                    continue

                key = (msg.channel, msg.note) # create unique key per note

                # velocity>0 because sometimes midi files are weird lol
                if msg.type=='note_on' and msg.velocity>0:
                    note_onsets.setdefault(key, []).append((elapsed_time, msg.velocity))

                elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
                    if key not in note_onsets:
                        continue
                    # end the note we recorded in note_onsets and write to NoteData
                    start_time, velocity = note_onsets[key].pop(0)
                    existing = note_lists[msg.channel].get(start_time)
                    if existing is not None:
                        # CHORD: another note already starts at this exact onset.
                        # NoteData is keyed by onset (one Note per start_time), so
                        # MERGE this pitch into that Note's midi_num list instead of
                        # overwriting it — otherwise every chord / double-stop
                        # collapses to a single note (only the last one survived).
                        # The chord's bar spans until its last member releases.
                        if msg.note not in existing.midi_num:
                            existing.midi_num.append(msg.note)
                        if elapsed_time > existing.end_time:
                            existing.end_time = elapsed_time
                            existing.base_end_time = elapsed_time
                    else:
                        note = Note(
                            i=i,
                            start_time=start_time,
                            end_time=elapsed_time,
                            midi_num=[msg.note],
                            velocity=velocity,
                            instrument=self.instruments.get(msg.channel, None)
                        )
                        note_lists[msg.channel][start_time] = note
                        i += 1  # one id per onset -> id stays aligned with note index

                    # cleanup our iteration variables
                    if not note_onsets[key]:
                        del note_onsets[key]

        note_datas = {}
        for channel, notes in note_lists.items():
            note_datas[channel] = NoteData()
            note_datas[channel].load_data(notes=notes)        
        return note_datas

    def init_metronome(self, beat_times: list[tuple[float, bool]]):
        """Set up the metronome click track on a new channel after all
        instruments, then lay down its clicks at the given beat times.

        Only registers the channel + woodblock program once; the actual click
        notes are (re)generated by set_metronome so they can be refreshed
        whenever the score timing changes.
        """
        # MIDI only has channels 0..15. Grab the lowest channel the score isn't
        # already using (don't assume channels are contiguous, so len() won't do).
        # A fully orchestrated file can use all 16 channels (e.g. Rite of Spring);
        # then there's no room for a click track, so skip it and let the rest of
        # the score load — set_metronome / consumers all no-op on a None channel.
        channel = next((c for c in range(16) if c not in self.instruments), None)
        if channel is None:
            print("Warning: all 16 MIDI channels in use; skipping metronome.")
            self.metronome_channel = None
            return
        self.metronome_channel = channel
        self.instruments[channel] = WOODBLOCK_PROGRAM

        # the program_change is tempo-independent (time 0): keep it in the
        # original program map so it survives change_tempo's rebuild. At load
        # `programs` aliases `programs_og`, so only add to both if they've
        # already diverged (avoids a duplicate program_change).
        pc = mido.Message(
            'program_change',
            program=WOODBLOCK_PROGRAM,
            channel=channel,
            time=0
        )
        self.programs_og.setdefault(0.0, []).append(pc)
        if self.programs is not self.programs_og:
            self.programs.setdefault(0.0, []).append(pc)

        self.set_metronome(beat_times)

    def set_metronome(self, beat_times: list[tuple[float, bool]]):
        """(Re)generate the metronome click notes from the beat grid.

        Strips any existing click notes from `messages` and re-adds a
        note_on/note_off pair at each beat time, so the audible metronome always
        matches the current beat grid (and thus the GuitarHero gridlines) after
        a tempo change or resize. Idempotent: safe to call on every beat update.

        Rebuilding each time bucket into a fresh list also de-aliases `messages`
        from `messages_og` (change_tempo reuses the original lists), so clicks
        never leak back into the pristine original score.
        """
        channel = self.metronome_channel
        if channel is None:
            return

        is_click = lambda m: (
            getattr(m, "channel", None) == channel
            and m.type in ("note_on", "note_off")
        )

        # drop existing clicks; the comprehension yields new lists (de-aliasing)
        messages: dict[float, list[mido.Message]] = {}
        for t, msgs in self.messages.items():
            kept = [m for m in msgs if not is_click(m)]
            if kept:
                messages[t] = kept

        for beat_time, is_downbeat in beat_times:
            t_on = round(float(beat_time), 9)
            t_off = round(float(beat_time) + CLICK_DURATION, 9)
            note_num = DOWNBEAT_NOTE if is_downbeat else BEAT_NOTE

            on = mido.Message(
                'note_on', channel=channel, note=note_num, velocity=100, time=t_on
            )
            off = mido.Message(
                'note_off', channel=channel, note=note_num, velocity=0, time=t_off
            )
            messages.setdefault(t_on, []).append(on)
            messages.setdefault(t_off, []).append(off)

        self.messages = dict(sorted(messages.items())) # keep time-sorted


    def transpose(self, semitones: int):
        """Shift every instrument note's MIDI number by `semitones` half steps,
        in place, so playback is transposed. Skips the metronome channel (its
        woodblock clicks must keep their pitch).

        Operates on `messages_og` — the pitch baseline. change_tempo rebuilds
        `messages` from these same Message objects (and set_metronome keeps them),
        so a single shift here carries through tempo changes / metronome rebuilds
        without double-applying. Clamped to the valid MIDI range (0..127)."""
        if not semitones:
            return
        for msgs in self.messages_og.values():
            for m in msgs:
                if (m.type in ('note_on', 'note_off')
                        and m.channel != self.metronome_channel):
                    m.note = int(max(0, min(127, m.note + semitones)))

    def change_tempo(self, factor: float):
        """Changes all beats by some factor of the original tempo.
        Updates messages, program changes, meta messages.
        """
         # update all message timings
        messages = {}
        for t, msgs in self.messages_og.items():
            new_time = t * factor
            messages[new_time] = msgs
        self.messages = messages

        metas = {}
        for t, msgs in self.metas_og.items():
            new_time = t * factor
            metas[new_time] = msgs
        self.metas = metas

        programs = {}
        for t, msgs in self.programs_og.items():
            new_time = t * factor
            programs[new_time] = msgs
        self.programs = programs
        
