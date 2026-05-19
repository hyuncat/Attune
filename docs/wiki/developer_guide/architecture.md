# Code structure

### Top-Level Architecture
`app.py` represents the general application shell. It initializes the frontend subcomponents with related logic and acts as the full app coordinator for the multi-panel UI. Additionally, it handles global state, loading, and mode switching. `practice.py` handles similar logic specific to practice mode in Attune.

### Logical Architecture
`NoteData.py` is the primary handler for storing and retrieving notes as it includes the NoteData and Note class. The Note class stores per-note attributes, while NoteData provides indexed access, updates, and utilities for working with note collections over time and indexing. It is used for both recording data and score data. 

`Alignment.py` compares user-recorded notes against score notes from MIDI data and captures match quality through mistake categories. It serves as the core comparison layer used by analysis and feedback features.

## Data Flow Overview

### Recording Data Flow
Recordings enter the system either by importing an audio file supported by the SoundFile Python library or by live microphone capture via `AudioRecorder.py`. Raw waveform data is stored in `AudioData.py`.

The recording pipeline then derives pitches into `PitchData.py`, segments notes into `NoteData.py`, and compares user notes against score notes using `Alignment.py`. 

### Score Data Flow
Score files (MusicXML, MIDI) are loaded into `ScoreData.py`, which parses and initializes metadata. Similar to the recording data pipeline, `NoteData.py` is used to store information related note collections initialized in `ScoreData.py`. Playback is handled in `MidiPlayer.py` and `MidiSynth.py`.

## /algorithms
PitchDetector.py
- online + offline pitch detection

NoteDetector.py

StringEditor.py