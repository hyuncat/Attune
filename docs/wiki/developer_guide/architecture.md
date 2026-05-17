# Code structure

### Top-Level Architecture
`app.py` represents the general application shell. It initializes the frontend subcomponents with related logic and acts as the full app coordinator for the multi-panel UI. Additionally, it handles global state, loading, and mode switching. `practice.py` handles similar logic specific to practice mode in Attune.

### Logical Architecture
`NoteData.py` is the primary handler for storing and retrieving notes as it includes the NoteData and Note class. The Note class stores per-note attributes, while NoteData provides indexed access, updates, and utilities for working with note collections over time and indexing.

`Alignment.py` compares user-recorded notes against score notes from MIDI data and captures match quality through mistake categories. It serves as the core comparison layer used by analysis and feedback features.

## /algorithms
PitchDetector.py
- online + offline pitch detection

NoteDetector.py

StringEditor.py