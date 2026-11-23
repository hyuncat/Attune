## Attune

Upload a MIDI / musicXML file, record yourself playing an instrument, and analyze your mistakes!

<img width="800" alt="attune_pitch" src="https://github.com/user-attachments/assets/1e7786bb-76ce-408d-b667-8a0239c7a87b" />

### Core features
Two modes
- Performance mode - Play as you would perform the piece, and get graded on your pitch and relative timing
- Practice mode - The score doesn't keep moving until you get the pitch right

Clip score
- Highlight the measures in the score you want to focus on, then "Clip" to analyze just that

Record, save, and upload audio
- Record in the app, save as .wav, and upload it back in next time
- Can compare multiple recording takes against a single score

Robust playback
- Select subset of instruments to play back at any time
- Can even record while instruments are playing (ie, piano accompaniment)

### Algorithms
Built upon:
- PYIN algorithm for pitch detection
- PELT (pruned exact linear time) L2 segmentation for note detection
- String edit algorithm for mistake correction

## Installation
To download the code on your machine, pull the repository and run the following commands:

Install the required packages:
```shell
pip install -r requirements.txt
```

And run the app with
```shell
python app.py
```

## Contact

If you find an issue or want to contribute, contact:
- hyuncat3@gmail.com
- gbciaobella@gmail.com


Research notebooks, datasets, competitor implementations, and results are indexed in [experiments/README.md](experiments/README.md). Paper-facing exports live in [experiments/results/](experiments/results/); experiments and results do not ship with the app.
