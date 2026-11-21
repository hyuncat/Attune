"""Export the bundled C major example using the desktop analysis pipeline.

Run from the repository root with its Python environment. Never modifies source
recordings or sidecars. The committed export keeps website builds Python-free.
"""
import base64
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app_logic.midi.ScoreData import ScoreData
from app_logic.JsonHandler import JsonHandler
from app_logic.user.ds.Recording import Recording

score_path = ROOT / 'resources/demo/scales/major/major.mxl'
audio_path = score_path.parent / 'flute_scale_bad.m4a'
s = ScoreData()
s.load(str(score_path))
h = JsonHandler()
# Naming overrides belong only to the bundled web example.
xml = ET.fromstring(s.to_musicxml_bytes())
for element in xml.iter():
    if element.tag in ('work-title', 'movement-title'):
        element.text = 'C Major Scale'
    elif element.tag == 'midi-program':
        element.text = '74'  # MusicXML uses one-based General MIDI programs.
    elif element.tag in ('part-name', 'instrument-name') and element.text == 'Violin':
        element.text = 'Flute'
    elif element.tag in ('part-abbreviation', 'instrument-abbreviation') and element.text in ('Vln', 'Vln.'):
        element.text = 'Fl.'
notes = dict(title='C Major Scale' , bpm=s.bpm, active_instrument=s.active_instrument,
    instruments=sorted(int(ch) for ch in s.instruments),
    note_data={str(ch): h._note_data_to_payload(nd) for ch, nd in s.note_datas.items()},
    musicxml_b64=base64.b64encode(ET.tostring(xml, encoding='utf-8', xml_declaration=True)).decode('ascii'),
    instrument_names={str(s.active_instrument): 'Flute'},
    metronome_channel=s.metronome_channel, measure_onsets_og=s.measure_onsets_og(),
    bpm_og=s.bpm_og, transpose_offset=s.transpose_offset)
r = Recording(score_data=s)
r.load_audio(str(audio_path), score_filepath=str(score_path))
if not r.pitch_data.data:
    r.detect_pitches()
r.reset_analysis()
r.detect_notes()
r.align_score_and_refine()
r.update_alignment_distances()
r.trim_end()
# Pair indices refer to the fitted score: publish that same timing for review.
notes["bpm"] = s.bpm
notes["note_data"] = {str(ch): h._note_data_to_payload(nd) for ch, nd in s.note_datas.items()}
payload = JsonHandler(r).to_cache_payload(score_filepath=str(score_path), recording_name='C major sample')
payload['alignment'] = {'pairs': payload['alignment']['pairs']}
payload['vibrato'] = JsonHandler._vibrato_to_payload(r.vibrato_data)
# Publish only fields consumed by the web UI, never local paths/cache metadata.
payload = {k: payload[k] for k in ('recording', 'config', 'pitch_data', 'note_data', 'alignment', 'vibrato')}
payload['recording'] = {'active_instrument': r.score_data.active_instrument}
# General MIDI Flute is zero-based program 73 in the playback note arrays.
for note in notes['note_data'][str(s.active_instrument)]:
    note[5] = 73
output = ROOT / 'web/sample'
output.mkdir(exist_ok=True)
(output / 'example.json').write_text(json.dumps({'score': notes, 'analysis': payload}, separators=(',', ':'), allow_nan=False))
(output / 'major.mxl').write_bytes(score_path.read_bytes())
(output / 'flute_scale_bad.m4a').write_bytes(audio_path.read_bytes())
print('Exported C major score, recording, and analysis to', output)
