# Injected CocoChorales audit

The saved `benchmarks/results/mistake_competitors_coco` run used overlapping
insertions. Its F1 scores do **not** describe the corrected monophonic protocol.
Preserve it as historical evidence; new full runs use
`benchmarks/results/mistake_competitors_coco_monophonic`.

## Findings from the historical run

The audit replays the saved smoothed pitch frames through current note detection
and both Attune configurations. All 39 unique cases (26 injected, 13 clean)
reproduce their saved TP/FP/FN for each configuration at the 100 ms gate.

- All 940 serialized MIDI notes are integer semitones, with zero pitch bends.
  Fractional injected pitches were not the cause of these results.
- All 17 **net insertions** overlap another performed note. Checker 3 detects
  **0/17**. The 48 canonical extra events also include 31 substitutions, so
  “14/48 extras found” must not be read as insertion recall.
- Mean per-note smoothed-pitch coverage within 50 cents is approximately 93%
  on clean notes, 89% on nonoverlapping injected-performance notes, and 58% on
  overlapping notes. These are diagnostic coverage values, not standard pitch F1.
- Checker 3 has 103 injected-case false positives. Heuristic classification:

| Predicted kind | Diagnostic evidence | Count |
|---|---|---:|
| Missed | Target pitch weak/absent in smoothed track | 48 |
| Missed | Matching initial note exists; inspect alignment/refinement | 2 |
| Extra | Same-pitch fragment or displaced onset | 28 |
| Extra | Pitch unsupported by nearby performed notes | 15 |
| Extra | Real performed note incorrectly flagged | 5 |
| Extra | Near a true error; timing gate or duplicate | 5 |

These categories localize inspection; they do not establish a causal breakdown.
Pitch coverage uses nominal performed MIDI time with up to 50 ms excluded at
note edges, without compensating for instrument attack latency. Comparisons with
clean pitch/note notebooks alone cannot measure extraction on injected audio.

## Corrected protocol: `monophonic_edits_v3`

Insertions occur sequentially after their host and shift following notes later
as necessary, consuming available silence first. Deletions advance subsequent
notes by the removed duration while preserving existing gaps. Duration overruns
also delay following notes. The final pass preserves event order and note
durations, prevents overlaps, and saves each event's nominal onset and signed
timing shift. Changed pitches are integer MIDI semitones; offsets cannot silently
clip back to the original pitch at MIDI range boundaries.

Truth is recounted from the serialized performed MIDI. The time map removes
only generated shifts for this assignment, preventing intact suffix notes from
becoming false errors. Source-note identities do not dictate matches. A removed
note plus an equivalent inserted note can cancel; a wrong-pitch replacement can
become a substitution. Evaluated algorithms do not receive the time map.

MIDI parsing can change IDs, so the map is relinked chronologically and checked
against serialized onsets and pitches. Serialized overlaps are rejected.
The new case fingerprint prevents reuse of historical overlapping audio.

To reproduce the diagnostic tables after a comparison:

```sh
at-venv/bin/python -m benchmarks.modules.mistake.provenance.InjectedCaseAudit benchmarks/results/mistake_competitors_coco_monophonic
```

The notebook displays the audit when present. A full corrected comparison must
be rerun before making claims about monophonic F1 or ranking competitors.

## Validation

20 focused tests pass, including insertion/deletion cancellation, signed time
mapping after parser ID changes, range-boundary pitch changes, and adjacent
insertions with identical nominal onsets. A further 100 randomized mixed-edit
cases are monophonic. All five competitors completed a clean/injected Coco horn
smoke run under the corrected timeline protocol; its serialized MIDI passed
validation and its clean control had empty mistake truth. This small smoke run
is not a replacement for the full comparison.

### MIDI serialization regression

The 25%-injected tuba case with seed 1 exposed sub-tick short notes. At 220 PPQ
and 120 BPM, a MIDI tick is about 2.27 ms; a roughly 1 ms note could round its
onset and offset to the same tick. The emitted note-off preceded note-on,
leaving a hanging note until a later same-pitch release. Protocol v3 quantizes
the final timeline and enforces at least one tick per note without overlaps.
The MIDI is validated before synthesis, and versioned case keys rebuild old
artifacts. All 52 default cases pass raw-MIDI and app-parser round trips.
