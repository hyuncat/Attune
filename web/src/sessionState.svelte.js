// Shared reactive session state, replacing what used to be prop-drilled
// through App -> UploadForm/ResultsView. Needed once the desktop app's
// actual panel split (toolbar upload, center score+overlay, right mistake
// table, bottom transport) pulled those three apart into separate
// components that all need the same underlying state.
//
// Single module-level instance: this app has exactly one score/recording
// in flight at a time (matching the desktop app's single active Recording),
// so a singleton is simpler than threading context through every component.
import { getNoteData } from "./noteDataCache.js";
import { realign, debounce } from "./realign.js";
import { classifyPitchMistakes, classifyTimingMistakes, noteName, noteNameToMidi, midiToHz, mistakeCategory } from "./mistakes.js";
import { buildAnnotations } from "./annotations.js";
import { playback } from "./playback.svelte.js";
import { makeNotifier } from "./notifier.js";

const API_BASE_URL = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "http://localhost:8000" : "https://at-c19b0e98085b4847b318cac88b97b75d.ecs.us-east-1.on.aws");

function createSessionState() {
  let scoreFile = $state(null);
  let audioFile = $state(null);
  let noteData = $state(null);
  let noteDataError = $state("");
  // User's Instrument-selector override (SettingsPanel); null means "use the
  // score's own default" (score_data.get_default_instrument()).
  let selectedInstrument = $state(null);

  let analysisResult = $state(null);
  let analyzeStatus = $state("idle"); // "idle" | "loading" | "error"
  let analyzeError = $state("");

  let pitchTolerance = $state(0.5);
  let timingTolerance = $state(0.25);
  let mode = $state("pitch"); // "pitch" | "timing" - mirrors MistakeWidget's mode dropdown

  // ScoreViewer's own "Colors:" dropdown (perform.py's score_color_mode) -
  // independent of `mode` above (MistakeWidget's table can show pitch rows
  // while the score itself is colored by timing, and vice versa).
  let scoreColorMode = $state("pitch"); // "pitch" | "timing" | "volume"

  // Range/Tuning (SettingsWidget) - these bound the server-side PYIN pitch
  // detector's search range (Config.fmin/fmax) and reference pitch
  // (Config.tuning), not just a mistake-classification display concern.
  // Narrowing fmin/fmax to the piece's actual range is a real accuracy win
  // (fewer octave-error candidates for PYIN to confuse itself with), which
  // is why these default to the SCORE's own range instead of the wide
  // general-purpose "G3"-"E7" span the desktop app's blank-state default
  // uses - re-derived per instrument, since a different line can have a
  // very different range.
  //
  // No padding beyond the score's exact min/max note, matching
  // ui/info/SettingsWidget.py's populate_range_from_score() precisely (a
  // straight min/max over the channel's midi_num values, no headroom).
  // Padding was tried and reverted: it doesn't match any tested desktop
  // behavior, and any value close to 12 semitones re-admits octave-alias
  // candidates right back into PYIN's search window - exactly the
  // confusion range-narrowing exists to prevent. Desktop's own answer to
  // "a real mistake falls outside the range" isn't a buffer, it's that
  // Range stays a user-adjustable control with an Apply-and-redetect loop
  // (see app.py's on_range_applied) - the same escape hatch this web port
  // already has via setRange()/runAnalyze().
  const DEFAULT_LOW = "G3";
  const DEFAULT_HIGH = "E7";
  let lowNoteName = $state(DEFAULT_LOW);
  let highNoteName = $state(DEFAULT_HIGH);
  let tuning = $state(440);
  let rangeError = $state("");

  let realignedPairs = $state(null);
  let realignError = $state("");
  let realigning = $state(false);

  // Client-side-only mistake dismissal (see ResultsView's original note on
  // this) - keyed by mode+pairIndex+type since a pair can appear in both an
  // onset AND duration timing mistake.
  let overridden = $state(new Set());

  // Mirrors MistakeWidget.selected -> perform.on_mistake_selected ->
  // GuitarHero.highlight_mistake: clicking a mistake row highlights its
  // note(s) in the pitch overlay and moves the shared time cursor there
  // (which also drags the score viewer along, via the same currentTime
  // effect that drives it during playback). Keyed the same way as
  // `overridden` for a stable identity across re-renders.
  let selectedMistakeKey = $state(null);

  // Transient StatusBar message (mirrors app.py's StatusBar.status_label).
  let statusMessage = $state("");

  // Fired imperatively whenever fresh note data lands (pickScore, below) -
  // App.svelte subscribes once (onMount) and pushes the new score into the
  // viewer directly, mirroring perform.py's refresh_score_viewer being
  // called explicitly rather than derived from a reactive watcher.
  const { on: onNoteDataLoaded, notify: notifyNoteDataLoaded } = makeNotifier("noteDataLoaded");

  const debouncedRealign = debounce(async (tolerance) => {
    if (!analysisResult || !noteData) return;
    const scoreNotes = scoreNotesForActiveInstrument();
    if (!scoreNotes) return;
    realigning = true;
    try {
      const result = await realign(analysisResult.note_data, scoreNotes, tolerance, API_BASE_URL);
      realignedPairs = result.pairs;
      realignError = "";
    } catch (err) {
      realignError = err instanceof Error ? err.message : String(err);
    } finally {
      realigning = false;
    }
  }, 250);

  function activeInstrument() {
    if (selectedInstrument != null) return selectedInstrument;
    if (analysisResult) return analysisResult.recording.active_instrument;
    if (noteData) return noteData.active_instrument;
    return null;
  }

  // Scans the given channel's raw note arrays (note_data[channel][i][3] is
  // the chord-aware midiNum array - see JsonHandler._note_to_payload) for
  // its exact pitch span - a straight min/max, no padding, matching
  // populate_range_from_score() exactly. Not a mistake-analysis path, so it
  // works directly off the raw payload rather than noteFromArray().
  function computeDefaultRange(channel) {
    if (!noteData) return null;
    const rawNotes = noteData.note_data[String(channel)];
    if (!rawNotes || rawNotes.length === 0) return null;
    let min = Infinity;
    let max = -Infinity;
    for (const n of rawNotes) {
      for (const m of n[3]) {
        if (m != null && m >= 0) {
          if (m < min) min = m;
          if (m > max) max = m;
        }
      }
    }
    if (min === Infinity) return null;
    return { low: noteName(min), high: noteName(max) };
  }

  function applyDefaultRangeForActiveInstrument() {
    const range = computeDefaultRange(activeInstrument());
    lowNoteName = range?.low ?? DEFAULT_LOW;
    highNoteName = range?.high ?? DEFAULT_HIGH;
  }

  function scoreNotesForActiveInstrument() {
    if (!noteData) return null;
    const ch = activeInstrument();
    return ch != null ? noteData.note_data[String(ch)] : null;
  }

  let currentPairs = $derived(realignedPairs ?? analysisResult?.alignment?.pairs ?? null);

  let scoreNotesActive = $derived(scoreNotesForActiveInstrument());

  let pitchMistakes = $derived.by(() => {
    if (!currentPairs || !analysisResult || !scoreNotesActive) return [];
    return classifyPitchMistakes(
      currentPairs,
      analysisResult.note_data,
      scoreNotesActive,
      pitchTolerance
    );
  });

  let timingMistakes = $derived.by(() => {
    if (!currentPairs || !analysisResult || !scoreNotesActive) return [];
    return classifyTimingMistakes(
      currentPairs,
      analysisResult.note_data,
      scoreNotesActive,
      timingTolerance
    );
  });

  // Score-viewer annotation payload (window.setMistakeAnnotations - see
  // annotations.js's header). Colocated here as a $derived.by rather than
  // computed inline in App.svelte's push effect: it's fed by ~9 different
  // mutations scattered across this module (runAnalyze, toggleOverride,
  // setSelectedInstrument, setPitchTolerance/setTimingTolerance,
  // debouncedRealign's completion, pickScore's own noteData/
  // selectedInstrument writes, _resetAnalysis) - hand-wiring an explicit
  // notify() at every one of those sites is exactly the kind of missed-site
  // bug this whole redesign exists to eliminate (a first manual pass at that
  // list actually missed two). A $derived.by tracks its real dependencies
  // automatically instead, so App.svelte only has to watch this one
  // collapsed value (see App.svelte's pushAnnotations/its $effect).
  let annotationsPayload = $derived.by(() =>
    buildAnnotations({
      scoreNotesActive,
      userNotesActive: analysisResult?.note_data,
      mistakes: pitchMistakes.concat(timingMistakes),
      overridden,
      overrideKey,
      currentPairs,
      pitchFrames: analysisResult?.pitch_data?.pitches,
    })
  );

  let visibleMistakes = $derived(
    (mode === "timing" ? timingMistakes : pitchMistakes).slice().sort((a, b) => {
      const ta = a.scoreNote?.startTime ?? a.userNote?.startTime ?? 0;
      const tb = b.scoreNote?.startTime ?? b.userNote?.startTime ?? 0;
      return ta - tb;
    })
  );

  // Keyed by the MISTAKE's own category, not the currently active mode tab -
  // ResultsView only ever toggles mistakes matching the active tab so this is
  // no different there, but the ScoreViewer annotation payload (annotations.js)
  // combines pitch AND timing mistakes at once regardless of which tab is
  // showing, and needs each one's override state checked correctly either way.
  function overrideKey(m) {
    return `${mistakeCategory(m.type)}:${m.pairIndex}:${m.type}`;
  }
  function toggleOverride(m) {
    const key = overrideKey(m);
    const next = new Set(overridden);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    overridden = next;
  }

  let selectedMistake = $derived(
    selectedMistakeKey ? visibleMistakes.find((m) => overrideKey(m) === selectedMistakeKey) ?? null : null
  );

  // GuitarHero.highlight_mistake mean-of-midpoints: mean of each involved
  // note's own (start+end)/2, not the midpoint of a bounding box spanning
  // all of them - matters when a substitution's user/score notes land at
  // different times.
  function mistakeMedianTime(m) {
    const notes =
      m.type === "substitution"
        ? [m.userNote, m.scoreNote]
        : m.type === "insertion"
          ? [m.userNote]
          : m.type === "deletion"
            ? [m.scoreNote]
            : [m.userNote, m.scoreNote]; // timing mistakes: both
    const times = notes.filter(Boolean).map((n) => 0.5 * (n.startTime + n.endTime));
    if (times.length === 0) return null;
    return times.reduce((a, b) => a + b, 0) / times.length;
  }

  function selectMistake(m) {
    const key = overrideKey(m);
    if (selectedMistakeKey === key) {
      selectedMistakeKey = null;
      return;
    }
    selectedMistakeKey = key;
    const t = mistakeMedianTime(m);
    if (t != null) playback.seek(t);
  }

  function clearMistakeSelection() {
    selectedMistakeKey = null;
  }

  // Mirrors perform.py's _clear_analysis: a new score or recording
  // invalidates any existing analysis (it was computed against the OLD
  // score/audio pair), so the mistake table and GuitarHero overlay must not
  // keep showing it - otherwise pitchMistakes/timingMistakes end up pairing
  // stale userNotes against the NEW scoreNotesActive, which is exactly what
  // produced the "mistake widget looks wrong after uploading a new score"
  // symptom (indices/notes that don't correspond to each other at all).
  function _resetAnalysis() {
    analysisResult = null;
    analyzeStatus = "idle";
    analyzeError = "";
    realignedPairs = null;
    realignError = "";
    overridden = new Set();
    selectedMistakeKey = null;
  }

  let selectionVersion = 0;
  let sampleLoading = $state(false);
  let sampleError = $state("");

  async function loadSample() {
    const version = ++selectionVersion;
    sampleLoading = true;
    sampleError = "";
    try {
      const responses = await Promise.all(["example.json", "major.mxl", "flute_scale_bad.m4a"].map(
        name => fetch(`/sample/${name}`).then(response => {
          if (!response.ok) throw new Error("Could not load the bundled example.");
          return response;
        })
      ));
      const [example, score, audio] = await Promise.all([responses[0].json(), responses[1].blob(), responses[2].blob()]);
      // A manual upload made while the example was downloading takes priority.
      if (version !== selectionVersion) return;
      playback.pause();
      _resetAnalysis();
      selectedInstrument = null;
      scoreFile = new File([score], "major.mxl");
      audioFile = new File([audio], "flute_scale_bad.m4a", { type: "audio/mp4" });
      noteData = example.score;
      noteDataError = "";
      analysisResult = example.analysis;
      pitchTolerance = 0.5;
      timingTolerance = 0.25;
      tuning = 440;
      rangeError = "";
      applyDefaultRangeForActiveInstrument();
      playback.loadNoteData(noteData);
      playback.loadUserAudio(audioFile);
      playback.seek(0);
      notifyNoteDataLoaded(noteData);
      statusMessage = "C Major Scale · flute_scale_bad · precomputed analysis";
    } catch (error) {
      if (version === selectionVersion) sampleError = error instanceof Error ? error.message : String(error);
    } finally {
      if (version === selectionVersion) sampleLoading = false;
    }
  }

  async function pickScore(file) {
    selectionVersion++;
    sampleLoading = false;
    sampleError = "";
    scoreFile = file;
    noteDataError = "";
    selectedInstrument = null;
    _resetAnalysis();
    if (!file) {
      noteData = null;
      return;
    }
    statusMessage = `Loading ${file.name}...`;
    try {
      noteData = await getNoteData(file, API_BASE_URL);
      statusMessage = `Loaded ${file.name}`;
      playback.loadNoteData(noteData);
      applyDefaultRangeForActiveInstrument();
      notifyNoteDataLoaded(noteData);
    } catch (err) {
      noteDataError = err instanceof Error ? err.message : String(err);
      statusMessage = "Failed to load score.";
    }
  }

  function pickAudio(file) {
    selectionVersion++;
    sampleLoading = false;
    sampleError = "";
    audioFile = file;
    statusMessage = file ? `Recording set: ${file.name}` : "";
    _resetAnalysis();
    playback.loadUserAudio(file);
  }

  async function runAnalyze() {
    if (!scoreFile || !audioFile) return;
    analyzeStatus = "loading";
    analyzeError = "";
    statusMessage = "Analyzing...";

    const formData = new FormData();
    formData.append("score", scoreFile);
    formData.append("audio", audioFile);
    if (pitchTolerance != null) {
      formData.append("pitch_tolerance", String(pitchTolerance));
    }
    if (selectedInstrument != null) {
      formData.append("active_instrument", String(selectedInstrument));
    }
    const lowMidi = noteNameToMidi(lowNoteName);
    const highMidi = noteNameToMidi(highNoteName);
    if (lowMidi != null) formData.append("fmin", String(midiToHz(lowMidi, tuning)));
    if (highMidi != null) formData.append("fmax", String(midiToHz(highMidi, tuning)));
    formData.append("tuning", String(tuning));

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || `Request failed (${response.status})`);
      }
      analysisResult = await response.json();
      analyzeStatus = "idle";
      overridden = new Set();
      realignedPairs = null; // fresh alignment supersedes any prior /realign result
      statusMessage = "Analysis complete.";
    } catch (err) {
      analyzeStatus = "error";
      analyzeError = err instanceof Error ? err.message : String(err);
      statusMessage = "Analysis failed.";
    }
  }

  function setPitchTolerance(value) {
    pitchTolerance = value;
    debouncedRealign(pitchTolerance); // no-op pre-analysis; guarded above
  }

  function setTimingTolerance(value) {
    // No /realign: timing-mistake reclassification is a pure client-side
    // threshold check over the existing pairs.
    timingTolerance = value;
  }

  function setSelectedInstrument(channel) {
    selectedInstrument = channel;
    applyDefaultRangeForActiveInstrument();
  }

  function setRange(low, high) {
    const lowMidi = noteNameToMidi(low);
    const highMidi = noteNameToMidi(high);
    if (lowMidi == null || highMidi == null) {
      rangeError = `Couldn't parse "${lowMidi == null ? low : high}" as a note name (e.g. G3, F#4, Bb2).`;
      return;
    }
    if (lowMidi >= highMidi) {
      rangeError = "Low note must be below the high note.";
      return;
    }
    rangeError = "";
    lowNoteName = low;
    highNoteName = high;
  }

  function setTuning(hz) {
    if (!Number.isFinite(hz) || hz <= 0) return;
    tuning = hz;
  }

  return {
    get scoreFile() { return scoreFile; },
    get audioFile() { return audioFile; },
    get noteData() { return noteData; },
    get noteDataError() { return noteDataError; },
    get analysisResult() { return analysisResult; },
    get analyzeStatus() { return analyzeStatus; },
    get analyzeError() { return analyzeError; },
    get pitchTolerance() { return pitchTolerance; },
    get timingTolerance() { return timingTolerance; },
    get mode() { return mode; },
    // Repopulating the tree on a mode switch clears the selection in
    // MistakeWidget.py too (its comment: "so GuitarHero drops its
    // highlight via cleared").
    set mode(v) {
      mode = v;
      selectedMistakeKey = null;
    },
    get scoreColorMode() { return scoreColorMode; },
    set scoreColorMode(v) { scoreColorMode = v; },
    get realigning() { return realigning; },
    get realignError() { return realignError; },
    get overridden() { return overridden; },
    get statusMessage() { return statusMessage; },
    get currentPairs() { return currentPairs; },
    get scoreNotesActive() { return scoreNotesActive; },
    get pitchMistakes() { return pitchMistakes; },
    get timingMistakes() { return timingMistakes; },
    get annotationsPayload() { return annotationsPayload; },
    get visibleMistakes() { return visibleMistakes; },
    get activeInstrument() { return activeInstrument(); },
    get selectedInstrument() { return selectedInstrument; },
    get lowNoteName() { return lowNoteName; },
    get highNoteName() { return highNoteName; },
    get tuning() { return tuning; },
    get rangeError() { return rangeError; },
    get selectedMistake() { return selectedMistake; },
    get selectedMistakeKey() { return selectedMistakeKey; },

    get sampleLoading() { return sampleLoading; },
    get sampleError() { return sampleError; },
    loadSample,
    pickScore,
    pickAudio,
    runAnalyze,
    setPitchTolerance,
    setTimingTolerance,
    setSelectedInstrument,
    setRange,
    setTuning,
    overrideKey,
    toggleOverride,
    selectMistake,
    clearMistakeSelection,
    onNoteDataLoaded,
  };
}

export const session = createSessionState();
