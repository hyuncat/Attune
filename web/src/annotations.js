// Client-side port of ui/score/ScoreAnnotations.py + ui/score/NotePopupSV.py:
// builds the score-note-indexed mistake-annotation payload the ported
// viewer.js expects (window.setMistakeAnnotations - see
// ui/score/verovio/viewer.js's "MISTAKE ANNOTATION STATE" section). Built
// here instead of on the server since every input (pitchMistakes/
// timingMistakes, scoreNotesActive, currentPairs, pitch frames) already
// lives in sessionState client-side - see the project note on why mistake
// classification itself is client-side.
import { noteFromArray, noteName } from "./mistakes.js";
import { meanLevelDb, volumeFractionDb, volumeColorScale, viridis, cssRgb } from "./colors.js";

function pitchName(midi, cents = false) {
  if (midi == null || midi < 0) return "-";
  const name = noteName(midi);
  if (!cents) return name;
  const centsValue = (midi - Math.round(midi)) * 100;
  if (Math.abs(centsValue) < 0.5) return name;
  return `${name} ${centsValue >= 0 ? "+" : ""}${centsValue.toFixed(0)}¢`;
}

function notePitchNames(note) {
  if (!note || !note.midiNum || !note.midiNum.length) return "-";
  const names = note.midiNum
    .filter((m) => m != null && m >= 0)
    .map((m) => pitchName(m));
  return names.length ? names.join("/") : "-";
}

function primaryMidi(note) {
  if (!note || !note.midiNum || !note.midiNum.length) return null;
  const midi = note.midiNum[0];
  return midi == null || midi < 0 ? null : midi;
}

// The chord member closest to what was played (mirrors NotePopupSV._nearest_target_midi).
function nearestTargetMidi(userNote, scoreNote) {
  const played = primaryMidi(userNote);
  const targets = (scoreNote?.midiNum ?? []).filter((m) => m != null && m >= 0);
  if (!targets.length) return null;
  if (played == null) return targets[0];
  return targets.reduce((best, m) => (Math.abs(played - m) < Math.abs(played - best) ? m : best), targets[0]);
}

function seconds(v) {
  return `${v.toFixed(2)}s`;
}

function duration(note) {
  return note ? Math.max(0, note.endTime - note.startTime) : 0;
}

// Mirrors NotePopupSV.mistake_payload(): the popup content for one mistake,
// rendered by viewer.js when its note/marker is clicked.
export function mistakePayload(mistake) {
  const { type } = mistake;
  if (type === "insertion") {
    return {
      category: "pitch",
      type,
      title: "Extra note!",
      rows: [
        { label: "Target pitch", value: "-" },
        { label: "Played pitch", value: notePitchNames(mistake.userNote) },
      ],
    };
  }
  if (type === "deletion") {
    return {
      category: "pitch",
      type,
      title: "Missed note!",
      rows: [
        { label: "Target pitch", value: notePitchNames(mistake.scoreNote) },
        { label: "Played pitch", value: "-" },
      ],
    };
  }
  if (type === "substitution") {
    const target = nearestTargetMidi(mistake.userNote, mistake.scoreNote);
    const played = primaryMidi(mistake.userNote);
    const title = played != null && target != null && played > target ? "Too sharp!" : "Too flat!";
    return {
      category: "pitch",
      type,
      title,
      rows: [
        { label: "Target pitch", value: pitchName(target) },
        { label: "Played pitch", value: pitchName(played, true) },
      ],
    };
  }
  if (type === "early" || type === "late") {
    return {
      category: "timing",
      type,
      title: type === "early" ? "Early!" : "Late!",
      rows: [
        { label: "Target onset", value: seconds(mistake.scoreNote.startTime) },
        { label: "Played onset", value: seconds(mistake.userNote.startTime) },
      ],
    };
  }
  if (type === "long" || type === "short") {
    return {
      category: "timing",
      type,
      title: type === "long" ? "Too long!" : "Too short!",
      rows: [
        { label: "Target duration", value: seconds(duration(mistake.scoreNote)) },
        { label: "Played duration", value: seconds(duration(mistake.userNote)) },
      ],
    };
  }
  return { category: "pitch", type, title: `${type}!`, rows: [] };
}

// The score notes immediately before/after an insertion's slot, walked by
// PAIR INDEX (not time) - mirrors Alignment.flanking_score_notes, which reads
// the same DP pair list. Pairs are already in score/chronological order, so
// walking outward from the insertion's own pairIndex for the nearest non-null
// scoreIdx on each side lands on the same neighbors Python finds.
function flankingScoreIndices(pairs, pairIndex) {
  let left = null;
  for (let i = pairIndex - 1; i >= 0; i--) {
    if (pairs[i][1] != null) { left = pairs[i][1]; break; }
  }
  let right = null;
  for (let i = pairIndex + 1; i < pairs.length; i++) {
    if (pairs[i][1] != null) { right = pairs[i][1]; break; }
  }
  if (left == null && right == null) return null;
  return [left, right];
}

// A score note's volume payload: the volume of the USER note it aligned to
// (Colors.viridis(frac, dim=True) - score-dimmed, unlike GuitarHero's own
// dots, so the playback cursor still reads on top). Mirrors
// ScoreAnnotations._score_note_volume/_user_note_volume.
function userNoteVolume(userNote, pitchFrames, volumeScale) {
  if (!userNote) return null;
  const level = meanLevelDb(pitchFrames, userNote.startTime, userNote.endTime);
  const frac = volumeFractionDb(level, volumeScale);
  if (frac == null) return null;
  return { frac, color: cssRgb(viridis(frac, true)) };
}

// Builds the {notes, insertions, noteMeta, volumes} payload for
// window.setMistakeAnnotations. `mistakes` is pitchMistakes.concat(timingMistakes)
// - viewer.js itself filters which category is VISIBLE per the active color
// mode (see visibleMistakesForNoteIndex), so both always go in together,
// mirroring ScoreAnnotations.build combining pitch_mistakes + timing_mistakes
// unconditionally. `userNotesActive`/`pitchFrames` are only used for the
// `volumes` map (skippable - volume mode just shows nothing colored without them).
export function buildAnnotations({
  scoreNotesActive,
  userNotesActive,
  mistakes,
  overridden,
  overrideKey,
  currentPairs,
  pitchFrames,
}) {
  const empty = { notes: {}, insertions: [], noteMeta: {}, volumes: {} };
  if (!scoreNotesActive || !currentPairs) return empty;

  const notes = {};
  const noteMeta = {};
  const volumes = {};
  const insertionSlots = new Map();

  const userNotesParsed = (userNotesActive ?? []).map(noteFromArray);
  const volumeScale = volumeColorScale(pitchFrames);
  // scoreIdx -> userIdx, the note each score note actually aligned to (a
  // clean match or a substitution both count - only deletions have none).
  const userIdxForScoreIdx = new Map();
  for (const [userIdx, scoreIdx] of currentPairs) {
    if (userIdx != null && scoreIdx != null) userIdxForScoreIdx.set(scoreIdx, userIdx);
  }

  scoreNotesActive.forEach((raw, idx) => {
    noteMeta[String(idx)] = {
      seekTime: raw[1], // [id, start, end, midi[], ...] - see JsonHandler._note_to_payload
      midis: (raw[3] ?? []).filter((m) => m != null && m >= 0),
    };
    if (!pitchFrames) return;
    const userIdx = userIdxForScoreIdx.get(idx);
    const volume = userIdx != null ? userNoteVolume(userNotesParsed[userIdx], pitchFrames, volumeScale) : null;
    if (volume) volumes[String(idx)] = volume;
  });

  for (const mistake of mistakes) {
    if (overridden.has(overrideKey(mistake))) continue;
    const payload = mistakePayload(mistake);

    if (mistake.type === "insertion") {
      const flanks = flankingScoreIndices(currentPairs, mistake.pairIndex);
      if (!flanks) continue;
      const [leftIndex, rightIndex] = flanks;
      if (leftIndex == null && rightIndex == null) continue;
      const key = `${leftIndex}:${rightIndex}`;
      let entry = insertionSlots.get(key);
      if (!entry) {
        entry = { leftIndex, rightIndex, seekTime: mistake.userNote.startTime, mistakes: [], midis: [] };
        insertionSlots.set(key, entry);
      }
      entry.mistakes.push(payload);
      entry.seekTime = Math.min(entry.seekTime, mistake.userNote.startTime);
      const midi = primaryMidi(mistake.userNote);
      if (midi != null) entry.midis.push(midi);
      if (pitchFrames) {
        const volume = userNoteVolume(mistake.userNote, pitchFrames, volumeScale);
        if (volume) entry.volume = volume;
      }
      continue;
    }

    if (mistake.scoreIdx == null) continue;
    const key = String(mistake.scoreIdx);
    (notes[key] ??= []).push(payload);
  }

  const insertions = [];
  for (const { midis, ...rest } of insertionSlots.values()) {
    if (midis.length) rest.midi = midis.reduce((a, b) => a + b, 0) / midis.length;
    insertions.push(rest);
  }

  return { notes, insertions, noteMeta, volumes };
}
