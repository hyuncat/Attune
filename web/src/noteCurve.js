// Client-side port of ui/note/NoteCurveWidget.py's "note under the cursor"
// window state machine, plus the shared pitch-contour/volume-curve
// extraction (app_logic/user/ds/PitchData.py's pitch_curve/volume_curve)
// that NotePanel's graphs (Volume now, Vibrato/Timbre later) draw against.
import { noteFromArray } from "./mistakes.js";
import { getPitchFramesInRange } from "./PitchFrameHandler.js";

// The transport is quantized (desktop: milliseconds), so seeking to an exact
// onset can round down just before the note - a small display-only
// lookahead prefers the newly selected note without moving playback itself.
export const NOTE_ONSET_LOOKAHEAD_SEC = 0.01;

// The user note sounding at time t (strict start<=t<=end), or null - mirrors
// NoteData.note_containing(clean=True): gaps between notes and rests
// (midiNum[0] === -1) both read as "no note", matching the panel's blank rule.
export function noteContaining(userNotesActive, t) {
  if (!userNotesActive?.length) return null;
  const tt = t + NOTE_ONSET_LOOKAHEAD_SEC;
  let candidate = null;
  for (const raw of userNotesActive) {
    if (raw[1] <= tt) candidate = raw;
    else break;
  }
  if (!candidate) return null;
  const note = noteFromArray(candidate);
  if (!(note.startTime <= tt && tt <= note.endTime)) return null;
  if (!note.midiNum?.length || note.midiNum[0] === -1) return null;
  return note;
}

// (time, midi|null) over frames in [t0, t1] - null where unvoiced, so a
// chart drawn with gaps breaks cleanly. Mirrors PitchData.pitch_curve.
const UNVOICED_THRESHOLD = 0.9;
export function pitchContour(pitchFrames, t0, t1) {
  const points = [];
  for (const frame of getPitchFramesInRange(pitchFrames, t0, t1)) {
    const [time, , , unvoicedProb, , , , value] = frame;
    const voiced = value !== -1 && unvoicedProb < UNVOICED_THRESHOLD;
    points.push({ time, midi: voiced ? value : null });
  }
  return points;
}

// (time, dBFS) over every frame in [t0, t1] - voicing is irrelevant to
// loudness, so (unlike pitchContour) this keeps unvoiced/noisy frames too;
// silence floors at floorDb rather than gapping. Mirrors PitchData.volume_curve.
export function volumeCurveDb(pitchFrames, t0, t1, floorDb) {
  const points = [];
  for (const frame of getPitchFramesInRange(pitchFrames, t0, t1)) {
    const [time, , volume] = frame;
    const db = volume > 0 ? Math.max(floorDb, 20 * Math.log10(volume)) : floorDb;
    points.push({ time, db });
  }
  return points;
}

// Take-wide reference, matching VolumeWidget: arithmetic mean of voiced
// frame levels in dB. Silence/noise do not pull the playing reference down.
export function volumeLevelStats(pitchFrames) {
  let count = 0, sum = 0, min = Infinity, max = -Infinity;
  for (const frame of pitchFrames ?? []) {
    if (!frame) continue;
    const [, , volume, unvoicedProb, , , , value] = frame;
    if (value == null || value === -1 || !(unvoicedProb < UNVOICED_THRESHOLD)
        || !Number.isFinite(volume) || volume <= 0) continue;
    const db = 20 * Math.log10(volume);
    count++;
    sum += db;
    min = Math.min(min, db);
    max = Math.max(max, db);
  }
  return count ? { referenceDb: sum / count, min, max } : null;
}

// Pads a raw [y0, y1] domain by `padding` fraction of its span on each side
// (falls back to a scale-aware span when the domain is flat/degenerate) -
// mirrors NoteCurveWidget.set_default_y_range's padding step.
export function paddedRange(y0, y1, padding) {
  let span = y1 - y0;
  if (span <= 0) span = Math.max(Math.abs(y0), Math.abs(y1), 1.0);
  const pad = padding * span;
  return [y0 - pad, y1 + pad];
}

// Maps a pitch contour's own midi span into a band of the chart's y-range
// (CONTOUR_BAND=(0.2,0.8), floored at CONTOUR_MIN_SPAN=1 semitone so a
// dead-flat note doesn't amplify jitter) - mirrors
// NoteCurveWidget._contour_transform's default (non-Timbre) behavior.
export function contourToBand(contour, y0, y1) {
  const midis = contour.map((p) => p.midi).filter((m) => m != null && Number.isFinite(m));
  if (!midis.length) return contour.map((p) => ({ time: p.time, y: null }));
  const lo = Math.min(...midis);
  const hi = Math.max(...midis);
  const center = 0.5 * (lo + hi);
  const span = Math.max(hi - lo, 1.0);
  const b0 = y0 + 0.2 * (y1 - y0);
  const b1 = y0 + 0.8 * (y1 - y0);
  return contour.map((p) => ({
    time: p.time,
    y: p.midi == null ? null : b0 + ((p.midi - (center - span / 2)) / span) * (b1 - b0),
  }));
}

// --- vibrato (JsonHandler._vibrato_to_payload's raw grid) ---
// Unpacks JsonHandler._pack_number's NaN/Infinity string sentinels - there's
// no native way to carry these through JSON, and VibratoData genuinely uses
// NaN as its "not yet computed" marker (unlike the rest of this app's
// payloads, which mostly avoid it).
export function unpackNumber(v) {
  if (v === "nan") return NaN;
  if (v === "inf") return Infinity;
  if (v === "-inf") return -Infinity;
  return v;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// 3-point median where both neighbors are finite - mirrors
// VibratoData._median3: one bad analysis window can't flick the curve
// (isolated nonzero islands and both array ends pass through unchanged).
function median3(values) {
  if (values.length < 3) return values.slice();
  const out = values.slice();
  for (let i = 1; i < values.length - 1; i++) {
    const [prev, cur, next] = [values[i - 1], values[i], values[i + 1]];
    if (Number.isFinite(prev) && Number.isFinite(cur) && Number.isFinite(next)) {
      out[i] = [prev, cur, next].sort((a, b) => a - b)[1];
    }
  }
  return out;
}

// (time, speed, peak-to-peak width) over [t0, t1] - mirrors
// VibratoData.curve(): reads one
// extra grid sample past each end so the 3-point median smooths edge values
// the same way it smooths the interior, then crops back to the window.
export function vibratoCurve(vibratoPoints, t0, t1) {
  if (!vibratoPoints?.length) return [];
  const i0 = vibratoPoints.findIndex((p) => p[0] >= t0);
  if (i0 === -1) return [];
  let i1 = vibratoPoints.length;
  for (let i = i0; i < vibratoPoints.length; i++) {
    if (vibratoPoints[i][0] > t1) { i1 = i; break; }
  }
  const j0 = Math.max(0, i0 - 1);
  const j1 = Math.min(vibratoPoints.length, i1 + 1);
  const slice = vibratoPoints.slice(j0, j1);
  const rates = median3(slice.map((p) => unpackNumber(p[1])));
  const widths = median3(slice.map((p) => unpackNumber(p[2])));
  const offset = i0 - j0;
  const count = i1 - i0;
  const out = [];
  for (let k = 0; k < count; k++) {
    out.push({ time: slice[offset + k][0], rate: rates[offset + k], width: widths[offset + k] });
  }
  return out;
}

// Recording-wide (min, max) for "rate" or "width", over the same
// median-smoothed values curve() uses, only samples with a positive detected
// rate (the stored 0/0 sentinel means "no measurable vibrato", not silence -
// including it would make every take's least-vibrato endpoint zero rather
// than the subtlest vibrato the performer actually produced). Mirrors
// VibratoData.global_characteristic_range.
export function vibratoGlobalRange(vibratoPoints, metric) {
  if (!vibratoPoints?.length) return null;
  const rates = median3(vibratoPoints.map((p) => unpackNumber(p[1])));
  const widths = median3(vibratoPoints.map((p) => unpackNumber(p[2])));
  const values = metric === "rate" ? rates : widths;
  let lo = Infinity, hi = -Infinity, any = false;
  for (let i = 0; i < rates.length; i++) {
    if (Number.isFinite(rates[i]) && Number.isFinite(widths[i]) && rates[i] > 0) {
      any = true;
      if (values[i] < lo) lo = values[i];
      if (values[i] > hi) hi = values[i];
    }
  }
  return any ? [lo, hi] : null;
}

// Per-note median (speed_hz, peak-to-peak width_cents), or null - mirrors
// VibratoData.note_summary: RAW (unsmoothed) samples within the note's own
// span, gated on the note being long enough to contain vibMinCycles at its
// own median rate (config.vib_min_cycles - see analysisResult.config).
export function vibratoNoteSummary(vibratoPoints, note, vibMinCycles) {
  if (!vibratoPoints?.length || !note) return null;
  const rates = [];
  const widths = [];
  for (const p of vibratoPoints) {
    if (p[0] < note.startTime || p[0] > note.endTime) continue;
    const rate = unpackNumber(p[1]);
    const width = unpackNumber(p[2]);
    if (Number.isFinite(rate) && Number.isFinite(width) && rate > 0) {
      rates.push(rate);
      widths.push(width);
    }
  }
  if (!rates.length) return null;
  const rate = median(rates);
  const width = median(widths);
  const duration = Math.max(0, note.endTime - note.startTime);
  if (duration * rate < Math.max(0, vibMinCycles ?? 0)) return null;
  return { rate, width };
}
