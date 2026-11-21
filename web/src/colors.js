// Client-side port of the volume-coloring half of ui/Colors.py + the
// PitchData volume methods it consumes (app_logic/user/ds/PitchData.py).
// Mistake-role colors (deletion/substitution/timing) already live as a
// static SCORE_THEME constant in ScoreViewer.svelte - this file is only the
// part that needs real computation: the take's own loudness range and the
// viridis ramp built from it.
import { getVoicedPitchFrames } from "./PitchFrameHandler.js";
import { volumeLevelStats } from "./noteCurve.js";

// --- ui.Colors.VIRIDIS_ANCHORS / SCORE_DIM / VOL_LIVE_FLOOR_DB ---
const VIRIDIS_ANCHORS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];
const SCORE_DIM = 0.9;
function dim(rgb, factor = SCORE_DIM) {
  return rgb.map((c) => Math.round(c * factor));
}

export function cssRgb(rgb) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

// Linear interpolation along an anchor list for a 0..1 fraction (ui.Colors.ramp).
function ramp(anchors, frac, dimmed = false) {
  const t = Math.max(0, Math.min(1, frac)) * (anchors.length - 1);
  const i = Math.min(Math.floor(t), anchors.length - 2);
  const localT = t - i;
  const [c0, c1] = [anchors[i], anchors[i + 1]];
  const rgb = c0.map((a, k) => Math.round(a + (c1[k] - a) * localT));
  return dimmed ? dim(rgb) : rgb;
}

export function viridis(frac, dimmed = false) {
  return ramp(VIRIDIS_ANCHORS, frac, dimmed);
}

// Review uses the same symmetric, linear-dB mapping as VolumeColorScale.
export function volumeColorScale(pitchFrames) {
  const stats = volumeLevelStats(pitchFrames);
  if (!stats) return null;
  return { referenceDb: stats.referenceDb,
           halfRangeDb: Math.max(stats.referenceDb - stats.min, stats.max - stats.referenceDb) };
}

export function volumeFractionDb(db, scale) {
  if (!scale || db == null || !Number.isFinite(db)) return null;
  const { referenceDb, halfRangeDb } = scale;
  if (halfRangeDb <= 1e-10) return 0.5;
  return Math.max(0, Math.min(1, 0.5 + (db - referenceDb) / (2 * halfRangeDb)));
}

export function volumeFrac(volume, scale) {
  return Number.isFinite(volume) && volume > 0
    ? volumeFractionDb(20 * Math.log10(volume), scale) : null;
}

export function meanLevelDb(pitchFrames, startTime, endTime) {
  const volumes = voicedVolumes(pitchFrames, startTime, endTime);
  return volumes.length ? volumes.reduce((sum, v) => sum + 20 * Math.log10(v), 0) / volumes.length : null;
}

// Voiced frames' volumes in [startTime, endTime] - mirrors PitchData.read(clean=True).
function voicedVolumes(pitchFrames, startTime = -Infinity, endTime = Infinity) {
  const vols = [];
  for (const frame of getVoicedPitchFrames(pitchFrames, startTime, endTime)) {
    const [, , volume] = frame;
    if (Number.isFinite(volume) && volume > 0) vols.push(volume);
  }
  return vols;
}

// (minDb, maxDb) over the whole take's voiced frames - PitchData.volume_range_db.
export function volumeRangeDb(pitchFrames) {
  const vols = voicedVolumes(pitchFrames);
  if (!vols.length) return [null, null];
  return [20 * Math.log10(Math.min(...vols)), 20 * Math.log10(Math.max(...vols))];
}

// Mean voiced-frame volume in [startTime, endTime] - PitchData.mean_volume.
export function meanVolume(pitchFrames, startTime, endTime) {
  const vols = voicedVolumes(pitchFrames, startTime, endTime);
  if (!vols.length) return 0;
  return vols.reduce((a, b) => a + b, 0) / vols.length;
}
