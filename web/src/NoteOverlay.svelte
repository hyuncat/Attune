<script>
  // Static note-level piano-roll matching ui/GuitarHero.py's color system as
  // closely as a note-level (not frame-level) view can:
  //   - the letter-based MIDI "piano key" rainbow background
  //     (GuitarHero.MidiBackground.LETTER_RGB, sharps darkened 70%)
  //   - insertion = GREEN, deletion = RED (GuitarHero.init_colors - the
  //     opposite of the "errors are red" scheme this file used before)
  //   - a substitution's color is a continuous green->red HSV ramp based on
  //     how far its pitch distance is beyond tolerance, not a flat color
  //     (GuitarHero._build_align_brushes / get_align_distance_brush,
  //     ALIGN_MAX_MULT = 4.0)
  //   - dashed white lines connecting matched/substituted pairs
  //     (GuitarHero.match_lines)
  //   - a solid green playhead line at the current playback time
  //     (GuitarHero.timeline, pg.InfiniteLine, colors['timeline'] =
  //     mkPen(0, 255, 0, 255))
  import { noteFromArray, noteName } from "./mistakes.js";
  import { meanVolume, volumeFrac, volumeRangeDb, viridis, cssRgb } from "./colors.js";
  import { vibratoNoteSummary } from "./noteCurve.js";
  import { getVoicedPitchFrames } from "./PitchFrameHandler.js";

  let {
    scoreNotes,
    userNotes,
    pairs,
    pitchMistakes,
    timingMistakes = [],
    pitchTolerance,
    pitchFrames = null,
    vibratoPoints = null,
    vibMinCycles = 1.5,
    currentTime = null,
    selectedMistake = null,
    selectedMistakeOverridden = false,
    onSeek = null,
  } = $props();

  // --- GuitarHero.MidiBackground, ported exactly ---
  const LETTER_RGB = {
    A: [230, 60, 60],
    B: [255, 150, 40],
    C: [245, 220, 70],
    D: [70, 200, 90],
    E: [70, 140, 240],
    F: [100, 90, 210],
    G: [170, 90, 210],
  };
  const PC_TO_LETTER = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"];
  const isSharp = (m) => [1, 3, 6, 8, 10].includes(((m % 12) + 12) % 12);
  function midiToRgba(m, alpha = 0.2) {
    const letter = PC_TO_LETTER[((m % 12) + 12) % 12];
    let [r, g, b] = LETTER_RGB[letter];
    if (isSharp(m)) {
      r *= 0.7;
      g *= 0.7;
      b *= 0.7;
    }
    return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha})`;
  }

  // --- GuitarHero._build_align_brushes / get_align_distance_brush, ported ---
  // QColor.setHsv(hue, 255, 255) (full saturation+value) is exactly
  // hsl(hue, 100%, 50%) - same color space, no conversion loss.
  const ALIGN_MAX_MULT = 4.0;
  function alignHue(distance, tolerance) {
    const greenThresh = Math.max(tolerance, 0);
    const maxDist = Math.max(ALIGN_MAX_MULT * greenThresh, greenThresh + 0.05);
    const d = Math.min(Math.abs(distance), maxDist);
    if (d <= greenThresh) return 120;
    const frac = Math.max(0, Math.min(1, (d - greenThresh) / (maxDist - greenThresh)));
    return 120 * (1 - frac);
  }
  const hsl = (hue, alpha = 1) => `hsla(${hue}, 100%, 50%, ${alpha})`;

  const SCORE_NOTE_COLOR = "rgba(255, 255, 255, 0.78)";
  const DELETION_COLOR = "rgba(255, 0, 0, 0.78)";
  const INSERTION_COLOR = "rgba(0, 200, 0, 0.78)";
  const MATCH_LINE_COLOR = "rgba(255, 255, 255, 0.55)";

  let mistakeByUserIdx = $derived(
    new Map(pitchMistakes.filter((m) => m.userIdx != null).map((m) => [m.userIdx, m]))
  );
  let deletedScoreIndices = $derived(
    new Set(pitchMistakes.filter((m) => m.type === "deletion").map((m) => m.scoreIdx))
  );

  let scoreNotesParsed = $derived(scoreNotes.map(noteFromArray));
  let userNotesParsed = $derived(userNotes.map(noteFromArray));

  // --- GuitarHero.update_user_items' pitch scatter, ported ---
  // Only the first (most probable) candidate is plotted per frame, matching
  // GuitarHero's `break` after the first candidate_pitches entry. Voicing
  // filter (PitchData.is_voiced_pitch) lives in getVoicedPitchFrames now.
  const REST_COLOR = "rgb(140, 140, 140)";

  // --- GuitarHero's own "Colors:" dropdown (color_mode) - independent of
  // ScoreViewer's scoreColorMode: pitch (distance-from-target ramp, default)
  // or volume (viridis ramp over this take's own loudest/quietest frame).
  // Only the pitch-frame scatter dots recolor - note bars always stay on the
  // mistake-based coloring above (PitchDataUI.update_view, ported).
  let colorMode = $state("pitch");
  let volumeRange = $derived(pitchFrames ? volumeRangeDb(pitchFrames) : [null, null]);

  let pitchPoints = $derived.by(() => {
    if (!pitchFrames) return [];
    const points = [];
    for (const frame of getVoicedPitchFrames(pitchFrames, -Infinity, Infinity)) {
      const [time, candidates, volume, , , alignedDistance, isTransition] = frame;
      if (!candidates || candidates.length === 0) continue;
      const midi = candidates[0][0];
      let color;
      if (colorMode === "volume") {
        const frac = volumeFrac(volume, volumeRange[0], volumeRange[1]);
        color = cssRgb(viridis(frac));
      } else {
        color = isTransition
          ? REST_COLOR
          : alignedDistance != null
            ? hsl(alignHue(alignedDistance, pitchTolerance))
            : REST_COLOR;
      }
      points.push({ time, midi, color });
    }
    return points;
  });

  function topPitch(note) {
    return Math.max(...note.midiNum.filter((m) => m >= 0));
  }

  // color for a user note bar: green/red for insertion/deletion-adjacent
  // cases don't apply to user notes directly (insertions are user notes with
  // no score match), substitutions ramp by distance, clean matches are solid
  // green (matches GuitarHero: d <= tolerance is always hue 120, no partial
  // ramp inside the tolerance zone itself).
  function userNoteColor(i) {
    const mistake = mistakeByUserIdx.get(i);
    if (!mistake) return hsl(120); // clean match
    if (mistake.type === "insertion") return INSERTION_COLOR;
    if (mistake.type === "substitution") return hsl(alignHue(mistake.distance, pitchTolerance));
    return hsl(120); // deletion has no user note to color
  }

  let allTimes = $derived([
    ...scoreNotesParsed.flatMap((n) => [n.startTime, n.endTime]),
    ...userNotesParsed.flatMap((n) => [n.startTime, n.endTime]),
    ...pitchPoints.map((p) => p.time),
  ]);
  let allPitches = $derived(
    [
      ...[...scoreNotesParsed, ...userNotesParsed].map(topPitch).filter((p) => Number.isFinite(p)),
      ...pitchPoints.map((p) => p.midi),
    ]
  );

  const WIDTH = 800;
  const HEIGHT = 280;
  const BAR_HEIGHT = 6;
  const PITCH_DOT_RADIUS = 2.5;
  const PITCH_PADDING = 3;

  let minTime = $derived(allTimes.length ? Math.min(...allTimes) : 0);
  let maxTime = $derived(allTimes.length ? Math.max(...allTimes) : 1);
  let minPitch = $derived((allPitches.length ? Math.min(...allPitches) : 60) - PITCH_PADDING);
  let maxPitch = $derived((allPitches.length ? Math.max(...allPitches) : 72) + PITCH_PADDING);

  function xPos(t) {
    const span = maxTime - minTime || 1;
    return ((t - minTime) / span) * WIDTH;
  }
  function yPos(midi) {
    const span = maxPitch - minPitch || 1;
    return HEIGHT - ((midi - minPitch) / span) * HEIGHT;
  }

  // one thin horizontal band per semitone in view, colored by the MIDI
  // rainbow key scheme - the piano-roll background GuitarHero always shows.
  let midiBands = $derived.by(() => {
    const bands = [];
    const lo = Math.floor(minPitch);
    const hi = Math.ceil(maxPitch);
    for (let m = lo; m <= hi; m++) {
      bands.push({ midi: m, y: yPos(m + 0.5), color: midiToRgba(m) });
    }
    return bands;
  });

  const semitoneHeight = $derived(HEIGHT / (maxPitch - minPitch || 1));

  // dashed lines between matched/substituted pairs (both sides present) -
  // mirrors GuitarHero's match_lines (goods + subs, not insertions/deletions).
  let matchLines = $derived.by(() => {
    if (!pairs) return [];
    const lines = [];
    for (const [userIdx, scoreIdx] of pairs) {
      if (userIdx == null || scoreIdx == null) continue;
      const u = userNotesParsed[userIdx];
      const s = scoreNotesParsed[scoreIdx];
      if (!u || !s) continue;
      lines.push({
        x1: xPos((u.startTime + u.endTime) / 2),
        y1: yPos(topPitch(u)),
        x2: xPos((s.startTime + s.endTime) / 2),
        y2: yPos(topPitch(s)),
      });
    }
    return lines;
  });

  // GuitarHero.highlight_mistake: one box per note involved in the selected
  // mistake (substitution/timing: both user+score notes; insertion: user
  // only; deletion: score only), each independently positioned - not a
  // single bounding box merging them, since a substitution's user/score
  // notes can land at different times/pitches.
  const HIGHLIGHT_HEIGHT = BAR_HEIGHT * 2.5;
  let highlightRects = $derived.by(() => {
    if (!selectedMistake) return [];
    const notes =
      selectedMistake.type === "substitution"
        ? [selectedMistake.userNote, selectedMistake.scoreNote]
        : selectedMistake.type === "insertion"
          ? [selectedMistake.userNote]
          : selectedMistake.type === "deletion"
            ? [selectedMistake.scoreNote]
            : [selectedMistake.userNote, selectedMistake.scoreNote]; // timing
    return notes.filter(Boolean).map((n) => ({
      x: xPos(n.startTime),
      width: Math.max(1.5, xPos(n.endTime) - xPos(n.startTime)),
      y: yPos(topPitch(n)) - HIGHLIGHT_HEIGHT / 2,
      height: HIGHLIGHT_HEIGHT,
    }));
  });

  // --- NOTE CLICK POPUP (GuitarHero.select_note / NotePopupGH, ported) ---
  // Own click/hover/popup surface, independent of the mistake-table
  // selection above (selectedMistake/highlightRects) - mirrors GuitarHero's
  // _popup_note being separate state from its mistake-highlight path.
  // Vibrato/volume rows are intentionally omitted (see NoteInfo.analyze):
  // neither vibrato detection nor per-frame volume has been ported to the
  // web client yet, so there's nothing to show there until they are - a
  // later, separate module, not a pitch-mode concern.
  let selectedNoteIdx = $state(null);
  let hoveredNoteIdx = $state(null);
  let popupPos = $state({ x: 0, y: 0 });

  function primaryMidi(note) {
    if (!note?.midiNum?.length) return null;
    const midi = note.midiNum[0];
    return midi == null || midi < 0 ? null : midi;
  }

  // userIdx -> {onset: "early"|"late"|null, duration: "long"|"short"|null} -
  // mirrors NoteInfo._timing_mistakes, which shows both independent of
  // whichever tab (pitch/timing) ResultsView currently has active.
  let timingByUserIdx = $derived.by(() => {
    const map = new Map();
    for (const m of timingMistakes) {
      if (m.userIdx == null) continue;
      const entry = map.get(m.userIdx) ?? { onset: null, duration: null };
      if (m.type === "early" || m.type === "late") entry.onset = m.type;
      else if (m.type === "long" || m.type === "short") entry.duration = m.type;
      map.set(m.userIdx, entry);
    }
    return map;
  });

  const TIMING_LABELS = { early: "Early", late: "Late", long: "Too long", short: "Too short" };

  // 0..1 bar position within the take's own [quietest, loudest] range - NULL
  // (not a color) when nothing voiced was measured, matching NoteInfo._volume.
  function noteVolumeFrac(note) {
    if (!pitchFrames || !note) return null;
    const volume = meanVolume(pitchFrames, note.startTime, note.endTime);
    if (volume <= 0) return null;
    return volumeFrac(volume, volumeRange[0], volumeRange[1]);
  }

  let noteInfo = $derived.by(() => {
    if (selectedNoteIdx == null) return null;
    const note = userNotesParsed[selectedNoteIdx];
    if (!note) return null;
    const midi = primaryMidi(note);
    const timing = timingByUserIdx.get(selectedNoteIdx) ?? { onset: null, duration: null };
    return {
      noteName: midi != null ? noteName(midi) : "—",
      cents: midi != null ? (midi - Math.round(midi)) * 100 : 0,
      onset: note.startTime,
      duration: note.endTime - note.startTime,
      onsetMistake: timing.onset,
      durationMistake: timing.duration,
      volumeFrac: noteVolumeFrac(note),
      vibrato: vibratoNoteSummary(vibratoPoints, note, vibMinCycles),
    };
  });

  function selectNote(i, clientPos = null) {
    selectedNoteIdx = i;
    const note = userNotesParsed[i];
    if (clientPos) popupPos = clientPos;
    if (note) onSeek?.(note.startTime);
  }

  function closePopup() {
    selectedNoteIdx = null;
  }

  function handleNoteClick(ev, i) {
    ev.stopPropagation();
    const wrapRect = ev.currentTarget.closest(".overlay-wrap").getBoundingClientRect();
    selectNote(i, { x: ev.clientX - wrapRect.left, y: ev.clientY - wrapRect.top });
  }

  function handleNoteHover(i) {
    hoveredNoteIdx = i;
  }
  function handleNoteLeave() {
    hoveredNoteIdx = null;
  }

  function stepNote(dir) {
    if (selectedNoteIdx == null) return;
    const next = selectedNoteIdx + dir;
    if (next < 0 || next >= userNotesParsed.length) return;
    selectNote(next);
  }

  function handleKeydown(ev) {
    if (selectedNoteIdx == null) return;
    if (ev.key === "ArrowLeft") { ev.preventDefault(); stepNote(-1); }
    else if (ev.key === "ArrowRight") { ev.preventDefault(); stepNote(1); }
    else if (ev.key === "Escape") { closePopup(); }
  }

  // dims the hovered note's own pitch dots (GuitarHero._hover_span) so its
  // frame-by-frame detail reads clearly against the flat summary bar
  let hoveredSpan = $derived(
    hoveredNoteIdx != null && userNotesParsed[hoveredNoteIdx]
      ? [userNotesParsed[hoveredNoteIdx].startTime, userNotesParsed[hoveredNoteIdx].endTime]
      : null
  );
  function pitchDotOpacity(t) {
    return hoveredSpan && t >= hoveredSpan[0] && t <= hoveredSpan[1] ? 0.35 : 1;
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="overlay-wrap">
<svg viewBox="0 0 {WIDTH} {HEIGHT}" class="overlay" role="img" aria-label="Pitch overlay" onclick={closePopup}>
  {#each midiBands as band}
    <rect x="0" y={band.y - semitoneHeight / 2} width={WIDTH} height={semitoneHeight} fill={band.color} />
  {/each}

  {#each matchLines as line}
    <line x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} class="match-line" />
  {/each}

  {#each scoreNotesParsed as note, i}
    <rect
      x={xPos(note.startTime)}
      y={yPos(topPitch(note)) - BAR_HEIGHT / 2}
      width={Math.max(1.5, xPos(note.endTime) - xPos(note.startTime))}
      height={BAR_HEIGHT}
      fill={deletedScoreIndices.has(i) ? DELETION_COLOR : SCORE_NOTE_COLOR}
    />
  {/each}

  {#each userNotesParsed as note, i}
    <rect
      x={xPos(note.startTime)}
      y={yPos(topPitch(note)) - BAR_HEIGHT / 2}
      width={Math.max(1.5, xPos(note.endTime) - xPos(note.startTime))}
      height={BAR_HEIGHT}
      fill={userNoteColor(i)}
      class="user-note"
      class:hovered={hoveredNoteIdx === i}
    />
  {/each}

  {#each pitchPoints as p}
    <circle cx={xPos(p.time)} cy={yPos(p.midi)} r={PITCH_DOT_RADIUS} fill={p.color} opacity={pitchDotOpacity(p.time)} />
  {/each}

  {#if currentTime != null && currentTime >= minTime && currentTime <= maxTime}
    <line x1={xPos(currentTime)} y1="0" x2={xPos(currentTime)} y2={HEIGHT} class="playhead" />
  {/if}

  {#each highlightRects as r}
    <rect x={r.x} y={r.y} width={r.width} height={r.height} class="mistake-highlight" class:overridden={selectedMistakeOverridden} />
  {/each}

  <!-- Invisible, enlarged click/hover targets for user notes - drawn last
       (highest z) so they always win hit-testing over the pitch dots/lines
       drawn on top of the thin visible bars. Height mirrors GuitarHero's own
       CLICK_SLACK (±0.5 semitones around the note's pitch, so it scales with
       the current pitch-range zoom the same way desktop's slack does,
       instead of a fixed pixel guess); width gets a slightly bigger floor
       than the visible bar's so short/grace notes stay clickable. -->
  {#each userNotesParsed as note, i}
    <rect
      x={xPos(note.startTime)}
      y={yPos(topPitch(note)) - semitoneHeight / 2}
      width={Math.max(4, xPos(note.endTime) - xPos(note.startTime))}
      height={semitoneHeight}
      fill="transparent"
      class="user-note-hit"
      onclick={(ev) => handleNoteClick(ev, i)}
      onmouseenter={() => handleNoteHover(i)}
      onmouseleave={handleNoteLeave}
    />
  {/each}
</svg>

{#if noteInfo}
  <div class="note-popup" style="left:{popupPos.x + 12}px; top:{popupPos.y + 12}px" onclick={(ev) => ev.stopPropagation()}>
    <p><b>Pitch:</b> {noteInfo.noteName} {noteInfo.cents >= 0 ? "+" : ""}{noteInfo.cents.toFixed(0)}¢</p>
    <p><b>Onset:</b> {noteInfo.onset.toFixed(2)}s{noteInfo.onsetMistake ? ` (${TIMING_LABELS[noteInfo.onsetMistake]})` : ""}</p>
    <p><b>Duration:</b> {noteInfo.duration.toFixed(2)}s{noteInfo.durationMistake ? ` (${TIMING_LABELS[noteInfo.durationMistake]})` : ""}</p>
    {#if noteInfo.vibrato}
      <p><b>Vibrato:</b> f={noteInfo.vibrato.rate.toFixed(1)}Hz, W={noteInfo.vibrato.width.toFixed(0)}¢ peak-to-peak</p>
    {:else}
      <p><b>Vibrato:</b> —</p>
    {/if}
    {#if noteInfo.volumeFrac != null}
      <p class="volume-row">
        <b>Volume:</b>
        <span class="volume-bar"><span class="volume-marker" style="left:{noteInfo.volumeFrac * 100}%"></span></span>
      </p>
    {:else}
      <p><b>Volume:</b> —</p>
    {/if}
  </div>
{/if}
</div>

<div class="legend-row">
  <p class="legend">
    {#if colorMode === "volume"}
      <span class="legend-item"><span class="swatch volume-swatch"></span>quiet → loud</span>
    {:else}
      <span class="legend-item"><span class="swatch" style="background:{hsl(120)}"></span>correct</span>
      <span class="legend-item"><span class="swatch" style="background:{hsl(60)}"></span>off-pitch</span>
      <span class="legend-item"><span class="swatch" style="background:{hsl(0)}"></span>way off</span>
    {/if}
    <span class="legend-item"><span class="swatch" style="background:{INSERTION_COLOR}"></span>extra note</span>
    <span class="legend-item"><span class="swatch" style="background:{DELETION_COLOR}"></span>missed note</span>
  </p>
  <label class="color-mode-picker">
    Colors:
    <select bind:value={colorMode}>
      <option value="pitch">Pitch</option>
      <option value="volume">Volume</option>
    </select>
  </label>
</div>

<style>
  .overlay {
    width: 100%;
    height: 280px;
    background: rgb(20, 20, 25);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .match-line {
    stroke: rgba(255, 255, 255, 0.55);
    stroke-width: 1.5;
    stroke-dasharray: 4, 3;
  }
  /* GuitarHero.timeline: solid green, above everything else in the plot */
  .playhead {
    stroke: rgb(0, 255, 0);
    stroke-width: 1.5;
  }
  /* GuitarHero.highlight_bar: red normally, green if the mistake is
     overridden - exact brush/pen values (255,80,80)/(80,255,80), alpha
     130/255. Drawn last (highest z, above the playhead even). */
  .mistake-highlight {
    fill: rgba(255, 80, 80, 0.51);
    stroke: rgb(255, 80, 80);
    stroke-width: 2;
  }
  .mistake-highlight.overridden {
    fill: rgba(80, 255, 80, 0.51);
    stroke: rgb(80, 255, 80);
  }
  .user-note {
    opacity: 0.95;
  }
  .user-note.hovered {
    opacity: 1;
  }
  .user-note-hit {
    cursor: pointer;
  }
  .overlay-wrap {
    position: relative;
  }
  /* NotePopupGH, ported: same dark rounded box, same row styling. */
  .note-popup {
    position: absolute;
    z-index: 10;
    background: rgb(32, 33, 38);
    border: 1px solid rgb(95, 95, 105);
    border-radius: 6px;
    padding: 9px 12px;
    font-size: 12px;
    color: rgb(228, 231, 235);
    white-space: nowrap;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  }
  .note-popup p {
    margin: 0;
    padding: 1.5px 0;
  }
  .legend-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-top: 0.5rem;
  }
  .legend {
    font-size: 0.8rem;
    color: var(--text-secondary);
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    row-gap: 0.3rem;
    column-gap: 0.7rem;
    margin: 0;
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
  }
  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 0.2rem;
  }
  .volume-swatch {
    background: linear-gradient(to right, rgb(68, 1, 84), rgb(59, 82, 139), rgb(33, 145, 140), rgb(94, 201, 98), rgb(253, 231, 37));
  }
  .color-mode-picker {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    flex-shrink: 0;
  }
  .color-mode-picker select {
    background: var(--bg-input, #2a2b30);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 4px;
    font-size: 0.8rem;
  }
  /* NotePopupGH's _volume_rows: a thin viridis gradient strip with a marker
     at the note's own quiet<->loud position, no absolute number. */
  .volume-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .volume-bar {
    position: relative;
    flex: 1;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(to right, rgb(68, 1, 84), rgb(59, 82, 139), rgb(33, 145, 140), rgb(94, 201, 98), rgb(253, 231, 37));
  }
  .volume-marker {
    position: absolute;
    top: -2px;
    width: 2px;
    height: 12px;
    background: white;
    transform: translateX(-1px);
  }
</style>
