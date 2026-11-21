<script>
  // Mirrors ui/score/ScoreViewer.py: owns the Verovio iframe, tracks readiness,
  // and exposes the same JS-API wrapper methods Python calls via runJavaScript
  // (window.loadScore, window.timeChanged, window.getMeasureTimemap,
  // window.getClipSelection, window.clearClipSelection, window.setClipRange,
  // window.clearClipRange, window.setMistakeAnnotations,
  // window.setAnnotationColorMode - see ui/score/verovio/viewer.js). Same-origin
  // iframe, so these are direct synchronous calls instead of Python's
  // runJavaScript(..., callback) dance. Note clicks push the other way (JS ->
  // host) via window.setBridge, since there's no QWebChannel outside Qt.
  // onNoteClicked/onAnnotationClicked: mirrors ScoreViewer.py's note_clicked/
  // annotation_clicked signals (perform.py's on_note_clicked/on_annotation_clicked) -
  // called with a time in seconds when a note or mistake annotation is clicked
  // in the score. Wired to the iframe's window.bridge via the ported viewer.js's
  // window.setBridge() (see ui/score/verovio/viewer.js), since a same-origin
  // iframe has no QWebChannel to push through.
  // onReady: fires exactly once, the instant the iframe finishes loading -
  // an explicit callback rather than relying on the parent reactively
  // noticing `ready` flip through a cross-component property read, which
  // proved unreliable in practice (see App.svelte's score-load effect).
  let { onNoteClicked = null, onAnnotationClicked = null, onReady = null } = $props();

  // Mirrors ui/Colors.py's score_theme(): the --score-<role> custom
  // properties viewer.css's mistake/insertion-marker rules read (see
  // "!important" fill/stroke rules keyed on .mistake-<role>). Python computes
  // these from MISTAKE_RGB/CURRENT_RGB (SCORE_DIM = 0.9, dimmed except
  // 'current'); ported as a static constant here rather than recomputing,
  // since this app has no theme switcher to make that dynamic yet.
  const SCORE_THEME = {
    substitution: "rgb(212, 130, 0)",
    insertion: "rgb(198, 30, 0)",
    timing: "rgb(198, 30, 0)",
    deletion: "rgb(198, 30, 0)",
    current: "rgb(0, 110, 154)",
  };

  let ready = $state(false);
  export { ready };
  let iframeEl;

  function handleLoad() {
    ready = true;
    iframeEl.contentWindow.setThemeColors?.(SCORE_THEME);
    iframeEl.contentWindow.setBridge?.({
      noteClicked: (sec) => onNoteClicked?.(sec),
      annotationClicked: (sec) => onAnnotationClicked?.(sec),
    });
    onReady?.();
  }

  function callIframe(fnName, ...args) {
    if (!ready || !iframeEl?.contentWindow) return undefined;
    const fn = iframeEl.contentWindow[fnName];
    if (typeof fn !== "function") return undefined;
    return fn(...args);
  }

  // 32KB chunks to avoid call-stack limits from spreading a large byte array
  // into String.fromCharCode at once.
  function bytesToBase64(bytes) {
    let binary = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
    }
    return btoa(binary);
  }

  // xmlBytes: Uint8Array of MusicXML content (uncompressed text, matching what
  // ScoreData.to_musicxml_bytes() produces server-side - not a raw .mxl zip).
  //
  // onRendered(height), if given, fires once the score has actually produced
  // visible geometry (see attemptLoad below) - App.svelte uses it to size the
  // score pane to the rendered line (mirrors perform.py's
  // _fit_score_viewer_height).
  let loadVersion = 0;
  export function loadScore(xmlBytes, onRendered) {
    if (!ready) {
      console.warn("[ScoreViewer] loadScore called before ready, ignoring.");
      return;
    }
    attemptLoad(bytesToBase64(xmlBytes), 0, onRendered, ++loadVersion);
  }

  // Verovio's toolkit can report itself constructed before its engraver is
  // actually able to produce real geometry (see viewer.js's calledRun
  // fast-path) - the very first renderToSVG() right after that can silently
  // come back as an empty 0x0-page SVG, with getContentHeight() reporting 0.
  // This doesn't self-heal by waiting (confirmed by polling for seconds with
  // no re-trigger), so it needs an explicit re-call - and the retry has to
  // live out here on the host rather than inside viewer.js's own script
  // context: an iframe-internal setTimeout retry was tried first and never
  // fired at all (a freshly-created/just-shown iframe's own timers can stall
  // independently of the host page's), while the host's timers fire
  // reliably.
  function attemptLoad(b64, attempt, onRendered, version) {
    if (version !== loadVersion) return;
    callIframe("loadScore", b64);
    const height = callIframe("getContentHeight");
    // The empty notation container has padding, so a positive height alone
    // does not mean the WASM toolkit has rendered the preloaded score.
    const hasNotes = iframeEl.contentDocument?.querySelector("#notation svg .note");
    if (height > 0 && hasNotes) {
      onRendered?.(height);
    } else if (attempt < 100) {
      setTimeout(() => attemptLoad(b64, attempt + 1, onRendered, version), 60);
    } else {
      console.warn("[ScoreViewer] score render kept coming back empty after retries");
    }
  }

  export function setPlaybackTime(sec) {
    callIframe("timeChanged", sec);
  }

  // Mirrors ScoreViewer.py's set_mistake_annotations/set_annotation_color_mode:
  // annotations is the {notes, insertions, noteMeta, volumes} shape built by
  // web/src/annotations.js's buildAnnotations().
  export function setMistakeAnnotations(annotations) {
    callIframe("setMistakeAnnotations", annotations);
  }

  export function setAnnotationColorMode(mode) {
    callIframe("setAnnotationColorMode", mode);
  }

  export function getMeasureTimemap() {
    return callIframe("getMeasureTimemap");
  }

  export function getClipSelection() {
    return callIframe("getClipSelection");
  }

  export function clearClipSelection() {
    callIframe("clearClipSelection");
  }

  export function setClipRange(startIdx, endIdx) {
    callIframe("setClipRange", startIdx, endIdx);
  }

  export function clearClipRange() {
    callIframe("clearClipRange");
  }

  export function isReady() {
    return ready;
  }
</script>

<div class="score-viewer">
  {#if !ready}
    <div class="loading">Loading...</div>
  {/if}
  <iframe
    bind:this={iframeEl}
    onload={handleLoad}
    src="/verovio/viewer.html"
    title="Score viewer"
    class:hidden={!ready}
  ></iframe>
</div>

<style>
  .score-viewer {
    /* NOT min-height:300px (used to be, harmlessly, before App.svelte's
       score-pane auto-fit existed - .score-pane always had plenty of height
       by default back then). Now that .score-pane is deliberately bounded
       to the rendered content's real height, this floor fought that sizing:
       the iframe (height:100% of this element) would overflow past its
       actual allotted height and visually cover .score-legend-row below it,
       even though the legend's own layout box was correctly positioned -
       getBoundingClientRect() has no idea an overflowing sibling is
       painting over it. */
    position: relative;
    width: 100%;
    height: 100%;
  }
  .loading {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary, #999);
    background: var(--bg-window, #202124);
  }
  iframe {
    width: 100%;
    height: 100%;
    border: none;
  }
  iframe.hidden {
    visibility: hidden;
  }
</style>
