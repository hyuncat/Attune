"""FastAPI /analyze endpoint.

Upload a score (MIDI/MusicXML) + a recording, run Attune's existing analysis
pipeline unchanged, get the results back as JSON. This is a thin adapter, not
a reimplementation: every analysis step below is the same call sequence
app.py/perform.py already make (Recording.load_audio + detect_pitches for
import, PerformTab.analyze()'s note/mistake pipeline for analysis), just
triggered by an HTTP request instead of a button click, and serialized with
the same JsonHandler used for the desktop app's local `.json.xz` cache.

Run (from the Attune project root):
    pip install -r web/api/requirements.txt
    uvicorn web.api.analyze_api:app --reload --app-dir .
"""
import base64
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# This file lives at web/api/analyze_api.py; make the Attune project root (two
# levels up) importable regardless of the working directory it's launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.Config import Config  # noqa: E402
from algorithms.MistakeDetector import MistakeDetector  # noqa: E402
from app_logic.JsonHandler import JsonHandler  # noqa: E402
from app_logic.midi.ScoreData import ScoreData  # noqa: E402
from app_logic.user.ds.Recording import Recording  # noqa: E402

# ScoreData.load()'s supported extensions
SUPPORTED_SCORE_EXTENSIONS = {".mid", ".midi", ".mxl", ".musicxml", ".xml", ".mei"}
# Audio extensions supported by Soundfile and pydub
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".wave", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}

app = FastAPI(title="Attune Analyze API")

# allow_origin_regex covers local dev (any localhost/127.0.0.1 port - Vite's
# port isn't fixed, autoPort reassigns it if 5173 is taken), allow_origins
# covers the real deployed frontend. Starlette's CORSMiddleware checks a
# request's origin against both independently, so a match on either is enough
# - no need to fold the deployed origin into the regex.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_origins=["https://main.dz0hpijjgdav5.amplifyapp.com"],
    #Post is the only relevant method because frontend will only send Post to the API
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}



@app.post("/analyze")
async def analyze(
    score: UploadFile = File(...),
    audio: UploadFile = File(...),
    pitch_tolerance: float | None = Form(None),
    active_instrument: int | None = Form(None),
    fmin: float | None = Form(None),
    fmax: float | None = Form(None),
    tuning: float | None = Form(None),
) -> dict:
    """`pitch_tolerance`, if given, overrides Config's 0.5-semitone default for
    this analysis's alignment step - lets a user who already adjusted the
    tolerance control before ever clicking Analyze (matching the desktop
    app's on_tolerance_applied/reanalyze_if_analyzed behavior - tolerance is
    settable the moment a recording exists, not gated on analysis having run
    yet) get a first alignment that already reflects it, instead of always
    starting at the fixed default regardless of the UI.

    `active_instrument`, if given, overrides ScoreData's own default channel
    pick (get_default_instrument()) - mirrors SettingsWidget's Instrument
    selector, which sets score_data.active_instrument directly. Set before
    Recording(score_data=...) is constructed, since Recording.__init__ copies
    it once at construction time (verified against Recording.py, not
    assumed) - setting it after would silently no-op.

    `fmin`/`fmax` (Hz) and `tuning` (Hz) mirror SettingsWidget's Range and
    Tuning fields, feeding PitchDetector's search bounds via Config. Range in
    particular is a real accuracy lever, not just a display concern -
    narrowing it to the piece's actual pitch span (the client defaults it to
    the score's own range, not the wide general-purpose default) measurably
    reduces PYIN octave-error candidates."""
    #checking filetype
    score_suffix = await check_upload_file(score, SUPPORTED_SCORE_EXTENSIONS)
    # AudioData.load_data only special-cases .m4a (via pydub); everything else
    # falls to soundfile.read(), which can't decode .mp3 - keep this list to
    # what that call path actually supports (mirrors app.py's writable_suffixes).
    audio_suffix = await check_upload_file(
        audio, {".wav", ".wave", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        score_path = Path(tmpdir) / f"score{score_suffix}"
        audio_path = Path(tmpdir) / f"audio{audio_suffix}"
        score_path.write_bytes(await score.read())
        audio_path.write_bytes(await audio.read())

        try:
            score_data = ScoreData()
            score_data.load(str(score_path))
            if active_instrument is not None:
                if active_instrument not in score_data.note_datas:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown instrument channel {active_instrument}.",
                    )
                score_data.active_instrument = active_instrument

            # update_config() runs before PitchDetector/MistakeDetector are
            # constructed in Recording.__init__, and both read recording.config
            # at construction time - so a custom Config here reaches both the
            # PYIN search range (PitchDetector reads fmin/fmax/tuning to set
            # tau_min/tau_max) and the alignment step correctly (verified
            # against Recording.py/PitchDetector.py, not assumed).
            config_kwargs = {}
            if pitch_tolerance is not None:
                config_kwargs["pitch_tolerance"] = pitch_tolerance
            if fmin is not None:
                config_kwargs["fmin"] = fmin
            if fmax is not None:
                config_kwargs["fmax"] = fmax
            if tuning is not None:
                config_kwargs["tuning"] = tuning
            config = Config(**config_kwargs) if config_kwargs else None
            rec = Recording(score_data=score_data, config=config)
            rec.load_audio(str(audio_path), score_filepath=str(score_path), load_cache=False)
            rec.detect_pitches()

            # mirrors perform.py analyze method's call sequence
            rec.reset_analysis()
            rec.detect_notes()
            rec.align_score_and_refine()
            rec.update_alignment_distances()
            rec.trim_end()

            handler = JsonHandler(rec)
            payload = handler.to_cache_payload(
                score_filepath=str(score_path), recording_name="upload"
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Mistake classification now happens client-side, against whatever
    # tolerance is currently set in the UI - the client needs the raw
    # alignment (which user note maps to which score note) to redo that
    # itself, not mistakes pre-filtered at whatever tolerance this server-side
    # Config happened to use. JsonHandler's own payload shape is shared with
    # the desktop app's local .json.xz cache format and deliberately left
    # unmodified - this trims only the API response.
    payload["alignment"] = {"pairs": payload["alignment"]["pairs"]}
    # Vibrato is already computed as a side effect of detect_notes() above
    # (Recording.detect_notes() -> recompute_vibrato(), unconditional) - this
    # is pure serialization, not extra analysis work. Added here rather than
    # in to_cache_payload() itself: vibrato is deliberately never persisted to
    # the desktop app's local cache (cheap to recompute on load), so this key
    # is web-API-only, not part of the shared cache payload shape.
    payload["vibrato"] = JsonHandler._vibrato_to_payload(rec.vibrato_data)
    return payload


@app.post("/notedata")
async def note_data(score: UploadFile = File(...)) -> dict:
    """Parse an uploaded score and return its note-by-note data, per
    instrument channel. No GET counterpart by design - the client caches this
    response itself (keyed by a hash of the score file's content) rather than
    the server persisting it behind an id; see project notes on why. Index i
    in note_data[channel] corresponds to index i in /analyze's alignment pairs
    for that same channel (both walk NoteData.times, which is kept sorted) -
    verified against app_logic/NoteData.py, not assumed.
    """
    score_data = await _parse_uploaded_score(score)
    # _note_data_to_payload doesn't touch JsonHandler's own recording state,
    # so a handler with no Recording attached is fine here - reusing it avoids
    # a second implementation of the same note serialization drifting out of
    # sync with the desktop app's cache format.
    handler = JsonHandler()
    # Full-score MusicXML (same export ScoreViewer.py feeds Verovio via
    # runJavaScript) so the center column can render the actual uploaded
    # score, not a placeholder - fetched once here and cached client-side
    # alongside the rest of /notedata's response (see noteDataCache.js),
    # not repeated in /analyze's response.
    musicxml_b64 = base64.b64encode(score_data.to_musicxml_bytes()).decode("ascii")
    return {
        "title": score_data.title,
        "bpm": score_data.bpm,
        "active_instrument": score_data.active_instrument,
        "instruments": sorted(int(ch) for ch in score_data.instruments),
        "note_data": {
            str(channel): handler._note_data_to_payload(nd)
            for channel, nd in score_data.note_datas.items()
        },
        "musicxml_b64": musicxml_b64,
        # lets the client mute/exclude the metronome click track by default
        # and drive its own Metronome toggle, mirroring Toolbar.py's
        # metronome_toggled/populate_instrument_menu (which skips this
        # channel from the per-instrument checkbox list and gives it its
        # own switch instead). Each note already carries its own MIDI
        # program in note_data[channel][i][5] (Note.instrument, set from
        # MidiData.instruments - a real GM program number, not guessed).
        "metronome_channel": score_data.metronome_channel,
        # ScoreTimeMap inputs (see ui/time/ScoreTimeMap.py): barline onsets in
        # the original-tempo timeframe, paired 1:1 by measure index with
        # Verovio's own measure timemap (ScoreViewer.svelte's
        # getMeasureTimemap()) to anchor the score cursor to the MIDI/NoteData
        # timeline instead of Verovio's own drifting one. channel=None (first
        # part) since the web port always renders the full score - there's no
        # per-instrument re-render here the way desktop's viewer_show_full has.
        "measure_onsets_og": score_data.measure_onsets_og(),
        "bpm_og": score_data.bpm_og,
        "transpose_offset": score_data.transpose_offset,
    }


class RealignRequest(BaseModel):
    # Same array shape /analyze's "note_data" and /notedata's per-channel
    # note_data already return - the client sends back exactly what it
    # already has, no reshaping on either side.
    user_notes: list
    score_notes: list
    pitch_tolerance: float


@app.post("/realign")
async def realign(payload: RealignRequest) -> dict:
    """Return production string-edit pairs while applying a new classification
    tolerance, without re-uploading audio or re-running pitch/note detection.
    Pairing itself uses weighted absolute pitch/onset/duration errors and is
    independent of the classification threshold; pitch_tolerance determines
    which returned diagonal pairs are substitutions.

    Only pitch tolerance goes through this endpoint - timing-mistake
    reclassification (early/late/short/long) is a simple fixed-pairs
    threshold check with no re-alignment involved, so that stays purely
    client-side against /analyze's existing pairs.
    """
    handler = JsonHandler()
    try:
        user_nd = handler._note_data_from_payload(payload.user_notes)
        score_nd = handler._note_data_from_payload(payload.score_notes)
    except (IndexError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Malformed note data: {e}")

    config = Config(pitch_tolerance=payload.pitch_tolerance)
    detector = MistakeDetector(config=config)
    try:
        alignment = detector.detect_mistakes(
            user_notes=user_nd,
            score_notes=score_nd,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Realignment failed: {e}")

    # Index maps built from the SAME ordering (NoteData.times) the client's
    # original arrays used, so returned indices resolve correctly against
    # data the client already holds - reuses the identical helpers
    # JsonHandler._alignment_to_payload uses for /analyze's own pairs, so
    # indexing behavior can't drift between the two endpoints.
    user_notes_full = [user_nd.data[t] for t in user_nd.times]
    score_notes_full = [score_nd.data[t] for t in score_nd.times]
    user_maps = JsonHandler._note_index_maps(user_notes_full)
    score_maps = JsonHandler._note_index_maps(score_notes_full)

    return {
        "pairs": [
            [
                JsonHandler._lookup_note_index(u, user_maps),
                JsonHandler._lookup_note_index(s, score_maps),
            ]
            for u, s in alignment.pairs
        ]
    }


#Helper function to check file extension (refactored for both score and audio to use the same call)
async def check_upload_file(file: UploadFile, allowed_extensions: set[str]) -> str:
    """Check that the uploaded file has an allowed extension, and return it."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {sorted(allowed_extensions)}"
            ),
        )
    return suffix

#helper method for converting an uploaded score file into a ScoreData object, with validation and cleanup - refactored for both analyze and notedata endpoints to use the same call
async def _parse_uploaded_score(score: UploadFile) -> ScoreData:
    """Validate + parse an uploaded score file into a ScoreData object.
    ScoreData.load() extracts everything it needs into the object graph, so
    the temp file is safe to clean up before returning - nothing reads from
    the path afterward."""
    score_suffix = await check_upload_file(score, SUPPORTED_SCORE_EXTENSIONS)
    with tempfile.TemporaryDirectory() as tmpdir:
        score_path = Path(tmpdir) / f"score{score_suffix}"
        score_path.write_bytes(await score.read())
        score_data = ScoreData()
        try:
            score_data.load(str(score_path))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return score_data
