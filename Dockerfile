# syntax=docker/dockerfile:1

# Backend image for web/api/analyze_api.py (the /analyze, /notedata, /realign
# FastAPI endpoints). Build context is the repo root (not web/api/) because
# the API imports algorithms/ and app_logic/ as siblings, unmodified from the
# desktop app - see analyze_api.py's PROJECT_ROOT sys.path insert.
#
# MUST be built for linux/amd64, e.g.:
#   docker build --platform linux/amd64 -t attune-api .
# PyQt6==6.10.1 (see below for why it's required at all) has no published
# wheel for linux/arm64 - only the source sdist, which needs a full Qt6 SDK
# + qmake to build and was not pursued. On an Apple Silicon Mac, `docker
# build` defaults to the host's arm64 and this fails at the pip install step;
# confirmed empirically (arm64 build fails, amd64 build and a subsequent
# headless `import PyQt6.QtCore` both succeed). AWS App Runner should be
# configured for x86_64, not the Graviton/ARM option, to match.
#
# Python 3.12, not the 3.14 local dev uses: numpy==1.26.4 (pinned here,
# shared with the desktop app's requirements.txt) predates Python 3.14 and
# has no prebuilt wheel for it, so pip falls back to a from-source build that
# fails outright without a full C toolchain - confirmed by actually building
# this image, not assumed. 3.12 is the newest version numpy 1.26.4 ships
# wheels for on all platforms.
FROM python:3.12-slim-bookworm

# System packages:
# - ffmpeg: pydub's AudioSegment.from_file(..., format="m4a") shells out to
#   it (app_logic/user/ds/AudioData.py) - the only upload format that isn't
#   handled directly by soundfile.
# - libgl1/libegl1/libxkbcommon0/libdbus-1-3/libnss3/libxcomposite1/
#   libxrandr2/libxi6/libxtst6/libfontconfig1/libglib2.0-0: PyQt6 is a hard
#   import-time dependency of algorithms/PitchDetector.py and NoteDetector.py
#   (they subclass QObject for Qt signals this API path never connects to -
#   see web/api/requirements.txt's header comment). No window is ever opened,
#   but the compiled Qt6 .so files still dlopen these shared libraries the
#   moment PyQt6.QtCore is imported, headless or not - without them the
#   import itself fails before any request-handling code runs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libegl1 \
    libxkbcommon0 \
    libdbus-1-3 \
    libnss3 \
    libxcomposite1 \
    libxrandr2 \
    libxi6 \
    libxtst6 \
    libfontconfig1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies installed before source is copied in, so editing algorithms/
# or app_logic/ doesn't invalidate this layer - only a requirements.txt
# change does. --no-cache-dir keeps pip's download cache out of the image
# layer (BuildKit's cache mount is a further option if rebuild speed matters
# more than a from-scratch build's simplicity - see below).
COPY web/api/requirements.txt web/api/requirements.txt
RUN pip install --no-cache-dir -r web/api/requirements.txt

# Only what /analyze's call path actually imports: algorithms/ + app_logic/
# (the shared analysis engine, unmodified from the desktop app) and web/api/
# (the FastAPI adapter). Deliberately excludes ui/ (desktop PyQt
# widgets/WebEngine - never imported by this path, confirmed via grep),
# resources/ (demo fixtures), web/src + web-resources/ (the frontend,
# deployed separately), notebooks/, experiments/, data/.
COPY algorithms/ algorithms/
COPY app_logic/ app_logic/
COPY web/api/ web/api/

# App Runner (and most container platforms) inject the port to listen on via
# $PORT rather than assuming a fixed value - default kept at 8000 to match
# local dev (see analyze_api.py's own docstring) when run standalone.
ENV PORT=8000
EXPOSE 8000

# No --reload here (that's dev-only: it adds a file-watcher and reimport
# overhead with no benefit in a container that gets rebuilt on every deploy
# instead of edited in place).
CMD ["sh", "-c", "uvicorn web.api.analyze_api:app --app-dir . --host 0.0.0.0 --port ${PORT}"]
