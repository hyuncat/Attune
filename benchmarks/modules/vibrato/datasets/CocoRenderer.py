"""Straight-tone SFZ rendering for the CocoChorales vibrato benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import pretty_midi

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

from benchmarks.paths import REPO_ROOT


COCO_SFIZZ_POLICY_VERSION = "coco_sfizz_straight_tone_v1"
DEFAULT_SOUNDFONTS_ROOT = REPO_ROOT / "resources" / "soundfonts"
_MIDI_RPN_CONTROLLERS = frozenset((6, 38, 100, 101))


class CocoRenderer:
    """Render one monophonic Coco stem through its straight-tone SFZ patch."""

    @dataclass(frozen=True)
    class Patch:
        library: str
        library_version: str
        relative_sfz: str
        pitch_bend_range_semitones: int = 2
        zero_vibrato_ccs: tuple[int, ...] = ()
        bend_range_source: str = "sfz_default"
        render_note_shift_semitones: int = 0

    PATCHES: ClassVar[dict[str, Patch]] = {
        "bassoon": Patch(
            "SSO",
            "4.0",
            "sso-4.0/Sonatina Symphonic Orchestra/"
            "Woodwinds - Performance/Bassoon Solo Sustain (looped).sfz",
        ),
        "cello": Patch(
            "Karoryfer x bigcat Cello",
            "repository checkout",
            "karoryfer-bigcat.cello/Programs/01- Bowed (velocity layer).sfz",
            pitch_bend_range_semitones=12,
            zero_vibrato_ccs=(111, 118, 119),
            bend_range_source="sfz_explicit",
            # This library uses the common sampler convention C3=60 while
            # Coco/pretty_midi use scientific C4=60.  Shift only the hidden
            # renderer copy; benchmark annotations remain concert MIDI.
            render_note_shift_semitones=-12,
        ),
        "clarinet": Patch(
            "SSO",
            "4.0",
            "sso-4.0/Sonatina Symphonic Orchestra/"
            "Woodwinds - Performance/Clarinet Solo Sustain (looped).sfz",
        ),
        "double bass": Patch(
            "SSO",
            "4.0",
            "sso-4.0/Sonatina Symphonic Orchestra/"
            "Strings - Performance/Bass Solo Sustain.sfz",
            zero_vibrato_ccs=(21,),
        ),
        "flute": Patch(
            "SSO",
            "4.0",
            "sso-4.0/Sonatina Symphonic Orchestra/"
            "Woodwinds - Performance/Flute Solo 2 Sustain Non-Vibrato.sfz",
        ),
        "horn": Patch(
            "SSO",
            "4.0",
            "sso-4.0/Sonatina Symphonic Orchestra/"
            "Brass - Performance/Horn Solo Sustain (looped).sfz",
        ),
        "oboe": Patch(
            "SSO",
            "4.0",
            "sso-4.0/Sonatina Symphonic Orchestra/"
            "Woodwinds - Performance/Oboe Solo Sustain (looped).sfz",
        ),
        "saxophone": Patch(
            "VCSL",
            "1.2.2-RC",
            "VCSL-1.2.2-RC/Aerophones/Reed Aerophones/"
            "Tenor Saxophone - Non-Vibrato.sfz",
        ),
        "trombone": Patch(
            "SSO",
            "4.0",
            "sso-4.0/Sonatina Symphonic Orchestra/"
            "Brass - Performance/Tenor Trombone Solo Sustain (looped).sfz",
        ),
        "trumpet": Patch(
            "SSO",
            "4.0",
            "sso-4.0/Sonatina Symphonic Orchestra/"
            "Brass - Performance/Trumpet Solo Sustain (looped).sfz",
        ),
        "tuba": Patch(
            "SSO",
            "4.0",
            "sso-4.0/Sonatina Symphonic Orchestra/"
            "Brass - Performance/Tuba Sustain (looped).sfz",
        ),
        "viola": Patch(
            "SSO",
            "4.0",
            "sso-4.0/Sonatina Symphonic Orchestra/"
            "Strings - Performance/Viola Solo Sustain.sfz",
            zero_vibrato_ccs=(21,),
        ),
        "violin": Patch(
            "SSO",
            "4.0",
            "sso-4.0/Sonatina Symphonic Orchestra/"
            "Strings - Performance/Violin Solo 2 Sustain Non-Vibrato.sfz",
            zero_vibrato_ccs=(21,),
        ),
    }

    def __init__(
        self,
        *,
        executable: str | Path | None = None,
        soundfonts_root: str | Path = DEFAULT_SOUNDFONTS_ROOT,
        executable_sha256: str | None = None,
    ) -> None:
        self.executable = self.resolve_executable(executable)
        self.soundfonts_root = Path(soundfonts_root).expanduser().resolve()
        self._executable_sha256 = executable_sha256

    @staticmethod
    def resolve_executable(configured: str | Path | None = None) -> Path:
        candidates: list[Path] = []
        if configured is not None:
            candidates.append(Path(configured).expanduser())
        env_path = os.environ.get("ATTUNE_SFIZZ_RENDER")
        if env_path:
            candidates.append(Path(env_path).expanduser())
        on_path = shutil.which("sfizz_render")
        if on_path:
            candidates.append(Path(on_path))
        candidates.append(
            Path.home()
            / "dev"
            / "sfizz-1.2.3"
            / "build"
            / "library"
            / "bin"
            / "sfizz_render"
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file() and os.access(resolved, os.X_OK):
                return resolved
        searched = ", ".join(str(path) for path in candidates)
        raise RuntimeError(
            "sfizz_render was not found or is not executable; pass its path "
            f"explicitly or set ATTUNE_SFIZZ_RENDER (searched: {searched})"
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @property
    def executable_sha256(self) -> str:
        if self._executable_sha256 is None:
            self._executable_sha256 = self._sha256(self.executable)
        return self._executable_sha256

    def patch_for(self, instrument: str) -> CocoRenderer.Patch:
        try:
            return self.PATCHES[instrument.lower()]
        except KeyError as error:
            supported = ", ".join(sorted(self.PATCHES))
            raise ValueError(
                f"no straight-tone SFZ mapping for {instrument!r}; "
                f"supported Coco instruments: {supported}"
            ) from error

    def sfz_path_for(self, instrument: str) -> Path:
        return self.soundfonts_root / self.patch_for(instrument).relative_sfz

    def validate(self, instruments: tuple[str, ...] | list[str] | set[str]) -> None:
        if not self.soundfonts_root.is_dir():
            raise RuntimeError(
                f"soundfont repository root was not found: {self.soundfonts_root}"
            )
        missing = [
            str(self.sfz_path_for(instrument))
            for instrument in sorted(set(instruments))
            if not self.sfz_path_for(instrument).is_file()
        ]
        if missing:
            raise RuntimeError(
                "required straight-tone SFZ files are missing:\n  "
                + "\n  ".join(missing)
            )

    def manifest_for(self, instrument: str) -> dict[str, object]:
        patch = self.patch_for(instrument)
        sfz_path = self.sfz_path_for(instrument)
        payload: dict[str, object] = {
            "policy": COCO_SFIZZ_POLICY_VERSION,
            "renderer": "sfizz_render",
            "renderer_sha256": self.executable_sha256,
            "instrument": instrument,
            **asdict(patch),
            "sfz_sha256": self._sha256(sfz_path),
        }
        identity_payload = {
            key: value for key, value in payload.items() if key != "instrument"
        }
        payload["identity"] = hashlib.sha256(
            json.dumps(
                identity_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload["renderer_path"] = str(self.executable)
        payload["sfz_path"] = str(sfz_path)
        return payload

    def prepare_midi(
        self,
        source_midi: str | Path,
        output_midi: str | Path,
        *,
        instrument: str,
    ) -> Path:
        """Remove FluidSynth RPNs and force patch vibrato controls to zero."""
        source_midi = Path(source_midi)
        output_midi = Path(output_midi)
        midi = pretty_midi.PrettyMIDI(str(source_midi))
        pitched = [item for item in midi.instruments if not item.is_drum]
        if not pitched:
            raise ValueError(f"no pitched instrument in {source_midi}")
        target = max(pitched, key=lambda item: len(item.notes))
        zero_ccs = set(self.patch_for(instrument).zero_vibrato_ccs)
        stripped = _MIDI_RPN_CONTROLLERS | zero_ccs
        target.control_changes = [
            cc for cc in target.control_changes if cc.number not in stripped
        ]
        target.control_changes.extend(
            pretty_midi.ControlChange(number=number, value=0, time=0.0)
            for number in sorted(zero_ccs)
        )
        target.control_changes.sort(key=lambda event: (event.time, event.number))
        if not any(
            event.time == 0.0 and event.pitch == 0 for event in target.pitch_bends
        ):
            target.pitch_bends.append(pretty_midi.PitchBend(0, 0.0))
            target.pitch_bends.sort(key=lambda event: event.time)
        output_midi.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(output_midi))
        return output_midi

    @staticmethod
    @contextmanager
    def _render_lock():
        """Serialize sfizz processes; parallel sample loads can truncate audio."""
        lock_path = Path(tempfile.gettempdir()) / "attune-coco-sfizz-render.lock"
        with lock_path.open("a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _renderer_midi(
        self,
        midi_path: Path,
        *,
        instrument: str,
        out_dir: Path,
        force: bool,
    ) -> Path:
        shift = self.patch_for(instrument).render_note_shift_semitones
        if shift == 0:
            return midi_path
        shifted_path = out_dir / f"{midi_path.stem}__sfizz.mid"
        if (
            not force
            and shifted_path.is_file()
            and shifted_path.stat().st_mtime >= midi_path.stat().st_mtime
        ):
            return shifted_path
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        pitched = [item for item in midi.instruments if not item.is_drum]
        if not pitched:
            raise ValueError(f"no pitched instrument in {midi_path}")
        target = max(pitched, key=lambda item: len(item.notes))
        for note in target.notes:
            shifted_pitch = int(note.pitch) + int(shift)
            if not 0 <= shifted_pitch <= 127:
                raise ValueError(
                    f"renderer note shift moves MIDI {note.pitch} outside [0, 127]"
                )
            note.pitch = shifted_pitch
        midi.write(str(shifted_path))
        return shifted_path

    def render(
        self,
        midi_path: str | Path,
        *,
        instrument: str,
        out_dir: str | Path,
        sample_rate: int = 44_100,
        force: bool = False,
    ) -> Path:
        midi_path = Path(midi_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{midi_path.stem}.wav"
        sfz_path = self.sfz_path_for(instrument)
        newest_input = max(
            midi_path.stat().st_mtime,
            sfz_path.stat().st_mtime,
            self.executable.stat().st_mtime,
        )
        if (
            not force
            and output.is_file()
            and output.stat().st_size > 44
            and output.stat().st_mtime >= newest_input
        ):
            return output

        render_midi = self._renderer_midi(
            midi_path,
            instrument=instrument,
            out_dir=out_dir,
            force=force,
        )

        command = [
            str(self.executable),
            "--sfz",
            str(sfz_path),
            "--midi",
            str(render_midi),
            "--wav",
            str(output),
            "--blocksize",
            "512",
            "--samplerate",
            str(int(sample_rate)),
            "--quality",
            "10",
            "--polyphony",
            "64",
            "--use-eot",
        ]
        with self._render_lock():
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode != 0:
            detail = completed.stdout.strip()
            raise RuntimeError(
                f"sfizz_render failed for {instrument} ({sfz_path}): {detail}"
            )
        if not output.is_file() or output.stat().st_size <= 44:
            raise RuntimeError(f"sfizz_render did not create non-empty audio: {output}")
        return output
